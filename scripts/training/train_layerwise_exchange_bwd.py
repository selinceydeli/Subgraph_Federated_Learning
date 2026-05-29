#!/usr/bin/env python3
"""
Layer-wise Embedding Exchange with FedAvg + backward gradient coupling.

Mirrors `train_layerwise_exchange.py` exactly, with one change to the
per-step exchange mechanism: the shared embedding cache stores LIVE
(gradient-attached, on-device) tensor references instead of detached CPU
copies, and the per-step loss is a single combined `sum(losses)` back-
propagated once. The combined backward routes each consumer client's
∂L/∂emb_g — for every owned node g consumed across all clients — through
the natural chain rule into the owning client's parameters. Forward
exchange semantics (per-step reset, write→read, zeros fallback for
unwritten ghosts) are otherwise identical.

Each epoch:
  1. FedAvg parameter aggregation — all client models start identical.
  2. Synchronous per-step exchange training using LiveEmbeddingCache and
     one combined backward per step.
  3. FedAvg again at the end of the epoch; consensus model evaluated on
     each client's val partition via the synchronous forward-only exchange.

Memory note: combined backward keeps every active client's full forward
activation graph alive until the single .backward() call. With many
clients and large batches this can dominate VRAM; if OOM occurs, reduce
batch_size in configs/pna_configs.json or num_clients in
configs/fed_configs.json.

Usage:
    python3 -m scripts.training.train_layerwise_exchange_bwd
"""

import os
import csv
import time
import json
from itertools import zip_longest
from types import SimpleNamespace
from datetime import datetime

import torch

from utils.loader import load_client_graphs, resolve_data_dirs
from utils.seed import set_seed
from utils.metrics import (
    append_f1_score_to_csv,
    append_pr_auc_to_csv,
    start_epoch_csv,
    append_epoch_csv,
    compute_minority_f1_score_per_task,
    compute_pr_auc_per_task,
)
from utils.train_utils import (
    ensure_node_features,
    evaluate_epoch,
)
from utils.hetero import make_bidirected_hetero
from utils.graph_helpers import (
    max_port_cols,
    check_and_strip_self_loops,
    build_hetero_neighbor_loader,
    build_full_eval_loader,
)
from utils.layerwise_exchange import EmbeddingTable, LiveEmbeddingCache
from task.node_cls import NodeClsTask

# Reuse the forward-only LE helpers verbatim — input projection, partition
# integrity check, FedAvg aggregation, and the forward-only eval pass are
# all unchanged. The bwd-coupled variant only changes the per-step training
# loop (combined backward instead of per-client backward) and the cache
# class (LiveEmbeddingCache vs EmbeddingTable).
from scripts.training.train_layerwise_exchange import (
    _prepare_client_state,
    _check_partition_integrity,
    _fedavg_aggregate,
    _synchronous_eval_epoch,
)

PNA_CONFIG_PATH = "./configs/pna_configs.json"
FED_CONFIG_PATH = "./configs/fed_configs.json"

TASKS = ["deg-in", "deg-out", "fan-in", "fan-out", "C2", "C3", "C4", "C5", "C6", "S-G", "B-C"]

# Set True once during development to verify cross-client gradient flow.
# When enabled, client 0's loss is numerically zeroed for the first step of
# the first epoch — any nonzero grad on client 0's params then proves
# gradient is routed through ghost-emb splices from other clients' losses.
CHECK_GRADIENTS_BWD = False


def build_args(pna_cfg, fed_cfg, partition_cfg):
    hparams = pna_cfg["default_hparams"]
    return SimpleNamespace(
        use_ego_ids=pna_cfg["use_ego_ids"],
        use_port_ids=pna_cfg["use_port_ids"],
        use_mini_batch=pna_cfg["use_mini_batch"],
        batch_size=pna_cfg["batch_size"],
        port_emb_dim=pna_cfg["port_emb_dim"],
        num_layers=hparams["num_layers"],
        neighbors_per_hop=hparams["neighbors_per_hop"],
        hidden_dim=hparams["hidden_dim"],
        dropout=hparams["dropout"],
        lr=hparams["lr"],
        weight_decay=hparams["weight_decay"],
        minority_class_weight=hparams["minority_class_weight"],
        ego_dim=hparams.get("ego_dim", None),
        local_epochs=fed_cfg["local_epochs"],
        global_epochs=fed_cfg["global_epochs"],
        base_seed=fed_cfg["base_seed"],
        partition_strategy=partition_cfg.get("partition_strategy", "partition_aware"),
        num_clients=partition_cfg["num_clients"],
        include_cross_edges=partition_cfg["include_cross_edges"],
        use_local_labels=partition_cfg.get("use_local_labels", False),
    )


def make_eval_loader(client_data, task, device, shuffle=False):
    """Preprocess a val/test subgraph and return a loader for evaluation."""
    data = check_and_strip_self_loops(client_data, "eval")
    data = ensure_node_features(data)
    hetero = make_bidirected_hetero(data)

    if task.use_port_ids:
        fwd_ea = hetero[("n", "fwd", "n")].edge_attr.clone()
        fwd_ea[:, 0].clamp_(max=task.in_port_vocab_size - 1)
        fwd_ea[:, 1].clamp_(max=task.out_port_vocab_size - 1)
        hetero[("n", "fwd", "n")].edge_attr = fwd_ea

        rev_ea = hetero[("n", "rev", "n")].edge_attr.clone()
        rev_ea[:, 0].clamp_(max=task.out_port_vocab_size - 1)
        rev_ea[:, 1].clamp_(max=task.in_port_vocab_size - 1)
        hetero[("n", "rev", "n")].edge_attr = rev_ea

    owned_idx = None
    if hasattr(hetero["n"], "owned_mask") and hetero["n"].owned_mask is not None:
        owned_idx = torch.where(hetero["n"].owned_mask)[0]

    num_hops = task.num_layers

    if task.use_mini_batch:
        return build_hetero_neighbor_loader(
            hetero,
            batch_size=task.batch_size,
            num_layers=num_hops,
            fanout=task.neighbors_per_hop,
            device=device,
            shuffle=shuffle,
            input_nodes=owned_idx,
        )
    else:
        return build_full_eval_loader(
            hetero,
            batch_size=hetero["n"].num_nodes,
            num_layers=num_hops,
            device=device,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Global ghost-routing index (built once per experiment)
# ──────────────────────────────────────────────────────────────────────────────

def _build_step_gid_index(client_states, num_nodes, device):
    """
    Build a PER-STEP mapping from global node ID → (owner_client_id,
    local_row_index_in_owner's_current_batch). Used by the bwd-coupled
    exchange loop to route a consumer's remote-gid lookup to the right owner's
    stored LIVE activation tensor.

    Crucially, the row index points into the owner's CURRENT mini-batch
    (`s['x']` / `s['gids']`), not the owner's full `data.x`. In mini-batch
    mode the stored activation tensor only spans the batch's nodes, so a
    static full-graph index would run off the end. Rebuilding from the
    current batches each step keeps the row indices valid in both mini-batch
    and full-batch modes.

    A gid an owner owns globally but did not draw into its current batch is
    left unmapped (-1); consumers requesting it fall back to their locally
    computed embedding, exactly as the forward-only EmbeddingTable's
    written-mask does.

    Returns:
        gid_to_owner_cid: [num_nodes] long tensor on `device`. Index by
                          global_nid. Value is the owner client id, or -1
                          if no client wrote that gid this step.
        gid_to_local_idx: [num_nodes] long tensor on `device`. Index by
                          global_nid. Value is the row index of that gid
                          within the owner's current-batch activation, or -1.
    """
    gid_to_owner_cid = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    gid_to_local_idx = torch.full((num_nodes,), -1, dtype=torch.long, device=device)

    for cid, s in enumerate(client_states):
        if s['owned'] is None or s['gids'] is None:
            continue
        owned = s['owned'].to(device)
        gids = s['gids'].to(device).long()
        owned_local = owned.nonzero(as_tuple=True)[0]
        owned_gids = gids[owned]
        gid_to_owner_cid[owned_gids] = cid
        gid_to_local_idx[owned_gids] = owned_local

    return gid_to_owner_cid, gid_to_local_idx


# ──────────────────────────────────────────────────────────────────────────────
# Bwd-coupled per-step exchange (live cache + gradient-attached injection)
# ──────────────────────────────────────────────────────────────────────────────

def _exchange_one_step_live(
    client_states, cache, num_layers, device,
    *, num_nodes, track_coverage=False,
):
    """
    One synchronous-exchange step with LIVE (autograd-tracked) ghost
    injection. Mirrors `_exchange_one_step` in train_layerwise_exchange.py
    except:

      • cache.write stores the full live activation tensor — no .detach(),
        no .cpu() — so autograd edges from the owner's parameters through
        the stored tensor stay intact.
      • cache.fetch_remote returns a live tensor whose backward propagates
        into the owning client's model via the chain rule.
      • The torch.where injection mixes two live branches; the x_fill
        branch carries gradient back to the owners through the ghost
        positions.

    All other semantics — per-step reset, conv → write → inject ordering,
    zeros fallback for unwritten ghosts — match the forward-only variant.

    Returns:
        (remote_total, remote_served) per-layer lists when
        track_coverage=True, else (None, None).
    """
    cache.reset()
    gid_to_owner, gid_to_local_idx = _build_step_gid_index(
        client_states, num_nodes, device
    )
    remote_total  = [0] * num_layers if track_coverage else None
    remote_served = [0] * num_layers if track_coverage else None

    for l in range(num_layers):
        for s in client_states:
            s['x'] = s['task'].model.compute_conv_layer(
                l, s['x'], s['ei'], s['pna_ea']
            )
        for cid, s in enumerate(client_states):
            cache.write(l, cid, s['x'])

        for s in client_states:
            if s['owned'] is None or s['gids'] is None:
                continue
            remote_mask = ~s['owned']
            if not remote_mask.any():
                continue

            rem_gids = s['gids'][remote_mask].to(device).long()
            owners   = gid_to_owner[rem_gids]
            loc_idx  = gid_to_local_idx[rem_gids]

            rem_embs, served = cache.fetch_remote(l, owners, loc_idx)

            if track_coverage:
                remote_total[l] += int(remote_mask.sum().item())
                remote_served[l] += int(served.sum().item())

            if not served.any():
                continue

            remote_indices = remote_mask.nonzero(as_tuple=True)[0]
            written_indices = remote_indices[served]
            written_full = torch.zeros(
                s['x'].size(0), dtype=torch.bool, device=device
            )
            written_full[written_indices] = True
            x_fill = torch.zeros_like(s['x'])
            x_fill[written_indices] = rem_embs[served]
            s['x'] = torch.where(
                (s['owned'] | ~written_full).unsqueeze(1).expand_as(s['x']),
                s['x'],
                x_fill,
            )

    return remote_total, remote_served


def _synchronous_train_epoch_live(
    tasks, cache, device, *, num_nodes,
    grad_check_epoch=None, current_epoch=None,
):
    """
    One pass of bwd-coupled per-step layer-wise exchange training.

    For each mini-batch step:
      1. All clients fetch one batch and project node features.
      2. _exchange_one_step_live runs the per-layer write→inject loop with
         the live cache. Owned positions stay in each client's own autograd
         graph; remote positions are spliced in as live references to the
         owner's graph.
      3. Each client computes its output + loss on owned/seed positions.
      4. A single combined backward — combined_loss = sum(per-client
         losses) — is back-propagated once. Autograd's chain rule
         automatically sums each consumer's ∂L/∂emb_g at the shared
         emb_g tensor, then continues back through the owning client's
         model into its parameters. Each client's optimizer.step()
         picks up the accumulated grad in its own params.

    Returns (avg_loss, exchange_stats) with keys 'remote_total' and
    'remote_served' (per-layer lists).
    """
    iters = [iter(t.train_loader) for t in tasks]
    num_steps = min(len(t.train_loader) for t in tasks)

    total_loss = 0.0
    total_count = 0

    num_layers = tasks[0].num_layers
    remote_total  = [0] * num_layers
    remote_served = [0] * num_layers

    step_idx = 0
    for _ in range(num_steps):
        client_states = []
        for task, it in zip(tasks, iters):
            task.model.train()
            client_states.append(_prepare_client_state(task, next(it), device))

        for t in tasks:
            t.optimizer.zero_grad()

        step_total, step_served = _exchange_one_step_live(
            client_states, cache, num_layers, device,
            num_nodes=num_nodes,
            track_coverage=True,
        )
        for l in range(num_layers):
            remote_total[l]  += step_total[l]
            remote_served[l] += step_served[l]

        losses_list = []
        for cid, s in enumerate(client_states):
            task = s['task']
            logits = task.model.compute_output(s['x'])

            out_used = logits[:s['B']] if s['B'] is not None else logits
            y_batch  = s['y'][:s['B']] if s['B'] is not None else s['y']

            if s['owned'] is not None and (s['B'] is None or s['B'] == s['N']):
                out_used = out_used[s['owned']]
                y_batch  = y_batch[s['owned']]
                count = int(s['owned'].sum().item())
            else:
                count = s['B'] if s['B'] is not None else s['N']

            loss = task.criterion(out_used, y_batch.float())
            losses_list.append(loss)
            total_loss += loss.item() * count
            total_count += count

        do_grad_check = (
            CHECK_GRADIENTS_BWD and step_idx == 0
            and grad_check_epoch is not None and current_epoch == grad_check_epoch
        )
        if do_grad_check and len(losses_list) >= 2:
            # Zero client 0's contribution numerically while keeping the
            # graph alive; any nonzero grad on client 0's params then came
            # purely from other clients' losses through ghost splices.
            losses_list[0] = 0.0 * losses_list[0]

        combined = torch.stack(losses_list).sum()
        combined.backward()

        if do_grad_check and len(losses_list) >= 2:
            p0 = next(tasks[0].model.parameters())
            g0 = p0.grad
            assert g0 is not None and g0.abs().sum().item() > 0, (
                "[BwdGradCheck] FAIL: client 0 has no grad with loss_0=0 — "
                "cross-client gradient is NOT flowing. Check that "
                "LiveEmbeddingCache.fetch_remote preserves autograd and that "
                "the torch.where x_fill branch is live."
            )
            print(
                f"[BwdGradCheck] PASS: cross-client gradient confirmed "
                f"(|grad_p0| sum = {g0.abs().sum().item():.4e})"
            )

        for t in tasks:
            t.optimizer.step()

        step_idx += 1

    exchange_stats = {
        'remote_total': remote_total,
        'remote_served': remote_served,
    }
    return total_loss / max(total_count, 1), exchange_stats


# ──────────────────────────────────────────────────────────────────────────────
# Main experiment loop (one seed)
# ──────────────────────────────────────────────────────────────────────────────

def run_exchange_bwd_experiment(train_list, val_list, test_list, args, device, seed, run_id):
    """
    Run the full bwd-coupled layer-wise exchange protocol for all clients
    under one seed. Returns a list of per-client dicts with
    test_f1_per_task and test_pr_auc_per_task.
    """
    num_clients = len(train_list)
    num_clients_cfg = getattr(args, "num_clients", num_clients)
    label_suffix = "_local_labels" if getattr(args, "use_local_labels", False) else ""
    strategy = getattr(args, "partition_strategy", "partition_aware")
    cross_suffix = "with_cross_edges" if getattr(args, "include_cross_edges", False) else "without_cross_edges"
    run_tag = f"{strategy}_{cross_suffix}"

    set_seed(seed)
    tasks = []
    for cid in range(num_clients):
        task = NodeClsTask(args, cid, train_list[cid], "./data", device)
        tasks.append(task)

    val_loaders = [
        make_eval_loader(val_list[cid], tasks[cid], device, shuffle=True)
        for cid in range(num_clients)
    ]
    test_loaders = [
        make_eval_loader(test_list[cid], tasks[cid], device, shuffle=True)
        for cid in range(num_clients)
    ]

    epoch_csv_paths = [
        start_epoch_csv(
            model_name=f"layerwise_exchange_bwd_client_{cid}",
            seed=seed,
            tasks=TASKS,
            out_dir=(
                f"./results/metrics/federated_logs/"
                f"layerwise_exchange_bwd{label_suffix}/{run_tag}/{num_clients_cfg}_clients/client_{cid}"
            ),
        )
        for cid in range(num_clients)
    ]

    # Live cache for training (one combined backward per step) and a
    # separate forward-only EmbeddingTable for eval (eval is no_grad, so
    # the simpler detached path is fine and avoids GPU memory pressure).
    all_gids = torch.cat([train_list[cid].global_nid for cid in range(num_clients)])
    num_nodes = int(all_gids.max().item()) + 1
    cache = LiveEmbeddingCache(num_clients=num_clients, num_layers=args.num_layers)
    eval_table = EmbeddingTable(
        num_nodes=num_nodes,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
    )

    _check_partition_integrity(train_list, num_nodes)

    coverage_csv_dir = (
        f"./results/metrics/federated_logs/"
        f"layerwise_exchange_bwd{label_suffix}/{run_tag}/{num_clients_cfg}_clients"
    )
    os.makedirs(coverage_csv_dir, exist_ok=True)
    coverage_csv_path = os.path.join(coverage_csv_dir, f"exchange_coverage_seed{seed}.csv")
    coverage_csv_header = [
        "epoch", "layer", "remote_total", "remote_served", "coverage_pct", "bytes_communicated_mb"
    ]
    if not os.path.exists(coverage_csv_path):
        with open(coverage_csv_path, "w", newline="") as f:
            csv.writer(f).writerow(coverage_csv_header)

    ckpt_dir = f"./checkpoints/layerwise_exchange_bwd{label_suffix}/{run_tag}/{num_clients_cfg}_clients"
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(ckpt_dir, f"seed{seed}_{run_id}_best.pt")
    best_val_pr_auc = float("-inf")

    local_epochs = getattr(args, 'local_epochs', 1)

    for epoch in range(1, args.global_epochs + 1):
        _fedavg_aggregate(tasks)

        epoch_remote_total  = [0] * args.num_layers
        epoch_remote_served = [0] * args.num_layers
        for _ in range(local_epochs):
            _, step_stats = _synchronous_train_epoch_live(
                tasks, cache, device,
                num_nodes=num_nodes,
                grad_check_epoch=1, current_epoch=epoch,
            )
            for l in range(args.num_layers):
                epoch_remote_total[l]  += step_stats['remote_total'][l]
                epoch_remote_served[l] += step_stats['remote_served'][l]

        total_served = sum(epoch_remote_served)
        total_remote = sum(epoch_remote_total)
        bytes_communicated = total_served * args.hidden_dim * 4
        mb = bytes_communicated / (1024 ** 2)

        layer_parts = [
            f"Layer {l}: {100.0 * epoch_remote_served[l] / max(epoch_remote_total[l], 1):.1f}%"
            f" ({epoch_remote_served[l]}/{epoch_remote_total[l]})"
            for l in range(args.num_layers)
        ]
        print(f"[Exchange] {' | '.join(layer_parts)} | Bytes: {mb:.1f} MB")

        with open(coverage_csv_path, "a", newline="") as f:
            w = csv.writer(f)
            for l in range(args.num_layers):
                rt = epoch_remote_total[l]
                rs = epoch_remote_served[l]
                pct = 100.0 * rs / rt if rt > 0 else 0.0
                layer_bytes_mb = rs * args.hidden_dim * 4 / (1024 ** 2)
                w.writerow([epoch, l, rt, rs, f"{pct:.4f}", f"{layer_bytes_mb:.4f}"])
            total_pct = 100.0 * total_served / total_remote if total_remote > 0 else 0.0
            w.writerow([epoch, "total", total_remote, total_served, f"{total_pct:.4f}", f"{mb:.4f}"])

        _fedavg_aggregate(tasks)

        global_model = tasks[0].model
        global_criterion = tasks[0].criterion

        val_losses, val_f1s, val_pr_aucs = _synchronous_eval_epoch(
            tasks, val_loaders, eval_table, device
        )

        avg_val_loss     = sum(val_losses) / len(val_losses)
        avg_val_f1       = torch.stack(val_f1s).mean(dim=0)
        avg_val_pr_auc   = torch.stack(val_pr_aucs).mean(dim=0)
        val_macro_f1     = avg_val_f1.mean().item()
        val_macro_pr_auc = avg_val_pr_auc.mean().item()

        train_loss, _, _, _ = evaluate_epoch(
            global_model, tasks[0].train_loader, global_criterion, device, tasks[0].use_port_ids
        )

        for cid in range(num_clients):
            append_epoch_csv(
                epoch_csv_paths[cid], epoch, train_loss,
                val_losses[cid], val_f1s[cid], val_pr_aucs[cid],
            )

        if val_macro_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_macro_pr_auc
            torch.save(global_model.state_dict(), best_ckpt_path)

        print(
            f"[Seed {seed}] Epoch {epoch:03d} | "
            f"train {train_loss:.4f} | val {avg_val_loss:.4f} | "
            f"val macro-minF1 {100 * val_macro_f1:.2f}% | "
            f"val macro-PR-AUC {100 * val_macro_pr_auc:.2f}%"
        )

    best_sd = torch.load(best_ckpt_path, map_location=device)
    for task in tasks:
        task.model.load_state_dict(best_sd)

    test_losses, test_f1s, test_pr_aucs = _synchronous_eval_epoch(
        tasks, test_loaders, eval_table, device
    )

    results = []
    for cid in range(num_clients):
        test_f1 = test_f1s[cid]
        test_pr_auc = test_pr_aucs[cid]
        print(
            f"[Seed {seed}][Client {cid}] Best ckpt → "
            f"test_loss={test_losses[cid]:.4f} | "
            f"test macro-minF1={100 * test_f1.mean().item():.2f}% | "
            f"test macro-PR-AUC={100 * test_pr_auc.mean().item():.2f}%"
        )
        results.append({
            "client_id": cid,
            "test_f1_per_task": test_f1.cpu(),
            "test_pr_auc_per_task": test_pr_auc.cpu(),
        })
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_ts = time.perf_counter()

    with open(PNA_CONFIG_PATH, "r") as f:
        pna_all = json.load(f)
    with open(FED_CONFIG_PATH, "r") as f:
        fed_all = json.load(f)

    pna_cfg = pna_all["reverse_mp_with_port_and_ego"]
    fed_cfg = fed_all["fed_learning_configs"]
    partition_cfg = fed_all["fed_splits"]

    args = build_args(pna_cfg, fed_cfg, partition_cfg)

    print(
        f"[Config] partition_strategy={args.partition_strategy}, "
        f"global_epochs={args.global_epochs}, local_epochs={args.local_epochs}, "
        f"num_clients={args.num_clients}, include_cross_edges={args.include_cross_edges}, "
        f"use_local_labels={args.use_local_labels}"
    )
    print(
        f"[Config] use_ego_ids={args.use_ego_ids}, use_port_ids={args.use_port_ids}, "
        f"use_mini_batch={args.use_mini_batch}, batch_size={args.batch_size}"
    )

    train_dir, val_dir, test_dir = resolve_data_dirs(partition_cfg)
    num_clients = args.num_clients
    train_list = load_client_graphs(train_dir, num_clients)
    val_list = load_client_graphs(val_dir, num_clients)
    test_list = load_client_graphs(test_dir, num_clients)
    print(f"[Data] Loaded {num_clients} clients.")

    if args.use_port_ids:
        global_unified_vocab = 0
        for raw_data in train_list:
            data_ = check_and_strip_self_loops(raw_data, "precompute")
            data_ = ensure_node_features(data_)
            in_max, out_max = max_port_cols(data_)
            global_unified_vocab = max(global_unified_vocab, in_max, out_max)
        args.in_port_vocab_size = global_unified_vocab + 1
        args.out_port_vocab_size = global_unified_vocab + 1
        print(f"[Port Vocab] Global unified vocab size: {global_unified_vocab + 1}")

    seeds = [args.base_seed, args.base_seed + 1]
    per_seed_mean_f1 = []
    per_seed_mean_pr_auc = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"[Seed {seed}] Starting bwd-coupled layer-wise exchange training ({num_clients} clients)...")
        print(f"{'='*60}")
        client_results = run_exchange_bwd_experiment(
            train_list, val_list, test_list, args, device, seed, run_id
        )

        seed_f1 = torch.stack([r["test_f1_per_task"] for r in client_results], dim=0)
        seed_pr_auc = torch.stack([r["test_pr_auc_per_task"] for r in client_results], dim=0)
        per_seed_mean_f1.append(seed_f1.mean(dim=0))
        per_seed_mean_pr_auc.append(seed_pr_auc.mean(dim=0))

    all_seeds_f1 = torch.stack(per_seed_mean_f1, dim=0)
    mean_f1 = all_seeds_f1.mean(dim=0)
    std_f1 = all_seeds_f1.std(dim=0, unbiased=False)
    macro_mean = mean_f1.mean().item() * 100

    all_seeds_pr_auc = torch.stack(per_seed_mean_pr_auc, dim=0)
    mean_pr_auc = all_seeds_pr_auc.mean(dim=0)
    std_pr_auc = all_seeds_pr_auc.std(dim=0, unbiased=False)
    macro_pr_auc = mean_pr_auc.mean().item() * 100

    print(f"\n{'='*60}")
    print(
        f"[Results] Layer-wise exchange + FedAvg + bwd coupling — "
        f"macro minority F1 (mean over {len(seeds)} seeds × {num_clients} clients): "
        f"{macro_mean:.2f}%"
    )
    row = " | ".join(
        f"{n}: {100*m:.2f}±{100*s:.2f}%"
        for n, m, s in zip(TASKS, mean_f1.tolist(), std_f1.tolist())
    )
    print(f"[Results] Per-task F1 (mean±std): {row}")
    print(f"[Results] macro PR-AUC: {macro_pr_auc:.2f}%")
    row_pr = " | ".join(
        f"{n}: {100*m:.2f}±{100*s:.2f}%"
        for n, m, s in zip(TASKS, mean_pr_auc.tolist(), std_pr_auc.tolist())
    )
    print(f"[Results] Per-task PR-AUC (mean±std): {row_pr}")

    runtime_sec = time.perf_counter() - start_ts

    label_suffix = "_local_labels" if args.use_local_labels else ""
    out_csv     = f"./results/metrics/federated_logs/layerwise_exchange_bwd{label_suffix}_results.csv"
    out_csv_auc = f"./results/metrics/federated_logs/layerwise_exchange_bwd{label_suffix}_pr_auc_results.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    model_name_str = (
        f"Layer-wise exchange + FedAvg + bwd coupling (live cache + combined backward) | "
        f"partition_strategy={args.partition_strategy}, "
        f"num_clients={num_clients}, "
        f"cross_edges={args.include_cross_edges}, "
        f"local_labels={args.use_local_labels}, "
        f"global_epochs={args.global_epochs}, "
        f"local_epochs={args.local_epochs}, "
        f"use_port_ids={args.use_port_ids}, "
        f"use_ego_ids={args.use_ego_ids}, "
        f"num_layers={args.num_layers}, "
        f"neighbors_per_hop={args.neighbors_per_hop}, "
        f"batch_size={args.batch_size}, "
        f"lr={args.lr}, "
        f"seeds={seeds}, "
        f"run_id={run_id}"
    )

    append_f1_score_to_csv(
        out_csv=out_csv,
        tasks=TASKS,
        mean_f1=mean_f1,
        std_f1=std_f1,
        macro_mean_percent=macro_mean,
        seeds=seeds,
        model_name=model_name_str,
        runtime_seconds=runtime_sec,
    )
    append_pr_auc_to_csv(
        out_csv=out_csv_auc,
        tasks=TASKS,
        mean_pr_auc=mean_pr_auc,
        std_pr_auc=std_pr_auc,
        macro_mean_prauc=macro_pr_auc,
        seeds=seeds,
        model_name=model_name_str,
        runtime_seconds=runtime_sec,
    )

    print(f"\n[Done] Runtime: {runtime_sec:.1f}s | F1 → {out_csv} | PR-AUC → {out_csv_auc}")


if __name__ == "__main__":
    main()
