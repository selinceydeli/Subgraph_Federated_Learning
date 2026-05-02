#!/usr/bin/env python3
"""
Layer-wise Embedding Exchange — once-per-epoch (OptimES-style ablation).

Mirrors the OptimES paper's federated GNN training methodology
(Naman & Simmhan, 2025): clients exchange `h^1...h^{L-1}` for boundary
vertices ONCE per round (= per epoch here, ε = 1) instead of per
mini-batch step. Inside an epoch the cache is frozen; ghost positions
are filled from the pre-pulled cache and gradients never flow through
remote embeddings.

Lifecycle (per seed):

    PRE-TRAINING (once, before epoch 1):
        each client does a full forward over its local subgraph and
        writes h^1...h^{L-1} for its OWNED nodes into cache_in. Cold
        start uses zeros for ghost positions.

    EPOCH e:
        FedAvg(tasks)                     # synchronise params
        each client trains independently on its train_loader; every
        mini-batch forward INJECTS ghost embeddings from cache_in
        (frozen, gradient-detached) at every conv layer. No writes.
        PUSH PHASE:                       # OptimES § 3.2.2 push
            cache_out.reset()
            each client's diverged model does a full forward and
            writes refreshed h^1...h^{L-1} for owned nodes into
            cache_out (still injecting from cache_in for ghosts).
        FedAvg(tasks)                     # synchronise for eval
        cache_in <-> cache_out swap
        EVAL with cache_in (per-client mini-batches, frozen inject)

Key contrasts with the per-step variant (`train_layerwise_exchange.py`):
  - Cache reset moves from per-step to per-epoch (one push pass).
  - Training NEVER writes to the cache; only the push phase does.
  - Clients train independently within an epoch (no per-step lockstep).
  - Gradient flow through ghosts is impossible by construction
    (ghost values are read from a CPU table, fetched detached).
  - One extra full forward per client at the end of every epoch (push).

Deviations from strict OptimES:
  - Ghost rows keep their raw features (h^0). Cache injection at every
    layer overrides ghost h^l for l >= 1, so this only affects the very
    first conv's aggregation. (User-confirmed pragmatic trade-off.)
  - All `num_layers` slots of the cache are populated, including h^L
    (the final conv output). h^L for ghosts is read but never used in
    loss, so this is a no-op deviation.

Usage:
    python3 -m scripts.training.train_layerwise_exchange_per_epoch
"""

import os
import csv
import time
import json
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
from utils.train_utils import ensure_node_features, evaluate_epoch
from utils.hetero import make_bidirected_hetero
from utils.graph_helpers import (
    max_port_cols,
    check_and_strip_self_loops,
    build_hetero_neighbor_loader,
    build_full_eval_loader,
)
from utils.layerwise_exchange import EmbeddingTable
from task.node_cls import NodeClsTask

# Reuse the shared exchange helpers verbatim — keeps train/eval mechanics
# byte-for-byte identical with the per-step variant where they overlap.
from scripts.training.train_layerwise_exchange import (
    _prepare_client_state,
    _check_partition_integrity,
    _fedavg_aggregate,
)

PNA_CONFIG_PATH = "./configs/pna_configs.json"
FED_CONFIG_PATH = "./configs/fed_configs.json"

TASKS = ["deg-in", "deg-out", "fan-in", "fan-out", "C2", "C3", "C4", "C5", "C6", "S-G", "B-C"]


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
# Per-epoch exchange primitives
# ──────────────────────────────────────────────────────────────────────────────

def _inject_ghosts_from_cache(state, x, layer, cache, device, *, attempts=None, hits=None):
    """
    Replace ghost-position embeddings at `x` with values from `cache` at
    the given conv layer. Gradient-safe: owned positions retain their
    gradient path; ghost positions are filled from a detached zero tensor.

    If `attempts` and `hits` lists are provided, increments
    attempts[layer] += #ghosts and hits[layer] += #ghosts that had a
    written cache entry.

    Returns the (possibly-updated) `x`.
    """
    if state['owned'] is None or state['gids'] is None:
        return x
    remote_mask = ~state['owned']
    if not remote_mask.any():
        return x

    rem_gids = state['gids'][remote_mask]
    rem_embs = cache.fetch(layer, rem_gids).to(x.device)
    written = cache.written_mask(layer, rem_gids).to(x.device)

    if attempts is not None:
        attempts[layer] += int(remote_mask.sum().item())
    if hits is not None:
        hits[layer] += int(written.sum().item())

    if not written.any():
        return x

    remote_indices = remote_mask.nonzero(as_tuple=True)[0]
    written_indices = remote_indices[written]
    written_full = torch.zeros(x.size(0), dtype=torch.bool, device=device)
    written_full[written_indices] = True

    x_fill = torch.zeros_like(x)
    x_fill[written_indices] = rem_embs[written]

    return torch.where(
        (state['owned'] | ~written_full).unsqueeze(1).expand_as(x),
        x,
        x_fill,
    )


def _collect_seed_owned(state, x, layer, cache_out):
    """
    Write the seed-slice (mini-batch) or all-owned (full-batch) embeddings
    at `x` into `cache_out` at the given conv layer, keyed by global node ID.

    Mini-batch: PyG NeighborLoader puts seeds in the first B positions, and
    they are owned by construction (input_nodes=owned_idx). Writing only the
    seed slice avoids overwriting earlier batches' high-quality (seed-rooted)
    embeddings with later batches' low-quality (peripheral) embeddings.

    Full-batch: every owned position is effectively a seed.

    Returns the count of embeddings written (for push-payload accounting).
    """
    if state['gids'] is None:
        return 0

    B = state['B']
    N = state['N']
    if B is not None and B > 0 and (N is None or B < N):
        seed_gids = state['gids'][:B].cpu()
        cache_out.update(layer, seed_gids, x[:B].detach().cpu())
        return int(B)
    elif state['owned'] is not None:
        owned_gids = state['gids'][state['owned']].cpu()
        cache_out.update(layer, owned_gids, x[state['owned']].detach().cpu())
        return int(state['owned'].sum().item())
    return 0


def _full_forward_collect(tasks, inject_cache, collect_cache, device, *, reset_collect=True):
    """
    Each client iterates its train_loader and does a full forward, optionally
    injecting ghost embeddings from `inject_cache` (read-only) at every layer
    and collecting the seed-slice owned embeddings into `collect_cache`.

    Used for:
      • Pre-training round: inject_cache=None,    collect_cache=cache_in
      • Push phase:         inject_cache=cache_in, collect_cache=cache_out

    Eval mode + no_grad — deterministic with frozen BN running stats so we
    don't pollute statistics nor compute gradients we won't use. Each client
    operates with its own (potentially diverged) model weights, mirroring
    OptimES § 3.2.2 where the push uses the locally-trained model.

    Returns total push-payload count (entries written across all clients
    and all layers); caller can convert to MB via × hidden_dim × 4.
    """
    if collect_cache is not None and reset_collect:
        collect_cache.reset()

    total_written = 0
    for t in tasks:
        t.model.eval()

    with torch.no_grad():
        for task in tasks:
            for batch in task.train_loader:
                state = _prepare_client_state(task, batch, device)
                x = state['x']
                for l in range(task.num_layers):
                    x = task.model.compute_conv_layer(l, x, state['ei'], state['pna_ea'])
                    if inject_cache is not None:
                        x = _inject_ghosts_from_cache(state, x, l, inject_cache, device)
                    if collect_cache is not None:
                        total_written += _collect_seed_owned(state, x, l, collect_cache)

    for t in tasks:
        t.model.train()

    return total_written


def _per_epoch_train_epoch(tasks, cache_in, device):
    """
    One pass of independent per-client training with frozen-cache injection.

    For each client (no inter-client lockstep), iterate its train_loader; in
    every mini-batch:
      1. Project the input via _prepare_client_state.
      2. For each conv layer: compute_conv_layer → inject ghost positions
         from cache_in (frozen, no writes).
      3. Compute output, restrict loss to seed/owned positions, backward, step.

    Returns (avg_loss, exchange_stats) where exchange_stats is:
        {'cache_attempts': [N_l]_l, 'cache_hits': [H_l]_l}
    """
    total_loss = 0.0
    total_count = 0
    num_layers = tasks[0].num_layers
    cache_attempts = [0] * num_layers
    cache_hits = [0] * num_layers

    for task in tasks:
        task.model.train()

    for task in tasks:
        for batch in task.train_loader:
            state = _prepare_client_state(task, batch, device)
            x = state['x']

            for l in range(num_layers):
                x = task.model.compute_conv_layer(l, x, state['ei'], state['pna_ea'])
                x = _inject_ghosts_from_cache(
                    state, x, l, cache_in, device,
                    attempts=cache_attempts, hits=cache_hits,
                )

            logits = task.model.compute_output(x)
            out_used = logits[:state['B']] if state['B'] is not None else logits
            y_batch = state['y'][:state['B']] if state['B'] is not None else state['y']

            if state['owned'] is not None and (state['B'] is None or state['B'] == state['N']):
                out_used = out_used[state['owned']]
                y_batch = y_batch[state['owned']]
                count = int(state['owned'].sum().item())
            else:
                count = state['B'] if state['B'] is not None else state['N']

            task.optimizer.zero_grad()
            loss = task.criterion(out_used, y_batch.float())
            loss.backward()
            task.optimizer.step()

            total_loss += loss.item() * count
            total_count += count

    exchange_stats = {
        'cache_attempts': cache_attempts,
        'cache_hits': cache_hits,
    }
    return total_loss / max(total_count, 1), exchange_stats


def _per_epoch_eval_epoch(tasks, loaders, cache_in, device):
    """
    Eval pass with frozen-cache injection — same mechanism as the training
    forward, so val/test metrics reflect the regime the model was optimised
    for. Each client iterates its own loader independently.

    Returns three parallel lists of length num_clients:
        (val_losses, val_f1_per_task, val_pr_auc_per_task)
    """
    for t in tasks:
        t.model.eval()

    num_clients = len(tasks)
    num_layers = tasks[0].num_layers

    per_client_logits = [[] for _ in range(num_clients)]
    per_client_labels = [[] for _ in range(num_clients)]
    per_client_loss_sum = [0.0] * num_clients
    per_client_count = [0] * num_clients

    with torch.no_grad():
        for cid, (task, loader) in enumerate(zip(tasks, loaders)):
            for batch in loader:
                state = _prepare_client_state(task, batch, device)
                x = state['x']
                for l in range(num_layers):
                    x = task.model.compute_conv_layer(l, x, state['ei'], state['pna_ea'])
                    x = _inject_ghosts_from_cache(state, x, l, cache_in, device)

                logits = task.model.compute_output(x)
                out_used = logits[:state['B']] if state['B'] is not None else logits
                y_batch = state['y'][:state['B']] if state['B'] is not None else state['y']

                if state['owned'] is not None and (state['B'] is None or state['B'] == state['N']):
                    out_used = out_used[state['owned']]
                    y_batch = y_batch[state['owned']]
                    count = int(state['owned'].sum().item())
                else:
                    count = state['B'] if state['B'] is not None else state['N']

                loss = task.criterion(out_used, y_batch.float())
                per_client_loss_sum[cid] += loss.item() * count
                per_client_count[cid] += count
                per_client_logits[cid].append(out_used.detach().cpu())
                per_client_labels[cid].append(y_batch.detach().cpu())

    losses, f1s, pr_aucs = [], [], []
    for cid in range(num_clients):
        avg_loss = per_client_loss_sum[cid] / max(per_client_count[cid], 1)
        if per_client_logits[cid]:
            logits = torch.cat(per_client_logits[cid], dim=0)
            labels = torch.cat(per_client_labels[cid], dim=0)
        else:
            logits = torch.empty((0,))
            labels = torch.empty((0,))
        f1 = compute_minority_f1_score_per_task(logits, labels)
        pr_auc = compute_pr_auc_per_task(logits, labels)
        losses.append(avg_loss)
        f1s.append(f1)
        pr_aucs.append(pr_auc)

    return losses, f1s, pr_aucs


# ──────────────────────────────────────────────────────────────────────────────
# Main experiment loop (one seed)
# ──────────────────────────────────────────────────────────────────────────────

def run_exchange_experiment(train_list, val_list, test_list, args, device, seed, run_id):
    """
    Run the OptimES-style per-epoch exchange protocol for all clients under
    one seed. Returns a list of per-client dicts with test_f1_per_task and
    test_pr_auc_per_task.
    """
    num_clients = len(train_list)
    num_clients_cfg = getattr(args, "num_clients", num_clients)
    label_suffix = "_local_labels" if getattr(args, "use_local_labels", False) else ""
    strategy = getattr(args, "partition_strategy", "partition_aware")
    cross_suffix = "with_cross_edges" if getattr(args, "include_cross_edges", False) else "without_cross_edges"
    run_tag = f"{strategy}_{cross_suffix}"

    # ── Initialise one independent model + optimizer per client ───────────────
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

    # ── Per-epoch CSV logging (one file per client) ───────────────────────────
    epoch_csv_paths = [
        start_epoch_csv(
            model_name=f"layerwise_exchange_per_epoch_client_{cid}",
            seed=seed,
            tasks=TASKS,
            out_dir=(
                f"./results/metrics/federated_logs/"
                f"layerwise_exchange_per_epoch{label_suffix}/{run_tag}/{num_clients_cfg}_clients/client_{cid}"
            ),
        )
        for cid in range(num_clients)
    ]

    # ── Two embedding tables: cache_in (read), cache_out (write/swap) ─────────
    all_gids = torch.cat([train_list[cid].global_nid for cid in range(num_clients)])
    num_nodes = int(all_gids.max().item()) + 1
    cache_in = EmbeddingTable(
        num_nodes=num_nodes, num_layers=args.num_layers, hidden_dim=args.hidden_dim,
    )
    cache_out = EmbeddingTable(
        num_nodes=num_nodes, num_layers=args.num_layers, hidden_dim=args.hidden_dim,
    )

    # ── Partition integrity check (once per seed, before training) ────────────
    _check_partition_integrity(train_list, num_nodes)

    # ── Coverage / payload CSV setup ──────────────────────────────────────────
    coverage_csv_dir = (
        f"./results/metrics/federated_logs/"
        f"layerwise_exchange_per_epoch{label_suffix}/{run_tag}/{num_clients_cfg}_clients"
    )
    os.makedirs(coverage_csv_dir, exist_ok=True)
    coverage_csv_path = os.path.join(coverage_csv_dir, f"exchange_coverage_seed{seed}.csv")
    coverage_csv_header = [
        "epoch", "layer", "cache_attempts", "cache_hits",
        "hit_rate_pct", "pull_bytes_mb", "push_bytes_mb",
    ]
    if not os.path.exists(coverage_csv_path):
        with open(coverage_csv_path, "w", newline="") as f:
            csv.writer(f).writerow(coverage_csv_header)

    # ── Checkpointing ─────────────────────────────────────────────────────────
    ckpt_dir = f"./checkpoints/layerwise_exchange_per_epoch{label_suffix}/{run_tag}/{num_clients_cfg}_clients"
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(ckpt_dir, f"seed{seed}_{run_id}_best.pt")
    best_val_pr_auc = float("-inf")

    # ── Pre-training round (OptimES § 3.2.1) ──────────────────────────────────
    # Synchronise to a single starting model so the cold-start cache is
    # consistent across clients. Then each client computes h^1..h^{L-1} for
    # its owned nodes (no injection — cache_in is empty / cold zeros for
    # ghost positions during this single forward).
    _fedavg_aggregate(tasks)
    pretraining_writes = _full_forward_collect(
        tasks, inject_cache=None, collect_cache=cache_in, device=device,
    )
    pretraining_mb = pretraining_writes * args.hidden_dim * 4 / (1024 ** 2)
    print(
        f"[Pre-training] Wrote {pretraining_writes} cache entries "
        f"({pretraining_mb:.2f} MB)"
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, args.global_epochs + 1):

        # Synchronise parameters across all clients via FedAvg before training.
        _fedavg_aggregate(tasks)

        # Independent per-client mini-batch training with frozen cache_in.
        _, stats = _per_epoch_train_epoch(tasks, cache_in, device)
        cache_attempts = stats['cache_attempts']
        cache_hits = stats['cache_hits']

        # Push phase (OptimES § 3.2.2): each client uses its OWN diverged
        # local model to recompute embeddings into cache_out, reading ghosts
        # from the still-frozen cache_in.
        push_writes = _full_forward_collect(
            tasks, inject_cache=cache_in, collect_cache=cache_out, device=device,
        )
        push_writes_per_layer = push_writes // args.num_layers if args.num_layers > 0 else 0

        # Synchronise post-push for consensus eval and checkpointing.
        _fedavg_aggregate(tasks)

        # Swap caches: next epoch reads from the freshly-pushed cache.
        cache_in, cache_out = cache_out, cache_in

        # ── Coverage / payload report ─────────────────────────────────────────
        total_attempts = sum(cache_attempts)
        total_hits = sum(cache_hits)
        total_pull_mb = total_hits * args.hidden_dim * 4 / (1024 ** 2)
        total_push_mb = push_writes * args.hidden_dim * 4 / (1024 ** 2)

        layer_parts = [
            f"L{l}: {100.0 * cache_hits[l] / max(cache_attempts[l], 1):.1f}%"
            f" ({cache_hits[l]}/{cache_attempts[l]})"
            for l in range(args.num_layers)
        ]
        print(
            f"[Exchange] {' | '.join(layer_parts)} | "
            f"pull {total_pull_mb:.1f} MB | push {total_push_mb:.2f} MB"
        )

        # ── Append to coverage CSV ────────────────────────────────────────────
        with open(coverage_csv_path, "a", newline="") as f:
            w = csv.writer(f)
            for l in range(args.num_layers):
                attempts_l = cache_attempts[l]
                hits_l = cache_hits[l]
                pct = 100.0 * hits_l / attempts_l if attempts_l > 0 else 0.0
                pull_l_mb = hits_l * args.hidden_dim * 4 / (1024 ** 2)
                push_l_mb = push_writes_per_layer * args.hidden_dim * 4 / (1024 ** 2)
                w.writerow([
                    epoch, l, attempts_l, hits_l,
                    f"{pct:.4f}", f"{pull_l_mb:.4f}", f"{push_l_mb:.4f}",
                ])
            total_pct = 100.0 * total_hits / total_attempts if total_attempts > 0 else 0.0
            w.writerow([
                epoch, "total", total_attempts, total_hits,
                f"{total_pct:.4f}", f"{total_pull_mb:.4f}", f"{total_push_mb:.4f}",
            ])

        # ── Validation on the consensus (post-FedAvg) model ───────────────────
        global_model = tasks[0].model
        global_criterion = tasks[0].criterion

        val_losses, val_f1s, val_pr_aucs = _per_epoch_eval_epoch(
            tasks, val_loaders, cache_in, device,
        )

        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_val_f1 = torch.stack(val_f1s).mean(dim=0)
        avg_val_pr_auc = torch.stack(val_pr_aucs).mean(dim=0)
        val_macro_f1 = avg_val_f1.mean().item()
        val_macro_pr_auc = avg_val_pr_auc.mean().item()

        # Diagnostic train-loss on client 0's training data.
        train_loss, _, _, _ = evaluate_epoch(
            global_model, tasks[0].train_loader, global_criterion, device, tasks[0].use_port_ids,
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

    # ── Test evaluation ───────────────────────────────────────────────────────
    best_sd = torch.load(best_ckpt_path, map_location=device)
    for task in tasks:
        task.model.load_state_dict(best_sd)

    # Refresh cache_in once more from the best-checkpoint model, so test-time
    # injection reflects the same model used for the test forward — same
    # principle as during training (cache always comes from the most recent
    # push pass of the model now in use).
    _full_forward_collect(
        tasks, inject_cache=cache_in, collect_cache=cache_out, device=device,
    )
    cache_in, cache_out = cache_out, cache_in

    test_losses, test_f1s, test_pr_aucs = _per_epoch_eval_epoch(
        tasks, test_loaders, cache_in, device,
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
        print(f"[Seed {seed}] Starting per-epoch layer-wise exchange ({num_clients} clients)...")
        print(f"{'='*60}")
        client_results = run_exchange_experiment(
            train_list, val_list, test_list, args, device, seed, run_id,
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
        f"[Results] Per-epoch layer-wise exchange (OptimES-style) — "
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
    out_csv = f"./results/metrics/federated_logs/layerwise_exchange_per_epoch{label_suffix}_results.csv"
    out_csv_auc = f"./results/metrics/federated_logs/layerwise_exchange_per_epoch{label_suffix}_pr_auc_results.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    model_name_str = (
        f"Layer-wise exchange per-epoch (OptimES-style + FedAvg) | "
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
