#!/usr/bin/env python3
"""
Layer-wise Embedding Exchange with Sync-SGD + backward gradient coupling.

Mirrors `train_layerwise_exchange_sync_sgd.py` in its Sync-SGD structure
(one shared model + one shared Adam optimizer; per-step gradient averaging
weighted by each client's owned-sample share), but replaces the persistent
detached CPU EmbeddingTable with a per-step LiveEmbeddingCache. The
per-step loss is a single weighted combined `sum(w_cid * loss_cid)`
back-propagated once; the combined backward routes each consumer's
∂L/∂emb_g into the shared model via the natural chain rule through the
ghost-emb splices.

The persistent-cache machinery and OptimES-style pretraining round from
the forward-only Sync-SGD variant are dropped: a live cache cannot
survive across optimizer.step() (autograd graphs are freed) and would
also retain GPU memory indefinitely. The cache is reset every step.

Eval uses a separate forward-only EmbeddingTable under no_grad (eval is
gradient-irrelevant, so detached fetches are fine and cheaper). This
matches the forward-only LE variant's eval mechanism.

Memory note: combined backward keeps every active client's full forward
activation graph alive until the single .backward() call. With many
clients and large batches this can dominate VRAM; if OOM occurs, reduce
batch_size in configs/pna_configs.json or num_clients in
configs/fed_configs.json.

Usage:
    python3 -m scripts.training.train_layerwise_exchange_sync_sgd_bwd
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
from utils.train_utils import ensure_node_features, evaluate_epoch
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
# integrity check, and the forward-only eval pass are all unchanged. The
# bwd-coupled Sync-SGD variant only changes the per-step training loop
# (live cache + single weighted combined backward).
from scripts.training.train_layerwise_exchange import (
    _prepare_client_state,
    _check_partition_integrity,
    _synchronous_eval_epoch,
)
from scripts.training.train_layerwise_exchange_bwd import (
    _build_gid_to_owner_index,
    _exchange_one_step_live,
)

PNA_CONFIG_PATH = "./configs/pna_configs.json"
FED_CONFIG_PATH = "./configs/fed_configs.json"

TASKS = ["deg-in", "deg-out", "fan-in", "fan-out", "C2", "C3", "C4", "C5", "C6", "S-G", "B-C"]

# Set True once during development to verify cross-client gradient flow.
# In Sync-SGD all clients share one model, so the assertion checks that
# the shared model's gradient changes when only client 1's loss is active
# (after numerically zeroing client 0's loss for the first step of the
# first epoch). Nonzero grad implies splice → owner routing works.
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
# Sync-SGD + bwd-coupled training step
# ──────────────────────────────────────────────────────────────────────────────

def _synchronous_train_epoch_sync_sgd_bwd(
    tasks, cache, shared_optim, device,
    *, gid_to_owner, gid_to_local_idx,
    grad_check_epoch=None, current_epoch=None,
):
    """
    One pass of bwd-coupled layer-wise exchange + per-step gradient
    averaging into one shared Adam state.

    Per step:
      1. Each client draws one batch; build client_state via
         _prepare_client_state. All tasks share `tasks[0].model` /
         `tasks[0].optimizer`.
      2. _exchange_one_step_live runs the per-layer write→inject loop
         with the LIVE cache (no detach, autograd preserved through ghost
         splices). Cache is reset every step.
      3. Compute each client's per-owned-sample loss and a per-client
         weight w_cid = owned_count_cid / total_step_count.
      4. Build combined = sum(w_cid * loss_cid) and call combined.backward()
         once. This is mathematically identical to the forward-only
         variant's per-client weighted backward (the gradients accumulate
         into the same shared params either way), but additionally routes
         each consumer's grad through the ghost splices into the shared
         model via the chain rule.
      5. One shared_optim.step() applies the averaged gradient.

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

    shared_model = tasks[0].model

    step_idx = 0
    for _ in range(num_steps):
        client_states = []
        for task, it in zip(tasks, iters):
            client_states.append(_prepare_client_state(task, next(it), device))

        owned_counts = []
        for s in client_states:
            if s['owned'] is not None:
                if s['B'] is not None and s['B'] != s['N']:
                    owned_counts.append(int(s['owned'][:s['B']].sum().item()))
                else:
                    owned_counts.append(int(s['owned'].sum().item()))
            else:
                owned_counts.append(s['B'] if s['B'] is not None else s['N'])
        total_step_count = max(sum(owned_counts), 1)

        shared_model.train()
        shared_optim.zero_grad()

        step_total, step_served = _exchange_one_step_live(
            client_states, cache, num_layers, device,
            gid_to_owner=gid_to_owner, gid_to_local_idx=gid_to_local_idx,
            track_coverage=True,
        )
        for l in range(num_layers):
            remote_total[l]  += step_total[l]
            remote_served[l] += step_served[l]

        losses_list = []
        for cid, s in enumerate(client_states):
            task = s['task']
            logits = shared_model.compute_output(s['x'])

            out_used = logits[:s['B']] if s['B'] is not None else logits
            y_batch  = s['y'][:s['B']] if s['B'] is not None else s['y']

            if s['owned'] is not None and (s['B'] is None or s['B'] == s['N']):
                out_used = out_used[s['owned']]
                y_batch  = y_batch[s['owned']]
                count = int(s['owned'].sum().item())
            else:
                count = s['B'] if s['B'] is not None else s['N']

            loss = task.criterion(out_used, y_batch.float())
            w = owned_counts[cid] / total_step_count
            losses_list.append(loss * w)

            total_loss += loss.item() * count
            total_count += count

        do_grad_check = (
            CHECK_GRADIENTS_BWD and step_idx == 0
            and grad_check_epoch is not None and current_epoch == grad_check_epoch
        )
        if do_grad_check and len(losses_list) >= 2:
            losses_list[0] = 0.0 * losses_list[0]

        combined = torch.stack(losses_list).sum()
        combined.backward()

        if do_grad_check and len(losses_list) >= 2:
            p0 = next(shared_model.parameters())
            g0 = p0.grad
            assert g0 is not None and g0.abs().sum().item() > 0, (
                "[BwdGradCheck] FAIL: shared model has no grad with loss_0=0 "
                "in a 2-client step — cross-client gradient is NOT flowing. "
                "Check LiveEmbeddingCache.fetch_remote and the torch.where "
                "x_fill branch."
            )
            print(
                f"[BwdGradCheck] PASS: cross-client gradient confirmed "
                f"(|grad_p0| sum = {g0.abs().sum().item():.4e})"
            )

        shared_optim.step()

        step_idx += 1

    exchange_stats = {
        'remote_total':  remote_total,
        'remote_served': remote_served,
    }
    return total_loss / max(total_count, 1), exchange_stats


# ──────────────────────────────────────────────────────────────────────────────
# Main experiment loop (one seed)
# ──────────────────────────────────────────────────────────────────────────────

def run_exchange_sync_sgd_bwd_experiment(train_list, val_list, test_list, args, device, seed, run_id):
    num_clients = len(train_list)
    num_clients_cfg = getattr(args, "num_clients", num_clients)
    label_suffix = "_local_labels" if getattr(args, "use_local_labels", False) else ""
    strategy = getattr(args, "partition_strategy", "partition_aware")
    cross_suffix = "with_cross_edges" if getattr(args, "include_cross_edges", False) else "without_cross_edges"
    run_tag = f"{strategy}_{cross_suffix}"

    # Sync-SGD: one shared model + one shared Adam across all clients.
    set_seed(seed)
    tasks = [NodeClsTask(args, cid, train_list[cid], "./data", device) for cid in range(num_clients)]
    shared_model = tasks[0].model
    shared_optim = tasks[0].optimizer
    for cid in range(1, num_clients):
        tasks[cid].model = shared_model
        tasks[cid].optimizer = shared_optim

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
            model_name=f"layerwise_exchange_sync_sgd_bwd_client_{cid}",
            seed=seed,
            tasks=TASKS,
            out_dir=(
                f"./results/metrics/federated_logs/"
                f"layerwise_exchange_sync_sgd_bwd{label_suffix}/{run_tag}/{num_clients_cfg}_clients/client_{cid}"
            ),
        )
        for cid in range(num_clients)
    ]

    # Live cache for training; separate detached eval table (no grad needed).
    all_gids = torch.cat([train_list[cid].global_nid for cid in range(num_clients)])
    num_nodes = int(all_gids.max().item()) + 1
    cache = LiveEmbeddingCache(num_clients=num_clients, num_layers=args.num_layers)
    eval_table = EmbeddingTable(
        num_nodes=num_nodes,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
    )

    _check_partition_integrity(train_list, num_nodes)

    gid_to_owner, gid_to_local_idx = _build_gid_to_owner_index(
        train_list, num_nodes, device,
    )

    coverage_csv_dir = (
        f"./results/metrics/federated_logs/"
        f"layerwise_exchange_sync_sgd_bwd{label_suffix}/{run_tag}/{num_clients_cfg}_clients"
    )
    os.makedirs(coverage_csv_dir, exist_ok=True)
    coverage_csv_path = os.path.join(coverage_csv_dir, f"exchange_coverage_seed{seed}.csv")
    coverage_csv_header = [
        "epoch", "layer", "remote_total", "remote_served", "coverage_pct", "bytes_communicated_mb"
    ]
    if not os.path.exists(coverage_csv_path):
        with open(coverage_csv_path, "w", newline="") as f:
            csv.writer(f).writerow(coverage_csv_header)

    ckpt_dir = f"./checkpoints/layerwise_exchange_sync_sgd_bwd{label_suffix}/{run_tag}/{num_clients_cfg}_clients"
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(ckpt_dir, f"seed{seed}_{run_id}_best.pt")
    best_val_pr_auc = float("-inf")

    local_epochs = getattr(args, 'local_epochs', 1)

    for epoch in range(1, args.global_epochs + 1):
        epoch_remote_total  = [0] * args.num_layers
        epoch_remote_served = [0] * args.num_layers
        for _ in range(local_epochs):
            _, step_stats = _synchronous_train_epoch_sync_sgd_bwd(
                tasks, cache, shared_optim, device,
                gid_to_owner=gid_to_owner, gid_to_local_idx=gid_to_local_idx,
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

        # Eval via the forward-only exchange (grad-irrelevant).
        val_losses, val_f1s, val_pr_aucs = _synchronous_eval_epoch(
            tasks, val_loaders, eval_table, device,
        )

        avg_val_loss     = sum(val_losses) / len(val_losses)
        avg_val_f1       = torch.stack(val_f1s).mean(dim=0)
        avg_val_pr_auc   = torch.stack(val_pr_aucs).mean(dim=0)
        val_macro_f1     = avg_val_f1.mean().item()
        val_macro_pr_auc = avg_val_pr_auc.mean().item()

        train_loss, _, _, _ = evaluate_epoch(
            shared_model, tasks[0].train_loader, tasks[0].criterion, device, tasks[0].use_port_ids
        )

        for cid in range(num_clients):
            append_epoch_csv(
                epoch_csv_paths[cid], epoch, train_loss,
                val_losses[cid], val_f1s[cid], val_pr_aucs[cid],
            )

        if val_macro_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_macro_pr_auc
            torch.save(shared_model.state_dict(), best_ckpt_path)

        print(
            f"[Seed {seed}] Epoch {epoch:03d} | "
            f"train {train_loss:.4f} | val {avg_val_loss:.4f} | "
            f"val macro-minF1 {100 * val_macro_f1:.2f}% | "
            f"val macro-PR-AUC {100 * val_macro_pr_auc:.2f}%"
        )

    shared_model.load_state_dict(torch.load(best_ckpt_path, map_location=device))

    test_losses, test_f1s, test_pr_aucs = _synchronous_eval_epoch(
        tasks, test_loaders, eval_table, device,
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
    val_list   = load_client_graphs(val_dir,   num_clients)
    test_list  = load_client_graphs(test_dir,  num_clients)
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
        print(f"[Seed {seed}] Starting bwd-coupled layer-wise exchange + Sync-SGD ({num_clients} clients)...")
        print(f"{'='*60}")
        client_results = run_exchange_sync_sgd_bwd_experiment(
            train_list, val_list, test_list, args, device, seed, run_id
        )

        seed_f1     = torch.stack([r["test_f1_per_task"]     for r in client_results], dim=0)
        seed_pr_auc = torch.stack([r["test_pr_auc_per_task"] for r in client_results], dim=0)
        per_seed_mean_f1.append(seed_f1.mean(dim=0))
        per_seed_mean_pr_auc.append(seed_pr_auc.mean(dim=0))

    all_seeds_f1 = torch.stack(per_seed_mean_f1, dim=0)
    mean_f1 = all_seeds_f1.mean(dim=0)
    std_f1  = all_seeds_f1.std(dim=0, unbiased=False)
    macro_mean = mean_f1.mean().item() * 100

    all_seeds_pr_auc = torch.stack(per_seed_mean_pr_auc, dim=0)
    mean_pr_auc = all_seeds_pr_auc.mean(dim=0)
    std_pr_auc  = all_seeds_pr_auc.std(dim=0, unbiased=False)
    macro_pr_auc = mean_pr_auc.mean().item() * 100

    print(f"\n{'='*60}")
    print(
        f"[Results] Layer-wise exchange + Sync-SGD + bwd coupling — "
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
    out_csv     = f"./results/metrics/federated_logs/layerwise_exchange_sync_sgd_bwd{label_suffix}_results.csv"
    out_csv_auc = f"./results/metrics/federated_logs/layerwise_exchange_sync_sgd_bwd{label_suffix}_pr_auc_results.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    model_name_str = (
        f"Layer-wise exchange + Sync-SGD + bwd coupling (live cache + combined weighted backward) | "
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
