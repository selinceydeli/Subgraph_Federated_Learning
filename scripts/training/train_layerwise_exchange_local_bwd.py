#!/usr/bin/env python3
"""
Layer-wise Embedding Exchange with Fully Local Training + backward coupling.

Combines the two ablations:
  • Fully local training (as in ``train_layerwise_exchange_local.py``):
    each client trains its OWN model + optimizer, initialised from its own
    derived seed, with NO FedAvg parameter aggregation at any point — so
    client models diverge throughout training.
  • Backward gradient coupling (as in ``train_layerwise_exchange_bwd.py``):
    the shared embedding cache stores LIVE (autograd-tracked, on-device)
    tensor references instead of detached CPU copies, and each step uses a
    single combined ``sum(per-client losses)`` back-propagated once. The
    combined backward routes each consumer client's ∂L/∂emb_g — for every
    owned node g consumed across all clients — through the natural chain
    rule back into the OWNING client's parameters.

This isolates the value of *gradient-coupled* embedding exchange from
parameter averaging: clients never share weights, but a node owner's
gradient still reflects every consumer's loss on its embeddings. The
per-step exchange step (``_synchronous_train_epoch_live``) is reused
verbatim from the FedAvg bwd variant — it already drives per-client
``zero_grad()`` / ``step()``, so dropping FedAvg yields the fully-local
semantics directly.

Each epoch:
  1. NO FedAvg — each client keeps its own (divergent) weights.
  2. Synchronous per-step exchange training using LiveEmbeddingCache and
     one combined backward per step; each client's optimizer applies the
     accumulated (cross-client-coupled) gradient to its own params.
  3. Each client is validated and checkpointed independently via the
     forward-only synchronous exchange (eval is no_grad, so the simpler
     detached EmbeddingTable path is used).

Memory note: combined backward keeps every active client's full forward
activation graph alive until the single .backward() call. With many
clients and large batches this can dominate VRAM; if OOM occurs, reduce
batch_size in configs/pna_configs.json or num_clients in
configs/fed_configs.json.

Usage:
    python3 -m scripts.training.train_layerwise_exchange_local_bwd
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
# bwd-coupled training step (live cache + combined backward) is reused from
# the FedAvg bwd variant; it already does per-client zero_grad/step, so it
# works unchanged for fully-local training once FedAvg is dropped.
from scripts.training.train_layerwise_exchange import (
    _prepare_client_state,
    _check_partition_integrity,
    _synchronous_eval_epoch,
)
from scripts.training.train_layerwise_exchange_bwd import (
    _exchange_one_step_live,
    _synchronous_train_epoch_live,
)

PNA_CONFIG_PATH = "./configs/pna_configs.json"
FED_CONFIG_PATH = "./configs/fed_configs.json"

TASKS = ["deg-in", "deg-out", "fan-in", "fan-out", "C2", "C3", "C4", "C5", "C6", "S-G", "B-C"]

# Set True once during development to verify cross-client gradient flow.
# When enabled, client 0's loss is numerically zeroed for the first step of
# the first epoch — any nonzero grad on client 0's params then proves
# gradient is routed through ghost-emb splices from other clients' losses.
# Reused (via _synchronous_train_epoch_live) from train_layerwise_exchange_bwd,
# which reads its own module-level CHECK_GRADIENTS_BWD flag.
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
# Main experiment loop (one seed)
# ──────────────────────────────────────────────────────────────────────────────

def run_exchange_local_bwd_experiment(train_list, val_list, test_list, args, device, seed, run_id):
    """
    Run the bwd-coupled layer-wise exchange protocol with fully local training
    for one seed. Each client has independently initialised and trained model
    weights — no FedAvg aggregation is performed at any point — but the
    combined backward still routes consumer gradients through the live cache
    into each owner's parameters.

    Returns a list of per-client dicts with test_f1_per_task and
    test_pr_auc_per_task.
    """
    num_clients = len(train_list)
    num_clients_cfg = getattr(args, "num_clients", num_clients)
    label_suffix = "_local_labels" if getattr(args, "use_local_labels", False) else ""
    strategy = getattr(args, "partition_strategy", "partition_aware")
    cross_suffix = "with_cross_edges" if getattr(args, "include_cross_edges", False) else "without_cross_edges"
    run_tag = f"{strategy}_{cross_suffix}"

    # ── Initialise one independent model + optimizer per client ───────────────
    # Each client gets its own derived seed so models diverge from the start.
    tasks = []
    for cid in range(num_clients):
        set_seed(seed + cid)
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
            model_name=f"layerwise_exchange_local_bwd_client_{cid}",
            seed=seed,
            tasks=TASKS,
            out_dir=(
                f"./results/metrics/federated_logs/"
                f"layerwise_exchange_local_bwd{label_suffix}/{run_tag}/{num_clients_cfg}_clients/client_{cid}"
            ),
        )
        for cid in range(num_clients)
    ]

    # ── Live cache for training (one combined backward per step) and a
    # separate forward-only EmbeddingTable for eval (eval is no_grad, so the
    # simpler detached path is fine and avoids GPU memory pressure). ──────────
    all_gids = torch.cat([train_list[cid].global_nid for cid in range(num_clients)])
    num_nodes = int(all_gids.max().item()) + 1
    cache = LiveEmbeddingCache(num_clients=num_clients, num_layers=args.num_layers)
    eval_table = EmbeddingTable(
        num_nodes=num_nodes,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
    )

    # ── Partition integrity check (once per seed, before training) ────────────
    _check_partition_integrity(train_list, num_nodes)

    # ── Coverage CSV setup ────────────────────────────────────────────────────
    coverage_csv_dir = (
        f"./results/metrics/federated_logs/"
        f"layerwise_exchange_local_bwd{label_suffix}/{run_tag}/{num_clients_cfg}_clients"
    )
    os.makedirs(coverage_csv_dir, exist_ok=True)
    coverage_csv_path = os.path.join(coverage_csv_dir, f"exchange_coverage_seed{seed}.csv")
    coverage_csv_header = [
        "epoch", "layer", "remote_total", "remote_served", "coverage_pct", "bytes_communicated_mb"
    ]
    if not os.path.exists(coverage_csv_path):
        with open(coverage_csv_path, "w", newline="") as f:
            csv.writer(f).writerow(coverage_csv_header)

    # ── Per-client checkpointing ──────────────────────────────────────────────
    ckpt_dir = f"./checkpoints/layerwise_exchange_local_bwd{label_suffix}/{run_tag}/{num_clients_cfg}_clients"
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt_paths = [
        os.path.join(ckpt_dir, f"client_{cid}_seed{seed}_{run_id}_best.pt")
        for cid in range(num_clients)
    ]
    best_val_pr_aucs = [float("-inf")] * num_clients

    local_epochs = getattr(args, 'local_epochs', 1)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, args.global_epochs + 1):

        # No FedAvg aggregation — each client keeps its own weights.

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

        # ── Coverage report ───────────────────────────────────────────────────
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

        # ── Per-client validation via synchronous layer-wise exchange ────────
        # Every client evaluates its own val partition, but all clients
        # participate in the per-step exchange — so remote embeddings injected
        # into client A come from client B's (divergent) model, exactly as in
        # training.
        val_losses, val_f1s, val_pr_aucs = _synchronous_eval_epoch(
            tasks, val_loaders, eval_table, device
        )

        # Per-client best-model checkpointing
        for cid, task in enumerate(tasks):
            client_pr_auc = val_pr_aucs[cid].mean().item()
            if client_pr_auc > best_val_pr_aucs[cid]:
                best_val_pr_aucs[cid] = client_pr_auc
                torch.save(task.model.state_dict(), best_ckpt_paths[cid])

        avg_val_loss     = sum(val_losses) / len(val_losses)
        avg_val_f1       = torch.stack(val_f1s).mean(dim=0)
        avg_val_pr_auc   = torch.stack(val_pr_aucs).mean(dim=0)
        val_macro_f1     = avg_val_f1.mean().item()
        val_macro_pr_auc = avg_val_pr_auc.mean().item()

        # Train-loss estimate on client 0's training data (diagnostic only;
        # kept as plain local eval so it doesn't depend on all clients).
        train_loss, _, _, _ = evaluate_epoch(
            tasks[0].model, tasks[0].train_loader, tasks[0].criterion, device, tasks[0].use_port_ids
        )

        for cid in range(num_clients):
            append_epoch_csv(
                epoch_csv_paths[cid], epoch, train_loss,
                val_losses[cid], val_f1s[cid], val_pr_aucs[cid],
            )

        print(
            f"[Seed {seed}] Epoch {epoch:03d} | "
            f"train {train_loss:.4f} | val {avg_val_loss:.4f} | "
            f"val macro-minF1 {100 * val_macro_f1:.2f}% | "
            f"val macro-PR-AUC {100 * val_macro_pr_auc:.2f}%"
        )

    # ── Test evaluation ───────────────────────────────────────────────────────
    # Load each client's own best checkpoint into its model, then run one
    # synchronous layer-wise exchange pass across all clients.
    for cid, task in enumerate(tasks):
        task.model.load_state_dict(
            torch.load(best_ckpt_paths[cid], map_location=device)
        )

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
        print(f"[Seed {seed}] Starting bwd-coupled layer-wise exchange (local) training ({num_clients} clients)...")
        print(f"{'='*60}")
        client_results = run_exchange_local_bwd_experiment(
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
        f"[Results] Layer-wise exchange (local, no FedAvg) + bwd coupling — "
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
    out_csv     = f"./results/metrics/federated_logs/layerwise_exchange_local_bwd{label_suffix}_results.csv"
    out_csv_auc = f"./results/metrics/federated_logs/layerwise_exchange_local_bwd{label_suffix}_pr_auc_results.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    model_name_str = (
        f"Layer-wise exchange (local, no FedAvg) + bwd coupling (live cache + combined backward) | "
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
