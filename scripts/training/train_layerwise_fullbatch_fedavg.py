#!/usr/bin/env python3
"""
Layer-wise Embedding Exchange — Full-batch Phase 1 + FedAvg variant.

Each epoch proceeds in three steps:

  1. FedAvg parameter aggregation: all client models are averaged so that
     every client starts the epoch with identical parameters θ^(k).

  2. Full-batch embedding collection (no gradient):
     Every client runs a complete forward pass over its entire training
     subgraph using θ^(k).  After each conv layer l, owned-node embeddings
     are written into a shared EmbeddingTable at depth l.  Because this
     pass is full-batch (no neighbor sampling), every owned node across
     every client is guaranteed to have an entry in the table.

  3. Mini-batch training with injection (local_epochs passes):
     Each client trains with its mini-batch loader.  At each conv layer l,
     remote-node embeddings are replaced with the Phase-2 table values at
     depth l — always from the owning client, always at the correct depth.
     Because full-batch collection guarantees complete coverage, no remote
     node ever receives a zero fallback.

Compared to train_layerwise_exchange.py (synchronous per-step design):
  - Coverage: guaranteed complete here vs. partial (mini-batch dependent) there.
  - Staleness: Phase-2 embeddings are from θ^(k); as Phase 3 proceeds through
    local steps the model drifts, so later steps inject slightly stale
    embeddings.  With local_epochs=1 this drift is one full pass.
  - Simplicity: the two phases are independent and easier to reason about.

Usage:
    python3 -m scripts.training.train_layerwise_fullbatch_fedavg
"""

import os
import time
import json
from types import SimpleNamespace
from datetime import datetime

import torch

from utils.loader import load_client_graphs
from utils.seed import set_seed
from utils.metrics import append_f1_score_to_csv, append_pr_auc_to_csv, start_epoch_csv, append_epoch_csv
from utils.train_utils import (
    ensure_node_features,
    evaluate_epoch,
    _unpack_io,
    _augment_with_ego_and_get_seed_slice,
)
from utils.hetero import make_bidirected_hetero
from utils.graph_helpers import max_port_cols, check_and_strip_self_loops, build_hetero_neighbor_loader, build_full_eval_loader
from utils.layerwise_exchange import EmbeddingTable
from task.node_cls import NodeClsTask

PNA_CONFIG_PATH = "./configs/pna_configs.json"
FED_CONFIG_PATH = "./configs/fed_configs.json"

TASKS = ["deg-in", "deg-out", "fan-in", "fan-out", "C2", "C3", "C4", "C5", "C6", "S-G", "B-C"]


# ──────────────────────────────────────────────────────────────────────────────
# Config / data helpers  (identical to train_local_baseline.py)
# ──────────────────────────────────────────────────────────────────────────────

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
        num_clients=partition_cfg["num_clients"],
        include_cross_edges=partition_cfg["include_cross_edges"],
        use_local_labels=partition_cfg.get("use_local_labels", False),
    )


def resolve_data_dirs(partition_cfg):
    num_clients = partition_cfg["num_clients"]
    cross_suffix = "with_cross_edges" if partition_cfg["include_cross_edges"] else "without_cross_edges"
    local_suffix = "_local_labels" if partition_cfg.get("use_local_labels", False) else ""
    base = f"./data/fed_partition_aware_splits_{cross_suffix}{local_suffix}/{num_clients}_clients"
    return (
        f"{base}/train/clients",
        f"{base}/val/clients",
        f"{base}/test/clients",
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
# FedAvg parameter aggregation
# ──────────────────────────────────────────────────────────────────────────────

def _fedavg_aggregate(tasks):
    """
    Weighted FedAvg: aggregate all client models into a single shared state.
    Each client's contribution is weighted by its number of owned training nodes.
    After aggregation, all clients hold identical parameters.
    """
    counts = [t.num_samples for t in tasks]
    total = sum(counts)
    weights = [c / total for c in counts]

    ref_sd = tasks[0].model.state_dict()
    global_sd = {}
    for k in ref_sd:
        global_sd[k] = sum(
            w * t.model.state_dict()[k].float()
            for w, t in zip(weights, tasks)
        )

    device = next(tasks[0].model.parameters()).device
    for t in tasks:
        t.model.load_state_dict({k: v.to(device) for k, v in global_sd.items()})


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2 — Full-batch embedding collection (no gradient)
# ──────────────────────────────────────────────────────────────────────────────

def _collect_embeddings(task: NodeClsTask, table: EmbeddingTable, device: torch.device) -> None:
    """
    Run a full-batch forward pass on the client's entire training subgraph
    (no gradient).  Owned-node embeddings at each conv layer are written into
    the shared EmbeddingTable.

    Because this pass covers all edges of the subgraph (no neighbor sampling),
    every owned node is guaranteed to have an entry in the table at every depth.
    This eliminates the zero-injection problem of mini-batch collection.

    Ego IDs are set to zero for all nodes in this pass — no meaningful seed
    distinction exists in a full-batch forward pass.
    """
    task.model.eval()
    hetero = task.hetero_data

    with torch.no_grad():
        x_raw = hetero['n'].x.to(device)

        if task.ego_dim > 0:
            ego = torch.zeros(x_raw.size(0), task.ego_dim, device=device)
            x_aug = torch.cat([x_raw, ego], dim=-1)
        else:
            x_aug = x_raw

        x_dict = {'n': x_aug}
        edge_index_dict = {
            ('n', 'fwd', 'n'): hetero[('n', 'fwd', 'n')].edge_index.to(device),
            ('n', 'rev', 'n'): hetero[('n', 'rev', 'n')].edge_index.to(device),
        }

        edge_attr_dict = None
        if task.use_port_ids:
            edge_attr_dict = {}
            for rel in [('n', 'fwd', 'n'), ('n', 'rev', 'n')]:
                if hasattr(hetero[rel], 'edge_attr'):
                    ea = hetero[rel].edge_attr.to(device)
                    if ea.dtype != torch.long:
                        ea = ea.long()
                    edge_attr_dict[rel] = ea

        owned_mask = hetero['n'].owned_mask.to(device)
        global_nids = hetero['n'].global_nid.to(device)

        task.model.forward_layerwise(
            x_dict, edge_index_dict,
            edge_attr_dict=edge_attr_dict,
            owned_mask=owned_mask,
            global_nids=global_nids,
            inject_table=None,
            collect_into=table,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3 — Mini-batch training with injection
# ──────────────────────────────────────────────────────────────────────────────

def _train_with_exchange(
    task: NodeClsTask,
    table: EmbeddingTable,
    device: torch.device,
) -> float:
    """
    Run local_epochs mini-batch training passes for one client with embedding
    injection.  At each conv layer, remote-node embeddings are replaced with
    the full-batch Phase-2 values from the EmbeddingTable.

    Because Phase 2 guarantees complete coverage, every remote node receives
    the owning client's embedding at the correct layer depth.

    Returns the average training loss across all steps.
    """
    task.model.train()
    local_epochs = getattr(task.args, 'local_epochs', 1)
    total_loss = 0.0
    total_count = 0

    for _ in range(local_epochs):
        for batch in task.train_loader:
            batch = batch.to(device)
            x_in, edge_in, y_true, n_nodes, is_hetero = _unpack_io(batch)
            x_in_aug, y_used, B = _augment_with_ego_and_get_seed_slice(
                x_in, y_true, batch, is_hetero, task.model
            )

            edge_attr_dict = None
            if is_hetero and task.use_port_ids:
                edge_attr_dict = {}
                for rel in [('n', 'fwd', 'n'), ('n', 'rev', 'n')]:
                    if 'edge_attr' in batch[rel]:
                        ea = batch[rel].edge_attr
                        if ea.dtype != torch.long:
                            ea = ea.long()
                        edge_attr_dict[rel] = ea

            owned_mask = batch['n'].owned_mask if is_hetero else getattr(batch, 'owned_mask', None)
            global_nids = batch['n'].global_nid if is_hetero else getattr(batch, 'global_nid', None)

            task.optimizer.zero_grad()

            logits = task.model.forward_layerwise(
                x_in_aug, edge_in,
                edge_attr_dict=edge_attr_dict,
                owned_mask=owned_mask,
                global_nids=global_nids,
                inject_table=table,
                collect_into=None,
            )

            out_used = logits[:B] if B is not None else logits
            y_batch = y_used[:B] if B is not None else y_used

            if owned_mask is not None and (B is None or B == n_nodes):
                out_used = out_used[owned_mask]
                y_batch = y_batch[owned_mask]
                count = int(owned_mask.sum().item())
            else:
                count = B if B is not None else n_nodes

            loss = task.criterion(out_used, y_batch.float())
            loss.backward()
            task.optimizer.step()

            total_loss += loss.item() * count
            total_count += count

    return total_loss / max(total_count, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Main experiment loop (one seed)
# ──────────────────────────────────────────────────────────────────────────────

def run_exchange_experiment(train_list, val_list, test_list, args, device, seed, run_id):
    """
    Run the full-batch Phase-1 + FedAvg layer-wise exchange protocol for all
    clients under one seed.
    Returns a list of per-client dicts with test_f1_per_task and test_pr_auc_per_task.
    """
    num_clients = len(train_list)
    num_clients_cfg = getattr(args, "num_clients", num_clients)
    label_suffix = "_local_labels" if getattr(args, "use_local_labels", False) else ""

    # ── Initialise one independent model + optimizer per client ───────────────
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

    # ── Per-epoch CSV logging ─────────────────────────────────────────────────
    epoch_csv_paths = [
        start_epoch_csv(
            model_name=f"layerwise_fullbatch_fedavg_client_{cid}",
            seed=seed,
            tasks=TASKS,
            out_dir=(
                f"./results/metrics/federated_logs/"
                f"layerwise_fullbatch_fedavg{label_suffix}/{num_clients_cfg}_clients/client_{cid}"
            ),
        )
        for cid in range(num_clients)
    ]

    # ── Embedding table ───────────────────────────────────────────────────────
    all_gids = torch.cat([train_list[cid].global_nid for cid in range(num_clients)])
    num_nodes = int(all_gids.max().item()) + 1
    table = EmbeddingTable(
        num_nodes=num_nodes,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
    )

    # ── Checkpointing ─────────────────────────────────────────────────────────
    ckpt_dir = f"./checkpoints/layerwise_fullbatch_fedavg{label_suffix}/{num_clients_cfg}_clients"
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt_paths = [
        os.path.join(ckpt_dir, f"client_{cid}_seed{seed}_{run_id}_best.pt")
        for cid in range(num_clients)
    ]
    best_val_pr_auc = [float("-inf")] * num_clients

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, args.global_epochs + 1):

        # Step 1: synchronise parameters across all clients via FedAvg.
        # All clients now hold identical weights θ^(k) for this epoch.
        _fedavg_aggregate(tasks)

        # Step 2: full-batch embedding collection with θ^(k) (no gradient).
        # Every owned node at every layer depth is written to the table.
        table.reset()
        for task in tasks:
            _collect_embeddings(task, table, device)

        # Step 3: mini-batch training with injection.
        # Remote nodes always receive the owning client's embedding at the
        # correct depth — guaranteed by the full-batch collection above.
        for task in tasks:
            _train_with_exchange(task, table, device)

        # Aggregate parameters after training so validation is always performed
        # on the clean averaged model — consistent with train_fedavg.py which
        # evaluates the server model (post-aggregation) rather than diverged
        # per-client models.
        _fedavg_aggregate(tasks)

        # ── Validation, epoch logging, checkpointing ──────────────────────────
        for cid, task in enumerate(tasks):
            val_loss, _, val_f1, val_pr_auc = evaluate_epoch(
                task.model, val_loaders[cid], task.criterion, device, task.use_port_ids
            )
            train_loss, _, _, _ = evaluate_epoch(
                task.model, task.train_loader, task.criterion, device, task.use_port_ids
            )

            append_epoch_csv(epoch_csv_paths[cid], epoch, train_loss, val_loss, val_f1, val_pr_auc)

            val_macro_f1     = val_f1.mean().item()
            val_macro_pr_auc = val_pr_auc.mean().item()

            if val_macro_pr_auc > best_val_pr_auc[cid]:
                best_val_pr_auc[cid] = val_macro_pr_auc
                torch.save(task.model.state_dict(), best_ckpt_paths[cid])

            print(
                f"[Seed {seed}][Client {cid}] Epoch {epoch:03d} | "
                f"train {train_loss:.4f} | val {val_loss:.4f} | "
                f"val macro-minF1 {100 * val_macro_f1:.2f}% | "
                f"val macro-PR-AUC {100 * val_macro_pr_auc:.2f}%"
            )

    # ── Test evaluation ───────────────────────────────────────────────────────
    results = []
    for cid, task in enumerate(tasks):
        task.model.load_state_dict(torch.load(best_ckpt_paths[cid], map_location=device))
        test_loss, _, test_f1, test_pr_auc = evaluate_epoch(
            task.model, test_loaders[cid], task.criterion, device, task.use_port_ids
        )
        test_macro        = test_f1.mean().item()
        test_macro_pr_auc = test_pr_auc.mean().item()
        print(
            f"[Seed {seed}][Client {cid}] Best ckpt → "
            f"test_loss={test_loss:.4f} | "
            f"test macro-minF1={100 * test_macro:.2f}% | "
            f"test macro-PR-AUC={100 * test_macro_pr_auc:.2f}%"
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
    partition_cfg = fed_all["partition_aware_splits"]

    args = build_args(pna_cfg, fed_cfg, partition_cfg)

    print(
        f"[Config] global_epochs={args.global_epochs}, local_epochs={args.local_epochs}, "
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

    # Global port vocab — all clients must share the same embedding dimensions
    # so that _fedavg_aggregate does not crash on shape mismatches.
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

    seeds = [args.base_seed, args.base_seed + 1, args.base_seed + 2]
    per_seed_mean_f1 = []
    per_seed_mean_pr_auc = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"[Seed {seed}] Starting layerwise full-batch+FedAvg training ({num_clients} clients)...")
        print(f"{'='*60}")
        client_results = run_exchange_experiment(
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
        f"[Results] Layerwise full-batch+FedAvg oracle — "
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
    out_csv     = f"./results/metrics/federated_logs/layerwise_fullbatch_fedavg{label_suffix}_results.csv"
    out_csv_auc = f"./results/metrics/federated_logs/layerwise_fullbatch_fedavg{label_suffix}_pr_auc_results.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    model_name_str = (
        f"Layer-wise full-batch+FedAvg oracle | "
        f"num_clients={num_clients}, "
        f"cross_edges={args.include_cross_edges}, "
        f"local_labels={args.use_local_labels}, "
        f"global_epochs={args.global_epochs}, "
        f"local_epochs={args.local_epochs}, "
        f"use_port_ids={args.use_port_ids}, "
        f"use_ego_ids={args.use_ego_ids}, "
        f"num_layers={args.num_layers}, "
        f"neighbors_per_hop={args.neighbors_per_hop}, "
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
