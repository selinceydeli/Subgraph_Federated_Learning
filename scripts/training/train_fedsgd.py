#!/usr/bin/env python3
"""
FedSGD federated training: gradients are averaged every mini-batch step
into a single shared Adam optimizer state.

Semantics:
  - All clients share one model and one Adam optimizer (the "server").
  - Each step: every client draws one mini-batch from its own train loader,
    runs a forward pass through the shared model using its own criterion
    (local class weights), and does a weighted backward — weights are
    proportional to each client's owned-node count in that batch so that
    the total accumulated gradient equals the sample-weighted average of
    per-client gradients.
  - One optimizer.step() per sync step.

Why average gradients (not parameters)?
  NodeClsTask uses Adam. With Adam, averaging parameters after per-client
  local steps is NOT equivalent to averaging gradients with one shared
  optimizer, because each local optimizer would maintain its own first/
  second moments. Averaging gradients into one shared Adam state keeps the
  "single global optimizer" semantics classical FedSGD assumes.

This script is the "no layer-wise exchange" cell of the federated grid; the
companion `train_layerwise_exchange_fedsgd.py` adds per-step embedding
exchange on top of the same per-step gradient averaging.

Usage:
    python3 -m scripts.training.train_fedsgd
"""

import os
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
from utils.train_utils import (
    ensure_node_features,
    evaluate_epoch,
    _unpack_io,
    _augment_with_ego_and_get_seed_slice,
)
from utils.hetero import make_bidirected_hetero
from utils.graph_helpers import (
    max_port_cols,
    check_and_strip_self_loops,
    build_hetero_neighbor_loader,
    build_full_eval_loader,
)
from task.node_cls import NodeClsTask

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
# FedSGD step
# ──────────────────────────────────────────────────────────────────────────────

def _build_edge_attr_dict(batch, task, is_hetero):
    """Extract per-relation edge_attr dict for port-ID-enabled models."""
    if not (is_hetero and task.use_port_ids):
        return None
    d = {}
    for rel in [('n', 'fwd', 'n'), ('n', 'rev', 'n')]:
        if 'edge_attr' in batch[rel]:
            ea = batch[rel].edge_attr
            d[rel] = ea if ea.dtype == torch.long else ea.long()
    return d or None


def _fedsgd_train_epoch(shared_model, shared_optim, client_tasks, device):
    """
    One pass of FedSGD training — gradients are averaged every mini-batch step
    into the shared Adam optimizer.

    Per step:
      1. Each client draws one batch from its own train loader.
      2. Forward through the shared model using the batch; compute loss with
         the client's own criterion (local class weights).
      3. Weighted backward: scale each client's loss by
         (owned_count_c / total_owned_count_this_step) so that the accumulated
         gradient equals the sample-weighted average of per-client gradients.
      4. One shared_optim.step() applies the averaged gradient.

    Returns the mean loss across all observed owned samples.
    """
    iters = [iter(t.train_loader) for t in client_tasks]
    num_steps = min(len(t.train_loader) for t in client_tasks)

    total_loss = 0.0
    total_count = 0

    for _ in range(num_steps):
        batches = [next(it).to(device) for it in iters]

        # Count owned samples per batch for weighting.
        owned_counts = []
        for batch in batches:
            is_hetero = hasattr(batch, '__getitem__') and hasattr(batch, 'node_types')
            if is_hetero:
                owned = batch['n'].owned_mask
            else:
                owned = getattr(batch, 'owned_mask', None)
            # Fall back to seed count if owned_mask absent.
            owned_counts.append(
                int(owned.sum().item()) if owned is not None else batch.num_nodes
            )
        total_step_count = max(sum(owned_counts), 1)

        shared_model.train()
        shared_optim.zero_grad()

        for cid, (batch, task) in enumerate(zip(batches, client_tasks)):
            x_in, edge_in, y_true, n_nodes, is_hetero = _unpack_io(batch)
            x_in_aug, y_used, B = _augment_with_ego_and_get_seed_slice(
                x_in, y_true, batch, is_hetero, shared_model
            )
            edge_attr_dict = _build_edge_attr_dict(batch, task, is_hetero)

            if task.use_port_ids:
                logits = shared_model(
                    x_in_aug, edge_in, edge_attr_dict=edge_attr_dict
                )
            else:
                logits = shared_model(x_in_aug, edge_in)

            out_used = logits[:B] if B is not None else logits
            y_batch  = y_used[:B] if B is not None else y_used

            owned_mask = (
                batch['n'].owned_mask if is_hetero else getattr(batch, 'owned_mask', None)
            )
            if owned_mask is not None and (B is None or B == n_nodes):
                out_used = out_used[owned_mask]
                y_batch  = y_batch[owned_mask]
                count = int(owned_mask.sum().item())
            else:
                count = B if B is not None else n_nodes

            loss = task.criterion(out_used, y_batch.float())
            # Sample-weighted average: total gradient = sum(owned_c * grad_c) / total_count
            w = owned_counts[cid] / total_step_count
            (loss * w).backward()

            total_loss += loss.item() * count
            total_count += count

        shared_optim.step()

    return total_loss / max(total_count, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Main experiment loop (one seed)
# ──────────────────────────────────────────────────────────────────────────────

def run_fedsgd_experiment(train_list, val_list, test_list, args, device, seed, run_id):
    num_clients = len(train_list)
    label_suffix = "_local_labels" if getattr(args, "use_local_labels", False) else ""
    strategy = getattr(args, "partition_strategy", "partition_aware")
    cross_suffix = "with_cross_edges" if getattr(args, "include_cross_edges", False) else "without_cross_edges"
    run_tag = f"{strategy}_{cross_suffix}"

    # ── Instantiate one NodeClsTask per client for per-client train loaders
    # and per-client criteria. Then share the same model + optimizer across all
    # clients so there is exactly one Adam state.
    set_seed(seed)
    tasks = []
    for cid in range(num_clients):
        t = NodeClsTask(args, cid, train_list[cid], "./data", device)
        tasks.append(t)

    shared_model = tasks[0].model
    shared_optim = tasks[0].optimizer
    for cid in range(1, num_clients):
        tasks[cid].model = shared_model
        tasks[cid].optimizer = shared_optim

    val_loaders  = [make_eval_loader(val_list[cid],  tasks[cid], device, shuffle=True) for cid in range(num_clients)]
    test_loaders = [make_eval_loader(test_list[cid], tasks[cid], device, shuffle=True) for cid in range(num_clients)]

    epoch_csv_path = start_epoch_csv(
        model_name=f"fedsgd_seed{seed}",
        seed=seed,
        tasks=TASKS,
        out_dir=f"./results/metrics/federated_logs/fedsgd{label_suffix}/{run_tag}/{num_clients}_clients",
    )

    ckpt_dir = f"./checkpoints/fedsgd{label_suffix}/{run_tag}/{num_clients}_clients"
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(ckpt_dir, f"seed{seed}_{run_id}_best.pt")
    best_val_pr_auc = float("-inf")

    local_epochs = getattr(args, 'local_epochs', 1)

    for epoch in range(1, args.global_epochs + 1):
        # One "epoch" = local_epochs passes over the shortest client loader.
        # Each pass does min(loader_len) FedSGD steps.
        for _ in range(local_epochs):
            _fedsgd_train_epoch(shared_model, shared_optim, tasks, device)

        # Validation — plain per-client eval using the shared model.
        val_losses, val_f1s, val_pr_aucs = [], [], []
        for cid, task in enumerate(tasks):
            val_loss, _, val_f1, val_pr_auc = evaluate_epoch(
                shared_model, val_loaders[cid], task.criterion, device, task.use_port_ids
            )
            val_losses.append(val_loss)
            val_f1s.append(val_f1)
            val_pr_aucs.append(val_pr_auc)

        avg_val_loss     = sum(val_losses) / len(val_losses)
        avg_val_f1       = torch.stack(val_f1s).mean(dim=0)
        avg_val_pr_auc   = torch.stack(val_pr_aucs).mean(dim=0)
        val_macro_f1     = avg_val_f1.mean().item()
        val_macro_pr_auc = avg_val_pr_auc.mean().item()

        # Diagnostic train-loss on client 0's train loader.
        train_loss, _, _, _ = evaluate_epoch(
            shared_model, tasks[0].train_loader, tasks[0].criterion, device, tasks[0].use_port_ids
        )

        append_epoch_csv(
            epoch_csv_path, epoch, train_loss, avg_val_loss, avg_val_f1, avg_val_pr_auc
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

    # ── Test evaluation ───────────────────────────────────────────────────────
    shared_model.load_state_dict(torch.load(best_ckpt_path, map_location=device))

    results = []
    for cid, task in enumerate(tasks):
        test_loss, _, test_f1, test_pr_auc = evaluate_epoch(
            shared_model, test_loaders[cid], task.criterion, device, task.use_port_ids
        )
        print(
            f"[Seed {seed}][Client {cid}] Best ckpt → "
            f"test_loss={test_loss:.4f} | "
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

    # Global port vocab precomputation — all clients must agree on vocab sizes
    # so the shared model's port embeddings are correctly sized.
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
        print(f"[Seed {seed}] Starting FedSGD training ({num_clients} clients)...")
        print(f"{'='*60}")
        client_results = run_fedsgd_experiment(
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
        f"[Results] FedSGD — macro minority F1 "
        f"(mean over {len(seeds)} seeds × {num_clients} clients): {macro_mean:.2f}%"
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
    out_csv     = f"./results/metrics/federated_logs/fedsgd{label_suffix}_results.csv"
    out_csv_auc = f"./results/metrics/federated_logs/fedsgd{label_suffix}_pr_auc_results.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    model_name_str = (
        f"FedSGD PNA (gradient averaging, shared Adam) | "
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
