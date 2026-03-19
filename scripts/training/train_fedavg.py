#!/usr/bin/env python3
"""
FedAvg federated training: clients train locally and periodically sync
with a server that aggregates weights via weighted averaging (FedAvg).

This script mirrors train_local_baseline.py exactly in config loading,
seed handling, data loading, evaluation, and CSV reporting; it replaces
the isolated per-client loop with a FedAvg communication loop.

Usage:
    python3 -m scripts.training.train_fedavg
"""

import os
import random
import time
import json
from types import SimpleNamespace
from datetime import datetime

import torch

from utils.loader import load_client_graphs
from utils.seed import set_seed
from utils.metrics import append_f1_score_to_csv, start_epoch_csv, append_epoch_csv
from utils.train_utils import ensure_node_features, evaluate_epoch
from utils.hetero import make_bidirected_hetero
from utils.graph_helpers import (
    max_port_cols,
    check_and_strip_self_loops,
    build_hetero_neighbor_loader,
    build_full_eval_loader,
)
from fed_algo.fedavg.client import FedAvgClient
from fed_algo.fedavg.server import FedAvgServer

PNA_CONFIG_PATH = "./configs/pna_configs.json"
FED_CONFIG_PATH = "./configs/fed_configs.json"

TASKS = ["deg-in", "deg-out", "fan-in", "fan-out", "C2", "C3", "C4", "C5", "C6", "S-G", "B-C"]


def build_args(pna_cfg, fed_cfg, partition_cfg):
    """Merge config dicts into a single SimpleNamespace for NodeClsTask / FedAvg."""
    hparams = pna_cfg["default_hparams"]
    ns = SimpleNamespace(
        # Required by load_task() in base.py
        task="node_cls",
        # Model architecture
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
        # FL settings
        local_epochs=fed_cfg["local_epochs"],
        global_epochs=fed_cfg["global_epochs"],
        client_fraction=fed_cfg["client_fraction"],
        base_seed=fed_cfg["base_seed"],
        # Partition settings
        num_clients=partition_cfg["num_clients"],
        include_cross_edges=partition_cfg["include_cross_edges"],
    )
    return ns


def resolve_data_dirs(partition_cfg):
    """Return (train_clients_dir, val_clients_dir, test_clients_dir)."""
    num_clients = partition_cfg["num_clients"]
    cross_suffix = "with_cross_edges" if partition_cfg["include_cross_edges"] else "without_cross_edges"
    base = f"./data/fed_partition_aware_splits_{cross_suffix}/{num_clients}_clients"
    return (
        f"{base}/train/clients",
        f"{base}/val/clients",
        f"{base}/test/clients",
    )


def make_eval_loader(client_data, task, device, shuffle=False):
    """
    Preprocess a val/test client subgraph and build a NeighborLoader for evaluation.
    Mirrors the preprocessing done by NodeClsTask on training data.
    Uses the server task's port vocab sizes to clamp unseen port degrees.
    """
    data = check_and_strip_self_loops(client_data, "eval")
    data = ensure_node_features(data)
    hetero = make_bidirected_hetero(data)

    # Clamp port IDs to the global vocab sizes to handle unseen port degrees
    # in val/test that exceed what the model saw during training.
    # fwd edge_attr layout: [in_port, out_port]
    # rev edge_attr layout: [out_port, in_port]  (swapped by make_bidirected_hetero)
    if task.use_port_ids:
        fwd_ea = hetero[("n", "fwd", "n")].edge_attr.clone()
        fwd_ea[:, 0].clamp_(max=task.in_port_vocab_size - 1)
        fwd_ea[:, 1].clamp_(max=task.out_port_vocab_size - 1)
        hetero[("n", "fwd", "n")].edge_attr = fwd_ea

        rev_ea = hetero[("n", "rev", "n")].edge_attr.clone()
        rev_ea[:, 0].clamp_(max=task.out_port_vocab_size - 1)
        rev_ea[:, 1].clamp_(max=task.in_port_vocab_size - 1)
        hetero[("n", "rev", "n")].edge_attr = rev_ea

    # Respect owned_mask if present (ignore ghost nodes)
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_ts = time.perf_counter()

    # Load configs
    with open(PNA_CONFIG_PATH, "r") as f:
        pna_all = json.load(f)
    with open(FED_CONFIG_PATH, "r") as f:
        fed_all = json.load(f)

    pna_cfg = pna_all["reverse_mp_with_port_and_ego"]
    fed_cfg = fed_all["fed_learning_configs"]
    partition_cfg = fed_all["partition_aware_splits"]

    args = build_args(pna_cfg, fed_cfg, partition_cfg)

    print(f"[Config] global_epochs={args.global_epochs}, local_epochs={args.local_epochs}, "
          f"client_fraction={args.client_fraction}, num_clients={args.num_clients}, "
          f"include_cross_edges={args.include_cross_edges}")
    print(f"[Config] use_ego_ids={args.use_ego_ids}, use_port_ids={args.use_port_ids}, "
          f"use_mini_batch={args.use_mini_batch}, batch_size={args.batch_size}")

    # Resolve data directories
    train_dir, val_dir, test_dir = resolve_data_dirs(partition_cfg)
    print(f"[Data] train_dir={train_dir}")
    print(f"[Data] val_dir={val_dir}")
    print(f"[Data] test_dir={test_dir}")

    # Load all client data
    num_clients = args.num_clients
    train_list = load_client_graphs(train_dir, num_clients)
    val_list   = load_client_graphs(val_dir,   num_clients)
    test_list  = load_client_graphs(test_dir,  num_clients)

    print(f"[Data] Loaded {num_clients} clients' train/val/test subgraphs.")

    # --- Global port vocab precomputation ---
    # All clients and server must share the same port embedding dimensions so that
    # FedAvg weight aggregation does not crash on shape mismatches.
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

    # data_dir placeholder; NodeClsTask only uses it for reference
    data_dir = "./data"

    seeds = [args.base_seed, args.base_seed + 1, args.base_seed + 2]
    per_seed_mean_f1     = []
    per_seed_mean_pr_auc = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"[Seed {seed}] Starting FedAvg training ({num_clients} clients)...")
        print(f"{'='*60}")

        set_seed(seed)
        message_pool = {}

        # Server initialises the global model using client 0's training data
        server = FedAvgServer(args, train_list[0], data_dir, message_pool, device)

        # Each client gets its own local model (same architecture, shared vocab)
        clients = [
            FedAvgClient(args, cid, train_list[cid], data_dir, message_pool, device)
            for cid in range(num_clients)
        ]

        # Build val/test loaders using the server task's port vocab sizes
        val_loaders  = [make_eval_loader(val_list[cid],  server.task, device) for cid in range(num_clients)]
        test_loaders = [make_eval_loader(test_list[cid], server.task, device) for cid in range(num_clients)]

        model_name = f"fedavg_seed{seed}"
        epoch_csv_path = start_epoch_csv(
            model_name=model_name,
            seed=seed,
            tasks=TASKS,
            out_dir="./results/metrics/federated_logs/fedavg",
        )

        ckpt_dir = "./checkpoints/fedavg"
        os.makedirs(ckpt_dir, exist_ok=True)
        best_ckpt_path = os.path.join(ckpt_dir, f"seed{seed}_{run_id}_best.pt")

        # Broadcast initial global model to all clients
        server.send_message()

        best_val_pr_auc = -1.0

        for epoch in range(1, args.global_epochs + 1):
            # Sample a fraction of clients
            num_sampled = max(1, round(args.client_fraction * num_clients))
            sampled = random.sample(range(num_clients), num_sampled)
            message_pool["sampled_clients"] = sampled

            # Client local updates
            for cid in sampled:
                clients[cid].execute()       # sync with server + train locally
                clients[cid].send_message()  # upload weights to message pool

            # Server aggregation + broadcast updated global model
            server.execute()
            server.send_message()

            # Evaluate the global model on each client's val partition
            val_losses, val_f1s, val_pr_aucs = [], [], []
            for cid in range(num_clients):
                val_loss, _, val_f1, val_pr_auc = evaluate_epoch(
                    server.task.model,
                    val_loaders[cid],
                    server.task.criterion,
                    device,
                    server.task.use_port_ids,
                )
                val_losses.append(val_loss)
                val_f1s.append(val_f1)
                val_pr_aucs.append(val_pr_auc)

            avg_val_loss   = sum(val_losses) / len(val_losses)
            avg_val_f1     = torch.stack(val_f1s).mean(dim=0)
            avg_val_pr_auc = torch.stack(val_pr_aucs).mean(dim=0)

            # Quick train-loss estimate: evaluate server model on client 0's train loader
            train_loss, _, _, _ = evaluate_epoch(
                server.task.model,
                server.task.train_loader,
                server.task.criterion,
                device,
                server.task.use_port_ids,
            )

            append_epoch_csv(epoch_csv_path, epoch, train_loss, avg_val_loss, avg_val_f1, avg_val_pr_auc)

            val_macro_f1     = avg_val_f1.mean().item()
            val_macro_pr_auc = avg_val_pr_auc.mean().item()
            print(
                f"[Seed {seed}] Epoch {epoch:03d} | "
                f"train {train_loss:.4f} | val {avg_val_loss:.4f} | "
                f"val macro-minF1 {100 * val_macro_f1:.2f}% | "
                f"val macro-PR-AUC {100 * val_macro_pr_auc:.2f}% | "
                f"sampled={sampled}"
            )

            if val_macro_pr_auc > best_val_pr_auc:
                best_val_pr_auc = val_macro_pr_auc
                torch.save(server.task.model.state_dict(), best_ckpt_path)

        # Load best checkpoint and evaluate on all clients' test partitions
        server.task.model.load_state_dict(
            torch.load(best_ckpt_path, map_location=device)
        )

        test_f1s, test_pr_aucs = [], []
        for cid in range(num_clients):
            test_loss, _, test_f1, test_pr_auc = evaluate_epoch(
                server.task.model,
                test_loaders[cid],
                server.task.criterion,
                device,
                server.task.use_port_ids,
            )
            test_macro = test_f1.mean().item()
            print(
                f"[Seed {seed}] Client {cid} best ckpt → "
                f"test_loss={test_loss:.4f} | test macro-minF1={100 * test_macro:.2f}%"
            )
            test_f1s.append(test_f1.cpu())
            test_pr_aucs.append(test_pr_auc.cpu())

        seed_test_f1     = torch.stack(test_f1s).mean(dim=0)
        seed_test_pr_auc = torch.stack(test_pr_aucs).mean(dim=0)
        per_seed_mean_f1.append(seed_test_f1)
        per_seed_mean_pr_auc.append(seed_test_pr_auc)

    # --- Aggregate across seeds ---
    all_seeds_f1 = torch.stack(per_seed_mean_f1, dim=0)  # [3, 11]
    mean_f1 = all_seeds_f1.mean(dim=0)
    std_f1  = all_seeds_f1.std(dim=0, unbiased=False)
    macro_mean = mean_f1.mean().item() * 100

    all_seeds_pr_auc = torch.stack(per_seed_mean_pr_auc, dim=0)  # [3, 11]
    mean_pr_auc = all_seeds_pr_auc.mean(dim=0)
    std_pr_auc  = all_seeds_pr_auc.std(dim=0, unbiased=False)
    macro_pr_auc = mean_pr_auc.mean().item() * 100

    print(f"\n{'='*60}")
    print(
        f"[Results] FedAvg — macro minority F1 (mean across {len(seeds)} seeds × {num_clients} clients): "
        f"{macro_mean:.2f}%"
    )
    row = " | ".join(
        f"{n}: {100*m:.2f}±{100*s:.2f}%"
        for n, m, s in zip(TASKS, mean_f1.tolist(), std_f1.tolist())
    )
    print(f"[Results] Per-task (mean±std across seeds): {row}")
    print(f"[Results] macro PR-AUC: {macro_pr_auc:.2f}%")
    row_pr = " | ".join(
        f"{n}: {100*m:.2f}±{100*s:.2f}%"
        for n, m, s in zip(TASKS, mean_pr_auc.tolist(), std_pr_auc.tolist())
    )
    print(f"[Results] Per-task PR-AUC (mean±std): {row_pr}")

    runtime_sec = time.perf_counter() - start_ts

    out_csv = "./results/metrics/federated_logs/fedavg_results.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    append_f1_score_to_csv(
        out_csv=out_csv,
        tasks=TASKS,
        mean_f1=mean_f1,
        std_f1=std_f1,
        macro_mean_percent=macro_mean,
        seeds=seeds,
        mean_pr_auc=mean_pr_auc,
        std_pr_auc=std_pr_auc,
        model_name=(
            f"FedAvg PNA | "
            f"num_clients={num_clients}, "
            f"cross_edges={args.include_cross_edges}, "
            f"global_epochs={args.global_epochs}, "
            f"local_epochs={args.local_epochs}, "
            f"client_fraction={args.client_fraction}, "
            f"use_port_ids={args.use_port_ids}, "
            f"use_ego_ids={args.use_ego_ids}, "
            f"num_layers={args.num_layers}, "
            f"neighbors_per_hop={args.neighbors_per_hop}, "
            f"seeds={seeds}, "
            f"run_id={run_id}"
        ),
        runtime_seconds=runtime_sec,
    )

    print(f"\n[Done] Runtime: {runtime_sec:.1f}s | Results appended to {out_csv}")


if __name__ == "__main__":
    main()
