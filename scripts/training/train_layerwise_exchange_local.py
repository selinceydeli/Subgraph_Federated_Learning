#!/usr/bin/env python3
"""
Layer-wise Embedding Exchange with Fully Local Training (no server).

Each client trains its own model independently (no FedAvg parameter
aggregation), but clients still exchange owned-node embeddings after
every conv layer via a shared EmbeddingTable.  This isolates the value
of the embedding exchange mechanism from parameter averaging.

Because client models diverge, remote embeddings injected into client A
come from client B's independently-trained model.  This is the key
semantic difference from ``train_layerwise_exchange.py``, where all
clients share identical weights after FedAvg.

Each epoch:
  1. Synchronous per-step exchange training: for each mini-batch step,
     all clients simultaneously compute one conv layer, update the
     shared EmbeddingTable with their owned-node embeddings, inject the
     freshly computed remote-node embeddings from the table, and only
     then proceed to the next layer.
  2. Each client is validated and checkpointed independently.

Usage:
    python3 -m scripts.training.train_layerwise_exchange_local
"""

import os
import csv
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

CHECK_GRADIENTS = False


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
# Synchronous exchange helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_edge_attr(batch, task, is_hetero):
    """Extract per-relation edge_attr dict from batch for port-ID-enabled models."""
    if not (is_hetero and task.use_port_ids):
        return None
    d = {}
    for rel in [('n', 'fwd', 'n'), ('n', 'rev', 'n')]:
        if 'edge_attr' in batch[rel]:
            ea = batch[rel].edge_attr
            d[rel] = ea if ea.dtype == torch.long else ea.long()
    return d or None


def _check_partition_integrity(train_list, num_nodes):
    """
    Verify the federated node partition before training begins.

    Checks:
      1. Non-overlap: no global node ID is owned by more than one client.
      2. Completeness: the union of all owned sets covers all training nodes.
      3. Usefulness: each client's remote nodes overlap with other clients'
         owned sets — confirms that table writes are meaningful.

    Raises AssertionError with details on any violation.
    Prints a concise PASS summary on success.
    """
    num_clients = len(train_list)
    owned_sets = []

    for cid, data in enumerate(train_list):
        owned_mask = data.owned_mask
        global_nid = data.global_nid
        owned_sets.append(set(global_nid[owned_mask].tolist()))

    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            overlap = owned_sets[i] & owned_sets[j]
            assert len(overlap) == 0, (
                f"[PartitionCheck] FAIL: clients {i} and {j} both own "
                f"{len(overlap)} node(s) (e.g. {list(overlap)[:5]})"
            )

    all_owned = set()
    for s in owned_sets:
        all_owned |= s
    all_train_gids = set()
    for data in train_list:
        all_train_gids |= set(data.global_nid.tolist())
        missing = all_train_gids - all_owned
    assert len(missing) == 0, (
        f"[PartitionCheck] FAIL: {len(missing)} node(s) appear in training "
        f"subgraphs but are not owned by any client (e.g. {list(missing)[:5]})"
    )

    other_owned_union = [
        set().union(*(owned_sets[j] for j in range(num_clients) if j != cid))
        for cid in range(num_clients)
    ]
    for cid, data in enumerate(train_list):
        global_nid = data.global_nid
        owned_mask = data.owned_mask
        remote_gids = set(global_nid[~owned_mask].tolist())
        useful = remote_gids & other_owned_union[cid]
        assert len(useful) > 0 or len(remote_gids) == 0, (
            f"[PartitionCheck] FAIL: client {cid} has {len(remote_gids)} remote "
            f"node(s) but none are owned by other clients — table writes would be useless"
        )

    total_owned = sum(len(s) for s in owned_sets)
    print(
        f"[PartitionCheck] PASS — {num_clients} clients, "
        f"{total_owned} total owned nodes, no overlaps, union complete."
    )


def _synchronous_train_epoch(tasks, table, device):
    """
    One pass of synchronous per-step layer-wise embedding exchange training
    with independently-trained client models (no FedAvg).

    Identical to the FedAvg variant except that client models have divergent
    weights, so remote embeddings are produced by a different model than the
    one consuming them.

    For each mini-batch step:
      1. All clients fetch one batch and project node features to the first
         hidden layer using their own (potentially divergent) model weights.
      2. For each conv layer l:
           a. All clients compute layer l (HeteroConv + BN + ReLU + Dropout).
           b. The shared EmbeddingTable is updated with each client's
              owned-node embeddings (detached — no gradient through injection).
           c. All clients replace remote-node embeddings with the freshly
              written table values (gradient-safe via torch.where).
      3. All clients compute the output logits and backpropagate independently.

    Returns:
        (avg_loss, exchange_stats) where exchange_stats is a dict:
            'remote_total':  list[int] — total remote node-instances per layer.
            'remote_served': list[int] — subset actually served from table.
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
        # ── Fetch one batch per client ────────────────────────────────────────
        batches = [next(it).to(device) for it in iters]

        # ── Input projection for all clients ──────────────────────────────────
        client_states = []
        for task, batch in zip(tasks, batches):
            task.model.train()
            x_in, edge_in, y_true, n_nodes, is_hetero = _unpack_io(batch)
            x_in_aug, y_used, B = _augment_with_ego_and_get_seed_slice(
                x_in, y_true, batch, is_hetero, task.model
            )
            edge_attr_dict = _build_edge_attr(batch, task, is_hetero)

            x, pna_ea, ei_dict = task.model.project_input(
                x_in_aug, edge_in, edge_attr_dict=edge_attr_dict
            )

            owned_mask = (
                batch['n'].owned_mask if is_hetero else getattr(batch, 'owned_mask', None)
            )
            global_nids = (
                batch['n'].global_nid if is_hetero else getattr(batch, 'global_nid', None)
            )

            client_states.append({
                'x': x,
                'pna_ea': pna_ea,
                'ei': ei_dict,
                'owned': owned_mask,
                'gids': global_nids,
                'y': y_used,
                'B': B,
                'N': n_nodes,
                'task': task,
            })

        # ── Layer-by-layer synchronous exchange ───────────────────────────────
        table.reset()

        if step_idx == 0:
            assert not table.written[0].any(), (
                "[FreshnessCheck] FAIL: table.written[0] is not all-False after "
                "reset() — reset() is broken"
            )

        do_grad_check = CHECK_GRADIENTS and step_idx == 0
        grad_check_refs = {}

        for l in range(tasks[0].num_layers):
            if do_grad_check and l == 0:
                for cid, s in enumerate(client_states):
                    s['x'].retain_grad()
                    grad_check_refs[cid] = s['x']

            # Step 2a: all clients compute conv layer l
            for s in client_states:
                s['x'] = s['task'].model.compute_conv_layer(
                    l, s['x'], s['ei'], s['pna_ea']
                )

            # Step 2b: update table with owned-node embeddings (detached)
            for s in client_states:
                if s['owned'] is not None and s['gids'] is not None:
                    owned_mask = s['owned']
                    table.update(
                        layer=l,
                        global_nids=s['gids'][owned_mask].cpu(),
                        embeddings=s['x'][owned_mask].detach().cpu(),
                    )

            # Step 2c: inject remote-node embeddings from the fresh table
            for s in client_states:
                if s['owned'] is None or s['gids'] is None:
                    continue
                remote_mask = ~s['owned']

                if remote_mask.any():
                    remote_total[l] += int(remote_mask.sum().item())
                    written_cpu = table.written_mask(l, s['gids'][remote_mask])
                    remote_served[l] += int(written_cpu.sum().item())

                if remote_mask.any():
                    rem_embs = table.fetch(l, s['gids'][remote_mask]).to(device)
                    written = table.written_mask(l, s['gids'][remote_mask]).to(device)
                    if written.any():
                        remote_indices = remote_mask.nonzero(as_tuple=True)[0]
                        written_indices = remote_indices[written]
                        written_full = torch.zeros(
                            s['x'].size(0), dtype=torch.bool, device=device
                        )
                        written_full[written_indices] = True
                        x_fill = torch.zeros_like(s['x'])
                        x_fill[written_indices] = rem_embs[written]
                        s['x'] = torch.where(
                            (s['owned'] | ~written_full).unsqueeze(1).expand_as(s['x']),
                            s['x'],
                            x_fill,
                        )

        # ── Output, loss, backprop ────────────────────────────────────────────
        for cid, s in enumerate(client_states):
            task = s['task']
            logits = task.model.compute_output(s['x'])

            out_used = logits[:s['B']] if s['B'] is not None else logits
            y_batch = s['y'][:s['B']] if s['B'] is not None else s['y']

            if s['owned'] is not None and (s['B'] is None or s['B'] == s['N']):
                out_used = out_used[s['owned']]
                y_batch = y_batch[s['owned']]
                count = int(s['owned'].sum().item())
            else:
                count = s['B'] if s['B'] is not None else s['N']

            task.optimizer.zero_grad()
            loss = task.criterion(out_used, y_batch.float())
            loss.backward()
            task.optimizer.step()

            total_loss += loss.item() * count
            total_count += count

            if do_grad_check and cid in grad_check_refs:
                x0 = grad_check_refs[cid]
                if x0.grad is not None and s['owned'] is not None:
                    remote_mask = ~s['owned'][:x0.size(0)]
                    remote_gids = s['gids'][:x0.size(0)][remote_mask]
                    written_at_any = table.written[
                        :, remote_gids.cpu().long()
                    ].any(dim=0)
                    if written_at_any.any():
                        injected_idx = remote_mask.nonzero(as_tuple=True)[0][written_at_any]
                        leaked = x0.grad[injected_idx].abs().max().item()
                        assert leaked < 1e-6, (
                            f"[GradCheck] FAIL client {cid}: gradient leaked into "
                            f"injected remote positions (max abs = {leaked:.6e}). "
                            f"torch.where is not detaching remote embeddings correctly."
                        )
                        print(
                            f"[GradCheck] PASS client {cid}: gradient at injected "
                            f"remote positions is zero (max abs = {leaked:.6e})"
                        )

        do_grad_check = False
        step_idx += 1

    exchange_stats = {
        'remote_total': remote_total,
        'remote_served': remote_served,
    }
    return total_loss / max(total_count, 1), exchange_stats


# ──────────────────────────────────────────────────────────────────────────────
# Main experiment loop (one seed)
# ──────────────────────────────────────────────────────────────────────────────

def run_exchange_experiment(train_list, val_list, test_list, args, device, seed, run_id):
    """
    Run the layer-wise exchange protocol with fully local training for one seed.
    Each client has independently initialised and trained model weights — no
    FedAvg aggregation is performed at any point.
    Returns a list of per-client dicts with test_f1_per_task and test_pr_auc_per_task.
    """
    num_clients = len(train_list)
    num_clients_cfg = getattr(args, "num_clients", num_clients)
    label_suffix = "_local_labels" if getattr(args, "use_local_labels", False) else ""

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
            model_name=f"layerwise_exchange_local_client_{cid}",
            seed=seed,
            tasks=TASKS,
            out_dir=(
                f"./results/metrics/federated_logs/"
                f"layerwise_exchange_local{label_suffix}/{num_clients_cfg}_clients/client_{cid}"
            ),
        )
        for cid in range(num_clients)
    ]

    # ── Embedding table (covers all nodes in the global training graph) ───────
    all_gids = torch.cat([train_list[cid].global_nid for cid in range(num_clients)])
    num_nodes = int(all_gids.max().item()) + 1
    table = EmbeddingTable(
        num_nodes=num_nodes,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
    )

    # ── Partition integrity check (once per seed, before training) ────────────
    _check_partition_integrity(train_list, num_nodes)

    # ── Coverage CSV setup ────────────────────────────────────────────────────
    coverage_csv_dir = (
        f"./results/metrics/federated_logs/"
        f"layerwise_exchange_local{label_suffix}/{num_clients_cfg}_clients"
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
    ckpt_dir = f"./checkpoints/layerwise_exchange_local{label_suffix}/{num_clients_cfg}_clients"
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
            _, step_stats = _synchronous_train_epoch(tasks, table, device)
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

        # ── Per-client validation ─────────────────────────────────────────────
        # Each client's own model is evaluated on its own val partition.
        val_losses, val_f1s, val_pr_aucs = [], [], []
        for cid, task in enumerate(tasks):
            val_loss, _, val_f1, val_pr_auc = evaluate_epoch(
                task.model, val_loaders[cid], task.criterion, device, task.use_port_ids
            )
            val_losses.append(val_loss)
            val_f1s.append(val_f1)
            val_pr_aucs.append(val_pr_auc)

            # Per-client best-model checkpointing
            client_pr_auc = val_pr_auc.mean().item()
            if client_pr_auc > best_val_pr_aucs[cid]:
                best_val_pr_aucs[cid] = client_pr_auc
                torch.save(task.model.state_dict(), best_ckpt_paths[cid])

        avg_val_loss     = sum(val_losses) / len(val_losses)
        avg_val_f1       = torch.stack(val_f1s).mean(dim=0)
        avg_val_pr_auc   = torch.stack(val_pr_aucs).mean(dim=0)
        val_macro_f1     = avg_val_f1.mean().item()
        val_macro_pr_auc = avg_val_pr_auc.mean().item()

        # Train-loss estimate on client 0's training data
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
    # Load each client's own best checkpoint and evaluate independently.
    results = []
    for cid, task in enumerate(tasks):
        task.model.load_state_dict(
            torch.load(best_ckpt_paths[cid], map_location=device)
        )
        test_loss, _, test_f1, test_pr_auc = evaluate_epoch(
            task.model, test_loaders[cid], task.criterion, device, task.use_port_ids
        )
        test_macro        = test_f1.mean().item()
        test_macro_pr_auc = test_pr_auc.mean().item()
        print(
            f"[Seed {seed}][Client {cid}] Best ckpt -> "
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

    # seeds = [args.base_seed, args.base_seed + 1, args.base_seed + 2]
    seeds = [args.base_seed, args.base_seed + 1]
    per_seed_mean_f1 = []
    per_seed_mean_pr_auc = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"[Seed {seed}] Starting layer-wise exchange (local) training ({num_clients} clients)...")
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
        f"[Results] Layer-wise exchange (local, no FedAvg) — "
        f"macro minority F1 (mean over {len(seeds)} seeds x {num_clients} clients): "
        f"{macro_mean:.2f}%"
    )
    row = " | ".join(
        f"{n}: {100*m:.2f}+/-{100*s:.2f}%"
        for n, m, s in zip(TASKS, mean_f1.tolist(), std_f1.tolist())
    )
    print(f"[Results] Per-task F1 (mean+/-std): {row}")
    print(f"[Results] macro PR-AUC: {macro_pr_auc:.2f}%")
    row_pr = " | ".join(
        f"{n}: {100*m:.2f}+/-{100*s:.2f}%"
        for n, m, s in zip(TASKS, mean_pr_auc.tolist(), std_pr_auc.tolist())
    )
    print(f"[Results] Per-task PR-AUC (mean+/-std): {row_pr}")

    runtime_sec = time.perf_counter() - start_ts

    label_suffix = "_local_labels" if args.use_local_labels else ""
    out_csv     = f"./results/metrics/federated_logs/layerwise_exchange_local{label_suffix}_results.csv"
    out_csv_auc = f"./results/metrics/federated_logs/layerwise_exchange_local{label_suffix}_pr_auc_results.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    model_name_str = (
        f"Layer-wise exchange (local, no FedAvg) | "
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

    print(f"\n[Done] Runtime: {runtime_sec:.1f}s | F1 -> {out_csv} | PR-AUC -> {out_csv_auc}")


if __name__ == "__main__":
    main()
