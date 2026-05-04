#!/usr/bin/env python3
"""
Layer-wise Embedding Exchange with Sync-SGD + persistent embedding cache.

Combines four sharing mechanisms:
  - Per-step gradient averaging into a single shared Adam state (Sync-SGD).
  - Layer-wise embedding exchange (the per-step write→read protocol).
  - Persistent embedding cache: the EmbeddingTable is NEVER reset within
    a seed's training lifetime, so ghost lookups can be served by
    embeddings written earlier in the current epoch OR in any prior epoch.
    Owned writes overwrite per-gid (EmbeddingTable.update is assignment),
    so an entry is naturally refreshed when its owner revisits it.
  - OptimES-style cold-start pre-training round (Naman & Simmhan 2025,
    §3.2.1): one local-only forward pass per client under the synchronized
    random-init model, populating `h^1..h^L` for every owned gid before
    epoch 1. Eliminates the epoch-1 cold-start coverage shortfall.

Coverage trajectory:
  • Pre-training (round 0): every training gid gets one cache entry per
    layer, computed under the random-init model with no ghost injection
    (purely local subgraph view, matching OptimES §3.2.1).
  • Epoch 1 step 1 onwards: ≈100% hit-rate from the start (every ghost
    lookup hits the pretraining-seeded entry, immediately overwritten as
    its owner revisits it under the current Sync-SGD model).
  • Bandwidth (bytes WRITTEN) per training epoch converges to "one write
    per owned gid per epoch", because served-from-cache consumptions
    cost nothing.

Trade-off vs. the per-step-reset baseline (`train_layerwise_exchange.py`'s
`_exchange_one_step` + this script's prior import of it):
  • Higher coverage from the start, near-zero cold-start cost from epoch 2.
  • Injected ghosts may reflect a model from many steps ago (Sync-SGD steps
    the optimizer every batch), so embeddings carry per-step staleness —
    same staleness assumption OptimES makes at the per-epoch granularity.

Why gradient averaging with one shared Adam (not parameter averaging)?
  `NodeClsTask` uses Adam. Averaging N clients' post-step parameters is not
  equivalent to averaging N clients' gradients with one shared Adam, because
  N local Adam states maintain their own first/second moments. Averaging
  gradients into one shared optimizer keeps the "single global optimizer"
  semantics classical Sync-SGD assumes.

Train/eval mechanism parity: both training and evaluation route through
`_exchange_one_step_persistent` (defined locally below) so the exchange
mechanic is byte-for-byte identical. Eval uses its OWN fresh EmbeddingTable
constructed per pass — train.pt / val.pt / test.pt are independent graphs
(distinct global_nid spaces), so the training table's entries would inject
embeddings computed for *different nodes* if read during eval.

Usage:
    python3 -m scripts.training.train_layerwise_exchange_sync_sgd
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
from utils.layerwise_exchange import EmbeddingTable
from task.node_cls import NodeClsTask

# `_prepare_client_state` and `_check_partition_integrity` are reused
# verbatim. `_exchange_one_step` and `_synchronous_eval_epoch` are NOT
# imported — the persistent-cache variants below replace them in this
# script (the imports' per-step `table.reset()` is incompatible with
# cross-step / cross-epoch reuse).
from scripts.training.train_layerwise_exchange import (
    _prepare_client_state,
    _check_partition_integrity,
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
# Persistent-cache exchange primitive (replaces _exchange_one_step locally)
# ──────────────────────────────────────────────────────────────────────────────

def _exchange_one_step_persistent(
    client_states, table, num_layers, device,
    *, track_coverage=False, write_tracker=None,
):
    """
    Per-layer write→read exchange identical to `_exchange_one_step` in
    train_layerwise_exchange.py, EXCEPT this never calls `table.reset()`.
    The caller controls table lifetime; overwrite-on-write semantics
    (EmbeddingTable.update is assignment, not merge) keep entries fresh
    as their owners revisit them.

    A ghost lookup is "served" iff written_mask is True at that gid/layer,
    regardless of when the writing happened (this step / earlier this
    epoch / a prior epoch).

    If `write_tracker` is provided ([num_layers, num_nodes] CPU bool),
    it is OR'd in-place with each layer's owned-gid writes so the caller
    can compute unique writes per layer per epoch via
    `write_tracker[l].sum()`.

    Returns (remote_total, remote_served, layer_writes):
      • remote_total / remote_served are per-layer lists if
        track_coverage=True, else None.
      • layer_writes is always a per-layer list of update-call sizes
        (#gids written) summed across clients this step.
    """
    remote_total  = [0] * num_layers if track_coverage else None
    remote_served = [0] * num_layers if track_coverage else None
    layer_writes  = [0] * num_layers

    for l in range(num_layers):
        for s in client_states:
            s['x'] = s['task'].model.compute_conv_layer(
                l, s['x'], s['ei'], s['pna_ea']
            )
        for s in client_states:
            if s['owned'] is not None and s['gids'] is not None:
                m = s['owned']
                gids_to_write = s['gids'][m].cpu()
                table.update(
                    layer=l,
                    global_nids=gids_to_write,
                    embeddings=s['x'][m].detach().cpu(),
                )
                layer_writes[l] += int(gids_to_write.numel())
                if write_tracker is not None:
                    write_tracker[l, gids_to_write.long()] = True
        for s in client_states:
            if s['owned'] is None or s['gids'] is None:
                continue
            remote_mask = ~s['owned']
            if not remote_mask.any():
                continue

            if track_coverage:
                remote_total[l] += int(remote_mask.sum().item())
                written_cpu = table.written_mask(l, s['gids'][remote_mask])
                remote_served[l] += int(written_cpu.sum().item())

            rem_embs = table.fetch(l, s['gids'][remote_mask]).to(device)
            written  = table.written_mask(l, s['gids'][remote_mask]).to(device)
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

    return remote_total, remote_served, layer_writes


# ──────────────────────────────────────────────────────────────────────────────
# Persistent-cache eval primitive (replaces _synchronous_eval_epoch locally)
# ──────────────────────────────────────────────────────────────────────────────

def _synchronous_eval_epoch_persistent(
    tasks, loaders, num_nodes, num_layers, hidden_dim, device,
):
    """
    Eval pass using `_exchange_one_step_persistent`. Constructs its OWN
    fresh EmbeddingTable for the duration of the pass — separate from the
    training table — for two reasons:

      1. train.pt / val.pt / test.pt are independent graphs (separate
         global_nid spaces), so a training-time write at gid `g` is
         computed for a different node than the val/test graph's gid `g`.
         Reading the training table at eval time would inject embeddings
         for the wrong node identities.
      2. Eval runs in `model.eval()` + `no_grad`; writing those (frozen-BN,
         no-grad) outputs into the training table would pollute training
         injections in the next epoch.

    Within a single eval pass the eval table is persistent across all
    eval steps, so coverage builds up the same way it does within a
    training epoch (cold start at step 1, warming up across the pass).

    Returns three parallel lists of length num_clients:
        (val_losses, val_f1_per_task, val_pr_auc_per_task)
    """
    eval_table = EmbeddingTable(
        num_nodes=num_nodes, num_layers=num_layers, hidden_dim=hidden_dim,
    )

    for t in tasks:
        t.model.eval()

    num_clients = len(tasks)

    per_client_logits   = [[] for _ in range(num_clients)]
    per_client_labels   = [[] for _ in range(num_clients)]
    per_client_loss_sum = [0.0] * num_clients
    per_client_count    = [0] * num_clients

    iters = [iter(ldr) for ldr in loaders]

    with torch.no_grad():
        for step_batches in zip_longest(*iters, fillvalue=None):
            active_cids = []
            client_states = []
            for cid, (task, batch) in enumerate(zip(tasks, step_batches)):
                if batch is None:
                    continue
                active_cids.append(cid)
                client_states.append(_prepare_client_state(task, batch, device))

            if not client_states:
                continue

            _exchange_one_step_persistent(
                client_states, eval_table, num_layers, device,
                track_coverage=False,
            )

            for cid, s in zip(active_cids, client_states):
                task = s['task']
                logits = task.model.compute_output(s['x'])
                out_used = logits[:s['B']] if s['B'] is not None else logits
                y_batch  = s['y'][:s['B']]  if s['B'] is not None else s['y']

                if s['owned'] is not None and (s['B'] is None or s['B'] == s['N']):
                    out_used = out_used[s['owned']]
                    y_batch  = y_batch[s['owned']]
                    count    = int(s['owned'].sum().item())
                else:
                    count = s['B'] if s['B'] is not None else s['N']

                loss = task.criterion(out_used, y_batch.float())
                per_client_loss_sum[cid] += loss.item() * count
                per_client_count[cid]    += count
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
        f1     = compute_minority_f1_score_per_task(logits, labels)
        pr_auc = compute_pr_auc_per_task(logits, labels)
        losses.append(avg_loss)
        f1s.append(f1)
        pr_aucs.append(pr_auc)

    return losses, f1s, pr_aucs


# ──────────────────────────────────────────────────────────────────────────────
# OptimES-style pre-training round (§3.2.1) — eliminates epoch-1 cold start
# ──────────────────────────────────────────────────────────────────────────────

def _pretraining_round(tasks, table, num_nodes, num_layers, device):
    """
    OptimES-faithful cold-start pre-training round (Naman & Simmhan 2025,
    §3.2.1). One pass over every client's train_loader using the
    synchronized initial model weights. Writes `h^1..h^L` for every owned
    gid into `table`; ghosts are NOT injected — the first conv at boundary
    nodes uses purely the local subgraph view (matching the paper's
    "purely local" cold-start computation).

    For Sync-SGD, all `tasks` already share `tasks[0].model`, so the
    "synchronization at round 0" is automatic — no FedAvg needed.

    Run under `model.eval()` + `torch.no_grad()` so BN running stats and
    gradients aren't polluted by this initialization pass.

    After this returns, epoch-1 step-1 ghost lookups will hit ≈100% (every
    training gid is owned by some client and has just been written), at
    the cost of those entries reflecting the random-init model — but
    overwritten as their owners revisit them during epoch 1.

    Returns (per_layer_writes, per_layer_unique_writes):
      per-layer total update sizes and per-layer count of distinct gids
      touched. Used for the [Pre-training] log line and CSV row.
    """
    write_tracker = torch.zeros(num_layers, num_nodes, dtype=torch.bool)
    layer_writes  = [0] * num_layers

    for t in tasks:
        t.model.eval()

    with torch.no_grad():
        for task in tasks:
            for batch in task.train_loader:
                state = _prepare_client_state(task, batch, device)
                x = state['x']
                for l in range(num_layers):
                    x = task.model.compute_conv_layer(
                        l, x, state['ei'], state['pna_ea']
                    )
                    if state['owned'] is not None and state['gids'] is not None:
                        m = state['owned']
                        gids_to_write = state['gids'][m].cpu()
                        table.update(
                            layer=l,
                            global_nids=gids_to_write,
                            embeddings=x[m].detach().cpu(),
                        )
                        layer_writes[l] += int(gids_to_write.numel())
                        write_tracker[l, gids_to_write.long()] = True

    for t in tasks:
        t.model.train()

    unique_writes = [int(write_tracker[l].sum().item()) for l in range(num_layers)]
    return layer_writes, unique_writes


# ──────────────────────────────────────────────────────────────────────────────
# Sync-SGD synchronous training step
# ──────────────────────────────────────────────────────────────────────────────

def _synchronous_train_epoch_sync_sgd(tasks, table, shared_optim, device, write_tracker):
    """
    One pass of layer-wise exchange + per-step gradient averaging, using
    the persistent EmbeddingTable. Per step:
      1. Each client draws one batch; build client_state via _prepare_client_state.
      2. _exchange_one_step_persistent runs the exchange (no table reset).
         A `write_tracker` accumulates per-epoch unique-gid writes.
      3. Every client computes its own output + loss on owned/seed positions
         using its own criterion. Per-client loss is weighted by the
         client's share of this step's total owned count; backward
         accumulates a sample-weighted average gradient into the shared model.
      4. One shared_optim.step() applies the averaged gradient.

    Returns (avg_loss, exchange_stats) with keys 'remote_total',
    'remote_served', 'epoch_writes' — all per-layer lists.
    """
    iters = [iter(t.train_loader) for t in tasks]
    num_steps = min(len(t.train_loader) for t in tasks)

    total_loss = 0.0
    total_count = 0

    num_layers = tasks[0].num_layers
    remote_total  = [0] * num_layers
    remote_served = [0] * num_layers
    epoch_writes  = [0] * num_layers

    shared_model = tasks[0].model

    for _ in range(num_steps):
        client_states = []
        for task, it in zip(tasks, iters):
            client_states.append(_prepare_client_state(task, next(it), device))

        # Sample-weighting: each client contributes in proportion to the
        # number of owned seeds in its current batch.
        owned_counts = []
        for s in client_states:
            if s['owned'] is not None:
                if s['B'] is not None and s['B'] != s['N']:
                    # Mini-batch: all seeds in [:B] are owned by construction
                    # (input_nodes=owned_idx), but count explicitly to stay honest.
                    owned_counts.append(int(s['owned'][:s['B']].sum().item()))
                else:
                    owned_counts.append(int(s['owned'].sum().item()))
            else:
                owned_counts.append(s['B'] if s['B'] is not None else s['N'])
        total_step_count = max(sum(owned_counts), 1)

        shared_model.train()
        shared_optim.zero_grad()

        # Persistent-cache exchange (no table reset between steps or epochs).
        step_total, step_served, step_writes = _exchange_one_step_persistent(
            client_states, table, num_layers, device,
            track_coverage=True, write_tracker=write_tracker,
        )
        for l in range(num_layers):
            remote_total[l]  += step_total[l]
            remote_served[l] += step_served[l]
            epoch_writes[l]  += step_writes[l]

        # Weighted per-client backward — gradients accumulate into shared_model.
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
            (loss * w).backward()

            total_loss += loss.item() * count
            total_count += count

        shared_optim.step()

    exchange_stats = {
        'remote_total':  remote_total,
        'remote_served': remote_served,
        'epoch_writes':  epoch_writes,
    }
    return total_loss / max(total_count, 1), exchange_stats


# ──────────────────────────────────────────────────────────────────────────────
# Main experiment loop (one seed)
# ──────────────────────────────────────────────────────────────────────────────

def run_exchange_sync_sgd_experiment(train_list, val_list, test_list, args, device, seed, run_id):
    num_clients = len(train_list)
    num_clients_cfg = getattr(args, "num_clients", num_clients)
    label_suffix = "_local_labels" if getattr(args, "use_local_labels", False) else ""
    strategy = getattr(args, "partition_strategy", "partition_aware")
    cross_suffix = "with_cross_edges" if getattr(args, "include_cross_edges", False) else "without_cross_edges"
    run_tag = f"{strategy}_{cross_suffix}"

    # ── One NodeClsTask per client, but all tasks share one model + one Adam ──
    # This is the key Sync-SGD structural choice: every client's backward
    # accumulates gradients into the same parameter tensors, and one
    # optimizer.step() per sync step applies the averaged gradient.
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
            model_name=f"layerwise_exchange_sync_sgd_persistent_client_{cid}",
            seed=seed,
            tasks=TASKS,
            out_dir=(
                f"./results/metrics/federated_logs/"
                f"layerwise_exchange_sync_sgd_persistent{label_suffix}/{run_tag}/{num_clients_cfg}_clients/client_{cid}"
            ),
        )
        for cid in range(num_clients)
    ]

    # Persistent EmbeddingTable: created once per seed, never reset for the
    # entire training lifetime. Owned writes overwrite per-gid (assignment
    # semantics in EmbeddingTable.update), so entries are naturally refreshed
    # when their owners revisit them; ghost lookups can be served by anything
    # already in the table — current step, earlier this epoch, or prior epochs.
    all_gids = torch.cat([train_list[cid].global_nid for cid in range(num_clients)])
    num_nodes = int(all_gids.max().item()) + 1
    table = EmbeddingTable(
        num_nodes=num_nodes,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
    )

    _check_partition_integrity(train_list, num_nodes)

    coverage_csv_dir = (
        f"./results/metrics/federated_logs/"
        f"layerwise_exchange_sync_sgd_persistent{label_suffix}/{run_tag}/{num_clients_cfg}_clients"
    )
    os.makedirs(coverage_csv_dir, exist_ok=True)
    coverage_csv_path = os.path.join(coverage_csv_dir, f"exchange_coverage_seed{seed}.csv")
    # Under persistent caching, `bytes_communicated_mb` counts bytes WRITTEN
    # this epoch (true bandwidth), not bytes consumed. `epoch_writes` is the
    # raw write-count that bytes_communicated_mb is derived from;
    # `unique_writes` is the count of distinct (layer, gid) pairs touched
    # this epoch. Together they distinguish bandwidth from reuse efficacy.
    coverage_csv_header = [
        "epoch", "layer", "remote_total", "remote_served", "coverage_pct",
        "bytes_communicated_mb", "epoch_writes", "unique_writes",
    ]
    if not os.path.exists(coverage_csv_path):
        with open(coverage_csv_path, "w", newline="") as f:
            csv.writer(f).writerow(coverage_csv_header)

    ckpt_dir = f"./checkpoints/layerwise_exchange_sync_sgd_persistent{label_suffix}/{run_tag}/{num_clients_cfg}_clients"
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(ckpt_dir, f"seed{seed}_{run_id}_best.pt")
    best_val_pr_auc = float("-inf")

    local_epochs = getattr(args, 'local_epochs', 1)

    # ── OptimES-style cold-start pre-training round (§3.2.1) ─────────────────
    # Populate the persistent cache once with random-init-model embeddings
    # for every owned gid, so epoch-1 step-1 ghost lookups hit ≈100% from
    # the start. Sync-SGD already shares a single model across clients,
    # so the round-0 synchronization is automatic.
    pretrain_writes, pretrain_unique = _pretraining_round(
        tasks, table, num_nodes, args.num_layers, device,
    )
    pretrain_total_writes = sum(pretrain_writes)
    pretrain_total_unique = sum(pretrain_unique)
    pretrain_mb = pretrain_total_writes * args.hidden_dim * 4 / (1024 ** 2)

    pretrain_layer_parts = [
        f"L{l}: writes={pretrain_writes[l]}, unique={pretrain_unique[l]}"
        for l in range(args.num_layers)
    ]
    print(
        f"[Pre-training] {' | '.join(pretrain_layer_parts)} | "
        f"Bandwidth: {pretrain_mb:.2f} MB | "
        f"Seeded {pretrain_total_unique} unique (layer, gid) entries"
    )

    # Pre-training row in coverage CSV: epoch="pretrain", remote_total/served=0
    # (no injection happens during pre-training), bandwidth = bytes written.
    with open(coverage_csv_path, "a", newline="") as f:
        w = csv.writer(f)
        for l in range(args.num_layers):
            layer_bytes_mb = pretrain_writes[l] * args.hidden_dim * 4 / (1024 ** 2)
            w.writerow([
                "pretrain", l, 0, 0, "0.0000",
                f"{layer_bytes_mb:.4f}", pretrain_writes[l], pretrain_unique[l],
            ])
        w.writerow([
            "pretrain", "total", 0, 0, "0.0000",
            f"{pretrain_mb:.4f}", pretrain_total_writes, pretrain_total_unique,
        ])

    for epoch in range(1, args.global_epochs + 1):
        epoch_remote_total  = [0] * args.num_layers
        epoch_remote_served = [0] * args.num_layers
        epoch_writes        = [0] * args.num_layers
        # Per-epoch unique-writes tracker (CPU bool). OR'd in-place by
        # _exchange_one_step_persistent on every owned write.
        write_tracker = torch.zeros(args.num_layers, num_nodes, dtype=torch.bool)
        for _ in range(local_epochs):
            _, step_stats = _synchronous_train_epoch_sync_sgd(
                tasks, table, shared_optim, device, write_tracker
            )
            for l in range(args.num_layers):
                epoch_remote_total[l]  += step_stats['remote_total'][l]
                epoch_remote_served[l] += step_stats['remote_served'][l]
                epoch_writes[l]        += step_stats['epoch_writes'][l]

        unique_writes = [int(write_tracker[l].sum().item()) for l in range(args.num_layers)]

        total_served = sum(epoch_remote_served)
        total_remote = sum(epoch_remote_total)
        total_writes = sum(epoch_writes)
        total_unique = sum(unique_writes)
        bytes_written = total_writes * args.hidden_dim * 4  # bandwidth = writes
        mb = bytes_written / (1024 ** 2)

        layer_parts = [
            f"Layer {l}: {100.0 * epoch_remote_served[l] / max(epoch_remote_total[l], 1):.1f}%"
            f" ({epoch_remote_served[l]}/{epoch_remote_total[l]}, "
            f"writes={epoch_writes[l]}, unique={unique_writes[l]})"
            for l in range(args.num_layers)
        ]
        print(f"[Exchange] {' | '.join(layer_parts)} | Bandwidth: {mb:.1f} MB")

        with open(coverage_csv_path, "a", newline="") as f:
            w = csv.writer(f)
            for l in range(args.num_layers):
                rt = epoch_remote_total[l]
                rs = epoch_remote_served[l]
                ew = epoch_writes[l]
                uw = unique_writes[l]
                pct = 100.0 * rs / rt if rt > 0 else 0.0
                layer_bytes_mb = ew * args.hidden_dim * 4 / (1024 ** 2)
                w.writerow([epoch, l, rt, rs, f"{pct:.4f}", f"{layer_bytes_mb:.4f}", ew, uw])
            total_pct = 100.0 * total_served / total_remote if total_remote > 0 else 0.0
            w.writerow([
                epoch, "total", total_remote, total_served,
                f"{total_pct:.4f}", f"{mb:.4f}", total_writes, total_unique,
            ])

        # Validation via the persistent-cache exchange. Eval uses its own
        # fresh table (val.pt has a separate gid space from train.pt).
        val_losses, val_f1s, val_pr_aucs = _synchronous_eval_epoch_persistent(
            tasks, val_loaders, num_nodes, args.num_layers, args.hidden_dim, device,
        )

        avg_val_loss     = sum(val_losses) / len(val_losses)
        avg_val_f1       = torch.stack(val_f1s).mean(dim=0)
        avg_val_pr_auc   = torch.stack(val_pr_aucs).mean(dim=0)
        val_macro_f1     = avg_val_f1.mean().item()
        val_macro_pr_auc = avg_val_pr_auc.mean().item()

        # Diagnostic train loss on client 0's train loader (plain local eval).
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

    # ── Test evaluation via the persistent-cache exchange (own fresh table) ──
    shared_model.load_state_dict(torch.load(best_ckpt_path, map_location=device))

    test_losses, test_f1s, test_pr_aucs = _synchronous_eval_epoch_persistent(
        tasks, test_loaders, num_nodes, args.num_layers, args.hidden_dim, device,
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
        print(f"[Seed {seed}] Starting layer-wise exchange + Sync-SGD + persistent cache ({num_clients} clients)...")
        print(f"{'='*60}")
        client_results = run_exchange_sync_sgd_experiment(
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
        f"[Results] Layer-wise exchange + Sync-SGD + persistent cache — "
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
    out_csv     = f"./results/metrics/federated_logs/layerwise_exchange_sync_sgd_persistent{label_suffix}_results.csv"
    out_csv_auc = f"./results/metrics/federated_logs/layerwise_exchange_sync_sgd_persistent{label_suffix}_pr_auc_results.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    model_name_str = (
        f"Layer-wise exchange + Sync-SGD + persistent cache (within-epoch reuse + cross-epoch fallback) | "
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
