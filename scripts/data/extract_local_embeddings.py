#!/usr/bin/env python3
"""
One-time embedding extraction for Cross-Client Embedding Bootstrap.

Loads locally-trained client checkpoints (from train_local_baseline.py),
aggregates their non-port parameters via weighted FedAvg into a canonical model,
and extracts node embeddings for all owned nodes across all clients.

Output: data/local_embeddings/{partition}/{num_clients}_clients/seed{seed}/node_embeddings.pt
        dict[int, Tensor] mapping global_nid -> [hidden_dim] float32 embedding

Usage:
    python3 -m scripts.data.extract_local_embeddings --num_clients 3
    python3 -m scripts.data.extract_local_embeddings --num_clients 5 --partition partition_aware_with_cross_edges
"""

import argparse
import glob
import json
import os
from types import SimpleNamespace

import torch

from task.node_cls import NodeClsTask
from utils.graph_helpers import (
    build_full_eval_loader,
    check_and_strip_self_loops,
    max_port_cols,
)
from utils.hetero import make_bidirected_hetero
from utils.loader import load_client_graphs
from utils.seed import set_seed
from utils.train_utils import ensure_node_features

PNA_CONFIG_PATH = "./configs/pna_configs.json"
FED_CONFIG_PATH = "./configs/fed_configs.json"

# Port embedding keys to skip during aggregation (mismatched vocab sizes across clients)
PORT_KEYS = {"in_port_emb.weight", "out_port_emb.weight"}


def build_args(pna_cfg, fed_cfg, partition_cfg):
    hparams = pna_cfg["default_hparams"]
    ns = SimpleNamespace(
        task="node_cls",
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
    )
    return ns


def resolve_data_dirs(num_clients, partition):
    """Return (train_clients_dir, val_clients_dir, test_clients_dir) for the given partition."""
    if partition == "partition_aware_with_cross_edges":
        base = f"./data/fed_partition_aware_splits_with_cross_edges/{num_clients}_clients"
    elif partition == "partition_aware_without_cross_edges":
        base = f"./data/fed_partition_aware_splits_without_cross_edges/{num_clients}_clients"
    else:
        raise ValueError(f"Unknown partition: {partition}")
    return (
        f"{base}/train/clients",
        f"{base}/val/clients",
        f"{base}/test/clients",
    )


def aggregate_state_dicts(local_state_dicts, sample_counts):
    """Weighted average of non-port parameters across clients."""
    total_samples = sum(sample_counts)
    agg_sd = {}
    for key in local_state_dicts[0]:
        if key in PORT_KEYS:
            continue  # skip — port tables have mismatched sizes
        agg_sd[key] = sum(
            (sample_counts[i] / total_samples) * local_state_dicts[i][key].float()
            for i in range(len(local_state_dicts))
        )
    return agg_sd


def extract_embeddings_for_client(cid, train_data, canon_model, args, device):
    """
    Run canon_model.encode() on all nodes of a client's training subgraph.
    Returns dict mapping global_nid (int) -> embedding [hidden_dim] for OWNED nodes only.
    """
    data = check_and_strip_self_loops(train_data, f"embed_client_{cid}")
    data = ensure_node_features(data)
    hetero = make_bidirected_hetero(data)

    # Clamp port IDs to global vocab if port embeddings are used
    if args.use_port_ids:
        fwd_ea = hetero[("n", "fwd", "n")].edge_attr.clone()
        fwd_ea[:, 0].clamp_(max=args.in_port_vocab_size - 1)
        fwd_ea[:, 1].clamp_(max=args.out_port_vocab_size - 1)
        hetero[("n", "fwd", "n")].edge_attr = fwd_ea

        rev_ea = hetero[("n", "rev", "n")].edge_attr.clone()
        rev_ea[:, 0].clamp_(max=args.out_port_vocab_size - 1)
        rev_ea[:, 1].clamp_(max=args.in_port_vocab_size - 1)
        hetero[("n", "rev", "n")].edge_attr = rev_ea

    # Full eval loader: all nodes as seeds, full k-hop neighborhood
    loader = build_full_eval_loader(
        hetero,
        batch_size=hetero["n"].num_nodes,
        num_layers=args.num_layers,
        device=device,
    )

    owned_mask_global = hetero["n"].owned_mask if hasattr(hetero["n"], "owned_mask") else None
    global_nids = hetero["n"].global_nid if hasattr(hetero["n"], "global_nid") else None

    client_embs = {}
    canon_model.eval()

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            x_in = {'n': batch['n'].x}
            edge_in = {
                ('n', 'fwd', 'n'): batch[('n', 'fwd', 'n')].edge_index,
                ('n', 'rev', 'n'): batch[('n', 'rev', 'n')].edge_index,
            }

            edge_attr_dict = None
            if args.use_port_ids:
                edge_attr_dict = {}
                for rel in [('n', 'fwd', 'n'), ('n', 'rev', 'n')]:
                    if 'edge_attr' in batch[rel]:
                        ea = batch[rel].edge_attr
                        if ea.dtype != torch.long:
                            ea = ea.long()
                        edge_attr_dict[rel] = ea

            embs = canon_model.encode(x_in, edge_in, edge_attr_dict=edge_attr_dict)  # [N, hidden_dim]

            # Recover node-level owned_mask and global_nid from the batch
            # NeighborLoader puts seed nodes first; for full-batch they cover all nodes
            batch_owned = batch['n'].owned_mask if hasattr(batch['n'], 'owned_mask') else None
            batch_gnids = batch['n'].global_nid if hasattr(batch['n'], 'global_nid') else None

            B = int(batch['n'].batch_size) if hasattr(batch['n'], 'batch_size') else embs.size(0)

            for i in range(B):
                is_owned = bool(batch_owned[i].item()) if batch_owned is not None else True
                if is_owned and batch_gnids is not None:
                    gnid = int(batch_gnids[i].item())
                    client_embs[gnid] = embs[i].cpu()

    return client_embs


def main():
    parser = argparse.ArgumentParser(description="Extract local embeddings for bootstrap FedAvg")
    parser.add_argument("--num_clients", type=int, required=True,
                        help="Number of clients (e.g. 3, 5, 10, 15)")
    parser.add_argument("--partition", type=str, default="partition_aware_with_cross_edges",
                        help="Partition strategy directory name")
    parsed = parser.parse_args()

    num_clients = parsed.num_clients
    partition = parsed.partition

    device = torch.device("cpu")  # embedding extraction is fast; CPU is fine

    # Load configs
    with open(PNA_CONFIG_PATH, "r") as f:
        pna_all = json.load(f)
    with open(FED_CONFIG_PATH, "r") as f:
        fed_all = json.load(f)

    pna_cfg = pna_all["reverse_mp_with_port_and_ego"]
    fed_cfg = fed_all["fed_learning_configs"]
    partition_cfg = fed_all["partition_aware_splits"]

    # Override num_clients from CLI; keep rest from config
    partition_cfg = dict(partition_cfg)
    partition_cfg["num_clients"] = num_clients

    args = build_args(pna_cfg, fed_cfg, partition_cfg)

    print(f"[Config] num_clients={num_clients}, partition={partition}")
    print(f"[Config] hidden_dim={args.hidden_dim}, num_layers={args.num_layers}")

    # Resolve data directories
    train_dir, val_dir, test_dir = resolve_data_dirs(num_clients, partition)
    print(f"[Data] train_dir={train_dir}")

    # Load all client training data
    train_list = load_client_graphs(train_dir, num_clients)
    print(f"[Data] Loaded {num_clients} clients' training subgraphs.")

    # --- Global port vocab precomputation (same as in train_fedavg.py) ---
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

    data_dir = "./data"
    base_seed = fed_cfg["base_seed"]
    seeds = [base_seed, base_seed + 1, base_seed + 2]

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"[Seed {seed}] Extracting embeddings for {num_clients} clients...")
        print(f"{'='*60}")

        set_seed(seed)

        # --- Step A: load local checkpoints ---
        # Local baseline uses seed+cid for client cid in outer seed loop starting at `seed`
        local_state_dicts = []
        sample_counts = []

        for cid in range(num_clients):
            client_seed = seed + cid  # matches run_local_client(seed=seed+cid) in train_local_baseline.py
            ckpt_glob = f"./checkpoints/local_baseline/{num_clients}_clients/client_{cid}_seed{client_seed}_*_best.pt"
            matches = sorted(glob.glob(ckpt_glob))
            if not matches:
                raise FileNotFoundError(
                    f"No checkpoint found for client {cid} seed {client_seed} ({num_clients} clients).\n"
                    f"  Looked for: {ckpt_glob}\n"
                    f"  Run train_local_baseline.py first (with num_clients={num_clients} in fed_configs.json)."
                )
            # Use the most recent run's checkpoint (sorted by run_id timestamp)
            ckpt_path = matches[-1]
            print(f"  [Client {cid}] Loading checkpoint: {ckpt_path}")
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            local_state_dicts.append(sd)
            sample_counts.append(int(train_list[cid].owned_mask.sum().item()))

        print(f"[Aggregation] Sample counts: {sample_counts}")

        # --- Step B: aggregate non-port parameters ---
        agg_sd = aggregate_state_dicts(local_state_dicts, sample_counts)
        print(f"[Aggregation] Averaged {len(agg_sd)} parameter tensors (skipped port embeddings)")

        # --- Step C: build canonical model using global port vocab ---
        # Use client 0's training data for degree histograms (minor approximation;
        # degree histograms only affect PNA scalers, not learned weights)
        canon_task = NodeClsTask(args, client_id=None, data=train_list[0], data_dir=data_dir, device=device)
        canon_model = canon_task.model

        # Load aggregated weights; port embeddings get fresh random init (strict=False expected)
        missing_keys, unexpected_keys = canon_model.load_state_dict(agg_sd, strict=False)
        expected_missing = {k for k in missing_keys if k in PORT_KEYS}
        unexpected_missing = {k for k in missing_keys if k not in PORT_KEYS}
        if unexpected_missing:
            raise RuntimeError(f"Unexpected missing keys when loading aggregated state dict: {unexpected_missing}")
        print(f"[Model] Missing keys (expected — port embeddings with fresh init): {expected_missing}")

        # --- Step D: extract embeddings for all clients ---
        canon_model.eval()
        global_emb = {}  # global_nid (int) -> embedding [hidden_dim]

        for cid in range(num_clients):
            print(f"  [Client {cid}] Extracting embeddings...")
            client_embs = extract_embeddings_for_client(cid, train_list[cid], canon_model, args, device)
            # Owned nodes from different clients should be disjoint; warn if overlap
            overlap = set(client_embs.keys()) & set(global_emb.keys())
            if overlap:
                print(f"  [WARNING] {len(overlap)} global_nid overlap between client {cid} and previous clients")
            global_emb.update(client_embs)
            print(f"  [Client {cid}] Extracted {len(client_embs)} owned-node embeddings")

        # --- Step E: save ---
        out_dir = f"./data/local_embeddings/{partition}/{num_clients}_clients/seed{seed}"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "node_embeddings.pt")
        torch.save(global_emb, out_path)
        print(f"[Saved] {len(global_emb)} node embeddings → {out_path}")

    print(f"\n[Done] Embeddings extracted for seeds {seeds}.")
    print(f"       Output dir: ./data/local_embeddings/{partition}/{num_clients}_clients/")


if __name__ == "__main__":
    main()
