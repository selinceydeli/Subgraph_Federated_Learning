#!/usr/bin/env python3
import os
import time
import json
from datetime import datetime
import torch
import torch.nn as nn
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import remove_self_loops

from utils.metrics import append_f1_score_to_csv, start_epoch_csv, append_epoch_csv
from utils.seed import set_seed
from utils.train_utils import load_datasets, ensure_node_features, train_epoch, evaluate_epoch
from utils.hetero import make_bidirected_hetero
from models.pna_reverse_mp import PNANetReverseMP, compute_directional_degree_hists

CONFIG_PATH = "./configs/pna_configs.json"

with open(CONFIG_PATH, "r") as f:
    ALL_CONFIG = json.load(f)

CONFIG = ALL_CONFIG["reverse_mp_with_port_and_ego"]

MODEL_NAME = CONFIG["model_name"]
BEST_MODEL_PATH = CONFIG["best_model_path"]

USE_EGO_IDS = CONFIG["use_ego_ids"]
USE_PORT_IDS = CONFIG["use_port_ids"]
BATCH_SIZE = CONFIG["batch_size"]
PORT_EMB_DIM = CONFIG["port_emb_dim"]
NUM_EPOCHS = CONFIG["num_epochs"]

DEFAULT_HPARAMS = CONFIG["default_hparams"]

def max_port_cols(d):
        in_col, out_col = d.edge_attr.size(-1) - 2, d.edge_attr.size(-1) - 1
        return int(d.edge_attr[:, in_col].max().item()), int(d.edge_attr[:, out_col].max().item())


def check_and_strip_self_loops(data, name):
    ei = data.edge_index
    has_loops = bool((ei[0] == ei[1]).any())
    #print(f"[{name}] self-loops? {has_loops}")
    if has_loops:
        ei_clean, ea_clean = remove_self_loops(ei, getattr(data, "edge_attr", None))
        data.edge_index = ei_clean
        if hasattr(data, "edge_attr"):
            data.edge_attr = ea_clean
        print(f"[{name}] removed self-loops → E={data.edge_index.size(1)}")
    return data


def build_hetero_neighbor_loader(hetero_data, batch_size, num_layers, fanout, device=None):
    """
    Creates mini-batches of seed nodes (the first batch_size nodes) and their sampled neighbors.

    Parameters:
    - hetero_data: HeteroData object
    - batch_size: number of seed nodes
    - num_layers: number of hops 
    - fanout: number of neighbors per hop, e.g. 15 (or a tuple/list per hop)
    """
    if isinstance(fanout, int):
        fanout_list = [fanout] * num_layers
    else:
        fanout_list = list(fanout)  # e.g. [20, 15, 10] if num_layers=3

    num_neighbors = {
        ('n','fwd','n'): fanout_list,
        ('n','rev','n'): fanout_list,
    }

    use_cuda = (device is not None and device.type == "cuda")
    num_workers = max(1, os.cpu_count() // 2)

    return NeighborLoader(
        hetero_data,
        num_neighbors=num_neighbors,
        input_nodes=('n', torch.arange(hetero_data['n'].num_nodes)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=use_cuda,                 # speeds host→GPU copy
        num_workers=num_workers,             # parallel sampling
        persistent_workers=True,             # keep workers alive
        prefetch_factor=2,                   # overlap next batch
        filter_per_worker=True,              # filter on worker side (PyG>=2.3)
    )


def build_full_eval_loader(hetero_data, batch_size, num_layers, device=None):
    """
    Covers ALL nodes as seeds and expands with ALL neighbors up to `num_layers`.
    This yields exact full-graph metrics without holding the whole graph at once.
    """
    fanout_all = [-1] * num_layers  # -1 => take all neighbors at that hop
    num_neighbors = {
        ('n','fwd','n'): fanout_all,
        ('n','rev','n'): fanout_all,
    }

    use_cuda = (device is not None and device.type == "cuda")
    num_workers = max(1, os.cpu_count() // 2)

    return NeighborLoader(
        hetero_data,
        num_neighbors=num_neighbors,
        input_nodes=('n', torch.arange(hetero_data['n'].num_nodes)),
        batch_size=batch_size,
        shuffle=False,                 # deterministic, cover each node once
        drop_last=False,
        pin_memory=use_cuda,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=2,
        filter_per_worker=True,
    )


def run_pna(seed, tasks, device, run_id, **hparams):
    """
    Run a single PNA experiment with the given seed and hyperparameters.

    Hyperparameters (all kwargs, with defaults):
      - use_ego_ids: bool
      - batch_size: int
      - ego_dim: int or None (defaults to batch_size if use_ego_ids)
      - use_port_ids: bool
      - port_emb_dim: int
      - num_epochs: int
      - num_layers: int          (PNA depth)
      - num_hops: int or None    (NeighborLoader hops; defaults to num_layers)
      - neighbors_per_hop: int or list[int]
      - hidden_dim: int
      - dropout: float
      - lr: float
      - weight_decay: float
      - minority_class_weight: float or None (for BCEWithLogitsLoss pos_weight)
    """
    set_seed(seed)

    # Default hyperparameters
    default_cfg = {
        "use_ego_ids": USE_EGO_IDS,
        "batch_size": BATCH_SIZE,
        "use_port_ids": USE_PORT_IDS,
        "port_emb_dim": PORT_EMB_DIM,
        "num_epochs": NUM_EPOCHS,
        # plus everything in DEFAULT_HPARAMS
        **DEFAULT_HPARAMS,
    }

    # Override with user-provided hparams
    cfg = {**default_cfg, **hparams}

    use_ego_ids = cfg["use_ego_ids"]
    batch_size = cfg["batch_size"]
    use_port_ids = cfg["use_port_ids"]
    port_emb_dim = cfg["port_emb_dim"]
    num_epochs = cfg["num_epochs"]
    num_layers = cfg["num_layers"]
    num_hops = cfg["num_hops"] if cfg["num_hops"] is not None else num_layers
    neighbors_per_hop = cfg["neighbors_per_hop"]
    hidden_dim = cfg["hidden_dim"]
    dropout = cfg["dropout"]
    lr = cfg["lr"]
    weight_decay = cfg["weight_decay"]
    minority_class_weight = cfg["minority_class_weight"]

    print("Training with hyperparameters:")
    print(cfg)

    # ego_dim default logic
    if use_ego_ids:
        ego_dim = batch_size if cfg["ego_dim"] is None else cfg["ego_dim"]
        print("Training with Ego IDs...")
    else:
        ego_dim = 0
        print("Training without Ego IDs...")

    model_dir = os.path.join(BEST_MODEL_PATH, f"run_{run_id}_seed{seed}")
    os.makedirs(model_dir, exist_ok=True)

    # Data loading
    train_data, val_data, test_data = load_datasets()

    # Check for self loops and remove if any
    train_data = check_and_strip_self_loops(train_data, "train")
    val_data   = check_and_strip_self_loops(val_data, "val")
    test_data  = check_and_strip_self_loops(test_data, "test")

    # Assign constant features
    train_data = ensure_node_features(train_data)
    val_data   = ensure_node_features(val_data)
    test_data  = ensure_node_features(test_data)

    # Find maximum port in and out degrees if port IDs are present
    if use_port_ids:
        tr_in_max, tr_out_max = max_port_cols(train_data)
        va_in_max, va_out_max = max_port_cols(val_data)
        te_in_max, te_out_max = max_port_cols(test_data)
        in_port_vocab_size  = max(tr_in_max,  va_in_max,  te_in_max)  + 1
        out_port_vocab_size = max(tr_out_max, va_out_max, te_out_max) + 1
    else:
        in_port_vocab_size  = 0
        out_port_vocab_size = 0

    # Convert the data into HeteroData format
    train_h = make_bidirected_hetero(train_data)
    val_h   = make_bidirected_hetero(val_data)
    test_h  = make_bidirected_hetero(test_data)

    # PNA degree histograms per direction
    deg_fwd_hist, deg_rev_hist = compute_directional_degree_hists(
        edge_index=train_data.edge_index,
        num_nodes=train_data.num_nodes,
    )

    # Define the model
    in_dim = train_h['n'].x.size(-1) if 'x' in train_h['n'] else 1
    out_dim = train_h['n'].y.size(-1)

    print(f"Number of layers using in training: {num_layers}")
    print(f"Number of hops for NeighborLoader: {num_hops}")
    print(f"Neighbors per hop: {neighbors_per_hop}")

    model = PNANetReverseMP(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        deg_fwd=deg_fwd_hist,
        deg_rev=deg_rev_hist,
        num_layers=num_layers,
        dropout=dropout,
        ego_dim=ego_dim,  # pass ego dimension
        combine="sum",
        in_port_vocab_size=in_port_vocab_size,
        out_port_vocab_size=out_port_vocab_size,
        port_emb_dim=(port_emb_dim if use_port_ids else 0),
    ).to(device)

    print(
        f"[PORT] enabled={use_port_ids} "
        f"vocab_in={in_port_vocab_size} vocab_out={out_port_vocab_size} "
        f"emb_dim={(port_emb_dim if use_port_ids else 0)}"
    )

    # Use hetero neighbor loader for the training data
    train_loader = build_hetero_neighbor_loader(
        train_h,
        batch_size=batch_size,
        num_layers=num_hops,
        fanout=neighbors_per_hop,
        device=device,
    )

    # For validation and test, same sampling scheme
    valid_loader = build_hetero_neighbor_loader(
        val_h,
        batch_size=batch_size,
        num_layers=num_hops,
        fanout=neighbors_per_hop,
        device=device,
    )

    test_loader  = build_hetero_neighbor_loader(
        test_h,
        batch_size=batch_size,
        num_layers=num_hops,
        fanout=neighbors_per_hop,
        device=device,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Minority class weighting via pos_weight
    if minority_class_weight is not None:
        pos_weight = torch.full((out_dim,), float(minority_class_weight), device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Using minority class weight: {minority_class_weight}")
    else:
        criterion = nn.BCEWithLogitsLoss()

    os.makedirs(BEST_MODEL_PATH, exist_ok=True)

    epoch_csv_path = start_epoch_csv(
        model_name=MODEL_NAME,
        seed=seed,
        tasks=tasks,
        out_dir=f"./results/metrics/epoch_logs/{MODEL_NAME}"
    )

   
    best_ckpt_path = os.path.join(model_dir, "best_model.pt")

    best_val = float("inf")
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            use_port_ids,
        )
        val_loss, _, val_f1 = evaluate_epoch(
            model,
            valid_loader,
            criterion,
            device,
            use_port_ids,
        )

        append_epoch_csv(epoch_csv_path, epoch, train_loss, val_loss, val_f1)

        val_macro = val_f1.mean().item()

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_ckpt_path)

        print(
            f"[seed {seed}] Epoch {epoch:03d} | "
            f"train {train_loss:.4f} | val {val_loss:.4f} | "
            f"val macro-minF1 {100*val_macro:.2f}%"
        )

    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    test_loss, _, test_f1 = evaluate_epoch(
        model,
        test_loader,
        criterion,
        device,
        use_port_ids,
    )
    return test_loss, test_f1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    start_ts = time.perf_counter()

    # Define the sub-tasks
    tasks = ["deg-in", "deg-out", "fan-in", "fan-out", "C2", "C3", "C4", "C5", "C6", "S-G", "B-C"]

    # Example hyperparameter config for this run.
    # For hyperparameter tuning, you can vary these and call run_pna with different kwargs.
    base_hparams = dict(
        num_layers=2,
        num_hops=2,
        neighbors_per_hop=[10, 4],
        minority_class_weight=None,  # e.g. 2.0 if you want to up-weight positives
        use_ego_ids=USE_EGO_IDS,
        batch_size=BATCH_SIZE,
        use_port_ids=USE_PORT_IDS,
        port_emb_dim=PORT_EMB_DIM,
        num_epochs=NUM_EPOCHS,
        hidden_dim=64,
        dropout=0.1,
        lr=1e-3,
        weight_decay=1e-4,
    )

    seeds = [0, 1, 2, 3, 4]
    test_f1_scores = []
    for s in seeds:
        _, test_f1 = run_pna(s, tasks, device, run_id=run_id, **base_hparams)
        test_f1_scores.append(test_f1.cpu())

    all_f1 = torch.stack(test_f1_scores, dim=0)
    mean_f1 = all_f1.mean(dim=0)
    std_f1  = all_f1.std(dim=0, unbiased=False)

    macro_mean = mean_f1.mean().item() * 100
    print(f"\nPNA reverse message passing with mini batch training, "
          f"port numbers, & ego IDs={USE_EGO_IDS} — macro minority F1 over 5 runs: {macro_mean:.2f}%")

    row = " | ".join(
        f"{n}: {100*m:.2f}±{100*s:.2f}%"
        for n, m, s in zip(tasks, mean_f1.tolist(), std_f1.tolist())
    )
    print("Per-task (mean±std over 5 runs):", row)

    runtime_sec = time.perf_counter() - start_ts

    # Append F1 scores to CSV
    append_f1_score_to_csv(
        out_csv="./results/metrics/f1_scores.csv",
        tasks=tasks,
        mean_f1=mean_f1,
        std_f1=std_f1,
        macro_mean_percent=macro_mean,
        seeds=seeds,
        model_name=f"PNA reverse MP with mini batch training, port numbers, & ego IDs={USE_EGO_IDS}",
        runtime_seconds=runtime_sec,
    )


if __name__ == "__main__":
    main()