#!/usr/bin/env python3
import os
import time
import json
from datetime import datetime
import torch
import torch.nn as nn

from utils.metrics import append_f1_score_to_csv, start_epoch_csv, append_epoch_csv
from utils.seed import set_seed
from utils.train_utils import load_datasets, ensure_node_features, train_epoch, evaluate_epoch
from utils.hetero import make_bidirected_hetero
from utils.graph_helpers import max_port_cols, check_and_strip_self_loops, build_hetero_neighbor_loader, build_full_eval_loader
from models.pna_reverse_mp import PNANetReverseMP, compute_directional_degree_hists

CONFIG_PATH = "./configs/pna_configs.json"

with open(CONFIG_PATH, "r") as f:
    ALL_CONFIG = json.load(f)

CONFIG = ALL_CONFIG["reverse_mp_with_port_and_ego"]

MODEL_NAME = CONFIG["model_name"]
BEST_MODEL_PATH = CONFIG["best_model_path"]

USE_EGO_IDS = CONFIG["use_ego_ids"]
USE_PORT_IDS = CONFIG["use_port_ids"]
USE_MINI_BATCH = CONFIG["use_mini_batch"]
BATCH_SIZE = CONFIG["batch_size"]
PORT_EMB_DIM = CONFIG["port_emb_dim"]
NUM_EPOCHS = CONFIG["num_epochs"]

DEFAULT_HPARAMS = CONFIG["default_hparams"]


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
        "use_mini_batch": USE_MINI_BATCH,
        "port_emb_dim": PORT_EMB_DIM,
        "num_epochs": NUM_EPOCHS,
        # plus everything in DEFAULT_HPARAMS
        **DEFAULT_HPARAMS,
    }

    # Override with user-provided hparams
    cfg = {**default_cfg, **hparams}

    use_ego_ids = cfg["use_ego_ids"]
    batch_size = cfg["batch_size"]
    use_mini_batch = cfg["use_mini_batch"]
    use_port_ids = cfg["use_port_ids"]
    port_emb_dim = cfg["port_emb_dim"]
    num_epochs = cfg["num_epochs"]
    num_layers = cfg["num_layers"]
    num_hops = num_layers # there is one hop per layer in PNA conv
    neighbors_per_hop = cfg["neighbors_per_hop"]
    hidden_dim = cfg["hidden_dim"]
    dropout = cfg["dropout"]
    lr = cfg["lr"]
    weight_decay = cfg["weight_decay"]
    minority_class_weight = cfg["minority_class_weight"]

    print(f"Training with hyperparameters:{cfg}")

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

    # Compute per-task pos_weight if enabled
    auto_pos_weight = None
    if isinstance(minority_class_weight, str) and minority_class_weight == "auto":
        y_train = train_h['n'].y.float()
        pos_counts = y_train.sum(dim=0)                 # [num_tasks]
        neg_counts = (1.0 - y_train).sum(dim=0)         # [num_tasks]
        eps = 1e-8
        auto_pos_weight = neg_counts / (pos_counts + eps)

    # Decide batch sizes depending on mini-batch vs full-batch mode
    if use_mini_batch:
        train_batch_size = batch_size
        val_batch_size   = batch_size
        test_batch_size  = batch_size
        # print(f"[TRAIN MODE] mini-batch | B={train_batch_size}")
    else:
        train_batch_size = train_h['n'].num_nodes
        val_batch_size   = val_h['n'].num_nodes
        test_batch_size  = test_h['n'].num_nodes
        # print(f"[TRAIN MODE] FULL-BATCH | "
        #     f"train B={train_batch_size}, val B={val_batch_size}, test B={test_batch_size}")
        
    # Set ego IDs
    if use_ego_ids:
        ego_dim = train_batch_size if cfg.get("ego_dim") is None else cfg["ego_dim"]
        #print(f"Training with Ego IDs... ego_dim={ego_dim}")
    else:
        ego_dim = 0
        #print("Training without Ego IDs...")

    # Define the model
    in_dim = train_h['n'].x.size(-1) if 'x' in train_h['n'] else 1
    out_dim = train_h['n'].y.size(-1)

    # print(f"Number of layers using in training: {num_layers}")
    # print(f"Number of hops for NeighborLoader: {num_hops}")
    # print(f"Neighbors per hop: {neighbors_per_hop}")

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

    # print(
    #     f"[PORT] enabled={use_port_ids} "
    #     f"vocab_in={in_port_vocab_size} vocab_out={out_port_vocab_size} "
    #     f"emb_dim={(port_emb_dim if use_port_ids else 0)}"
    # )

    if use_mini_batch:
        # Mini-batch training + mini-batch validation/test
        train_loader = build_hetero_neighbor_loader(
            train_h,
            batch_size=train_batch_size,
            num_layers=num_hops,
            fanout=neighbors_per_hop,
            device=device,
        )

        valid_loader = build_hetero_neighbor_loader(
            val_h,
            batch_size=val_batch_size,
            num_layers=num_hops,
            fanout=neighbors_per_hop,
            device=device,
        )

        test_loader = build_hetero_neighbor_loader(
            test_h,
            batch_size=test_batch_size,
            num_layers=num_hops,
            fanout=neighbors_per_hop,
            device=device,
        )
    else:
        # Full-batch training + full-batch validation/test
        # Use -1 neighbors to pull the full k-hop neighborhood, one batch per split
        train_loader = build_full_eval_loader(
            train_h,
            batch_size=train_batch_size,   # equal to num_nodes
            num_layers=num_hops,
            device=device,
        )

        valid_loader = build_full_eval_loader(
            val_h,
            batch_size=val_batch_size,    
            num_layers=num_hops,
            device=device,
        )

        test_loader = build_full_eval_loader(
            test_h,
            batch_size=test_batch_size,   
            num_layers=num_hops,
            device=device,
        )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Minority class weighting via pos_weight
    if isinstance(minority_class_weight, str) and minority_class_weight == "auto":
        # Use per-task pos_weight computed from training labels
        assert auto_pos_weight is not None, "auto_pos_weight should have been computed above"
        criterion = nn.BCEWithLogitsLoss(pos_weight=auto_pos_weight.to(device))
        print(f"Using automatic per-task minority weighting: {auto_pos_weight.tolist()}")
    elif minority_class_weight is not None:
        # Use a uniform scalar pos_weight across all tasks
        pos_weight = torch.full((out_dim,), float(minority_class_weight), device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Using uniform minority class weight: {minority_class_weight}")
    else:
        # No weighting
        criterion = nn.BCEWithLogitsLoss()
        print("Using unweighted BCEWithLogitsLoss.")

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
        num_layers=DEFAULT_HPARAMS["num_layers"],
        neighbors_per_hop=DEFAULT_HPARAMS["neighbors_per_hop"],
        minority_class_weight=DEFAULT_HPARAMS["minority_class_weight"],  
        use_ego_ids=USE_EGO_IDS,
        use_mini_batch=USE_MINI_BATCH,
        batch_size=BATCH_SIZE,
        use_port_ids=USE_PORT_IDS,
        port_emb_dim=PORT_EMB_DIM,
        num_epochs=NUM_EPOCHS,
        hidden_dim=DEFAULT_HPARAMS["hidden_dim"],
        dropout=DEFAULT_HPARAMS["dropout"],
        lr=DEFAULT_HPARAMS["lr"],
        weight_decay=DEFAULT_HPARAMS["weight_decay"],
    )

    # seeds = [0, 1, 2, 3, 4]
    seeds = [0] # for testing
    test_f1_scores = []
    for s in seeds:
        _, test_f1 = run_pna(s, tasks, device, run_id=run_id, **base_hparams)
        test_f1_scores.append(test_f1.cpu())

    all_f1 = torch.stack(test_f1_scores, dim=0)
    mean_f1 = all_f1.mean(dim=0)
    std_f1  = all_f1.std(dim=0, unbiased=False)

    macro_mean = mean_f1.mean().item() * 100

    mode_str = "mini-batch" if USE_MINI_BATCH else "full-batch"
    print(f"\nPNA reverse message passing with {mode_str} training, "
        f"port numbers={USE_PORT_IDS}, & ego IDs={USE_EGO_IDS} — macro minority F1 over 5 runs: {macro_mean:.2f}%")
    
    row = " | ".join(
        f"{n}: {100*m:.2f}±{100*s:.2f}%"
        for n, m, s in zip(tasks, mean_f1.tolist(), std_f1.tolist())
    )
    print("Per-task (mean±std over 5 runs):", row)

    runtime_sec = time.perf_counter() - start_ts

    append_f1_score_to_csv(
        out_csv="./results/metrics/f1_scores.csv",
        tasks=tasks,
        mean_f1=mean_f1,
        std_f1=std_f1,
        macro_mean_percent=macro_mean,
        seeds=seeds,
        model_name=f"PNA reverse MP with {mode_str} training, port numbers={USE_PORT_IDS}, & ego IDs={USE_EGO_IDS}",
        runtime_seconds=runtime_sec,
    )

if __name__ == "__main__":
    main()