#!/usr/bin/env python3
import os
import time
import torch
import torch.nn as nn
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import remove_self_loops

from utils.metrics import append_f1_score_to_csv, start_epoch_csv, append_epoch_csv
from utils.seed import set_seed
from utils.train_utils import load_datasets, ensure_node_features, train_epoch, evaluate_epoch
from utils.hetero import make_bidirected_hetero
from models.pna_reverse_mp import PNANetReverseMP, compute_directional_degree_hists

# Model configs
BEST_MODEL_PATH = "./checkpoints/pna_reverse_mp_with_ego"
MODEL_NAME = "pna_reverse_mp_with_ego"

# Train configs
USE_EGO_IDS = False
BATCH_SIZE = 32
EGO_DIM = BATCH_SIZE 
NUM_EPOCHS = 100

def check_and_strip_self_loops(data, name):
    ei = data.edge_index
    has_loops = bool((ei[0] == ei[1]).any())
    print(f"[{name}] self-loops? {has_loops}")
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


def run_pna(seed, tasks, device):
    set_seed(seed)

    train_data, val_data, test_data = load_datasets()

    # Check for self loops and remove if any
    train_data = check_and_strip_self_loops(train_data, "train")
    val_data   = check_and_strip_self_loops(val_data, "val")
    test_data  = check_and_strip_self_loops(test_data, "test")

    # Assign constant features
    train_data = ensure_node_features(train_data)
    val_data = ensure_node_features(val_data)
    test_data = ensure_node_features(test_data)

    # Convert the data into HeteroData format
    # using forward and backward edge relations
    train_h = make_bidirected_hetero(train_data)
    val_h   = make_bidirected_hetero(val_data)
    test_h  = make_bidirected_hetero(test_data)

    # PNA degree histograms per direction
    # computed once on the full training graph (before small batches are sampled)
    deg_fwd_hist, deg_rev_hist = compute_directional_degree_hists(
        edge_index=train_data.edge_index,  # original edges
        num_nodes=train_data.num_nodes,
    )

    # Define the model
    in_dim = train_h['n'].x.size(-1) if 'x' in train_h['n'] else 1
    out_dim = train_h['n'].y.size(-1)

    # Define the number of layers
    # Best number of layers found so far is 2
    num_layers = 2      
    print(f"Number of layers using in training: {num_layers}")

    if USE_EGO_IDS:
        ego_dim = EGO_DIM
    else:
        ego_dim = 0
    
    model = PNANetReverseMP(
        in_dim=in_dim,
        hidden_dim=64,
        out_dim=out_dim,
        deg_fwd=deg_fwd_hist,
        deg_rev=deg_rev_hist,
        num_layers=num_layers,
        dropout=0.1,
        ego_dim=ego_dim, # pass ego dimension
        combine="sum",   # other aggregation options: 'mean' or 'max'
    ).to(device)

    # Load the hetero datasets
    # Use hetero neighbor loader for the training data
    train_loader = build_hetero_neighbor_loader(train_h, BATCH_SIZE, num_layers, fanout=[10, 4], device=device) # 1st hop 10, 2nd hop 4

    # For validation and test, again use hetero neighbor loader for consistency
    valid_loader = build_hetero_neighbor_loader(val_h,   BATCH_SIZE, num_layers, fanout=[10, 4], device=device)
    test_loader  = build_hetero_neighbor_loader(test_h,  BATCH_SIZE, num_layers, fanout=[10, 4], device=device)

    # Define optimizer and loss functions
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4) # Define optimizer as Adam
    criterion = nn.BCEWithLogitsLoss() # Define loss as binary cross-entropy (preferred for multi-label classification task we have here)

    # Log the epoch results
    os.makedirs(BEST_MODEL_PATH, exist_ok=True)

    epoch_csv_path = start_epoch_csv(
        model_name=MODEL_NAME,
        seed=seed,
        tasks=tasks,
        out_dir=f"./results/metrics/epoch_logs/{MODEL_NAME}"
    )

    # Training loop
    best_val = float("inf")
    for epoch in range(1, NUM_EPOCHS + 1):  # a few more epochs helps stabilize F1
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, _, val_f1 = evaluate_epoch(model, valid_loader, criterion, device)

        append_epoch_csv(epoch_csv_path, epoch, train_loss, val_loss, val_f1)

        val_macro = val_f1.mean().item()

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(BEST_MODEL_PATH, f"best_pna_reverse_mp_seed{seed}.pt"))

        # Print training and validation results after each epoch
        print(f"[seed {seed}] Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f} | val macro-minF1 {100*val_macro:.2f}%")

    # Save the best model and evaluate on test dataset
    model.load_state_dict(torch.load(os.path.join(BEST_MODEL_PATH, f"best_pna_reverse_mp_seed{seed}.pt"), map_location=device))
    test_loss, _, test_f1 = evaluate_epoch(model, test_loader, criterion, device)
    return test_loss, test_f1  


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_ts = time.perf_counter() 

    # Define the sub-tasks
    tasks = ["deg-in","deg-out","fan-in","fan-out","C2","C3","C4","C5","C6","S-G","B-C"]

    seeds = [0,1,2,3,4]
    test_f1_scores = []
    for s in seeds:
        _, test_f1 = run_pna(s, tasks, device)
        test_f1_scores.append(test_f1.cpu())

    all_f1 = torch.stack(test_f1_scores, dim=0)        
    mean_f1 = all_f1.mean(dim=0)              
    std_f1  = all_f1.std(dim=0, unbiased=False)

    macro_mean = mean_f1.mean().item()*100
    print(f"\nPNA reverse message passing with ego IDs — macro minority F1 over 5 runs: {macro_mean:.2f}%")

    row = " | ".join(f"{n}: {100*m:.2f}±{100*s:.2f}%" for n, m, s in zip(tasks, mean_f1.tolist(), std_f1.tolist()))
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
        model_name=f"PNA reverse MP with mini batch training & ego IDs={USE_EGO_IDS}",
        runtime_seconds=runtime_sec,
    )


if __name__ == "__main__":
    main()
