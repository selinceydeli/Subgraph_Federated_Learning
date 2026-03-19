#!/usr/bin/env python3
import os
import json
import torch
import torch.nn as nn
from torch_geometric.utils import degree

from utils.metrics import append_f1_score_to_csv, append_pr_auc_to_csv, start_epoch_csv, append_epoch_csv
from utils.seed import set_seed
from utils.train_utils import load_datasets, ensure_node_features, train_epoch, evaluate_epoch
from models.pna_baseline import PNANet

CONFIG_PATH = "./configs/pna_configs.json"

with open(CONFIG_PATH, "r") as f:
    ALL_CONFIG = json.load(f)

CONFIG = ALL_CONFIG["pna_baseline"]

MODEL_NAME = CONFIG["model_name"]
BEST_MODEL_PATH = CONFIG["best_model_path"]

NUM_EPOCHS = CONFIG["num_epochs"]
DEFAULT_HPARAMS = CONFIG["default_hparams"]

def run_pna(seed, tasks, device):
    set_seed(seed)

    # Default hyperparameters
    cfg = {
        "num_epochs": NUM_EPOCHS,
        **DEFAULT_HPARAMS,
    }

    print(cfg)

    num_epochs = cfg["num_epochs"]
    num_layers = cfg["num_layers"]
    hidden_dim = cfg["hidden_dim"]
    dropout = cfg["dropout"]
    lr = cfg["lr"]
    weight_decay = cfg["weight_decay"]

    train_data, val_data, test_data = load_datasets()

    # Assign constant features
    train_data = ensure_node_features(train_data)
    val_data = ensure_node_features(val_data)
    test_data = ensure_node_features(test_data)

    d = degree(train_data.edge_index[1], num_nodes=train_data.num_nodes).long()
    deg_hist = torch.bincount(d, minlength=int(d.max()) + 1)

    # Define the model
    in_dim = train_data.num_node_features if train_data.x is not None else 1
    out_dim = train_data.y.size(-1)
    model = PNANet(
        in_dim=in_dim, 
        hidden_dim=hidden_dim, 
        out_dim=out_dim, 
        deg=deg_hist, 
        num_layers=num_layers, 
        dropout=dropout
    ).to(device)

    # Load the datasets
    # Note: Because we currently have only 1 graph per split, there is no need for batching.
    train_loader = [train_data]
    valid_loader = [val_data]
    test_loader  = [test_data]

    # Define optimizer and loss functions
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) # Define optimizer as Adam
    criterion = nn.BCEWithLogitsLoss() # Define loss as binary cross-entropy (preferred for multi-label classification task we have here)

    os.makedirs(BEST_MODEL_PATH, exist_ok=True)

    # Log the epoch results
    epoch_csv_path = start_epoch_csv(
        model_name=MODEL_NAME,
        seed=seed,
        tasks=tasks,
        out_dir=f"./results/metrics/epoch_logs/{MODEL_NAME}"
    )

    # Training loop
    best_val = float("inf")
    for epoch in range(1, num_epochs):  # a few more epochs helps stabilize F1
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, _, val_f1, val_pr_auc = evaluate_epoch(model, valid_loader, criterion, device)

        append_epoch_csv(epoch_csv_path, epoch, train_loss, val_loss, val_f1, val_pr_auc)

        val_macro = val_f1.mean().item()

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(BEST_MODEL_PATH, f"best_pna_baseline_seed{seed}.pt"))

        if epoch % 10 == 0:
            print(f"[seed {seed}] Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f} | val macro-minF1 {100*val_macro:.2f}%")

    # Save the best model and evaluate on test dataset
    model.load_state_dict(torch.load(os.path.join(BEST_MODEL_PATH, f"best_pna_baseline_seed{seed}.pt"), map_location=device))
    test_loss, _, test_f1, test_pr_auc = evaluate_epoch(model, test_loader, criterion, device)
    return test_loss, test_f1, test_pr_auc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the sub-tasks
    tasks = ["deg-in","deg-out","fan-in","fan-out","C2","C3","C4","C5","C6","S-G","B-C"]

    seeds = [0,1,2,3,4]
    test_f1_scores     = []
    test_pr_auc_scores = []
    for s in seeds:
        _, test_f1, test_pr_auc = run_pna(s, tasks, device)
        test_f1_scores.append(test_f1.cpu())
        test_pr_auc_scores.append(test_pr_auc.cpu())

    all_f1 = torch.stack(test_f1_scores, dim=0)
    mean_f1 = all_f1.mean(dim=0)
    std_f1  = all_f1.std(dim=0, unbiased=False)
    macro_mean = mean_f1.mean().item() * 100

    all_pr_auc = torch.stack(test_pr_auc_scores, dim=0)
    mean_pr_auc = all_pr_auc.mean(dim=0)
    std_pr_auc  = all_pr_auc.std(dim=0, unbiased=False)
    macro_pr_auc = mean_pr_auc.mean().item() * 100

    print(f"\nPNA baseline — macro minority F1 over 5 runs: {macro_mean:.2f}%")
    row = " | ".join(f"{n}: {100*m:.2f}±{100*s:.2f}%" for n, m, s in zip(tasks, mean_f1.tolist(), std_f1.tolist()))
    print("Per-task (mean±std over 5 runs):", row)
    print(f"macro PR-AUC: {macro_pr_auc:.2f}%")
    row_pr = " | ".join(f"{n}: {100*m:.2f}±{100*s:.2f}%" for n, m, s in zip(tasks, mean_pr_auc.tolist(), std_pr_auc.tolist()))
    print("Per-task PR-AUC (mean±std):", row_pr)

    append_f1_score_to_csv(
        out_csv="./results/metrics/f1_scores.csv",
        tasks=tasks,
        mean_f1=mean_f1,
        std_f1=std_f1,
        macro_mean_percent=macro_mean,
        seeds=seeds,
        model_name="PNA baseline",
    )

    append_pr_auc_to_csv(
        out_csv="./results/metrics/pr_auc_scores.csv",
        tasks=tasks,
        mean_pr_auc=mean_pr_auc,
        std_pr_auc=std_pr_auc,
        macro_mean_prauc=macro_pr_auc,
        seeds=seeds,
        model_name="PNA baseline",
    )


if __name__ == "__main__":
    main()
