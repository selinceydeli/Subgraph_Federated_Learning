import os
import csv
from datetime import datetime

import torch
from scripts.train_pna_reverse_mp_with_ego import run_pna

def hops_for_layers(L):
    """
    Number of hops should usually be ≥ num_layers
    To ensure that the receptive field covers enough neighbors for each message-passing layer
    """
    return [L, L + 1]

def neighbors_for_hops(H):
    grids = []
    # Uniform small / medium / large
    # grids.append([10] * H)
    # grids.append([15] * H)
    # grids.append([20] * H)

    # Decaying fanout variant
    if H == 2:
        grids.append([15, 10])
        grids.append([20, 10])
    elif H == 3:
        grids.append([15, 10, 5])
        grids.append([20, 15, 10])
    elif H >= 4:
        # e.g. 4 hops:  [15, 10, 5, 5]
        grids.append([15] + [10] * (H - 2) + [5])
    return grids

# Define the sub-tasks
tasks = ["deg-in", "deg-out", "fan-in", "fan-out", "C2", "C3", "C4", "C5", "C6", "S-G", "B-C"]

num_layers_grid = [2, 3]

grid = []
for L in num_layers_grid:
    for H in hops_for_layers(L):           
        for neigh in neighbors_for_hops(H):
            cfg = {
                "num_layers": L,
                "num_hops": H,
                "neighbors_per_hop": neigh,
                # fixed hyperparams:
                "use_ego_ids": False,
                "use_port_ids": False,
                "use_mini_batch": True,
                "batch_size": 32,
                "port_emb_dim": 8,
            }
            grid.append(cfg)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

seed = 0

results = [] 

for i, cfg in enumerate(grid):
    run_id = f"grid_{i:03d}"  
    print(f"\n=== Running config {i+1}/{len(grid)}: "
          f"L={cfg['num_layers']}, H={cfg['num_hops']}, neigh={cfg['neighbors_per_hop']} ===")

    test_loss, test_f1 = run_pna(
        seed=seed,
        tasks=tasks,
        device=device,
        run_id=run_id,
        **cfg,         
    )

    results.append((cfg, test_loss, test_f1.cpu()))

# Find best config by macro F1 score
best_cfg = None
best_macro_f1 = -1.0

for cfg, loss, f1 in results:
    macro_f1 = f1.mean().item()
    if macro_f1 > best_macro_f1:
        best_macro_f1 = macro_f1
        best_cfg = cfg

print("\nBest config by macro F1:")
print(best_cfg)
print(f"Macro F1: {best_macro_f1:.4f}")

output_dir = os.path.join("results", "parameter_tuning")
os.makedirs(output_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = os.path.join(output_dir, f"pna_reverse_mp_grid_{timestamp}.csv")

header = [
    "run_index",
    "num_layers",
    "num_hops",
    "neighbors_per_hop",
    "use_ego_ids",
    "use_port_ids",
    "use_mini_batch",
    "batch_size",
    "test_loss",
    "macro_f1",
]
header.extend([f"f1_{t}" for t in tasks])

with open(csv_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for idx, (cfg, loss, f1_tensor, macro_f1) in enumerate(results):
        row = [
            idx,
            cfg["num_layers"],
            cfg["num_hops"],
            " ".join(map(str, cfg["neighbors_per_hop"])),  
            cfg["use_ego_ids"],
            cfg["use_port_ids"],
            cfg["use_mini_batch"],
            cfg["batch_size"],
            float(loss),
            macro_f1,
        ]
        f1_list = f1_tensor.tolist()
        row.extend([float(v) for v in f1_list])

        writer.writerow(row)

print(f"\nSaved grid search results to: {csv_path}")