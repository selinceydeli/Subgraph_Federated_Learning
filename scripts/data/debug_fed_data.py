import os
import json
import torch

FED_CONFIG_PATH = "./configs/fed_configs.json"

with open(FED_CONFIG_PATH, "r") as f:
    ALL_FED_CONFIG = json.load(f)

FED_DATA_CONFIG = ALL_FED_CONFIG["federated_dataset_simulation"]
NUM_CLIENTS = FED_DATA_CONFIG["num_clients"]

FED_TRAIN_SPLITS_DIR = "./data/fed_louvain"  # or fed_metis

for cid in range(NUM_CLIENTS):  
    path = os.path.join(FED_TRAIN_SPLITS_DIR, f"client_{cid}.pt")
    g = torch.load(path, weights_only=False)
    print(f"\n=== client_{cid} ===")
    print(g)

    # basic structural checks
    ei = g.edge_index
    num_nodes = g.num_nodes

    print("num_nodes:", num_nodes)
    print("edge_index max:", int(ei.max()) if ei.numel() > 0 else None)

    if ei.numel() > 0 and int(ei.max()) >= num_nodes:
        print(">>> INVALID: edge_index has node >= num_nodes!")

print("has x:", hasattr(g, "x"), None if not hasattr(g, "x") else g.x.shape)
print("has y:", hasattr(g, "y"), None if not hasattr(g, "y") else g.y.shape)
print("has edge_attr:", hasattr(g, "edge_attr"),
      None if not hasattr(g, "edge_attr") else g.edge_attr.shape)
