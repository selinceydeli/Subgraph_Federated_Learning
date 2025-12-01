# make_federated_splits.py
import os
import json
import torch

from utils.seed import set_seed, derive_seed
from utils.fed_partitioning import graphdata_to_pyg
from utils.fed_simulation import louvain_label_imbalance_split, metis_label_imbalance_split

CONFIG_PATH = "./configs/fed_configs.json"

with open(CONFIG_PATH, "r") as f:
    ALL_CONFIG = json.load(f)

CONFIG = ALL_CONFIG["federated_dataset_simulation"]

NUM_CLIENTS = CONFIG["num_clients"]
LOUVAIN_RESOLUTION = CONFIG["louvain_resolution"]
METIS_NUM_COMS = CONFIG["metis_num_coms"]  # taken to be greater than NUM_CLIENTS
BASE_SEED = CONFIG.get("base_seed", 0) 

def main():
    # set a global seed
    set_seed(BASE_SEED)

    # derive different seeds for each splitting method
    louvain_seed = derive_seed(BASE_SEED, "louvain_split")
    metis_seed = derive_seed(BASE_SEED, "metis_split")

    # load synthetic graph and convert to PyG Data
    train_graphdata = torch.load("./data/train.pt", weights_only=False)
    global_data = graphdata_to_pyg(train_graphdata)

    # Louvain-based Label Imbalance Split
    louvain_clients = louvain_label_imbalance_split(
        global_data,
        num_clients=NUM_CLIENTS,
        resolution=LOUVAIN_RESOLUTION,
        seed=louvain_seed,
    )

    # Metis-based Label Imbalance Split
    metis_clients = metis_label_imbalance_split(
        global_data,
        num_clients=NUM_CLIENTS,
        metis_num_coms=METIS_NUM_COMS,
        seed=metis_seed,
    )

    # save federated splits for later training
    louvain_dir = "./data/fed_louvain"
    metis_dir = "./data/fed_metis"
    os.makedirs(louvain_dir, exist_ok=True)
    os.makedirs(metis_dir, exist_ok=True)

    for cid, data in enumerate(louvain_clients):
        torch.save(data, os.path.join(louvain_dir, f"client_{cid}.pt"))
    for cid, data in enumerate(metis_clients):
        torch.save(data, os.path.join(metis_dir, f"client_{cid}.pt"))

if __name__ == "__main__":
    main()
