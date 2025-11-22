# make_federated_splits.py
import os
import torch
from utils.federated_partitioning import graphdata_to_pyg
from utils.federated_simulation import louvain_label_imbalance_split, metis_label_imbalance_split

NUM_CLIENTS = 10
LOUVAIN_RESOLUTION = 1.0
METIS_NUM_COMS = 40  # taken to be greater than NUM_CLIENTS

def main():
    # load synthetic graph and convert to PyG Data
    train_graphdata = torch.load("./data/train.pt", weights_only=False)
    global_data = graphdata_to_pyg(train_graphdata)

    # Louvain-based Label Imbalance Split
    louvain_clients = louvain_label_imbalance_split(
        global_data,
        num_clients=NUM_CLIENTS,
        resolution=LOUVAIN_RESOLUTION,
    )

    # Metis-based Label Imbalance Split
    metis_clients = metis_label_imbalance_split(
        global_data,
        num_clients=NUM_CLIENTS,
        metis_num_coms=METIS_NUM_COMS,
    )

    # save federated splits for later training
    louvain_dir = "./data/fed_louvain"
    metis_dir = "./data/fed_metis"
    os.makedirs(louvain_dir, exist_ok=True)
    os.makedirs(metis_dir, exist_ok=True)

    for cid, data in enumerate(louvain_clients):
        torch.save(data, f"./data/fed_louvain/client_{cid}.pt")
    for cid, data in enumerate(metis_clients):
        torch.save(data, f"./data/fed_metis/client_{cid}.pt")

if __name__ == "__main__":
    main()
