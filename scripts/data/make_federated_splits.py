import os
import json
import torch

from utils.seed import set_seed, derive_seed
from utils.fed_partitioning import graphdata_to_pyg, save_community_clients
from utils.fed_simulation import louvain_original_split, metis_original_split

"""
Generates the canonical Louvain/Metis subgraph-FL splits with approximately
equal-sized clients (the version commonly reported in federated GNN
benchmarks). Runs once per client count in `num_clients_list` and emits
both with/without cross-edge variants.

Each of train.pt / val.pt / test.pt is partitioned INDEPENDENTLY (same
protocol as the partition-aware witness splits), yielding the layout:

    ./data/fed_{strategy}_splits_{with,without}_cross_edges/
           {num_clients}_clients/
               {train,val,test}/
                   clients/client_XXXX.pt
                   node_to_client.pt
                   client_sizes.csv

Other assignment variants (zipf_skewed, label-imbalance) remain available
in utils/fed_simulation.py but are not generated here.
"""

CONFIG_PATH = "./configs/fed_configs.json"

with open(CONFIG_PATH, "r") as f:
    ALL_CONFIG = json.load(f)

CONFIG = ALL_CONFIG["louvain_and_metis_splits"]

NUM_CLIENTS_LIST = CONFIG["num_clients_list"]
LOUVAIN_RESOLUTION = CONFIG["louvain_resolution"]
METIS_NUM_COMS = CONFIG["metis_num_coms"]  # >= max(NUM_CLIENTS_LIST)
BASE_SEED = CONFIG.get("base_seed", 0)

GRAPH_SPLITS = ("train", "val", "test")


def _split_dir(strategy: str, num_clients: int, include_cross_edges: bool, graph_split: str) -> str:
    """
    Mirror the partition-aware layout:
        ./data/fed_{strategy}_splits_{with,without}_cross_edges/
               {num_clients}_clients/{train,val,test}
    """
    cross_suffix = "with_cross_edges" if include_cross_edges else "without_cross_edges"
    return os.path.join(
        "./data",
        f"fed_{strategy}_splits_{cross_suffix}",
        f"{num_clients}_clients",
        graph_split,
    )


def _partition_one(global_data, num_clients: int, strategy: str, seed: int):
    if strategy == "louvain":
        return louvain_original_split(
            global_data,
            num_clients=num_clients,
            resolution=LOUVAIN_RESOLUTION,
            seed=seed,
            client_assignment="equal",
            return_node_indices=True,
        )
    if strategy == "metis":
        return metis_original_split(
            global_data,
            num_clients=num_clients,
            metis_num_coms=METIS_NUM_COMS,
            seed=seed,
            client_assignment="equal",
            return_node_indices=True,
        )
    raise ValueError(f"Unknown strategy: {strategy}")


def _generate_for_num_clients(graphs_by_split: dict, num_clients: int, seed_log: dict):
    """
    Produce Louvain and Metis equal-sized-client splits for a single
    num_clients value, across all three graph splits (train/val/test),
    and save both cross-edge variants. Each (strategy, num_clients,
    graph_split) triple gets its own derived seed.
    """
    for strategy in ("louvain", "metis"):
        for graph_split, global_data in graphs_by_split.items():
            tag = f"{strategy}_equal_{num_clients}c_{graph_split}"
            seed = derive_seed(BASE_SEED, tag)
            seed_log[tag] = seed

            node_splits = _partition_one(global_data, num_clients, strategy, seed)

            for include_cross_edges in (True, False):
                out_dir = _split_dir(strategy, num_clients, include_cross_edges, graph_split)
                save_community_clients(
                    out_dir, global_data, node_splits,
                    include_cross_edges=include_cross_edges,
                )
                cross_tag = "with" if include_cross_edges else "without"
                print(f"  {strategy}/{num_clients}c/{graph_split} ({cross_tag} cross edges) -> {out_dir}")


def main():
    set_seed(BASE_SEED)

    graphs_by_split = {}
    for graph_split in GRAPH_SPLITS:
        graphdata = torch.load(f"./data/{graph_split}.pt", weights_only=False)
        graphs_by_split[graph_split] = graphdata_to_pyg(graphdata)

    seed_log: dict[str, int] = {}

    print("Saving client subgraphs (partition-aware-aligned layout, train/val/test)...")
    for num_clients in NUM_CLIENTS_LIST:
        _generate_for_num_clients(graphs_by_split, num_clients, seed_log)

    out_dir = "./data/seeds"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "fed_seeds.txt"), "w") as f:
        for k, v in seed_log.items():
            f.write(f"{k}:{v}\n")

    print("Done. Federated splits successfully generated.")


if __name__ == "__main__":
    main()
