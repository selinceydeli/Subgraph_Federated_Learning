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


def _split_dir(strategy: str, num_clients: int, include_cross_edges: bool) -> str:
    """
    Mirror the partition-aware layout:
        ./data/fed_{strategy}_splits_{with,without}_cross_edges/
               {num_clients}_clients/
    """
    cross_suffix = "with_cross_edges" if include_cross_edges else "without_cross_edges"
    return os.path.join(
        "./data",
        f"fed_{strategy}_splits_{cross_suffix}",
        f"{num_clients}_clients",
    )


def _generate_for_num_clients(global_data, num_clients: int, seed_log: dict):
    """
    Produce the Louvain and Metis equal-sized-client splits for a single
    num_clients value, saving both cross-edge variants. Each (strategy,
    num_clients) pair gets its own derived seed so runs remain
    independently reproducible.
    """
    louvain_seed = derive_seed(BASE_SEED, f"louvain_equal_{num_clients}c")
    metis_seed = derive_seed(BASE_SEED, f"metis_equal_{num_clients}c")
    seed_log[f"louvain_equal_{num_clients}c"] = louvain_seed
    seed_log[f"metis_equal_{num_clients}c"] = metis_seed

    louvain_node_splits = louvain_original_split(
        global_data,
        num_clients=num_clients,
        resolution=LOUVAIN_RESOLUTION,
        seed=louvain_seed,
        client_assignment="equal",
        return_node_indices=True,
    )
    metis_node_splits = metis_original_split(
        global_data,
        num_clients=num_clients,
        metis_num_coms=METIS_NUM_COMS,
        seed=metis_seed,
        client_assignment="equal",
        return_node_indices=True,
    )

    strategies = [
        ("louvain", louvain_node_splits),
        ("metis", metis_node_splits),
    ]
    for strategy, node_splits in strategies:
        for include_cross_edges in (True, False):
            out_dir = _split_dir(strategy, num_clients, include_cross_edges)
            save_community_clients(
                out_dir, global_data, node_splits,
                include_cross_edges=include_cross_edges,
            )
            cross_tag = "with" if include_cross_edges else "without"
            print(f"  {strategy}/{num_clients}c ({cross_tag} cross edges) -> {out_dir}")


def main():
    set_seed(BASE_SEED)

    train_graphdata = torch.load("./data/train.pt", weights_only=False)
    global_data = graphdata_to_pyg(train_graphdata)

    seed_log: dict[str, int] = {}

    print("Saving client subgraphs (partition-aware-aligned layout)...")
    for num_clients in NUM_CLIENTS_LIST:
        _generate_for_num_clients(global_data, num_clients, seed_log)

    out_dir = "./data/seeds"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "fed_seeds.txt"), "w") as f:
        for k, v in seed_log.items():
            f.write(f"{k}:{v}\n")

    print("Done. Federated splits successfully generated.")


if __name__ == "__main__":
    main()
