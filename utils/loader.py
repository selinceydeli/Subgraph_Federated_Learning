# utils/loader.py
import os
import torch


def load_client_graphs(clients_dir, num_clients):
    out = []
    for cid in range(num_clients):
        p1 = os.path.join(clients_dir, f"client_{cid:04d}.pt")
        p2 = os.path.join(clients_dir, f"client_{cid}.pt")
        path = p1 if os.path.exists(p1) else p2
        out.append(torch.load(path, weights_only=False))
    return out


_STRATEGY_PREFIX = {
    "partition_aware": "fed_partition_aware_splits",
    "louvain": "fed_louvain_splits",
    "metis": "fed_metis_splits",
}


def resolve_data_dirs(partition_cfg: dict):
    """
    Resolve (train_clients_dir, val_clients_dir, test_clients_dir) for a
    given fed_splits block. Supports partition_strategy in
    {partition_aware, louvain, metis}. `use_local_labels` is only
    meaningful for partition_aware; it is ignored for community strategies.
    """
    strategy = partition_cfg.get("partition_strategy", "partition_aware")
    if strategy not in _STRATEGY_PREFIX:
        raise ValueError(
            f"Unknown partition_strategy='{strategy}'. "
            f"Expected one of {list(_STRATEGY_PREFIX)}."
        )

    num_clients = partition_cfg["num_clients"]
    cross_suffix = "with_cross_edges" if partition_cfg["include_cross_edges"] else "without_cross_edges"

    local_suffix = ""
    if strategy == "partition_aware" and partition_cfg.get("use_local_labels", False):
        local_suffix = "_local_labels"

    base = f"./data/{_STRATEGY_PREFIX[strategy]}_{cross_suffix}{local_suffix}/{num_clients}_clients"
    return (
        f"{base}/train/clients",
        f"{base}/val/clients",
        f"{base}/test/clients",
    )
