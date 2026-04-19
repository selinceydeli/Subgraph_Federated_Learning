from typing import Optional
import numpy as np
import pymetis as metis
from sknetwork.clustering import Louvain
from sklearn.cluster import KMeans
from torch_geometric.utils import to_scipy_sparse_matrix, to_networkx
from torch_geometric.data import Data

from utils.fed_partitioning import (
    get_subgraph_pyg_data,
    zipf_assign_communities_to_clients,
    equal_assign_communities_to_clients,
)

"""
The implementation of the Label Imbalance Split (LIS) simulation strategy
for the Metis-based and Louvain-based data partitioning techniques is adapted
from the GitHub repository of the "OpenFGL: A Comprehensive Benchmark for
Federated Graph Learning" paper by Li et al., (2024).

In the original OpenFGL implementation, at the label distribution clustering step,
both partitioning techniques assume that each node has a single class label.
However, in the global graph generated for this work, each node has 11 binary labels.
To address this, the clustering feature per community is the per-label prevalence
(mean of each binary task across the community's nodes), which treats tasks
independently instead of sum-normalizing across labels.
"""


def _metis_options(seed: Optional[int]):
    """Build a pymetis.Options with a deterministic seed (if provided)."""
    opts = metis.Options()
    opts.set_defaults()
    if seed is not None:
        # METIS accepts any non-negative int; keep it in int32 range.
        opts.seed = int(seed) % (2**31 - 1)
    return opts


def _reassign_to_empty_clients(client_indices: dict, num_clients: int):
    """
    Ensure every client has at least one node by donating nodes from the
    largest clients to any empty ones. Keeps the outer contract (every
    cid in range(num_clients) maps to a non-empty node list) while
    preserving the overall partitioning as much as possible.
    """
    empty = [cid for cid in range(num_clients) if len(client_indices.get(cid, [])) == 0]
    if not empty:
        return client_indices

    # Donor order: clients with the most nodes first.
    donors = sorted(
        (cid for cid in range(num_clients) if len(client_indices.get(cid, [])) > 1),
        key=lambda cid: len(client_indices[cid]),
        reverse=True,
    )

    for empty_cid in empty:
        if not donors:
            raise ValueError(
                f"Cannot fill empty client {empty_cid}: no donor with >1 node. "
                f"num_clients={num_clients} is too large for the number of nodes available."
            )
        donor_cid = donors[0]
        # Move one node; pick the last so the donor's prefix stays stable.
        node = client_indices[donor_cid].pop()
        client_indices[empty_cid].append(node)
        # Resort donors if this donor dropped below the next one.
        donors.sort(key=lambda cid: len(client_indices[cid]), reverse=True)
        # Drop donors that can no longer give.
        donors = [cid for cid in donors if len(client_indices[cid]) > 1]

    return client_indices


def _per_label_prevalence_features(communities: dict, num_classes: int) -> np.ndarray:
    """
    Per-community KMeans feature = mean of each binary task over the
    community's nodes (so tasks are treated independently).
    communities[com_id] must contain "nodes" (list) and "label_distribution"
    (summed label vector of length num_classes).
    """
    com_ids = sorted(communities.keys())
    feats = np.zeros((len(com_ids), num_classes), dtype=float)
    for row, com_id in enumerate(com_ids):
        n_nodes = len(communities[com_id]["nodes"])
        if n_nodes == 0:
            continue
        feats[row, :] = communities[com_id]["label_distribution"] / float(n_nodes)
    return feats


def metis_original_split(global_data: Data,
                         num_clients: int,
                         metis_num_coms: int,
                         seed: Optional[int] = None,
                         alpha: float = 1.2,
                         client_assignment: str = "zipf",
                         return_node_indices: bool = False):
    """
    Metis-based original subgraph-FL split (structure-only, no label handling).

    Supports two client-assignment strategies:
      - 'zipf'   : Zipf-skewed client sizes (realistic, heavy-tailed)
      - 'equal'  : Approximately equal-sized clients (controlled setting)
    """
    print(f"Conducting subgraph-FL Metis (original, {client_assignment}-assigned) simulation...")

    # convert to NetworkX and partition with Metis (seeded)
    graph_nx = to_networkx(global_data, to_undirected=True)
    _, membership = metis.part_graph(metis_num_coms, graph_nx, options=_metis_options(seed))
    membership = np.array(membership)  # shape [num_nodes], value = com_id

    # build structural communities
    communities: dict[int, list[int]] = {com_id: [] for com_id in range(metis_num_coms)}
    for node_id, com_id in enumerate(membership):
        com_id = int(com_id)
        communities[com_id].append(node_id)

    # assign communities to clients according to the chosen strategy
    if client_assignment == "zipf":
        client_indices = zipf_assign_communities_to_clients(
            communities=communities,
            num_clients=num_clients,
            alpha=alpha,
            seed=seed,
        )
    elif client_assignment == "equal":
        # alpha is ignored in the equal-sized case
        client_indices = equal_assign_communities_to_clients(
            communities=communities,
            num_clients=num_clients,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown client_assignment='{client_assignment}'. Expected 'zipf' or 'equal'.")

    client_indices = _reassign_to_empty_clients(client_indices, num_clients)

    # if the user only wants the node indices, return them without computing local subgraphs
    if return_node_indices:
        return [sorted(client_indices[cid]) for cid in range(num_clients)]

    # else, build local subgraphs
    local_data = []
    for client_id in range(num_clients):
        node_list = sorted(client_indices[client_id])
        local_subgraph = get_subgraph_pyg_data(global_data, node_list)
        local_data.append(local_subgraph)

    return local_data


def louvain_original_split(global_data: Data,
                           num_clients: int,
                           resolution: float = 1.0,
                           seed: Optional[int] = None,
                           alpha: float = 1.2,
                           client_assignment: str = "zipf",
                           return_node_indices: bool = False):
    """
    Louvain-based original subgraph-FL split (structure-only, no label handling).

    Supports two client-assignment strategies:
      - 'zipf'   : Zipf-skewed client sizes (realistic, heavy-tailed)
      - 'equal'  : Approximately equal-sized clients (controlled setting)
    """
    print(f"Conducting subgraph-FL Louvain (original, {client_assignment}-assigned) simulation...")

    num_nodes = global_data.num_nodes

    # Louvain communities on the adjacency
    adj_csr = to_scipy_sparse_matrix(global_data.edge_index, num_nodes=num_nodes)
    louvain = Louvain(
        modularity="newman",
        resolution=resolution,
        return_aggregate=True,
        random_state=seed,
    )
    com_assignments = louvain.fit_predict(adj_csr)  # community ID per node

    # build structural communities (no labels involved)
    communities: dict[int, list[int]] = {}
    for node_id, com_id in enumerate(com_assignments):
        com_id = int(com_id)
        if com_id not in communities:
            communities[com_id] = []
        communities[com_id].append(node_id)

    # assign communities to clients according to the chosen strategy
    if client_assignment == "zipf":
        client_indices = zipf_assign_communities_to_clients(
            communities=communities,
            num_clients=num_clients,
            alpha=alpha,
            seed=seed,
        )
    elif client_assignment == "equal":
        # alpha is ignored in the equal-sized case
        client_indices = equal_assign_communities_to_clients(
            communities=communities,
            num_clients=num_clients,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown client_assignment='{client_assignment}'. Expected 'zipf' or 'equal'.")

    client_indices = _reassign_to_empty_clients(client_indices, num_clients)

    # if the user only wants the node indices, return them without computing local subgraphs
    if return_node_indices:
        return [sorted(client_indices[cid]) for cid in range(num_clients)]

    # else, build local subgraphs
    local_data = []
    for client_id in range(num_clients):
        node_list = sorted(client_indices[client_id])
        local_subgraph = get_subgraph_pyg_data(global_data, node_list)
        local_data.append(local_subgraph)

    return local_data


def metis_label_imbalance_split(global_data: Data,
                                num_clients: int,
                                metis_num_coms: int,
                                seed: Optional[int] = None,
                                return_node_indices: bool = False):
    """
    Metis-based Label Imbalance Split for a single big graph
    with multi-task labels y in {0,1}^{N x T}.

    Returns:
        If return_node_indices is False (default):
            list[Data]: one subgraph per client.
        If return_node_indices is True:
            list[list[int]]: node indices per client.
    """
    print("Conducting subgraph-FL metis+ (label-imbalance) simulation...")

    num_classes = global_data.num_classes

    # seeded Metis partition
    graph_nx = to_networkx(global_data, to_undirected=True)
    _, membership = metis.part_graph(metis_num_coms, graph_nx, options=_metis_options(seed))
    membership = np.array(membership)

    # build communities with label distributions (multi-task)
    communities = {
        com_id: {
            "nodes": [],
            "label_distribution": np.zeros(num_classes, dtype=float),
        }
        for com_id in range(metis_num_coms)
    }

    for node_id, com_id in enumerate(membership):
        com_id = int(com_id)
        communities[com_id]["nodes"].append(node_id)
        label_vec = global_data.y[node_id].cpu().numpy()
        communities[com_id]["label_distribution"] += label_vec

    # Per-label prevalence (mean over nodes) — treats the 11 binary tasks independently.
    clustering_data = _per_label_prevalence_features(communities, num_classes)

    # drop empty communities so KMeans does not cluster zero rows
    com_ids_sorted = sorted(communities.keys())
    nonempty_rows = [i for i, cid in enumerate(com_ids_sorted) if len(communities[cid]["nodes"]) > 0]
    if len(nonempty_rows) < num_clients:
        raise ValueError(
            f"num_clients={num_clients} exceeds the number of non-empty Metis "
            f"communities ({len(nonempty_rows)}); lower num_clients or raise metis_num_coms."
        )
    clustering_data = clustering_data[nonempty_rows]
    nonempty_com_ids = [com_ids_sorted[i] for i in nonempty_rows]

    kmeans = KMeans(n_clusters=num_clients, n_init="auto", random_state=seed)
    clustering_labels = kmeans.fit_predict(clustering_data)

    client_indices = {cid: [] for cid in range(num_clients)}
    for local_idx, com_id in enumerate(nonempty_com_ids):
        client_id = int(clustering_labels[local_idx])
        client_indices[client_id] += communities[com_id]["nodes"]

    client_indices = _reassign_to_empty_clients(client_indices, num_clients)

    if return_node_indices:
        return [sorted(client_indices[cid]) for cid in range(num_clients)]

    local_data = []
    for client_id in range(num_clients):
        node_list = sorted(client_indices[client_id])
        local_subgraph = get_subgraph_pyg_data(global_data, node_list)
        local_data.append(local_subgraph)

    return local_data


def louvain_label_imbalance_split(global_data: Data,
                                  num_clients: int,
                                  resolution: float = 1.0,
                                  seed: Optional[int] = None,
                                  return_node_indices: bool = False):
    """
    Louvain-based Label Imbalance Split for a single big graph
    with multi-task labels y in {0,1}^{N x T}.

    Returns:
        If return_node_indices is False (default):
            list[Data]: one subgraph per client.
        If return_node_indices is True:
            list[list[int]]: node indices per client.
    """
    print("Conducting subgraph-FL louvain+ (label-imbalance) simulation...")

    num_nodes = global_data.num_nodes
    num_classes = global_data.num_classes  # == num_tasks

    # louvain communities on the adjacency
    adj_csr = to_scipy_sparse_matrix(global_data.edge_index, num_nodes=num_nodes)
    louvain = Louvain(modularity="newman",
                      resolution=resolution,
                      return_aggregate=True,
                      random_state=seed)
    com_assignments = louvain.fit_predict(adj_csr)  # community ID per node

    # build per-community label sums (multi-task)
    communities = {}
    for node_id, com_id in enumerate(com_assignments):
        com_id = int(com_id)
        if com_id not in communities:
            communities[com_id] = {
                "nodes": [],
                "label_distribution": np.zeros(num_classes, dtype=float),
            }
        communities[com_id]["nodes"].append(node_id)
        label_vec = global_data.y[node_id].cpu().numpy()
        communities[com_id]["label_distribution"] += label_vec

    # Per-label prevalence features — see comment in metis_label_imbalance_split.
    clustering_data = _per_label_prevalence_features(communities, num_classes)

    com_ids_sorted = sorted(communities.keys())
    nonempty_rows = [i for i, cid in enumerate(com_ids_sorted) if len(communities[cid]["nodes"]) > 0]
    if len(nonempty_rows) < num_clients:
        raise ValueError(
            f"num_clients={num_clients} exceeds the number of non-empty Louvain "
            f"communities ({len(nonempty_rows)}); lower num_clients or lower resolution."
        )
    clustering_data = clustering_data[nonempty_rows]
    nonempty_com_ids = [com_ids_sorted[i] for i in nonempty_rows]

    kmeans = KMeans(n_clusters=num_clients, n_init="auto", random_state=seed)
    clustering_labels = kmeans.fit_predict(clustering_data)

    client_indices = {cid: [] for cid in range(num_clients)}
    for local_idx, com_id in enumerate(nonempty_com_ids):
        client_id = int(clustering_labels[local_idx])
        client_indices[client_id] += communities[com_id]["nodes"]

    client_indices = _reassign_to_empty_clients(client_indices, num_clients)

    if return_node_indices:
        return [sorted(client_indices[cid]) for cid in range(num_clients)]

    local_data = []
    for client_id in range(num_clients):
        node_list = sorted(client_indices[client_id])
        local_subgraph = get_subgraph_pyg_data(global_data, node_list)
        local_data.append(local_subgraph)

    return local_data
