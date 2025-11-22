import numpy as np
from sknetwork.clustering import Louvain
from sklearn.cluster import KMeans
from torch_geometric.utils import to_scipy_sparse_matrix, to_networkx
from torch_geometric.data import Data
import pymetis as metis

from utils.fed_partitioning import get_subgraph_pyg_data

def metis_label_imbalance_split(global_data: Data,
                                num_clients: int,
                                metis_num_coms: int):
    """
    Metis-based Label Imbalance Split for a single big graph
    with multi-task labels y in {0,1}^{N x T}.

    Args:
        global_data: PyG Data for the full graph.
        num_clients: number of federated clients.
        metis_num_coms: number of Metis communities (often > num_clients).

    Returns:
        list[Data]: one subgraph per client.
    """
    print("Conducting subgraph-FL metis+ (label-imbalance) simulation...")

    num_classes = global_data.num_classes

    # 1) Convert to NetworkX and partition with Metis
    graph_nx = to_networkx(global_data, to_undirected=True)
    n_cuts, membership = metis.part_graph(metis_num_coms, graph_nx)
    membership = np.array(membership)  # shape [num_nodes], value = com_id

    # 2) Build communities with label distributions (multi-task)
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

    num_communities = len(communities)

    # 3) Normalize and create clustering features
    clustering_data = np.zeros((num_communities, num_classes), dtype=float)
    for com_id in communities.keys():
        dist = communities[com_id]["label_distribution"]
        total = dist.sum()
        if total > 0:
            dist = dist / total
        clustering_data[com_id, :] = dist

    # 4) KMeans: communities -> clients
    kmeans = KMeans(n_clusters=num_clients, n_init="auto")
    clustering_labels = kmeans.fit_predict(clustering_data)

    client_indices = {cid: [] for cid in range(num_clients)}
    for com_id in range(num_communities):
        client_id = int(clustering_labels[com_id])
        client_indices[client_id] += communities[com_id]["nodes"]

    # 5) Build local subgraphs
    local_data = []
    for client_id in range(num_clients):
        node_list = sorted(client_indices[client_id])
        local_subgraph = get_subgraph_pyg_data(global_data, node_list)
        if local_subgraph.edge_index.numel() == 0:
            # optional: add random edges if you want connectivity
            pass
        local_data.append(local_subgraph)

    return local_data


def louvain_label_imbalance_split(global_data: Data,
                                  num_clients: int,
                                  resolution: float = 1.0):
    """
    Louvain-based Label Imbalance Split for a single big graph
    with multi-task labels y in {0,1}^{N x T}.

    Returns:
        list[Data]: one subgraph per client.
    """
    print("Conducting subgraph-FL louvain+ (label-imbalance) simulation...")

    num_nodes = global_data.num_nodes
    num_classes = global_data.num_classes  # == num_tasks

    # 1) Louvain communities on the adjacency
    adj_csr = to_scipy_sparse_matrix(global_data.edge_index, num_nodes=num_nodes)
    louvain = Louvain(modularity="newman",
                      resolution=resolution,
                      return_aggregate=True)
    com_assignments = louvain.fit_predict(adj_csr)  # community ID per node

    # 2) Build per-community label distributions (vectors of length num_classes)
    communities = {}
    for node_id, com_id in enumerate(com_assignments):
        com_id = int(com_id)
        if com_id not in communities:
            communities[com_id] = {
                "nodes": [],
                "label_distribution": np.zeros(num_classes, dtype=float),
            }
        communities[com_id]["nodes"].append(node_id)
        # accumulate the label vector (multi-task)
        label_vec = global_data.y[node_id].cpu().numpy()  # shape [num_classes]
        communities[com_id]["label_distribution"] += label_vec

    num_communities = len(communities)

    # 3) Normalize label distributions and create clustering features
    clustering_data = np.zeros((num_communities, num_classes), dtype=float)
    for com_id in communities.keys():
        dist = communities[com_id]["label_distribution"]
        total = dist.sum()
        if total > 0:
            dist = dist / total
        clustering_data[com_id, :] = dist

    # 4) KMeans over communities by label distribution
    kmeans = KMeans(n_clusters=num_clients, n_init="auto")
    clustering_labels = kmeans.fit_predict(clustering_data)  # community -> client

    # 5) Aggregate communities into clients
    client_indices = {cid: [] for cid in range(num_clients)}
    for com_id in range(num_communities):
        client_id = int(clustering_labels[com_id])
        client_indices[client_id] += communities[com_id]["nodes"]

    # 6) Build local subgraphs
    local_data = []
    for client_id in range(num_clients):
        node_list = sorted(client_indices[client_id])
        local_subgraph = get_subgraph_pyg_data(global_data, node_list)
        if local_subgraph.edge_index.numel() == 0:
            # if no edges, you can optionally add random edges or leave as empty
            pass
        local_data.append(local_subgraph)

    return local_data

