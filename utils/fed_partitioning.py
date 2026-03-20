from typing import Optional
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

def equal_assign_communities_to_clients(
    communities: dict[int, list[int]],
    num_clients: int,
    seed: Optional[int] = None):
    """
    Assign communities to clients to get approximately equal-sized clients.

    Algorithm used for this purpose is greedy bin-packing: sort communities by size in desc order, 
    then iteratively assign each community to the client with the smallest current total size.
    """
    rng = np.random.default_rng(seed)

    # shuffle communities
    com_items = list(communities.items())
    rng.shuffle(com_items)
    com_items.sort(key=lambda x: len(x[1]), reverse=True)

    client_indices: dict[int, list[int]] = {cid: [] for cid in range(num_clients)}
    client_sizes = np.zeros(num_clients, dtype=int)

    for _, nodes in com_items:
        # assign to the client with the smallest current size
        cid = int(np.argmin(client_sizes))
        client_indices[cid].extend(nodes)
        client_sizes[cid] += len(nodes)

    return client_indices


def zipf_assign_communities_to_clients(communities: dict,
                                        num_clients: int,
                                        alpha: float = 1.2,
                                        seed: Optional[int] = None):
    """
    Assigns each community to a client using a Zipf-like skew:

        P(client = i) ∝ 1 / (i + 1)^alpha

    So client 0 is most likely to receive communities, client 1 a bit less,
    etc. This creates heterogeneous client sizes (big vs small clients).
    
    Additionally, ensure that every client receives at least one community
    to avoid empty client graphs.
    """
    rng = np.random.default_rng(seed)

    com_ids = list(communities.keys())
    rng.shuffle(com_ids)  # randomize community ids for extra variability

    # zipf-like probabilities over client ids
    ranks = np.arange(1, num_clients + 1, dtype=float)
    probs = 1.0 / (ranks ** alpha)
    probs = probs / probs.sum()

    client_indices = {cid: [] for cid in range(num_clients)}

    # Step 1: guarantee at least one community per client (if enough communities)
    num_coms = len(com_ids)
    first_pass = min(num_clients, num_coms)
    for cid in range(first_pass):
        com_id = com_ids[cid]
        client_indices[cid].extend(communities[com_id])

    # Step 2: assign remaining communities using Zipf-skewed sampling
    for idx in range(first_pass, num_coms):
        com_id = com_ids[idx]
        cid = int(rng.choice(num_clients, p=probs))
        client_indices[cid].extend(communities[com_id])

    return client_indices


def graphdata_to_pyg(data_g):
    """
    Convert the GraphData object into a torch_geometric.data.Data
    that is compatible with the OpenFGL-style partitioning code.
    """
    # Use existing node features or create dummy ones if not available
    if getattr(data_g, "x", None) is not None:
        x = data_g.x
    else:
        x = torch.zeros(data_g.num_nodes, 1)  # simple dummy feature

    y = data_g.y  # [N, num_tasks] multi-label

    data = Data(
        x=x,
        edge_index=data_g.edge_index,
        y=y,
    )

    # Set num_classes = number of label dimensions
    data.num_classes = y.size(1) 

    # Keep edge attributes to use later if needed
    if getattr(data_g, "edge_attr", None) is not None:
        data.edge_attr = data_g.edge_attr

    return data


def get_subgraph_pyg_data(global_data: Data, node_list):
    """
    Extract an induced subgraph from `global_data` on the given `node_list`.

    - Relabels node indices to be 0..(sub_num_nodes-1)
    - Slices x, y, edge_attr consistently
    """

    node_idx = torch.tensor(node_list, dtype=torch.long)

    # extract the subgraph and relabel nodes to be 0..(sub_num_nodes-1)
    sub_edge_index, sub_edge_attr = subgraph(
        node_idx,
        global_data.edge_index,
        edge_attr=getattr(global_data, "edge_attr", None),
        relabel_nodes=True,
        num_nodes=global_data.num_nodes,
    )

    # build new data object
    sub_data = Data()
    sub_data.edge_index = sub_edge_index

    if sub_edge_attr is not None:
        sub_data.edge_attr = sub_edge_attr

    # x / y sliced in the same order as node_idx
    if hasattr(global_data, "x") and global_data.x is not None:
        sub_data.x = global_data.x[node_idx]

    if hasattr(global_data, "y") and global_data.y is not None:
        sub_data.y = global_data.y[node_idx]

    # explicitly set num_nodes to be safe
    sub_data.num_nodes = node_idx.numel()

    return sub_data