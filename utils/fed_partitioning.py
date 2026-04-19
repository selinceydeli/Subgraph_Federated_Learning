import csv
import os
from typing import List, Optional

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

from utils.gcn_utils import GraphData

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


def save_community_clients(
    split_dir: str,
    global_data: Data,
    node_splits: List[List[int]],
    include_cross_edges: bool = False,
):
    """
    Persist Louvain/Metis community-style client subgraphs using the same
    on-disk layout as the partition-aware splits:

        {split_dir}/
            clients/client_{cid:04d}.pt
            node_to_client.pt
            client_sizes.csv

    If include_cross_edges=False, each client stores a strict induced
    subgraph on its owned nodes; owned_mask is all-ones and
    num_cross_edges = 0.

    If include_cross_edges=True, each client additionally keeps edges
    with exactly one owned endpoint and pulls the foreign endpoint in
    as a "ghost" node. owned_mask identifies owned vs ghost nodes so
    downstream loss/metrics can restrict to owned nodes only.
    """
    os.makedirs(split_dir, exist_ok=True)
    clients_out = os.path.join(split_dir, "clients")
    os.makedirs(clients_out, exist_ok=True)

    num_clients = len(node_splits)
    num_global_nodes = int(global_data.num_nodes)

    node_to_client = torch.full((num_global_nodes,), -1, dtype=torch.long)
    for cid, node_list in enumerate(node_splits):
        if len(node_list) == 0:
            continue
        idx = torch.tensor(node_list, dtype=torch.long)
        node_to_client[idx] = cid
    torch.save(node_to_client, os.path.join(split_dir, "node_to_client.pt"))

    src = global_data.edge_index[0]
    dst = global_data.edge_index[1]
    edge_attr = getattr(global_data, "edge_attr", None)

    # Port ID columns (last 2 of edge_attr) depend on per-node adjacency
    # and must be recomputed after subgraphing to stay consistent with
    # each client's local edges. Strip them here and re-add via GraphData.add_ports().
    if edge_attr is not None and edge_attr.size(1) >= 2:
        base_edge_attr = edge_attr[:, :-2].contiguous()
    else:
        base_edge_attr = edge_attr

    rows = []
    for cid in range(num_clients):
        owned_global = (node_to_client == cid)  # [N] bool over *global* node ids

        if include_cross_edges:
            edge_keep = owned_global[src] | owned_global[dst]
        else:
            edge_keep = owned_global[src] & owned_global[dst]

        kept_src = src[edge_keep]
        kept_dst = dst[edge_keep]

        if include_cross_edges and int(edge_keep.sum().item()) > 0:
            # owned nodes + foreign endpoints appearing as ghosts
            nodes_keep = torch.unique(torch.cat([kept_src, kept_dst], dim=0))
        else:
            # strictly owned nodes only
            nodes_keep = torch.where(owned_global)[0]

        sub_edge_index, sub_edge_attr = subgraph(
            subset=nodes_keep,
            edge_index=global_data.edge_index[:, edge_keep],
            edge_attr=base_edge_attr[edge_keep] if base_edge_attr is not None else None,
            relabel_nodes=True,
            num_nodes=num_global_nodes,
        )

        owned_mask_local = owned_global[nodes_keep].clone()  # [n_client_nodes] bool

        gd = GraphData(
            x=global_data.x[nodes_keep].clone() if global_data.x is not None else None,
            y=global_data.y[nodes_keep].clone() if global_data.y is not None else None,
            edge_index=sub_edge_index,
            edge_attr=sub_edge_attr.clone() if sub_edge_attr is not None else None,
        )

        # Recompute ports against the client's local adjacency.
        if gd.edge_attr is not None and gd.edge_attr.size(1) >= 1:
            gd = gd.add_ports()

        gd.owned_mask = owned_mask_local
        gd.global_nid = nodes_keep.clone()
        gd.client_id = int(cid)
        gd.include_cross_edges = bool(include_cross_edges)

        torch.save(gd, os.path.join(clients_out, f"client_{cid:04d}.pt"))

        # Cross-edge = exactly one owned endpoint (in client-local indexing)
        if gd.edge_index.numel() > 0:
            cs, cd = gd.edge_index[0], gd.edge_index[1]
            cross_edges = int((gd.owned_mask[cs] ^ gd.owned_mask[cd]).sum().item())
        else:
            cross_edges = 0

        num_total = int(gd.num_nodes)
        num_owned = int(owned_mask_local.sum().item())
        num_ghost = num_total - num_owned
        num_edges = int(gd.edge_index.size(1))
        rows.append((cid, num_total, num_owned, num_ghost, num_edges, cross_edges))

    with open(os.path.join(split_dir, "client_sizes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["client_id", "num_nodes_total", "num_owned", "num_ghost", "num_edges", "num_cross_edges"])
        w.writerows(rows)