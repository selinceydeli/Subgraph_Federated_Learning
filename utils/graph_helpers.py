# utils/graph_helpers.py
import os
from typing import Any, Optional

import torch
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import remove_self_loops


def max_port_cols(d):
    """
    Returns (max_in_port_id, max_out_port_id) for a homogeneous Data object
    whose edge_attr last two columns correspond to (in_port_id, out_port_id).
    """
    in_col, out_col = d.edge_attr.size(-1) - 2, d.edge_attr.size(-1) - 1
    return int(d.edge_attr[:, in_col].max().item()), int(d.edge_attr[:, out_col].max().item())


def check_and_strip_self_loops(data, name: str = ""):
    """
    Removes self-loops from a PyG Data object (if any) and log a message.
    """
    ei = data.edge_index
    has_loops = bool((ei[0] == ei[1]).any())
    if has_loops:
        ei_clean, ea_clean = remove_self_loops(ei, getattr(data, "edge_attr", None))
        data.edge_index = ei_clean
        if hasattr(data, "edge_attr"):
            data.edge_attr = ea_clean
        print(f"[{name}] removed self-loops → E={data.edge_index.size(1)}")
    return data


def build_hetero_neighbor_loader(
    hetero_data,
    batch_size: int,
    num_layers: int,
    fanout: Any,
    device: Optional[torch.device] = None,
    shuffle: bool = True,
):
    """
    NeighborLoader for mini-batch training/eval on a hetero graph.

    fanout can be int or list[int].
    """
    if isinstance(fanout, int):
        fanout_list = [fanout] * num_layers
    else:
        fanout_list = list(fanout)

    num_neighbors = {
        ("n", "fwd", "n"): fanout_list,
        ("n", "rev", "n"): fanout_list,
    }

    use_cuda = (device is not None and device.type == "cuda")
    num_workers = max(1, os.cpu_count() // 2)

    return NeighborLoader(
        hetero_data,
        num_neighbors=num_neighbors,
        input_nodes=("n", torch.arange(hetero_data["n"].num_nodes)),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=use_cuda,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=2,
        filter_per_worker=True,
    )


def build_full_eval_loader(
    hetero_data,
    batch_size: int,
    num_layers: int,
    device: Optional[torch.device] = None,
    shuffle: bool = False,
):
    """
    Covers all nodes as seeds and expands with all neighbors up to num_layers.
    Uses -1 neighbors, i.e., full k-hop neighborhoods.
    """
    fanout_all = [-1] * num_layers
    num_neighbors = {
        ("n", "fwd", "n"): fanout_all,
        ("n", "rev", "n"): fanout_all,
    }

    use_cuda = (device is not None and device.type == "cuda")
    num_workers = max(1, os.cpu_count() // 2)

    return NeighborLoader(
        hetero_data,
        num_neighbors=num_neighbors,
        input_nodes=("n", torch.arange(hetero_data["n"].num_nodes)),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=use_cuda,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=2,
        filter_per_worker=True,
    )
