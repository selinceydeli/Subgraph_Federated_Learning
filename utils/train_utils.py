#!/usr/bin/env python3
import os
from typing import Optional
import torch
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import HeteroData
import copy

from utils.metrics import compute_minority_f1_score_per_task

DATA_PATH = "./data"

def load_datasets(log_dir=DATA_PATH, train_data_file="train.pt", val_data_file="val.pt", test_data_file="test.pt"):
    train = torch.load(os.path.join(log_dir, train_data_file), weights_only=False, map_location="cpu")
    val= torch.load(os.path.join(log_dir, val_data_file), weights_only=False, map_location="cpu")
    test = torch.load(os.path.join(log_dir, test_data_file), weights_only=False, map_location="cpu")
    return train, val, test


def ensure_node_features(g):
    '''
    Ensure that the graph has node features.
    The Provably Powerful GNNs paper uses constant node features in its baseline models.
    As a baseline approach, assign ones to each node.
    '''
    if getattr(g, 'x', None) is None:
        N = g.y.shape[0] if getattr(g, 'y', None) is not None else int(g.edge_index.max()) + 1
        # Assign constant features (all ones) to each node
        g.x = torch.ones((N, 1), dtype=torch.float)
    return g


def make_reverse_neighbor_loader(data, num_neighbors=[15, 10, 5], batch_size=2048, shuffle=False, input_nodes=None):
    """
    Neighbor sampling expands neighborhoods using the given edge_index
    by duplicating the data and flipping its edge_index so the sampling is in backward direction.
    """
    data_rev = copy.copy(data)  
    data_rev.edge_index = data.edge_index[[1, 0], :]  # Reversed edge indices for sampling only
    loader = NeighborLoader(
        data_rev,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=shuffle,
        input_nodes=input_nodes  
    )
    # Keep originals for later reconstruction of forward edges per batch:
    loader._full_edge_index_fwd = data.edge_index
    loader._num_nodes_full = data.num_nodes
    return loader


def _unpack_io(batch):
    """
    Helper method to unpack the batch into x_in, edge_in, y_true
    by differentiating between homogeneous and heterogeneous graphs

    Returns a tuple of (x_in, edge_in, y_true, num_nodes, is_hetero)
      - if homogeneous: edge_in is a Tensor edge_index
      - if hetero: edge_in is a dict {edge_type: edge_index}
    """
    is_hetero = isinstance(batch, HeteroData)
    if is_hetero:
        x_in = {'n': batch['n'].x}  # x_in should be a dict for hetero
        y_true = batch['n'].y
        edge_in = {
            ('n','fwd','n'): batch[('n','fwd','n')].edge_index,
            ('n','rev','n'): batch[('n','rev','n')].edge_index,
        }
        num_nodes = int(getattr(batch['n'], 'num_nodes', 0) or batch['n'].x.size(0))
    else:
        x_in = batch.x
        y_true = batch.y
        edge_in = batch.edge_index
        num_nodes = int(batch.num_nodes)
    return x_in, edge_in, y_true, num_nodes, is_hetero


def _augment_with_ego_and_get_seed_slice(x_in, y_true, batch, is_hetero, model):
    """
    - If the model has `ego_dim > 0`, concatenate an Ego one-hot of width `ego_dim`
      to node features.
    - Detect number of seeds B (first B nodes) if available and return the seed slice.
    - Returns: x_in_aug (same type as x_in), y_used (sliced to seeds if B known), B (int or None)
    """
    # Detect seed count B (NeighborLoader puts seeds first)
    B = None
    if is_hetero and hasattr(batch['n'], 'batch_size'):
        B = int(batch['n'].batch_size)
    elif not is_hetero and hasattr(batch, 'batch_size'):
        B = int(batch.batch_size)

    # If no Ego support on the model, just seed-slice labels and return
    ego_dim = int(getattr(model, 'ego_dim', 0))
    if ego_dim <= 0:
        y_used = y_true[:B] if B is not None else y_true
        return x_in, y_used, B

    # Build Ego one-hots and concatenate
    if is_hetero:
        # x_in is a dict with key 'n'
        x = x_in['n']
        device = x.device
        ego = torch.zeros((x.size(0), ego_dim), device=device)
        if B is not None and B > 0:
            k = min(B, ego_dim)
            idx = torch.arange(k, device=device)
            ego[idx, idx] = 1.0
        x_aug = torch.cat([x, ego], dim=-1)
        x_in_aug = {'n': x_aug}
    else:
        # homogeneous tensor
        x = x_in
        device = x.device
        ego = torch.zeros((x.size(0), ego_dim), device=device)
        if B is not None and B > 0:
            k = min(B, ego_dim)
            idx = torch.arange(k, device=device)
            ego[idx, idx] = 1.0
        x_in_aug = torch.cat([x, ego], dim=-1)

    y_used = y_true[:B] if B is not None else y_true
    return x_in_aug, y_used, B


def train_epoch(model, loader, optimizer, criterion, device, use_port_ids=False, loss_fn=None, step_preprocess=None):
    """
    This method can be used for training both homogeneous and heterogeneous graphs
    If loss_fn is provided, it will be used (OpenFGL-style hook).
    """
    model.train()
    total_loss  = 0.0
    total_count = 0

    for batch in loader:
        batch = batch.to(device)
        x_in, edge_in, y_true, n_nodes, is_hetero = _unpack_io(batch)

        # Add Ego (if enabled) and slice labels to seeds (if B known)
        x_in_aug, y_used, B = _augment_with_ego_and_get_seed_slice(
            x_in, y_true, batch, is_hetero, model
        )

        # Assemble per-relation edge_attr dict [in_port, out_port]
        edge_attr_dict = None
        if is_hetero and use_port_ids:
            edge_attr_dict = {}
            for rel in [('n','fwd','n'), ('n','rev','n')]:
                if 'edge_attr' in batch[rel]:
                    ea = batch[rel].edge_attr
                    if ea.dtype != torch.long: ea = ea.long()
                    edge_attr_dict[rel] = ea

        # Print port information
        if not getattr(model, "_port_dbg_printed", False):
            if edge_attr_dict:
                f_ea, r_ea = edge_attr_dict[('n','fwd','n')], edge_attr_dict[('n','rev','n')]
                print(f"[PORT] fwd edge_attr: {tuple(f_ea.shape)} (dtype={f_ea.dtype}) | rev edge_attr: {tuple(r_ea.shape)}")
            else:
                print("[PORT] edge_attr_dict missing (did make_bidirected_hetero set edge_attr?)")
            model._port_dbg_printed = True

        # Print training mode (full-batch or mini-batch)
        is_full_batch = (B is None) or (B == n_nodes)
        if not getattr(model, "_mode_printed", False):
            mode = "full-batch" if is_full_batch else "mini-batch (seed-only)"
            print(f"[TRAIN MODE] {mode} | is_hetero={is_hetero} | ego_dim={getattr(model, 'ego_dim', 0)}")
            model._mode_printed = True

        # Print input dimensions as a sanity check
        if not getattr(model, "_ego_dbg_printed", False):
            base_dim = x_in['n'].shape[-1] if is_hetero else x_in.shape[-1]
            aug_dim  = x_in_aug['n'].shape[-1] if is_hetero else x_in_aug.shape[-1]
            ego_dim  = int(getattr(model, "ego_dim", 0))
            print(f"[EGO-CHECK] base_dim={base_dim}  ego_dim={ego_dim}  aug_dim={aug_dim}  seeds(B)={B}")
            model._ego_dbg_printed = True

        optimizer.zero_grad()

        if use_port_ids:
            out = model(
                x_in_aug,
                edge_in,
                edge_attr_dict=edge_attr_dict,
            )
        else:
            out = model(
                x_in_aug,
                edge_in,
            )

        out_used = out[:B] if B is not None else out
        if B is not None:
            y_used = y_used[:B]

        # If client graph has owned_mask, compute loss only on owned nodes
        owned_mask_full = None
        if is_hetero:
            if hasattr(batch['n'], 'owned_mask'):
                owned_mask_full = batch['n'].owned_mask
        else:
            if hasattr(batch, 'owned_mask'):
                owned_mask_full = batch.owned_mask

        # If full-batch (or seed slicing returns all nodes), apply owned_mask
        if owned_mask_full is not None and (B is None or B == n_nodes):
            out_used = out_used[owned_mask_full]
            y_used = y_used[owned_mask_full]
            # count should be owned nodes
            count = int(owned_mask_full.sum().item())
        else:
            count = (B if B is not None else n_nodes)

        if loss_fn is not None:
            # When we want to keep signature compatible with OpenFGL: (embedding, logits, label, mask)
            loss = loss_fn(None, out_used, y_used.float(), None)
        else:
            loss = criterion(out_used, y_used.float())

        loss.backward()

        if step_preprocess is not None:
            step_preprocess()
        
        optimizer.step()

        total_loss  += loss.item() * count
        total_count += count

    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate_epoch(
    model,
    loader,
    criterion,
    device,
    use_port_ids=False,
    return_logits_labels: bool = False,
):
    """
    Evaluate on a loader. Supports homo/hetero graphs.

    If return_logits_labels=True, also returns (logits, labels, count),
    where logits/labels correspond to the evaluated nodes (owned seeds).
    """
    model.eval()

    total_loss = 0.0
    total_count = 0
    total_pairs = 0
    correct_pairs = 0

    all_logits = []
    all_labels = []

    for batch in loader:
        batch = batch.to(device)
        x_in, edge_in, y_true, n_nodes, is_hetero = _unpack_io(batch)

        x_in_aug, y_used, B = _augment_with_ego_and_get_seed_slice(
            x_in, y_true, batch, is_hetero, model
        )

        # Assemble per-relation edge_attr dict [in_port, out_port]
        edge_attr_dict = None
        if is_hetero and use_port_ids:
            edge_attr_dict = {}
            for rel in [('n','fwd','n'), ('n','rev','n')]:
                if 'edge_attr' in batch[rel]:
                    ea = batch[rel].edge_attr
                    if ea.dtype != torch.long:
                        ea = ea.long()
                    edge_attr_dict[rel] = ea

        if use_port_ids:
            out = model(
                x_in_aug,
                edge_in,
                edge_attr_dict=edge_attr_dict,
            )
        else:
            out = model(
                x_in_aug,
                edge_in,
            )

        out_used = out[:B] if B is not None else out
        if B is not None:
            y_used = y_used[:B]

        owned_mask_full = None
        if is_hetero:
            if hasattr(batch['n'], 'owned_mask'):
                owned_mask_full = batch['n'].owned_mask
        else:
            if hasattr(batch, 'owned_mask'):
                owned_mask_full = batch.owned_mask

        if owned_mask_full is not None and (B is None or B == n_nodes):
            out_used = out_used[owned_mask_full]
            y_used = y_used[owned_mask_full]
            count = int(owned_mask_full.sum().item())
        else:
            count = int(B if B is not None else n_nodes)

        loss = criterion(out_used, y_used.float())
        total_loss += loss.item() * count
        total_count += count

        preds = (torch.sigmoid(out_used) > 0.5)
        correct_pairs += (preds == y_used.bool()).sum().item()
        total_pairs += y_used.numel()

        all_logits.append(out_used.detach().cpu())
        all_labels.append(y_used.detach().cpu())

    avg_loss = total_loss / max(total_count, 1)
    per_node_acc = correct_pairs / max(total_pairs, 1)

    logits = torch.cat(all_logits, dim=0) if len(all_logits) else torch.empty((0,))
    labels = torch.cat(all_labels, dim=0) if len(all_labels) else torch.empty((0,))

    f1_score_per_task = compute_minority_f1_score_per_task(logits, labels)

    if return_logits_labels:
        return avg_loss, per_node_acc, f1_score_per_task, logits, labels, total_count

    return avg_loss, per_node_acc, f1_score_per_task