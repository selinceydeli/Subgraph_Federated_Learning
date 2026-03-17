# utils/federated_eval.py
from typing import List, Tuple
import torch

from utils.hetero import make_bidirected_hetero
from utils.graph_helpers import build_hetero_neighbor_loader, build_full_eval_loader
from utils.train_utils import evaluate_epoch
from utils.metrics import compute_minority_f1_score_per_task

"""
Logic of federated evaluation is to run 
evaluate_epoch(..., return_logits_labels=True) per client

This way, we
- aggregate loss weighted by owned nodes of the clients
- concatenate logits/labels to compute one F1 score
"""

def build_federated_eval_loaders(
    client_graphs: List,
    *,
    num_layers: int,
    neighbors_per_hop,
    batch_size: int,
    device: torch.device,
    use_mini_batch: bool,
) -> Tuple[List, List[int]]:
    """
    Returns:
      loaders: list of NeighborLoaders (one per client, or None if no owned nodes)
      owned_counts: list of owned seed counts per client (0 if skipped)
    """
    loaders = []
    owned_counts = []

    for gd in client_graphs:
        hd = make_bidirected_hetero(gd)

        if not hasattr(hd["n"], "owned_mask") or hd["n"].owned_mask is None:
            raise ValueError("Federated eval requires owned_mask on client graphs.")

        owned_idx = torch.where(hd["n"].owned_mask)[0]
        owned_counts.append(int(owned_idx.numel()))

        if owned_idx.numel() == 0:
            loaders.append(None)
            continue

        if use_mini_batch:
            loader = build_hetero_neighbor_loader(
                hd,
                batch_size=batch_size,
                num_layers=num_layers,
                fanout=neighbors_per_hop,
                device=device,
                shuffle=False,
                input_nodes=owned_idx,
            )
        else:
            # single-batch full-neighborhood eval on owned seeds
            loader = build_full_eval_loader(
                hd,
                batch_size=int(owned_idx.numel()),
                num_layers=num_layers,
                device=device,
                shuffle=False,
                input_nodes=owned_idx,
            )

        loaders.append(loader)

    return loaders, owned_counts


@torch.no_grad()
def evaluate_federated(
    model,
    loaders,
    criterion,
    device: torch.device,
    *,
    use_port_ids: bool,
):
    total_loss = 0.0
    total_count = 0
    all_logits = []
    all_labels = []

    for loader in loaders:
        if loader is None:
            continue

        loss, _, _, logits, labels, count = evaluate_epoch(
            model,
            loader,
            criterion,
            device,
            use_port_ids,
            return_logits_labels=True,
        )

        total_loss += loss * count
        total_count += count
        all_logits.append(logits)
        all_labels.append(labels)

    if total_count == 0:
        return float("nan"), torch.zeros(0, device=device)

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    f1 = compute_minority_f1_score_per_task(logits, labels)

    return (total_loss / total_count), f1