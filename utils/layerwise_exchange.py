#!/usr/bin/env python3
"""
EmbeddingTable: single-machine simulation of layer-wise cross-client
embedding exchange.

Usage (per epoch):
    table = EmbeddingTable(num_nodes=8192, num_layers=num_layers, hidden_dim=64)

    # Phase 1 — collect owned embeddings from all clients (no gradient):
    table.reset()
    for each client c:
        model_c.forward_layerwise(..., collect_into=table)

    # Phase 2 — train with injection (owned gradient intact, remote detached):
    for each client c:
        model_c.forward_layerwise(..., inject_table=table)
"""

import torch
from torch import Tensor


class EmbeddingTable:
    """
    Stores per-layer node embeddings for all owned nodes across all clients.

    Internal layout: storage[layer, global_nid] → hidden_dim-dimensional vector.
    Stored on CPU; the caller is responsible for moving fetched tensors to the
    target device.

    Nodes that have never been written return zero vectors on fetch, which is
    the correct cold-start behaviour (first epoch).
    """

    def __init__(self, num_nodes: int, num_layers: int, hidden_dim: int) -> None:
        """
        Args:
            num_nodes:  Total number of nodes in the global graph (e.g. 8192).
            num_layers: Number of GNN conv layers (one entry per layer).
            hidden_dim: Width of the hidden embeddings.
        """
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        # Float32 tensor; kept on CPU to avoid GPU memory pressure when
        # many clients share the same table.
        self.storage = torch.zeros(
            num_layers, num_nodes, hidden_dim, dtype=torch.float32
        )
        # Boolean mask tracking which entries have been written in the current step.
        # Used to distinguish "genuinely written zero" from "never written" entries.
        self.written = torch.zeros(num_layers, num_nodes, dtype=torch.bool)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Zero out all stored embeddings and written flags.  Call once per step."""
        self.storage.zero_()
        self.written.zero_()

    # ------------------------------------------------------------------
    def update(self, layer: int, global_nids: Tensor, embeddings: Tensor) -> None:
        """
        Store owned-node embeddings for the given conv layer.

        Args:
            layer:       0-based conv-layer index.
            global_nids: [K] long CPU tensor of global node IDs.
            embeddings:  [K, hidden_dim] float CPU tensor (must be detached).
        """
        self.storage[layer, global_nids.long()] = embeddings.float()
        self.written[layer, global_nids.long()] = True

    # ------------------------------------------------------------------
    def fetch(self, layer: int, global_nids: Tensor) -> Tensor:
        """
        Retrieve stored embeddings for the given global node IDs and layer.
        Nodes whose entries have never been written return zero vectors.

        Args:
            layer:       0-based conv-layer index.
            global_nids: [K] long tensor (any device — moved to CPU internally).

        Returns:
            [K, hidden_dim] float32 CPU tensor.
            The caller must move this to the appropriate device.
        """
        return self.storage[layer, global_nids.cpu().long()]

    # ------------------------------------------------------------------
    def written_mask(self, layer: int, global_nids: Tensor) -> Tensor:
        """
        Return a boolean mask indicating which node IDs were written at this layer.

        Args:
            layer:       0-based conv-layer index.
            global_nids: [K] long tensor (any device — moved to CPU internally).

        Returns:
            [K] bool CPU tensor.  True means the owning client wrote an embedding
            for that node in the current step; False means the entry is still zero.
        """
        return self.written[layer, global_nids.cpu().long()]


class LiveEmbeddingCache:
    """
    Per-step, per-layer cache of LIVE (gradient-attached, on-device) full
    embedding tensors, indexed by (layer, client_id).

    Unlike EmbeddingTable, this stores tensor REFERENCES — no detach, no CPU
    offload — so the autograd graph from the owning client's parameters
    through the stored tensor remains intact. When a consumer client fetches
    a ghost embedding, the returned tensor stays in the autograd graph, and
    gradients flowing back through the consumer's loss reach the owning
    client's parameters via the natural chain rule. Used by the bwd-coupled
    layer-wise exchange variants (train_layerwise_exchange_bwd.py and
    train_layerwise_exchange_sync_sgd_bwd.py).

    Must be reset() at the start of every training step — autograd graphs
    cannot survive across optimizer.step(), and stale references would also
    pin GPU memory.

    Indexing scheme: storage[layer][client_id] holds the FULL [N_cid,
    hidden_dim] activation tensor that client `client_id` produced at layer
    `layer`. Lookups into ghost positions go through (owner_cid, local_idx)
    pairs precomputed once outside the cache (see _build_gid_to_owner_index
    in the bwd training scripts).
    """

    def __init__(self, num_clients: int, num_layers: int) -> None:
        """
        Args:
            num_clients: Number of clients participating in the per-step
                         exchange. Storage is allocated for all of them
                         (each entry starts as None until write() is called).
            num_layers:  Number of GNN conv layers.
        """
        self.num_clients = num_clients
        self.num_layers = num_layers
        self.storage: list[list[Tensor | None]] = [
            [None] * num_clients for _ in range(num_layers)
        ]

    def reset(self) -> None:
        """
        Drop all stored tensor references. Call once per training step,
        before any write(), to release autograd graphs and free GPU memory
        from the previous step.
        """
        for l in range(self.num_layers):
            for c in range(self.num_clients):
                self.storage[l][c] = None

    def write(self, layer: int, client_id: int, x_full: Tensor) -> None:
        """
        Store a LIVE reference to the full activation tensor produced by
        `client_id` at conv layer `layer`. No detach, no copy — autograd
        edges back to the owning client's parameters are preserved.
        """
        self.storage[layer][client_id] = x_full

    def fetch_remote(
        self, layer: int, owner_cids: Tensor, local_idxs: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Gather ghost-node embeddings from the stored owner tensors, keeping
        the autograd graph intact.

        Args:
            layer:      0-based conv-layer index.
            owner_cids: [K] long tensor on device. owner_cids[k] is the
                        client id that owns the k-th ghost lookup.
                        A value < 0 means the gid is unowned (e.g., unknown
                        node) — returns a zeros row with served_mask=False.
            local_idxs: [K] long tensor on device. local_idxs[k] is the
                        local row index within owner_cids[k]'s data.x
                        for that gid.

        Returns:
            (rem_embs, served_mask):
              rem_embs:    [K, hidden_dim] LIVE tensor on the owner's device.
                           Rows where the owner has not yet written this
                           layer (or owner_cids[k] < 0) are zeros.
              served_mask: [K] bool tensor on the same device, True where
                           the lookup hit a stored entry.
        """
        if owner_cids.numel() == 0:
            return (
                torch.zeros((0, 0), device=owner_cids.device),
                torch.zeros((0,), dtype=torch.bool, device=owner_cids.device),
            )

        device = owner_cids.device
        K = owner_cids.size(0)

        served_mask = torch.zeros(K, dtype=torch.bool, device=device)
        hidden_dim = None
        for cid in range(self.num_clients):
            if self.storage[layer][cid] is not None:
                hidden_dim = self.storage[layer][cid].size(1)
                break
        if hidden_dim is None:
            return (
                torch.zeros((K, 1), device=device),
                served_mask,
            )

        rem_embs = torch.zeros((K, hidden_dim), device=device)

        unique_cids = torch.unique(owner_cids)
        for cid_t in unique_cids:
            cid = int(cid_t.item())
            if cid < 0 or self.storage[layer][cid] is None:
                continue
            sel = (owner_cids == cid)
            if not sel.any():
                continue
            owner_x = self.storage[layer][cid]
            idxs = local_idxs[sel].long()
            rem_embs[sel] = owner_x[idxs]
            served_mask[sel] = True

        return rem_embs, served_mask
