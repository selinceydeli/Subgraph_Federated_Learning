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
