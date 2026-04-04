#!/usr/bin/env python3
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, PNAConv, BatchNorm
from torch_geometric.utils import degree

__all__ = ["PNANetReverseMP"]

class PNANetReverseMP(nn.Module):
    """
    Single node type 'n' with two relations:
      ('n','fwd','n') uses PNA with in-degree histogram of the original graph.
      ('n','rev','n') uses PNA with in-degree histogram of the reversed graph
                      (= out-degree histogram of the original graph).
    We combine both directions via HeteroConv(..., aggr='sum').
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        deg_fwd,
        deg_rev,
        num_layers: int = 6,
        dropout: float = 0.1,
        ego_dim: int = 0,
        aggregators=None,
        scalers=None,
        towers: int = 4,
        pre_layers: int = 1,
        post_layers: int = 1,
        divide_input: bool = False,
        combine: str = "sum",
        in_port_vocab_size=0,
        out_port_vocab_size=0,
        port_emb_dim=0,
    ):
        super().__init__()

        if aggregators is None:
            aggregators = ["mean", "min", "max", "std"]
        if scalers is None:
            scalers = ["amplification", "attenuation", "identity"]

        self.in_port_vocab_size = int(in_port_vocab_size)
        self.out_port_vocab_size = int(out_port_vocab_size)
        self.port_emb_dim = int(port_emb_dim)

        self.in_port_emb = None
        self.out_port_emb = None
        edge_dim = 0

        if self.in_port_vocab_size > 0 and self.port_emb_dim > 0:
            self.in_port_emb = nn.Embedding(self.in_port_vocab_size, self.port_emb_dim)
            edge_dim += self.port_emb_dim

        if self.out_port_vocab_size > 0 and self.port_emb_dim > 0:
            self.out_port_emb = nn.Embedding(self.out_port_vocab_size, self.port_emb_dim)
            edge_dim += self.port_emb_dim

        self.ego_dim = int(ego_dim)
        self.input = nn.Linear(in_dim + self.ego_dim, hidden_dim)
        self.dropout = dropout
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for _ in range(num_layers):
            conv_dict = {
                ('n', 'fwd', 'n'): PNAConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    aggregators=aggregators,
                    scalers=scalers,
                    deg=deg_fwd,
                    towers=towers,
                    pre_layers=pre_layers,
                    post_layers=post_layers,
                    divide_input=divide_input,
                    edge_dim=edge_dim if edge_dim > 0 else None,
                ),
                ('n', 'rev', 'n'): PNAConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    aggregators=aggregators,
                    scalers=scalers,
                    deg=deg_rev,
                    towers=towers,
                    pre_layers=pre_layers,
                    post_layers=post_layers,
                    divide_input=divide_input,
                    edge_dim=edge_dim if edge_dim > 0 else None,
                ),
            }
            self.convs.append(HeteroConv(conv_dict, aggr=combine))
            self.bns.append(BatchNorm(hidden_dim))

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    @torch.no_grad()
    def _ensure_dicts(self, x_dict, edge_index_dict):
        if isinstance(x_dict, torch.Tensor):
            x_dict = {'n': x_dict}
        return x_dict, edge_index_dict

    def _edge_ports_to_attr(self, edge_attr_dict):
        """
        Expect edge_attr_dict[('n','fwd','n')] and edge_attr_dict[('n','rev','n')]
        each of shape [E_rel, 2] with columns [in_port, out_port].
        Returns dict of float tensors [E_rel, edge_dim] for PNA.
        """
        if self.in_port_emb is None and self.out_port_emb is None:
            return None

        out = {}
        for rel, ea in edge_attr_dict.items():
            assert ea.dim() == 2 and ea.size(-1) == 2, "Expect [in_port, out_port]"
            in_ids = ea[:, 0].long()
            out_ids = ea[:, 1].long()

            parts = []
            if self.in_port_emb is not None:
                parts.append(self.in_port_emb(in_ids))
            if self.out_port_emb is not None:
                parts.append(self.out_port_emb(out_ids))

            out[rel] = torch.cat(parts, dim=-1).float()

        return out

    def forward(
        self,
        x_dict,
        edge_index_dict,
        *,
        edge_attr_dict=None,
    ):
        x_dict, edge_index_dict = self._ensure_dicts(x_dict, edge_index_dict)
        x = x_dict['n']
        x = F.relu(self.input(x))

        pna_edge_attrs = self._edge_ports_to_attr(edge_attr_dict) if edge_attr_dict is not None else None

        for conv, bn in zip(self.convs, self.bns):
            if pna_edge_attrs is not None:
                out_dict = conv({'n': x}, edge_index_dict, edge_attr_dict=pna_edge_attrs)
            else:
                out_dict = conv({'n': x}, edge_index_dict)

            x = out_dict['n']
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.mlp(x)

    def project_input(self, x_dict, edge_index_dict, *, edge_attr_dict=None):
        """Apply input linear + ReLU.  Returns (x, pna_edge_attrs, edge_index_dict).
        Used by the synchronous layer-wise exchange training loop."""
        x_dict, edge_index_dict = self._ensure_dicts(x_dict, edge_index_dict)
        x = F.relu(self.input(x_dict['n']))
        pna_edge_attrs = (
            self._edge_ports_to_attr(edge_attr_dict) if edge_attr_dict is not None else None
        )
        return x, pna_edge_attrs, edge_index_dict

    def compute_conv_layer(self, l: int, x, edge_index_dict, pna_edge_attrs=None):
        """Compute conv layer l (HeteroConv + BatchNorm + ReLU + Dropout).
        Returns updated x.  Used by the synchronous layer-wise exchange loop."""
        conv, bn = self.convs[l], self.bns[l]
        if pna_edge_attrs is not None:
            out_dict = conv({'n': x}, edge_index_dict, edge_attr_dict=pna_edge_attrs)
        else:
            out_dict = conv({'n': x}, edge_index_dict)
        x = bn(out_dict['n'])
        x = F.relu(x)
        return F.dropout(x, p=self.dropout, training=self.training)

    def compute_output(self, x):
        """Apply MLP classification head.  Returns logits [N, out_dim].
        Used by the synchronous layer-wise exchange loop."""
        return self.mlp(x)

    def encode(self, x_dict, edge_index_dict, *, edge_attr_dict=None):
        """Return hidden node representations after all conv layers (before MLP head).
        Shape: [N, hidden_dim]. Call with model.eval() and torch.no_grad()."""
        x_dict, edge_index_dict = self._ensure_dicts(x_dict, edge_index_dict)
        x = x_dict['n']
        x = F.relu(self.input(x))
        pna_edge_attrs = self._edge_ports_to_attr(edge_attr_dict) if edge_attr_dict is not None else None
        for conv, bn in zip(self.convs, self.bns):
            if pna_edge_attrs is not None:
                out_dict = conv({'n': x}, edge_index_dict, edge_attr_dict=pna_edge_attrs)
            else:
                out_dict = conv({'n': x}, edge_index_dict)
            x = out_dict['n']
            x = bn(x)
            x = F.relu(x)
            # No dropout during encode (intended for eval mode)
        return x

    def forward_layerwise(
        self,
        x_dict,
        edge_index_dict,
        *,
        edge_attr_dict=None,
        owned_mask=None,
        global_nids=None,
        inject_table=None,
        collect_into=None,
    ):
        """
        Forward pass with optional per-layer embedding injection and collection,
        used to simulate the layer-wise cross-client embedding exchange.

        Phase 1 (collect):  call with collect_into=curr_table, inject_table=None.
          After each conv+BN+ReLU+Dropout step the owned-node embeddings are
          written into collect_into before any injection takes place.

        Phase 2 (train):    call with inject_table=curr_table, collect_into=None.
          After each conv+BN+ReLU+Dropout step the remote-node (owned_mask=False)
          embeddings are replaced with values from inject_table.  The replacement
          is gradient-safe: owned nodes retain their gradient path; remote nodes
          are filled from a detached zero tensor so they contribute no gradient.

        Args:
            owned_mask:   [N] bool tensor — True for nodes owned by this client.
            global_nids:  [N] long tensor — global node ID for each local position.
            inject_table: EmbeddingTable — source of remote embeddings (Phase 2).
            collect_into: EmbeddingTable — destination for owned embeddings (Phase 1).

        Returns:
            logits: Tensor [N, out_dim].
        """
        x_dict, edge_index_dict = self._ensure_dicts(x_dict, edge_index_dict)
        x = x_dict['n']
        x = F.relu(self.input(x))
        pna_edge_attrs = (
            self._edge_ports_to_attr(edge_attr_dict) if edge_attr_dict is not None else None
        )

        for l, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            if pna_edge_attrs is not None:
                out_dict = conv({'n': x}, edge_index_dict, edge_attr_dict=pna_edge_attrs)
            else:
                out_dict = conv({'n': x}, edge_index_dict)

            x = out_dict['n']
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # Collection: store owned-node embeddings before any injection.
            if (collect_into is not None
                    and owned_mask is not None
                    and global_nids is not None):
                collect_into.update(
                    layer=l,
                    global_nids=global_nids[owned_mask].cpu(),
                    embeddings=x[owned_mask].detach().cpu(),
                )

            # Injection: replace remote-node embeddings with table values.
            # Gradient-safe: owned nodes keep their gradient; remote nodes are
            # filled from a zero tensor (no gradient path).
            if (inject_table is not None
                    and owned_mask is not None
                    and global_nids is not None):
                remote_mask = ~owned_mask
                if remote_mask.any():
                    remote_gids = global_nids[remote_mask]
                    remote_embs = inject_table.fetch(l, remote_gids).to(x.device)
                    # Build a detached fill tensor: zeros for owned, table for remote.
                    x_fill = torch.zeros_like(x)   # no gradient
                    x_fill[remote_mask] = remote_embs
                    # torch.where: owned positions → gradient flows through x;
                    #              remote positions → zero gradient (x_fill detached).
                    x = torch.where(owned_mask.unsqueeze(1).expand_as(x), x, x_fill)

        return self.mlp(x)


def compute_directional_degree_hists(edge_index, num_nodes):
    """
    Returns (deg_fwd_hist, deg_rev_hist):
      deg_fwd uses in-degree wrt original edges (target = edge_index[1]).
      deg_rev uses in-degree wrt reversed edges, i.e. out-degree of original (source = edge_index[0]).
    """
    d_fwd = degree(edge_index[1], num_nodes=num_nodes).long()
    d_rev = degree(edge_index[0], num_nodes=num_nodes).long()

    dfh = torch.bincount(d_fwd, minlength=int(d_fwd.max()) + 1)
    drh = torch.bincount(d_rev, minlength=int(d_rev.max()) + 1)
    return dfh, drh