#!/usr/bin/env python3
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
        deg_fwd,                # histogram for in-degrees w.r.t. fwd edges
        deg_rev,                # histogram for in-degrees w.r.t. rev edges
        num_layers: int = 6,
        dropout: float = 0.1,
        ego_dim:int = 0,        # pass ego-ID dimension
        aggregators=None,
        scalers=None,
        towers: int = 4,
        pre_layers: int = 1,
        post_layers: int = 1,
        divide_input: bool = False,
        combine: str = "sum",   # how to combine relations in HeteroConv - possible values: 'sum', 'mean', or 'max'
        in_port_vocab_size=0,
        out_port_vocab_size=0, 
        port_emb_dim=0,
    ):
        super().__init__()
        if aggregators is None:
            aggregators = ["mean", "min", "max", "std"]
        if scalers is None:
            scalers = ["amplification", "attenuation", "identity"]

        self.in_port_vocab_size  = int(in_port_vocab_size)
        self.out_port_vocab_size = int(out_port_vocab_size)
        self.port_emb_dim        = int(port_emb_dim)

        self.in_port_emb  = None
        self.out_port_emb = None
        edge_dim = 0
        if self.in_port_vocab_size > 0 and self.port_emb_dim > 0:
            self.in_port_emb = nn.Embedding(self.in_port_vocab_size,  self.port_emb_dim)
            edge_dim += self.port_emb_dim
        if self.out_port_vocab_size > 0 and self.port_emb_dim > 0:
            self.out_port_emb = nn.Embedding(self.out_port_vocab_size, self.port_emb_dim)
            edge_dim += self.port_emb_dim

        self.ego_dim = int(ego_dim)
        self.input = nn.Linear(in_dim + self.ego_dim, hidden_dim)
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        for _ in range(num_layers):
            conv_dict = {
                # Define PNA with in-degree histogram of the original graph.
                ('n','fwd','n'): PNAConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    aggregators=aggregators,
                    scalers=scalers,
                    deg=deg_fwd,    # histogram for in-degrees w.r.t. fwd edges
                    towers=towers,
                    pre_layers=pre_layers,
                    post_layers=post_layers,
                    divide_input=divide_input,
                    edge_dim=edge_dim if edge_dim > 0 else None,
                ),
                # Define PNA with in-degree histogram of the reversed graph.
                ('n','rev','n'): PNAConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    aggregators=aggregators,
                    scalers=scalers,
                    deg=deg_rev,    # histogram for in-degrees w.r.t. rev edges
                    towers=towers,
                    pre_layers=pre_layers,
                    post_layers=post_layers,
                    divide_input=divide_input,
                    edge_dim=edge_dim if edge_dim > 0 else None,
                ),
            }
            self.convs.append(HeteroConv(conv_dict, aggr=combine))
            self.bns.append(BatchNorm(hidden_dim))  # one BN per layer

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),  # per-node logits
        )

    @torch.no_grad()
    def _ensure_dicts(self, x_dict, edge_index_dict):
        # Convenience: allow passing homogeneous x & edge_index via ('n','*','n')
        if isinstance(x_dict, torch.Tensor):
            x_dict = {'n': x_dict}
        return x_dict, edge_index_dict
    
    def _edge_ports_to_attr(self, edge_attr_dict):
        """
        Expect edge_attr_dict[('n','fwd','n')] and edge_attr_dict[('n','rev','n')]
        each of shape [E_rel, 2] with columns [in_port, out_port] as prepared in make_bidirected_hetero().
        Returns dict of float tensors [E_rel, edge_dim] for PNA.
        """
        if self.in_port_emb is None and self.out_port_emb is None:
            return None

        out = {}
        for rel, ea in edge_attr_dict.items():
            # ea: [E, 2] longs: [in_port, out_port]
            assert ea.dim() == 2 and ea.size(-1) == 2, "Expect [in_port, out_port]"
            in_ids  = ea[:, 0].long()
            out_ids = ea[:, 1].long()
            parts = []
            if self.in_port_emb is not None:
                parts.append(self.in_port_emb(in_ids))
            if self.out_port_emb is not None:
                parts.append(self.out_port_emb(out_ids))
            out[rel] = torch.cat(parts, dim=-1).float()  # [E, edge_dim]
        return out

    def forward(self, x_dict, edge_index_dict, *, edge_attr_dict=None):
        x_dict, edge_index_dict = self._ensure_dicts(x_dict, edge_index_dict)
        x = x_dict['n']
        x = F.relu(self.input(x))

        # Build edge attrs for PNA from (in_port, out_port)
        pna_edge_attrs = self._edge_ports_to_attr(edge_attr_dict) if edge_attr_dict is not None else None

        for conv, bn in zip(self.convs, self.bns):
            out_dict = conv({'n': x}, edge_index_dict, edge_attr_dict=pna_edge_attrs)
            x = out_dict['n']
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

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
