#!/usr/bin/env python3
from __future__ import annotations
import math
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
        combine: str = "sum",   # 'sum', 'mean', or 'max'
        in_port_vocab_size=0,
        out_port_vocab_size=0, 
        port_emb_dim=0,
        *,
        enable_cross_client_comm: bool = False,
        comm=None,
        client_id: int | None = None,
        init_lambda: float = 0.5,
        consensus_start_layer: int = 0
    ):
        super().__init__()
        if aggregators is None:
            aggregators = ["mean", "min", "max", "std"]
        if scalers is None:
            scalers = ["amplification", "attenuation", "identity"]

        # cross-client communication flags
        self.enable_cross_client_comm = bool(enable_cross_client_comm)
        self.comm = comm
        self.client_id = client_id
        self.init_lambda = float(init_lambda)
        self.consensus_start_layer = int(consensus_start_layer)

        # Controls behavior of _cross_client_sync:
        # - apply_consensus = False: Phase A (push stats only, no blending)
        # - apply_consensus = True:  Phase C / eval (read stats, blend; no pushes)
        self.apply_consensus = True

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
                ('n','fwd','n'): PNAConv(
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
                ('n','rev','n'): PNAConv(
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
            self.bns.append(BatchNorm(hidden_dim))  # one BN per layer

        # Consensus MLPs (one per layer) to transform
        # shared summary statistics z_v received from the server per node
        self.num_layers = num_layers
        if self.enable_cross_client_comm:
            self.consensus_mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(4 * hidden_dim, hidden_dim), # Received concatenation has 4 summary stats, hence the length is "4*hidden_dim"
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_layers)
            ])

            # learnable per-layer lambda logits (global scalar per layer)
            # initialize so that sigmoid(lambda_logit) ≈ init_lambda
            # init_lambda is initialized in the pna_config file using "cross_client_initial_lambda" variable
            init_lambda = max(0.0, min(1.0, self.init_lambda))  # clamp its value to [0,1]
            eps = 1e-6
            init_lambda = max(eps, min(1.0 - eps, init_lambda))
            init_alpha = math.log(init_lambda / (1.0 - init_lambda))  # convert probability to logit

            # create a learnable vector of logits per layer
            self.lambda_logit = nn.Parameter(
                torch.full((num_layers,), float(init_alpha))
            )
        else:
            self.consensus_mlps = None
            self.lambda_logit = None

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
        each of shape [E_rel, 2] with columns [in_port, out_port] as prepared
        in make_bidirected_hetero().
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

    def _cross_client_sync(
        self,
        x: torch.Tensor,               # [N, hidden_dim]
        layer_idx: int,
        global_nids: torch.Tensor | None,
        owned_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Cross-client consensus step after each GNN layer.

        Two modes controlled by self.apply_consensus:

        - Phase A (apply_consensus = False):
            * If training: push local embeddings into comm as stats.
            * Never read or blend consensus -> x is returned unchanged.

        - Phase C / eval (apply_consensus = True):
            * Do NOT push to comm.
            * Read consensus stats from comm, map via MLP, and blend:
                x <- (1 - lambda_l) * x + lambda_l * consensus_emb

        This lets us implement the A/B/C scheme per global round.
        """
        if (
            not self.enable_cross_client_comm or
            self.comm is None or
            self.client_id is None or
            global_nids is None
        ):
            return x

        # Phase A: stats-only mode
        if not self.apply_consensus:
            # Only accumulate stats during training; in eval, do nothing.
            if self.training:
                self.comm.push_local(
                    client_id=self.client_id,
                    layer=layer_idx,
                    global_nids=global_nids,
                    node_embs=x,
                )
            # No consensus blending in Phase A
            return x

        # Phase C / eval: blend-only mode
        # Read consensus features [mu, max, min, std] from comm
        z = self.comm.get_consensus_features(
            layer=layer_idx,
            global_nids=global_nids,
            device=x.device,
        )

        # If no stats (e.g., first ever round before any Phase A), skip
        if z.numel() == 0:
            return x

        # Apply client-specific consensus MLP for this layer
        consensus_mlp = self.consensus_mlps[layer_idx]
        consensus_emb = consensus_mlp(z)  # [N, hidden_dim]

        # Learnable lambda^(l) in (0,1) via sigmoid
        lambda_l = torch.sigmoid(self.lambda_logit[layer_idx])

        # Option A: update both owned & ghost nodes  
        x = (1.0 - lambda_l) * x + lambda_l * consensus_emb

        # Option B: ghost-only update
        # if owned_mask is not None:
        #     owned_mask = owned_mask.to(x.device)
        #     ghost_mask = (~owned_mask).float().unsqueeze(-1)
        #     x = x * (1.0 - ghost_mask * lambda_l) + consensus_emb * (ghost_mask * lambda_l)

        return x

    def forward(
        self,
        x_dict,
        edge_index_dict,
        *,
        edge_attr_dict=None,
        global_nids: torch.Tensor | None = None,   # [N] global ids for 'n'
        owned_mask: torch.Tensor | None = None,    # [N] bool for 'n'
        device=None,                               # not strictly needed here
    ):
        x_dict, edge_index_dict = self._ensure_dicts(x_dict, edge_index_dict)
        x = x_dict['n']
        x = F.relu(self.input(x))

        # Build edge attrs for PNA from (in_port, out_port)
        pna_edge_attrs = self._edge_ports_to_attr(edge_attr_dict) if edge_attr_dict is not None else None

        for layer_idx, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            if pna_edge_attrs is not None:
                out_dict = conv({'n': x}, edge_index_dict, edge_attr_dict=pna_edge_attrs)
            else:
                out_dict = conv({'n': x}, edge_index_dict)

            x = out_dict['n']
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            if (
                self.enable_cross_client_comm 
                and self.comm is not None 
                and self.client_id is not None
            ):
                assert global_nids is not None, \
                    "global_nids must be provided when cross-client comm is enabled for client models"

            # apply cross-client sync hook only at the final few layers
            # configured by the consensus_start_layer parameter
            if self.enable_cross_client_comm and layer_idx >= self.consensus_start_layer:
                x = self._cross_client_sync(
                    x=x,
                    layer_idx=layer_idx,
                    global_nids=global_nids,
                    owned_mask=owned_mask,
                )

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
