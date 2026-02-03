from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import torch

@dataclass
class NodeStats:
    count: int
    sum: torch.Tensor
    sum_sq: torch.Tensor
    min: torch.Tensor
    max: torch.Tensor

    def update(self, emb: torch.Tensor) -> None:
        # emb is 1D tensor [d] on CPU
        self.count += 1
        self.sum += emb
        self.sum_sq += emb * emb
        self.min = torch.minimum(self.min, emb)
        self.max = torch.maximum(self.max, emb)


class CrossClientComm:
    """
    Cross-client communication bus for embedding exchange.

    For each (layer, global node id), we store running statistics over all
    client-specific embeddings that have been pushed during the current round.
    This allows us to compute [mean, max, min, std] on demand for each gid.
    """

    def __init__(self, global_owner: Dict[int, int]):
        # we don't actually need global_owner for the new logic,
        # but we keep it for compatibility / potential sanity checks
        self.global_owner = global_owner
        self.reset_round()

    def reset_round(self) -> None:
        """
        Clear all per-round statistics. Call this once per global FL round.
        """
        # layer_stats[layer][gid] = NodeStats(...)
        self.layer_stats: Dict[int, Dict[int, NodeStats]] = {}

    def _get_layer_stats(self, layer: int) -> Dict[int, NodeStats]:
        if layer not in self.layer_stats:
            self.layer_stats[layer] = {}
        return self.layer_stats[layer]

    def push_local(
        self,
        client_id: int,
        layer: int,
        global_nids: torch.Tensor,  # [N]
        node_embs: torch.Tensor,    # [N, d]
    ) -> None:
        """
        Accumulate statistics for all local copies (owned + ghost) of nodes.

        Called by each client after a GNN layer.
        """
        stats_layer = self._get_layer_stats(layer)

        # Move to CPU & detach for sharing
        global_nids_cpu = global_nids.detach().cpu()
        node_embs_cpu = node_embs.detach().cpu()

        for gid, emb in zip(global_nids_cpu.tolist(), node_embs_cpu):
            if gid is None:
                continue
            if gid not in stats_layer:
                stats_layer[gid] = NodeStats(
                    count=1,
                    sum=emb.clone(),
                    sum_sq=emb * emb,
                    min=emb.clone(),
                    max=emb.clone(),
                )
            else:
                stats_layer[gid].update(emb)

    def get_consensus_features(
        self,
        layer: int,
        global_nids: torch.Tensor,  # [N]
        device: torch.device,
    ) -> torch.Tensor:
        """
        For each node in global_nids, compute [mean, max, min, std] from the
        statistics accumulated so far at this layer.

        Returns tensor of shape [N, 4 * d].
        """
        stats_layer = self._get_layer_stats(layer)

        if len(stats_layer) == 0:
            # No stats yet: just return zeros
            N = global_nids.size(0)
            return torch.zeros(N, 0, device=device)  # handled by caller

        # Infer dimensionality from any existing entry
        some_stats = next(iter(stats_layer.values()))
        d = some_stats.sum.size(0)

        N = global_nids.size(0)
        mu = torch.zeros(N, d, device=device)
        mx = torch.zeros(N, d, device=device)
        mn = torch.zeros(N, d, device=device)
        std = torch.zeros(N, d, device=device)

        gids_cpu = global_nids.detach().cpu().tolist()

        for i, gid in enumerate(gids_cpu):
            s = stats_layer.get(gid, None)
            if s is None or s.count == 0:
                # no stats for this gid yet; leave zeros
                continue
            # move stats tensors to device lazily
            sum_ = s.sum.to(device)
            sum_sq = s.sum_sq.to(device)
            min_ = s.min.to(device)
            max_ = s.max.to(device)

            mu_i = sum_ / s.count
            var_i = sum_sq / s.count - mu_i * mu_i
            var_i = torch.clamp(var_i, min=1e-12)
            std_i = torch.sqrt(var_i)

            mu[i] = mu_i
            mx[i] = max_
            mn[i] = min_
            std[i] = std_i

        z = torch.cat([mu, mx, mn, std], dim=-1)  # [N, 4d]
        return z

    def push_owned(self, *args, **kwargs):
        # no longer needed
        return

    def pull_ghost_and_merge(self, *args, **kwargs):
        # In the new design, all merging happens inside the model.
        # Here we just return the local_embs unchanged if needed in the future.
        local_embs = kwargs.get("local_embs")
        return local_embs
