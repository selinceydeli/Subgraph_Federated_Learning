from typing import Optional
import torch
import torch.nn as nn

from utils.train_utils import ensure_node_features, train_epoch
from utils.hetero import make_bidirected_hetero
from utils.graph_helpers import max_port_cols, check_and_strip_self_loops, build_hetero_neighbor_loader, build_full_eval_loader
from models.pna_reverse_mp import PNANetReverseMP, compute_directional_degree_hists

# NodeClsTask class is used by BaseClient and BaseServer
class NodeClsTask:
    """
    Node classification task wrapper for PNA reverse-MP model,
    adapted from run_pna() for the federated setting.

    Used by:
      - FedAvgClient (for local training)
      - FedAvgServer (to hold the global model)

    Expected attributes in `args`:
      - use_ego_ids : bool
      - use_port_ids : bool
      - use_mini_batch : bool
      - batch_size : int
      - port_emb_dim : int
      - num_layers : int
      - neighbors_per_hop : int or list[int]
      - hidden_dim : int
      - dropout : float
      - lr : float
      - weight_decay : float
      - minority_class_weight : float | "auto" | None
      - local_epochs : int   (or fallback to num_epochs)
      - Optionally (to share across clients):
          * deg_fwd_hist, deg_rev_hist
          * in_port_vocab_size, out_port_vocab_size
          * ego_dim
    """

    def __init__(self,
                 args,
                 client_id: Optional[int],
                 data,
                 data_dir: str,
                 device: torch.device):

        self.args = args
        self.client_id = client_id
        self.device = device

        # hyperparams (mirroring run_pna)
        self.use_ego_ids = getattr(args, "use_ego_ids", False)
        self.use_port_ids = getattr(args, "use_port_ids", False)
        self.use_mini_batch = getattr(args, "use_mini_batch", True)
        self.batch_size = getattr(args, "batch_size", 1024)
        self.port_emb_dim = getattr(args, "port_emb_dim", 0)
        self.num_layers = getattr(args, "num_layers", 2)
        self.neighbors_per_hop = getattr(args, "neighbors_per_hop", [10] * self.num_layers)
        self.hidden_dim = getattr(args, "hidden_dim", 64)
        self.dropout = getattr(args, "dropout", 0.1)
        self.lr = getattr(args, "lr", 1e-3)
        self.weight_decay = getattr(args, "weight_decay", 1e-4)
        self.minority_class_weight = getattr(args, "minority_class_weight", None)

        # 1) Pre-process local homogeneous graph (client's split)
        name = f"client_{client_id}" if client_id is not None else "server"
        data = check_and_strip_self_loops(data, name)
        data = ensure_node_features(data)
        self.homo_data = data

        # 2) Convert to hetero + PNA degree histograms
        self.hetero_data = make_bidirected_hetero(self.homo_data)

        # Degree histograms: prefer global ones in args, else compute locally
        if hasattr(args, "deg_fwd_hist") and hasattr(args, "deg_rev_hist"):
            deg_fwd_hist = args.deg_fwd_hist
            deg_rev_hist = args.deg_rev_hist
        else:
            deg_fwd_hist, deg_rev_hist = compute_directional_degree_hists(
                edge_index=self.homo_data.edge_index,
                num_nodes=self.homo_data.num_nodes,
            )

        self.deg_fwd_hist = deg_fwd_hist
        self.deg_rev_hist = deg_rev_hist

        # 3) Port vocabulary sizes (shared or per-client)
        if self.use_port_ids:
            if hasattr(args, "in_port_vocab_size") and hasattr(args, "out_port_vocab_size"):
                in_port_vocab_size = int(args.in_port_vocab_size)
                out_port_vocab_size = int(args.out_port_vocab_size)
            else:
                in_max, out_max = max_port_cols(self.homo_data)
                in_port_vocab_size = in_max + 1
                out_port_vocab_size = out_max + 1
        else:
            in_port_vocab_size = 0
            out_port_vocab_size = 0

        self.in_port_vocab_size = in_port_vocab_size
        self.out_port_vocab_size = out_port_vocab_size

        # 4) Basic dimensions and num_samples
        in_dim = self.hetero_data['n'].x.size(-1) if 'x' in self.hetero_data['n'] else 1
        out_dim = self.hetero_data['n'].y.size(-1)
        self.out_dim = out_dim

        self.num_samples = int(self.hetero_data['n'].num_nodes)

        # Batch size for local training
        if self.use_mini_batch:
            train_batch_size = self.batch_size
        else:
            train_batch_size = self.hetero_data['n'].num_nodes

        # Ego IDs
        if self.use_ego_ids:
            ego_dim = getattr(args, "ego_dim", None)
            if ego_dim is None:
                ego_dim = train_batch_size
        else:
            ego_dim = 0
        self.ego_dim = ego_dim

        # 5) Build the PNA model
        self.model = PNANetReverseMP(
            in_dim=in_dim,
            hidden_dim=self.hidden_dim,
            out_dim=out_dim,
            deg_fwd=self.deg_fwd_hist,
            deg_rev=self.deg_rev_hist,
            num_layers=self.num_layers,
            dropout=self.dropout,
            ego_dim=self.ego_dim,
            combine="sum",
            in_port_vocab_size=self.in_port_vocab_size,
            out_port_vocab_size=self.out_port_vocab_size,
            port_emb_dim=(self.port_emb_dim if self.use_port_ids else 0),
        ).to(self.device)

        # 6) Build local training loader (mini-batch or full-batch)
        num_hops = self.num_layers  # one hop per PNA layer
        if self.use_mini_batch:
            self.train_loader = build_hetero_neighbor_loader(
                self.hetero_data,
                batch_size=train_batch_size,
                num_layers=num_hops,
                fanout=self.neighbors_per_hop,
                device=self.device,
            )
        else:
            self.train_loader = build_full_eval_loader(
                self.hetero_data,
                batch_size=train_batch_size,
                num_layers=num_hops,
                device=self.device,
            )

        # 7) Optimizer and loss (BCEWithLogits + optional pos_weight)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # auto pos_weight (per task) if requested
        auto_pos_weight = None
        if isinstance(self.minority_class_weight, str) and self.minority_class_weight == "auto":
            y_train = self.hetero_data['n'].y.float()
            pos_counts = y_train.sum(dim=0)                  # [num_tasks]
            neg_counts = (1.0 - y_train).sum(dim=0)          # [num_tasks]
            eps = 1e-8
            auto_pos_weight = neg_counts / (pos_counts + eps)

        if isinstance(self.minority_class_weight, str) and self.minority_class_weight == "auto":
            assert auto_pos_weight is not None
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=auto_pos_weight.to(self.device))
            print(f"[{name}] Using automatic per-task minority weighting: {auto_pos_weight.tolist()}")
        elif self.minority_class_weight is not None:
            pos_weight = torch.full((out_dim,), float(self.minority_class_weight), device=self.device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            print(f"[{name}] Using uniform minority class weight: {self.minority_class_weight}")
        else:
            self.criterion = nn.BCEWithLogitsLoss()
            print(f"[{name}] Using unweighted BCEWithLogitsLoss.")


    # Local training used by FedAvgClient.execute() method
    def train(self):
        """
        Local training on this client's data.

        Runs args.local_epochs epochs if present, otherwise falls back to args.num_epochs (or 1).
        Based on the centralized `train_epoch` utility.
        """
        local_epochs = getattr(self.args, "local_epochs",
                               getattr(self.args, "num_epochs", 1))

        self.model.train()
        for ep in range(local_epochs):
            _ = train_epoch(
                self.model,
                self.train_loader,
                self.optimizer,
                self.criterion,
                self.device,
                self.use_port_ids,
            )
