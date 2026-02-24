# flcore/fedavg/client.py
import torch
from fed_algo.base import BaseClient

class FedAvgClient(BaseClient):
    """
    FedAvgClient implements the client-side logic for the Federated Averaging (FedAvg) algorithm,
    introduced in the paper "Communication-Efficient Learning of Deep Networks from Decentralized Data"
    by McMahan et al. (2017). This class extends the BaseClient class and manages local training
    and communication with the server.
    """

    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedAvgClient, self).__init__(args, client_id, data, data_dir, message_pool, device)

    # Helper: sync local model with the latest global model from server
    def _sync_with_server(self):
        """
        Copy global model parameters from the server into the local model.
        Assumes server has written its weights into message_pool["server"]["weight"].
        """
        with torch.no_grad():
            global_weights = self.message_pool["server"]["weight"]
            for local_param, global_param in zip(self.task.model.parameters(), global_weights):
                local_param.data.copy_(global_param.to(self.device))

    # Phase A: stats-only forward pass to populate CrossClientComm
    def phase_a_collect_stats(self):
        """
        Phase A of the A/B/C consensus scheme.

        - Sync local model with the global model.
        - Run a forward-only pass over local data to populate cross-client
          consensus statistics via CrossClientComm.
        - No loss is computed, no gradients are taken, no optimizer steps.

        This phase assumes that:
          - self.args.enable_cross_client_comm indicates whether consensus is used.
          - self.args.cross_client_comm is the shared CrossClientComm instance.
        """
        # If cross-client comm is disabled, nothing to do
        if not getattr(self.args, "enable_cross_client_comm", False):
            return
        if getattr(self.args, "cross_client_comm", None) is None:
            return

        # 1) Sync local model with global model
        self._sync_with_server()

        model = self.task.model
        comm  = self.args.cross_client_comm

        # 2) Configure model for Phase A: stats-only mode, no blending
        model.enable_cross_client_comm = True
        model.apply_consensus = False      # only push stats, don't blend
        model.comm = comm
        model.client_id = self.client_id

        model.train()  # training mode (dropout etc.), but we'll use no_grad

        # 3) Let the task run a stats-collection pass over its train data
        with torch.no_grad():
            if hasattr(self.task, "collect_consensus_stats"):
                self.task.collect_consensus_stats()
            else:
                raise AttributeError(
                    "NodeClsTask is missing 'collect_consensus_stats()'. "
                    "Please implement it to run a forward-only pass that "
                    "triggers CrossClientComm.push_local via PNANetReverseMP."
                )

    # Phase C: normal local training with consensus applied
    def execute(self):
        """
        Phase C of the A/B/C consensus scheme.

        - Sync local model with the global model.
        - Configure the model to APPLY consensus (read-only from comm).
        - Run the standard local training implemented in NodeClsTask.train().
        """
        # Sync local model with global model
        self._sync_with_server()

        model = self.task.model

        # If cross-client comm is enabled, turn on consensus blending
        if getattr(self.args, "enable_cross_client_comm", False) and getattr(self.args, "cross_client_comm", None) is not None:
            model.enable_cross_client_comm = True
            model.apply_consensus = True    # use consensus, no pushes
            model.comm = self.args.cross_client_comm
            model.client_id = self.client_id
        else:
            # No cross-client communication
            model.enable_cross_client_comm = False
            model.apply_consensus = True    # still True, but comm=None => no-op
            model.comm = None

        # Local training (implemented inside NodeClsTask)
        self.task.train()

    def send_message(self):
        """
        Sends a message to the server containing the model parameters after training
        and the number of samples in the client's dataset.
        """
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": [p.data.detach().cpu().clone() for p in self.task.model.parameters()],
        }