# flcore/fedavg/client.py
import torch
from fed_algo.base import BaseClient

class FedAvgClient(BaseClient):
    """
    FedAvgClient implements the client-side logic for the Federated Averaging (FedAvg) algorithm,
    introduced in the paper "Communication-Efficient Learning of Deep Networks from Decentralized Data"
    by McMahan et al. (2017). This class extends the BaseClient class and manages local training
    and communication with the server.

    Standard FedAvg client:
    1. Pull latest global model from server
    2. Train locally
    3. Send updated parameters back to server
    """

    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedAvgClient, self).__init__(args, client_id, data, data_dir, message_pool, device)

    def execute(self):
        """
        Standard FedAvg local step:
        synchronize with server, then train locally.
        """
        with torch.no_grad():
            for local_param, global_param in zip(
                self.task.model.parameters(),
                self.message_pool["server"]["weight"]
            ):
                local_param.data.copy_(global_param.data.to(self.device))

        self.task.train()

    def send_message(self):
        """
        Send local model weights and sample count to server.
        """
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": [param.detach().clone() for param in self.task.model.parameters()]
        }