# flcore/fedavg/server.py
import torch
from fed_algo.base import BaseServer

class FedAvgServer(BaseServer):
    """
    FedAvgServer implements the server-side logic for the Federated Averaging (FedAvg) algorithm,
    as introduced in the paper "Communication-Efficient Learning of Deep Networks from Decentralized Data"
    by McMahan et al. (2017). This class is responsible for aggregating model updates from clients
    and broadcasting the updated global model to all participants in the federated learning process.

    Attributes:
        None (inherits attributes from BaseServer)

    Standard FedAvg server:
    aggregate sampled client models with sample-size weighting,
    then broadcast updated global model.
    """
    
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedAvgServer, self).__init__(args, global_data, data_dir, message_pool, device)

    def execute(self):
        """
        Aggregate sampled client parameters using weighted averaging.
        """
        with torch.no_grad():
            sampled_clients = self.message_pool["sampled_clients"]
            num_tot_samples = sum(
                self.message_pool[f"client_{client_id}"]["num_samples"]
                for client_id in sampled_clients
            )

            for it, client_id in enumerate(sampled_clients):
                weight = self.message_pool[f"client_{client_id}"]["num_samples"] / num_tot_samples

                for local_param, global_param in zip(
                    self.message_pool[f"client_{client_id}"]["weight"],
                    self.task.model.parameters()
                ):
                    local_tensor = local_param.data.to(self.device)

                    if it == 0:
                        global_param.data.copy_(weight * local_tensor)
                    else:
                        global_param.data += weight * local_tensor

    def send_message(self):
        """
        Broadcast current global model to clients.
        """
        self.message_pool["server"] = {
            "weight": list(self.task.model.parameters())
        }