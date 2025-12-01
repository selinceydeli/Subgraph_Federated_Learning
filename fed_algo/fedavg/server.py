# flcore/fedavg.py
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
    """
    
    def __init__(self, args, global_data, data_dir, message_pool, device):
        """
        Initializes the FedAvgServer.

        Attributes:
            args (Namespace): Arguments containing model and training configurations.
            global_data (object): Global dataset accessible by the server.
            data_dir (str): Directory containing the data.
            message_pool (object): Pool for managing messages between server and clients.
            device (torch.device): Device to run the computations on.
        """
        super(FedAvgServer, self).__init__(args, global_data, data_dir, message_pool, device)

    def execute(self):
        """
        Executes the server-side operations. This method aggregates model updates from the 
        clients by computing a weighted average of the model parameters, based on the number 
        of samples each client used for training.
        """
        sampled_clients = self.message_pool["sampled_clients"]
        with torch.no_grad():
            num_tot_samples = sum(
                self.message_pool[f"client_{cid}"]["num_samples"]
                for cid in sampled_clients
            )

            # Initialize global parameters as zero
            for g_param in self.task.model.parameters():
                g_param.data.zero_()

            # Weighted sum of client params
            for cid in sampled_clients:
                client_msg = self.message_pool[f"client_{cid}"]
                weight_factor = client_msg["num_samples"] / num_tot_samples

                for g_param, c_param in zip(self.task.model.parameters(),
                                            client_msg["weight"]):
                    g_param.data += weight_factor * c_param.to(self.device)

    def send_message(self):
        """
        Sends a message to the clients containing the updated global model parameters after 
        aggregation.
        """
        self.message_pool["server"] = {
            "weight": [p.data.detach().cpu().clone() for p in self.task.model.parameters()]
        }
