import torch
from argparse import Namespace
from typing import Any, Dict, Optional

from task.node_cls import NodeClsTask

def load_task(args: Namespace,
              client_id: Optional[int],
              data: Any,
              data_dir: str,
              device: torch.device):
    """
    Loads and returns a task instance based on the task type specified in args.task.

    For this project, only the node classification task ("node_cls") is supported.
    The tasks definition can be extended later if more tasks are added.
    """
    if args.task == "node_cls":
        return NodeClsTask(args, client_id, data, data_dir, device)
    else:
        raise ValueError(f"Unknown task type: {args.task}")


class BaseClient:
    """
    Base class for a client in a federated learning setup.

    Attributes:
        args: Arguments containing model and training configurations.
        client_id: ID of the client.
        message_pool: Shared dict for messages between client and server.
        device: Torch device.
        task: Task instance wrapping model, data, and train/eval routines.
        personalized: Flag for personalized FL algorithms.
    """
    def __init__(self,
                 args: Namespace,
                 client_id: int,
                 data: Any,
                 data_dir: str,
                 message_pool: Dict[str, Any],
                 device: torch.device,
                 personalized: bool = False):
        self.args = args
        self.client_id = client_id
        self.message_pool = message_pool
        self.device = device
        self.task = load_task(args, client_id, data, data_dir, device)
        self.personalized = personalized
    
    def execute(self):
        """Client local execution (to be implemented by subclasses)."""
        raise NotImplementedError

    def send_message(self):
        """Send a message to the server (to be implemented by subclasses)."""
        raise NotImplementedError


class BaseServer:
    """
    Base class for a server in a federated learning setup.

    Attributes:
        args: Arguments containing model and training configurations.
        message_pool: Shared dict for messages between server and clients.
        device: Torch device.
        task: Task instance wrapping the global model and (optional) global data.
        personalized: Flag for personalized FL algorithms.
    """
    def __init__(self,
                 args: Namespace,
                 global_data: Any,
                 data_dir: str,
                 message_pool: Dict[str, Any],
                 device: torch.device,
                 personalized: bool = False):
        self.args = args
        self.message_pool = message_pool
        self.device = device
        self.task = load_task(args, None, global_data, data_dir, device)
        self.personalized = personalized
   
    def execute(self):
        """Server global execution (to be implemented by subclasses)."""
        raise NotImplementedError

    def send_message(self):
        """Send messages to clients (to be implemented by subclasses)."""
        raise NotImplementedError
