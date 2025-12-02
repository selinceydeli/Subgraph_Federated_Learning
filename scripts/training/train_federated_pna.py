#!/usr/bin/env python3
import os
import time
import json
import random
from datetime import datetime
from types import SimpleNamespace
import torch

from utils.metrics import append_f1_score_to_csv, start_epoch_csv, append_epoch_csv
from utils.seed import set_seed
from utils.train_utils import load_datasets, ensure_node_features, evaluate_epoch
from utils.hetero import make_bidirected_hetero
from utils.graph_helpers import max_port_cols, check_and_strip_self_loops, build_hetero_neighbor_loader, build_full_eval_loader
from models.pna_reverse_mp import compute_directional_degree_hists

from fed_algo.fedavg.client import FedAvgClient
from fed_algo.fedavg.server import FedAvgServer


PNA_CONFIG_PATH = "./configs/pna_configs.json"
FED_CONFIG_PATH = "./configs/fed_configs.json"

with open(PNA_CONFIG_PATH, "r") as f:
    ALL_PNA_CONFIG = json.load(f)

with open(FED_CONFIG_PATH, "r") as f:
    ALL_FED_CONFIG = json.load(f)

PNA_CONFIG = ALL_PNA_CONFIG["reverse_mp_with_port_and_ego"]
FED_CONFIG = ALL_FED_CONFIG["fed_learning_configs"]
FED_DATA_CONFIG = ALL_FED_CONFIG["federated_dataset_simulation"]

ALGORITHM = FED_CONFIG["algorithm"]  # e.g. "fedavg"

MODEL_NAME = f"{PNA_CONFIG['model_name']}_{ALGORITHM.lower()}"
BEST_MODEL_PATH = f"{PNA_CONFIG['best_model_path']}_{ALGORITHM.lower()}"

USE_EGO_IDS = PNA_CONFIG["use_ego_ids"]
USE_PORT_IDS = PNA_CONFIG["use_port_ids"]
USE_MINI_BATCH = PNA_CONFIG["use_mini_batch"]
BATCH_SIZE = PNA_CONFIG["batch_size"]
PORT_EMB_DIM = PNA_CONFIG["port_emb_dim"]

DEFAULT_HPARAMS = PNA_CONFIG["default_hparams"]

NUM_CLIENTS = FED_DATA_CONFIG["num_clients"]
BASE_SEED = FED_DATA_CONFIG["base_seed"]

PARTITION_STRATEGY = FED_CONFIG["partition_strategy"]
GLOBAL_EPOCHS = FED_CONFIG["global_epochs"]      # number of global communication rounds
GLOBAL_LOCAL_EPOCHS = FED_CONFIG["local_epochs"] # client epochs per round
CLIENT_FRACTION = FED_CONFIG["client_fraction"]  # fraction of clients per round, domain:(0,1]


def get_fl_classes(algorithm: str):
    """
    Return (ClientClass, ServerClass) for a given federated algorithm name.
    Extend this function as you add new algorithms.
    """
    alg = algorithm.lower()
    if alg == "fedavg":
        return FedAvgClient, FedAvgServer
    # elif alg == "fedadam":
    #     from fed_algo.fedadam.client import FedAdamClient
    #     from fed_algo.fedadam.server import FedAdamServer
    #     return FedAdamClient, FedAdamServer
    else:
        raise NotImplementedError(
            f"Algorithm '{algorithm}' is not implemented. "
            f"Supported algorithms: ['fedavg']"
        )


# Load the correct client and server classes based on selected algorithm 
ClientClass, ServerClass = get_fl_classes(ALGORITHM)

if PARTITION_STRATEGY == "louvain imbalance split":
    FED_TRAIN_SPLITS_DIR = "./data/fed_louvain_imbalance_splits"
elif PARTITION_STRATEGY == "metis imbalance split":
    FED_TRAIN_SPLITS_DIR = "./data/fed_metis_imbalance_splits"

def run_federated_experiment(seed, tasks, device, run_id, **hparams):
    """
    Run a single federated experiment with the chosen FL algorithm (e.g., FedAvg),
    using PNA as the underlying model via NodeClsTask.
    """
    set_seed(seed)

    # Default hyperparams: PNA + FL 
    default_cfg = {
        "use_ego_ids": USE_EGO_IDS,
        "batch_size": BATCH_SIZE,
        "use_port_ids": USE_PORT_IDS,
        "use_mini_batch": USE_MINI_BATCH,
        "port_emb_dim": PORT_EMB_DIM,
        # FL-specific defaults
        "num_epochs": GLOBAL_EPOCHS,          # number of global communication rounds
        "local_epochs": GLOBAL_LOCAL_EPOCHS,  # client epochs per round
        "client_fraction": CLIENT_FRACTION,
        **DEFAULT_HPARAMS,
    }
    cfg = {**default_cfg, **hparams}

    use_ego_ids = cfg["use_ego_ids"]
    batch_size = cfg["batch_size"]
    use_mini_batch = cfg["use_mini_batch"]
    use_port_ids = cfg["use_port_ids"]
    port_emb_dim = cfg["port_emb_dim"]
    num_rounds = cfg["num_epochs"]              # global communication rounds
    num_layers = cfg["num_layers"]
    num_hops = num_layers
    neighbors_per_hop = cfg["neighbors_per_hop"]
    hidden_dim = cfg["hidden_dim"]
    dropout = cfg["dropout"]
    lr = cfg["lr"]
    weight_decay = cfg["weight_decay"]
    minority_class_weight = cfg["minority_class_weight"]

    local_epochs = cfg["local_epochs"]          # how many epochs per client per round
    client_fraction = cfg["client_fraction"]    # fraction of clients per round, domain:(0,1]

    print(f"[FL-SETUP] Algorithm={ALGORITHM}")
    print(f"[FL-SETUP] Hyperparameters: {cfg}")
    print(
        f"[FL-SETUP] num_clients={NUM_CLIENTS}, "
        f"num_rounds={num_rounds}, local_epochs={local_epochs}, "
        f"client_fraction={client_fraction}"
    )

    model_dir = os.path.join(BEST_MODEL_PATH, f"run_{run_id}_seed{seed}")
    os.makedirs(model_dir, exist_ok=True)

    # load centralized datasets
    train_data, val_data, test_data = load_datasets()

    train_data = check_and_strip_self_loops(train_data, "train")
    val_data = check_and_strip_self_loops(val_data, "val")
    test_data = check_and_strip_self_loops(test_data, "test")

    train_data = ensure_node_features(train_data)
    val_data = ensure_node_features(val_data)
    test_data = ensure_node_features(test_data)

    # port vocab sizes (computed from full central data, shared across clients)
    if use_port_ids:
        tr_in_max, tr_out_max = max_port_cols(train_data)
        va_in_max, va_out_max = max_port_cols(val_data)
        te_in_max, te_out_max = max_port_cols(test_data)
        in_port_vocab_size = max(tr_in_max, va_in_max, te_in_max) + 1
        out_port_vocab_size = max(tr_out_max, va_out_max, te_out_max) + 1
    else:
        in_port_vocab_size = 0
        out_port_vocab_size = 0

    # degree histograms from global train graph
    deg_fwd_hist, deg_rev_hist = compute_directional_degree_hists(
        edge_index=train_data.edge_index,
        num_nodes=train_data.num_nodes,
    )

    # convert val/test to HeteroData for evaluation
    val_h = make_bidirected_hetero(val_data)
    test_h = make_bidirected_hetero(test_data)

    # decide batch sizes for val/test
    if use_mini_batch:
        val_batch_size = batch_size
        test_batch_size = batch_size
    else:
        val_batch_size = val_h["n"].num_nodes
        test_batch_size = test_h["n"].num_nodes

    # build loaders for validation & test
    if use_mini_batch:
        valid_loader = build_hetero_neighbor_loader(
            val_h,
            batch_size=val_batch_size,
            num_layers=num_hops,
            fanout=neighbors_per_hop,
            device=device,
        )
        test_loader = build_hetero_neighbor_loader(
            test_h,
            batch_size=test_batch_size,
            num_layers=num_hops,
            fanout=neighbors_per_hop,
            device=device,
        )
    else:
        valid_loader = build_full_eval_loader(
            val_h,
            batch_size=val_batch_size,
            num_layers=num_hops,
            device=device,
        )
        test_loader = build_full_eval_loader(
            test_h,
            batch_size=test_batch_size,
            num_layers=num_hops,
            device=device,
        )

    # load federated train splits
    print(f"[FL-SETUP] Loading federated train splits from {FED_TRAIN_SPLITS_DIR}")
    client_graphs = []
    for cid in range(NUM_CLIENTS):
        path = os.path.join(FED_TRAIN_SPLITS_DIR, f"client_{cid}.pt")
        client_graphs.append(torch.load(path, weights_only=False))

    assert len(client_graphs) == NUM_CLIENTS, (
        f"Expected {NUM_CLIENTS} client graphs, got {len(client_graphs)}"
    )

    args = SimpleNamespace(
        task="node_cls",
        # model / training hyperparams
        use_ego_ids=use_ego_ids,
        use_port_ids=use_port_ids,
        use_mini_batch=use_mini_batch,
        batch_size=batch_size,
        port_emb_dim=port_emb_dim,
        num_layers=num_layers,
        neighbors_per_hop=neighbors_per_hop,
        hidden_dim=hidden_dim,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        minority_class_weight=minority_class_weight,
        # federated-specific
        num_epochs=num_rounds,
        local_epochs=local_epochs,
        # global PNA stats shared across all clients
        deg_fwd_hist=deg_fwd_hist,
        deg_rev_hist=deg_rev_hist,
        in_port_vocab_size=in_port_vocab_size,
        out_port_vocab_size=out_port_vocab_size,
    )

    # set up FL server & clients (algorithm-agnostic)
    message_pool = {}

    # server uses the central train_data to build the global model
    server = ServerClass(
        args=args,
        global_data=train_data,
        data_dir="./data",
        message_pool=message_pool,
        device=device,
    )

    # use server's criterion (from NodeClsTask) for evaluation
    criterion = server.task.criterion

    # build clients from federated splits
    clients = []
    for cid in range(NUM_CLIENTS):
        c = ClientClass(
            args=args,
            client_id=cid,
            data=client_graphs[cid],
            data_dir="./data",
            message_pool=message_pool,
            device=device,
        )
        clients.append(c)

    # initial broadcast of global model to clients
    server.send_message()

    # log metrics
    os.makedirs(BEST_MODEL_PATH, exist_ok=True)

    epoch_csv_path = start_epoch_csv(
        model_name=MODEL_NAME,
        seed=seed,
        tasks=tasks,
        out_dir=f"./results/metrics/epoch_logs/{MODEL_NAME}",
    )

    best_ckpt_path = os.path.join(model_dir, "best_model.pt")
    best_val = float("inf")

    # federated training loop
    for round_idx in range(1, num_rounds + 1):
        print(f"\n=== [{ALGORITHM}] Round {round_idx:03d}/{num_rounds:03d} ===")

        # sample clients according to client_fraction
        num_sampled = max(1, int(round(client_fraction * NUM_CLIENTS)))
        if num_sampled == NUM_CLIENTS:
            sampled_clients = list(range(NUM_CLIENTS))
        else:
            sampled_clients = sorted(random.sample(range(NUM_CLIENTS), num_sampled))

        message_pool["sampled_clients"] = sampled_clients
        print(f"[FL-ROUND] Sampled clients: {sampled_clients}")

        # local execution on each sampled client
        for cid in sampled_clients:
            clients[cid].execute()
            clients[cid].send_message()

        # global aggregation on server
        server.execute()
        server.send_message()  # broadcast updated global state (e.g., model weights)

        # validation on centralized validation graph
        with torch.no_grad():
            val_loss, _, val_f1 = evaluate_epoch(
                server.task.model,
                valid_loader,
                criterion,
                device,
                USE_PORT_IDS,
            )

        # We don't have a clean single scalar train_loss for all clients,
        # store NaN in the CSV for train_loss
        train_loss = float("nan")
        append_epoch_csv(epoch_csv_path, round_idx, train_loss, val_loss, val_f1)

        val_macro = val_f1.mean().item()
        print(
            f"[seed {seed}] Round {round_idx:03d} | "
            f"val {val_loss:.4f} | val macro-minF1 {100*val_macro:.2f}%"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(server.task.model.state_dict(), best_ckpt_path)

    # final test evaluation on best global model
    server.task.model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    test_loss, _, test_f1 = evaluate_epoch(
        server.task.model,
        test_loader,
        criterion,
        device,
        USE_PORT_IDS,
    )
    return test_loss, test_f1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    start_ts = time.perf_counter()

    # Define the sub-tasks (same order as centralized)
    tasks = [
        "deg-in",
        "deg-out",
        "fan-in",
        "fan-out",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "S-G",
        "B-C",
    ]

    base_hparams = dict(
        num_layers=DEFAULT_HPARAMS["num_layers"],
        neighbors_per_hop=DEFAULT_HPARAMS["neighbors_per_hop"],
        minority_class_weight=DEFAULT_HPARAMS["minority_class_weight"],
        use_ego_ids=USE_EGO_IDS,
        use_mini_batch=USE_MINI_BATCH,
        batch_size=BATCH_SIZE,
        use_port_ids=USE_PORT_IDS,
        port_emb_dim=PORT_EMB_DIM,
        # Use FL config for rounds / local epochs / client fraction
        num_epochs=GLOBAL_EPOCHS,              # global rounds
        local_epochs=GLOBAL_LOCAL_EPOCHS,
        client_fraction=CLIENT_FRACTION,
        hidden_dim=DEFAULT_HPARAMS["hidden_dim"],
        dropout=DEFAULT_HPARAMS["dropout"],
        lr=DEFAULT_HPARAMS["lr"],
        weight_decay=DEFAULT_HPARAMS["weight_decay"],
    )

    # For testing, use single seed
    # seeds = [BASE_SEED, BASE_SEED+1, BASE_SEED+2, BASE_SEED+3, BASE_SEED+4]
    seeds = [BASE_SEED]

    test_f1_scores = []
    for s in seeds:
        _, test_f1 = run_federated_experiment(s, tasks, device, run_id=run_id, **base_hparams)
        test_f1_scores.append(test_f1.cpu())

    all_f1 = torch.stack(test_f1_scores, dim=0)
    mean_f1 = all_f1.mean(dim=0)
    std_f1 = all_f1.std(dim=0, unbiased=False)

    macro_mean = mean_f1.mean().item() * 100

    mode_str = f"federated {ALGORITHM} (mini-batch)" if USE_MINI_BATCH else f"federated {ALGORITHM} (full-batch)"
    print(
        f"\nPNA reverse message passing in {mode_str}, "
        f"port numbers={USE_PORT_IDS}, & ego IDs={USE_EGO_IDS} — "
        f"macro minority F1 over {len(seeds)} runs: {macro_mean:.2f}%"
    )

    row = " | ".join(
        f"{n}: {100*m:.2f}±{100*s:.2f}%"
        for n, m, s in zip(tasks, mean_f1.tolist(), std_f1.tolist())
    )
    print("Per-task (mean±std over runs):", row)

    runtime_sec = time.perf_counter() - start_ts

    append_f1_score_to_csv(
        out_csv="./results/metrics/f1_scores_federated.csv",
        tasks=tasks,
        mean_f1=mean_f1,
        std_f1=std_f1,
        macro_mean_percent=macro_mean,
        seeds=seeds,
        model_name=f"PNA reverse MP {mode_str}, num_clients={NUM_CLIENTS} ,local_epochs={base_hparams['local_epochs']}, client_fraction={base_hparams['client_fraction']}",
        runtime_seconds=runtime_sec,
    )


if __name__ == "__main__":
    main()
