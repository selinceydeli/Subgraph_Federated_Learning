#!/usr/bin/env python3
import os
import time
import json
import random
from datetime import datetime
from types import SimpleNamespace
import torch

from utils.loader import load_client_graphs
from utils.sanity_check import sanity_check_client_graphs
from utils.metrics import append_f1_score_to_csv, start_epoch_csv, append_epoch_csv, compute_minority_f1_score_per_task
from utils.seed import set_seed
from utils.train_utils import load_datasets, ensure_node_features
from utils.graph_helpers import max_port_cols, check_and_strip_self_loops
from utils.federated_eval import build_federated_eval_loaders, evaluate_federated, evaluate_epoch
from models.pna_reverse_mp import compute_directional_degree_hists

from fed_algo.fedavg.client import FedAvgClient
from fed_algo.fedavg.server import FedAvgServer

from fed_algo.fedprox.client import FedProxClient
from fed_algo.fedprox.server import FedProxServer

from fed_algo.scaffold.client import ScaffoldClient
from fed_algo.scaffold.server import ScaffoldServer


PNA_CONFIG_PATH = "./configs/pna_configs.json"
FED_CONFIG_PATH = "./configs/fed_configs.json"

with open(PNA_CONFIG_PATH, "r") as f:
    ALL_PNA_CONFIG = json.load(f)

with open(FED_CONFIG_PATH, "r") as f:
    ALL_FED_CONFIG = json.load(f)

PNA_CONFIG = ALL_PNA_CONFIG["reverse_mp_with_port_and_ego"]
FED_CONFIG = ALL_FED_CONFIG["fed_learning_configs"]
COMMUNITY_SPLITS_CONFIG = ALL_FED_CONFIG["louvain_and_metis_splits"]
PARTITION_AWARE_SPLITS_CONFIG = ALL_FED_CONFIG["partition_aware_splits"]
INCLUDE_CROSS_EDGES = PARTITION_AWARE_SPLITS_CONFIG["include_cross_edges"]

ALGORITHM = FED_CONFIG["algorithm"]  # e.g. "fedavg"
LOCAL_ONLY_TRAINING = FED_CONFIG.get("local_only_training", False)
FULLY_LOCAL_EXPERIMENT = FED_CONFIG.get("fully_local_experiment", False)

MODEL_NAME = f"{PNA_CONFIG['model_name']}_{ALGORITHM.lower()}"
BEST_MODEL_PATH = f"{PNA_CONFIG['best_model_path']}_{ALGORITHM.lower()}"

USE_EGO_IDS = PNA_CONFIG["use_ego_ids"]
USE_PORT_IDS = PNA_CONFIG["use_port_ids"]
USE_MINI_BATCH = PNA_CONFIG["use_mini_batch"]
BATCH_SIZE = PNA_CONFIG["batch_size"]
PORT_EMB_DIM = PNA_CONFIG["port_emb_dim"]

DEFAULT_HPARAMS = PNA_CONFIG["default_hparams"]

PARTITION_STRATEGY = FED_CONFIG["partition_strategy"]
GLOBAL_EPOCHS = FED_CONFIG["global_epochs"]      # number of global communication rounds
GLOBAL_LOCAL_EPOCHS = FED_CONFIG["local_epochs"] # client epochs per round
CLIENT_FRACTION = FED_CONFIG["client_fraction"]  # fraction of clients per round, domain:(0,1]
BASE_SEED = FED_CONFIG["base_seed"]

def get_fl_classes(algorithm: str):
    """
    Return (ClientClass, ServerClass) for a given federated algorithm name.
    Extend this function as you add new algorithms.
    """
    alg = algorithm.lower()
    if alg == "fedavg":
        return FedAvgClient, FedAvgServer
    elif alg == "fedprox":
        return FedProxClient, FedProxServer
    elif alg == "scaffold":
        return ScaffoldClient, ScaffoldServer
    # elif alg == "fedadam":
    #     from fed_algo.fedadam.client import FedAdamClient
    #     from fed_algo.fedadam.server import FedAdamServer
    #     return FedAdamClient, FedAdamServer
    else:
        raise NotImplementedError(
            f"Algorithm '{algorithm}' is not implemented. "
            f"Supported algorithms: ['fedavg', 'fedprox', 'scaffold']"
        )


# Load the correct client and server classes based on selected algorithm 
ClientClass, ServerClass = get_fl_classes(ALGORITHM)

# Initialize number of clients and base seed using community splits config
NUM_CLIENTS = COMMUNITY_SPLITS_CONFIG["num_clients"]

if PARTITION_STRATEGY == "louvain imbalance":
    FED_TRAIN_SPLITS_DIR = "./data/fed_louvain_imbalance_splits"
elif PARTITION_STRATEGY == "metis imbalance":
    FED_TRAIN_SPLITS_DIR = "./data/fed_metis_imbalance_splits"
elif PARTITION_STRATEGY == "louvain original":              
    FED_TRAIN_SPLITS_DIR = "./data/fed_louvain_splits"             # with equal client sizes
elif PARTITION_STRATEGY == "metis original":                
    FED_TRAIN_SPLITS_DIR = "./data/fed_metis_splits"
elif PARTITION_STRATEGY == "louvain original skewed":              # with zipf-skewed client sizes
    FED_TRAIN_SPLITS_DIR = "./data/fed_louvain_splits_zipf_skewed"
elif PARTITION_STRATEGY == "metis original skewed":                
    FED_TRAIN_SPLITS_DIR = "./data/fed_metis_splits_zipf_skewed"
elif PARTITION_STRATEGY == "partition aware":
    NUM_CLIENTS = PARTITION_AWARE_SPLITS_CONFIG["num_clients"]
    if INCLUDE_CROSS_EDGES:
        data_folder_path = os.path.join("./data/fed_partition_aware_splits_with_cross_edges", f"{NUM_CLIENTS}_clients")
    else:
        data_folder_path = os.path.join("./data/fed_partition_aware_splits_without_cross_edges", f"{NUM_CLIENTS}_clients")
    FED_TRAIN_SPLITS_DIR = os.path.join(data_folder_path, "train/clients")
    FED_VAL_SPLITS_DIR  = os.path.join(data_folder_path, "val/clients")
    FED_TEST_SPLITS_DIR = os.path.join(data_folder_path, "test/clients")
else:
    raise ValueError(
        f"Unknown partition_strategy='{PARTITION_STRATEGY}'. "
        "Expected one of: "
        "['louvain imbalance', 'metis imbalance', "
        "'louvain original', 'metis original', "
        "'louvain original skewed', 'metis original skewed', "
        "'partition aware']"
    )

if PARTITION_STRATEGY != "partition aware":
    raise ValueError("This script currently expects partition-aware train/val/test client splits.")

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
        "local_only_training": LOCAL_ONLY_TRAINING,
        "fully_local_experiment": FULLY_LOCAL_EXPERIMENT,
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
    local_only_training = cfg.get("local_only_training", False)
    fully_local_experiment = cfg.get("fully_local_experiment", False)

    print(f"[FL-SETUP] Algorithm={ALGORITHM}")
    print(f"[FL-SETUP] PNA model hyperparameters: {cfg}")
    print(
        f"[FL-SETUP] num_clients={NUM_CLIENTS}, "
        f"num_rounds={num_rounds}, local_epochs={local_epochs}, "
        f"client_fraction={client_fraction}, "
        f"cross_edges={INCLUDE_CROSS_EDGES}, "
        f"local_only_training={local_only_training}, "
        f"fully_local_experiment={fully_local_experiment}"
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

    # load federated train, test, and val splits
    print(f"[FL-SETUP] Loading federated train splits from {FED_TRAIN_SPLITS_DIR}")
    client_graphs = load_client_graphs(FED_TRAIN_SPLITS_DIR, NUM_CLIENTS)

    print(f"[FL-SETUP] Loading federated val splits from {FED_VAL_SPLITS_DIR}")
    val_client_graphs = load_client_graphs(FED_VAL_SPLITS_DIR, NUM_CLIENTS)

    print(f"[FL-SETUP] Loading federated test splits from {FED_TEST_SPLITS_DIR}")
    test_client_graphs = load_client_graphs(FED_TEST_SPLITS_DIR, NUM_CLIENTS)

    # sanity checks:
    # train splits
    node_to_client_path = os.path.join(os.path.dirname(FED_TRAIN_SPLITS_DIR), "node_to_client.pt")
    if os.path.exists(node_to_client_path):
        node_to_client = torch.load(node_to_client_path)
        sanity_check_client_graphs(client_graphs, node_to_client)
    else:
        print(f"[SANITY] node_to_client.pt not found at {node_to_client_path}, skipping mapping checks.")

    # val splits
    val_node_to_client_path = os.path.join(os.path.dirname(FED_VAL_SPLITS_DIR), "node_to_client.pt")
    if os.path.exists(val_node_to_client_path):
        val_node_to_client = torch.load(val_node_to_client_path)
        sanity_check_client_graphs(val_client_graphs, val_node_to_client)
    else:
        print(f"[SANITY] node_to_client.pt not found at {val_node_to_client_path}, skipping val mapping checks.")

    # test splits
    test_node_to_client_path = os.path.join(os.path.dirname(FED_TEST_SPLITS_DIR), "node_to_client.pt")
    if os.path.exists(test_node_to_client_path):
        test_node_to_client = torch.load(test_node_to_client_path)
        sanity_check_client_graphs(test_client_graphs, test_node_to_client)
    else:
        print(f"[SANITY] node_to_client.pt not found at {test_node_to_client_path}, skipping test mapping checks.")

    base_args_kwargs = dict(
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
        # local-training only flags
        local_only_training=local_only_training,
        fully_local_experiment=fully_local_experiment,
        # global PNA stats shared across all clients
        deg_fwd_hist=deg_fwd_hist,
        deg_rev_hist=deg_rev_hist,
        in_port_vocab_size=in_port_vocab_size,
        out_port_vocab_size=out_port_vocab_size,
    )

    args = SimpleNamespace(**base_args_kwargs)
    # val_args = SimpleNamespace(**base_args_kwargs)
    # test_args = SimpleNamespace(**base_args_kwargs)

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

    # build federated eval loaders once (re-used every round)
    val_eval_loaders, _ = build_federated_eval_loaders(
        val_client_graphs,
        num_layers=num_hops,
        neighbors_per_hop=neighbors_per_hop,
        batch_size=batch_size,
        device=device,
        use_mini_batch=use_mini_batch,
    )

    test_eval_loaders, _ = build_federated_eval_loaders(
        test_client_graphs,
        num_layers=num_hops,
        neighbors_per_hop=neighbors_per_hop,
        batch_size=batch_size,
        device=device,
        use_mini_batch=use_mini_batch,
    )

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


    def eval_fully_local_split(clients, eval_loaders, split_name):
        """
        Evaluate fully local models by aggregating confusion matrix across clients.

        - For each client cid:
            - use its own model: clients[cid].task.model
            - evaluate on its own eval_loaders[cid]
        - Collect logits/labels across all clients
        - Compute one F1 score from the concatenated logits/labels
          (equivalent to summing confusion matrices across clients).

        Returns:
            split_loss: float
            split_f1:   torch.Tensor shape = (num_tasks,)
        """
        total_loss = 0.0
        total_count = 0
        all_logits = []
        all_labels = []

        for cid in range(NUM_CLIENTS):
            loader = eval_loaders[cid]
            if loader is None:
                continue

            model = clients[cid].task.model
            model.eval()

            # evaluate_epoch is the same function used by evaluate_federated
            loss_c, _, _, logits_c, labels_c, count_c = evaluate_epoch(
                model,
                loader,
                criterion,
                device,
                use_port_ids,
                return_logits_labels=True,
            )

            total_loss += loss_c * count_c
            total_count += count_c
            all_logits.append(logits_c)
            all_labels.append(labels_c)

        if total_count == 0:
            print(f"[FULLY-LOCAL][{split_name}] no samples found.")
            return float("nan"), torch.zeros(0, device=device)

        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)

        # this method computes an F1 score by aggregating client confusion matrices
        f1 = compute_minority_f1_score_per_task(logits, labels)

        print(
            f"[FULLY-LOCAL][{split_name}] "
            f"macro-minority F1: {100*f1.mean().item():.2f}%"
        )

        return (total_loss / total_count), f1


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

    # Fully-local experiment training loop
    # if in fully-local experiment mode, conduct fully local training and evaluation
    # return early without proceeding with the federated training loop
    if fully_local_experiment:
        print("[EXPERIMENT] Running fully local training & evaluation (no aggregation).")

        # Fully local training: each client trains on its own subgraph, no server interaction.
        for round_idx in range(1, num_rounds + 1):
            print(f"\n=== [FULLY-LOCAL] Round {round_idx:03d}/{num_rounds:03d} ===")
            for cid in range(NUM_CLIENTS):
                print(f"[FULLY-LOCAL] Training client {cid}")
                # Directly call the underlying task's train() – no sync_with_server, no comm.
                clients[cid].task.train()

        # After training, evaluate each local model using aggregated confusion across clients
        val_loss, val_f1 = eval_fully_local_split(
            clients, val_eval_loaders, split_name="val"
        )
        test_loss, test_f1 = eval_fully_local_split(
            clients, test_eval_loaders, split_name="test"
        )

        # write a final epoch row (use num_rounds as "epoch index")
        train_loss = float("nan")
        append_epoch_csv(epoch_csv_path, num_rounds, train_loss, val_loss, val_f1)

        return test_loss, test_f1


    # federated training loop
    for round_idx in range(1, num_rounds + 1):
        print(f"\n=== [{ALGORITHM}] Round {round_idx:03d}/{num_rounds:03d} ===")

        # 1) Sample clients
        num_sampled = max(1, int(round(client_fraction * NUM_CLIENTS)))
        if num_sampled == NUM_CLIENTS:
            sampled_clients = list(range(NUM_CLIENTS))
        else:
            sampled_clients = sorted(random.sample(range(NUM_CLIENTS), num_sampled))

        message_pool["sampled_clients"] = sampled_clients
        print(f"[FL-ROUND] Sampled clients: {sampled_clients}")

        # 2) Local training on sampled clients
        for cid in sampled_clients:
            clients[cid].execute()
            clients[cid].send_message()

        # 3) Server aggregation and broadcast
        server.execute()
        server.send_message()

        # 4) Evaluate on validation client splits
        with torch.no_grad():
            server.task.model.eval()
            val_loss, val_f1 = evaluate_federated(
                server.task.model,
                val_eval_loaders,
                criterion,
                device,
                use_port_ids=use_port_ids,
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

    # # Broadcast best weights so *_clients can sync from message_pool["server"]["weight"]
    # server.send_message()

    server.task.model.eval()
    test_loss, test_f1 = evaluate_federated(
        server.task.model,
        test_eval_loaders,
        criterion,
        device,
        use_port_ids=use_port_ids,
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
    #seeds = [BASE_SEED]
    seeds = [BASE_SEED, BASE_SEED+1, BASE_SEED+2]

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

    model_name = (
        f"PNA reverse MP {mode_str}, "
        f"partition_strategy={PARTITION_STRATEGY}, "
        f"num_clients={NUM_CLIENTS}, "
        f"local_epochs={base_hparams['local_epochs']}, "
        f"client_fraction={base_hparams['client_fraction']}, "
        f"cross_edges={INCLUDE_CROSS_EDGES}, "
        f"num_layers={DEFAULT_HPARAMS['num_layers']}, "
        f"neighbors_per_hop={DEFAULT_HPARAMS['neighbors_per_hop']}, "
        f"batch_size={BATCH_SIZE}, "
        f"local_only_training={LOCAL_ONLY_TRAINING}, "
        f"fully_local_train_and_eval={FULLY_LOCAL_EXPERIMENT}, "
        f"seed={seeds}"
    )

    append_f1_score_to_csv(
        out_csv="./results/parameter_tuning/fl_embed_exchange_tuning_f1_scores.csv",
        tasks=tasks,
        mean_f1=mean_f1,
        std_f1=std_f1,
        macro_mean_percent=macro_mean,
        seeds=seeds,
        model_name=model_name,
        runtime_seconds=runtime_sec,
    )


if __name__ == "__main__":
    main()
