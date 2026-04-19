#!/usr/bin/env python3
import os
import json
import logging
import torch

from scripts.data.simulator import GraphSimulator
from utils.metrics import compute_label_percentages
from utils.seed import set_seed, derive_seed

from utils.fed_motif_splitting import (
    define_subtasks_thresholds_and_witness_builders,
    set_y_with_labels_and_witnesses,
    assign_clients_from_witnesses,
    save_federated_clients,
    write_witness_split_sanity,
)

FED_CONFIG_PATH = "./configs/fed_configs.json"

with open(FED_CONFIG_PATH, "r") as f:
    ALL_FED_CONFIG = json.load(f)

FED_DATA_CONFIG = ALL_FED_CONFIG["fed_splits"]
NUM_CLIENTS = FED_DATA_CONFIG["num_clients"]
INCLUDE_CROSS_EDGES = FED_DATA_CONFIG["include_cross_edges"]

# Print logs on the terminal screen
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

# Here define the global variables
BASE_SEED = FED_DATA_CONFIG["base_seed"] 

def check_port_columns(data, name="data"):
    assert data.edge_attr is not None, f"{name}: edge_attr is None"
    E, F = data.edge_attr.shape
    print(f"[{name}] edges={E}  edge_attr_dim={F}")

    # by construction in add_ports(): the last 2 columns are [in_port, out_port]
    in_col, out_col = F - 2, F - 1
    in_ports  = data.edge_attr[:, in_col].long()
    out_ports = data.edge_attr[:, out_col].long()

    print(f"[{name}] in_port  min={int(in_ports.min())}  max={int(in_ports.max())}")
    print(f"[{name}] out_port min={int(out_ports.min())} max={int(out_ports.max())}")

    # check first 5 edges for sanity
    ei = data.edge_index
    for i in range(min(5, ei.size(1))):
        u, v = int(ei[0, i]), int(ei[1, i])
        print(f"  e#{i}: {u}->{v} | in_port={int(in_ports[i])} out_port={int(out_ports[i])}")


def write_label_stats(path, names, datasets):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("split," + ",".join(names) + ",total\n")
        for split_name, d in zip(["train", "val", "test"], datasets):
            sums = list(torch.sum(d.y, dim=0).long().cpu().numpy().tolist())
            f.write(split_name + "," + ",".join(str(x) for x in sums) + f",{d.y.shape[0]}\n")


def main():
    # Set seed for reproducibility
    set_seed(BASE_SEED)

    # Each split has a distinct and reproducible seed
    split_seeds = {
        "train": derive_seed(BASE_SEED, "train"),
        "val":   derive_seed(BASE_SEED, "val"),
        "test":  derive_seed(BASE_SEED, "test"),
    }
    logging.info("Split seeds: %s", split_seeds)

    # Below parameters are defined based on Appendix D.2 of the original paper
    n = 8192        # number of nodes
    d = 6           # average degree
    r = 11.1        # radius
    num_graphs = 1  # one connected component generator call per data split to prevent data leakeage
    generator = "chordal"  # describes the random-circulant-like generator mentioned in the paper (to my understanding)
    bidirectional = False  # have a directed multigraph (needed for directed cycles)

    # Build simulator once per split to ensure independent graphs, as described in the paper
    def make_sim():
        return GraphSimulator(
            num_nodes=n,
            avg_degree=d,
            num_edges=None,
            network_type="type1",
            readout="node",
            node_feats=False,   # for simplicity, ignore node features
            bidirectional=bidirectional,
            delta=r,                
            num_graphs=num_graphs,
            generator=generator,    
        )

    logging.info("Generating train/val/test graphs (independent random circulant graphs).")
    set_seed(split_seeds["train"])
    tr = make_sim().generate_pytorch_graph().add_ports()

    set_seed(split_seeds["val"])
    va = make_sim().generate_pytorch_graph().add_ports()

    set_seed(split_seeds["test"])
    te = make_sim().generate_pytorch_graph().add_ports()

    # Check that generated graphs have expected structure
    check_port_columns(tr, "train")
    check_port_columns(va, "val")
    check_port_columns(te, "test")

    # Label + witness extraction
    label_only_funcs, label_only_thresholds, motif_builders, names, thresholds = \
        define_subtasks_thresholds_and_witness_builders(
            cycle_max_instances_per_k=None,  # set e.g. 5000 if cycle enumeration is slow
            sg_max_instances=None,
            bp_max_instances=None,
        )

    logging.info("Computing labels + witnesses for train/val/test splits.")
    tr, tr_w = set_y_with_labels_and_witnesses(tr, label_only_funcs, label_only_thresholds, motif_builders)
    va, va_w = set_y_with_labels_and_witnesses(va, label_only_funcs, label_only_thresholds, motif_builders)
    te, te_w = set_y_with_labels_and_witnesses(te, label_only_funcs, label_only_thresholds, motif_builders)

    # Sanity checks: witnesses must imply positive labels
    # Sanity checks: witnesses must imply positive labels (task-semantic aware)
    name_to_col = {name: i for i, name in enumerate(names)}

    def _check_witnesses(data, witnesses, split_name):
        for task, insts in witnesses.items():
            col = name_to_col[task]

            # only check a few to keep it cheap
            for inst in insts[:10]:
                if task.startswith("cycle"):
                    # cycle witness contains only cycle nodes; all must be labeled
                    for u in inst:
                        assert data.y[u, col] == 1, (
                            f"[{split_name}] Witness-label mismatch: "
                            f"node {u} in {task} witness but y[{u},{col}] != 1"
                        )
                elif task in {"scatter_gather", "biclique"}:
                    # witness = (..., i) where only i is labeled positive by SG2/BP2 definition
                    i = int(inst[-1])
                    assert data.y[i, col] == 1, (
                        f"[{split_name}] Witness-label mismatch: "
                        f"gather node {i} in {task} witness but y[{i},{col}] != 1"
                    )
                else:
                    raise ValueError(f"Unknown witness task: {task}")

    _check_witnesses(tr, tr_w, "train")
    _check_witnesses(va, va_w, "val")
    _check_witnesses(te, te_w, "test")

    out_dir = "./data"
    os.makedirs(out_dir, exist_ok=True)

    labels_out_dir = "./results/metrics"
    os.makedirs(labels_out_dir, exist_ok=True)

    # Log label stats for sanity check
    write_label_stats(os.path.join(out_dir, "y_sums.csv"), names, [tr, va, te])
    logging.info("Wrote label totals to %s", os.path.join(out_dir, "y_sums.csv"))

    # Compute label percentages
    compute_label_percentages(
        input_csv=os.path.join(out_dir, "y_sums.csv"),
        output_csv=os.path.join(labels_out_dir, "label_percentages.csv"),
        add_mean=True,
    )
    logging.info("Wrote label percentages to %s", os.path.join(labels_out_dir, "label_percentages.csv"))

    # Federated splits from witnesses
    if INCLUDE_CROSS_EDGES:
        fed_root = os.path.join(out_dir, "fed_partition_aware_splits_with_cross_edges", f"{NUM_CLIENTS}_clients")
        os.makedirs(fed_root, exist_ok=True)
    else:
        fed_root = os.path.join(out_dir, "fed_partition_aware_splits_without_cross_edges", f"{NUM_CLIENTS}_clients")
        os.makedirs(fed_root, exist_ok=True)

    tr_node_to_client = assign_clients_from_witnesses(
        num_nodes=tr.num_nodes,
        witnesses=tr_w,
        num_clients=NUM_CLIENTS,
        seed=split_seeds["train"],
    )
    va_node_to_client = assign_clients_from_witnesses(
        num_nodes=va.num_nodes,
        witnesses=va_w,
        num_clients=NUM_CLIENTS,
        seed=split_seeds["val"],
    )
    te_node_to_client = assign_clients_from_witnesses(
        num_nodes=te.num_nodes,
        witnesses=te_w,
        num_clients=NUM_CLIENTS,
        seed=split_seeds["test"],
    )

    write_witness_split_sanity(
        out_csv=os.path.join(fed_root, "train", "witness_split_sanity.csv"),
        node_to_client=tr_node_to_client,
        witnesses=tr_w,
    )
    write_witness_split_sanity(
        out_csv=os.path.join(fed_root, "val", "witness_split_sanity.csv"),
        node_to_client=va_node_to_client,
        witnesses=va_w,
    )
    write_witness_split_sanity(
        out_csv=os.path.join(fed_root, "test", "witness_split_sanity.csv"),
        node_to_client=te_node_to_client,
        witnesses=te_w,
    )

    save_federated_clients(os.path.join(fed_root, "train"), tr, tr_node_to_client, include_cross_edges=INCLUDE_CROSS_EDGES)
    save_federated_clients(os.path.join(fed_root, "val"),   va, va_node_to_client, include_cross_edges=INCLUDE_CROSS_EDGES)
    save_federated_clients(os.path.join(fed_root, "test"),  te, te_node_to_client, include_cross_edges=INCLUDE_CROSS_EDGES)

    logging.info("Saved federated witness splits under %s", fed_root)

    # Local-label splits (same partitioning, labels recomputed from each client's subgraph)
    if INCLUDE_CROSS_EDGES:
        fed_root_local = os.path.join(
            out_dir, "fed_partition_aware_splits_with_cross_edges_local_labels",
            f"{NUM_CLIENTS}_clients"
        )
    else:
        fed_root_local = os.path.join(
            out_dir, "fed_partition_aware_splits_without_cross_edges_local_labels",
            f"{NUM_CLIENTS}_clients"
        )
    os.makedirs(fed_root_local, exist_ok=True)

    save_federated_clients(
        os.path.join(fed_root_local, "train"), tr, tr_node_to_client,
        include_cross_edges=INCLUDE_CROSS_EDGES, label_mode="local"
    )
    save_federated_clients(
        os.path.join(fed_root_local, "val"), va, va_node_to_client,
        include_cross_edges=INCLUDE_CROSS_EDGES, label_mode="local"
    )
    save_federated_clients(
        os.path.join(fed_root_local, "test"), te, te_node_to_client,
        include_cross_edges=INCLUDE_CROSS_EDGES, label_mode="local"
    )
    logging.info("Saved local-label federated splits under %s", fed_root_local)

    torch.save(tr, os.path.join(out_dir, "train.pt"))
    torch.save(va, os.path.join(out_dir, "val.pt"))
    torch.save(te, os.path.join(out_dir, "test.pt"))
    logging.info("Saved train/val/test GraphData objects.")

    # Log the seeds used
    seeds_out_dir = "./data/seeds"
    os.makedirs(seeds_out_dir, exist_ok=True)

    with open(os.path.join(seeds_out_dir, "global_data_seeds.txt"), "w") as f:
        for k, v in split_seeds.items():
            f.write(f"{k}:{v}\n")


if __name__ == "__main__":
    main()
