# Subgraph Federated Learning <!-- omit in toc -->

A repository for **synthetic subgraph-detection benchmarking** and **PNA-based baselines** on directed multigraphs.

It provides a fully reproducible pipeline for generating synthetic multigraphs with node-level pattern labels, partitioning them into federated client subgraphs using both **community-detection–based methods** (Louvain, Metis) and a custom **pattern-aware splitting strategy**, and training **centralized and federated PNA models** for financial crime detection.

## Table of Contents <!-- omit in toc -->

- [Synthetic Pattern Detection Dataset Generation](#synthetic-pattern-detection-dataset-generation)
  - [Label Tasks](#label-tasks)
  - [Default Generation Settings for Synthetic Graph](#default-generation-settings-for-synthetic-graph)
  - [How to Generate Synthetic Graph](#how-to-generate-synthetic-graph)
- [Federated Subgraph Partitioning](#federated-subgraph-partitioning)
  - [Community-Detection-Based Partitioning](#community-detection-based-partitioning)
    - [Original Splits (Equal-Sized Clients)](#original-splits-equal-sized-clients)
    - [Original Splits (Zipf-Skewed Clients)](#original-splits-zipf-skewed-clients)
    - [Label-Controlled Splits (Imbalance-Handled)](#label-controlled-splits-imbalance-handled)
    - [How to Generate Metis- and Louvain-based Splits](#how-to-generate-metis--and-louvain-based-splits)
  - [Pattern-Aware Federated Splits (Witness-Based)](#pattern-aware-federated-splits-witness-based)
    - [Motivation](#motivation)
    - [Cross-Client Edge Handling](#cross-client-edge-handling)
    - [How to Generate Pattern-Aware Splits](#how-to-generate-pattern-aware-splits)
- [Principal Neighborhood Aggregation (PNA)](#principal-neighborhood-aggregation-pna)
  - [1. Baseline PNA (Full-Batch Training)](#1-baseline-pna-full-batch-training)
  - [2. PNA with Reverse Message Passing (Mini-Batch Training)](#2-pna-with-reverse-message-passing-mini-batch-training)
  - [Training Configuration for Centralized PNA Model](#training-configuration-for-centralized-pna-model)
- [PNA Training Under Federated Setting](#pna-training-under-federated-setting)
  - [Fully-Local Federated Baseline](#fully-local-federated-baseline)
  - [Standard FedAvg](#standard-fedavg)
  - [Sync-SGD](#sync-sgd)
  - [Layer-wise Embedding Exchange](#layer-wise-embedding-exchange)
  - [Per-Epoch Layer-wise Embedding Exchange](#per-epoch-layer-wise-embedding-exchange)
  - [Federated Learning Configuration](#federated-learning-configuration)
    - [Federated Training Hyperparameters](#federated-training-hyperparameters)
    - [Hyperparameters for Partition-Aware Splits](#hyperparameters-for-partition-aware-splits)
    - [Hyperparameters for Louvain- and Metis-Based Splits](#hyperparameters-for-louvain--and-metis-based-splits)
- [Reproducibility](#reproducibility)

## Synthetic Pattern Detection Dataset Generation

This repository includes a **synthetic subgraph-detection dataset** used for benchmarking graph models for the pattern detection task. The graphs and labels are generated following the pseudocode and configurations described in [Provably Powerful Graph Neural Networks for Directed Multigraphs](https://arxiv.org/abs/2306.11586) (Egressy et al., 2023).

### Label Tasks

The generated synthetic subgraph-detection dataset consists of eleven money laundering patterns. These patterns are randomly injected into graphs using a _random circulant–like graph generator_ (Egressy et al., 2023). The eleven tasks consist of **four degree-based motifs** and **seven higher-order structural motifs.**

The degree-based motifs are **degree-in/out** (the number of incoming and outgoing edges) and **fan-in/out** (the number of unique incoming and outgoing neighbors). For each of these four tasks, a node’s label is set to _true_ if the corresponding quantity is greater than three.

The remaining seven tasks are defined based on a node’s participation in higher-order structural motifs: **scatter–gather patterns, directed bicliques, and directed cycles of length up to six**. For these motif tasks, a node’s label is set to _true_ if it participates in at least one instance of the corresponding motif.

---

### Default Generation Settings for Synthetic Graph

The default generation config (see the generator script `scripts/data/generate_synthetic.py`) follows the paper’s setup:

- Nodes `n = 8192`
- Average degree `d = 6`
- Radius parameter `r = 11.1`
- Directed multigraphs (for directed cycles)
- Generator: `chordal` (random-circulant-like graph generator)
- One connected component per split (prevents data leakage)

---

### How to Generate Synthetic Graph

From the repository root, run:

```bash
python3 -m scripts.data.generate_synthetic
```

This command generates the synthetic pattern-detection graphs and saves the following files:

- `./data/train.pt`
- `./data/val.pt`
- `./data/test.pt`
- `./data/y_sums.csv` — positive-label counts per sub-task
- `./results/metrics/label_percentages.csv` — label percentages for sanity checking against the statistics reported by the original paper

## Federated Subgraph Partitioning

### Community-Detection-Based Partitioning

In the federated setting, each client is represented by a subgraph of the global synthetic graph. We use two community-detection–based partitioning techniques:

- **Metis:** balanced k-way graph partitioning
- **Louvain:** modularity-based community detection

Both follow the methodology of
[OpenFGL: A Comprehensive Benchmark for Federated Graph Learning](https://arxiv.org/abs/2408.16288) (Li et al., 2024), extended here for multi-task labels.

#### Original Splits (Equal-Sized Clients)

The default experimental setup uses **approximately equal-sized clients**. After detecting communities, we assign them to clients using a greedy bin-packing strategy, producing subgraphs with similar node counts. This provides a controlled and stable federated environment for evaluating performance differences between centralized and decentralized training.

#### Original Splits (Zipf-Skewed Clients)

To simulate more realistic financial crime settings with different client sizes, we additionally support **Zipf-skewed** splits. Communities are assigned to clients according to a Zipf-like distribution, producing:

- a few large clients,
- many small clients.

These splits model strongly **non-uniform client sizes**, common in real-world networks.

#### Label-Controlled Splits (Imbalance-Handled)

We also provide **label-controlled (imbalance-handled)** splits following the OpenFGL label-distribution clustering strategy. Communities are clustered based on their multi-task label distributions and grouped to explicitly reduce extreme label skew across clients. These splits are intended for **controlled benchmarking**, where label distributions and task difficulty are kept comparable across clients.

---

#### How to Generate Metis- and Louvain-based Splits

From the repository root:

```bash
python3 -m scripts.data.make_federated_splits
```

This produces **six federated split directories** (each containing per-client `.pt` files) **plus a client size summary** under `./data/`:

- `fed_louvain_splits/` — Louvain, equal-sized
- `fed_metis_splits/` — Metis, equal-sized
- `fed_louvain_splits_zipf_skewed/` — Louvain, Zipf-skewed
- `fed_metis_splits_zipf_skewed/` — Metis, Zipf-skewed
- `fed_louvain_imbalance_splits/` — Louvain, label-controlled (imbalance-handled)
- `fed_metis_imbalance_splits/` — Metis, label-controlled (imbalance-handled)

Additionally:

- `client_sizes.csv` — Number of nodes and edges for each client across all split types

The training script automatically selects the appropriate directory using:

```json
"partition_strategy": "<strategy_name>"
```

Examples:
`"metis original"`, `"louvain original skewed"`, `"metis imbalance"`

---

### Pattern-Aware Federated Splits (Witness-Based)

In addition to community-detection–based partitioning, this repository provides a custom **pattern-aware federated splitting strategy** that explicitly divides _structural subgraph patterns_ across clients.

Unlike Metis- or Louvain-based approaches, which operate purely on graph topology, this strategy uses **pattern witnesses**, the exact node sets that form each labeled subgraph instance (e.g., cycles, scatter–gather motifs, or bicliques), to guide client assignment.

#### Motivation

In realistic federated financial crime detection settings, illicit activity patterns are often **distributed across institutions** rather than localized within a single organization. Community-based partitioning can unintentionally concentrate entire subgraph patterns within a single client, making the federated task artificially easy.

The witness-based strategy addresses this by enforcing **instance-level pattern dispersion**:

> Nodes participating in the same structural pattern are, whenever possible, assigned to different clients.

This yields a federated dataset with **stronger non-IID structure** and a more realistic financial crime setting.

---

#### Cross-Client Edge Handling

The inclusion of edges that span multiple clients is controlled by the configuration flag:

```json
"partition_aware_splits": {
  "num_clients": 3,
  "include_cross_edges": true,
  "base_seed": 42
}
```

where **`include_cross_edges = true`** is the default.

---

#### How to Generate Pattern-Aware Splits

Running the synthetic data generation script:

```bash
python3 -m scripts.data.generate_synthetic
```

produces the following federated split directories:

- `./data/fed_partition_aware_splits_with_cross_edges/`
- `./data/fed_partition_aware_splits_without_cross_edges/`
- `./data/fed_partition_aware_splits_with_cross_edges_local_labels/`
- `./data/fed_partition_aware_splits_without_cross_edges_local_labels/`

The `_local_labels` variants use labels **recomputed from each client's local subgraph topology** rather than copied from the global graph.

All four directories contain **pattern-aware federated splits** for each global graph split (`train/`, `val/`, `test/`). Each split directory has the following structure:

- `clients/client_XXXX.pt` — per-client subgraphs
- `node_to_client.pt` — node-to-client assignment
- `client_sizes.csv` — number of nodes and edges per client
- `witness_split_sanity.csv` — sanity check showing how often pattern instances are split across distinct clients

The training script automatically uses these **pattern-aware splits** when the federated configuration specifies:

```json
"partition_strategy": "partition aware"
```

## Principal Neighborhood Aggregation (PNA)

This repository provides two implementations of the **Principal Neighborhood Aggregation (PNA)** model, one baseline version using standard message passing, and an enhanced version that incorporates **Reverse Message Passing**, **Ego IDs**, **Port IDs**, and **mini-batch neighborhood sampling** for scalable training.

Both implementations follow the PNA architecture introduced in
[Principal Neighbourhood Aggregation for Graph Nets](https://arxiv.org/abs/2004.05718) (Corso et al., 2020).

### 1. Baseline PNA (Full-Batch Training)

The baseline model uses the original PNAConv layers from PyTorch Geometric and is trained in the **full-batch** setting.

To train and evaluate the baseline model:

```bash
python3 -m scripts.training.train_pna_baseline
```

The baseline model:

- operates directly on the homogeneous directed multigraph,
- uses full-batch message passing over the entire graph,
- does **not** use Ego IDs or Port IDs,
- serves as the reference for evaluating all incremental adaptations.

---

### 2. PNA with Reverse Message Passing (Mini-Batch Training)

This extended version incorporates several adaptations designed to improve pattern detection in directed multigraphs:

- **Reverse Message Passing** (direction-aware PNA aggregation)
- **Heterogeneous graph transformation** (`fwd` and `rev` edge types)
- **Ego ID embeddings** (to preserve seed-identity across sampled mini-batches)
- **Port ID embeddings** (to encode in/out-port numbers)
- **Mini-batch neighborhood sampling** using PyG’s `NeighborLoader`
- **Configurable fanout per hop** (default: `[10, 10, 10, 10, 5, 5]`)
- **`encode()` method** — returns pre-MLP node representations of shape `[N, hidden_dim]`, used by the Cross-Client Embedding Bootstrap pipeline to extract structural embeddings without passing through the classification head

To train and evaluate this model:

```bash
python3 -m scripts.training.train_pna_reverse_mp
```

This version serves as the foundation for future **federated** extensions.

---

### Training Configuration for Centralized PNA Model

Both PNA variants share the following core hyperparameters:

- **`hidden_dim = 64`**
  Dimensionality of node embeddings throughout the network.

- **`dropout = 0.1`**
  Dropout rate applied during training to reduce overfitting.

- **`lr = 0.001`**
  Learning rate used by the Adam optimizer.

- **`weight_decay = 0.0001`**
  L2 regularization strength to prevent overfitting.

The two models differ in depth:

- **`num_layers = 2`** (baseline PNA)
- **`num_layers = 6`** (PNA with reverse message passing)

Additional hyperparameters apply to the extended PNA model with reverse message passing:

- **`batch_size = 32`**
  Number of seed nodes sampled per mini-batch.

- **`neighbors_per_hop = [10, 10, 10, 10, 5, 5]`**
  Number of neighbors sampled at each hop for scalable neighborhood expansion. The three entries correspond to the three GNN layers.

- **`ego_dim = null`**
  When set to _null_, the ego embedding dimension is derived automatically from `batch_size` (i.e., 32 at runtime). This encodes ego-node identity across sampled mini-batches.

- **`port_emb_dim = 8`**
  Embedding dimension for port IDs, capturing in-/out-port structural information.

- **`minority_class_weight = "auto"`**
  Class-weighting strategy. When set to _auto_, the loss function computes per-task positive-class weights from the training labels.

All configurations are available in `configs/pna_configs.json`.

## PNA Training Under Federated Setting

### Fully-Local Federated Baseline

Before evaluating federated learning algorithms, a **fully-local baseline** is used to establish a lower-bound reference point. In this setting, each client trains its own PNA model **independently** on its local subgraph partition, with **no parameter sharing** across clients.

This experiment answers the question: _how well can a model trained on a client's local subgraph alone perform, without any cross-client coordination?_ The resulting per-client test F1 scores serve as a lower bound against which all federated algorithms can be compared.

Key properties of this baseline:

- Each client builds its own PNA model with **local degree histograms** and **local port vocabulary sizes** using entirely its own subgraph.
- During training, **remote nodes** (owned by other clients but present in the subgraph due to cross-client edges) participate in **message passing** but are excluded from the loss computation.
- During evaluation, only **owned nodes** are scored; remote nodes contribute only as neighbors for neighborhood sampling.

To run the fully-local baseline:

```bash
python3 -m scripts.training.train_local_baseline
```

**Outputs:**

- `results/metrics/federated_logs/local_baseline/{num_clients}_clients/client_{id}/` — per-epoch train/val loss and F1 per client
- `checkpoints/local_baseline/{num_clients}_clients/client_{id}_seed{seed}_{run_id}_best.pt` — best checkpoint (by val PR-AUC) per client per seed, namespaced by client-count to prevent cross-experiment contamination
- `results/metrics/federated_logs/local_baseline_results.csv` — aggregated mean ± std test F1 per task across all clients and seeds
- `results/metrics/federated_logs/local_baseline_pr_auc_results.csv` — same, but for PR-AUC

---

### Standard FedAvg

The **standard FedAvg** training script implements the classic federated averaging algorithm: in each communication round, a random fraction of clients perform local gradient updates and send their model weights to the server, which aggregates them via weighted averaging and broadcasts the updated global model back.

Unlike the fully-local baseline, all clients share a single **global port vocabulary size**, computed from the union of all client training data. This ensures that port embedding tables have compatible shapes for weight aggregation.

To run standard FedAvg:

```bash
python3 -m scripts.training.train_fedavg
```

**Outputs:**

- `results/metrics/federated_logs/fedavg/` — per-epoch train/val loss and F1 logs per seed
- `checkpoints/fedavg/seed{seed}_{run_id}_best.pt` — best global model checkpoint per seed
- `results/metrics/federated_logs/fedavg_results.csv` — aggregated mean ± std test F1 per task
- `results/metrics/federated_logs/fedavg_pr_auc_results.csv` — same, but for PR-AUC

---

### Sync-SGD

A more synchronized alternative to FedAvg: all clients share one model and one Adam state. Each step, every client backpropagates a sample-weighted loss into the shared parameters and a single `optimizer.step()` applies the averaged gradient.

```bash
python3 -m scripts.training.train_sync_sgd
```

---

### Layer-wise Embedding Exchange

Synchronous per-step exchange of intermediate hidden representations across clients via a shared `EmbeddingTable`. At every conv layer, each client writes its owned-node embeddings to the table and replaces ghost-node positions with the freshly written values from other clients, so each client's forward sees the full graph one layer at a time.

Three variants by parameter-aggregation strategy:

```bash
# (a) Per-epoch FedAvg aggregation
python3 -m scripts.training.train_layerwise_exchange

# (b) No parameter aggregation (isolates the value of embedding exchange)
python3 -m scripts.training.train_layerwise_exchange_local

# (c) Per-step Sync-SGD with persistent cache and OptimES-style pre-training
python3 -m scripts.training.train_layerwise_exchange_sync_sgd
```

---

### Per-Epoch Layer-wise Embedding Exchange

OptimES-style ablation (Naman & Simmhan 2025): clients exchange `h^1..h^{L-1}` only at epoch boundaries via two `EmbeddingTable` instances (`cache_in` / `cache_out`) with an end-of-epoch push pass and a swap. Inside an epoch the cache is frozen and gradient-detached.

```bash
python3 -m scripts.training.train_layerwise_exchange_per_epoch
```

---

### Federated Learning Configuration

The federated setting introduces additional hyperparameters governing both the **federated splits generation** and the **federated training procedure**. This section documents the default configuration used throughout the experiments, along with a brief rationale for each choice.

#### Federated Training Hyperparameters

- **`partition_strategy`**
  Selects the partitioning strategy used in the experiment.
  Supported options:
  `"partition aware"`, `"metis original"`, `"louvain original"`, `"metis original skewed"`, `"louvain original skewed"`, `"metis imbalance"`, `"louvain imbalance"`.

- **`global_epochs = 100`**
  The total number of global communication rounds.

- **`local_epochs = 1`**
  Number of local training epochs performed by each client per communication round. A single local epoch is used by default to limit client drift and emphasize the effects of graph partitioning.

- **`client_fraction = 0.5`**
  In each communication round, a randomly sampled 50% of clients participate in training.

- **`algorithm`**
  Specifies the federated learning algorithm used in the experiment.
  Currently supported options:
  `"fedavg"`, `"fedprox"`, `"scaffold"`.

---

#### Hyperparameters for Partition-Aware Splits

- **`num_clients = 3`**
  The default configuration splits the global graph into 3 clients under the pattern-aware partitioning scheme.

- **`include_cross_edges = true`**
  Enables the inclusion of cross-client edges when constructing client subgraphs.
  This allows clients to observe edges connecting to remote nodes owned by other clients, which is essential for studying **cross-client communication**.

- **`use_local_labels = false`**
  When set to `true`, training uses the `_local_labels` splits where each client's node labels are recomputed from its local subgraph topology instead of the global graph.

---

#### Hyperparameters for Louvain- and Metis-Based Splits

- **`num_clients = 32`**
  The 8192-node global graph is partitioned into 32 subgraphs, yielding approximately 256 nodes per client.

- **`louvain_resolution = 1.0`**
  Uses the default modularity resolution for Louvain community detection.

- **`metis_num_coms = 32`**
  The Metis partitioning strategy is configured to produce exactly 32 partitions, ensuring that **each client corresponds to one contiguous graph community**, which maximizes structural separation between clients.

All configurations are available in `configs/fed_configs.json`.

## Reproducibility

All datasets, federated splits, and training results in this repository are **fully reproducible**.

The entire pipeline uses a shared seed-derivation mechanism:

- A global `BASE_SEED` is defined in the config.
- Each script calls `set_seed(BASE_SEED)`.
- Task-specific seeds (e.g., `"train"`, `"val"`, `"louvain"`, `"metis"`) are deterministically obtained via `derive_seed(BASE_SEED, tag)`.

This ensures:

- **Synthetic graphs** (`train.pt`, `val.pt`, `test.pt`) are identical across runs.
- **Federated subgraph partitions** (Metis/Louvain, equal-sized, Zipf-skewed, LIS) are reproduced exactly.
- **Training runs** (both centralized and federated) are stable and repeatable, including model initialization, mini-batch sampling, and client sampling.

Changing the `BASE_SEED` produces a new, independent experiment instance while preserving internal consistency across all components.
