# Federated_Graph_Learning

A repository for **synthetic subgraph-detection** benchmarking and **PNA** baselines on directed multigraphs.

## Synthetic Graph Generation

This repository includes a **synthetic subgraph-detection dataset** used for benchmarking graph models for the pattern detection task. The graphs and labels are generated following the pseudocode/configurations described in **_Provably Powerful Graph Neural Networks for Directed Multigraphs_** (Egressy et al.).

### What’s Included

- Three splits: **train**, **val**, **test**
- Saved as PyTorch tensors under `./data/`:

  - `train.pt`, `val.pt`, `test.pt` objects with node-level labels
  - `y_sums.csv` — per-split counts of positive labels per sub-task

- Per-split label percentages and mean across splits are stored under `./results/metrics/`, useful to sanity-check against the paper’s reported marginals

---

### Label Tasks

Each node is labeled for the presence of the following patterns (11 sub-tasks):

- `deg_in > 3`
- `deg_out > 3`
- `fan_in > 3`
- `fan_out > 3`
- `cycle2`
- `cycle3`
- `cycle4`
- `cycle5`
- `cycle6`
- `scatter_gather`
- `biclique`

---

### Reproducibility

Graph instances are **reproducible**. A single `BASE_SEED` deterministically derives distinct seeds for each split (train/val/test), ensuring:

- different graphs **within** a run for the splits,
- identical graphs **across** runs with the same `BASE_SEED`.

---

### Default Generation Settings

The default config (see the generator script `scripts/generate_synthetic.py`) follows the paper’s setup:

- Nodes `n = 8192`
- Average degree `d = 6`
- Radius parameter `r = 11.1`
- Directed multigraphs (for directed cycles)
- Generator: “chordal” / random-circulant-like
- One connected component per split (prevents data leakage)

---

### How to Generate

From the repo root:

```bash
# Generate graphs and labels
python3 -m scripts.generate_synthetic
```

After step (1), you’ll find `train.pt`, `val.pt`, `test.pt`, and `y_sums.csv` under `./data/`. The `label_percentages.csv` will be saved under `./results/metrics/`.

## Principal Neighborhood Aggregation (PNA)

This repository provides two implementations of the **Principal Neighborhood Aggregation (PNA)** model, one baseline version using standard message passing, and an enhanced version that incorporates **Reverse Message Passing**, **Ego IDs**, **Port IDs**, and **mini-batch neighborhood sampling** for scalable training.

Both implementations follow the PNA architecture introduced in
**_Principal Neighbourhood Aggregation for Graph Nets_** (Corso et al., 2020).

### 1. Baseline PNA (Full-Batch Training)

The baseline model uses the original PNAConv layers from PyTorch Geometric and is trained in the **full-batch** setting.

To train and evaluate the baseline model:

```bash
python3 -m scripts.train_pna_baseline
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
- **Configurable fanout per hop** (default: `[10, 4]`)

To train and evaluate this model:

```bash
python3 -m scripts.train_pna_reverse_mp_with_ego
```

This version serves as the foundation for future **federated** extensions.

---

### 3. Training Configuration

Both PNA variants share the same core hyperparameters:

- hidden dimension: **64**
- number of GNN layers: **2**
- message-passing hops per layer: **2**
- dropout: **0.1**
- learning rate: **0.001**
- weight decay: **0.0001**

Additional hyperparameters apply to the extended model:

- mini-batch size: **32**
- neighbors per hop (fanout): **[10, 4]**
- ego-embedding dimension: **32**
- port-embedding dimension: **8**

All configurations are available under:

```
configs/pna_configs.json
```
