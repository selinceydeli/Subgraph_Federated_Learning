# Federated_Graph_Learning

A repository for **synthetic subgraph-detection** benchmarking and **PNA** baselines on directed multigraphs.

## Synthetic Graph Generation

This repository includes a **synthetic subgraph-detection dataset** used for benchmarking graph models for the pattern detection task. The graphs and labels are generated following the pseudocode/configurations described in _Provably Powerful Graph Neural Networks for Directed Multigraphs_ (Egressy et al.).

### What’s Included

- Three splits: **train**, **val**, **test**
- Saved as PyTorch tensors under `./data/`:

  - `train.pt`, `val.pt`, `test.pt` objects with node-level labels
  - `y_sums.csv` — per-split counts of positive labels per sub-task

- Per-split label percentages and mean across splits are stored under `./results/metrics/`, useful to sanity-check against the paper’s reported marginals

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

### Reproducibility

Graph instances are **reproducible**. A single `BASE_SEED` deterministically derives distinct seeds for each split (train/val/test), ensuring:

- different graphs **within** a run for the splits,
- identical graphs **across** runs with the same `BASE_SEED`.

### Default Generation Settings

The default config (see the generator script `scripts/generate_synthetic.py`) follows the paper’s setup:

- Nodes `n = 8192`
- Average degree `d = 6`
- Radius parameter `r = 11.1`
- Directed multigraphs (for directed cycles)
- Generator: “chordal” / random-circulant-like
- One connected component per split (prevents data leakage)

### How to Generate

From the repo root:

```bash
# Generate graphs and labels
python3 -m scripts.generate_synthetic
```

After step (1), you’ll find `train.pt`, `val.pt`, `test.pt`, and `y_sums.csv` under `./data/`. The `label_percentages.csv` will be saved under `./results/metrics/`.

---

## Principal Neighborhood Aggregation (PNA)

PNA model is implemented by following the model architecture described in _Principal Neighbourhood Aggregation for Graph Nets_ (Corso et al.).

### Train Baseline PNA

From the repo root:

```bash
# Train and test PNA model on the generated graph data
python3 -m scripts.train_pna_baseline
```

### Train PNA with Reverse Message Passing

Training the `PNA model with Reverse Message Passing`, which uses a `Heterogeneous Graph` and `Ego IDs` to detect the fraud patterns listed under [Label Tasks](#label-tasks).

From the repo root:

```bash
# Train and test enhanced PNA model on the generated graph data
python3 -m scripts.train_pna_reverse_mp_with_ego
```

### Hyperparameter Tuning

From the repo root:

```bash
python3 -m scripts.pna_hyperparameter_tuning
```
