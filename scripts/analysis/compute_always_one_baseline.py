"""
Compute PR-AUC for an always-predict-1 (constant positive) baseline, and
report per-client label distributions (% positive per task).

Analytically, the always-predict-1 PR-AUC equals the minority-class prevalence
per task, mirroring the minority-class-aware logic in
utils/metrics.py:compute_pr_auc_per_task.

Two settings:
  - Global labels: evaluated on the full test set (data/test.pt) and on each
                   client's owned nodes using global labels
  - Local labels:  per-client owned nodes using labels recomputed from each
                   client's local subgraph (_local_labels splits)

Usage:
    python3 -m scripts.analysis.compute_always_one_baseline
"""

import os
import csv
from datetime import datetime

import torch

TASKS = ["deg-in", "deg-out", "fan-in", "fan-out", "C2", "C3", "C4", "C5", "C6", "S-G", "B-C"]
DATA_DIR = "./data"
RESULTS_DIR = "./results/metrics/always_one_baseline"
OUT_PRAUC_CSV = os.path.join(RESULTS_DIR, "always_one_baseline_pr_auc_results.csv")
OUT_DIST_CSV  = os.path.join(RESULTS_DIR, "label_distributions.csv")


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def minority_class_prevalence(y: torch.Tensor) -> torch.Tensor:
    """
    For each task column in y [N, T], return the minority-class prevalence,
    matching the logic in utils/metrics.py:compute_pr_auc_per_task.
    - If positive is minority (pos <= neg): returns pos / N
    - If negative is minority (neg < pos):  returns neg / N
    Returns a [T] float64 tensor of baseline PR-AUC values.
    """
    N, T = y.shape
    scores = torch.zeros(T, dtype=torch.float64)
    for t in range(T):
        y_t = y[:, t].float()
        pos = int(y_t.sum().item())
        neg = N - pos
        if pos == 0 or neg == 0:
            scores[t] = 0.0
            continue
        scores[t] = pos / N if pos <= neg else neg / N
    return scores


def positive_ratio(y: torch.Tensor) -> torch.Tensor:
    """Return fraction of positive labels per task: [T] float64 tensor."""
    return y.float().mean(dim=0).double()


def load_client_y(clients_dir: str, num_clients: int) -> list[torch.Tensor]:
    """
    Load owned-node labels for each client.
    Returns a list of [N_owned, T] tensors (one per client).
    Missing clients are skipped with a warning.
    """
    ys = []
    for cid in range(num_clients):
        path = os.path.join(clients_dir, f"client_{cid:04d}.pt")
        if not os.path.exists(path):
            path = os.path.join(clients_dir, f"client_{cid}.pt")
        if not os.path.exists(path):
            print(f"  [warn] client {cid} not found in {clients_dir}")
            continue
        data = torch.load(path, weights_only=False)
        y = data.y.bool().float()
        if hasattr(data, "owned_mask") and data.owned_mask is not None:
            y = y[data.owned_mask]
        ys.append(y)
    return ys


def clients_dir_path(num_clients: int, variant: str, local_labels: bool) -> str:
    ll_suffix = "_local_labels" if local_labels else ""
    return os.path.join(
        DATA_DIR,
        f"fed_partition_aware_splits_{variant}{ll_suffix}",
        f"{num_clients}_clients",
        "test", "clients",
    )


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def print_prauc_table(label: str, scores: torch.Tensor) -> None:
    print(f"\n{'='*62}")
    print(f"  {label}")
    print(f"{'='*62}")
    print(f"  {'Task':<10} {'PR-AUC (always-1 baseline)':>30}")
    print(f"  {'-'*42}")
    for name, val in zip(TASKS, scores.tolist()):
        print(f"  {name:<10} {val*100:>29.2f}%")
    print(f"  {'-'*42}")
    print(f"  {'Macro':<10} {scores.mean().item()*100:>29.2f}%")


def print_dist_table(label: str, per_client_ratios: list[torch.Tensor]) -> None:
    """Print a table where rows = clients, columns = tasks (% positive)."""
    col_w = 8
    print(f"\n{'='*62}")
    print(f"  {label}")
    print(f"{'='*62}")
    header = f"  {'Client':<8}" + "".join(f"{t:>{col_w}}" for t in TASKS)
    print(header)
    print(f"  {'-'*(8 + col_w * len(TASKS))}")
    for cid, ratios in enumerate(per_client_ratios):
        row = f"  {cid:<8}" + "".join(f"{v*100:>{col_w-1}.1f}%" for v in ratios.tolist())
        print(row)
    # overall average row
    avg = torch.stack(per_client_ratios).mean(dim=0)
    print(f"  {'-'*(8 + col_w * len(TASKS))}")
    avg_row = f"  {'Avg':<8}" + "".join(f"{v*100:>{col_w-1}.1f}%" for v in avg.tolist())
    print(avg_row)


# ---------------------------------------------------------------------------
# PR-AUC baseline computations
# ---------------------------------------------------------------------------

def compute_global_baseline() -> torch.Tensor:
    test_data = torch.load(os.path.join(DATA_DIR, "test.pt"), weights_only=False)
    y = test_data.y.bool().float()
    scores = minority_class_prevalence(y)
    print_prauc_table("Global labels — always-predict-1 baseline (test.pt)", scores)
    return scores


def compute_client_baseline(
    num_clients: int, variant: str, local_labels: bool
) -> torch.Tensor | None:
    """Per-client minority-class prevalence averaged across clients."""
    cdir = clients_dir_path(num_clients, variant, local_labels)
    if not os.path.isdir(cdir):
        return None
    ys = load_client_y(cdir, num_clients)
    if not ys:
        return None
    client_scores = [minority_class_prevalence(y) for y in ys if y.shape[0] > 0]
    if not client_scores:
        return None
    avg = torch.stack(client_scores).mean(dim=0)
    ll_tag = "local labels" if local_labels else "global labels per client"
    print_prauc_table(
        f"Always-predict-1 baseline — {ll_tag} ({num_clients} clients, {variant})",
        avg,
    )
    return avg


# ---------------------------------------------------------------------------
# Label distribution computations
# ---------------------------------------------------------------------------

def compute_global_dist() -> torch.Tensor:
    """Positive ratio per task on the full test set."""
    test_data = torch.load(os.path.join(DATA_DIR, "test.pt"), weights_only=False)
    y = test_data.y.bool().float()
    ratios = positive_ratio(y)
    print_dist_table(
        "Label distribution — global test set (test.pt)",
        [ratios],
    )
    return ratios


def compute_client_dist(
    num_clients: int, variant: str, local_labels: bool
) -> list[torch.Tensor] | None:
    """Per-client positive ratio per task (owned nodes only)."""
    cdir = clients_dir_path(num_clients, variant, local_labels)
    if not os.path.isdir(cdir):
        return None
    ys = load_client_y(cdir, num_clients)
    if not ys:
        return None
    ratios = [positive_ratio(y) for y in ys if y.shape[0] > 0]
    if not ratios:
        return None
    ll_tag = "local labels" if local_labels else "global labels per client"
    print_dist_table(
        f"Label distribution — {ll_tag} ({num_clients} clients, {variant})",
        ratios,
    )
    return ratios


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def write_prauc_csv(rows: list[dict]) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fieldnames = (
        ["timestamp_iso", "setting", "macro_mean_prauc"]
        + [f"{t}_mean_prauc" for t in TASKS]
    )
    write_header = not os.path.exists(OUT_PRAUC_CSV)
    with open(OUT_PRAUC_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nPR-AUC baseline results appended to {OUT_PRAUC_CSV}")


def write_dist_csv(rows: list[dict]) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fieldnames = (
        ["timestamp_iso", "setting", "num_clients", "client_id"]
        + [f"{t}_pos_pct" for t in TASKS]
    )
    write_header = not os.path.exists(OUT_DIST_CSV)
    with open(OUT_DIST_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Label distribution results appended to {OUT_DIST_CSV}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    now = datetime.now().isoformat(timespec="seconds")
    prauc_rows: list[dict] = []
    dist_rows:  list[dict] = []

    variants         = ["with_cross_edges"]
    num_clients_list = [3, 5, 10, 15]

    # ---- Global test set ----
    global_scores = compute_global_baseline()
    prauc_rows.append({
        "timestamp_iso":   now,
        "setting":         "global_labels",
        "macro_mean_prauc": round(global_scores.mean().item() * 100, 2),
        **{f"{t}_mean_prauc": round(v * 100, 2)
           for t, v in zip(TASKS, global_scores.tolist())},
    })

    global_dist = compute_global_dist()
    dist_rows.append({
        "timestamp_iso": now,
        "setting":       "global_test_set",
        "num_clients":   "N/A",
        "client_id":     "all",
        **{f"{t}_pos_pct": round(v * 100, 2)
           for t, v in zip(TASKS, global_dist.tolist())},
    })

    # ---- Per-client splits ----
    for num_clients in num_clients_list:
        for variant in variants:
            for local_labels in [False, True]:
                cdir = clients_dir_path(num_clients, variant, local_labels)
                if not os.path.isdir(cdir):
                    continue

                ll_tag = "local_labels" if local_labels else "global_labels_per_client"
                setting = f"{ll_tag}_{num_clients}clients_{variant}"

                # PR-AUC baseline
                scores = compute_client_baseline(num_clients, variant, local_labels)
                if scores is not None:
                    prauc_rows.append({
                        "timestamp_iso":    now,
                        "setting":          setting,
                        "macro_mean_prauc": round(scores.mean().item() * 100, 2),
                        **{f"{t}_mean_prauc": round(v * 100, 2)
                           for t, v in zip(TASKS, scores.tolist())},
                    })

                # Label distribution
                ratios = compute_client_dist(num_clients, variant, local_labels)
                if ratios is not None:
                    for cid, r in enumerate(ratios):
                        dist_rows.append({
                            "timestamp_iso": now,
                            "setting":       setting,
                            "num_clients":   num_clients,
                            "client_id":     cid,
                            **{f"{t}_pos_pct": round(v * 100, 2)
                               for t, v in zip(TASKS, r.tolist())},
                        })

    write_prauc_csv(prauc_rows)
    write_dist_csv(dist_rows)


if __name__ == "__main__":
    main()
