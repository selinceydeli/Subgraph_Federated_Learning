#!/usr/bin/env python3
import os
import glob
import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


"""
How to run this script:
- python -m scripts.analysis.analyze_client_label_shift --with-cross-edges
- python -m scripts.analysis.analyze_client_label_shift --without-cross-edges
"""


TASK_NAMES = [
    "deg_in>3",
    "deg_out>3",
    "fan_in>3",
    "fan_out>3",
    "cycle2",
    "cycle3",
    "cycle4",
    "cycle5",
    "cycle6",
    "scatter_gather",
    "biclique",
]


def safe_mkdir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def list_client_files(split_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(split_dir, "client_*.pt")))
    return files


def load_client_graph(path: str):
    return torch.load(path, map_location="cpu")


def compute_label_prevalence(y: torch.Tensor) -> np.ndarray:
    """
    y: [num_nodes, num_tasks], binary 0/1
    returns per-task percentage in [0, 100]
    """
    if y.ndim != 2:
        raise ValueError(f"Expected y to be 2D, got shape {tuple(y.shape)}")
    num_nodes = y.shape[0]
    if num_nodes == 0:
        return np.zeros(y.shape[1], dtype=float)
    prev = y.float().mean(dim=0).cpu().numpy() * 100.0
    return prev


def normalize_distribution(prevalence_pct: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Convert prevalence vector into a probability distribution over tasks.
    If all-zero, return uniform distribution.
    """
    x = np.asarray(prevalence_pct, dtype=float)
    x = np.clip(x, 0.0, None)
    s = x.sum()
    if s <= eps:
        return np.ones_like(x) / len(x)
    return x / s


def js_divergence_from_prevalence(a_pct: np.ndarray, b_pct: np.ndarray, eps: float = 1e-12) -> float:
    """
    Jensen-Shannon divergence between two task distributions
    induced by prevalence vectors.
    """
    p = normalize_distribution(a_pct, eps=eps)
    q = normalize_distribution(b_pct, eps=eps)
    m = 0.5 * (p + q)

    def kl(x, y):
        mask = x > eps
        return np.sum(x[mask] * np.log2(x[mask] / np.clip(y[mask], eps, None)))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def l1_shift(a_pct: np.ndarray, b_pct: np.ndarray) -> float:
    return float(np.mean(np.abs(a_pct - b_pct)))


def l2_shift(a_pct: np.ndarray, b_pct: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a_pct - b_pct) ** 2)))


def max_shift(a_pct: np.ndarray, b_pct: np.ndarray) -> float:
    return float(np.max(np.abs(a_pct - b_pct)))


def client_id_from_path(path: str) -> int:
    name = os.path.basename(path)
    # client_0000.pt -> 0
    stem = os.path.splitext(name)[0]
    return int(stem.split("_")[-1])


def load_split_client_stats(split_clients_dir: str, split_name: str) -> Dict[int, Dict]:
    """
    Returns:
      client_id -> {
          "num_nodes": int,
          "prevalence": np.ndarray shape [num_tasks]
      }
    """
    stats = {}
    files = list_client_files(split_clients_dir)

    for fp in files:
        cid = client_id_from_path(fp)
        gd = load_client_graph(fp)

        if not hasattr(gd, "y"):
            raise ValueError(f"{fp} does not contain attribute 'y'.")

        y = gd.y
        if y.shape[1] != len(TASK_NAMES):
            raise ValueError(
                f"{fp}: expected {len(TASK_NAMES)} tasks, got {y.shape[1]}. "
                f"Please update TASK_NAMES if your task list changed."
            )

        prev = compute_label_prevalence(y)
        stats[cid] = {
            "split": split_name,
            "num_nodes": int(y.shape[0]),
            "prevalence": prev,
        }

    return stats


def merge_client_splits(
    train_stats: Dict[int, Dict],
    val_stats: Dict[int, Dict],
    test_stats: Dict[int, Dict],
    num_clients: int,
) -> pd.DataFrame:
    rows = []

    all_client_ids = sorted(set(train_stats) | set(val_stats) | set(test_stats))

    for cid in all_client_ids:
        tr = train_stats.get(cid)
        va = val_stats.get(cid)
        te = test_stats.get(cid)

        if tr is None or va is None or te is None:
            # keep row anyway, but mark missing
            rows.append({
                "client_id": cid,
                "num_clients_setting": num_clients,
                "missing_split": True,
            })
            continue

        tr_prev = tr["prevalence"]
        va_prev = va["prevalence"]
        te_prev = te["prevalence"]

        row = {
            "client_id": cid,
            "num_clients_setting": num_clients,
            "missing_split": False,
            "train_num_nodes": tr["num_nodes"],
            "val_num_nodes": va["num_nodes"],
            "test_num_nodes": te["num_nodes"],

            "train_val_js": js_divergence_from_prevalence(tr_prev, va_prev),
            "train_test_js": js_divergence_from_prevalence(tr_prev, te_prev),
            "val_test_js": js_divergence_from_prevalence(va_prev, te_prev),

            "train_val_l1": l1_shift(tr_prev, va_prev),
            "train_test_l1": l1_shift(tr_prev, te_prev),
            "val_test_l1": l1_shift(va_prev, te_prev),

            "train_val_l2": l2_shift(tr_prev, va_prev),
            "train_test_l2": l2_shift(tr_prev, te_prev),
            "val_test_l2": l2_shift(va_prev, te_prev),

            "train_val_max": max_shift(tr_prev, va_prev),
            "train_test_max": max_shift(tr_prev, te_prev),
            "val_test_max": max_shift(va_prev, te_prev),
        }

        for i, task in enumerate(TASK_NAMES):
            row[f"train_{task}"] = tr_prev[i]
            row[f"val_{task}"] = va_prev[i]
            row[f"test_{task}"] = te_prev[i]
            row[f"tv_absdiff_{task}"] = abs(tr_prev[i] - va_prev[i])
            row[f"tt_absdiff_{task}"] = abs(tr_prev[i] - te_prev[i])
            row[f"vt_absdiff_{task}"] = abs(va_prev[i] - te_prev[i])

        rows.append(row)

    return pd.DataFrame(rows)


def plot_prevalence_heatmap(df: pd.DataFrame, split_name: str, save_path: str):
    """
    Heatmap: rows = clients, cols = tasks, values = prevalence %
    """
    split_cols = [f"{split_name}_{t}" for t in TASK_NAMES]
    available = df[df["missing_split"] == False].sort_values("client_id")

    if available.empty:
        return

    mat = available[split_cols].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(12, max(4, 0.5 * len(available))))
    im = ax.imshow(mat, aspect="auto")

    ax.set_xticks(np.arange(len(TASK_NAMES)))
    ax.set_xticklabels(TASK_NAMES, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(available)))
    ax.set_yticklabels([f"client_{cid:04d}" for cid in available["client_id"].tolist()])
    ax.set_title(f"{split_name.capitalize()} label prevalence per client (%)")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Prevalence (%)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_shift_bars(df: pd.DataFrame, metric_col: str, title: str, save_path: str):
    available = df[df["missing_split"] == False].sort_values("client_id")
    if available.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(
        [f"c{cid}" for cid in available["client_id"]],
        available[metric_col].to_numpy(dtype=float),
    )
    ax.set_title(title)
    ax.set_xlabel("Client")
    ax.set_ylabel(metric_col)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_top_shifted_tasks_for_client(row: pd.Series, pair_prefix: str, save_path: str, top_k: int = 11):
    if pair_prefix == "train_val":
        diff_prefix = "tv_absdiff_"
    elif pair_prefix == "train_test":
        diff_prefix = "tt_absdiff_"
    elif pair_prefix == "val_test":
        diff_prefix = "vt_absdiff_"
    else:
        raise ValueError(f"Unknown pair_prefix: {pair_prefix}")

    vals = [(t, float(row[f"{diff_prefix}{t}"])) for t in TASK_NAMES]
    vals = sorted(vals, key=lambda x: x[1], reverse=True)[:top_k]

    tasks = [x[0] for x in vals]
    diffs = [x[1] for x in vals]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(tasks, diffs)
    ax.set_title(
        f"Top task shifts for client {int(row['client_id'])} ({pair_prefix.replace('_', ' vs ')})"
    )
    ax.set_ylabel("Absolute prevalence difference (%)")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def summarize_across_client_counts(all_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for df in all_dfs:
        available = df[df["missing_split"] == False]
        if available.empty:
            continue

        k = int(available["num_clients_setting"].iloc[0])
        rows.append({
            "num_clients_setting": k,
            "num_available_clients": len(available),

            "mean_train_val_js": available["train_val_js"].mean(),
            "mean_train_test_js": available["train_test_js"].mean(),
            "mean_val_test_js": available["val_test_js"].mean(),

            "mean_train_val_l1": available["train_val_l1"].mean(),
            "mean_train_test_l1": available["train_test_l1"].mean(),
            "mean_val_test_l1": available["val_test_l1"].mean(),

            "max_train_val_l1": available["train_val_l1"].max(),
            "max_train_test_l1": available["train_test_l1"].max(),
            "max_val_test_l1": available["val_test_l1"].max(),
        })

    return pd.DataFrame(rows).sort_values("num_clients_setting")


def analyze_one_setting(root_dir: str, num_clients: int, out_dir: str) -> pd.DataFrame:
    setting_dir = os.path.join(root_dir, f"{num_clients}_clients")
    train_dir = os.path.join(setting_dir, "train", "clients")
    val_dir = os.path.join(setting_dir, "val", "clients")
    test_dir = os.path.join(setting_dir, "test", "clients")

    if not (os.path.isdir(train_dir) and os.path.isdir(val_dir) and os.path.isdir(test_dir)):
        raise FileNotFoundError(
            f"Could not find expected split dirs for {num_clients} clients under:\n{setting_dir}"
        )

    train_stats = load_split_client_stats(train_dir, "train")
    val_stats = load_split_client_stats(val_dir, "val")
    test_stats = load_split_client_stats(test_dir, "test")

    df = merge_client_splits(train_stats, val_stats, test_stats, num_clients=num_clients)

    setting_out = os.path.join(out_dir, f"{num_clients}_clients")
    safe_mkdir(setting_out)

    detail_csv = os.path.join(setting_out, "client_label_shift_detailed.csv")
    df.to_csv(detail_csv, index=False)

    plot_prevalence_heatmap(df, "train", os.path.join(setting_out, "heatmap_train.png"))
    plot_prevalence_heatmap(df, "val", os.path.join(setting_out, "heatmap_val.png"))
    plot_prevalence_heatmap(df, "test", os.path.join(setting_out, "heatmap_test.png"))

    plot_shift_bars(
        df,
        "train_val_l1",
        f"Train vs Val label shift per client ({num_clients} clients)",
        os.path.join(setting_out, "shift_train_val_l1.png"),
    )
    plot_shift_bars(
        df,
        "train_test_l1",
        f"Train vs Test label shift per client ({num_clients} clients)",
        os.path.join(setting_out, "shift_train_test_l1.png"),
    )
    plot_shift_bars(
        df,
        "val_test_l1",
        f"Val vs Test label shift per client ({num_clients} clients)",
        os.path.join(setting_out, "shift_val_test_l1.png"),
    )

    plot_shift_bars(
        df,
        "train_val_js",
        f"Train vs Val JS divergence per client ({num_clients} clients)",
        os.path.join(setting_out, "shift_train_val_js.png"),
    )
    plot_shift_bars(
        df,
        "train_test_js",
        f"Train vs Test JS divergence per client ({num_clients} clients)",
        os.path.join(setting_out, "shift_train_test_js.png"),
    )
    plot_shift_bars(
        df,
        "val_test_js",
        f"Val vs Test JS divergence per client ({num_clients} clients)",
        os.path.join(setting_out, "shift_val_test_js.png"),
    )

    # Save top shifted clients
    available = df[df["missing_split"] == False].copy()
    if not available.empty:
        worst_tv = available.sort_values("train_val_l1", ascending=False).iloc[0]
        worst_tt = available.sort_values("train_test_l1", ascending=False).iloc[0]
        worst_vt = available.sort_values("val_test_l1", ascending=False).iloc[0]

        plot_top_shifted_tasks_for_client(
            worst_tv,
            "train_val",
            os.path.join(setting_out, f"top_shift_tasks_client_{int(worst_tv['client_id']):04d}_train_val.png"),
        )
        plot_top_shifted_tasks_for_client(
            worst_tt,
            "train_test",
            os.path.join(setting_out, f"top_shift_tasks_client_{int(worst_tt['client_id']):04d}_train_test.png"),
        )
        plot_top_shifted_tasks_for_client(
            worst_vt,
            "val_test",
            os.path.join(setting_out, f"top_shift_tasks_client_{int(worst_vt['client_id']):04d}_val_test.png"),
        )

    return df


def main():
    parser = argparse.ArgumentParser(description="Analyze per-client label distribution shift across train/val/test.")
    parser.add_argument(
        "--with-cross-edges",
        action="store_true",
        help="Use fed_partition_aware_splits_with_cross_edges as input root."
    )
    parser.add_argument(
        "--without-cross-edges",
        action="store_true",
        help="Use fed_partition_aware_splits_without_cross_edges as input root."
    )
    parser.add_argument(
        "--client-counts",
        type=int,
        nargs="+",
        default=[3, 5, 10, 15],
        help="Client counts to analyze."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="Base data directory."
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="./results/label_shift_analysis",
        help="Where to save CSVs and plots."
    )

    args = parser.parse_args()

    if args.with_cross_edges and args.without_cross_edges:
        raise ValueError("Choose only one of --with-cross-edges or --without-cross-edges.")

    if args.with_cross_edges:
        input_root = os.path.join(args.data_root, "fed_partition_aware_splits_with_cross_edges")
        mode_name = "with_cross_edges"
    elif args.without_cross_edges:
        input_root = os.path.join(args.data_root, "fed_partition_aware_splits_without_cross_edges")
        mode_name = "without_cross_edges"
    else:
        # default
        input_root = os.path.join(args.data_root, "fed_partition_aware_splits_with_cross_edges")
        mode_name = "with_cross_edges"

    out_dir = os.path.join(args.output_root, mode_name)
    safe_mkdir(out_dir)

    all_dfs = []
    for k in args.client_counts:
        print(f"[INFO] Analyzing {k} clients from: {input_root}")
        df = analyze_one_setting(input_root, num_clients=k, out_dir=out_dir)
        all_dfs.append(df)

    summary_df = summarize_across_client_counts(all_dfs)
    summary_csv = os.path.join(out_dir, "summary_across_client_counts.csv")
    summary_df.to_csv(summary_csv, index=False)

    # Summary plot across k
    if not summary_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = summary_df["num_clients_setting"].to_numpy(dtype=int)

        ax.plot(x, summary_df["mean_train_val_l1"], marker="o", label="train vs val")
        ax.plot(x, summary_df["mean_train_test_l1"], marker="o", label="train vs test")
        ax.plot(x, summary_df["mean_val_test_l1"], marker="o", label="val vs test")

        ax.set_xlabel("Number of clients")
        ax.set_ylabel("Mean L1 label shift (%)")
        ax.set_title("Average client-level label shift vs number of clients")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "summary_mean_l1_vs_num_clients.png"), dpi=200, bbox_inches="tight")
        plt.close()

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(x, summary_df["mean_train_val_js"], marker="o", label="train vs val")
        ax.plot(x, summary_df["mean_train_test_js"], marker="o", label="train vs test")
        ax.plot(x, summary_df["mean_val_test_js"], marker="o", label="val vs test")

        ax.set_xlabel("Number of clients")
        ax.set_ylabel("Mean JS divergence")
        ax.set_title("Average task-distribution divergence vs number of clients")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "summary_mean_js_vs_num_clients.png"), dpi=200, bbox_inches="tight")
        plt.close()

    print(f"[INFO] Done. Results saved under: {out_dir}")


if __name__ == "__main__":
    main()