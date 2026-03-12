#!/usr/bin/env python3
import os
import glob
import argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
How to run this script:
- python -m scripts.analysis.analyze_cycle2_shift --with-cross-edges
- python -m scripts.analysis.analyze_cycle2_shift --without-cross-edges
- python -m scripts.analysis.analyze_cycle2_shift --with-cross-edges --client-counts 3 5 10 15
"""

CYCLE2_INDEX = 4  # task index in y
DEFAULT_CLIENT_COUNTS = [3, 5, 10, 15]


def load_cycle2_prevalence_from_graphdata(gd, owned_only=True):
    if not hasattr(gd, "y"):
        raise ValueError("GraphData object does not contain attribute 'y'.")

    y = gd.y

    if owned_only and hasattr(gd, "owned_mask"):
        y = y[gd.owned_mask]

    if y.ndim != 2:
        raise ValueError(f"Expected y to be 2D, got shape {tuple(y.shape)}")

    if y.shape[0] == 0:
        return 0.0

    if CYCLE2_INDEX >= y.shape[1]:
        raise ValueError(
            f"CYCLE2_INDEX={CYCLE2_INDEX} is out of bounds for y with shape {tuple(y.shape)}"
        )

    cycle2 = y[:, CYCLE2_INDEX]
    return float(cycle2.float().mean().item() * 100.0)


def load_cycle2_prevalence(client_path):
    gd = torch.load(client_path, map_location="cpu")
    return load_cycle2_prevalence_from_graphdata(gd, owned_only=True)


def load_split(split_dir):
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    files = sorted(glob.glob(os.path.join(split_dir, "client_*.pt")))
    if not files:
        raise FileNotFoundError(f"No client files found under: {split_dir}")

    res = {}
    for f in files:
        cid = int(os.path.basename(f).split("_")[-1].split(".")[0])
        res[cid] = load_cycle2_prevalence(f)

    return res


def analyze_setting(root_dir, k):
    train_dir = os.path.join(root_dir, f"{k}_clients", "train", "clients")
    val_dir = os.path.join(root_dir, f"{k}_clients", "val", "clients")
    test_dir = os.path.join(root_dir, f"{k}_clients", "test", "clients")

    train = load_split(train_dir)
    val = load_split(val_dir)
    test = load_split(test_dir)

    rows = []
    all_clients = sorted(set(train.keys()) | set(val.keys()) | set(test.keys()))

    for cid in all_clients:
        tr = train.get(cid, np.nan)
        va = val.get(cid, np.nan)
        te = test.get(cid, np.nan)

        rows.append(
            {
                "client": cid,
                "num_clients": k,
                "train_cycle2": tr,
                "val_cycle2": va,
                "test_cycle2": te,
                "train_val_shift": abs(tr - va) if pd.notna(tr) and pd.notna(va) else np.nan,
                "train_test_shift": abs(tr - te) if pd.notna(tr) and pd.notna(te) else np.nan,
                "val_test_shift": abs(va - te) if pd.notna(va) and pd.notna(te) else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values("client").reset_index(drop=True)


def analyze_centralized(data_root="./data"):
    train_path = os.path.join(data_root, "train.pt")
    val_path = os.path.join(data_root, "val.pt")
    test_path = os.path.join(data_root, "test.pt")

    for p in [train_path, val_path, test_path]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Centralized split file not found: {p}")

    train_gd = torch.load(train_path, map_location="cpu")
    val_gd = torch.load(val_path, map_location="cpu")
    test_gd = torch.load(test_path, map_location="cpu")

    train_prev = load_cycle2_prevalence_from_graphdata(train_gd, owned_only=False)
    val_prev = load_cycle2_prevalence_from_graphdata(val_gd, owned_only=False)
    test_prev = load_cycle2_prevalence_from_graphdata(test_gd, owned_only=False)

    df = pd.DataFrame(
        [
            {"split": "train", "cycle2_prevalence": train_prev},
            {"split": "val", "cycle2_prevalence": val_prev},
            {"split": "test", "cycle2_prevalence": test_prev},
        ]
    )

    return df


def plot_shift(df, k, out_dir):
    plt.figure(figsize=(8, 4.2))

    x = np.arange(len(df))
    width = 0.25

    plt.bar(x - width, df["train_val_shift"], width=width, label="train-val")
    plt.bar(x, df["train_test_shift"], width=width, label="train-test")
    plt.bar(x + width, df["val_test_shift"], width=width, label="val-test")

    plt.xlabel("Client")
    plt.ylabel("Cycle2 prevalence shift (%)")
    plt.title(f"Cycle2 distribution shift per client ({k} clients)")
    plt.xticks(x, df["client"].astype(int))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"cycle2_shift_{k}_clients.png"), dpi=200, bbox_inches="tight")
    plt.close()


def plot_prevalence(df, k, out_dir, centralized_df=None):
    plt.figure(figsize=(8, 4.2))

    x = df["client"].astype(int)

    plt.plot(x, df["train_cycle2"], marker="o", label="train")
    plt.plot(x, df["val_cycle2"], marker="o", label="val")
    plt.plot(x, df["test_cycle2"], marker="o", label="test")

    if centralized_df is not None:
        ctrain = float(centralized_df.loc[centralized_df["split"] == "train", "cycle2_prevalence"].iloc[0])
        cval = float(centralized_df.loc[centralized_df["split"] == "val", "cycle2_prevalence"].iloc[0])
        ctest = float(centralized_df.loc[centralized_df["split"] == "test", "cycle2_prevalence"].iloc[0])

        plt.axhline(ctrain, linestyle="--", linewidth=1.5, label="centralized train")
        plt.axhline(cval, linestyle="--", linewidth=1.5, label="centralized val")
        plt.axhline(ctest, linestyle="--", linewidth=1.5, label="centralized test")

    plt.xlabel("Client")
    plt.ylabel("Cycle2 prevalence (%)")
    plt.title(f"Cycle2 prevalence per client ({k} clients)")
    plt.xticks(x)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"cycle2_prevalence_{k}_clients.png"), dpi=200, bbox_inches="tight")
    plt.close()


def plot_cycle2_prevalence_bars(df, k, out_dir, centralized_df=None):
    plt.figure(figsize=(9, 4.5))

    x = np.arange(len(df))
    width = 0.25

    plt.bar(x - width, df["train_cycle2"], width=width, label="train")
    plt.bar(x, df["val_cycle2"], width=width, label="val")
    plt.bar(x + width, df["test_cycle2"], width=width, label="test")

    if centralized_df is not None:
        ctrain = float(centralized_df.loc[centralized_df["split"] == "train", "cycle2_prevalence"].iloc[0])
        cval = float(centralized_df.loc[centralized_df["split"] == "val", "cycle2_prevalence"].iloc[0])
        ctest = float(centralized_df.loc[centralized_df["split"] == "test", "cycle2_prevalence"].iloc[0])

        plt.axhline(ctrain, linestyle="--", linewidth=1.2, label="centralized train")
        plt.axhline(cval, linestyle="--", linewidth=1.2, label="centralized val")
        plt.axhline(ctest, linestyle="--", linewidth=1.2, label="centralized test")

    plt.xlabel("Client")
    plt.ylabel("Cycle2 prevalence (%)")
    plt.title(f"Absolute Cycle2 prevalence per client ({k} clients)")
    plt.xticks(x, df["client"].astype(int))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"cycle2_prevalence_bars_{k}_clients.png"), dpi=200, bbox_inches="tight")
    plt.close()


def plot_summary_shift(summary, out_dir):
    plt.figure(figsize=(7, 4.2))

    x = summary["clients"]

    plt.plot(x, summary["mean_train_val_shift"], marker="o", label="train-val")
    plt.plot(x, summary["mean_train_test_shift"], marker="o", label="train-test")
    plt.plot(x, summary["mean_val_test_shift"], marker="o", label="val-test")

    plt.xlabel("Number of clients")
    plt.ylabel("Mean cycle2 shift (%)")
    plt.title("Cycle2 distribution shift vs number of clients")
    plt.xticks(x)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cycle2_shift_vs_clients.png"), dpi=200, bbox_inches="tight")
    plt.close()


def plot_summary_prevalence(summary, out_dir, centralized_df=None):
    plt.figure(figsize=(7, 4.2))

    x = summary["clients"]

    plt.plot(x, summary["mean_train_cycle2"], marker="o", label="mean train")
    plt.plot(x, summary["mean_val_cycle2"], marker="o", label="mean val")
    plt.plot(x, summary["mean_test_cycle2"], marker="o", label="mean test")

    if centralized_df is not None:
        ctrain = float(centralized_df.loc[centralized_df["split"] == "train", "cycle2_prevalence"].iloc[0])
        cval = float(centralized_df.loc[centralized_df["split"] == "val", "cycle2_prevalence"].iloc[0])
        ctest = float(centralized_df.loc[centralized_df["split"] == "test", "cycle2_prevalence"].iloc[0])

        plt.axhline(ctrain, linestyle="--", linewidth=1.5, label="centralized train")
        plt.axhline(cval, linestyle="--", linewidth=1.5, label="centralized val")
        plt.axhline(ctest, linestyle="--", linewidth=1.5, label="centralized test")

    plt.xlabel("Number of clients")
    plt.ylabel("Mean Cycle2 prevalence (%)")
    plt.title("Mean Cycle2 prevalence vs number of clients")
    plt.xticks(x)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cycle2_mean_prevalence_vs_clients.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 4.2))
    plt.plot(x, summary["max_test_cycle2"], marker="o", label="max test")
    if centralized_df is not None:
        ctest = float(centralized_df.loc[centralized_df["split"] == "test", "cycle2_prevalence"].iloc[0])
        plt.axhline(ctest, linestyle="--", linewidth=1.5, label="centralized test")
    plt.xlabel("Number of clients")
    plt.ylabel("Max test Cycle2 prevalence (%)")
    plt.title("Maximum test Cycle2 prevalence vs number of clients")
    plt.xticks(x)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cycle2_max_test_prevalence_vs_clients.png"), dpi=200, bbox_inches="tight")
    plt.close()


def plot_centralized_prevalence(centralized_df, out_dir):
    plt.figure(figsize=(6.5, 4.2))

    x = np.arange(len(centralized_df))
    plt.bar(x, centralized_df["cycle2_prevalence"])

    plt.xticks(x, centralized_df["split"])
    plt.ylabel("Cycle2 prevalence (%)")
    plt.title("Centralized Cycle2 prevalence across train/val/test")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "centralized_cycle2_prevalence.png"), dpi=200, bbox_inches="tight")
    plt.close()


def plot_federated_vs_centralized(summary, centralized_df, out_dir):
    plt.figure(figsize=(7.5, 4.5))

    x = summary["clients"]

    plt.plot(x, summary["mean_train_cycle2"], marker="o", label="federated mean train")
    plt.plot(x, summary["mean_val_cycle2"], marker="o", label="federated mean val")
    plt.plot(x, summary["mean_test_cycle2"], marker="o", label="federated mean test")

    ctrain = float(centralized_df.loc[centralized_df["split"] == "train", "cycle2_prevalence"].iloc[0])
    cval = float(centralized_df.loc[centralized_df["split"] == "val", "cycle2_prevalence"].iloc[0])
    ctest = float(centralized_df.loc[centralized_df["split"] == "test", "cycle2_prevalence"].iloc[0])

    plt.axhline(ctrain, linestyle="--", linewidth=1.5, label="centralized train")
    plt.axhline(cval, linestyle="--", linewidth=1.5, label="centralized val")
    plt.axhline(ctest, linestyle="--", linewidth=1.5, label="centralized test")

    plt.xlabel("Number of clients")
    plt.ylabel("Cycle2 prevalence (%)")
    plt.title("Federated mean vs centralized Cycle2 prevalence")
    plt.xticks(x)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "federated_vs_centralized_cycle2_prevalence.png"), dpi=200, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Cycle2 prevalence and distribution shift across clients and centralized data."
    )
    parser.add_argument(
        "--with-cross-edges",
        action="store_true",
        help="Use partition-aware splits with cross edges.",
    )
    parser.add_argument(
        "--without-cross-edges",
        action="store_true",
        help="Use partition-aware splits without cross edges.",
    )
    parser.add_argument(
        "--client-counts",
        type=int,
        nargs="+",
        default=DEFAULT_CLIENT_COUNTS,
        help="Client-count settings to analyze. Default: 3 5 10 15",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="Base data directory containing centralized train.pt/val.pt/test.pt",
    )
    args = parser.parse_args()

    if args.with_cross_edges and args.without_cross_edges:
        raise ValueError("Choose only one of --with-cross-edges or --without-cross-edges.")

    if args.without_cross_edges:
        fed_root = os.path.join(args.data_root, "fed_partition_aware_splits_without_cross_edges")
        out_dir = "./results/cycle2_analysis/without_cross_edges"
    else:
        fed_root = os.path.join(args.data_root, "fed_partition_aware_splits_with_cross_edges")
        out_dir = "./results/cycle2_analysis/with_cross_edges"

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Centralized analysis
    centralized_df = analyze_centralized(data_root=args.data_root)
    centralized_df.to_csv(os.path.join(out_dir, "centralized_cycle2_prevalence.csv"), index=False)
    plot_centralized_prevalence(centralized_df, out_dir)

    all_rows = []

    for k in args.client_counts:
        print(f"Analyzing {k} clients from {fed_root}")

        df = analyze_setting(fed_root, k)
        df.to_csv(os.path.join(out_dir, f"cycle2_clients_{k}.csv"), index=False)

        plot_shift(df, k, out_dir)
        plot_prevalence(df, k, out_dir, centralized_df=centralized_df)
        plot_cycle2_prevalence_bars(df, k, out_dir, centralized_df=centralized_df)

        all_rows.append(
            {
                "clients": k,
                "mean_train_cycle2": df["train_cycle2"].mean(),
                "mean_val_cycle2": df["val_cycle2"].mean(),
                "mean_test_cycle2": df["test_cycle2"].mean(),
                "std_train_cycle2": df["train_cycle2"].std(),
                "std_val_cycle2": df["val_cycle2"].std(),
                "std_test_cycle2": df["test_cycle2"].std(),
                "max_test_cycle2": df["test_cycle2"].max(),
                "min_test_cycle2": df["test_cycle2"].min(),
                "mean_train_val_shift": df["train_val_shift"].mean(),
                "mean_train_test_shift": df["train_test_shift"].mean(),
                "mean_val_test_shift": df["val_test_shift"].mean(),
            }
        )

    summary = pd.DataFrame(all_rows).sort_values("clients").reset_index(drop=True)
    summary.to_csv(os.path.join(out_dir, "cycle2_summary.csv"), index=False)

    plot_summary_shift(summary, out_dir)
    plot_summary_prevalence(summary, out_dir, centralized_df=centralized_df)
    plot_federated_vs_centralized(summary, centralized_df, out_dir)

    print(f"Saved Cycle2 analysis outputs to: {out_dir}")


if __name__ == "__main__":
    main()