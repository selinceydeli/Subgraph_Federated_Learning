"""Plot train_loss, val_loss, and a configurable C2 validation metric across epochs from a fedavg log CSV."""

import argparse
import pandas as pd
import matplotlib.pyplot as plt


"""
How to run:

# default (val_C2_minF1):
python3 -m scripts.analysis.plot_fedavg_training_curves

# use PR-AUC instead of minF1:
python3 -m scripts.analysis.plot_fedavg_training_curves --c2_metric pr_auc

# only show train/val loss (no C2 metric):
python3 -m scripts.analysis.plot_fedavg_training_curves --no_c2

# or point to a different log:
python3 -m scripts.analysis.plot_fedavg_training_curves --log <path> --output fig.png
"""

def main():
    parser = argparse.ArgumentParser(description="Plot training curves from fedavg log CSV.")
    parser.add_argument(
        "--log",
        default="results/metrics/federated_logs/fedavg/20260319-191011_fedavg_seed0_seed0.csv",
        help="Path to the fedavg log CSV file.",
    )
    parser.add_argument("--output", default=None, help="Path to save the figure (optional).")
    parser.add_argument(
        "--c2_metric",
        choices=["minF1", "pr_auc"],
        default="minF1",
        help="C2 validation metric to plot: 'minF1' (default) or 'pr_auc'.",
    )
    parser.add_argument(
        "--no_c2",
        action="store_true",
        help="Only plot train_loss and val_loss; omit the C2 metric line.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.log, comment="#")

    _, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(df["epoch"], df["train_loss"], label="train_loss", color="tab:blue")
    ax1.plot(df["epoch"], df["val_loss"], label="val_loss", color="tab:orange")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    if not args.no_c2:
        c2_col = f"val_C2_{args.c2_metric}"
        if c2_col not in df.columns:
            raise ValueError(f"Column '{c2_col}' not found in {args.log}. Available: {list(df.columns)}")
        ax2 = ax1.twinx()
        ax2.plot(df["epoch"], df[c2_col], label=c2_col, color="tab:green", linestyle="--")
        ax2.set_ylabel(c2_col)
        lines2, labels2 = ax2.get_legend_handles_labels()
    else:
        lines2, labels2 = [], []

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title("FedAvg Training Curves")
    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150)
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
