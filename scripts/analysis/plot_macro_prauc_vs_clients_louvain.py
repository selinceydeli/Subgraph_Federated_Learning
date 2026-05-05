"""Plot line chart of Macro PR-AUC vs. number of clients across FL methods (Louvain split).

Methods compared (Louvain split with cross-client edges, 6-layer PNA):
  - Fully Local
  - FedAvg
  - OptimES (FedAvg + per-epoch layer-wise exchange, stale baseline)
  - FedAvg + Layer-wise Exchange (per-step, fresh)

Reference lines: Centralized PNA (upper bound) and Always-Positive classifier (naive baseline).

Usage:
    python3 -m scripts.analysis.plot_macro_prauc_vs_clients_louvain
    python3 -m scripts.analysis.plot_macro_prauc_vs_clients_louvain --output figures/macro_prauc_vs_clients_louvain.pdf
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt


CLIENTS = [3, 5, 10, 15]

MACRO_PRAUC = {
    "Fully Local":   [93.71, 85.01, 72.56, 70.61],
    "FedAvg":        [87.94, 85.60, 78.77, 73.82],
    "OptimES":       [94.95, 91.44, 86.75, 83.13],
    "FedAvg + LE":   [95.44, 92.84, 86.79, 84.22],
    "Sync-SGD + LE": [97.13, 96.23, 94.98, 94.02],
}

CENTRALIZED = 97.09
ALWAYS_POSITIVE = 31.04

STYLE = {
    "Fully Local":   {"color": "#999999", "marker": "o", "linestyle": "--"},
    "FedAvg":        {"color": "#1f77b4", "marker": "s", "linestyle": "-"},
    "OptimES":       {"color": "#ff7f0e", "marker": "D", "linestyle": "-."},
    "FedAvg + LE":   {"color": "#2ca02c", "marker": "^", "linestyle": "-"},
    "Sync-SGD + LE": {"color": "#d62728", "marker": "P", "linestyle": "-"}
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="results/figures/macro_prauc_vs_clients_louvain.pdf",
        help="Output figure path (PDF recommended for LaTeX).",
    )
    parser.add_argument("--show_baseline", action="store_true",
                        help="Also draw the always-positive classifier reference line.")
    args = parser.parse_args()

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": ":",
    })

    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    # Reference: centralized upper bound
    ax.axhline(CENTRALIZED, color="black", linestyle=":", linewidth=1.2,
               label="Centralized PNA")

    if args.show_baseline:
        ax.axhline(ALWAYS_POSITIVE, color="0.6", linestyle=(0, (1, 2)), linewidth=1.0,
                   label=f"Always-Positive ({ALWAYS_POSITIVE:.2f})")

    for name, values in MACRO_PRAUC.items():
        s = STYLE[name]
        ax.plot(CLIENTS, values,
                color=s["color"], marker=s["marker"], linestyle=s["linestyle"],
                linewidth=1.8, markersize=6.5, markeredgecolor="white",
                markeredgewidth=0.8, label=name)

    ax.set_xlabel("Number of clients")
    ax.set_ylabel("Macro PR-AUC (\\%)" if plt.rcParams.get("text.usetex") else "Macro PR-AUC (%)")
    ax.set_xticks(CLIENTS)
    ax.set_xlim(min(CLIENTS) - 0.6, max(CLIENTS) + 0.6)

    ymin = ALWAYS_POSITIVE - 2 if args.show_baseline else 65
    ax.set_ylim(ymin, 100)

    ax.legend(loc="lower left", frameon=True, framealpha=0.95,
              edgecolor="0.8", borderpad=0.5)

    fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    # Also save a PNG copy alongside, for quick previews.
    if out.suffix.lower() == ".pdf":
        fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"Saved figure to {out}")


if __name__ == "__main__":
    main()
