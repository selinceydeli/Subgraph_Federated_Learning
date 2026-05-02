"""Plot line chart of Macro PR-AUC vs. number of clients across FL methods.

Methods compared (METIS split with cross-client edges, 6-layer PNA):
  - Fully Local
  - FedAvg
  - FedAvg + Layer-wise Exchange
  - Sync-SGD + Layer-wise Exchange

Reference lines: Centralized PNA (upper bound) and Always-Positive classifier (naive baseline).

Usage:
    python3 -m scripts.analysis.plot_macro_prauc_vs_clients_metis
    python3 -m scripts.analysis.plot_macro_prauc_vs_clients_metis --output figures/macro_prauc_vs_clients_metis.pdf
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt


CLIENTS = [3, 5, 10, 15]

MACRO_PRAUC = {
    "Fully Local":              [89.42, 81.94, 73.68, 69.42],
    "FedAvg":                   [83.62, 82.30, 80.17, 71.93],
    "FedAvg + LE":              [91.17, 88.02, 84.08, 81.96],
    "Sync-SGD + LE":            [93.32, 91.71, 90.30, 88.05],
}

CENTRALIZED = 97.09
ALWAYS_POSITIVE = 31.04

STYLE = {
    "Fully Local":   {"color": "#999999", "marker": "o", "linestyle": "--"},
    "FedAvg":        {"color": "#1f77b4", "marker": "s", "linestyle": "-"},
    "FedAvg + LE":   {"color": "#2ca02c", "marker": "^", "linestyle": "-"},
    "Sync-SGD + LE": {"color": "#d62728", "marker": "D", "linestyle": "-"},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="results/figures/macro_prauc_vs_clients_metis.pdf",
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
               label=f"Centralized PNA ({CENTRALIZED:.2f})")

    if args.show_baseline:
        ax.axhline(ALWAYS_POSITIVE, color="0.6", linestyle=(0, (1, 2)), linewidth=1.0,
                   label=f"Always-Positive ({ALWAYS_POSITIVE:.2f})")

    for name, values in MACRO_PRAUC.items():
        s = STYLE[name]
        ax.plot(CLIENTS, values,
                color=s["color"], marker=s["marker"], linestyle=s["linestyle"],
                linewidth=1.8, markersize=6.5, markeredgecolor="white",
                markeredgewidth=0.8, label=name)

    ax.set_title("Macro PR-AUC vs. Number of Clients (METIS split)",
                 pad=10)
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
