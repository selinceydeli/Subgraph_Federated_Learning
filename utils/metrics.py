#!/usr/bin/env python3
from typing import Optional
import os
import csv
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import torch


@torch.no_grad()
def compute_confusion_matrix_per_task(logits: torch.Tensor, labels: torch.Tensor):
    """
    Compute binary confusion matrices per task for multi-label predictions.

    Args:
        logits: Tensor of shape [N, T]
        labels: Tensor of shape [N, T]

    Returns:
        dict mapping task_idx -> {"tp": int, "fp": int, "tn": int, "fn": int}
    """
    preds = (torch.sigmoid(logits) > 0.5)
    labels = labels.bool()

    out = {}
    num_tasks = labels.size(1)

    for t in range(num_tasks):
        p = preds[:, t]
        y = labels[:, t]

        tp = int(((p == 1) & (y == 1)).sum().item())
        fp = int(((p == 1) & (y == 0)).sum().item())
        tn = int(((p == 0) & (y == 0)).sum().item())
        fn = int(((p == 0) & (y == 1)).sum().item())

        out[t] = {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }

    return out


def sum_confusion_matrices(cm_list):
    """
    Sum a list of per-task confusion-matrix dicts.
    """
    if not cm_list:
        return {}

    task_ids = cm_list[0].keys()
    out = {}

    for t in task_ids:
        out[t] = {
            "tp": sum(cm[t]["tp"] for cm in cm_list),
            "fp": sum(cm[t]["fp"] for cm in cm_list),
            "tn": sum(cm[t]["tn"] for cm in cm_list),
            "fn": sum(cm[t]["fn"] for cm in cm_list),
        }

    return out

    
def compute_minority_f1_score_per_task(logits, labels, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold)
    y = labels.bool()

    N, C = y.shape
    f1_scores = torch.zeros(C, dtype=torch.float32, device=logits.device)
    epsilon = 1e-12
    
    for c in range(C):
        y_c = y[:, c]

        # Find the minority class (either 0 or 1)
        pos = y_c.sum()
        neg = y_c.numel() - pos
        minority_is_one = (pos <= neg) 

        if minority_is_one:
            y_pos    = y_c
            pred_pos = preds[:, c]
        else:
            y_pos    = ~y_c
            pred_pos = ~preds[:, c]

        true_pos = (y_pos & pred_pos).sum().float()
        false_pos = ((~y_pos) & pred_pos).sum().float()
        false_neg = (y_pos & (~pred_pos)).sum().float()
        
        precision = true_pos / (true_pos + false_pos + epsilon)
        recall = true_pos / (true_pos + false_neg + epsilon)
        f1 = 2 * precision * recall / (precision + recall + epsilon)
        f1_scores[c] = f1

    return f1_scores


def compute_label_percentages(
    input_csv="./data/y_sums.csv",
    output_csv="./results/metrics/label_percentages.csv",
    add_mean=True
):
    '''
    Read label totals from `input_csv`, compute per-split
    percentages for each task, optionally append
    a 'mean_over_splits' row, and save to CSV.
    '''
    df = pd.read_csv(input_csv)

    if "total" not in df.columns:
        raise ValueError("Input CSV must include a 'total' column.")

    has_split_col = "split" in df.columns
    task_cols = [c for c in df.columns if c not in {"split", "total"}]

    if not task_cols:
        raise ValueError("No task columns found (columns other than 'split' and 'total').")

    # Ensure numeric columns are actually numeric
    for col in task_cols + ["total"]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    # per-split percentages
    pct = (df[task_cols].div(df["total"], axis=0) * 100.0).round(2)

    # Preserve split names if available
    if has_split_col:
        pct.insert(0, "split", df["split"])
    else:
        pct.index = [f"split_{i+1}" for i in range(len(pct))]

    if add_mean:
        mean_values = pct[task_cols].mean(axis=0).round(2).to_dict()
        if has_split_col:
            mean_values = {"split": "mean_over_splits", **mean_values}
            pct = pd.concat([pct, pd.DataFrame([mean_values])], ignore_index=True)
        else:
            pct.loc["mean_over_splits"] = pct.mean(axis=0).round(2)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    pct.to_csv(output_csv, index=not has_split_col)

    return pct


def _fmt_runtime_hhmmss(seconds: float) -> str:
    # Robust, clamps negatives to 0
    seconds = max(0.0, float(seconds))
    td = timedelta(seconds=round(seconds))
    # Format as HH:MM:SS (e.g., "01:23:45")
    total_seconds = int(td.total_seconds())
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def append_f1_score_to_csv(
    out_csv: str,
    tasks: list[str],
    mean_f1,
    std_f1,
    macro_mean_percent: float,
    seeds: list[int],
    model_name: str = "PNA baseline",
    runtime_seconds: Optional[float] = None,  
):
    """
    Append a single row with mean/std per task (in %), macro mean (in %), runtime, and metadata.
    Creates the CSV (with header) if it doesn't exist.
    If the CSV exists without a 'runtime' column, this will append that column for new rows.
    """
    # Ensure directory exists (if any)
    dir_ = os.path.dirname(out_csv)
    if dir_:
        os.makedirs(dir_, exist_ok=True)

    # Build the default header for this run
    mean_cols = [f"{t}_mean_pct" for t in tasks]
    std_cols  = [f"{t}_std_pct"  for t in tasks]
    default_header = (
        ["timestamp_iso", "model", "n_runs", "seeds", "macro_mean_pct"]
        + mean_cols + std_cols + ["runtime"]  # <— include runtime by default
    )

    # See if file already exists and if it has a header; keep compatibility
    file_exists = os.path.exists(out_csv)
    if file_exists:
        existing_header = None
        try:
            with open(out_csv, "r", newline="") as f_in:
                r = csv.reader(f_in)
                existing_header = next(r, None)
        except Exception:
            existing_header = None

        if existing_header:
            # Ensure 'runtime' is present at least at the end
            header = list(existing_header)
            if "runtime" not in header:
                header = header + ["runtime"]
        else:
            header = default_header
    else:
        header = default_header

    # Prepare row values
    mean_pct = (mean_f1 * 100.0).tolist()
    std_pct  = (std_f1  * 100.0).tolist()

    row = {
        "timestamp_iso": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "n_runs": len(seeds),
        "seeds": ",".join(map(str, seeds)),
        "macro_mean_pct": round(macro_mean_percent, 2),
        **{c: round(v, 2) for c, v in zip(mean_cols, mean_pct)},
        **{c: round(v, 2) for c, v in zip(std_cols,  std_pct)},
    }

    # Add runtime field (string "HH:MM:SS"); if missing, leave empty
    row["runtime"] = _fmt_runtime_hhmmss(runtime_seconds) if runtime_seconds is not None else ""

    # Write (create header if file doesn't exist)
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        if not file_exists:
            w.writeheader()
        # If the existing file lacked 'runtime', DictWriter will write an extra column at the end
        # (older rows won't have it, which is OK for CSV readers).
        w.writerow(row)


def start_epoch_csv(model_name: str,
                    seed: int,
                    tasks: list,
                    out_dir: str = "./results/metrics/epoch_logs") -> str:
    """
    Creates a timestamped CSV for per-epoch metrics and writes the header.
    Returns the full path to the CSV file.
    """
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = f"{ts}_{model_name}_seed{seed}.csv"
    path = os.path.join(out_dir, fname)

    header = ["epoch", "train_loss", "val_loss", "val_macro_minF1"] + [f"val_{t}_minF1" for t in tasks]

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"# model_name={model_name}"])
        w.writerow([f"# seed={seed}"])
        w.writerow([f"# started_at={ts}"])
        w.writerow(header)

    return path


def append_epoch_csv(csv_path: str,
                     epoch: int,
                     train_loss: float,
                     val_loss: float,
                     val_f1_tensor) -> None:
    """
    Appends a single epoch row to the CSV. val_f1_tensor is shape [num_tasks].
    """
    if hasattr(val_f1_tensor, "detach"):
        vals = val_f1_tensor.detach().cpu().tolist()
    else:
        vals = list(val_f1_tensor)
    macro = float(sum(vals) / len(vals))

    row = [int(epoch), float(train_loss), float(val_loss), float(macro)] + [float(v) for v in vals]

    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow(row)


def write_fed_split_sizes(writer, split_type: str, client_list):
    for cid, data in enumerate(client_list):
        # number of nodes
        if hasattr(data, "num_nodes") and data.num_nodes is not None:
            num_nodes = int(data.num_nodes)
        else:
            num_nodes = data.x.size(0)
        # number of edges 
        num_edges = data.edge_index.size(1) if getattr(data, "edge_index", None) is not None else 0
        writer.writerow([split_type, cid, num_nodes, num_edges])