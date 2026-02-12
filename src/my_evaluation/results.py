#!/usr/bin/env python3
import os
import pandas as pd

EXPERIMENTS_ROOT = "./experiments"
OUTPUT_CSV = "./experiments/all_metrics_summary.csv"

def main():
    rows = []
    for exp_name in os.listdir(EXPERIMENTS_ROOT):
        exp_dir = os.path.join(EXPERIMENTS_ROOT, exp_name)
        if not os.path.isdir(exp_dir):
            continue

        metrics_path = os.path.join(exp_dir, "metrics_summary.csv")
        if not os.path.exists(metrics_path):
            print(f"WARNING: metrics_summary.csv not found in {exp_dir}")
            continue

        df = pd.read_csv(metrics_path)
        df["experiment"] = exp_name  # add experiment name as a column
        rows.append(df)

    if not rows:
        print("No metrics_summary.csv files found.")
        return

    all_metrics = pd.concat(rows, ignore_index=True)
    # Optional: reorder columns
    cols = ["experiment", "task", "precision", "recall", "f1", "accuracy", "auroc"]
    all_metrics = all_metrics[cols]

    all_metrics.to_csv(OUTPUT_CSV, index=False)
    print(f"Combined metrics written to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
