#!/usr/bin/env python3
import os
import copy
import subprocess

import yaml
import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    roc_auc_score,
)

# Cesty potrebne na spravne spustenie trenovania a klasifikacie

BASE_CONFIG = "./model/experiments.yaml" # konfiguracny subor, v ktorom sa menia parametre
TRAIN_SCRIPT = "./src/plASgraph2_train.py" # skript na trenovanie modelu
CLASSIFY_SCRIPT = "./src/plASgraph2_classify.py" # skript na klasifikaciu
TRAIN_FILE_LIST = "./model/plasgraph2-datasets/eskapee-train.csv" # trenovaci dataset
TEST_FILE_LIST = "./model/plasgraph2-datasets/eskapee-test.csv" # dataset pouzity na klasifikaciu a zistenie presnosti modelu
FILE_PREFIX = "./model/plasgraph2-datasets/" # kde vieme najst vsetky datasety, s ktorymi pracujeme
EXPERIMENTS_ROOT = "./experiments" # do akeho adresara sa ulozia vysledne
os.makedirs(EXPERIMENTS_ROOT, exist_ok=True)

EXPERIMENTS = [
    {
        "name": "original_plasgraph",
        "dropout_rate": 0.1,
        "loss_function": "crossentropy",
        "use_attention": False,
    },

    {
        "name": "exp0_added_attention",
        "dropout_rate": 0.1,
        "loss_function": "crossentropy",
        "use_attention": True,
    },

    {
        "name": "exp1_crossentropy_do0.2",
        "dropout_rate": 0.2,
        "loss_function": "crossentropy",
        "use_attention": True,
    },
    {
        "name": "exp2_squaredhinge_do0.2",
        "dropout_rate": 0.2,
        "loss_function": "squaredhinge",
        "use_attention": True,
    },
    {
        "name": "exp3_crossentropy_do0.3",
        "dropout_rate": 0.3,
        "loss_function": "crossentropy",
        "use_attention": True,
    },
    {
        "name": "exp4_squaredhinge_do0.3",
        "dropout_rate": 0.3,
        "loss_function": "squaredhinge",
        "use_attention": True,
    },
]


def load_truth_df(truth_path: str) -> pd.DataFrame:
    """
    Načíta *.gfa.csv ground-truth súbor.
    """
    df = pd.read_csv(truth_path)

    if "contig" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Ground-truth CSV {truth_path} has no 'contig' or 'label' column")

    df = df[["contig", "label"]].rename(columns={"label": "true_label"})
    df["contig"] = df["contig"].astype(str)
    df["true_label"] = df["true_label"].astype(str)
    return df


def compute_metrics_for_test_set(
    preds_csv: str,
    test_list_csv: str,
    file_prefix: str,
    metrics_per_sample_csv: str,
    metrics_summary_csv: str,
):
    """
    preds_csv: výstup z plASgraph2_classify.py set (predikcie pre všetky vzorky z eskapee-test.csv)
    test_list_csv: eskapee-test.csv (gfa, gfa.csv, sample_id)
    file_prefix: root adresar, kde sú súbory z test_list_csv
    """
    preds = pd.read_csv(preds_csv)

    required_pred_cols = {"sample", "contig", "plasmid_score", "chrom_score", "label"}
    missing = required_pred_cols - set(preds.columns)
    if missing:
        raise ValueError(f"Predictions CSV {preds_csv} missing columns: {missing}")

    test_df = pd.read_csv(
        test_list_csv,
        header=None,
        names=["gfa", "truth_csv", "sample"],
    )

    rows = []

    for _, row in test_df.iterrows():
        sample_id = row["sample"]
        truth_path = os.path.join(file_prefix, row["truth_csv"])

        if not os.path.exists(truth_path):
            print(f"WARNING: truth file not found for sample {sample_id}: {truth_path}")
            continue

        sample_preds = preds[preds["sample"] == sample_id].copy()
        if sample_preds.empty:
            print(f"WARNING: no predictions for sample {sample_id}")
            continue

        sample_preds["contig"] = sample_preds["contig"].astype(str)

        truth = load_truth_df(truth_path)

        df = sample_preds.merge(truth, on="contig", how="inner")
        if df.empty:
            print(f"WARNING: no overlapping contigs between preds and truth for sample {sample_id}")
            continue

        # ignorujeme ground-truth 'unlabeled'
        df = df[df["true_label"] != "unlabeled"].copy()
        if df.empty:
            print(f"WARNING: only unlabeled contigs for sample {sample_id}")
            continue

        def binarize(task: str):
            if task == "plasmid":
                y_true = df["true_label"].isin(["plasmid", "ambiguous"]).astype(int)
                y_pred = df["label"].isin(["plasmid", "ambiguous"]).astype(int)
                scores = df["plasmid_score"].values
            elif task == "chromosome":
                y_true = df["true_label"].isin(["chromosome", "ambiguous"]).astype(int)
                y_pred = df["label"].isin(["chromosome", "ambiguous"]).astype(int)
                scores = df["chrom_score"].values
            else:
                raise ValueError("task must be 'plasmid' or 'chromosome'")
            return y_true, y_pred, scores

        def metrics_for_task(task: str):
            y_true, y_pred, scores = binarize(task)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )
            acc = accuracy_score(y_true, y_pred)
            try:
                auroc = roc_auc_score(y_true, scores)
            except ValueError:
                auroc = np.nan
            return prec, rec, f1, acc, auroc, len(y_true)

        for task in ["plasmid", "chromosome"]:
            prec, rec, f1, acc, auroc, n = metrics_for_task(task)
            rows.append(
                {
                    "sample": sample_id,
                    "task": task,
                    "n_contigs": n,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "accuracy": acc,
                    "auroc": auroc,
                }
            )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(metrics_per_sample_csv, index=False)

    if not metrics_df.empty:
        summary_rows = []
        for task in ["plasmid", "chromosome"]:
            mtask = metrics_df[metrics_df["task"] == task]
            if mtask.empty:
                continue
            summary_rows.append(
                {
                    "task": task,
                    "precision": mtask["precision"].median(),
                    "recall": mtask["recall"].median(),
                    "f1": mtask["f1"].median(),
                    "accuracy": mtask["accuracy"].median(),
                    "auroc": mtask["auroc"].median(),
                }
            )
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(metrics_summary_csv, index=False)
    else:
        summary_df = pd.DataFrame()
        summary_df.to_csv(metrics_summary_csv, index=False)

    return metrics_df


def run_experiment(exp_cfg: dict):
    name = exp_cfg["name"]
    exp_dir = os.path.join(EXPERIMENTS_ROOT, name)
    os.makedirs(exp_dir, exist_ok=True)

    # vytvorenie kopie experiments.py
    with open(BASE_CONFIG, "r") as f:
        base_cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(base_cfg)

    # prepíš kľúče z exp_cfg (okrem "name") - kopia experiments.yaml
    for key, value in exp_cfg.items():
        if key == "name":
            continue
        cfg[key] = value

    # vytvorenie konfigu pre tento experiment
    exp_config_path = os.path.join(exp_dir, f"{name}.yaml")
    with open(exp_config_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    model_dir = os.path.join(exp_dir, "model") # lde sa dany model ulozi

    # Trenovanie modelu
    train_cmd = [
        "python",
        TRAIN_SCRIPT,
        exp_config_path,
        TRAIN_FILE_LIST,
        FILE_PREFIX,
        model_dir,
    ]
    print(f"\n=== Running training for {name} ===")
    print("Command:", " ".join(train_cmd))

    # vytvorenie log suborov na ulozenie priebehu behu a errorov, ktore sa vyskytli pocas behu, trenovanie modelu
    with open(os.path.join(exp_dir, "train.log"), "w") as train_log, \
         open(os.path.join(exp_dir, "train.err"), "w") as train_err:
        subprocess.run(train_cmd, stdout=train_log, stderr=train_err, check=True)

    # Klasifikacia modelu
    preds_csv = os.path.join(exp_dir, "eskapee_test_predictions.csv")
    classify_cmd = [
        "python",
        CLASSIFY_SCRIPT,
        "set",
        TEST_FILE_LIST,
        FILE_PREFIX,
        model_dir,
        preds_csv,
    ]
    print(f"\n=== Running classification on eskapee-test for {name} ===")
    print("Command:", " ".join(classify_cmd))
    subprocess.run(classify_cmd, check=True)

    # Vypocet skor (F1, accuracy, auroc, recall) pre dany experiment
    metrics_per_sample_csv = os.path.join(exp_dir, "metrics_per_sample.csv")
    metrics_summary_csv = os.path.join(exp_dir, "metrics_summary.csv")

    print(f"\n=== Computing metrics on eskapee-test for {name} ===")
    metrics_df = compute_metrics_for_test_set(
        preds_csv,
        TEST_FILE_LIST,
        FILE_PREFIX,
        metrics_per_sample_csv,
        metrics_summary_csv,
    )

    print("\nPer-sample metrics (head):")
    if not metrics_df.empty:
        print(metrics_df.head().to_string(index=False))
    else:
        print("No metrics computed (empty DataFrame).")

    print(f"\nFinished experiment {name}")


def main():
    for exp in EXPERIMENTS:
        run_experiment(exp)


if __name__ == "__main__":
    main()
