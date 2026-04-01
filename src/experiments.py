import os
import copy
import subprocess

import yaml
import pandas as pd
import numpy as np

# Cesty potrebne na spravne spustenie trenovania a klasifikacie

BASE_CONFIG = "./model/experiments.yaml" # konfiguracny subor, v ktorom sa menia parametre
TRAIN_SCRIPT = "./src/plASgraph2_train.py" # skript na trenovanie modelu
CLASSIFY_SCRIPT = "./src/plASgraph2_classify.py" # skript na klasifikaciu
TRAIN_FILE_LIST = "./model/plasgraph2-datasets/eskapee-train.csv" # trenovaci dataset
TEST_FILE_LIST = "./model/plasgraph2-datasets/eskapee-test.csv" # dataset pouzity na klasifikaciu a zistenie presnosti modelu
FILE_PREFIX = "./model/plasgraph2-datasets/" # kde vieme najst vsetky datasety, s ktorymi pracujeme
EXPERIMENTS_ROOT = "./experiments/with_normalization" # do akeho adresara sa ulozia vysledky
RESULTS_ROOT = "./results/eval_original_scripts"
EVAL = "./src/evaluation/eval.py"
SUMMARY = "./src/evaluation/eval-summary.py"
os.makedirs(EXPERIMENTS_ROOT, exist_ok=True)

head = [2]
learning_rate = [0.003, 0.005]
tie = [True]
normalization = [None, "batch", "layer"]
name = ""

EXPERIMENTS = []

for h in head:
    for lr in learning_rate:
        for t in tie:
            for n in normalization:
                name = f"{h}_h_lr_{lr}_tie_{t}_norm_{n}"
                exp = {'name': name, 'number_of_heads': h, 'learning_rate': lr, 'tie_gnn_layers': t, 'normalization': n}
                EXPERIMENTS.append(exp)


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

    print(f"\nFinished experiment {name}")

# "python ./src/evaluation/eval.py model/gold_all.csv vsetky_experimenty -n nazvy_experimentov -s default > kde_sa_ma_csv_ulozit"
# "python ./src/evaluation/eval-summary.py results/eval_original_scripts/eval_no_normalization.csv MEDIAN -o csv > results/eval_original_scripts/summary_no_normalization.csv"

def resutls(eval_name, summary_name):
    name_experiments = os.listdir(EXPERIMENTS_ROOT)
    path_experiments = []
    for exp in name_experiments:
        path_experiments.append(EXPERIMENTS_ROOT + "/" + exp + "/" + "eskapee_test_predictions.csv")

    name = ','.join(name_experiments)
    path = ','.join(path_experiments)

    cmd_eval = [
        "python",
        EVAL,
        "model/gold_all.csv",
        *path_experiments,
        "-n",
        name,
        "-s",
        "default",
    ]

    print("\n=== Running evaluation ===")
    with open(os.path.join(RESULTS_ROOT, eval_name), "w") as eval_out:
        subprocess.run(cmd_eval, stdout=eval_out, check=True)

    cmd_summary = [
        "python",
        SUMMARY,
        os.path.join(RESULTS_ROOT, eval_name),
        "MEDIAN",
        "-o",
        "csv",
    ]

    print("\n=== Running summary of the evaluation ===")
    with open(os.path.join(RESULTS_ROOT, summary_name), "w") as summary_out:
        subprocess.run(cmd_summary, stdout=summary_out, check=True)



if __name__ == "__main__":
    for exp in EXPERIMENTS:
        run_experiment(exp)
    resutls("eval_with_normalization.csv", "summary_with_normalization.csv")