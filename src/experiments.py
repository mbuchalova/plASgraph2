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
EXPERIMENTS_ROOT = "./experiments/no_normalization" # do akeho adresara sa ulozia vysledky
os.makedirs(EXPERIMENTS_ROOT, exist_ok=True)

EXPERIMENTS = [
    {
        "name": "1_h_lr_0.005_tie_on",
        "use_attention": True,
        "number_of_heads": 1,
        "learning_rate": 0.005,
        "tie_gnn_layers": True
    },
    {
        "name": "1_h_lr_0.0001_tie_on",
        "use_attention": True,
        "number_of_heads": 1,
        "learning_rate": 0.0001,
        "tie_gnn_layers": True
    },
    {
        "name": "1_h_lr_0.0003_tie_on",
        "use_attention": True,
        "number_of_heads": 1,
        "learning_rate": 0.0003,
        "tie_gnn_layers": True
    },
{
        "name": "1_h_lr_0.005_tie_off",
        "use_attention": True,
        "number_of_heads": 1,
        "learning_rate": 0.005,
        "tie_gnn_layers": False
    },
    {
        "name": "1_h_lr_0.0001_tie_off",
        "use_attention": True,
        "number_of_heads": 1,
        "learning_rate": 0.0001,
        "tie_gnn_layers": False
    },
    {
        "name": "1_h_lr_0.0003_tie_off",
        "use_attention": True,
        "number_of_heads": 1,
        "learning_rate": 0.0003,
        "tie_gnn_layers": False
    },


    {
        "name": "2_h_lr_0.005_tie_on",
        "use_attention": True,
        "number_of_heads": 2,
        "learning_rate": 0.005,
        "tie_gnn_layers": True
    },
    {
        "name": "2_h_lr_0.0001_tie_on",
        "use_attention": True,
        "number_of_heads": 2,
        "learning_rate": 0.0001,
        "tie_gnn_layers": True
    },
    {
        "name": "2_h_lr_0.0003_tie_on",
        "use_attention": True,
        "number_of_heads": 2,
        "learning_rate": 0.0003,
        "tie_gnn_layers": True
    },
    {
        "name": "2_h_lr_0.005_tie_off",
        "use_attention": True,
        "number_of_heads": 2,
        "learning_rate": 0.005,
        "tie_gnn_layers": False
    },
    {
        "name": "2_h_lr_0.0001_tie_off",
        "use_attention": True,
        "number_of_heads": 2,
        "learning_rate": 0.0001,
        "tie_gnn_layers": False
    },
    {
        "name": "2_h_lr_0.0003_tie_off",
        "use_attention": True,
        "number_of_heads": 2,
        "learning_rate": 0.0003,
        "tie_gnn_layers": False
    },


    {
        "name": "3_h_lr_0.005_tie_on",
        "use_attention": True,
        "number_of_heads": 3,
        "learning_rate": 0.005,
        "tie_gnn_layers": True
    },
    {
        "name": "3_h_lr_0.0001_tie_on",
        "use_attention": True,
        "number_of_heads": 3,
        "learning_rate": 0.0001,
        "tie_gnn_layers": True
    },
    {
        "name": "3_h_lr_0.0003_tie_on",
        "use_attention": True,
        "number_of_heads": 3,
        "learning_rate": 0.0003,
        "tie_gnn_layers": True
    },
    {
        "name": "3_h_lr_0.005_tie_off",
        "use_attention": True,
        "number_of_heads": 3,
        "learning_rate": 0.005,
        "tie_gnn_layers": False
    },
    {
        "name": "3_h_lr_0.0001_tie_off",
        "use_attention": True,
        "number_of_heads": 3,
        "learning_rate": 0.0001,
        "tie_gnn_layers": False
    },
    {
        "name": "3_h_lr_0.0003_tie_off",
        "use_attention": True,
        "number_of_heads": 3,
        "learning_rate": 0.0003,
        "tie_gnn_layers": False
    },
]

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


if __name__ == "__main__":
    for exp in EXPERIMENTS:
        run_experiment(exp)