#!/usr/bin/env python3
"""
STEP 3 (Mac, inside the EGMOF folder).

Builds everything EGMOF needs to retrain Prop2Desc on the K-ion label V1:

  <data>/total.csv                 all MOFs: 183 descriptors + V1 + filename
  <data>/train_V1.csv / val_V1.csv / test_V1.csv   80 / 10 / 10 split
  <data>/rf_V1.pkl                 Random Forest (scaled descriptors -> scaled V1)
  config/k_ion/prop2desc_V1_train.yaml   training config (for prop2desc.train)
  config/k_ion/V1.yaml             generation config (scaler + feature importances)

It also prints the Random-Forest test R^2: a quick check that the 183
descriptors can actually predict V1 (want roughly R^2 >= 0.4).

Usage:
  uv run python make_dataset.py --data k_v1_data
"""
import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

EPS = 1e-6  # same epsilon EGMOF's Scaler uses
TARGET = "V1"

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="folder with descriptors.csv + training_labels.csv")
ap.add_argument("--egmof", default=".", help="EGMOF repo root")
ap.add_argument("--epochs", type=int, default=1000)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

data = Path(args.data)
root = Path(args.egmof)
names = json.load(open(root / "src/egmof/data/descriptor_name.json"))


def rename_rac(df):
    """Same renaming EGMOF applies (only needed for older descriptor CSVs)."""
    m = {}
    for c in df.columns:
        if c.startswith("f-sbu-"):
            m[c] = f"f-{c[6:]}-all"
        elif c.startswith("f-link-"):
            m[c] = f"f-lig-{c[7:]}"
        elif re.match(r"^(D_mc|D_lc|D_func|mc|lc|func)-", c) and not c.endswith("-all"):
            m[c] = f"{c}-all"
    return df.rename(columns=m)


# ---------- 1. merge descriptors + labels ----------
desc = pd.read_csv(data / "descriptors.csv")
desc = desc.loc[:, ~desc.columns.duplicated()].copy()
if not set(names) <= set(desc.columns):
    desc = rename_rac(desc)
missing_cols = [n for n in names if n not in desc.columns]
if missing_cols:
    raise SystemExit(f"Descriptor columns missing: {missing_cols[:10]} ...")

desc["qmof_id"] = desc["filename"].astype(str).str.replace(r"\.cif$", "", regex=True)
lab = pd.read_csv(data / "training_labels.csv")[["qmof_id", TARGET]]
df = desc.merge(lab, on="qmof_id", how="inner")

n0 = len(df)
X = df[names].apply(pd.to_numeric, errors="coerce")
ok = np.isfinite(X).all(axis=1) & np.isfinite(df[TARGET])
df = df[ok].reset_index(drop=True).copy()
print(f"Labels: {len(lab)}  descriptors: {len(desc)}  merged: {n0}  usable: {len(df)}")

# ---------- 2. split 80/10/10, stratified on V1 ----------
rng = np.random.default_rng(args.seed)
bins = pd.qcut(df[TARGET].rank(method="first"), q=10, labels=False)
split = np.empty(len(df), dtype=object)
for b in range(10):
    idx = rng.permutation(np.where(bins == b)[0])
    n_val = n_test = max(1, round(0.1 * len(idx)))
    split[idx[:n_val]] = "val"
    split[idx[n_val:n_val + n_test]] = "test"
    split[idx[n_val + n_test:]] = "train"
df["split"] = split

keep = names + [TARGET, "filename"]
df[keep].to_csv(data / "total.csv", index=False)
for s in ("train", "val", "test"):
    df.loc[df.split == s, keep].to_csv(data / f"{s}_{TARGET}.csv", index=False)
print("Split:", df.split.value_counts().to_dict())

# ---------- 3. scaler from the training split (as EGMOF does) ----------
tr = df[df.split == "train"]
mean = tr[names].mean().astype(float).tolist()
std = tr[names].std().astype(float).tolist()
t_mean, t_std = float(tr[TARGET].mean()), float(tr[TARGET].std())
scaler_value = {"mean": mean, "std": std, "target_mean": t_mean, "target_std": t_std}


def enc_x(frame):
    return (frame[names].values - np.array(mean)) / (np.array(std) + EPS)


def enc_y(y):
    return (y - t_mean) / (t_std + EPS)


# ---------- 4. Random Forest (feature importances for WMSE + pred_value) ----------
rf = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=args.seed)
rf.fit(enc_x(tr), enc_y(tr[TARGET].values))
te = df[df.split == "test"]
pred = rf.predict(enc_x(te)) * t_std + t_mean
r2, mae = r2_score(te[TARGET], pred), mean_absolute_error(te[TARGET], pred)
joblib.dump(rf, data / f"rf_{TARGET}.pkl")
print(f"Random Forest test:  R2 = {r2:.3f}   MAE = {mae:.3f} V")
print("  ->", "OK, descriptors capture V1" if r2 >= 0.4 else
      "WEAK: descriptors predict V1 poorly; guidance will be limited")

fi = rf.feature_importances_.tolist()
top = sorted(zip(names, fi), key=lambda t: -t[1])[:10]
print("Top-10 descriptors:", ", ".join(f"{n} ({v:.3f})" for n, v in top))

# ---------- 5. configs ----------
cfg_dir = root / "config/k_ion"
cfg_dir.mkdir(parents=True, exist_ok=True)

train_cfg = {
    "seed": args.seed,
    "logger": {"project": "EGMOF_K", "exp_name": TARGET, "log_dir": "./logs_k"},
    "model": {
        "in_channels": 96, "out_channels": None, "dim": 183, "dim_mults": [1, 2, 4],
        "timestep": 1000, "condition": "numeric", "num_classes": None, "cond_dim": None,
        "seed": 0, "lr": 0.001, "scaler_mode": "standard", "scaler_value": scaler_value,
    },
    "datamodule": {"path": str(data.resolve()), "task": TARGET, "dataset_cls": "csv",
                   "batch_size": 64, "num_workers": 4},
    "trainer": {"accelerator": "auto", "devices": 1, "max_epochs": args.epochs,
                "precision": 32, "per_gpu_batchsize": None, "num_nodes": 1,
                "val_check_interval": 1.0},
    "test_only": False, "resume_from": None, "ckpt_path": None,
}
yaml.safe_dump(train_cfg, open(cfg_dir / f"prop2desc_{TARGET}_train.yaml", "w"), sort_keys=False)

# generation config: same layout as config/various_dataset/bandgap.yaml
gen = yaml.safe_load(open(root / "config/various_dataset/bandgap.yaml"))
gen.update({
    "project": "EGMOF_K", "exp_name": TARGET, "target": TARGET, "task": TARGET,
    "path": str(data.resolve()), "dim_mults": [1, 2, 4], "lr": 0.001,
    "max_epochs": args.epochs, "devices": 1, "accelerator": "cpu", "precision": 32,
    "scaler_value": scaler_value, "feature_importances": fi,
})
gen.pop("num_training_steps_per_epochs", None)
yaml.safe_dump(gen, open(cfg_dir / f"{TARGET}.yaml", "w"), sort_keys=False)

print(f"\nV1 range in training data: {df[TARGET].min():.2f} to {df[TARGET].max():.2f} V "
      f"(median {df[TARGET].median():.2f})")
print(f"Wrote {data}/total.csv, train/val/test_{TARGET}.csv, rf_{TARGET}.pkl")
print(f"Wrote {cfg_dir}/prop2desc_{TARGET}_train.yaml and {cfg_dir}/{TARGET}.yaml")
