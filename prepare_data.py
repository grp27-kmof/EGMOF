#!/usr/bin/env python3
"""
STEP 1 (Mac, inside the EGMOF folder).

Reads the CHGNet results.csv, makes the label V1 = -E_ads_1K_eV, and copies the
CIF of every labelled MOF into k_v1_data/cifs/ as <qmof_id>.cif.

Usage:
  uv run python prepare_data.py --results /path/to/results.csv \
         --cifs /path/to/Results/training_set/cifs --out k_v1_data
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--results", required=True, help="CHGNet results.csv (MOF, E_ads_1K_eV, ...)")
ap.add_argument("--cifs", required=True, help="folder with the bare CIFs (<qmof_id>.cif)")
ap.add_argument("--out", default="k_v1_data")
args = ap.parse_args()

df = pd.read_csv(args.results)
df = df[df["MOF"].astype(str).str.startswith("qmof")].drop_duplicates("MOF")
df = df.dropna(subset=["E_ads_1K_eV"])
df["V1"] = -df["E_ads_1K_eV"]

out = Path(args.out)
(out / "cifs").mkdir(parents=True, exist_ok=True)
found, missing = [], []
for mof in df["MOF"]:
    src = Path(args.cifs) / f"{mof}.cif"
    if src.exists():
        shutil.copy2(src, out / "cifs" / src.name)
        found.append(mof)
    else:
        missing.append(mof)

lab = df[df["MOF"].isin(found)][["MOF", "E_ads_1K_eV", "V1"]].rename(columns={"MOF": "qmof_id"})
lab.to_csv(out / "training_labels.csv", index=False)

print(f"MOFs with E_ads_1K in results.csv: {len(df)}")
print(f"CIFs copied: {len(found)}   CIF not found: {len(missing)} {missing[:5]}")
print(lab["V1"].describe().round(3).to_string())
print(f"-> {out}/training_labels.csv and {out}/cifs/")
