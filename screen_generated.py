#!/usr/bin/env python3
"""
STEP 6 (Mac, inside the EGMOF folder): first check of the generated CIFs.

For every target folder in generated_V1/ (V1_0.60, V1_1.00, ...):
  - reads every CIF
  - runs egmof.linker_check.check_structure() (the same check used by generate_V1.py):
    rejects broken files, overlapping atoms (minimum interatomic distance < 0.9 A),
    structures without metal or organic linker, and structures whose organic
    fragments do not form exactly the required number of unique molecular graphs
    (--organic-mode: 1 for single/single-linker, 2 for ligand-linker, any = overlap only)
  - copies the passing CIFs to screened_V1/V1_x.xx/
  - writes one summary table, screened_V1/summary.csv, plus one line per target

Usage:
  uv run python screen_generated.py
  uv run python screen_generated.py --gen generated_V1 --out screened_V1 --dmin 1.0
  uv run python screen_generated.py --gen generated_ligand_linker --organic-mode ligand-linker
"""
import argparse
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read
from pymatgen.core import Element

from egmof.linker_check import MIN_DIST_THRESHOLD, check_structure, expected_unique_for_mode

warnings.filterwarnings("ignore")

# metal building blocks = single tokens in Desc2MOF; everything else (except E0) is an organic ligand
try:
    from egmof.desc2mof import __desc2mof_dir__
    METAL_BBS = set(open(f"{__desc2mof_dir__}/data/metal_node.txt").read().split()) | \
                set(open(f"{__desc2mof_dir__}/data/metal_edge.txt").read().split())
except Exception:
    METAL_BBS = None


def ligand_linker_counts(name):
    """(distinct organic ligands = N/Gen_N blocks, distinct organic linkers = E/Gen_E blocks)."""
    if METAL_BBS is None:
        return None, None
    blocks = [b for b in name.split("+")[1:] if b != "E0" and b not in METAL_BBS]
    lig = {b for b in blocks if b.startswith(("N", "Gen_N"))}
    lnk = {b for b in blocks if b.startswith(("E", "Gen_E"))}
    return len(lig), len(lnk)


def n_ligands(name):
    """Distinct organic ligands in a MOF name like 'pcu+N85+E72' (None if lists unavailable)."""
    if METAL_BBS is None:
        return None
    blocks = [b for b in name.split("+")[1:] if b != "E0"]
    return len({b for b in blocks if b not in METAL_BBS})


ap = argparse.ArgumentParser()
ap.add_argument("--gen", default="generated_V1", help="folder written by generate_V1.py")
ap.add_argument("--out", default="screened_V1", help="where passing CIFs are copied")
ap.add_argument("--dmin", type=float, default=MIN_DIST_THRESHOLD,
                help=f"minimum allowed interatomic distance (A, default {MIN_DIST_THRESHOLD})")
ap.add_argument(
    "--require-one-organic",
    action="store_true",
    help="kept for compatibility; use --organic-mode instead",
)
ap.add_argument(
    "--organic-mode",
    default="single-linker",
    choices=["any", "single", "single-linker", "ligand-linker"],
    help="unique linker graphs required per CIF: 1 for single/single-linker, "
         "2 for ligand-linker, any = overlap check only (use the mode of generate_V1.py)",
)
args = ap.parse_args()
expected_unique = expected_unique_for_mode(args.organic_mode)

gen, out = Path(args.gen), Path(args.out)
out.mkdir(exist_ok=True)
rows = []

for tdir in sorted(gen.glob("V1_*")):
    target = float(tdir.name.split("_")[1])
    cifs = sorted((tdir / "cifs").glob("*.cif"))
    if not cifs:
        print(f"{tdir.name}: no CIFs yet, skipped")
        continue

    # RF-predicted V1 per MOF, if generate_V1.py saved it
    pred = {}
    csv = tdir / "generated.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        if "pred_value" in df:
            for _, r in df.iterrows():
                for v in r.values:
                    if isinstance(v, str):
                        pred[Path(v).stem] = r["pred_value"]

    dest = out / tdir.name
    dest.mkdir(exist_ok=True)
    n_ok = 0
    for cif in cifs:
        row = {"target_V1": target, "mof": cif.stem, "pred_V1": pred.get(cif.stem, np.nan),
               "n_ligands": n_ligands(cif.stem),
               "n_ligand_types": ligand_linker_counts(cif.stem)[0],
               "n_linker_types": ligand_linker_counts(cif.stem)[1]}
        res = check_structure(cif, expected_unique=expected_unique, min_dist_threshold=args.dmin)
        try:
            atoms = read(str(cif))
            symbols = set(atoms.get_chemical_symbols()) - {"X"}
            metals = " ".join(sorted(e for e in symbols if Element(e).is_metal))
            n_atoms = len(atoms)
        except Exception:  # unreadable CIF; check_structure already reports the error
            metals, n_atoms = "", np.nan
        row.update(n_atoms=n_atoms,
                   min_dist=round(res["min_dist"], 3),
                   metals=metals,
                   passed=res["passed"],
                   reason=res["reason"],
                   n_fragments=res["n_fragments"],
                   n_unique_graphs=res["n_unique_graphs"],
                   organic_inchikeys=";".join(res["linker_inchikeys"]),
                   organic_smiles=";".join(res["linker_smiles"]),
                   organic_formulas=res["linker_formula"],
                   linker_reference=res["linker_reference"],
                   chemistry_issue=res["chemistry_issue"])
        if row["passed"]:
            shutil.copy(cif, dest / cif.name)
            n_ok += 1
        rows.append(row)
    print(f"{tdir.name}: {n_ok}/{len(cifs)} CIFs pass the [{args.organic_mode}] linker check "
          f"(min dist >= {args.dmin} A) -> {dest}")

if rows:
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    ok = summary[summary["passed"]]
    print("\nPassed per target (with mean RF-predicted V1):")
    if summary["n_ligands"].notna().any():
        print("\nLigands per passing MOF:", ok["n_ligands"].value_counts().sort_index().to_dict())
        comp = ok["n_ligand_types"].astype(str) + " ligand + " + ok["n_linker_types"].astype(str) + " linker"
        print("Composition of passing MOFs:", comp.value_counts().to_dict())
    print("Failure reasons:", summary.loc[~summary["passed"], "reason"].value_counts().to_dict())
    print(ok.groupby("target_V1").agg(n_passed=("mof", "count"),
                                      mean_pred_V1=("pred_V1", "mean"),
                                      mean_atoms=("n_atoms", "mean")).round(2))
    print(f"\nSummary table: {out / 'summary.csv'}")