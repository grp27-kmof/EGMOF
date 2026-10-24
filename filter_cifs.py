"""
Check generated MOF CIFs for overlapping atoms and sort them.

Good CIFs  (min distance >= threshold) -> good_cifs/
Bad CIFs   (min distance <  threshold) -> broken_cifs/
Summary of every CIF                   -> cif_check_summary.csv
"""
import argparse
import csv
import glob
import os
import shutil
import warnings

import numpy as np

warnings.filterwarnings("ignore")
from pymatgen.core import Structure


def main():
    parser = argparse.ArgumentParser(description="Filter MOF CIFs by minimum interatomic distance")
    parser.add_argument("--in_dir", default="test_output/cifs", help="Folder with generated CIFs")
    parser.add_argument("--out_dir", default="test_output", help="Where good_cifs/ and broken_cifs/ go")
    parser.add_argument("--threshold", type=float, default=0.9, help="Minimum allowed distance in Angstrom")
    args = parser.parse_args()

    good_dir = os.path.join(args.out_dir, "good_cifs")
    bad_dir = os.path.join(args.out_dir, "broken_cifs")
    os.makedirs(good_dir, exist_ok=True)
    os.makedirs(bad_dir, exist_ok=True)

    cif_files = sorted(glob.glob(os.path.join(args.in_dir, "*.cif")))
    if not cif_files:
        print(f"No CIF files found in {args.in_dir}")
        return

    rows = []
    n_good = 0

    for path in cif_files:
        name = os.path.basename(path)
        try:
            s = Structure.from_file(path)
            d = s.distance_matrix
            np.fill_diagonal(d, np.inf)
            dmin = float(d.min())
            i, j = np.unravel_index(d.argmin(), d.shape)
            pair = f"{s[i].specie}-{s[j].specie}"
            metals = ",".join(sorted({el.symbol for el in s.composition.elements if el.is_metal}))
            natoms = len(s)
            formula = s.composition.reduced_formula
            status = "good" if dmin >= args.threshold else "broken"
        except Exception as e:
            dmin, pair, metals, natoms, formula = None, "", "", None, ""
            status = "unreadable"
            print(f"  [ERROR] {name}: {e}")

        target = good_dir if status == "good" else bad_dir
        shutil.copy2(path, os.path.join(target, name))
        if status == "good":
            n_good += 1

        dmin_txt = f"{dmin:.2f}" if dmin is not None else "NA"
        print(f"  [{status.upper():10s}] {name:35s} min_dist={dmin_txt} A ({pair})")

        rows.append({
            "cif": name,
            "status": status,
            "min_dist_A": dmin_txt,
            "closest_pair": pair,
            "n_atoms": natoms,
            "metals": metals,
            "formula": formula,
        })

    summary_path = os.path.join(args.out_dir, "cif_check_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    total = len(cif_files)
    print(f"\nPassed: {n_good}/{total} ({100 * n_good / total:.0f}%)")
    print(f"Good CIFs   -> {good_dir}")
    print(f"Broken CIFs -> {bad_dir}")
    print(f"Summary     -> {summary_path}")


if __name__ == "__main__":
    main()
