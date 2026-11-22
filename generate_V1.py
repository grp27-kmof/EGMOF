#!/usr/bin/env python3
"""
STEP 5 (Mac, inside the EGMOF folder): generate MOFs for target V1 values
with the retrained Prop2Desc, and build their CIFs.

Usage:
  uv run python generate_V1.py --data k_v1_data --targets 0.8 1.0 1.2 --n 100
  # one kind of organic ligand per MOF (best-matching ligand chosen by WMSE):
  uv run python generate_V1.py --data k_v1_data --targets 1.0 --n 100 \
         --single-ligand --topk 20 --out generated_single
  # experimental-style MOFs: metal node + ONE kind of 2-connected linker:
  uv run python generate_V1.py --data k_v1_data --targets 1.0 --n 100 \
         --organic-mode single-linker --topk 20 --out generated_single_linker
  # one ligand type (3+ connections) + one linker type (2 connections) per MOF:
  uv run python generate_V1.py --data k_v1_data --targets 1.0 --n 100 \
         --ligand-linker --topk 30 --out generated_ligand_linker
  # keep sampling in rounds until exactly 50 unique valid ligand+linker MOFs exist:
  uv run python generate_V1.py --data k_v1_data --targets 1.0 --n-valid 50 \
         --organic-mode ligand-linker --topk 30 --out generated_ligand_linker_50

Every CIF is checked with egmof.linker_check.check_structure() before it is written
(lower-ranked beam candidates are tried when the best one fails), and all CIFs in
<out>/V1_x.xx/cifs/ are re-checked at the end: failures are moved to rejected/ and
listed in linker_report.csv.

With --n-valid N, descriptors are sampled in rounds of --chunk (seed = --seed + round)
until N unique valid CIFs are collected in <out>/V1_x.xx/cifs/ (or --max-rounds is
reached). Each round has its own rounds/round_XX/new_bbs/; kept CIFs whose name
contains a round-local Gen_N*/Gen_E* block are saved as <name>__rXX.cif. Duplicates
(same CIF name, or same topology + metal node + linker InChIKeys) and failures go
to rejected/; valid unique CIFs found after N is reached go to extra/. See
summary.csv and attempts.csv (rewritten after every round). An interrupted run is
continued with --resume and the same --out.

With --exclude-dirs generation/batch1 generation/batch2 ..., every CIF there (and every
recipe in generation/seen_recipes.txt) counts as already generated: such beam candidates
are skipped before PORMAKE builds them and the next candidate is tried; kept recipes are
appended to seen_recipes.txt after every round. Each round prints how every candidate
ended (built OK / duplicate of previous batch / duplicate within run / overlap / other).
Sampling knobs: --noise-scale (Prop2Desc), --decode-mode sample --temperature --top-p
(Desc2MOF nucleus sampling instead of beam search).
"""
import argparse
import glob
import json
import os
import random
import re
import shutil
import subprocess
from collections import Counter
from datetime import datetime
from importlib import metadata
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from egmof import EGMOF
from egmof.linker_check import (
    MIN_DIST_THRESHOLD,
    check_structure,
    expected_unique_for_mode,
    strip_run_tags,
    topology_node_linker_key,
)

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="k_v1_data folder (holds rf_V1.pkl)")
ap.add_argument("--targets", type=float, nargs="+", default=[1.0], help="target V1 values (V)")
ap.add_argument("--n", type=int, default=100,
                help="descriptors sampled per target (single pass; ignored with --n-valid)")
ap.add_argument("--n-valid", type=int, default=None,
                help="sample in rounds until exactly N unique valid CIFs exist per target")
ap.add_argument("--chunk", type=int, default=250,
                help="descriptors sampled per round with --n-valid")
ap.add_argument("--max-rounds", type=int, default=20,
                help="maximum number of rounds with --n-valid")
ap.add_argument("--exclude-from", type=Path, nargs="+", default=[],
                help="earlier output folders (e.g. generation/batch2); their kept CIFs "
                     "(**/cifs/*.cif) count as duplicates, so they are not generated again")
ap.add_argument("--exclude-dirs", type=Path, nargs="*", default=None, metavar="DIR",
                help="earlier output folders: EVERY CIF below them (kept, extra, rejected, "
                     "rounds) plus the recipes in --seen-file count as already generated; "
                     "such beam candidates are skipped before PORMAKE builds them and the "
                     "next candidate is tried. Kept recipes are appended to --seen-file "
                     "after every round. '--exclude-dirs' alone uses only --seen-file")
ap.add_argument("--seen-file", type=Path, default=Path("generation/seen_recipes.txt"),
                help="recipe list read and appended with --exclude-dirs")
ap.add_argument("--exclusion-cache", type=Path, default=Path("generation/exclusion_cache.csv"),
                help="cached check results of excluded CIFs (re-checked when the mtime changes)")
ap.add_argument("--noise-scale", type=float, default=1.0,
                help="Prop2Desc: multiplier on the reverse-diffusion noise (default 1.0)")
ap.add_argument("--decode-mode", choices=["beam", "sample"], default="beam",
                help="Desc2MOF: beam search (default) or nucleus sampling of --topk sequences")
ap.add_argument("--temperature", type=float, default=1.0,
                help="Desc2MOF softmax temperature (default 1.0; also applies to beam scores)")
ap.add_argument("--top-p", type=float, default=0.9,
                help="Desc2MOF nucleus mass for --decode-mode sample (default 0.9)")
ap.add_argument("--resume", action="store_true",
                help="continue an interrupted --n-valid run in the same --out "
                     "(keeps its CIFs, sorts unprocessed round CIFs, then generates new rounds)")
ap.add_argument("--seed", type=int, default=0,
                help="base random seed; round r of --n-valid uses seed + r")
ap.add_argument("--min-dist", type=float, default=MIN_DIST_THRESHOLD,
                help=f"minimum allowed interatomic distance in A (default {MIN_DIST_THRESHOLD})")
ap.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
ap.add_argument("--decode-device", choices=["cpu", "cuda", "mps"], default=None,
                help="device for desc2mof/mof2desc; default cpu when --device mps, else same as --device")
ap.add_argument("--out", default="generated_V1")
ap.add_argument("--batch-size", type=int, default=None,
                help="inference batch size; default 256 on cpu/cuda, 32 on mps")
ap.add_argument("--topk", type=int, default=5,
                help="beam width per descriptor (use 20 with --single-ligand)")
ap.add_argument("--organic-mode", default="any",
                choices=["any", "single", "single-linker", "ligand-linker"],
                help="any = original EGMOF; single = one kind of organic block; "
                     "single-linker = metal node + one 2-connected linker (ZIF-8/UiO-66 type); "
                     "ligand-linker = one multi-arm ligand + one linker")
ap.add_argument("--ligand-linker", action="store_true",
                help="keep only MOFs with exactly one kind of ligand (3+ connections) "
                     "and exactly one kind of linker (2 connections)")
ap.add_argument("--repair", dest="repair", action="store_true",
                help="relax overlapping CIF candidates once with CHGNet (if installed; fixed "
                     "cell, FIRE fmax 0.1, 200 steps) and re-run the full check")
ap.add_argument("--repair-max-per-descriptor", type=int, default=2,
                help="--repair: relax at most this many overlapping candidates per descriptor")
ap.add_argument("--repair-budget-min", type=float, default=20.0,
                help="--repair: stop relaxing in a round after this many minutes of CHGNet time")
ap.add_argument("--no-repair", dest="repair", action="store_false",
                help="disable CHGNet repair (default)")
ap.set_defaults(repair=False)
ap.add_argument("--decode-constraint", action=argparse.BooleanOptionalAction, default=False,
                help="single-linker constrained beam search (default OFF = original EGMOF decoding)")
ap.add_argument("--single-ligand", action="store_true",
                help="keep only MOFs with exactly one kind of organic ligand")
args = ap.parse_args()
if args.n_valid is not None and args.n_valid < 1:
    ap.error("--n-valid must be >= 1")
if args.chunk < 1 or args.max_rounds < 1:
    ap.error("--chunk and --max-rounds must be >= 1")
if args.temperature <= 0 or not 0 < args.top_p <= 1 or args.noise_scale < 0:
    ap.error("need --temperature > 0, 0 < --top-p <= 1 and --noise-scale >= 0")
if args.device == "mps" and not torch.backends.mps.is_available():
    print("WARNING: Apple MPS is not available; falling back to CPU.")
    args.device = "cpu"
if args.batch_size is None:
    args.batch_size = 32 if args.device == "mps" else 256
if args.decode_device is None:
    args.decode_device = "cpu" if args.device == "mps" else args.device
print(f"Sampling device: {args.device} | Decoding device: {args.decode_device}")
# rough CHGNet cost estimate: 0.35 min of repair time per descriptor of a round
round_size = args.chunk if args.n_valid is not None else args.n
if args.repair and args.repair_budget_min < round_size * 0.35 * 60 / 60:
    print(f"WARNING: repair budget may be too small for this chunk size "
          f"(--repair-budget-min {args.repair_budget_min} < {round_size} x 0.35 = "
          f"{round_size * 0.35:.1f} min)")

# best checkpoint = the one that is not last.ckpt
ckpts = [c for c in glob.glob("checkpoints/prop2desc_V1/*.ckpt") if not c.endswith("last.ckpt")]
if not ckpts:
    raise SystemExit("No trained checkpoint in checkpoints/prop2desc_V1/ (run train_V1.sh first)")
ckpt = max(ckpts, key=lambda p: Path(p).stat().st_mtime)
print("Using checkpoint:", ckpt)

egmof = EGMOF(
    load_pretrained_modules=True,                      # desc2mof + mof2desc stay pretrained
    prop2desc_ckpt_path=ckpt,
    prop2desc_config_path="config/k_ion/V1.yaml",
    skmodel_ckpt_dir=str(Path(args.data) / "rf_V1.pkl"),
    accelerator=args.device,
    devices=1,
)


def verify_cifs(out: Path, expected_unique: int | None) -> pd.DataFrame:
    """Re-check every CIF in out/cifs; move failures to out/rejected; write linker_report.csv."""
    cif_dir, rejected_dir = out / "cifs", out / "rejected"
    rows = []
    for cif in sorted(cif_dir.glob("*.cif")):
        # mof_name / new_bbs folder are taken from the CIF path (linker_check.default_new_bb_dir)
        res = check_structure(cif, expected_unique=expected_unique, min_dist_threshold=args.min_dist)
        rows.append({
            "cif": cif.name,
            "passed": res["passed"],
            "reason": res["reason"],
            "n_unique_graphs": res["n_unique_graphs"],
            "linker_smiles": ";".join(res["linker_smiles"]),
            "linker_inchikeys": ";".join(res["linker_inchikeys"]),
            "min_dist": res["min_dist"],
            "linker_reference": res["linker_reference"],
            "chemistry_issue": res["chemistry_issue"],
        })
        if not res["passed"]:
            rejected_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(cif), str(rejected_dir / cif.name))
            print(f"WARNING: final check failed for {cif.name} ({res['reason']}) "
                  f"-> moved to {rejected_dir}")
    report = pd.DataFrame(rows, columns=["cif", "passed", "reason", "n_unique_graphs",
                                         "linker_smiles", "linker_inchikeys", "min_dist",
                                         "linker_reference", "chemistry_issue"])
    report.to_csv(out / "linker_report.csv", index=False)
    return report


# unique linker graphs per CIF: 1 (single, single-linker), 2 (ligand-linker), None (any)
EXPECTED_UNIQUE = expected_unique_for_mode(args.organic_mode, args.single_ligand, args.ligand_linker)

SUMMARY_COLUMNS = ["name", "round", "candidate_rank", "min_dist", "n_unique_graphs",
                   "linker_smiles", "linker_inchikey", "predicted_property", "repaired",
                   "decode_mode"]


def write_run_config(out: Path) -> None:
    """<out>/run_config.json: CLI arguments, time, git commit and package versions.

    --resume keeps the original file and writes run_config_resume_<time>.json instead.
    """
    now = datetime.now()
    path = out / "run_config.json"
    if path.exists():
        path = out / f"run_config_resume_{now:%Y%m%d-%H%M%S}.json"

    def git(*cmd: str) -> str | None:
        try:
            result = subprocess.run(["git", *cmd], cwd=Path(__file__).resolve().parent,
                                    capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):  # git missing / hung
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    def version(package: str) -> str | None:
        try:
            return metadata.version(package)
        except metadata.PackageNotFoundError:
            return None

    status = git("status", "--porcelain", "--untracked-files=no")
    config = {
        "datetime": now.isoformat(timespec="seconds"),
        "args": vars(args),
        "prop2desc_checkpoint": ckpt,
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),  # uncommitted tracked changes
        "versions": {pkg: version(pkg) for pkg in ("torch", "pormake", "pymatgen", "chgnet", "rdkit")},
    }
    path.write_text(json.dumps(config, indent=2, default=str))
    print(f"run config: {path}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA / MPS generators


def generate_round(
    t: float,
    num_samples: int,
    cif_dir: Path,
    new_bb_dir: Path,
    prev_names: set[str],
    prev_keys: set[tuple],
    run_names: set[str] | None = None,
    run_keys: set[tuple] | None = None,
) -> pd.DataFrame:
    """One call of the existing EGMOF pipeline with the CLI options.

    prev_* (earlier batches) and run_* (kept earlier in this run) duplicates are skipped
    inside egmof.generate(), which then tries the next beam candidate.
    """
    return egmof.generate(
        num_samples=num_samples,
        target_value=t,
        topk=args.topk,
        temperature=args.temperature,
        batch_size=args.batch_size,
        decode_device=args.decode_device,
        single_ligand=args.single_ligand,
        ligand_linker=args.ligand_linker,
        organic_mode=args.organic_mode,
        repair=args.repair,
        repair_max_per_descriptor=args.repair_max_per_descriptor,
        repair_budget_min=args.repair_budget_min,
        decode_constraint=args.decode_constraint,
        min_dist=args.min_dist,
        noise_scale=args.noise_scale,
        decode_mode=args.decode_mode,
        top_p=args.top_p,
        exclude_names=prev_names,
        exclude_keys=prev_keys,
        run_names=run_names,
        run_keys=run_keys,
        build_cif=True,
        cif_dir=str(cif_dir),
        new_bb_dir=str(new_bb_dir),
    )


def load_exclusions(out: Path) -> tuple[set[str], set[tuple]]:
    """Names / recipes and topology+node+linker keys of earlier batches.

    --exclude-dirs: every CIF below the folders plus the recipes in --seen-file;
    --exclude-from: only their kept CIFs (**/cifs/*.cif outside rounds/). CIF names are
    stripped of the rXX_ / __rXX tags. Names with round-local Gen_* blocks cannot be
    compared (Gen numbers restart per new_bbs/), so --exclude-dirs also adds the recipe
    (SELFIES form, "filename") of every row with a CIF in the generated.csv files below
    the folders. Keys come from kept / extra CIFs that pass the check. Files inside the
    current --out are ignored.
    """
    names: set[str] = set()
    keys: set[tuple] = set()
    exclude_dirs = args.exclude_dirs or []
    if args.exclude_dirs is None and not args.exclude_from:
        return names, keys
    out = out.resolve()

    def outside_out(path: Path) -> bool:
        return out not in path.resolve().parents

    files = {cif for folder in exclude_dirs for cif in folder.glob("**/*.cif") if outside_out(cif)}
    files |= {cif for folder in args.exclude_from for cif in folder.glob("**/cifs/*.cif")
              if "rounds" not in cif.parts and outside_out(cif)}
    n_recipes = 0
    for folder in exclude_dirs:
        for csv in folder.glob("**/generated.csv"):
            if not outside_out(csv):
                continue
            table = pd.read_csv(csv, usecols=lambda c: c in ("cif", "filename"))
            if "cif" not in table or "filename" not in table:
                continue
            built = table.loc[table["cif"].notna(), "filename"].dropna()
            names.update(built)
            n_recipes += len(built)

    to_check = []
    for cif in sorted(files):
        local_name = strip_run_tags(cif.stem)
        if "Gen_" not in local_name:
            names.add(local_name)
        if cif.parent.name not in ("cifs", "extra") or "rounds" in cif.parts:
            continue  # rejected / unsorted round CIFs: name only, no key
        to_check.append((cif, local_name))

    # check_structure() results are cached per file; a row is reused only if the file's
    # mtime and the check settings (organic mode, --min-dist) are unchanged
    check_settings = f"{EXPECTED_UNIQUE}|{args.min_dist}"
    cache: dict[str, dict] = {}
    if args.exclusion_cache.exists():
        cached = pd.read_csv(args.exclusion_cache, dtype=str, keep_default_na=False)
        cache = {row["path"]: row for row in cached.to_dict("records")}
    n_keys = n_cached = 0

    def save_cache() -> None:
        args.exclusion_cache.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(list(cache.values()), columns=["path", "mtime", "name", "key", "check"]).to_csv(
            args.exclusion_cache, index=False
        )

    try:
        for i, (cif, local_name) in enumerate(
            tqdm(to_check, desc="exclusions: checking CIFs", unit="cif"), start=1
        ):
            path, mtime = str(cif.resolve()), str(cif.stat().st_mtime_ns)
            row = cache.get(path)
            if row and row["mtime"] == mtime and row.get("check") == check_settings:
                n_cached += 1
            else:
                res = check_structure(cif, expected_unique=EXPECTED_UNIQUE, min_dist_threshold=args.min_dist)
                key = topology_node_linker_key(local_name, res["linker_inchikeys"])
                # "" = no key (invalid structure or unlabelled linker)
                row = {"path": path, "mtime": mtime, "name": local_name, "check": check_settings,
                       "key": json.dumps(key) if is_valid(res) and key is not None else ""}
                cache[path] = row
                if (i - n_cached) % 25 == 0:
                    save_cache()
            if row["key"]:
                topology, metal_nodes, inchikeys = json.loads(row["key"])
                keys.add((topology, tuple(metal_nodes), tuple(inchikeys)))
                n_keys += 1
    finally:  # also on Ctrl-C: checks done so far are not repeated next time
        if to_check:
            save_cache()
    print(f"exclusions: {len(to_check)} kept/extra CIFs, {n_cached} from {args.exclusion_cache}, "
          f"{len(to_check) - n_cached} re-checked")

    n_seen = 0
    if args.exclude_dirs is not None and args.seen_file.exists():
        seen = {line.strip() for line in args.seen_file.read_text().splitlines() if line.strip()}
        names |= seen
        n_seen = len(seen)
    print(f"exclusions: {len(files)} earlier CIFs, {n_recipes} recipes from generated.csv, "
          f"{n_seen} from {args.seen_file if args.exclude_dirs is not None else '-'} "
          f"-> {len(names)} names/recipes, {n_keys} topology+node+linker keys")
    return names, keys


def append_seen(recipes: list[str]) -> None:
    """--exclude-dirs: record newly kept recipes so later batches skip them."""
    if args.exclude_dirs is None or not recipes:
        return
    args.seen_file.parent.mkdir(parents=True, exist_ok=True)
    with args.seen_file.open("a") as f:
        f.writelines(f"{recipe}\n" for recipe in recipes)


def print_build_stats(label: str, n_sampled: int, df: pd.DataFrame) -> None:
    """Per-round explanation of every beam candidate egmof.generate() tried."""
    st = getattr(egmof, "last_build_stats", Counter())
    status = df["build_status"] if "build_status" in df else pd.Series(dtype=str)
    print(f"  {label}: {n_sampled} descriptors -> {len(df)} accepted recipes "
          f"(WMSE<0.5, organic mode '{args.organic_mode}') -> {int((status == 'written').sum())} with a CIF")
    print(f"    candidates: built OK {st['built_ok']} | duplicate of previous batch "
          f"{st['dup_previous']} | duplicate within run {st['dup_within_run']} "
          f"({st['dup_prebuild']} duplicates skipped before PORMAKE) | overlap {st['overlap']} "
          f"| other {st['other']}")
    other = {k.split(":", 1)[1]: v for k, v in st.items() if k.startswith("other:")}
    if other:
        print(f"    other reasons: {dict(sorted(other.items(), key=lambda kv: -kv[1]))}")
    if args.repair:
        n_relaxed = st["repair_attempted"]
        avg = f"{st['repair_seconds'] / n_relaxed:.1f} s/repair" if n_relaxed else "-"
        print(f"    repaired (CHGNet): {st['repaired']} of {n_relaxed} overlapping candidates "
              f"relaxed ({avg}, {st['repair_seconds'] / 60:.1f} min total) | not relaxed: "
              f"{st['overlap_severe']} overlap_severe (min_dist < 0.5 A), "
              f"{st['repair_skipped_limit']} over per-descriptor limit, "
              f"{st['repair_skipped_budget']} over time budget")


def is_valid(res: dict) -> bool:
    """check_structure() passed with the number of linker graphs the mode requires."""
    if EXPECTED_UNIQUE is None:  # organic mode "any": overlap check only
        return bool(res["passed"])
    return res["reason"] == "ok" and res["n_unique_graphs"] == EXPECTED_UNIQUE


def global_name(local_name: str, round_idx: int) -> str:
    """Run-wide CIF name. Gen_N*/Gen_E* numbers restart in every round's new_bbs/
    (make_bb counts per folder), so names containing them are tagged with the round."""
    if any(tok.startswith("Gen_") for tok in local_name.split("+")):
        return f"{local_name}__r{round_idx:02d}"
    return local_name


def dedup_key(name: str, res: dict) -> tuple | None:
    """(topology, metal node(s), linker InChIKeys); None if the linkers could not be labelled."""
    return topology_node_linker_key(name, res["linker_inchikeys"])


def run_single_pass(t: float, out: Path) -> None:
    """Original behaviour: one pass of --n descriptors, then the final check."""
    prev_names, prev_keys = load_exclusions(out)
    df = generate_round(t, args.n, out / "cifs", out / "new_bbs", prev_names, prev_keys)
    out.mkdir(parents=True, exist_ok=True)
    print_build_stats(f"target V1={t:.2f} V", args.n, df)

    # ── final verification of everything in cifs/ (same checker as the build loop) ──
    report = verify_cifs(out, EXPECTED_UNIQUE)
    rejected = set(report.loc[~report["passed"].astype(bool), "cif"])
    if "cif" in df and rejected:
        df.loc[df["cif"].isin(rejected), "build_status"] = "rejected_final_check"
    df.to_csv(out / "generated.csv", index=False)
    if "build_status" in df:
        append_seen(df.loc[df["build_status"] == "written", "filename"].tolist())

    status = df["build_status"] if "build_status" in df else pd.Series(dtype=str)
    written = status == "written"
    n_fallback = int((written & (df["candidate_rank"].fillna(0) > 1)).sum()) if "candidate_rank" in df else 0
    pv = df["pred_value"].describe().round(2).to_dict() if "pred_value" in df else {}
    print(f"\n=== target V1={t:.2f} V ===")
    print(f"  descriptors sampled:     {args.n}")
    print(f"  accepted recipes:        {len(df)} (WMSE<0.5, organic mode '{args.organic_mode}')")
    print(f"  CIFs written:            {int(written.sum())}")
    fail_reason = df["fail_reason"].fillna("").astype(str) if "fail_reason" in df else pd.Series(dtype=str)
    print(f"  no CIF, last was dup:    {int(fail_reason.str.startswith('duplicate_').sum())}")
    print(f"  fallbacks (rank > 1):    {n_fallback}")
    print(f"  repaired:                {int(df['repaired'].fillna(False).astype(bool).sum()) if 'repaired' in df else 0}")
    print(f"  no_valid_structure:      {int((status == 'no_valid_structure').sum())}")
    print(f"  final check passed:      {int(report['passed'].astype(bool).sum())} / {len(report)} CIFs in cifs/")
    print(f"  final check failed:      {len(rejected)} (moved to {out / 'rejected'})")
    print(f"  RF-predicted V1:         {pv}")
    if "n_ligands" in df:
        print(f"  ligands per MOF: {df['n_ligands'].value_counts().sort_index().to_dict()}")
    if "n_ligand_types" in df:
        pairs = (df["n_ligand_types"].astype(str) + " ligand + " + df["n_linker_types"].astype(str) + " linker")
        print(f"  composition: {pairs.value_counts().to_dict()}")
    print(f"  report: {out / 'linker_report.csv'}")


def rows_by_cif_of(df: pd.DataFrame) -> dict:
    """Rows of a generated.csv keyed by CIF file name.

    egmof.generate() stores the bare file name ("<name>.cif") in "cif" and the
    RF-predicted property in "pred_value"; Path(...).name also tolerates full paths.
    """
    if "cif" not in df:
        return {}
    return {Path(r["cif"]).name: r for _, r in df.iterrows() if isinstance(r.get("cif"), str)}


def run_n_valid(t: float, out: Path, n_valid: int) -> None:
    """Sample in rounds until n_valid unique valid CIFs are in out/cifs."""
    cif_dir, rejected_dir, extra_dir = out / "cifs", out / "rejected", out / "extra"
    rounds_root = out / "rounds"
    previous_run = (cif_dir.exists() and any(cif_dir.glob("*.cif"))) or rounds_root.exists()
    if previous_run and not args.resume:
        raise SystemExit(f"{out} already holds a previous run; "
                         f"use --resume to continue it, or a new --out")
    for folder in (cif_dir, rejected_dir, extra_dir):
        folder.mkdir(parents=True, exist_ok=True)
    write_run_config(out)

    kept: list[dict] = []
    attempts: list[dict] = []
    kept_names: set[str] = set()
    kept_keys: set[tuple] = set()
    kept_recipes: set[str] = set()  # recipe ("filename", SELFIES form) of every kept CIF
    kept_round: dict[str, int | None] = {}  # kept name -> round it came from
    prev_names, prev_keys = load_exclusions(out)  # earlier batches (--exclude-dirs/-from)

    def reject(cif: Path, round_idx: int, reason: str) -> None:
        shutil.move(str(cif), str(rejected_dir / f"r{round_idx:02d}_{cif.name}"))
        attempts.append({"name": cif.stem, "round": round_idx, "status": "rejected", "reason": reason})

    def set_aside(cif: Path, name: str, round_idx: int) -> None:
        """Valid and unique, but N was already reached."""
        shutil.move(str(cif), str(extra_dir / f"{name}.cif"))
        attempts.append({"name": name, "round": round_idx, "status": "extra",
                         "reason": "valid, surplus after n-valid reached"})

    def keep(res: dict, name: str, local_name: str, round_idx: int | None, row) -> None:
        kept_names.add(name)
        kept_round[name] = round_idx
        key = dedup_key(local_name, res)
        if key is not None:
            kept_keys.add(key)
        recipe = row.get("filename")
        if isinstance(recipe, str):
            kept_recipes.add(recipe)
        elif "Gen_" not in local_name:
            kept_recipes.add(local_name)
        kept.append({
            "name": name,
            "round": round_idx,
            "candidate_rank": row.get("candidate_rank"),
            "min_dist": res["min_dist"],
            "n_unique_graphs": res["n_unique_graphs"],
            "linker_smiles": ";".join(res["linker_smiles"]),
            "linker_inchikey": ";".join(res["linker_inchikeys"]),
            "predicted_property": row.get("pred_value"),
            # generated.csv values are bools or "True"/"False" strings after --resume
            "repaired": str(row.get("repaired", False)) == "True",
            "decode_mode": row.get("decode_mode", args.decode_mode),
        })

    def write_tables() -> None:
        """Written after every round, so a crash keeps the state for --resume."""
        pd.DataFrame(kept, columns=SUMMARY_COLUMNS).to_csv(out / "summary.csv", index=False)
        pd.DataFrame(attempts, columns=["name", "round", "status", "reason"]).to_csv(
            out / "attempts.csv", index=False
        )

    def harvest(
        round_idx: int, round_dir: Path, rows_by_cif: dict, skip_kept: bool = False
    ) -> tuple[int, Counter]:
        """Check, deduplicate and sort the CIFs of one round.

        Returns the number kept and the reasons of the CIFs not kept. egmof.generate()
        already skipped duplicates, so rejections here are only a safety net.
        """
        n_round_kept = 0
        not_kept: Counter = Counter()
        new_recipes: list[str] = []
        for cif in sorted((round_dir / "cifs").glob("*.cif")):
            name = global_name(cif.stem, round_idx)
            if skip_kept and kept_round.get(name, -1) == round_idx:
                continue  # --resume: this round's own copy of a kept CIF (older runs copied)
            row = rows_by_cif.get(cif.name, {})
            recipe = row.get("filename") if isinstance(row.get("filename"), str) else None
            # new_bbs of this round: rounds/round_XX/cifs/<name>.cif -> rounds/round_XX/new_bbs
            res = check_structure(cif, expected_unique=EXPECTED_UNIQUE, min_dist_threshold=args.min_dist)
            key = dedup_key(cif.stem, res)
            if not is_valid(res):
                reason = res["reason"]
            elif recipe in prev_names or ("Gen_" not in cif.stem and cif.stem in prev_names):
                reason = "duplicate_previous_batch_name"
            elif key is not None and key in prev_keys:
                reason = "duplicate_previous_batch_topology_node_linker"
            elif name in kept_names or recipe in kept_recipes:
                reason = "duplicate_name"
            elif key is not None and key in kept_keys:
                reason = "duplicate_topology_node_linker"
            else:
                reason = ""
            if reason:
                reject(cif, round_idx, reason)
                not_kept[reason] += 1
                continue
            if len(kept) >= n_valid:
                set_aside(cif, name, round_idx)
                not_kept["extra (n-valid reached)"] += 1
                continue

            # moved, not copied: a CIF left in rounds/*/cifs/ is always unprocessed
            shutil.move(str(cif), str(cif_dir / f"{name}.cif"))
            keep(res, name, cif.stem, round_idx, row)
            attempts.append({"name": name, "round": round_idx, "status": "kept", "reason": "ok"})
            new_recipes.append(recipe or cif.stem)
            n_round_kept += 1
        append_seen(new_recipes)
        return n_round_kept, not_kept

    def restore() -> int:
        """--resume: rebuild the state from cifs/, rounds/ and attempts.csv.

        Returns the index of the next round to generate.
        """
        round_dirs = sorted(rounds_root.glob("round_*"))
        tables = {
            int(rd.name.split("_")[1]): rows_by_cif_of(pd.read_csv(rd / "generated.csv"))
            for rd in round_dirs
            if (rd / "generated.csv").exists()
        }
        if (out / "attempts.csv").exists():
            attempts.extend(pd.read_csv(out / "attempts.csv").to_dict("records"))

        # 1. CIFs already kept: re-check, recover round / rank / property from generated.csv
        for cif in sorted(cif_dir.glob("*.cif")):
            res = check_structure(cif, expected_unique=EXPECTED_UNIQUE, min_dist_threshold=args.min_dist)
            if not is_valid(res):
                shutil.move(str(cif), str(rejected_dir / cif.name))
                print(f"WARNING: --resume: {cif.name} fails the check ({res['reason']}) -> rejected/")
                continue
            match = re.search(r"__r(\d+)$", cif.stem)
            local_name = re.sub(r"__r\d+$", "", cif.stem)
            if match:
                round_idx = int(match.group(1))
            else:
                # older runs copied kept CIFs, so the round folder holding the copy is the
                # source; otherwise the first generated.csv that lists the name
                round_idx = next(
                    (int(rd.name.split("_")[1]) for rd in round_dirs
                     if (rd / "cifs" / f"{local_name}.cif").exists()),
                    next((i for i, rows in sorted(tables.items())
                          if f"{local_name}.cif" in rows), None),
                )
            row = tables.get(round_idx, {}).get(f"{local_name}.cif", {})
            keep(res, cif.stem, local_name, round_idx, row)
        print(f"--resume: {len(kept)} valid CIFs restored from {cif_dir}")

        # 2. CIFs a crashed round wrote but never sorted
        for rd in round_dirs:
            round_idx = int(rd.name.split("_")[1])
            n_new, _ = harvest(round_idx, rd, tables.get(round_idx, {}), skip_kept=True)
            if n_new:
                print(f"--resume: round {round_idx + 1}: {n_new} unsorted valid CIFs kept "
                      f"-> {len(kept)}/{n_valid}")
        write_tables()
        return max((int(rd.name.split("_")[1]) for rd in round_dirs), default=-1) + 1

    first_round = restore() if previous_run else 0
    # --resume: this run's own kept recipes may already be in --seen-file
    prev_names -= kept_recipes
    for round_idx in range(first_round, args.max_rounds):
        if len(kept) >= n_valid:
            break
        seed = args.seed + round_idx
        seed_everything(seed)
        round_dir = rounds_root / f"round_{round_idx:02d}"
        print(f"\n--- target V1={t:.2f} V | round {round_idx + 1}/{args.max_rounds} | "
              f"seed {seed} | {len(kept)}/{n_valid} valid so far ---")
        # own new_bbs/ per round: egmof.generate() passes it to make_bb (writes Gen_*.xyz)
        # and to name_to_mof (PORMAKE reads Gen_*.xyz), so every CIF is built from the
        # building blocks of its own round
        df = generate_round(t, args.chunk, round_dir / "cifs", round_dir / "new_bbs",
                            prev_names, prev_keys, kept_recipes, kept_keys)
        round_dir.mkdir(parents=True, exist_ok=True)
        df["decode_mode"] = args.decode_mode  # read back by keep(), also after --resume
        df.to_csv(round_dir / "generated.csv", index=False)

        n_round_kept, not_kept = harvest(round_idx, round_dir, rows_by_cif_of(df))
        write_tables()
        print_build_stats(f"round {round_idx + 1}", args.chunk, df)
        if not_kept:
            print(f"    built but not kept (final sort): {dict(not_kept)}")
        print(f"  round {round_idx + 1}: {n_round_kept} new unique valid CIFs -> {len(kept)}/{n_valid}")

    # ── final check of every file in cifs/ ──
    # verify_cifs() moves every failing file to rejected/ and prints its name
    report = verify_cifs(out, EXPECTED_UNIQUE)
    passed_names = {Path(n).stem for n in report.loc[report["passed"].astype(bool), "cif"]}
    final_names = {path.stem for path in cif_dir.glob("*.cif")}
    assert final_names == passed_names, "a CIF left in cifs/ did not pass the final check"
    kept = [row for row in kept if row["name"] in final_names]
    write_tables()

    n_rounds = len(list(rounds_root.glob("round_*")))
    reasons = pd.Series([a["reason"] for a in attempts if a["status"] == "rejected"])
    print(f"\n=== target V1={t:.2f} V (--n-valid {n_valid}) ===")
    print(f"  rounds run:              {n_rounds} x {args.chunk} descriptors")
    print(f"  unique valid CIFs:       {len(kept)} in {cif_dir}")
    print(f"  rejected attempts:       {len(reasons)} -> {rejected_dir}")
    print(f"  valid extras (> N):      {sum(a['status'] == 'extra' for a in attempts)} -> {extra_dir}")
    if len(reasons):
        print(f"  rejection reasons:       {reasons.value_counts().to_dict()}")
    print(f"  summary: {out / 'summary.csv'} | attempts: {out / 'attempts.csv'}")
    if len(kept) < n_valid:
        print(f"\nWARNING: only {len(kept)} of {n_valid} requested valid MOFs were found for "
              f"V1={t:.2f} V after {n_rounds} rounds (--max-rounds {args.max_rounds}). "
              f"Increase --max-rounds, --chunk or --topk.")


for t in args.targets:
    out = Path(args.out) / f"V1_{t:.2f}"
    if args.n_valid is None:
        run_single_pass(t, out)
    else:
        run_n_valid(t, out, args.n_valid)
