from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import List, Literal, Optional, Union

# import joblib
import numpy as np
import pandas as pd
import torch
import yaml


from .utils import (
    _require_ckpt,
    download_desc2mof,
    download_mof2desc,
    download_prop2desc,
    download_rf,
    download_all,
)

from .prop2desc import (
    Prop2Desc,
    run_train_prop2desc,
)

from .desc2mof import (
    Desc2MOF as Desc2MOFModel,
    Scaler,
)

from .constants import (
    DEFAULT_DESC2MOF_CKPT,
    DEFAULT_DESC2MOF_CONFIG,
    DEFAULT_DESC2MOF_FEATURE_NAME,
    DEFAULT_DESC2MOF_MEAN,
    DEFAULT_DESC2MOF_STD,
    DEFAULT_MOF2DESC_CKPT,
    DEFAULT_MOF2DESC_CONFIG,
)

from .data import Datamodule
from .data.dataset import CSVDataset, TextSplitDataset, JsonSplitDataset
from .train import train_desc2mof, train_mof2desc
from .utils import create_scaler, load_feature_names, load_config, _load_sk_scaler
from .generate import run_desc2mof, run_mof2desc_and_select


class EGMOF:
    """Orchestrator that wires configs → model/datamodule/trainer."""

    def __init__(
        self,
        prop2desc_ckpt_path: Optional[str | Path] = None,
        prop2desc_config_path: Optional[str | Path] = None,
        skmodel_ckpt_dir: Optional[str | Path] = None,
        load_pretrained_modules: bool = True,
        accelerator: Literal["cpu", "cuda", "mps"] = "cpu",
        devices: int | List[int] = 1,
    ) -> None:
        self.prop2desc_ckpt_path: Optional[str | Path] = prop2desc_ckpt_path
        self.prop2desc_config_path: Optional[str | Path] = prop2desc_config_path
        self.skmodel_ckpt_dir: Optional[str | Path] = skmodel_ckpt_dir
        self.load_pretrained_modules: bool = load_pretrained_modules
        self.accelerator: Literal["cpu", "cuda", "mps"] = accelerator
        self.devices: int | List[int] = devices

        self.prop2desc: Optional[Prop2Desc] = None
        self.desc2mof: Optional[Desc2MOFModel] = None
        self.mof2desc: Optional[object] = None
        self.sk_model: Optional[object] = None
        self._sk_feature_importances: Optional[list[float]] = None
        self._sk_scaler: Optional[object] = None
        self.setup()

    def setup(self):
        if self.load_pretrained_modules:
            self._load_pretrained_desc2mof()
            self._load_pretrained_mof2desc()

        if self.prop2desc_ckpt_path and self.prop2desc_config_path:
            self.load_prop2desc(
                ckpt_path=self.prop2desc_ckpt_path,
                config_path=self.prop2desc_config_path,
            )

        self._load_sk_model()

    def _load_sk_model(self):
        """Load sklearn model and/or scaler with feature importances."""
        if self.skmodel_ckpt_dir:
            skmodel_path = _require_ckpt(
                Path(self.skmodel_ckpt_dir).name,
                Path(self.skmodel_ckpt_dir).parent,
            )
            import joblib

            self.sk_model = joblib.load(skmodel_path)
            self._sk_feature_importances = self.sk_model.feature_importances_.tolist()
        else:
            print(
                "WARNING: skmodel not provided. Guided Decoding (WMSE calculation) will be skipped.\n\n"
                "To download sklearn model:\n"
                "  from egmof import download_rf\n"
                "  download_rf()\n"
                f"See: {ZENODO_RECORD}\n\n"
                "Or use prop2desc_config_path containing 'feature_importances' for WMSE.\n"
            )

        if self.prop2desc_config_path:
            scaler, fi_from_yaml = _load_sk_scaler(self.prop2desc_config_path)
            self._sk_scaler = scaler
            if self._sk_feature_importances is None:
                self._sk_feature_importances = fi_from_yaml

    def _load_pretrained_desc2mof(self):
        self.load_desc2mof(
            ckpt_path=DEFAULT_DESC2MOF_CKPT,
            config_path=DEFAULT_DESC2MOF_CONFIG,
            mean_path=DEFAULT_DESC2MOF_MEAN,
            std_path=DEFAULT_DESC2MOF_STD,
            feature_name_path=DEFAULT_DESC2MOF_FEATURE_NAME,
        )

    def _load_pretrained_mof2desc(self):
        if not hasattr(self, "_desc2mof_scaler") or self._desc2mof_scaler is None:
            self._load_pretrained_desc2mof()
        self.load_mof2desc(
            ckpt_path=DEFAULT_MOF2DESC_CKPT,
            config_path=DEFAULT_MOF2DESC_CONFIG,
            scaler=self._desc2mof_scaler,
        )

    def load_prop2desc(self, ckpt_path: str | Path, config_path: str | Path) -> None:
        ckpt_path = _require_ckpt(
            Path(ckpt_path).name,
            Path(ckpt_path).parent,
        )
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(
                f"prop2desc config not found: {config_path}\n"
                f"Expected: {config_path}\n\n"
                f"To download:\n"
                f"  from egmof import download_prop2desc\n"
                f"  download_prop2desc()\n\n"
                f"See: {ZENODO_RECORD}"
            )

        if self.prop2desc is not None:
            return
        self.prop2desc = Prop2Desc.load(
            ckpt_path=ckpt_path, config_path=config_path
        ).to(device=self.accelerator)

    def load_desc2mof(
        self,
        ckpt_path: str | Path,
        config_path: str | Path,
        mean_path: str | Path,
        std_path: str | Path,
        feature_name_path: str | Path,
    ) -> None:
        """Load desc2mof model from checkpoint.

        Args:
            ckpt_path: Path to desc2mof checkpoint (.ckpt)
            config_path: Path to desc2mof config YAML
            mean_path: Path to mean CSV for scaler
            std_path: Path to std CSV for scaler
            feature_name_path: Path to feature name text file
        """
        ckpt_path = _require_ckpt(
            Path(ckpt_path).name,
            Path(ckpt_path).parent,
        )
        config_path = Path(config_path)
        mean_path = Path(mean_path)
        std_path = Path(std_path)
        feature_name_path = Path(feature_name_path)

        for path, name in [
            (config_path, "desc2mof config"),
            (mean_path, "desc2mof mean"),
            (std_path, "desc2mof std"),
            (feature_name_path, "feature name"),
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"{name} not found: {path}\n"
                    f"Expected: {path}\n\n"
                    f"To download:\n"
                    f"  from egmof import download_desc2mof\n"
                    f"  download_desc2mof()\n\n"
                    f"See: {ZENODO_RECORD}"
                )

        # Load config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Load scaler
        with open(feature_name_path, "r") as f:
            feature_names = [line.strip() for line in f.readlines()]

        mean = pd.read_csv(mean_path)[feature_names]
        std = pd.read_csv(std_path)[feature_names]
        scaler = Scaler(
            np.array(mean).squeeze(),
            np.array(std).squeeze(),
            0,  # target_mean
            1,  # target_std
        )

        # Load model
        self.desc2mof = Desc2MOFModel.load_from_checkpoint(
            ckpt_path,
            config=config,
            strict=False,
            weights_only=False,
        )

        # Store for later use
        self._desc2mof_scaler = scaler
        self._desc2mof_feature_names = feature_names
        self._desc2mof_config = config

    def load_mof2desc(
        self,
        ckpt_path: str | Path,
        config_path: str | Path,
        scaler: Scaler,
    ) -> None:
        """Load mof2desc model from checkpoint.

        Args:
            ckpt_path: Path to mof2desc checkpoint (.ckpt)
            config_path: Path to mof2desc config YAML
            scaler: Scaler for descriptor normalization (can use desc2mof's scaler)
        """
        ckpt_path = _require_ckpt(
            Path(ckpt_path).name,
            Path(ckpt_path).parent,
        )
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(
                f"mof2desc config not found: {config_path}\n"
                f"Expected: {config_path}\n\n"
                f"To download:\n"
                f"  from egmof import download_mof2desc\n"
                f"  download_mof2desc()\n\n"
                f"See: {ZENODO_RECORD}"
            )

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        from .mof2desc import MOF2Desc as MOF2DescModel

        self.mof2desc = MOF2DescModel.load_from_checkpoint(
            ckpt_path,
            config=config,
            scaler=scaler,
            strict=False,
            weights_only=False,
        )

    def train(
        self,
        data_path: Optional[str | Path] = None,
        task: Optional[str] = None,
        prop2desc_config_path: Optional[str | Path] = None,
        desc2mof_config_path: Optional[str | Path] = None,
        mof2desc_config_path: Optional[str | Path] = None,
    ):
        if data_path and self.prop2desc is None:
            self.train_prop2desc(data_path, task, prop2desc_config_path)
        if self.desc2mof is None:
            self.train_desc2mof(desc2mof_config_path)
        if self.mof2desc is None:
            self.train_mof2desc(mof2desc_config_path)

    def train_prop2desc(
        self,
        data_path: str | Path = None,
        task: Optional[str] = None,
        prop2desc_config_path: Optional[str | Path] = None,
    ) -> Prop2Desc:
        """train prop2desc model"""
        self.prop2desc = run_train_prop2desc(
            config_path=prop2desc_config_path,
            data_path=data_path,
            task=task,
            accelerator=self.accelerator,
            devices=self.devices,
        )
        return self.prop2desc

    def train_desc2mof(
        self,
        config_path: Optional[str | Path] = None,
        train_data_dir: Optional[str | Path] = None,
        val_data_dir: Optional[str | Path] = None,
        test_data_dir: Optional[str | Path] = None,
        accelerator: str = "gpu",
        devices: int = 1,
        max_epochs: int = 200,
        log_dir: str = "./logs_desc2mof",
        ckpt_dir: str = "./ckpt_desc2mof",
    ) -> Desc2MOFModel:
        if config_path is None:
            config_path = DEFAULT_DESC2MOF_CONFIG

        config = load_config(config_path)

        feature_names = load_feature_names(
            config.get("feature_name_dir", DEFAULT_DESC2MOF_FEATURE_NAME)
        )

        if hasattr(self, "_desc2mof_scaler") and self._desc2mof_scaler is not None:
            scaler = self._desc2mof_scaler
        else:
            scaler = create_scaler(
                config["mean_dir"],
                config["std_dir"],
                feature_names,
            )

        train_dir = train_data_dir or config.get("train_data_dir")
        val_dir = val_data_dir or config.get("val_data_dir")
        test_dir = test_data_dir or config.get("test_data_dir")

        self.desc2mof = train_desc2mof(
            config=config,
            scaler=scaler,
            train_dir=train_dir,
            val_dir=val_dir,
            test_dir=test_dir,
            accelerator=accelerator,
            devices=devices,
            max_epochs=max_epochs,
            log_dir=log_dir,
            ckpt_dir=ckpt_dir,
        )

        return self.desc2mof

    def train_mof2desc(
        self,
        config_path: Optional[str | Path] = None,
        train_data_dir: Optional[str | Path] = None,
        valid_data_dir: Optional[str | Path] = None,
        test_data_dir: Optional[str | Path] = None,
        accelerator: str = "gpu",
        devices: int = 1,
        max_epochs: int = 500,
        log_dir: str = "./logs_mof2desc",
        ckpt_dir: str = "./ckpt_mof2desc",
    ):
        if config_path is None:
            config_path = DEFAULT_MOF2DESC_CONFIG

        config = load_config(config_path)

        feature_names = load_feature_names(
            config.get("feature_name_dir", DEFAULT_DESC2MOF_FEATURE_NAME)
        )

        if hasattr(self, "_desc2mof_scaler") and self._desc2mof_scaler is not None:
            scaler = self._desc2mof_scaler
        else:
            scaler = create_scaler(
                config["mean_dir"],
                config["std_dir"],
                feature_names,
            )

        train_dir = train_data_dir or config.get("train_data_dir")
        valid_dir = valid_data_dir or config.get("valid_data_dir")
        test_dir = test_data_dir or config.get("test_data_dir")

        self.mof2desc = train_mof2desc(
            config=config,
            scaler=scaler,
            train_dir=train_dir,
            valid_dir=valid_dir,
            test_dir=test_dir,
            accelerator=accelerator,
            devices=devices,
            max_epochs=max_epochs,
            log_dir=log_dir,
            ckpt_dir=ckpt_dir,
        )

        return self.mof2desc

    def train_sk_model(self, sk_model_config_path: str | Path):
        # TODO: Train sklearn model. (make train_sklearn.py and call it here.)
        raise NotImplementedError("Training sklearn model is not implemented yet.")

    def _generate(
        self,
        num_samples: int = 100,
        target_value: Optional[float | int] = None,
        output_type: Literal["df", "token"] = "df",
        topk: int = 5,
        temperature: float = 1.0,
        wmse_target: float = 0.5,
        batch_size: int = 256,
        decode_device: Optional[str] = None,
        repair: bool = False,
        num_workers: int = 0,
        return_descriptor: bool = False,
        single_ligand: bool = False,
        ligand_linker: bool = False,
        organic_mode: str = "any",
        decode_constraint: bool = False,
        noise_scale: float = 1.0,
        decode_mode: Literal["beam", "sample"] = "beam",
        top_p: float = 0.9,
    ) -> Union[pd.DataFrame, List[str]]:
        """Generate MOF structures from target property.

        Pipeline:
        1. prop2desc.sample() → descriptor tensor
        2. desc2mof inference → MOF tokens (beam search)
        3. mof2desc + wmse → select best MOFs

        Args:
            num_samples: Number of MOFs to generate per target
            target_value: Target property value for conditional generation
            output_type: 'df' returns DataFrame, 'token' returns MOF name list
            topk: Beam width for desc2mof beam search
            temperature: Sampling temperature for desc2mof
            wmse_target: WMSE threshold for filtering generated MOFs
            batch_size: Batch size for inference
            num_workers: Num workers for DataLoader
            return_descriptor: If True, also return predicted descriptors DataFrame
            single_ligand: If True, keep only MOFs with exactly one kind of organic
                ligand (the best-matching ligand by WMSE is chosen)
            decode_constraint: If True, apply the single-linker constrained decoding
                inside beam search. If False, beam search is the original EGMOF one.
            noise_scale: Multiplier on the prop2desc reverse-diffusion noise (1.0 = default)
            decode_mode: "beam" (default) keeps the topk beams; "sample" draws topk
                sequences per descriptor by nucleus sampling
            top_p: Nucleus mass for decode_mode="sample"

        Returns:
            DataFrame with generated MOFs (if output_type='df') or list of MOF names.
            If return_descriptor=True, returns (result, descriptor_df) tuple.
        """
        if self.prop2desc is None:
            raise ValueError("prop2desc not loaded. Please load or train first.")

        sampling_device = self.accelerator if self.accelerator in ["cuda", "cpu", "mps"] else "cpu"
        device = decode_device or sampling_device
        if device not in ["cuda", "cpu", "mps"]:
            device = "cpu"

        self.prop2desc = self.prop2desc.to(device=sampling_device)
        if sampling_device == "mps":
            self.prop2desc = self.prop2desc.float()
        self.desc2mof = self.desc2mof.to(device=device)
        self.mof2desc = self.mof2desc.to(device=device)
        if device == "mps":
            self.prop2desc = self.prop2desc.float()
            self.desc2mof = self.desc2mof.float()
            self.mof2desc = self.mof2desc.float()
            if getattr(self, "_desc2mof_scaler", None) is not None:
                self._desc2mof_scaler.to(dtype=torch.float32, device=device)

        desc_tensor = self.prop2desc.sample(
            num_samples=num_samples,
            target=target_value,
            noise_scale=noise_scale,
        )
        if len(desc_tensor.shape) == 3:
            desc_tensor = desc_tensor.squeeze(1)
        desc_tensor = desc_tensor.detach().cpu()

        if self.desc2mof is None or self.mof2desc is None:
            raise ValueError(
                "desc2mof/mof2desc not loaded. Please load or train first."
            )

        desc2mof_kwargs = {}
        if decode_constraint:
            desc2mof_kwargs["single_linker_constraint"] = True
        all_output, mof_names, target_data = run_desc2mof(
            model=self.desc2mof.model,
            target_desc=desc_tensor,
            scaler=self._desc2mof_scaler,
            feature_names=self._desc2mof_feature_names,
            feature_name_dir=DEFAULT_DESC2MOF_FEATURE_NAME,
            topk=topk,
            temperature=temperature,
            decode_mode=decode_mode,
            top_p=top_p,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            desc2mof_batch_size=(
                max(1, 64 // topk)
                if device == "cpu"
                else batch_size
            ),
            **desc2mof_kwargs,
        )

        weights = getattr(self, "_sk_feature_importances", None)

        if weights is None:
            print(
                "Warning: _sk_feature_importances not found. "
                "Guided Decoding (WMSE calculation) will be skipped."
            )

        desc2mof_scaler = getattr(self, "_desc2mof_scaler", None)

        valid_df, _, log_list, candidate_records = run_mof2desc_and_select(
            model=self.mof2desc,
            all_output=all_output,
            target_data=target_data,
            feature_size=self._desc2mof_config.get("feature_size", 183),
            weights=weights,
            sk_model=self.sk_model,
            sk_scaler=getattr(self, "_sk_scaler", None),
            desc2mof_scaler=desc2mof_scaler,
            topk=topk,
            wmse_target=wmse_target,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            return_descriptor=return_descriptor,
            feature_names=self._desc2mof_feature_names,
            single_ligand=single_ligand,
            ligand_linker=ligand_linker,
            organic_mode=organic_mode,
            return_candidates=True,
        )
        self._candidate_records = candidate_records

        print(f"Generated {len(valid_df)} valid MOFs out of {num_samples}")

        if output_type == "df":
            return valid_df
        elif output_type == "token":
            return valid_df["filename"].tolist()
        else:
            raise ValueError(f"Unknown output_type: {output_type}")

    def generate(
        self,
        num_samples: int = 100,
        target_value: Optional[float | int] = None,
        topk: int = 5,
        temperature: float = 1.0,
        wmse_target: float = 0.5,
        batch_size: int = 256,
        num_workers: int = 0,
        return_descriptor: bool = False,
        build_cif: bool = False,
        cif_dir: Optional[str | Path] = None,
        new_bb_dir: Optional[str | Path] = None,
        single_ligand: bool = False,
        ligand_linker: bool = False,
        organic_mode: str = "any",
        decode_device: Optional[str] = None,
        repair: bool = False,
        decode_constraint: bool = False,
        min_dist: Optional[float] = None,
        noise_scale: float = 1.0,
        decode_mode: Literal["beam", "sample"] = "beam",
        top_p: float = 0.9,
        exclude_names: Optional[set[str]] = None,
        exclude_keys: Optional[set[tuple]] = None,
        run_names: Optional[set[str]] = None,
        run_keys: Optional[set[tuple]] = None,
        repair_max_per_descriptor: int = 2,
        repair_min_dist: float = 0.5,
        repair_budget_min: Optional[float] = 20.0,
    ) -> pd.DataFrame:
        """Generate MOF structures, optionally building CIF files.

        With build_cif=True every accepted descriptor's valid beam candidates
        (WMSE < wmse_target, passing organic_mode) are tried best-WMSE first: each is
        built with PORMAKE and checked with ``linker_check.check_structure`` (unique linker
        graphs required: 1 for single/single-linker, 2 for ligand-linker, none for any).
        The first candidate that passes is written; a CIF that failed the check is never
        written.

        Args:
            num_samples: Number of MOFs to generate
            target_value: Target property value for conditional generation
            topk: Beam width for desc2mof beam search
            temperature: Sampling temperature for desc2mof
            wmse_target: WMSE threshold for filtering generated MOFs
            batch_size: Batch size for inference
            num_workers: Num workers for DataLoader
            return_descriptor: If True, include predicted descriptors in DataFrame
            build_cif: If True, run make_bb + build MOF CIF for each generated MOF
            cif_dir: Directory to save CIF files (required if build_cif=True)
            new_bb_dir: Directory for custom BB xyz files (default: builder/new_bbs)
            single_ligand: If True, only MOFs with one kind of organic ligand
            ligand_linker: If True, only MOFs with exactly one kind of organic ligand
                (3+ connections) and exactly one kind of organic linker (2 connections)
            organic_mode: "any" | "single" | "single-linker" | "ligand-linker"
                ("single-linker" = metal node + one 2-connected linker, ZIF-8/UiO-66 type)
            decode_device: Device for desc2mof/mof2desc (default: sampling device)
            repair: If True and chgnet is importable, relax atom positions of a candidate
                that failed with "overlap" once (FIRE, fmax=0.1 eV/A, 200 steps, fixed
                cell) and re-run the full check (linker reference + chemistry included)
            decode_constraint: If True, use single-linker constrained beam search
            min_dist: Minimum allowed interatomic distance in Angstrom for
                check_structure (default: linker_check.MIN_DIST_THRESHOLD = 1.0)
            noise_scale: Multiplier on the prop2desc reverse-diffusion noise (1.0 = default)
            decode_mode: "beam" (default) or "sample" (nucleus sampling of topk sequences)
            top_p: Nucleus mass for decode_mode="sample"
            exclude_names: Recipes / CIF names of earlier batches. A candidate whose
                recipe or resolved name is in it is skipped before make_bb / PORMAKE.
            exclude_keys: (topology, metal nodes, linker InChIKeys) keys of earlier
                batches (``linker_check.topology_node_linker_key``); checked after build.
            run_names: Like exclude_names, for MOFs already kept earlier in this run
                (counted as duplicate-within-run instead of duplicate-of-previous-batch).
            run_keys: Like exclude_keys, for MOFs already kept earlier in this run.
            repair_max_per_descriptor: Only the first N overlapping candidates of each
                descriptor are relaxed.
            repair_min_dist: Overlaps with min_dist below this (Angstrom) are not relaxed
                (atoms nearly on top of each other) and fail as "overlap_severe".
            repair_budget_min: Total CHGNet wall time (minutes) per call; later overlaps
                are not relaxed. None = no limit.

        Per-candidate outcomes are stored in ``self.last_build_stats`` (Counter):
        built_ok, dup_previous, dup_within_run, dup_prebuild (subset of the two
        duplicate counts caught before PORMAKE), overlap, overlap_severe (subset of
        overlap), other, other:<reason>, repair_attempted, repaired, repair_seconds,
        repair_skipped_limit, repair_skipped_budget.

        Returns:
            DataFrame with filename, wmse, pred_value (+ descriptors if return_descriptor=True).
            If build_cif=True, also includes cif, candidate_rank, repaired, build_status
            ("written" | "no_valid_structure"), fail_reason, min_dist,
            linker_smiles, linker_inchikeys and rank1_filename. filename / wmse /
            pred_value then describe the candidate that was actually used. A duplicate
            candidate (earlier batch or this run) is skipped and the next one is tried.
        """
        df = self._generate(
            num_samples=num_samples,
            target_value=target_value,
            output_type="df",
            topk=topk,
            temperature=temperature,
            wmse_target=wmse_target,
            batch_size=batch_size,
            num_workers=num_workers,
            return_descriptor=return_descriptor,
            single_ligand=single_ligand,
            ligand_linker=ligand_linker,
            organic_mode=organic_mode,
            decode_device=decode_device,
            repair=repair,
            decode_constraint=decode_constraint,
            noise_scale=noise_scale,
            decode_mode=decode_mode,
            top_p=top_p,
        )

        if not build_cif:
            return df

        # ── CIF building ───────────────────────────────────────────────────
        from ase.io import write as ase_write
        import pormake as pm

        from .builder import __builder_dir__
        from .builder.build_MOFs import name_to_mof
        from .builder.make_bbs import find_xtb_bin, make_bb
        from .linker_check import (
            MIN_DIST_THRESHOLD,
            check_structure,
            expected_unique_for_mode,
            topology_node_linker_key,
        )

        if cif_dir is None:
            raise ValueError("cif_dir must be provided when build_cif=True")
        cif_dir = Path(cif_dir)
        cif_dir.mkdir(parents=True, exist_ok=True)

        if new_bb_dir is None:
            new_bb_dir = Path(__builder_dir__) / "new_bbs"
        new_bb_dir = Path(new_bb_dir)
        new_bb_dir.mkdir(parents=True, exist_ok=True)

        db = pm.Database()
        relaxer = self._get_chgnet_relaxer(decode_device or self.accelerator) if repair else None
        written_this_run: set[str] = set()
        written_recipes: set[str] = set()
        written_keys: set[tuple] = set()
        exclude_names = exclude_names or set()
        exclude_keys = exclude_keys or set()
        run_names = run_names or set()
        run_keys = run_keys or set()
        stats: Counter = Counter()
        self.last_build_stats = stats
        # 1 linker graph for single / single-linker, 2 for ligand-linker, None (overlap only) for any
        expected_unique = expected_unique_for_mode(organic_mode, single_ligand, ligand_linker)
        if min_dist is None:
            min_dist = MIN_DIST_THRESHOLD

        def run_check(structure, resolved_name: str) -> dict:
            # the CIF name gives the expected linker; Gen_* blocks live in new_bb_dir
            return check_structure(
                structure,
                expected_unique=expected_unique,
                min_dist_threshold=min_dist,
                mof_name=resolved_name,
                new_bb_dir=new_bb_dir,
            )

        def run_repair_check(structure, resolved_name: str) -> dict:
            """run_check, plus linker_reference / chemistry also in organic mode "any"."""
            check = run_check(structure, resolved_name)
            if not check["passed"] or expected_unique is not None:
                return check
            # "any" checks overlap only, but a relaxed structure must keep its linkers
            full = check_structure(
                structure,
                expected_unique=1,
                min_dist_threshold=min_dist,
                mof_name=resolved_name,
                new_bb_dir=new_bb_dir,
            )
            if full["reason"].startswith("error"):
                problem = full["reason"]
            elif full["linker_reference"] not in ("matched", "no organic block"):
                problem = "linker_distorted"
            elif full["chemistry_issue"]:
                problem = "bad_chemistry"
            else:
                return check
            return {**check, "passed": False, "reason": problem,
                    "linker_reference": full["linker_reference"],
                    "chemistry_issue": full["chemistry_issue"]}

        def fail(record: dict, reason: str) -> None:
            record["fail_reason"] = reason
            if reason.startswith("overlap"):
                stats["overlap"] += 1
            else:
                stats["other"] += 1
                stats[f"other:{reason.split(':')[0]}"] += 1

        def duplicate(record: dict, previous: bool, prebuild: bool) -> None:
            record["fail_reason"] = "duplicate_previous_batch" if previous else "duplicate_within_run"
            stats["dup_previous" if previous else "dup_within_run"] += 1
            if prebuild:
                stats["dup_prebuild"] += 1

        # New building blocks (SELFIES tokens) are optimised with xTB. Without an xTB
        # binary, make_bb would try to download a Linux build (useless on macOS) and raise,
        # so candidates that need a new block are skipped instead of aborting the run.
        xtb_available = find_xtb_bin() is not None
        if not xtb_available:
            print(
                "WARNING: xTB not found (no 'xtb' on PATH, no builder/xtb-dist); beam "
                "candidates that need a new SELFIES building block are skipped. Install it "
                "e.g. with `conda install -c conda-forge xtb` to use them."
            )

        def resolve_name(filename: str) -> tuple[str, str]:
            """(PORMAKE name with Gen_* blocks, "") or ("", failure reason)."""
            tokens = filename.split("+")
            resolved_tokens = []
            for token in tokens:
                if token.startswith("[") and "[Lr]" in token:
                    if not xtb_available:
                        return "", "new_block_needs_xtb"
                    try:
                        success, bb_name = make_bb(token, run_dir=str(new_bb_dir))
                    except RuntimeError as e:  # xTB setup failed
                        print(f"  [BB ERROR] {e}")
                        return "", "bb_build_failed"
                    if not success:
                        return "", "bb_build_failed"
                    resolved_tokens.append(bb_name)
                else:
                    resolved_tokens.append(token)
            return "+".join(resolved_tokens), ""

        candidate_records = getattr(self, "_candidate_records", [])
        records = []
        for idx, row in df.iterrows():
            # df keeps the descriptor index of result_df, which indexes candidate_records
            if isinstance(idx, (int, np.integer)) and 0 <= idx < len(candidate_records):
                candidates = candidate_records[idx]
            else:
                candidates = [{
                    "filename": row["filename"],
                    "wmse": row.get("wmse"),
                    "pred_value": row.get("pred_value"),
                }]

            record = {
                "rank1_filename": row["filename"],
                "filename": row["filename"],
                "wmse": row.get("wmse"),
                "pred_value": row.get("pred_value"),
                "cif": None,
                "candidate_rank": None,
                "repaired": False,
                "build_status": "no_valid_structure",
                "fail_reason": "" if candidates else "no_candidates",
                "min_dist": float("nan"),
                "linker_smiles": "",
                "linker_inchikeys": "",
            }

            n_repairs = 0  # CHGNet relaxations spent on this descriptor
            for candidate_rank, candidate in enumerate(candidates, start=1):
                filename = candidate["filename"]
                # the recipe (SELFIES, before make_bb) is stable across runs; Gen_* names are not
                if filename in exclude_names:
                    duplicate(record, previous=True, prebuild=True)
                    continue
                if filename in run_names or filename in written_recipes:
                    duplicate(record, previous=False, prebuild=True)
                    continue
                resolved, bb_failure = resolve_name(filename)
                if not resolved:
                    fail(record, bb_failure)
                    continue
                cif_path = cif_dir / f"{resolved}.cif"
                # written earlier in this call first: those files also make cif_path.exists() true
                if cif_path.name in written_this_run:
                    duplicate(record, previous=False, prebuild=True)
                    continue
                # Gen_* numbers restart in every new_bbs/, so only Gen-free names are compared
                if ("Gen_" not in resolved and resolved in exclude_names) or cif_path.exists():
                    duplicate(record, previous=True, prebuild=True)
                    continue

                try:
                    mof = name_to_mof(resolved, db=db, new_bb_dir=str(new_bb_dir))
                except Exception as e:  # PORMAKE raises many error types
                    fail(record, f"build_error: {type(e).__name__}")
                    print(f"  [ERROR] {resolved}: {e}")
                    continue
                if isinstance(mof, str):
                    fail(record, "build_failed")
                    continue
                if np.min(mof.atoms.cell.cellpar()[:3]) < 4.5:
                    fail(record, "small_cell")
                    continue

                atoms = mof.atoms
                check = run_check(atoms, resolved)
                repaired = False
                if check["reason"] == "overlap" and relaxer is not None:
                    relaxed = None
                    if not check["min_dist"] >= repair_min_dist:
                        # atoms nearly on top of each other: a relaxation cannot separate them
                        stats["overlap_severe"] += 1
                        check = {**check, "reason": "overlap_severe"}
                    elif n_repairs >= repair_max_per_descriptor:
                        stats["repair_skipped_limit"] += 1
                    elif (
                        repair_budget_min is not None
                        and stats["repair_seconds"] >= 60 * repair_budget_min
                    ):
                        if not stats["repair_skipped_budget"]:
                            print(f"  [REPAIR] budget of {repair_budget_min} min used up; "
                                  f"no more CHGNet repairs in this round")
                        stats["repair_skipped_budget"] += 1
                    else:
                        n_repairs += 1
                        stats["repair_attempted"] += 1
                        t_start = time.perf_counter()
                        relaxed = self._relax_positions(relaxer, atoms)
                        stats["repair_seconds"] += time.perf_counter() - t_start
                    if relaxed is not None:
                        recheck = run_repair_check(relaxed, resolved)
                        if recheck["passed"]:
                            atoms, check, repaired = relaxed, recheck, True
                        else:
                            check = {**recheck, "reason": f"{recheck['reason']} (after repair)"}

                if not check["passed"]:
                    fail(record, check["reason"])
                    continue

                key = topology_node_linker_key(resolved, check["linker_inchikeys"])
                if key is not None and key in exclude_keys:
                    duplicate(record, previous=True, prebuild=False)
                    continue
                if key is not None and (key in run_keys or key in written_keys):
                    duplicate(record, previous=False, prebuild=False)
                    continue

                if repaired:
                    ase_write(str(cif_path), atoms, format="cif")
                    stats["repaired"] += 1
                else:
                    mof.write_cif(str(cif_path))
                written_this_run.add(cif_path.name)
                written_recipes.add(filename)
                if key is not None:
                    written_keys.add(key)
                stats["built_ok"] += 1

                record.update(
                    filename=filename,
                    wmse=candidate.get("wmse"),
                    pred_value=candidate.get("pred_value"),
                    cif=cif_path.name,
                    candidate_rank=candidate_rank,
                    repaired=repaired,
                    build_status="written",
                    fail_reason="",
                    min_dist=check["min_dist"],
                    linker_smiles=";".join(check["linker_smiles"]),
                    linker_inchikeys=";".join(check["linker_inchikeys"]),
                )
                print(
                    f"  [OK] {cif_path.name} (rank {candidate_rank}/{len(candidates)}, "
                    f"repaired={repaired})"
                )
                break
            else:
                print(
                    f"  [NONE] {row['filename']}: no candidate passed "
                    f"({len(candidates)} tried, last reason: {record['fail_reason']})"
                )
            records.append(record)

        df = df.copy()
        build_columns = [
            "rank1_filename", "filename", "wmse", "pred_value", "cif", "candidate_rank",
            "repaired", "build_status", "fail_reason", "min_dist", "linker_smiles",
            "linker_inchikeys",
        ]
        build_df = pd.DataFrame(records, index=df.index, columns=build_columns)
        for column in build_columns:
            df[column] = build_df[column]
        return df

    def _get_chgnet_relaxer(self, device: Optional[str] = None) -> Optional[object]:
        """CHGNet FIRE optimizer for repair, or None if chgnet is not importable.

        CHGNet runs on cuda when requested, otherwise on cpu (its float64 ops are not
        supported on mps).
        """
        if getattr(self, "_chgnet_relaxer", None) is not None:
            return self._chgnet_relaxer
        import importlib.util

        if importlib.util.find_spec("chgnet") is None:
            print("WARNING: repair requested but chgnet is not installed; repair disabled.")
            return None
        try:
            from chgnet.model.dynamics import StructOptimizer

            use_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
            self._chgnet_relaxer = StructOptimizer(optimizer_class="FIRE", use_device=use_device)
        except Exception as e:  # model download / init failures
            print(f"WARNING: could not initialise CHGNet ({e}); repair disabled.")
            return None
        return self._chgnet_relaxer

    @staticmethod
    def _relax_positions(relaxer: object, atoms) -> Optional[object]:
        """Relax atom positions once (FIRE, fmax 0.1 eV/A, 200 steps, fixed cell).

        Returns new ASE Atoms (dummy X atoms removed) or None.
        """
        from ase import Atoms
        from pymatgen.io.ase import AseAtomsAdaptor

        try:
            keep = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s != "X"]
            # plain Atoms: PORMAKE's extra per-atom arrays are not passed to CHGNet
            to_relax = Atoms(
                symbols=[atoms.get_chemical_symbols()[i] for i in keep],
                positions=atoms.get_positions()[keep],
                cell=atoms.get_cell(),
                pbc=True,
            )
            to_relax.wrap()
            result = relaxer.relax(
                to_relax,
                fmax=0.1,
                steps=200,
                relax_cell=False,
                verbose=False,
            )
            return AseAtomsAdaptor.get_atoms(result["final_structure"])
        except Exception as e:  # CHGNet can fail on unusual chemistry
            print(f"  [REPAIR FAILED] {type(e).__name__}: {e}")
            return None
