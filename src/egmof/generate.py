import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


_METAL_BLOCK_ORGANICS = None


def _load_metal_block_organics() -> dict:
    global _METAL_BLOCK_ORGANICS
    if _METAL_BLOCK_ORGANICS is None:
        from .desc2mof import __desc2mof_dir__

        path = Path(__desc2mof_dir__) / "data" / "metal_block_organics.json"
        if not path.exists():
            _METAL_BLOCK_ORGANICS = {}
        else:
            with path.open("r") as f:
                _METAL_BLOCK_ORGANICS = json.load(f)
    return _METAL_BLOCK_ORGANICS


def cal_wmse(
    pred_desc: torch.Tensor, target_desc: torch.Tensor, weights
) -> torch.Tensor:
    """Calculate weighted MSE between predicted and target descriptors."""
    mse = F.mse_loss(pred_desc, target_desc, reduction="none")
    w = torch.tensor(weights, dtype=mse.dtype, device=mse.device)
    weighted_mse = mse * w
    wmse = weighted_mse.sum(dim=-1) / w.sum()
    return wmse


def sk_predict(desc_pred: np.ndarray, sk_model, sk_scaler) -> np.ndarray:
    """Predict property from descriptor using sklearn model."""
    desc_scaled = sk_scaler.encode(desc_pred)
    pred_scaled = sk_model.predict(desc_scaled)
    return sk_scaler.decode_target(pred_scaled)


def _build_mofgen_dataset(desc_tensor: torch.Tensor, scaler, feature_name_dir: str):
    """Build MOFGenDataset from descriptor tensor [N, D]."""
    from .desc2mof import MOFGenDataset

    desc_np = desc_tensor.detach().cpu().numpy()
    df = pd.DataFrame(desc_np, columns=None)
    return MOFGenDataset(
        df, scaled=True, scaler=scaler, feature_name_dir=feature_name_dir
    )


def _parse_mof_output(all_output: list, SEP_TOKEN_ID: int):
    """Parse MOF token output, returns (valid_mask, log_list)."""
    from .desc2mof.utils import is_valid as _is_valid

    return _is_valid(all_output, SEP_TOKEN_ID)


def organic_blocks(seq, SEP_TOKEN_ID: int):
    """Split one Desc2MOF token sequence into building blocks and classify them.

    Token layout:  topo, [CN_n] block [SEP], [CN_n] block [SEP], ...
      - metal blocks (metal_node.txt / metal_edge.txt) are ONE token
      - organic blocks are SELFIES = several tokens ([Lr] = connection point)
      - the [CN_n] token before a block gives its number of connection points:
          n >= 3 -> organic LIGAND  (multi-arm, e.g. tetracarboxylate)
          n == 2 -> organic LINKER  (2-connected, e.g. dipyridyl pillar)

    Returns
        (n_distinct_ligands, n_distinct_linkers, n_metal_blocks,
         embedded_organic_identities, metal_block_organic,
         selfies_identities, metal_linker_stub_types).
    """
    from .desc2mof.dataset import EOS_TOKEN, PAD_TOKEN, CN_IDS, MOF_DECODE_DICT

    body = [t for t in seq if t not in (EOS_TOKEN, PAD_TOKEN)][1:]  # drop topology token
    segments, cur, cur_cn = [], [], None
    for t in body:
        if t == SEP_TOKEN_ID:
            segments.append((cur_cn, cur))
            cur, cur_cn = [], None
        elif t in CN_IDS:
            digits = re.findall(r"\d+", str(MOF_DECODE_DICT.get(t, "")))
            cur_cn = int(digits[0]) if digits else None
        else:
            cur.append(t)
    if cur:
        segments.append((cur_cn, cur))

    ligands = {tuple(b) for cn, b in segments if len(b) > 1 and (cn is None or cn >= 3)}
    linkers = {tuple(b) for cn, b in segments if len(b) > 1 and cn == 2}
    selfies_identities = ligands | linkers

    metals = [b for cn, b in segments if len(b) == 1]
    metal_block_organics = _load_metal_block_organics()
    embedded_identities = set()
    metal_block_organic = False
    metal_linker_stub_types = set()
    for block in metals:
        block_name = MOF_DECODE_DICT.get(block[0])
        entry = metal_block_organics.get(block_name, {})
        metal_linker_stub_types.update(entry.get("linker_stubs", []))
        if entry.get("has_organic", False):
            metal_block_organic = True
            identities = (
                entry.get("graph_keys")
                or entry.get("inchikeys")
                or entry.get("smiles")
                or []
            )
            embedded_identities.update(identities)

    ligands.update(("embedded", identity) for identity in embedded_identities)
    return (
        len(ligands),
        len(linkers),
        len(metals),
        embedded_identities,
        metal_block_organic,
        selfies_identities,
        len(metal_linker_stub_types),
    )


def count_ligands(seq, SEP_TOKEN_ID: int) -> int:
    """Number of DISTINCT organic ligands in one Desc2MOF token sequence.

    Token layout:  topo, [CN] bb [SEP], [CN] bb [SEP], ...
    Metal building blocks (metal_node.txt / metal_edge.txt) are ONE token.
    Organic building blocks are always SELFIES = several tokens ([Lr] = connection point).
    So every segment with more than one token is an organic ligand.
    """
    from .desc2mof.dataset import EOS_TOKEN, PAD_TOKEN, CN_IDS

    body = [t for t in seq if t not in (EOS_TOKEN, PAD_TOKEN)][1:]  # drop topology token
    segments, cur = [], []
    for t in body:
        if t == SEP_TOKEN_ID:
            segments.append(cur)
            cur = []
        elif t not in CN_IDS:
            cur.append(t)
    if cur:
        segments.append(cur)
    organic = {tuple(seg) for seg in segments if len(seg) > 1}
    return len(organic)


def run_desc2mof(
    model,
    target_desc: torch.Tensor,
    scaler,
    feature_names: list[str],
    feature_name_dir: str,
    topk: int = 5,
    temperature: float = 1.0,
    batch_size: int = 256,
    num_workers: int = 0,
    device: str = "cuda",
    desc2mof_batch_size: int | None = None,
    single_linker_constraint: bool = False,
    decode_mode: str = "beam",
    top_p: float = 0.9,
):
    """Run desc2mof inference: descriptor → MOF tokens (beam search or nucleus sampling).

    decode_mode="sample" draws topk sequences per descriptor by top-p sampling
    (see Decoder.generate_beam) instead of keeping the topk beams.
    """
    from .desc2mof import MOFGenDataset, decode_token2mof

    if desc2mof_batch_size is None:
        desc2mof_batch_size = max(1, 64 // topk) if device == "mps" else batch_size

    model.eval()
    model = model.to(device)
    if device == "mps":
        model = model.float()
        target_desc = target_desc.float()

    desc_np = target_desc.detach().cpu().numpy()
    target_df = pd.DataFrame(desc_np, columns=feature_names)
    target_data = MOFGenDataset(
        target_df, scaled=True, scaler=scaler, feature_name_dir=feature_name_dir
    )
    target_loader = DataLoader(
        target_data, batch_size=desc2mof_batch_size, num_workers=num_workers, shuffle=False
    )

    preds_output = []
    for batch in tqdm(target_loader, desc="desc2mof inference"):
        with torch.no_grad():
            desc_batch, _ = batch
            desc_batch = desc_batch.to(device)
            if device == "mps":
                desc_batch = desc_batch.float()
            preds, _, _, _ = model.generate_beam(
                desc_batch,
                temperature=temperature,
                beam_width=topk,
                single_linker_constraint=single_linker_constraint,
                decode_mode=decode_mode,
                top_p=top_p,
            )
            preds_output.append(preds.detach().cpu())
        if device == "mps" and hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    preds_output = torch.concat(preds_output)
    all_output = (
        preds_output.reshape(-1, preds_output.shape[-1]).detach().cpu().tolist()
    )
    mof_names = decode_token2mof(all_output)
    return all_output, mof_names, target_data


def run_mof2desc_and_select(
    model,
    all_output: list,
    target_data,
    feature_size: int,
    weights=None,
    sk_model=None,
    sk_scaler=None,
    desc2mof_scaler=None,
    topk: int = 5,
    wmse_target: float = 0.5,
    batch_size: int = 256,
    num_workers: int = 0,
    device: str = "cuda",
    return_descriptor: bool = False,
    feature_names: list[str] | None = None,
    single_ligand: bool = False,
    ligand_linker: bool = False,
    organic_mode: str = "any",
    return_candidates: bool = False,
):
    """Run mof2desc, compute wmse, and select best MOFs.

    single_ligand=True: a beam candidate is valid only if it contains exactly ONE kind
    of organic ligand. Among the valid candidates of each descriptor the lowest-WMSE
    one is kept, i.e. the ligand that best matches the target descriptor wins.

    ligand_linker=True: exactly one multi-arm organic ligand type (3+ connections) and
    exactly one 2-connected organic linker type.

    organic_mode (overrides the two flags above when not "any"):
      "any"           - no restriction (original EGMOF)
      "single"        - exactly one kind of organic block (linker OR multi-arm ligand)
      "single-linker" - metal node(s) + exactly one kind of 2-connected organic linker,
                        no multi-arm organic ligand (ZIF-8 / UiO-66-type MOFs)
      "ligand-linker" - one multi-arm ligand type + one 2-connected linker type
    O2-/OH- groups belong to the metal node block in PORMAKE and are never counted.
    """
    from .desc2mof import SEP_TOKEN
    from .mof2desc import Desc2MOFOutputDataset
    from .desc2mof import decode_token2mof

    model = model.to(device)
    model.eval()
    if device == "mps":
        model = model.float()

    valid_mask, log_list = _parse_mof_output(all_output, SEP_TOKEN_ID=SEP_TOKEN)

    # ── organic-composition rules ───────────────────────────────────────
    n_lig = np.array([count_ligands(seq, SEP_TOKEN) for seq in all_output])
    blk = [organic_blocks(seq, SEP_TOKEN) for seq in all_output]
    n_ligand_types = np.array([x[0] for x in blk])
    n_linker_types = np.array([x[1] for x in blk])
    n_metal = np.array([x[2] for x in blk])
    embedded_identities = [x[3] for x in blk]
    metal_block_organic = np.array([x[4] for x in blk], dtype=bool)
    selfies_identities = [x[5] for x in blk]
    union_metal_linker_stub_types = np.array([x[6] for x in blk])

    mode = organic_mode
    if mode == "any":
        if ligand_linker:
            mode = "ligand-linker"
        elif single_ligand:
            mode = "single"

    if mode == "single":
        total_organic_types = np.array([
            len(embedded | set(selfies))
            for embedded, selfies in zip(embedded_identities, selfies_identities)
        ])
        ok = (total_organic_types == 1) & (n_metal >= 1)
    elif mode == "single-linker":
        ok = (
            (n_ligand_types == 0)
            & (n_linker_types == 1)
            & ~metal_block_organic
            & (union_metal_linker_stub_types <= 1)
            & (n_metal >= 1)
        )
    elif mode == "ligand-linker":
        ok = (
            (n_ligand_types == 1)
            & (n_linker_types == 1)
            & ~metal_block_organic
            & (n_metal >= 1)
        )
    elif mode == "any":
        ok = np.ones(len(all_output), dtype=bool)
    else:
        raise ValueError(f"Unknown organic_mode: {organic_mode}")

    if mode != "any":
        if mode == "single-linker":
            stub_type_ok = union_metal_linker_stub_types <= 1
            removed_by_mixed_stub_rule = int(
                (valid_mask & ~stub_type_ok & (n_ligand_types == 0) & (n_linker_types == 1)
                 & ~metal_block_organic & (n_metal >= 1)).sum()
            )
            print(
                f"single-linker mixed-stub filter removed "
                f"{removed_by_mixed_stub_rule} beam candidates"
            )
        print(
            f"organic filter [{mode}]: {(valid_mask & ok).sum()} of {valid_mask.sum()} "
            f"valid beam candidates pass"
        )
        valid_mask = valid_mask & ok

    any_valid_mask = valid_mask.reshape(-1, topk).any(axis=1)

    x_np = np.array(target_data.x.squeeze(-1))
    x_tensor = (
        torch.from_numpy(x_np).unsqueeze(1).repeat(1, topk, 1).view(-1, feature_size)
    )
    if device == "mps":
        x_tensor = x_tensor.float()
    mof2desc_dataset = Desc2MOFOutputDataset(all_output, x_tensor, scaled=False)
    mof2desc_loader = DataLoader(
        mof2desc_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    wmse_output, mof2desc_preds = [], []
    for batch in tqdm(mof2desc_loader, desc="mof2desc inference"):
        with torch.no_grad():
            target_desc, token_ids, attention_mask = batch
            token_ids = token_ids.to(device)
            attention_mask = attention_mask.to(device)
            target_desc = target_desc.to(device)
            if device == "mps":
                target_desc = target_desc.float()
            pred_desc = model(token_ids, attention_mask)
            if weights is not None:
                wmse = cal_wmse(pred_desc, target_desc, weights)
                wmse_output.append(wmse.detach().cpu())
            mof2desc_preds.append(pred_desc.detach().cpu())
        if device == "mps" and hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    mof2desc_preds = torch.concat(mof2desc_preds).detach().cpu().numpy()
    if desc2mof_scaler is not None:
        mof2desc_preds_decoded = desc2mof_scaler.decode(mof2desc_preds)
    else:
        mof2desc_preds_decoded = mof2desc_preds

    if weights is not None:
        wmse_output = torch.concat(wmse_output).detach().cpu().numpy()
        wmse_reshaped = wmse_output.reshape(-1, topk)
    else:
        wmse_reshaped = None

    mof_names_reshaped = np.array(decode_token2mof(all_output)).reshape(-1, topk)
    valid_mask_reshaped = valid_mask.reshape(-1, topk)

    if weights is not None:
        wmse_masked = np.where(valid_mask_reshaped, wmse_reshaped, np.inf)
        min_indices = wmse_masked.argmin(axis=1)
        row_idx = np.arange(wmse_reshaped.shape[0])
        best_wmse = wmse_reshaped[row_idx, min_indices]
    else:
        num_samples = len(mof_names_reshaped)
        min_indices = np.zeros(num_samples, dtype=int)
        row_idx = np.arange(num_samples)
        best_wmse = None

    best_mof_names = mof_names_reshaped[row_idx, min_indices]

    result_df = pd.DataFrame({"filename": best_mof_names})
    result_df["n_ligands"] = n_lig.reshape(-1, topk)[row_idx, min_indices]
    result_df["n_ligand_types"] = n_ligand_types.reshape(-1, topk)[row_idx, min_indices]
    result_df["n_linker_types"] = n_linker_types.reshape(-1, topk)[row_idx, min_indices]
    result_df["metal_block_organic"] = metal_block_organic.reshape(-1, topk)[row_idx, min_indices]
    if best_wmse is not None:
        result_df["wmse"] = best_wmse

    prop_preds_reshaped = None
    if sk_model is not None and sk_scaler is not None:
        prop_preds = sk_predict(mof2desc_preds_decoded, sk_model, sk_scaler)
        prop_preds_reshaped = prop_preds.reshape(-1, topk)
        best_prop_preds = prop_preds_reshaped[row_idx, min_indices]
        result_df["pred_value"] = best_prop_preds

    if best_wmse is not None:
        wmse_mask = result_df["wmse"] < wmse_target
        final_mask = wmse_mask & any_valid_mask
    else:
        final_mask = any_valid_mask

    if return_descriptor:
        mof2desc_preds_reshaped = mof2desc_preds_decoded.reshape(-1, topk, feature_size)
        best_descriptors = mof2desc_preds_reshaped[row_idx, min_indices]
        desc_cols = feature_names if feature_names is not None else list(range(feature_size))
        desc_df = pd.DataFrame(best_descriptors, columns=desc_cols, index=result_df.index)
        result_df = pd.concat([result_df, desc_df], axis=1)

    # candidate_records[i] = ALL valid beam candidates of descriptor i (i == result_df
    # index), ranked by WMSE (best first); with weights only those with WMSE < wmse_target.
    candidate_records = []
    for row_idx_candidate in range(mof_names_reshaped.shape[0]):
        if weights is not None:
            ranked = np.argsort(wmse_masked[row_idx_candidate], kind="stable")
        else:
            ranked = np.arange(topk)
        records = []
        for beam_idx in ranked:
            if not valid_mask_reshaped[row_idx_candidate, beam_idx]:
                continue
            wmse_value = None
            if weights is not None:
                wmse_value = float(wmse_reshaped[row_idx_candidate, beam_idx])
                if not (np.isfinite(wmse_value) and wmse_value < wmse_target):
                    continue
            records.append({
                "filename": mof_names_reshaped[row_idx_candidate, beam_idx],
                "wmse": wmse_value,
                "pred_value": (
                    float(prop_preds_reshaped[row_idx_candidate, beam_idx])
                    if prop_preds_reshaped is not None
                    else None
                ),
            })
        candidate_records.append(records)

    if return_candidates:
        return result_df[final_mask], result_df[~final_mask], log_list, candidate_records
    return result_df[final_mask], result_df[~final_mask], log_list