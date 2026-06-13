#!/usr/bin/env python3
"""
Collect NTP/RHM transformer runs into one numpy dictionary.

Expected raw layout from run_ntp.pbs:

    data/<RUN_NAME>/<dataset>_<model>_P_<P>_seed_.../results.pt

Legacy ``results.pkl`` files from the previous RHM repository are also accepted.

Launch example:

    python -u collect_file.py --run_name ntp_rhm_transformer_default

Output:

    results/<RUN_NAME>.npy

The saved .npy contains seed-resolved arrays and mean/std/count arrays.
Scalar curves have shape [nP, nT, nSeeds] for *_seeds.
Per-token next-token curves have shape [nP, nT, nSeeds, d-1].
M_l curves have shape [nP, nT, nSeeds, L] for *_seeds.
Position-resolved M_l curves have shape [nP, nT, nSeeds, d-1, L].
"""

from __future__ import annotations

import argparse
import io
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch


SEED_KEYS = ("seed_rules", "seed_sample", "seed_model")

# These may legitimately differ across files belonging to the same run family.
IGNORE_COMPARE_KEYS = {
    "outname",
    "output_dir",
    "latest_model_path",
    "best_model_path",
    "rules_path",
    "device",
    "repo_dir",
    "data_root",
    "run_name",
    "train_size",
    "git",
    *SEED_KEYS,
}

# Raw dynamics keys written by the NTP code.
# We keep short aliases in the aggregated output, matching the older project logic.
SCALAR_KEY_MAP = {
    "trainloss": "trainloss",
    "trainacc": "trainacc",
    "trainerr": "trainerr",
    "testloss": "testloss",
    "testacc": "testacc",
    "testerr": "err",
    "specnorm": "spectral",
    "log_specnorm": "log_spectral",
    "specnorm_no_qk": "spectral_no_qk",
    "log_specnorm_no_qk": "log_spectral_no_qk",
    "l2norm": "l2",
    "log_l2norm": "log_l2",
}

# Level-wise vectors saved by the NTP code when COMPUTE_M_l=true.
ML_VECTOR_KEYS = (
    "rhm_M_mean",
    "rhm_M_pos_frac",
    "rhm_peeled_loss",
    "rhm_valid_frac",
)

# Position-resolved arrays saved by the NTP code when COMPUTE_M_l=true.
ML_POSITION_KEYS = (
    "rhm_M_mean_by_position",
    "rhm_M_pos_frac_by_position",
    "rhm_peeled_loss_by_position",
    "rhm_valid_count_by_position",
)

# Position-resolved next-token metrics saved at every validation.
# Shape in one raw dynamics entry: [d-1], where index k predicts token i=k+2.
TOKEN_POSITION_KEYS = (
    "trainloss_by_position",
    "trainacc_by_position",
    "trainerr_by_position",
    "train_count_by_position",
    "testloss_by_position",
    "testacc_by_position",
    "testerr_by_position",
    "test_count_by_position",
)

# Position-resolved effective-dimension diagnostics of centered prediction logits.
# Shape in one raw dynamics entry: [d-1].
LOGIT_EFFDIM_POSITION_KEYS = (
    "train_logit_energy_mean_by_position",
    "train_logit_input_variance_by_position",
    "train_logit_effdim_entropy_by_position",
    "train_logit_effdim_pr_by_position",
    "train_logit_effdim_entropy_norm_by_position",
    "train_logit_effdim_pr_norm_by_position",
    "train_logit_effdim_count_by_position",
    "test_logit_energy_mean_by_position",
    "test_logit_input_variance_by_position",
    "test_logit_effdim_entropy_by_position",
    "test_logit_effdim_pr_by_position",
    "test_logit_effdim_entropy_norm_by_position",
    "test_logit_effdim_pr_norm_by_position",
    "test_logit_effdim_count_by_position",
)

# Per-position covariance eigenvalues. Shape in one raw dynamics entry: [d-1, vocab_size].
LOGIT_EFFDIM_EIGVAL_KEYS = (
    "train_logit_cov_eigvals_by_position",
    "test_logit_cov_eigvals_by_position",
)

SPLITS = ("train", "test")


def _torch_load_cpu_from_bytes(b: bytes):
    return torch.load(io.BytesIO(b), map_location="cpu")


def _args_to_dict(args: Any, path: Path) -> Dict[str, Any]:
    if hasattr(args, "__dict__"):
        return vars(args).copy()
    if isinstance(args, dict):
        return dict(args)
    raise TypeError(f"Unsupported config object in {path}: {type(args)}")


def load_one_file(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load current ``results.pt`` or legacy two-object ``results.pkl`` files."""
    if path.suffix == ".pt":
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch versions predating weights_only
            payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise TypeError(f"Expected a dictionary in {path}, got {type(payload)}")
        config = payload.get("config", payload.get("args", {}))
        return _args_to_dict(config, path), payload

    old_load_from_bytes = torch.storage._load_from_bytes
    torch.storage._load_from_bytes = _torch_load_cpu_from_bytes
    try:
        with open(path, "rb") as f:
            args = pickle.load(f)
            output = pickle.load(f)
    finally:
        torch.storage._load_from_bytes = old_load_from_bytes
    return _args_to_dict(args, path), output


def comparable_params(args_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in args_dict.items() if k not in IGNORE_COMPARE_KEYS}


def _as_float_1d(value: Any, length: int) -> np.ndarray:
    out = np.full(length, np.nan, dtype=float)
    if value is None or length == 0:
        return out
    arr = np.asarray(value, dtype=float).reshape(-1)
    n = min(length, arr.size)
    if n > 0:
        out[:n] = arr[:n]
    return out


def _as_float_2d(value: Any, shape: Tuple[int, int]) -> np.ndarray:
    out = np.full(shape, np.nan, dtype=float)
    if value is None or shape[0] == 0 or shape[1] == 0:
        return out
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1 and shape[0] == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        arr = arr.reshape(shape) if arr.size == shape[0] * shape[1] else np.asarray([], dtype=float)
    if arr.size > 0:
        n0 = min(shape[0], arr.shape[0])
        n1 = min(shape[1], arr.shape[1])
        out[:n0, :n1] = arr[:n0, :n1]
    return out


def _get_time_from_entry(d: Dict[str, Any]) -> int:
    return int(d.get("t", d.get("epoch", d.get("global_update", -1))))


def dynamics_to_arrays(output: Dict[str, Any]) -> Dict[str, Any]:
    dyn = output.get("dynamics", []) or []
    nT = len(dyn)

    times = np.array([_get_time_from_entry(d) for d in dyn], dtype=int)
    out: Dict[str, Any] = {"epochs": times}

    for raw_key, out_key in SCALAR_KEY_MAP.items():
        if raw_key == "testerr":
            values = []
            for d in dyn:
                if "testerr" in d:
                    values.append(d["testerr"])
                elif "err" in d:
                    values.append(d["err"])
                elif "testacc" in d:
                    values.append(1.0 - d["testacc"])
                else:
                    values.append(np.nan)
            out[out_key] = np.asarray(values, dtype=float)
        elif raw_key == "trainerr":
            values = []
            for d in dyn:
                if "trainerr" in d:
                    values.append(d["trainerr"])
                elif "trainacc" in d:
                    values.append(1.0 - d["trainacc"])
                else:
                    values.append(np.nan)
            out[out_key] = np.asarray(values, dtype=float)
        else:
            out[out_key] = np.asarray([d.get(raw_key, np.nan) for d in dyn], dtype=float)

    # Also store evaluation sample counts when present.
    out["train_num_samples_eval"] = np.asarray([d.get("train_num_samples_eval", np.nan) for d in dyn], dtype=float)
    out["test_num_samples_eval"] = np.asarray([d.get("test_num_samples_eval", np.nan) for d in dyn], dtype=float)

    # Infer per-token, logit-effective-dimension, and M_l dimensions in this single file.
    L = 0
    Dpos = 0
    TokenD = 0
    LogitEigD = 0
    for d in dyn:
        for key in TOKEN_POSITION_KEYS + LOGIT_EFFDIM_POSITION_KEYS:
            value = d.get(key)
            if value is not None:
                arr = np.asarray(value)
                if arr.size > 0:
                    TokenD = max(TokenD, int(arr.reshape(-1).size))

        for key in LOGIT_EFFDIM_EIGVAL_KEYS:
            value = d.get(key)
            if value is not None:
                arr = np.asarray(value)
                if arr.ndim >= 2 and arr.size > 0:
                    TokenD = max(TokenD, int(arr.shape[-2]))
                    LogitEigD = max(LogitEigD, int(arr.shape[-1]))

        for split in SPLITS:
            for key in ML_VECTOR_KEYS:
                value = d.get(f"{split}_{key}")
                if value is not None:
                    arr = np.asarray(value)
                    if arr.size > 0:
                        L = max(L, int(arr.reshape(-1).size))
            for key in ML_POSITION_KEYS:
                value = d.get(f"{split}_{key}")
                if value is not None:
                    arr = np.asarray(value)
                    if arr.ndim >= 2:
                        Dpos = max(Dpos, int(arr.shape[-2]))
                        L = max(L, int(arr.shape[-1]))

    out["ntp_num_positions"] = np.array(TokenD, dtype=int)
    for key in TOKEN_POSITION_KEYS:
        out[key] = (
            np.stack([_as_float_1d(d.get(key), TokenD) for d in dyn], axis=0)
            if TokenD > 0 else np.full((nT, 0), np.nan)
        )
    for key in LOGIT_EFFDIM_POSITION_KEYS:
        out[key] = (
            np.stack([_as_float_1d(d.get(key), TokenD) for d in dyn], axis=0)
            if TokenD > 0 else np.full((nT, 0), np.nan)
        )

    out["logit_effdim_num_eigvals"] = np.array(LogitEigD, dtype=int)
    for key in LOGIT_EFFDIM_EIGVAL_KEYS:
        out[key] = (
            np.stack([_as_float_2d(d.get(key), (TokenD, LogitEigD)) for d in dyn], axis=0)
            if (TokenD > 0 and LogitEigD > 0) else np.full((nT, 0, 0), np.nan)
        )

    out["rhm_M_l_num_levels"] = np.array(L, dtype=int)
    out["rhm_M_l_num_positions"] = np.array(Dpos, dtype=int)

    for split in SPLITS:
        for key in ML_VECTOR_KEYS:
            full_key = f"{split}_{key}"
            arr = np.stack([_as_float_1d(d.get(full_key), L) for d in dyn], axis=0) if L > 0 else np.full((nT, 0), np.nan)
            out[full_key] = arr

        for key in ML_POSITION_KEYS:
            full_key = f"{split}_{key}"
            arr = np.stack([_as_float_2d(d.get(full_key), (Dpos, L)) for d in dyn], axis=0) if (Dpos > 0 and L > 0) else np.full((nT, 0, 0), np.nan)
            out[full_key] = arr

        n_key = f"{split}_rhm_num_samples"
        out[n_key] = np.asarray([d.get(n_key, np.nan) for d in dyn], dtype=float)

    return out


def nanmean_std_count(raw: np.ndarray, axis: int):
    valid = ~np.isnan(raw)
    counts = valid.sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(raw, axis=axis)
        std = np.nanstd(raw, axis=axis, ddof=1)
    mean = np.where(counts == 0, np.nan, mean)
    std = np.where(counts < 2, -1.0, std)
    return mean, std, counts


def _transpose_seed_axis(raw: np.ndarray) -> np.ndarray:
    # raw is [nP, nSeeds, nT, ...] -> [nP, nT, nSeeds, ...]
    if raw.ndim < 3:
        raise ValueError(f"Expected ndim >= 3, got shape {raw.shape}")
    axes = [0, 2, 1] + list(range(3, raw.ndim))
    return np.transpose(raw, axes)


def _add_metric(result: Dict[str, Any], name: str, raw: np.ndarray):
    seeds = _transpose_seed_axis(raw)
    mean, std, n = nanmean_std_count(raw, axis=1)
    result[f"{name}_raw"] = raw
    result[f"{name}_seeds"] = seeds
    result[f"{name}_mean"] = mean
    result[f"{name}_std"] = std
    result[f"{name}_n"] = n


def find_result_files(run_dir: Path) -> List[Path]:
    # Current small-language-modeling format plus the legacy RHM format.
    files = list(run_dir.glob("**/results.pt"))
    files.extend(run_dir.glob("**/results.pkl"))
    # Avoid duplicates if a path was reached through more than one pattern.
    return sorted(set(path.resolve() for path in files))


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect small-language-modeling text/RHM results into one .npy file")
    parser.add_argument("--run_name", type=str, required=True, help="folder name inside data_root")
    parser.add_argument("--data_root", type=str, default="data", help="root folder containing run folders")
    parser.add_argument("--results_dir", type=str, default="collected_results", help="folder where aggregated .npy is saved")
    parser.add_argument("--experiment_name", type=str, default=None, help="output file basename; default is run_name")
    parser.add_argument("--allow_mixed_params", action="store_true", help="do not stop if non-seed/non-P hyperparameters differ")
    args = parser.parse_args()

    run_dir = Path(args.data_root).expanduser().resolve() / args.run_name
    results_dir = Path(args.results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    files = find_result_files(run_dir)
    if len(files) == 0:
        raise FileNotFoundError(f"No results.pt or results.pkl files found under {run_dir}")

    entries: List[Dict[str, Any]] = []
    for path in files:
        args_dict, output = load_one_file(path)
        dyn = dynamics_to_arrays(output)
        best = output.get("best", {}) or {}

        entry: Dict[str, Any] = {
            "path": str(path),
            "args": args_dict,
            "params_compare": comparable_params(args_dict),
            "train_size": int(args_dict.get("train_size", -1)),
            "seed_rules": int(args_dict.get("seed_rules", -1)),
            "seed_sample": int(args_dict.get("seed_sample", -1)),
            "seed_model": int(args_dict.get("seed_model", -1)),
            "epochs": dyn["epochs"],
            "best_loss": float(best.get("loss", np.nan)),
            "best_acc": float(best.get("acc", np.nan)),
            "best_err": float(best.get("err", np.nan)),
            "best_epoch": float(best.get("epoch", np.nan)),
            "last_saved_epoch": float(output.get("epoch", np.nan)),
        }
        entry.update(dyn)
        entries.append(entry)

    if not args.allow_mixed_params:
        ref = entries[0]["params_compare"]
        for e in entries[1:]:
            if e["params_compare"] != ref:
                raise ValueError(
                    "Files under this run folder do not belong to one experiment family.\n"
                    "Use --allow_mixed_params to collect anyway, or split them into different RUN_NAME folders.\n\n"
                    f"Reference file: {entries[0]['path']}\n"
                    f"Reference params: {ref}\n\n"
                    f"Different file: {e['path']}\n"
                    f"Different params: {e['params_compare']}"
                )
        fixed_params = ref
    else:
        fixed_params = {"mixed_params": True}

    P_values = np.array(sorted({e["train_size"] for e in entries}), dtype=int)
    epoch_values = np.array(sorted({int(t) for e in entries for t in e["epochs"]}), dtype=int)
    epoch_to_idx = {int(t): i for i, t in enumerate(epoch_values)}

    entries_by_P: Dict[int, List[Dict[str, Any]]] = {int(P): [] for P in P_values}
    for e in entries:
        entries_by_P[e["train_size"]].append(e)
    for P in P_values:
        entries_by_P[int(P)] = sorted(
            entries_by_P[int(P)],
            key=lambda e: (e["seed_rules"], e["seed_sample"], e["seed_model"], e["path"]),
        )

    nP = len(P_values)
    max_seeds = max(len(v) for v in entries_by_P.values())
    nT = len(epoch_values)
    max_L = max(int(e.get("rhm_M_l_num_levels", 0)) for e in entries)
    max_Dpos = max(int(e.get("rhm_M_l_num_positions", 0)) for e in entries)
    max_token_pos = max(int(e.get("ntp_num_positions", 0)) for e in entries)
    max_logit_eigs = max(int(e.get("logit_effdim_num_eigvals", 0)) for e in entries)

    scalar_names = sorted(set(SCALAR_KEY_MAP.values()) | {"train_num_samples_eval", "test_num_samples_eval"})
    scalar_raw = {name: np.full((nP, max_seeds, nT), np.nan, dtype=float) for name in scalar_names}
    token_position_raw = {
        name: np.full((nP, max_seeds, nT, max_token_pos), np.nan, dtype=float)
        for name in TOKEN_POSITION_KEYS
    }
    logit_effdim_position_raw = {
        name: np.full((nP, max_seeds, nT, max_token_pos), np.nan, dtype=float)
        for name in LOGIT_EFFDIM_POSITION_KEYS
    }
    logit_effdim_eigval_raw = {
        name: np.full((nP, max_seeds, nT, max_token_pos, max_logit_eigs), np.nan, dtype=float)
        for name in LOGIT_EFFDIM_EIGVAL_KEYS
    }

    ml_vector_raw: Dict[str, np.ndarray] = {}
    ml_position_raw: Dict[str, np.ndarray] = {}
    for split in SPLITS:
        for key in ML_VECTOR_KEYS:
            name = f"{split}_{key}"
            ml_vector_raw[name] = np.full((nP, max_seeds, nT, max_L), np.nan, dtype=float)
        for key in ML_POSITION_KEYS:
            name = f"{split}_{key}"
            ml_position_raw[name] = np.full((nP, max_seeds, nT, max_Dpos, max_L), np.nan, dtype=float)
        scalar_raw[f"{split}_rhm_num_samples"] = np.full((nP, max_seeds, nT), np.nan, dtype=float)

    best_loss_raw = np.full((nP, max_seeds), np.nan, dtype=float)
    best_acc_raw = np.full((nP, max_seeds), np.nan, dtype=float)
    best_err_raw = np.full((nP, max_seeds), np.nan, dtype=float)
    best_epoch_raw = np.full((nP, max_seeds), np.nan, dtype=float)
    last_saved_epoch_raw = np.full((nP, max_seeds), np.nan, dtype=float)
    seed_triplets = np.full((nP, max_seeds, 3), -1, dtype=int)
    file_index = np.full((nP, max_seeds), "", dtype=object)
    num_seeds = np.zeros(nP, dtype=int)

    for iP, P in enumerate(P_values):
        plist = entries_by_P[int(P)]
        num_seeds[iP] = len(plist)
        for iseed, e in enumerate(plist):
            seed_triplets[iP, iseed] = [e["seed_rules"], e["seed_sample"], e["seed_model"]]
            file_index[iP, iseed] = e["path"]
            best_loss_raw[iP, iseed] = e["best_loss"]
            best_acc_raw[iP, iseed] = e["best_acc"]
            best_err_raw[iP, iseed] = e["best_err"]
            best_epoch_raw[iP, iseed] = e["best_epoch"]
            last_saved_epoch_raw[iP, iseed] = e["last_saved_epoch"]

            for local_i, ep in enumerate(e["epochs"]):
                j = epoch_to_idx[int(ep)]
                for name in scalar_names:
                    if name in e and local_i < len(e[name]):
                        scalar_raw[name][iP, iseed, j] = e[name][local_i]

                for name in TOKEN_POSITION_KEYS:
                    arr = np.asarray(e.get(name, np.empty((0, 0))), dtype=float)
                    if arr.ndim >= 2 and local_i < arr.shape[0] and max_token_pos > 0:
                        n = min(max_token_pos, arr.shape[1])
                        token_position_raw[name][iP, iseed, j, :n] = arr[local_i, :n]

                for name in LOGIT_EFFDIM_POSITION_KEYS:
                    arr = np.asarray(e.get(name, np.empty((0, 0))), dtype=float)
                    if arr.ndim >= 2 and local_i < arr.shape[0] and max_token_pos > 0:
                        n = min(max_token_pos, arr.shape[1])
                        logit_effdim_position_raw[name][iP, iseed, j, :n] = arr[local_i, :n]

                for name in LOGIT_EFFDIM_EIGVAL_KEYS:
                    arr = np.asarray(e.get(name, np.empty((0, 0, 0))), dtype=float)
                    if arr.ndim >= 3 and local_i < arr.shape[0] and max_token_pos > 0 and max_logit_eigs > 0:
                        n0 = min(max_token_pos, arr.shape[1])
                        n1 = min(max_logit_eigs, arr.shape[2])
                        logit_effdim_eigval_raw[name][iP, iseed, j, :n0, :n1] = arr[local_i, :n0, :n1]

                for split in SPLITS:
                    ns_name = f"{split}_rhm_num_samples"
                    if ns_name in e and local_i < len(e[ns_name]):
                        scalar_raw[ns_name][iP, iseed, j] = e[ns_name][local_i]

                    for key in ML_VECTOR_KEYS:
                        name = f"{split}_{key}"
                        arr = np.asarray(e.get(name, np.empty((0, 0))), dtype=float)
                        if arr.ndim >= 2 and local_i < arr.shape[0] and max_L > 0:
                            n = min(max_L, arr.shape[1])
                            ml_vector_raw[name][iP, iseed, j, :n] = arr[local_i, :n]

                    for key in ML_POSITION_KEYS:
                        name = f"{split}_{key}"
                        arr = np.asarray(e.get(name, np.empty((0, 0, 0))), dtype=float)
                        if arr.ndim >= 3 and local_i < arr.shape[0] and max_Dpos > 0 and max_L > 0:
                            n0 = min(max_Dpos, arr.shape[1])
                            n1 = min(max_L, arr.shape[2])
                            ml_position_raw[name][iP, iseed, j, :n0, :n1] = arr[local_i, :n0, :n1]

    # Keep older convention: err = 1 - testacc if explicit err missing.
    missing_err = np.isnan(scalar_raw["err"])
    scalar_raw["err"] = np.where(missing_err, 1.0 - scalar_raw["testacc"], scalar_raw["err"])
    missing_trainerr = np.isnan(scalar_raw["trainerr"])
    scalar_raw["trainerr"] = np.where(missing_trainerr, 1.0 - scalar_raw["trainacc"], scalar_raw["trainerr"])

    result: Dict[str, Any] = {
        "run_name": np.array(args.run_name),
        "experiment_name": np.array(args.experiment_name or args.run_name),
        "fixed_params": np.array(fixed_params, dtype=object),
        "dataset": np.array(entries[0]["args"].get("dataset", "unknown")),
        "model": np.array(entries[0]["args"].get("model", "unknown")),
        "P_values": P_values,
        "epoch_values": epoch_values,
        "T_arr": epoch_values.copy(),
        "num_seeds": num_seeds,
        "seed_triplets": seed_triplets,
        "file_index": file_index,
        "ntp_num_positions": np.array(max_token_pos, dtype=int),
        "logit_effdim_num_eigvals": np.array(max_logit_eigs, dtype=int),
        "rhm_M_l_num_levels": np.array(max_L, dtype=int),
        "rhm_M_l_num_positions": np.array(max_Dpos, dtype=int),
        "best_loss_raw": best_loss_raw,
        "best_acc_raw": best_acc_raw,
        "best_err_raw": best_err_raw,
        "best_epoch_raw": best_epoch_raw,
        "last_saved_epoch_raw": last_saved_epoch_raw,
    }

    for name, raw in scalar_raw.items():
        _add_metric(result, name, raw)
    for name, raw in token_position_raw.items():
        _add_metric(result, name, raw)
    for name, raw in logit_effdim_position_raw.items():
        _add_metric(result, name, raw)
    for name, raw in logit_effdim_eigval_raw.items():
        _add_metric(result, name, raw)
    for name, raw in ml_vector_raw.items():
        _add_metric(result, name, raw)
    for name, raw in ml_position_raw.items():
        _add_metric(result, name, raw)

    for name, raw in (
        ("best_loss", best_loss_raw),
        ("best_acc", best_acc_raw),
        ("best_err", best_err_raw),
        ("best_epoch", best_epoch_raw),
        ("last_saved_epoch", last_saved_epoch_raw),
    ):
        mean, std, n = nanmean_std_count(raw, axis=1)
        result[f"{name}_mean"] = mean
        result[f"{name}_std"] = std
        result[f"{name}_n"] = n

    save_name = args.experiment_name or args.run_name
    save_path = results_dir / f"{save_name}.npy"
    np.save(save_path, result, allow_pickle=True)

    print(f"Saved aggregated numpy dict to:\n  {save_path}")
    print(f"Found {len(files)} raw result files under:\n  {run_dir}")
    print(f"P values: {P_values.tolist()}")
    print(f"Global validation epochs: {epoch_values.tolist()}")
    print(f"num_seeds per P: {num_seeds.tolist()}")
    print(f"trainloss_seeds shape: {result['trainloss_seeds'].shape}")
    print(f"testloss_seeds shape: {result['testloss_seeds'].shape}")
    print(f"err_seeds shape: {result['err_seeds'].shape}")
    print(f"spectral_seeds shape: {result['spectral_seeds'].shape}")
    print(f"spectral_no_qk_seeds shape: {result['spectral_no_qk_seeds'].shape}")
    print(f"l2_seeds shape: {result['l2_seeds'].shape}")
    if max_token_pos > 0:
        print(f"Per-token NTP diagnostics detected: positions={max_token_pos}")
        print(f"testloss_by_position_seeds shape: {result['testloss_by_position_seeds'].shape}")
        print(f"testacc_by_position_seeds shape: {result['testacc_by_position_seeds'].shape}")
        if "test_logit_effdim_entropy_by_position_seeds" in result:
            print(f"test_logit_effdim_entropy_by_position_seeds shape: {result['test_logit_effdim_entropy_by_position_seeds'].shape}")
        if "test_logit_cov_eigvals_by_position_seeds" in result:
            print(f"test_logit_cov_eigvals_by_position_seeds shape: {result['test_logit_cov_eigvals_by_position_seeds'].shape}")
    else:
        print("Per-token NTP diagnostics not detected; saved empty by-position arrays.")
    if max_L > 0:
        print(f"M_l diagnostics detected: L={max_L}, positions={max_Dpos}")
        print(f"test_rhm_M_pos_frac_seeds shape: {result['test_rhm_M_pos_frac_seeds'].shape}")
        print(f"test_rhm_peeled_loss_seeds shape: {result['test_rhm_peeled_loss_seeds'].shape}")
        print(f"test_rhm_M_pos_frac_by_position_seeds shape: {result['test_rhm_M_pos_frac_by_position_seeds'].shape}")
    else:
        print("M_l diagnostics not detected; saved empty M_l arrays.")


if __name__ == "__main__":
    main()
