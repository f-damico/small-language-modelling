"""Training loop with logarithmic validation/checkpoint schedules and RHM diagnostics."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.distributed as dist

import measures
from rhm_margins import compute_rhm_m_l_metrics


def _is_distributed(config) -> bool:
    return bool(getattr(config, "ddp", False)) and dist.is_available() and dist.is_initialized()


def _is_main_process(config) -> bool:
    return int(getattr(config, "rank", 0)) == 0


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _ddp_barrier(config):
    if _is_distributed(config):
        dist.barrier()


def _broadcast_stop_flag(config, stop: bool) -> bool:
    if not _is_distributed(config):
        return bool(stop)
    device = torch.device(getattr(config, "device", "cpu"))
    flag = torch.tensor([1 if stop else 0], device=device, dtype=torch.int32)
    dist.broadcast(flag, src=0)
    return bool(flag.item())



def train_step(model, trainset, criterion, optimizer, scheduler, device):
    """One optimizer step, retained as a public helper for old notebooks."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    inputs, targets = trainset.next_batch()
    inputs = inputs.to(device, non_blocking=True).long()
    targets = targets.to(device, non_blocking=True).long()
    logits = model(inputs)
    loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    loss.backward()
    optimizer.step()
    scheduler.step()
    return float(loss.item())


def _cpu_state_dict(model) -> Dict[str, torch.Tensor]:
    raw_model = _unwrap_model(model)
    return {name: value.detach().cpu().clone() for name, value in raw_model.state_dict().items()}


def _select_log_spaced_steps(total_steps: int, requested: int) -> np.ndarray:
    """Return exactly min(requested,total_steps) increasing integer checkpoints.

    Endpoints 1 and total_steps are always included.  Rounding is constrained so
    that duplicate points at early time do not reduce the requested count.
    """
    total_steps = int(total_steps)
    requested = int(requested)
    if total_steps <= 0 or requested <= 0:
        return np.empty(0, dtype=np.int64)
    count = min(total_steps, requested)
    if count == total_steps:
        return np.arange(1, total_steps + 1, dtype=np.int64)
    if count == 1:
        return np.array([total_steps], dtype=np.int64)

    targets = np.geomspace(1.0, float(total_steps), num=count)
    selected = []
    for i, value in enumerate(targets):
        remaining = count - i - 1
        lower = 1 if not selected else selected[-1] + 1
        upper = total_steps - remaining
        candidate = min(max(int(round(value)), lower), upper)
        selected.append(candidate)
    selected[0] = 1
    selected[-1] = total_steps
    return np.asarray(selected, dtype=np.int64)


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _save_run_data(run_dir: Path, config, data_info) -> Dict[str, str]:
    """Save rules/metadata and, only offline, the finite RHM sequences."""
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    args_path = run_dir / "args.json"
    with open(args_path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(vars(config)), handle, indent=2, sort_keys=True)
    paths["args"] = str(args_path)

    if not getattr(data_info, "is_rhm", False):
        return paths

    rules = getattr(data_info, "rules", None)
    if rules is not None:
        cpu_rules = {
            int(k): v.detach().cpu() if isinstance(v, torch.Tensor) else torch.as_tensor(v)
            for k, v in rules.items()
        }
        rules_path = run_dir / "rules.pt"
        tree_rules_path = data_dir / "tree_rules.pt"
        torch.save(cpu_rules, rules_path)
        torch.save(cpu_rules, tree_rules_path)
        paths["rules"] = str(rules_path)
        paths["tree_rules"] = str(tree_rules_path)

    metadata = {
        "dataset": "rhm",
        "online": bool(getattr(data_info, "online", False)),
        "sampling": getattr(data_info, "sampling", None),
        "num_features": int(config.num_features),
        "num_classes": int(config.num_classes),
        "num_synonyms": int(config.num_synonyms),
        "tuple_size": int(config.tuple_size),
        "num_layers": int(config.num_layers),
        "num_tokens": int(config.num_tokens),
        "train_size": int(config.train_size),
        "val_size": int(config.val_size),
        "seed_rules": int(config.seed_rules),
        "seed_sample": int(config.seed_sample),
    }
    metadata_path = data_dir / "tree_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    paths["tree_metadata"] = str(metadata_path)

    # Online training intentionally stores no finite train/test set: the fixed
    # tree and seeds are sufficient to reproduce the stream.  Offline runs can
    # optionally store their exact finite sequences.
    if not getattr(data_info, "online", False) and bool(config.save_run_data):
        train_sequences = getattr(data_info, "train_sequences", None)
        val_sequences = getattr(data_info, "val_sequences", None)
        if train_sequences is not None and val_sequences is not None:
            dataset_path = data_dir / "dataset_full.npz"
            np.savez_compressed(
                dataset_path,
                train_sequences=torch.as_tensor(train_sequences).cpu().numpy().astype(np.int64),
                val_sequences=torch.as_tensor(val_sequences).cpu().numpy().astype(np.int64),
            )
            paths["dataset_full"] = str(dataset_path)
    return paths


def _safe_exp(log_value: float) -> float:
    if not np.isfinite(log_value):
        return float(np.exp(min(log_value, 700.0))) if log_value != -np.inf else 0.0
    return float(np.exp(min(log_value, 700.0)))


@torch.no_grad()
def _norm_measures(model) -> Dict[str, float]:
    model = _unwrap_model(model)
    out: Dict[str, float] = {}
    if hasattr(model, "compute_model_log_norm"):
        out["log_specnorm"] = float(model.compute_model_log_norm().detach().cpu())
        out["specnorm"] = _safe_exp(out["log_specnorm"])
    elif hasattr(model, "compute_model_norm"):
        out["specnorm"] = float(model.compute_model_norm().detach().cpu())
        out["log_specnorm"] = float(np.log(out["specnorm"])) if out["specnorm"] > 0 else -np.inf

    if hasattr(model, "compute_model_log_norm_no_qk"):
        out["log_specnorm_no_qk"] = float(model.compute_model_log_norm_no_qk().detach().cpu())
        out["specnorm_no_qk"] = _safe_exp(out["log_specnorm_no_qk"])
    elif "specnorm" in out:
        out["log_specnorm_no_qk"] = out["log_specnorm"]
        out["specnorm_no_qk"] = out["specnorm"]

    if hasattr(model, "compute_l2_log_norm"):
        out["log_l2norm"] = float(model.compute_l2_log_norm().detach().cpu())
        out["l2norm"] = _safe_exp(out["log_l2norm"])
    elif hasattr(model, "compute_l2_norm"):
        out["l2norm"] = float(model.compute_l2_norm().detach().cpu())
        out["log_l2norm"] = float(np.log(out["l2norm"])) if out["l2norm"] > 0 else -np.inf
    return out


def _clone_for_diagnostics(loader, batch_size: int):
    if loader is None:
        return None
    if hasattr(loader, "clone"):
        try:
            return loader.clone(batch_size=batch_size)
        except TypeError:
            return loader.clone()
    return loader


def _format_vector(values) -> str:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return "[" + ", ".join("nan" if not np.isfinite(v) else f"{v:.5f}" for v in arr) + "]"


def _rank_eval_limit(max_samples, config):
    if max_samples is None or int(max_samples) <= 0:
        return max_samples
    if not _is_distributed(config):
        return int(max_samples)
    return int(math.ceil(int(max_samples) / max(1, int(getattr(config, "world_size", 1)))))


def _ranked_clone(loader, batch_size, config, *, seed_offset=0):
    """Clone an evaluation loader and make each DDP rank evaluate a different shard/stream."""
    out = _clone_for_diagnostics(loader, batch_size)
    if out is None or not _is_distributed(config):
        return out

    rank = int(getattr(config, "rank", 0))
    world = max(1, int(getattr(config, "world_size", 1)))

    # Online RHM: same rules, but different deterministic reference stream per rank.
    if hasattr(out, "seed"):
        out.seed = int(out.seed) + 10000019 * rank + int(seed_offset)
    if hasattr(out, "_generator"):
        out._generator = torch.Generator().manual_seed(int(getattr(out, "seed", 0)))

    # Offline loaders with a stored finite tensor: make a non-overlapping rank shard.
    if hasattr(out, "sequences") and out.sequences is not None:
        out.sequences = out.sequences[rank::world].contiguous()
        out.num_samples = int(len(out.sequences))
        out._order = torch.arange(out.num_samples)
        out.num_batches = max(1, math.ceil(out.num_samples / max(1, int(out.B))))

    if hasattr(out, "reset"):
        out.reset()
    return out


def _all_reduce_tensor(tensor, config):
    if _is_distributed(config):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def _aggregate_eval_detailed(local, config):
    """Aggregate evaluate_detailed outputs over DDP ranks."""
    if not _is_distributed(config):
        return local

    device = torch.device(getattr(config, "device", "cpu"))
    # Scalars: reconstruct additive totals.
    num_tokens = float(local.get("num_tokens", 0) or 0)
    num_samples = float(local.get("num_samples", 0) or 0)
    loss_sum = float(local.get("loss", float("nan"))) * num_tokens if num_tokens > 0 else 0.0
    correct_sum = float(local.get("acc", 0.0)) * num_tokens if num_tokens > 0 else 0.0
    totals = torch.tensor([loss_sum, correct_sum, num_tokens, num_samples], dtype=torch.float64, device=device)
    _all_reduce_tensor(totals, config)

    out = dict(local)
    global_tokens = float(totals[2].item())
    global_samples = int(round(float(totals[3].item())))
    if global_tokens <= 0:
        return measures._empty_detailed()

    out["loss"] = float(totals[0].item() / global_tokens)
    out["acc"] = float(totals[1].item() / global_tokens)
    out["err"] = 1.0 - out["acc"]
    out["num_tokens"] = int(round(global_tokens))
    out["num_samples"] = global_samples

    # Position-wise losses/accuracies: aggregate sums using count_by_position.
    count_np = np.asarray(local.get("count_by_position", []), dtype=np.float64).reshape(-1)
    D = int(count_np.size)
    if D > 0:
        count = torch.as_tensor(count_np, dtype=torch.float64, device=device)
        loss_by = torch.as_tensor(np.asarray(local["loss_by_position"], dtype=np.float64).reshape(-1), dtype=torch.float64, device=device)
        acc_by = torch.as_tensor(np.asarray(local["acc_by_position"], dtype=np.float64).reshape(-1), dtype=torch.float64, device=device)
        loss_sum_by = torch.nan_to_num(loss_by, nan=0.0) * count
        correct_sum_by = torch.nan_to_num(acc_by, nan=0.0) * count
        packed = torch.cat([loss_sum_by, correct_sum_by, count])
        _all_reduce_tensor(packed, config)
        loss_sum_by, correct_sum_by, count = packed[:D], packed[D:2*D], packed[2*D:]
        denom = count.clamp_min(1.0)
        loss_pos = (loss_sum_by / denom).detach().cpu().numpy()
        acc_pos = (correct_sum_by / denom).detach().cpu().numpy()
        invalid = count.detach().cpu().numpy() <= 0
        loss_pos[invalid] = np.nan
        acc_pos[invalid] = np.nan
        out["loss_by_position"] = loss_pos
        out["acc_by_position"] = acc_pos
        out["err_by_position"] = 1.0 - acc_pos
        out["count_by_position"] = count.detach().cpu().numpy()

    # Logit effective dimensions/eigenvalues are not additive after eigendecomposition.
    # Keep rank-0 values as a deterministic estimate so the saved keys remain compatible.
    for key in (
        "logit_effdim_entropy_by_position",
        "logit_effdim_pr_by_position",
        "logit_effdim_entropy_norm_by_position",
        "logit_effdim_pr_norm_by_position",
        "logit_cov_eigvals_by_position",
    ):
        value = local.get(key)
        # Broadcast rank-0 arrays through object collectives for shape compatibility.
        obj = [value if _is_main_process(config) else None]
        dist.broadcast_object_list(obj, src=0)
        out[key] = obj[0]
    return out


def _aggregate_rhm_metrics(local, config):
    """Aggregate compute_rhm_m_l_metrics outputs over DDP ranks."""
    if not _is_distributed(config):
        return local

    device = torch.device(getattr(config, "device", "cpu"))
    count_np = np.asarray(local["rhm_valid_count_by_position"], dtype=np.float64)
    Dpos, L = count_np.shape
    count = torch.as_tensor(count_np, dtype=torch.float64, device=device)

    def weighted_sum(key):
        arr = np.asarray(local[key], dtype=np.float64)
        return torch.nan_to_num(torch.as_tensor(arr, dtype=torch.float64, device=device), nan=0.0) * count

    sum_M = weighted_sum("rhm_M_mean_by_position")
    sum_pos = weighted_sum("rhm_M_pos_frac_by_position")
    sum_peel = weighted_sum("rhm_peeled_loss_by_position")
    samples = torch.tensor([float(local.get("rhm_num_samples", 0) or 0)], dtype=torch.float64, device=device)

    packed = torch.cat([sum_M.reshape(-1), sum_pos.reshape(-1), sum_peel.reshape(-1), count.reshape(-1), samples])
    _all_reduce_tensor(packed, config)

    n = Dpos * L
    sum_M = packed[:n].reshape(Dpos, L)
    sum_pos = packed[n:2*n].reshape(Dpos, L)
    sum_peel = packed[2*n:3*n].reshape(Dpos, L)
    count = packed[3*n:4*n].reshape(Dpos, L)
    total_samples = int(round(float(packed[-1].item())))

    denom = count.clamp_min(1.0)
    mean_by_pos = (sum_M / denom).detach().cpu().numpy()
    pos_by_pos = (sum_pos / denom).detach().cpu().numpy()
    peel_by_pos = (sum_peel / denom).detach().cpu().numpy()
    valid_count_by_pos = count.detach().cpu().numpy().astype(np.int64)
    invalid = valid_count_by_pos == 0
    mean_by_pos[invalid] = np.nan
    pos_by_pos[invalid] = np.nan
    peel_by_pos[invalid] = np.nan

    level_count = count.sum(dim=0).clamp_min(1.0)
    M_mean = (sum_M.sum(dim=0) / level_count).detach().cpu().numpy()
    M_pos = (sum_pos.sum(dim=0) / level_count).detach().cpu().numpy()
    peeled_level = (sum_peel.sum(dim=0) / level_count).detach().cpu().numpy()
    valid_frac = (count.sum(dim=0) / max(1, total_samples * Dpos)).detach().cpu().numpy()
    zero = count.sum(dim=0).detach().cpu().numpy() == 0
    M_mean[zero] = np.nan
    M_pos[zero] = np.nan
    peeled_level[zero] = np.nan

    return {
        "rhm_M_mean": M_mean,
        "rhm_M_pos_frac": M_pos,
        "rhm_peeled_loss": peeled_level,
        "rhm_valid_frac": valid_frac,
        "rhm_M_mean_by_position": mean_by_pos,
        "rhm_M_pos_frac_by_position": pos_by_pos,
        "rhm_peeled_loss_by_position": peel_by_pos,
        "rhm_valid_count_by_position": valid_count_by_pos,
        "rhm_num_samples": total_samples,
    }


def _compute_rhm_diagnostics_pair(model, train_eval_loader, val_loader, config, data_info):
    model = _unwrap_model(model)
    if not bool(config.compute_rhm_diagnostics):
        return None, None
    if not getattr(data_info, "is_rhm", False):
        raise ValueError("--compute_rhm_diagnostics/--compute_M_l is only valid with --dataset rhm")

    train_diag_loader = _ranked_clone(train_eval_loader, config.rhm_margins_batch_size, config, seed_offset=33331)
    val_diag_loader = _ranked_clone(val_loader, config.rhm_margins_batch_size, config, seed_offset=77773)
    train_local = compute_rhm_m_l_metrics(
        model,
        train_diag_loader,
        config,
        data_info.rules,
        max_samples=_rank_eval_limit(config.rhm_margins_max_train_samples, config),
    )
    val_local = compute_rhm_m_l_metrics(
        model,
        val_diag_loader,
        config,
        data_info.rules,
        max_samples=_rank_eval_limit(config.rhm_margins_max_val_samples, config),
    )
    return _aggregate_rhm_metrics(train_local, config), _aggregate_rhm_metrics(val_local, config)


def _add_rhm_diagnostics(entry, train_metrics, val_metrics):
    if train_metrics is None or val_metrics is None:
        return
    for key, value in train_metrics.items():
        entry[f"train_{key}"] = value
    for key, value in val_metrics.items():
        entry[f"test_{key}"] = value

    print(
        f"[RHM] step={entry['t']} train_peeled_loss_l="
        f"{_format_vector(train_metrics['rhm_peeled_loss'])} "
        f"val_peeled_loss_l={_format_vector(val_metrics['rhm_peeled_loss'])}",
        flush=True,
    )
    print(
        f"[RHM] step={entry['t']} train_Mpos_l="
        f"{_format_vector(train_metrics['rhm_M_pos_frac'])} "
        f"val_Mpos_l={_format_vector(val_metrics['rhm_M_pos_frac'])} "
        f"train_valid_l={_format_vector(train_metrics['rhm_valid_frac'])} "
        f"val_valid_l={_format_vector(val_metrics['rhm_valid_frac'])}",
        flush=True,
    )

def _atomic_torch_save(payload, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _result_payload(config, dynamics, best, model, data_paths, schedules, step):
    return {
        "config": config,
        "dynamics": dynamics,
        "best": best,
        "model": _cpu_state_dict(model),
        "step": int(step),
        "epoch": float(step) / max(1, int(config.steps_per_epoch)),
        "time_unit": "step",
        "data_paths": data_paths,
        "validation_steps": schedules["validation_steps"],
        "weight_save_steps": schedules["weight_save_steps"],
    }


def _validate(
    *,
    step: int,
    recent_loss: float,
    model,
    train_eval_loader,
    val_loader,
    config,
    data_info,
    dynamics,
    best,
    criterion,
    run_dir: Path,
    data_paths,
    schedules,
    start_time: float,
):
    eval_model = _unwrap_model(model)
    compute_effdim = bool(config.compute_rhm_diagnostics and getattr(data_info, "is_rhm", False))

    # Every DDP rank evaluates a different shard/stream; then additive statistics are reduced.
    val_eval_loader_rank = _ranked_clone(val_loader, config.eval_batch_size, config, seed_offset=91001)
    train_eval_loader_rank = _ranked_clone(train_eval_loader, config.eval_batch_size, config, seed_offset=120011)

    val_local = measures.evaluate_detailed(
        eval_model,
        val_eval_loader_rank,
        config.device,
        max_samples=_rank_eval_limit(config.eval_val_size, config),
        compute_logit_effdim=compute_effdim,
    )
    val_eval = _aggregate_eval_detailed(val_local, config)

    if config.measure_train or config.compute_rhm_diagnostics:
        train_local = measures.evaluate_detailed(
            eval_model,
            train_eval_loader_rank,
            config.device,
            max_samples=_rank_eval_limit(config.eval_train_size, config),
            compute_logit_effdim=compute_effdim,
        )
        train_eval = _aggregate_eval_detailed(train_local, config)
    else:
        train_eval = measures._empty_detailed()

    train_rhm_metrics, val_rhm_metrics = _compute_rhm_diagnostics_pair(
        model,
        train_eval_loader,
        val_loader,
        config,
        data_info,
    )

    entry = {
        "t": int(step),
        "step": int(step),
        "epoch": float(step) / max(1, int(config.steps_per_epoch)),
        "running": float(recent_loss),
        "val_loss": float(val_eval["loss"]),
        "train_loss": float(train_eval["loss"]),
        "testloss": float(val_eval["loss"]),
        "testacc": float(val_eval["acc"]),
        "testerr": float(val_eval["err"]),
        "trainloss": float(train_eval["loss"]),
        "trainacc": float(train_eval["acc"]),
        "trainerr": float(train_eval["err"]),
        "ngram_loss": np.asarray(val_eval["loss_by_position"], dtype=np.float64),
        "testloss_by_position": val_eval["loss_by_position"],
        "testacc_by_position": val_eval["acc_by_position"],
        "testerr_by_position": val_eval["err_by_position"],
        "test_count_by_position": val_eval["count_by_position"],
        "trainloss_by_position": train_eval["loss_by_position"],
        "trainacc_by_position": train_eval["acc_by_position"],
        "trainerr_by_position": train_eval["err_by_position"],
        "train_count_by_position": train_eval["count_by_position"],
        "wall_time": time.time() - start_time,
    }
    for split, evaluated in (("train", train_eval), ("test", val_eval)):
        for key in (
            "logit_effdim_entropy_by_position",
            "logit_effdim_pr_by_position",
            "logit_effdim_entropy_norm_by_position",
            "logit_effdim_pr_norm_by_position",
            "logit_cov_eigvals_by_position",
        ):
            entry[f"{split}_{key}"] = evaluated[key]

    if not _is_main_process(config):
        return float(val_eval["loss"])

    entry.update(_norm_measures(model))
    _add_rhm_diagnostics(entry, train_rhm_metrics, val_rhm_metrics)
    dynamics.append(entry)

    if val_eval["loss"] < best["loss"]:
        best.update(
            {
                "step": int(step),
                "epoch": entry["epoch"],
                "loss": float(val_eval["loss"]),
                "acc": float(val_eval["acc"]),
                "err": float(val_eval["err"]),
                "model": _cpu_state_dict(model),
            }
        )
        _atomic_torch_save(
            {"model": best["model"], "step": step, "state": entry, "config": config},
            run_dir / "best_model.pt",
        )

    print(
        f"[VALID] step={step} epoch={entry['epoch']:.4f} running={recent_loss:.6f} "
        f"train_loss={train_eval['loss']:.6f} val_loss={val_eval['loss']:.6f} "
        f"val_acc={val_eval['acc']:.6f} log_specnorm={entry.get('log_specnorm', float('nan')):.6f}",
        flush=True,
    )
    _atomic_torch_save(
        {"model": _cpu_state_dict(model), "step": step, "state": entry, "config": config},
        run_dir / "latest_model.pt",
    )
    _atomic_torch_save(
        _result_payload(config, dynamics, best, model, data_paths, schedules, step),
        run_dir / "results.pt",
    )
    return float(val_eval["loss"])

def train_model(
    model,
    train_loader,
    val_loader,
    train_eval_loader,
    criterion,
    optimizer,
    scheduler,
    config,
    data_info,
):
    """Train and continuously save a recoverable run under ``config.output_dir``."""
    run_dir = Path(config.output_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    if _is_main_process(config):
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        data_paths = _save_run_data(run_dir, config, data_info)
    else:
        data_paths = {}
    _ddp_barrier(config)

    max_steps = int(config.max_steps)
    config.steps_per_epoch = int(train_loader.num_batches)
    validation_steps = _select_log_spaced_steps(max_steps, config.num_validations)
    weight_save_steps = _select_log_spaced_steps(max_steps, config.num_weight_saves)
    validation_set = set(validation_steps.tolist())
    weight_set = set(weight_save_steps.tolist())
    if _is_main_process(config):
        np.savetxt(run_dir / "validation_steps.txt", validation_steps, fmt="%d")
        np.savetxt(checkpoint_dir / "checkpoint_save_steps.txt", weight_save_steps, fmt="%d")
    schedules = {
        "validation_steps": validation_steps,
        "weight_save_steps": weight_save_steps,
    }

    if _is_main_process(config):
        print(f"[INFO] output_dir={run_dir}", flush=True)
        print(f"[INFO] world_size={getattr(config, 'world_size', 1)} batch_size_per_rank={config.batch_size} effective_batch_size={int(config.batch_size) * int(getattr(config, 'world_size', 1))}", flush=True)
        print(f"[INFO] validation steps={validation_steps.tolist()}", flush=True)
        print(f"[INFO] weight checkpoint steps={weight_save_steps.tolist()}", flush=True)
        print(
            f"[INFO] RHM diagnostics={config.compute_rhm_diagnostics}; "
            f"online={config.online}; dataset={config.dataset}; model={config.model}",
            flush=True,
        )

    best = {
        "step": -1,
        "epoch": -1.0,
        "loss": float("inf"),
        "acc": float("nan"),
        "err": float("nan"),
        "model": None,
    }
    dynamics = []
    start_time = time.time()
    initial_loss = math.log(max(2, int(config.vocab_size)))
    _validate(
            step=0,
            recent_loss=initial_loss,
            model=model,
            train_eval_loader=train_eval_loader,
            val_loader=val_loader,
            config=config,
            data_info=data_info,
            dynamics=dynamics,
            best=best,
            criterion=criterion,
            run_dir=run_dir,
            data_paths=data_paths,
            schedules=schedules,
            start_time=start_time,
        )
    _ddp_barrier(config)

    recent_loss = initial_loss
    for step in range(1, max_steps + 1):
        t0 = time.time()
        recent_loss = train_step(model, train_loader, criterion, optimizer, scheduler, config.device)
        dt_ms = (time.time() - t0) * 1000.0

        if _is_main_process(config) and config.print_freq > 0 and (step == 1 or step % config.print_freq == 0):
            print(
                f"[TRAIN] step={step}/{max_steps} epoch={step / config.steps_per_epoch:.4f} "
                f"loss={recent_loss:.6f} dt={dt_ms:.2f}ms",
                flush=True,
            )

        if _is_main_process(config) and step in weight_set:
            checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pt"
            _atomic_torch_save(
                {
                    "model": _cpu_state_dict(model),
                    "step": int(step),
                    "epoch": float(step) / config.steps_per_epoch,
                    "config": config,
                },
                checkpoint_path,
            )
            print(f"[CHECKPOINT] saved {checkpoint_path}", flush=True)

        do_validate = step in validation_set
        if not validation_set and config.save_freq > 0:
            do_validate = step % int(config.save_freq) == 0 or step == max_steps

        stop_now = False
        if do_validate:
            val_loss = _validate(
                step=step,
                recent_loss=recent_loss,
                model=model,
                train_eval_loader=train_eval_loader,
                val_loader=val_loader,
                config=config,
                data_info=data_info,
                dynamics=dynamics,
                best=best,
                criterion=criterion,
                run_dir=run_dir,
                data_paths=data_paths,
                schedules=schedules,
                start_time=start_time,
            )
            if _is_main_process(config) and config.loss_threshold is not None and val_loss <= config.loss_threshold:
                print(
                    f"[STOP] val_loss={val_loss:.6g} <= loss_threshold={config.loss_threshold}",
                    flush=True,
                )
                stop_now = True
            stop_now = _broadcast_stop_flag(config, stop_now)
            _ddp_barrier(config)
            if stop_now:
                break

    _ddp_barrier(config)
    if _is_main_process(config):
        return _result_payload(config, dynamics, best, model, data_paths, schedules, step)
    return None


# Compatibility alias for code that imports ``train.train``.
train = train_model
