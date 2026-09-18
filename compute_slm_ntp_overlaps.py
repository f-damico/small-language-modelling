#!/usr/bin/env python3
"""Unified overlap driver for Transformer next-token-prediction runs.

This script keeps the established weight/CKA implementations in
``compute_slm_representation_overlaps.py`` and adds three inexpensive,
checkpoint-to-checkpoint representation overlaps:

    weights             existing cosine weight overlap
    cka                 existing state CKA
    block_update_cka    existing CKA of h_l - h_{l-1}
    low_rank            finite-rank sample-space subspace overlap
    novelty             CKA of layer novelty after regressing h_l on h_{l-1}
    task                CKA of linearly next-token-predictive representations
    stage1              matched probe, rank, novelty and RHM loss diagnostics

The new learned maps are fitted once per checkpoint and layer, pooling all token
positions of a fixed *training* reference set, and are evaluated only on a
separate fixed *validation* reference set.  Thus no map is fitted separately for
checkpoint pairs (t,t').

For novelty, one map A_l(t) is fitted as

    h_l ~= mean_y + (h_{l-1} - mean_x) A_l(t)

and the held-out residual is compared by CKA.

For task overlap, one ridge probe W_l(t) is fitted from h_l to the centered
one-hot next-token target.  On validation data the task-predictive geometry is
that of Z = H W.  Its Gram matrix is evaluated efficiently as

    K_Z = H_c (W W^T) H_c^T,

so TinyStories does not require materialising [samples, vocab_size] probe
outputs at every token position.

Expected repository layout and checkpoint format are the same as
``compute_slm_representation_overlaps.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from numpy.lib.format import open_memmap
from torch.utils.data import DataLoader, TensorDataset

import compute_slm_representation_overlaps as base


EPS = 1e-12
NEW_METHODS = {"low_rank", "novelty", "task"}
ALL_DEFINITIONS = (
    "stage1",
    "weights",
    "cka",
    "block_update_cka",
    "low_rank",
    "novelty",
    "task",
)


# =============================================================================
# Small generic helpers
# =============================================================================


def parse_bool_flag(value: bool) -> bool:
    return bool(value)


def require_transformer(config: argparse.Namespace) -> None:
    model_name = str(getattr(config, "model", "gpt2")).lower()
    if model_name not in {"gpt2", "transformer_v2"}:
        raise ValueError(
            "This unified NTP overlap job is intentionally restricted to "
            "Transformer runs. Expected config.model in {'gpt2','transformer_v2'}, "
            f"got {model_name!r}."
        )


def selected_checkpoints(
    run_dir: Path,
    max_step: Optional[int],
    number_selected: int,
) -> Tuple[List[Path], np.ndarray, List[Path]]:
    all_checkpoints = base.find_checkpoints(run_dir, max_step)
    indices = base.log_select_indices(len(all_checkpoints), int(number_selected))
    return [all_checkpoints[int(i)] for i in indices], indices, all_checkpoints


def default_output_name(run_dir: Path, definition: str) -> str:
    return f"{run_dir.parent.name}_{run_dir.name}_{definition}_overlaps"


def resolved_temp_dir(
    repo_dir: Path,
    output_dir: Path,
    output_name: str,
    temp_root: Optional[Path],
) -> Path:
    root = output_dir if temp_root is None else base.resolve_relative(temp_root, repo_dir)
    return root / f"_tmp_{output_name}"


def torch_dtype(name: str) -> torch.dtype:
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unknown dtype {name!r}")


def fit_device(cli: argparse.Namespace) -> torch.device:
    requested = str(cli.linear_fit_device)
    if requested == "auto":
        requested = str(cli.device)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] linear-fit CUDA unavailable; using CPU", flush=True)
        requested = "cpu"
    return torch.device(requested)


def cleanup_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# Fixed NTP reference data with explicit next-token targets
# =============================================================================


def _normalize_full_token_sequences(
    value: Any,
    config: argparse.Namespace,
    required_length: int,
) -> np.ndarray:
    """Normalize saved/generated sequences without truncating away the target."""
    if torch.is_tensor(value):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)

    if array.ndim == 3:
        vocab = int(config.vocab_size)
        if array.shape[1] == vocab:
            array = array.argmax(axis=1)
        elif array.shape[2] == vocab:
            array = array.argmax(axis=2)
        else:
            raise ValueError(f"Cannot infer one-hot layout for {array.shape}")

    if array.ndim != 2:
        raise ValueError(f"Expected token sequences [N,T], got {array.shape}")
    if array.shape[1] < required_length:
        raise ValueError(
            f"Need at least {required_length} tokens per sequence for NTP inputs+targets; "
            f"got {array.shape[1]}."
        )

    array = array[:, :required_length].astype(np.int64, copy=False)
    if array.size:
        vocab = int(config.vocab_size)
        mn = int(array.min())
        mx = int(array.max())
        if mn >= 1 and mx == vocab:
            array = array - 1
            mn = int(array.min())
            mx = int(array.max())
        if mn < 0 or mx >= vocab:
            raise ValueError(
                f"Token ids outside [0,{vocab - 1}]: min={mn}, max={mx}"
            )
    return np.ascontiguousarray(array)


def _load_saved_rhm_full(
    run_dir: Path,
    config: argparse.Namespace,
    train_size: int,
    valid_size: int,
    subset_seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    candidates = (
        run_dir / "data" / "dataset_full.npz",
        run_dir / "dataset_full.npz",
        run_dir / "data" / "dataset_reference_subset.npz",
    )
    train_names = (
        "train_sequences",
        "train_rhm_sequences",
        "train_tokens",
        "train_inputs",
    )
    valid_names = (
        "val_sequences",
        "valid_sequences",
        "test_sequences",
        "val_rhm_sequences",
        "test_rhm_sequences",
        "valid_tokens",
        "test_tokens",
    )
    required_length = int(config.block_size) + 1

    for path in candidates:
        if not path.exists():
            continue
        with np.load(path, allow_pickle=True) as payload:
            train_raw = base.npz_first(payload, train_names)
            valid_raw = base.npz_first(payload, valid_names)
            if train_raw is None or valid_raw is None:
                continue

            rng = np.random.default_rng(int(subset_seed))
            train_all = _normalize_full_token_sequences(
                train_raw, config, required_length
            )
            valid_all = _normalize_full_token_sequences(
                valid_raw, config, required_length
            )
            train, train_idx = base.choose_rows(train_all, train_size, rng)
            valid, valid_idx = base.choose_rows(valid_all, valid_size, rng)
            return (
                torch.from_numpy(train).long(),
                torch.from_numpy(valid).long(),
                {
                    "reference_source": "saved_rhm_dataset_full_ntp",
                    "reference_detail": str(path),
                    "subset_seed": int(subset_seed),
                    "train_indices": train_idx.tolist(),
                    "valid_indices": valid_idx.tolist(),
                    "full_sequence_length": required_length,
                },
            )

    raise FileNotFoundError("No saved full RHM dataset was found.")


def _generate_rhm_full(
    run_dir: Path,
    config: argparse.Namespace,
    train_size: int,
    valid_size: int,
    subset_seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    rules_path = base.find_rules(run_dir)
    rules = base.normalize_rules(base.safe_load(rules_path))
    generator = torch.Generator(device="cpu").manual_seed(int(subset_seed))
    number_classes = int(
        getattr(config, "num_classes", rules[min(rules)].shape[0])
    )
    labels = torch.randint(
        0,
        number_classes,
        (train_size + valid_size,),
        generator=generator,
    )
    features = labels.view(-1, 1)
    for level in sorted(rules):
        rule = rules[level]
        selected_rules = torch.randint(
            0,
            rule.shape[1],
            features.shape,
            generator=generator,
        )
        features = rule[features, selected_rules].flatten(start_dim=1)

    required_length = int(config.block_size) + 1
    full = _normalize_full_token_sequences(features, config, required_length)
    return (
        torch.from_numpy(full[:train_size]).long(),
        torch.from_numpy(full[train_size:]).long(),
        {
            "reference_source": "generated_from_rules_full_ntp",
            "reference_detail": str(rules_path),
            "subset_seed": int(subset_seed),
            "train_indices": list(range(train_size)),
            "valid_indices": list(range(valid_size)),
            "full_sequence_length": required_length,
        },
    )


def load_ntp_task_references(
    config: argparse.Namespace,
    cli: argparse.Namespace,
    repo_dir: Path,
    source_dir: Path,
    run_dir: Path,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Return fixed train/valid (inputs, next-token targets)."""
    train_size = int(cli.subset_train_size)
    valid_size = int(cli.subset_valid_size)
    subset_seed = int(cli.subset_seed)
    if train_size <= 0 or valid_size <= 0:
        raise ValueError("subset sizes must be positive")

    if base.is_rhm(config, run_dir):
        reference_source = str(cli.reference_source)
        online = bool(getattr(config, "online", False))
        if reference_source == "generate" or (reference_source == "auto" and online):
            train_full, valid_full, metadata = _generate_rhm_full(
                run_dir, config, train_size, valid_size, subset_seed
            )
        else:
            try:
                train_full, valid_full, metadata = _load_saved_rhm_full(
                    run_dir, config, train_size, valid_size, subset_seed
                )
            except FileNotFoundError:
                if reference_source == "saved":
                    raise
                train_full, valid_full, metadata = _generate_rhm_full(
                    run_dir, config, train_size, valid_size, subset_seed
                )
    else:
        train_path = base.resolve_text_corpus(
            config, "train", repo_dir, source_dir, run_dir
        )
        valid_path = base.resolve_text_corpus(
            config, "valid", repo_dir, source_dir, run_dir
        )
        window = int(config.block_size) + 1
        train_full, train_starts = base.sample_text_windows(
            train_path, train_size, window, subset_seed
        )
        valid_full, valid_starts = base.sample_text_windows(
            valid_path, valid_size, window, subset_seed + 1
        )
        metadata = {
            "reference_source": "text_corpus_windows_full_ntp",
            "reference_detail": {"train": str(train_path), "valid": str(valid_path)},
            "subset_seed": subset_seed,
            "train_indices": train_starts.tolist(),
            "valid_indices": valid_starts.tolist(),
            "full_sequence_length": window,
        }

    train_inputs = train_full[:, :-1].contiguous()
    train_targets = train_full[:, 1:].contiguous()
    valid_inputs = valid_full[:, :-1].contiguous()
    valid_targets = valid_full[:, 1:].contiguous()

    expected = int(config.block_size)
    if train_inputs.shape[1] != expected or valid_inputs.shape[1] != expected:
        raise RuntimeError(
            f"NTP reference length mismatch: expected block_size={expected}, "
            f"train={train_inputs.shape[1]}, valid={valid_inputs.shape[1]}"
        )

    metadata = dict(metadata)
    metadata.update(
        {
            "train_num_sequences": int(len(train_inputs)),
            "valid_num_sequences": int(len(valid_inputs)),
            "input_length": expected,
            "target_rule": "target[:,i] = full_sequence[:,i+1]",
            "fit_split": "train",
            "evaluation_split": "valid",
        }
    )
    return train_inputs, train_targets, valid_inputs, valid_targets, metadata


# =============================================================================
# Raw Transformer state extraction for the three new methods
# =============================================================================


def state_path(
    temp_dir: Path,
    split: str,
    checkpoint_index: int,
    layer_index: int,
) -> Path:
    return (
        temp_dir
        / "states"
        / split
        / f"state_k{checkpoint_index:05d}_l{layer_index:03d}.npy"
    )


@torch.no_grad()
def store_state_checkpoint(
    checkpoint_path: Path,
    checkpoint_index: int,
    config: argparse.Namespace,
    init_module: Any,
    references: Mapping[str, torch.Tensor],
    cli: argparse.Namespace,
    temp_dir: Path,
    *,
    include_embedding: bool,
    include_final_norm: bool,
) -> Tuple[List[str], Dict[str, Any], str]:
    """Stream raw state-mode representations to [sample,position,feature] memmaps."""
    model, metadata = base.reconstruct_model(
        checkpoint_path, config, init_module, cli.device
    )
    model_architecture = base.architecture(model)
    if model_architecture != "transformer":
        raise ValueError("New NTP overlap definitions are Transformer-only.")

    names = base.representation_names(
        model,
        "state",
        include_embedding,
        include_final_norm,
    )

    if getattr(cli, "overlap_definition", None) == "stage1":
        grouped = {}
        for name, parameter in model.named_parameters():
            group, _, _ = base.classify_parameter(name, model_architecture)
            grouped.setdefault(group, []).append(parameter.detach().cpu().reshape(-1))
        values = {name: torch.cat(parts).double().numpy() for name, parts in grouped.items()}
        ref_path = temp_dir / 'stage1_weight_reference.npz'
        if not ref_path.exists():
            np.savez(ref_path, **values)
        stats = {}
        with np.load(ref_path, allow_pickle=False) as ref:
            for name, value in values.items():
                old = ref[name]
                norm, oldnorm = np.linalg.norm(value), np.linalg.norm(old)
                stats[name] = np.array([
                    np.dot(value, old) / (norm*oldnorm) if norm*oldnorm>EPS else np.nan,
                    norm, np.linalg.norm(value-old) / oldnorm if oldnorm>EPS else np.nan])
        np.savez(temp_dir / f'stage1_weights_{checkpoint_index}.npz', **stats)

    for split, reference in references.items():
        loader = DataLoader(
            TensorDataset(reference),
            batch_size=int(cli.batch_size),
            shuffle=False,
            num_workers=0,
            pin_memory=str(cli.device).startswith("cuda"),
        )
        writers: Optional[List[np.memmap]] = None
        offset = 0
        logits_writer = None

        for (batch,) in loader:
            reps = base.forward_representations(
                model,
                batch.to(cli.device, non_blocking=True),
                "state",
                include_embedding,
                include_final_norm,
            )
            if getattr(cli, "overlap_definition", None) == "stage1" and split == "valid":
                if not include_final_norm:
                    raise ValueError("Stage 1 requires final_norm for logits.")
                logits = model.lm_head(reps[-1].to(cli.device)).detach().float().cpu().numpy()
                if logits_writer is None:
                    logits_writer = open_memmap(temp_dir / f"stage1_logits_{checkpoint_index}.npy",
                        mode="w+", dtype=np.float32,
                        shape=(len(reference), logits.shape[1], logits.shape[2]))
                logits_writer[offset:offset+len(batch)] = logits
            if writers is None:
                writers = []
                for layer_index, rep in enumerate(reps):
                    path = state_path(temp_dir, split, checkpoint_index, layer_index)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    writers.append(
                        open_memmap(
                            path,
                            mode="w+",
                            dtype=np.float32,
                            shape=(
                                len(reference),
                                int(rep.shape[1]),
                                int(rep.shape[2]),
                            ),
                        )
                    )
            assert writers is not None
            bs = int(batch.shape[0])
            for layer_index, rep in enumerate(reps):
                writers[layer_index][offset : offset + bs] = rep.numpy()
            offset += bs

        if writers is None or offset != len(reference):
            raise RuntimeError(
                f"State extraction wrote {offset}/{len(reference)} samples for {split}."
            )
        for writer in writers:
            writer.flush()
        del writers
        if logits_writer is not None:
            logits_writer.flush()
            del logits_writer

    del model
    cleanup_cuda()
    return names, metadata, model_architecture


def remove_checkpoint_states(temp_dir: Path, checkpoint_index: int) -> None:
    root = temp_dir / "states"
    if not root.exists():
        return
    for path in root.glob(f"*/state_k{checkpoint_index:05d}_l*.npy"):
        path.unlink(missing_ok=True)


# =============================================================================
# Low-rank subspace overlap
# =============================================================================


def low_rank_path(temp_dir: Path, checkpoint_index: int, layer_index: int) -> Path:
    return (
        temp_dir
        / "low_rank"
        / f"basis_k{checkpoint_index:05d}_l{layer_index:03d}.npy"
    )


def randomized_left_basis(
    states: np.ndarray,
    rank: int,
    oversample: int,
    power_iters: int,
    device: torch.device,
    seed: int,
    position_chunk: int,
) -> np.ndarray:
    """Approximate leading left singular vectors independently at each position."""
    number_samples, number_positions, feature_dim = states.shape
    maximum_rank = min(number_samples - 1, feature_dim)
    if rank <= 0 or rank > maximum_rank:
        raise ValueError(
            f"low_rank_rank={rank} must be in [1,{maximum_rank}] for "
            f"states shape {states.shape}."
        )
    qdim = min(maximum_rank, rank + max(0, int(oversample)))
    out = np.empty((number_positions, number_samples, rank), dtype=np.float32)
    chunk_size = max(1, int(position_chunk))

    for p0 in range(0, number_positions, chunk_size):
        p1 = min(number_positions, p0 + chunk_size)
        # [position, sample, feature]
        block = np.array(states[:, p0:p1, :], dtype=np.float32, copy=True)
        h = torch.as_tensor(block, device=device).permute(1, 0, 2).contiguous()
        h = h - h.mean(dim=1, keepdim=True)

        omegas = []
        for position in range(p0, p1):
            rng = np.random.default_rng(int(seed) + 104729 * int(position + 1))
            omegas.append(rng.standard_normal((feature_dim, qdim), dtype=np.float32))
        omega = torch.as_tensor(np.stack(omegas), device=device)

        y = h @ omega
        for _ in range(max(0, int(power_iters))):
            y = h @ (h.transpose(-2, -1) @ y)

        qmat, _ = torch.linalg.qr(y, mode="reduced")
        small = qmat.transpose(-2, -1) @ h
        u_small, _, _ = torch.linalg.svd(small, full_matrices=False)
        u = qmat @ u_small[:, :, :rank]
        out[p0:p1] = u.detach().float().cpu().numpy()
        del h, omega, y, qmat, small, u_small, u
        cleanup_cuda()

    return out


def pairwise_low_rank_overlap(
    temp_dir: Path,
    number_checkpoints: int,
    number_layers: int,
    number_positions: int,
    rank: int,
    device: torch.device,
    position_chunk: int,
) -> np.ndarray:
    output = np.full(
        (number_positions, number_layers, number_checkpoints, number_checkpoints),
        np.nan,
        dtype=np.float32,
    )
    chunk_size = max(1, int(position_chunk))

    for layer_index in range(number_layers):
        files = [
            np.load(low_rank_path(temp_dir, k, layer_index), mmap_mode="r")
            for k in range(number_checkpoints)
        ]
        for p0 in range(0, number_positions, chunk_size):
            p1 = min(number_positions, p0 + chunk_size)
            stacked = np.stack(
                [np.asarray(array[p0:p1], dtype=np.float32) for array in files],
                axis=0,
            )
            q = torch.as_tensor(stacked, device=device)
            # cross[p,k,l,a,b] = sum_n Q[k,p,n,a] Q[l,p,n,b]
            cross = torch.einsum("kpnr,lpns->pklrs", q, q)
            values = (cross * cross).sum(dim=(-1, -2)) / float(rank)
            output[p0:p1, layer_index] = values.detach().float().cpu().numpy()
            del q, cross, values
            cleanup_cuda()

    return np.clip(output, 0.0, 1.0)


def compute_low_rank(
    checkpoints: Sequence[Path],
    config: argparse.Namespace,
    init_module: Any,
    valid_reference: torch.Tensor,
    cli: argparse.Namespace,
    temp_dir: Path,
) -> Tuple[np.ndarray, List[str], List[Dict[str, Any]], str]:
    layer_names_ref: Optional[List[str]] = None
    metadata_list: List[Dict[str, Any]] = []
    architecture_ref: Optional[str] = None
    fit_dev = fit_device(cli)
    n_positions = int(valid_reference.shape[1])

    for checkpoint_index, checkpoint in enumerate(checkpoints):
        print(
            f"[INFO] low-rank extraction {checkpoint_index + 1}/{len(checkpoints)} "
            f"{checkpoint.name}",
            flush=True,
        )
        names, metadata, arch = store_state_checkpoint(
            checkpoint,
            checkpoint_index,
            config,
            init_module,
            {"valid": valid_reference},
            cli,
            temp_dir,
            include_embedding=bool(cli.include_embedding),
            include_final_norm=bool(cli.include_final_norm),
        )
        if layer_names_ref is None:
            layer_names_ref = names
            architecture_ref = arch
        elif names != layer_names_ref or arch != architecture_ref:
            raise RuntimeError("Representation layout changed between checkpoints.")
        metadata_list.append(metadata)

        for layer_index, _ in enumerate(names):
            states = np.load(
                state_path(temp_dir, "valid", checkpoint_index, layer_index),
                mmap_mode="r",
            )
            basis = randomized_left_basis(
                states,
                int(cli.low_rank_rank),
                int(cli.low_rank_oversample),
                int(cli.low_rank_power_iters),
                fit_dev,
                int(cli.subset_seed)
                + 1_000_003 * checkpoint_index
                + 10_007 * layer_index,
                int(cli.low_rank_position_chunk),
            )
            path = low_rank_path(temp_dir, checkpoint_index, layer_index)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, basis)
            del states, basis

        remove_checkpoint_states(temp_dir, checkpoint_index)

    if layer_names_ref is None or architecture_ref is None:
        raise RuntimeError("No checkpoint processed.")

    overlap = pairwise_low_rank_overlap(
        temp_dir,
        len(checkpoints),
        len(layer_names_ref),
        n_positions,
        int(cli.low_rank_rank),
        fit_dev,
        int(cli.low_rank_position_chunk),
    )
    return overlap, layer_names_ref, metadata_list, architecture_ref


# =============================================================================
# Ridge sufficient-statistic fits used by novelty and task overlap
# =============================================================================


def _array_rows(array: np.ndarray) -> np.ndarray:
    return array.reshape(-1, array.shape[-1])


def fit_ridge_map(
    x_array: np.ndarray,
    y_array: np.ndarray,
    *,
    alpha: float,
    device: torch.device,
    dtype_name: str,
    chunk_rows: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit centered multivariate ridge using pooled sample x position rows.

    Objective is mean squared error + alpha * ||A||_F^2, so alpha does not
    scale with the number of pooled rows.
    """
    x = _array_rows(x_array)
    y = _array_rows(y_array)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"X/Y row mismatch: {x.shape} vs {y.shape}")
    if x.shape[1] != y.shape[1]:
        raise ValueError("Novelty regression expects equal source/target widths.")

    dtype = torch_dtype(dtype_name)
    d = int(x.shape[1])
    n = int(x.shape[0])
    sum_x = torch.zeros(d, dtype=dtype, device=device)
    sum_y = torch.zeros(d, dtype=dtype, device=device)
    xtx = torch.zeros((d, d), dtype=dtype, device=device)
    xty = torch.zeros((d, d), dtype=dtype, device=device)
    step = max(1, int(chunk_rows))
    np_dtype = np.float64 if dtype == torch.float64 else np.float32

    for start in range(0, n, step):
        stop = min(n, start + step)
        xb = torch.as_tensor(
            np.array(x[start:stop], dtype=np_dtype, copy=True),
            dtype=dtype,
            device=device,
        )
        yb = torch.as_tensor(
            np.array(y[start:stop], dtype=np_dtype, copy=True),
            dtype=dtype,
            device=device,
        )
        sum_x += xb.sum(dim=0)
        sum_y += yb.sum(dim=0)
        xtx += xb.T @ xb
        xty += xb.T @ yb
        del xb, yb

    mean_x = sum_x / float(n)
    mean_y = sum_y / float(n)
    cov_x = xtx / float(n) - torch.outer(mean_x, mean_x)
    cov_xy = xty / float(n) - torch.outer(mean_x, mean_y)
    cov_x = 0.5 * (cov_x + cov_x.T)
    system = cov_x + float(alpha) * torch.eye(d, dtype=dtype, device=device)
    try:
        weights = torch.linalg.solve(system, cov_xy)
    except RuntimeError:
        weights = torch.linalg.pinv(system) @ cov_xy
    return mean_x, mean_y, weights


def fit_task_probe(
    x_array: np.ndarray,
    targets: torch.Tensor,
    *,
    vocab_size: int,
    alpha: float,
    device: torch.device,
    dtype_name: str,
    chunk_rows: int,
    return_statistics: bool = False,
) -> torch.Tensor:
    """Fit centered ridge from hidden states to one-hot next-token labels.

    The one-hot design is accumulated sparsely; no [N*T, vocab] matrix is built.
    """
    x = _array_rows(x_array)
    y = targets.detach().cpu().numpy().reshape(-1).astype(np.int64, copy=False)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"State/target row mismatch: {x.shape[0]} vs {y.shape[0]}")
    if y.size and (int(y.min()) < 0 or int(y.max()) >= int(vocab_size)):
        raise ValueError(
            f"Task targets outside [0,{vocab_size - 1}]: "
            f"min={int(y.min())}, max={int(y.max())}"
        )

    dtype = torch_dtype(dtype_name)
    np_dtype = np.float64 if dtype == torch.float64 else np.float32
    d = int(x.shape[1])
    n = int(x.shape[0])
    v = int(vocab_size)
    sum_x = torch.zeros(d, dtype=dtype, device=device)
    xtx = torch.zeros((d, d), dtype=dtype, device=device)
    class_sums = torch.zeros((d, v), dtype=dtype, device=device)
    counts = torch.zeros(v, dtype=dtype, device=device)
    step = max(1, int(chunk_rows))

    for start in range(0, n, step):
        stop = min(n, start + step)
        xb = torch.as_tensor(
            np.array(x[start:stop], dtype=np_dtype, copy=True),
            dtype=dtype,
            device=device,
        )
        labels = torch.as_tensor(y[start:stop], dtype=torch.long, device=device)
        sum_x += xb.sum(dim=0)
        xtx += xb.T @ xb
        class_sums.index_add_(1, labels, xb.T)
        counts += torch.bincount(labels, minlength=v).to(dtype)
        del xb, labels

    mean_x = sum_x / float(n)
    cov_x = xtx / float(n) - torch.outer(mean_x, mean_x)
    cov_x = 0.5 * (cov_x + cov_x.T)
    class_freq = counts / float(n)
    cov_xy = class_sums / float(n) - mean_x[:, None] * class_freq[None, :]
    if return_statistics:
        return mean_x, class_freq, cov_x, cov_xy
    system = cov_x + float(alpha) * torch.eye(d, dtype=dtype, device=device)
    try:
        weights = torch.linalg.solve(system, cov_xy)
    except RuntimeError:
        weights = torch.linalg.pinv(system) @ cov_xy
    return weights


# =============================================================================
# CKA temporary helpers for derived representations
# =============================================================================


def save_normalized_gram(path: Path, gram: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gram64 = np.asarray(gram, dtype=np.float64)
    norm = float(np.sqrt(np.sum(gram64 * gram64)))
    if not np.isfinite(norm) or norm <= EPS:
        normalized = np.zeros_like(gram64, dtype=np.float32)
    else:
        normalized = (gram64 / norm).astype(np.float32)
    np.save(path, normalized)


def derived_cka_path(
    temp_dir: Path,
    method: str,
    checkpoint_index: int,
    layer_index: int,
    position_index: int,
) -> Path:
    return base.temporary_path(
        temp_dir,
        f"{method}_valid",
        "gram",
        checkpoint_index,
        layer_index,
        position_index,
    )


def pairwise_derived_gram_cka(
    temp_dir: Path,
    method: str,
    number_checkpoints: int,
    number_layers: int,
    number_positions: int,
) -> np.ndarray:
    dummy_norms = np.ones(
        (number_checkpoints, number_layers, number_positions), dtype=np.float64
    )
    return base.pairwise_cka(
        temp_dir,
        f"{method}_valid",
        "gram",
        dummy_norms,
        number_checkpoints,
        number_layers,
        number_positions,
    )


# =============================================================================
# Novelty overlap
# =============================================================================


def compute_novelty(
    checkpoints: Sequence[Path],
    config: argparse.Namespace,
    init_module: Any,
    train_reference: torch.Tensor,
    valid_reference: torch.Tensor,
    cli: argparse.Namespace,
    temp_dir: Path,
) -> Tuple[np.ndarray, List[str], List[Dict[str, Any]], str, np.ndarray, Dict[str, np.ndarray]]:
    fit_dev = fit_device(cli)
    layer_names_ref: Optional[List[str]] = None
    metadata_list: List[Dict[str, Any]] = []
    architecture_ref: Optional[str] = None
    n_positions = int(valid_reference.shape[1])
    energy_fraction = np.full(
        (len(checkpoints), int(getattr(config, "depth"))), np.nan, dtype=np.float64
    )
    # Final selected checkpoint is evaluated first, but output time order is unchanged.
    reference_index = len(checkpoints) - 1
    metadata_list = [None] * len(checkpoints)
    dynamics = {
        "novelty_dynamics_reference_index": np.asarray(reference_index),
        "novelty_dynamics_axes": np.asarray("time,block,position; map_relative_change: time,block"),
        "novelty_dynamics_definition": np.asarray(
            "f_t(x)=(x-mean_x_t)@A_t+mean_y_t; reference *=last selected checkpoint; "
            "prediction_change=mean||f_t(x_t)-f_*(x_*)||^2; "
            "fixed_input_change=mean||f_t(x_*)-f_*(x_*)||^2; "
            "prediction_cka=position-centered CKA(f_t(x_t),f_*(x_*)); "
            "map_relative_change=||A_t-A_*||_F^2/||A_*||_F^2; "
            "residual_mse=mean||y_t-f_t(x_t)||^2; "
            "target energies=mean||y-mean(y)||^2; all evaluation means over held-out samples; "
            "raw coordinates, no alignment; fixed and own-input changes are not additive."),
    }
    reference_dir = temp_dir / "novelty_predictor_reference"
    reference_dir.mkdir(exist_ok=True)
    for checkpoint_index in [reference_index] + list(range(reference_index)):
        checkpoint = checkpoints[checkpoint_index]
        print(
            f"[INFO] novelty checkpoint {checkpoint_index + 1}/{len(checkpoints)} "
            f"{checkpoint.name}",
            flush=True,
        )
        state_names, metadata, arch = store_state_checkpoint(
            checkpoint,
            checkpoint_index,
            config,
            init_module,
            {"train": train_reference, "valid": valid_reference},
            cli,
            temp_dir,
            include_embedding=True,
            include_final_norm=False,
        )
        expected_names = ["embedding"] + [
            f"block_{index + 1}" for index in range(len(state_names) - 1)
        ]
        if state_names != expected_names:
            raise RuntimeError(
                "Novelty expects embedding followed by block states; got "
                f"{state_names}"
            )
        novelty_names = [
            f"block_novelty_{index + 1}" for index in range(len(state_names) - 1)
        ]
        if layer_names_ref is None:
            layer_names_ref = novelty_names
            architecture_ref = arch
            energy_fraction = np.full(
                (len(checkpoints), len(novelty_names)), np.nan, dtype=np.float64
            )
            for key in ("prediction_change", "fixed_input_change", "prediction_cka",
                        "reference_target_energy", "target_energy", "residual_mse"):
                dynamics["novelty_" + key] = np.full(
                    (len(checkpoints), len(novelty_names), n_positions), np.nan)
            dynamics["novelty_map_relative_change"] = np.full(energy_fraction.shape, np.nan)
        elif novelty_names != layer_names_ref or arch != architecture_ref:
            raise RuntimeError("Novelty representation layout changed between checkpoints.")
        metadata_list[checkpoint_index] = metadata

        for novelty_layer in range(len(novelty_names)):
            x_train = np.load(
                state_path(temp_dir, "train", checkpoint_index, novelty_layer),
                mmap_mode="r",
            )
            y_train = np.load(
                state_path(temp_dir, "train", checkpoint_index, novelty_layer + 1),
                mmap_mode="r",
            )
            mean_x, mean_y, weights = fit_ridge_map(
                x_train,
                y_train,
                alpha=float(cli.novelty_ridge_alpha),
                device=fit_dev,
                dtype_name=str(cli.linear_fit_dtype),
                chunk_rows=int(cli.linear_fit_chunk_rows),
            )
            del x_train, y_train

            x_valid = np.load(
                state_path(temp_dir, "valid", checkpoint_index, novelty_layer),
                mmap_mode="r",
            )
            y_valid = np.load(
                state_path(temp_dir, "valid", checkpoint_index, novelty_layer + 1),
                mmap_mode="r",
            )
            map_path = reference_dir / f"map_{novelty_layer}.npz"
            if checkpoint_index == reference_index:
                np.savez(map_path, weights=weights.detach().cpu().numpy(),
                         mean_x=mean_x.detach().cpu().numpy(), mean_y=mean_y.detach().cpu().numpy())
                np.save(reference_dir / f"input_{novelty_layer}.npy", x_valid)
                np.save(reference_dir / f"target_{novelty_layer}.npy", y_valid)
            with np.load(map_path) as ref:
                rw, rxm, rym = [torch.as_tensor(ref[k], device=fit_dev, dtype=weights.dtype)
                                for k in ("weights", "mean_x", "mean_y")]
            ref_x = np.load(reference_dir / f"input_{novelty_layer}.npy", mmap_mode="r")
            ref_y = np.load(reference_dir / f"target_{novelty_layer}.npy", mmap_mode="r")
            map_den = float(rw.square().sum())
            dynamics["novelty_map_relative_change"][checkpoint_index, novelty_layer] = (
                float((weights-rw).square().sum()) / map_den if map_den > EPS else np.nan)
            residual_sq = 0.0
            target_sq = 0.0
            for position in range(n_positions):
                x = torch.as_tensor(
                    np.array(x_valid[:, position, :], copy=True),
                    dtype=weights.dtype,
                    device=fit_dev,
                )
                y = torch.as_tensor(
                    np.array(y_valid[:, position, :], copy=True),
                    dtype=weights.dtype,
                    device=fit_dev,
                )
                residual = (y - mean_y) - (x - mean_x) @ weights
                # Same held-out sequences, no extra fitting or model forwards.
                xr = torch.as_tensor(np.array(ref_x[:, position]), device=fit_dev, dtype=weights.dtype)
                yr = torch.as_tensor(np.array(ref_y[:, position]), device=fit_dev, dtype=weights.dtype)
                predicted = y - residual
                reference_prediction = (xr-rxm) @ rw + rym
                fixed_prediction = (xr-mean_x) @ weights + mean_y
                idx = (checkpoint_index, novelty_layer, position)
                values = {
                    "prediction_change": (predicted-reference_prediction).square().sum(-1).mean(),
                    "fixed_input_change": (fixed_prediction-reference_prediction).square().sum(-1).mean(),
                    "reference_target_energy": (yr-yr.mean(0)).square().sum(-1).mean(),
                    "target_energy": (y-y.mean(0)).square().sum(-1).mean(),
                    "residual_mse": residual.square().sum(-1).mean(),
                    "prediction_cka": _stage1_cka(predicted.double()[None],
                                                  reference_prediction.double()[None])[0],
                }
                for key, value in values.items():
                    dynamics["novelty_" + key][idx] = float(value)
                del xr, yr, predicted, reference_prediction, fixed_prediction, values
                residual_np = residual.detach().float().cpu().numpy()
                residual_np = residual_np - residual_np.mean(axis=0, keepdims=True)
                gram = residual_np @ residual_np.T
                save_normalized_gram(
                    derived_cka_path(
                        temp_dir,
                        "novelty",
                        checkpoint_index,
                        novelty_layer,
                        position,
                    ),
                    gram,
                )
                target_centered = y - y.mean(dim=0, keepdim=True)
                residual_sq += float(torch.sum(residual * residual).detach().cpu())
                target_sq += float(
                    torch.sum(target_centered * target_centered).detach().cpu()
                )
                del x, y, residual, target_centered

            energy_fraction[checkpoint_index, novelty_layer] = (
                residual_sq / target_sq if target_sq > EPS else np.nan
            )
            del x_valid, y_valid, mean_x, mean_y, weights, ref_x, ref_y, rw, rxm, rym
            cleanup_cuda()

        remove_checkpoint_states(temp_dir, checkpoint_index)

    if layer_names_ref is None or architecture_ref is None:
        raise RuntimeError("No novelty checkpoint processed.")

    overlap = pairwise_derived_gram_cka(
        temp_dir,
        "novelty",
        len(checkpoints),
        len(layer_names_ref),
        n_positions,
    )
    return overlap, layer_names_ref, metadata_list, architecture_ref, energy_fraction, dynamics


# =============================================================================
# Task-relevant overlap
# =============================================================================


def compute_task(
    checkpoints: Sequence[Path],
    config: argparse.Namespace,
    init_module: Any,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    valid_inputs: torch.Tensor,
    cli: argparse.Namespace,
    temp_dir: Path,
) -> Tuple[np.ndarray, List[str], List[Dict[str, Any]], str]:
    fit_dev = fit_device(cli)
    layer_names_ref: Optional[List[str]] = None
    metadata_list: List[Dict[str, Any]] = []
    architecture_ref: Optional[str] = None
    n_positions = int(valid_inputs.shape[1])
    vocab = int(config.vocab_size)

    for checkpoint_index, checkpoint in enumerate(checkpoints):
        print(
            f"[INFO] task checkpoint {checkpoint_index + 1}/{len(checkpoints)} "
            f"{checkpoint.name}",
            flush=True,
        )
        names, metadata, arch = store_state_checkpoint(
            checkpoint,
            checkpoint_index,
            config,
            init_module,
            {"train": train_inputs, "valid": valid_inputs},
            cli,
            temp_dir,
            include_embedding=bool(cli.include_embedding),
            include_final_norm=bool(cli.include_final_norm),
        )
        if layer_names_ref is None:
            layer_names_ref = names
            architecture_ref = arch
        elif names != layer_names_ref or arch != architecture_ref:
            raise RuntimeError("Task representation layout changed between checkpoints.")
        metadata_list.append(metadata)

        for layer_index, _ in enumerate(names):
            x_train = np.load(
                state_path(temp_dir, "train", checkpoint_index, layer_index),
                mmap_mode="r",
            )
            weights = fit_task_probe(
                x_train,
                train_targets,
                vocab_size=vocab,
                alpha=float(cli.task_ridge_alpha),
                device=fit_dev,
                dtype_name=str(cli.linear_fit_dtype),
                chunk_rows=int(cli.linear_fit_chunk_rows),
            )
            del x_train
            # B gives the task-output Gram without materializing H @ W in vocab space.
            task_metric = weights @ weights.T
            task_metric = 0.5 * (task_metric + task_metric.T)

            x_valid = np.load(
                state_path(temp_dir, "valid", checkpoint_index, layer_index),
                mmap_mode="r",
            )
            for position in range(n_positions):
                h = torch.as_tensor(
                    np.array(x_valid[:, position, :], copy=True),
                    dtype=weights.dtype,
                    device=fit_dev,
                )
                h = h - h.mean(dim=0, keepdim=True)
                gram = h @ task_metric @ h.T
                save_normalized_gram(
                    derived_cka_path(
                        temp_dir,
                        "task",
                        checkpoint_index,
                        layer_index,
                        position,
                    ),
                    gram.detach().float().cpu().numpy(),
                )
                del h, gram
            del x_valid, weights, task_metric
            cleanup_cuda()

        remove_checkpoint_states(temp_dir, checkpoint_index)

    if layer_names_ref is None or architecture_ref is None:
        raise RuntimeError("No task checkpoint processed.")

    overlap = pairwise_derived_gram_cka(
        temp_dir,
        "task",
        len(checkpoints),
        len(layer_names_ref),
        n_positions,
    )
    return overlap, layer_names_ref, metadata_list, architecture_ref


# =============================================================================
# Saving new-method outputs
# =============================================================================


def save_new_overlap(
    *,
    output_dir: Path,
    definition: str,
    overlap: np.ndarray,
    layer_names: Sequence[str],
    checkpoints: Sequence[Path],
    checkpoint_metadata: Sequence[Mapping[str, Any]],
    selected_indices: np.ndarray,
    config: argparse.Namespace,
    run_dir: Path,
    model_architecture: str,
    reference_metadata: Mapping[str, Any],
    cli: argparse.Namespace,
    novelty_energy_fraction: Optional[np.ndarray] = None,
    novelty_dynamics: Optional[Mapping[str, np.ndarray]] = None,
) -> Path:
    arrays = base.common_arrays(
        checkpoints,
        checkpoint_metadata,
        config,
        run_dir,
        model_architecture,
    )
    arrays.update(
        {
            "overlap_definition": np.asarray(definition),
            "valid_overlap_by_position": overlap,
            "test_overlap_by_position": overlap,
            "valid_overlap_position_mean": np.nanmean(overlap, axis=0),
            "test_overlap_position_mean": np.nanmean(overlap, axis=0),
            "layer_names": np.asarray(layer_names, dtype=object),
            "input_token_positions_1based": np.arange(
                1, overlap.shape[0] + 1, dtype=np.int64
            ),
            "target_token_positions_1based": np.arange(
                2, overlap.shape[0] + 2, dtype=np.int64
            ),
            "selected_checkpoint_indices": np.asarray(selected_indices, dtype=np.int64),
            "reference_source": np.asarray(reference_metadata["reference_source"]),
            "reference_metadata_json": np.asarray(
                json.dumps(base.jsonify(reference_metadata), sort_keys=True)
            ),
            "fit_split": np.asarray("train" if definition in {"novelty", "task"} else "none"),
            "evaluation_split": np.asarray("valid"),
        }
    )

    if definition == "low_rank":
        arrays.update(
            {
                "low_rank_overlap_by_position": overlap,
                "low_rank_overlap_position_mean": np.nanmean(overlap, axis=0),
                "low_rank_rank": np.asarray(int(cli.low_rank_rank), dtype=np.int64),
                "low_rank_oversample": np.asarray(
                    int(cli.low_rank_oversample), dtype=np.int64
                ),
                "low_rank_power_iters": np.asarray(
                    int(cli.low_rank_power_iters), dtype=np.int64
                ),
            }
        )
        filename = "low_rank_overlaps.npz"
    elif definition == "novelty":
        arrays.update(
            {
                "valid_cka_by_position": overlap,
                "test_cka_by_position": overlap,
                "valid_cka_position_mean": np.nanmean(overlap, axis=0),
                "test_cka_position_mean": np.nanmean(overlap, axis=0),
                "novelty_ridge_alpha": np.asarray(float(cli.novelty_ridge_alpha)),
            }
        )
        if novelty_energy_fraction is not None:
            arrays["novelty_valid_residual_energy_fraction"] = novelty_energy_fraction
        if novelty_dynamics is not None:
            arrays.update(novelty_dynamics)
        filename = "novelty_overlaps.npz"
    elif definition == "task":
        arrays.update(
            {
                "valid_cka_by_position": overlap,
                "test_cka_by_position": overlap,
                "valid_cka_position_mean": np.nanmean(overlap, axis=0),
                "test_cka_position_mean": np.nanmean(overlap, axis=0),
                "task_ridge_alpha": np.asarray(float(cli.task_ridge_alpha)),
                "task_target": np.asarray("next_token_one_hot"),
            }
        )
        filename = "task_overlaps.npz"
    else:
        raise ValueError(definition)

    output_path = output_dir / filename
    np.savez_compressed(output_path, **arrays)

    metadata_payload = {
        "run_dir": run_dir,
        "output_path": output_path,
        "overlap_definition": definition,
        "architecture": model_architecture,
        "num_checkpoints": len(checkpoints),
        "selected_checkpoint_indices": selected_indices,
        "layer_names": list(layer_names),
        "num_positions": int(overlap.shape[0]),
        "fit_split": "train" if definition in {"novelty", "task"} else None,
        "evaluation_split": "valid",
        "reference": reference_metadata,
        "low_rank_rank": int(cli.low_rank_rank) if definition == "low_rank" else None,
        "low_rank_oversample": int(cli.low_rank_oversample)
        if definition == "low_rank"
        else None,
        "low_rank_power_iters": int(cli.low_rank_power_iters)
        if definition == "low_rank"
        else None,
        "novelty_ridge_alpha": float(cli.novelty_ridge_alpha)
        if definition == "novelty"
        else None,
        "task_ridge_alpha": float(cli.task_ridge_alpha)
        if definition == "task"
        else None,
        "linear_fit_device": str(fit_device(cli))
        if definition in {"low_rank", "novelty", "task"}
        else None,
        "linear_fit_dtype": str(cli.linear_fit_dtype),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(base.jsonify(metadata_payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"[INFO] saved {output_path}", flush=True)
    return output_path


# =============================================================================
# CLI
# =============================================================================


# =============================================================================
# Stage 1: matched, inexpensive checkpoint diagnostics (one reference, no T^2)
# =============================================================================


def _stage1_cka(x, y):
    """Position-wise CKA; zero-variance representations are undefined, not 1."""
    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)
    xx, yy = x @ x.transpose(1, 2), y @ y.transpose(1, 2)
    den = torch.linalg.vector_norm(xx, dim=(1, 2)) * torch.linalg.vector_norm(yy, dim=(1, 2))
    value = (xx * yy).sum(dim=(1, 2)) / den.clamp_min(1e-300)
    return torch.where(den > 1e-24, value.clamp(0, 1), torch.full_like(value, float('nan')))


def _stage1_spectrum(h, ranks, rtol):
    """Exact small SVD on [position,sample,feature], rank-aware subspaces."""
    h = h - h.mean(dim=1, keepdim=True)
    u, s, _ = torch.linalg.svd(h, full_matrices=False)
    energy = s.square()
    total = energy.sum(-1)
    supported = (s > rtol * s[:, :1]).sum(-1)
    supported = torch.where(total > 1e-24, supported, torch.zeros_like(supported))
    cumulative = energy.cumsum(-1) / total[:, None].clamp_min(1e-300)
    captured, gap, relative = [], [], []
    for r in ranks:
        if r <= s.shape[-1]:
            captured.append(cumulative[:, r-1])
            relative.append(s[:, r-1] / s[:, 0].clamp_min(1e-300))
            next_s = s[:, r] if r < s.shape[-1] else torch.zeros_like(s[:, 0])
            gap.append((s[:, r-1] - next_s) / s[:, 0].clamp_min(1e-300))
        else:
            nan = torch.full_like(total, float('nan'))
            captured.append(nan); gap.append(nan); relative.append(nan)
    return u, s, supported, torch.stack(captured, -1), torch.stack(gap, -1), torch.stack(relative, -1)


def _stage1_loss(logits, targets, masks=None):
    """All quantities on exactly the same held-out tokens, in float64."""
    z = torch.as_tensor(np.array(logits, copy=True), dtype=torch.float64)
    y = torch.as_tensor(targets, dtype=torch.long)
    lp = z.log_softmax(-1)
    true_lp = lp.gather(-1, y[..., None]).squeeze(-1)
    result = {'ntp_loss': (-true_lp).mean(0).numpy()}
    if masks is not None:
        a = torch.as_tensor(masks, dtype=torch.bool)
        # [sample,position,level]; nested sets telescope exactly.
        log_mass = torch.logsumexp(lp[:, :, None, :].masked_fill(~a, -torch.inf), -1)
        prev = torch.cat((torch.zeros_like(log_mass[..., :1]), log_mass[..., :-1]), -1)
        peel = prev - log_mass
        within = log_mass[..., -1] - true_lp
        if not torch.isfinite(peel).all() or (peel < -1e-10).any():
            raise ValueError('Invalid or non-nested RHM compatibility masks.')
        error = ((-true_lp) - peel.sum(-1) - within).abs().max().item()
        if error > 1e-9:
            raise RuntimeError(f'Loss decomposition failed: {error}')
        result.update(peeled_contribution=peel.mean(0).numpy(),
                      within_loss=within.mean(0).numpy())
    return result


def _stage1_masks(config, run_dir, valid_inputs, valid_targets):
    """Reuse existing compatibility code and saved rules, once for all checkpoints."""
    from rhm_margins import CompatibilityComputer, RHMParamsLite
    params = RHMParamsLite(**{key: int(getattr(config, key)) for key in
                             ('num_features', 'num_classes', 'num_synonyms', 'tuple_size', 'num_layers')})
    rules = base.normalize_rules(base.safe_load(base.find_rules(run_dir)))
    full = torch.cat((valid_inputs[:, :1], valid_targets), dim=1).numpy()
    computer = CompatibilityComputer(params, rules)
    a, b = computer.masks_for_sequences(full)
    computer.clear_caches()
    previous = np.concatenate((np.ones_like(a[:, :, :1]), a[:, :, :-1]), axis=2)
    if np.any(a & ~previous):
        raise ValueError('RHM sets are not nested.')
    truth = np.take_along_axis(a, valid_targets.numpy()[:, :, None, None], axis=-1)
    if not truth.all():
        raise ValueError('A true token is absent from a compatibility set.')
    return a, b.any(-1).mean(0)


def compute_stage1(checkpoints, config, init_module, cli, repo_dir, source_dir,
                   run_dir, output_dir, temp_dir, selected_indices):
    """One state extraction per checkpoint; pooled train fits, held-out diagnostics.

    Diagnostic arrays use [time,layer,position,(alpha or rank)]. The fixed
    reference is the last selected checkpoint. No hidden states are required
    to have been saved during training; they are reconstructed from checkpoints.
    """
    if not base.is_rhm(config, run_dir):
        raise ValueError('Stage 1 currently supports RHM NTP runs only.')
    # Do not silently generate a new dataset for an old run.
    if cli.reference_source == 'auto':
        cli.reference_source = 'saved'
    train_x, train_y, valid_x, valid_y, refmeta = load_ntp_task_references(
        config, cli, repo_dir, source_dir, run_dir)
    if len(train_x) < 2 or len(valid_x) < 2:
        raise ValueError('Stage 1 needs at least two sequences per split.')
    alphas = np.asarray([float(x) for x in cli.stage1_alphas.split(',')])
    ranks = np.asarray([int(x) for x in cli.stage1_ranks.split(',')])
    if not np.isfinite(alphas).all() or np.any(alphas <= 0) or np.any(ranks <= 0):
        raise ValueError('Stage-1 alphas must be finite/positive, and ranks positive.')
    if not 0 < cli.stage1_rank_rtol < 1:
        raise ValueError('stage1_rank_rtol must lie in (0,1).')
    fit_dev = fit_device(cli)
    dtype = torch.float64  # diagnostics and rank checks need stable arithmetic
    nt, np_, vocab = len(checkpoints), valid_x.shape[1], int(config.vocab_size)
    arrays, metadata_list = {}, [None] * nt
    masks = None
    if cli.stage1_loss:
        print('[STAGE1] Building compatibility masks once on the fixed validation subset.', flush=True)
        masks, valid_fraction = _stage1_masks(config, run_dir, valid_x, valid_y)
        arrays['rhm_valid_fraction_by_position'] = valid_fraction
    labels = valid_y.numpy().T
    # Pooled train class frequencies; also a position-dependent baseline.
    freq = torch.bincount(train_y.reshape(-1), minlength=vocab).double().numpy()
    freq /= freq.sum()
    pos_freq = np.stack([np.bincount(train_y[:, p].numpy(), minlength=vocab) / len(train_y)
                         for p in range(np_)])
    arrays['probe_baseline_error'] = np.array([
        1 + np.sum(freq**2) - 2*np.mean(freq[labels[p]]) for p in range(np_)])
    arrays['probe_position_baseline_error'] = np.array([
        1 + np.sum(pos_freq[p]**2) - 2*np.mean(pos_freq[p, labels[p]]) for p in range(np_)])
    reference_index = nt - 1
    order = [reference_index] + list(range(nt - 1))
    names_ref = None
    cache_dir = temp_dir / 'stage1_reference'
    cache_dir.mkdir(parents=True, exist_ok=True)
    for ti in order:
        print(f'[STAGE1] checkpoint {ti+1}/{nt}: {checkpoints[ti].name}', flush=True)
        names, meta, arch = store_state_checkpoint(
            checkpoints[ti], ti, config, init_module, {'train': train_x, 'valid': valid_x},
            cli, temp_dir, include_embedding=True, include_final_norm=True)
        metadata_list[ti] = meta
        weight_path = temp_dir / f'stage1_weights_{ti}.npz'
        with np.load(weight_path, allow_pickle=False) as weight_data:
            groups = sorted(weight_data.files)
            if 'weight_group_names' not in arrays:
                arrays['weight_group_names'] = np.asarray(groups)
                for key in ('weight_cosine', 'weight_norm', 'weight_relative_displacement'):
                    arrays[key] = np.full((nt, len(groups)), np.nan)
            for gi, group in enumerate(groups):
                for mi, key in enumerate(('weight_cosine', 'weight_norm', 'weight_relative_displacement')):
                    arrays[key][ti, gi] = weight_data[group][mi]
        weight_path.unlink()
        if names_ref is None:
            names_ref = names
            shape = (nt, len(names), np_)
            for key in ('state_energy', 'update_energy', 'novelty_energy',
                        'novelty_uncentered_energy', 'state_cka', 'update_cka',
                        'novelty_cka', 'numerical_rank', 'state_rms'):
                arrays[key] = np.full(shape, np.nan)
            for key in ('probe_error', 'probe_energy', 'probe_output_energy',
                        'probe_cka', 'probe_displacement', 'probe_centered_displacement'):
                arrays[key] = np.full(shape + (len(alphas),), np.nan)
            for key in ('low_rank_cka', 'rank_energy_fraction', 'rank_gap_relative', 'rank_singular_relative'):
                arrays[key] = np.full(shape + (len(ranks),), np.nan)
            arrays['train_feature_variance'] = np.full(shape[:2], np.nan)
        elif names != names_ref:
            raise RuntimeError('Representation names changed across checkpoints.')
        prev_train = prev_valid = None
        for ki, name in enumerate(names):
            train = np.load(state_path(temp_dir, 'train', ti, ki), mmap_mode='r')
            valid = np.load(state_path(temp_dir, 'valid', ti, ki), mmap_mode='r')
            h = torch.as_tensor(np.array(valid.transpose(1, 0, 2)), device=fit_dev, dtype=dtype)
            hc = h - h.mean(1, keepdim=True)
            energy = hc.square().sum(-1).mean(-1)
            arrays['state_energy'][ti, ki] = energy.cpu().numpy()
            arrays['state_rms'][ti, ki] = h.square().mean((1, 2)).sqrt().cpu().numpy()
            u, s, supported, captured, gap, relative = _stage1_spectrum(h, ranks, cli.stage1_rank_rtol)
            if 'singular_values' not in arrays:
                arrays['singular_values'] = np.full((nt, len(names), np_, s.shape[-1]), np.nan)
            arrays['singular_values'][ti, ki] = s.cpu().numpy()
            arrays['numerical_rank'][ti, ki] = supported.cpu().numpy()
            arrays['rank_energy_fraction'][ti, ki] = captured.cpu().numpy()
            arrays['rank_gap_relative'][ti, ki] = gap.cpu().numpy()
            arrays['rank_singular_relative'][ti, ki] = relative.cpu().numpy()
            # Existing sufficient-statistics accumulator, reused once for all alphas.
            mean_x, class_freq, cov_x, cov_xy = fit_task_probe(
                train, train_y, vocab_size=vocab, alpha=0, device=fit_dev,
                dtype_name='float64', chunk_rows=cli.linear_fit_chunk_rows,
                return_statistics=True)
            values, vectors = torch.linalg.eigh(cov_x)
            values = values.clamp_min(0)
            projected = vectors.T @ cov_xy
            arrays['train_feature_variance'][ti, ki] = float(values.mean().cpu())
            predictions = []
            for ai, alpha in enumerate(alphas):
                w = vectors @ (projected / (values[:, None] + float(alpha)))
                z = (h - mean_x) @ w + class_freq
                predictions.append(z)
                centered = z - z.mean(1, keepdim=True)
                lab = torch.as_tensor(labels, device=fit_dev, dtype=torch.long)
                err = z.square().sum(-1) + 1 - 2*z.gather(-1, lab[..., None]).squeeze(-1)
                arrays['probe_error'][ti, ki, :, ai] = err.mean(1).cpu().numpy()
                arrays['probe_energy'][ti, ki, :, ai] = centered.square().sum(-1).mean(1).cpu().numpy()
                arrays['probe_output_energy'][ti, ki, :, ai] = z.square().sum(-1).mean(1).cpu().numpy()
            z_all = torch.stack(predictions)  # [alpha,position,sample,vocab]
            residual = update = None
            if name.startswith('block_'):
                mx, my, w = fit_ridge_map(prev_train, train, alpha=cli.novelty_ridge_alpha,
                    device=fit_dev, dtype_name='float64', chunk_rows=cli.linear_fit_chunk_rows)
                residual = (h-my) - (prev_valid-mx) @ w
                update = h-prev_valid
                rc = residual-residual.mean(1, keepdim=True)
                uc = update-update.mean(1, keepdim=True)
                arrays['novelty_energy'][ti, ki] = rc.square().sum(-1).mean(1).cpu().numpy()
                arrays['novelty_uncentered_energy'][ti, ki] = residual.square().sum(-1).mean(1).cpu().numpy()
                arrays['update_energy'][ti, ki] = uc.square().sum(-1).mean(1).cpu().numpy()
            cache_file = cache_dir / f'layer_{ki}.npz'
            if ti == reference_index:
                # Only the reference's states/probe predictions/subspaces are retained.
                payload = dict(h=h.cpu().numpy(), u=u[:, :, :max(ranks)].cpu().numpy(),
                               supported=supported.cpu().numpy(), z=z_all.cpu().numpy(),
                               singular_values=s.cpu().numpy())
                if residual is not None:
                    payload.update(residual=residual.cpu().numpy(), update=update.cpu().numpy())
                np.savez(cache_file, **payload)
            with np.load(cache_file, allow_pickle=False) as ref:
                rh = torch.as_tensor(ref['h'], device=fit_dev, dtype=dtype)
                arrays['state_cka'][ti, ki] = _stage1_cka(h, rh).cpu().numpy()
                for ai in range(len(alphas)):
                    rz = torch.as_tensor(ref['z'][ai], device=fit_dev, dtype=dtype)
                    z = z_all[ai]
                    arrays['probe_cka'][ti, ki, :, ai] = _stage1_cka(z, rz).cpu().numpy()
                    arrays['probe_displacement'][ti, ki, :, ai] = (z-rz).square().sum(-1).mean(1).cpu().numpy()
                    zc, rzc = z-z.mean(1, keepdim=True), rz-rz.mean(1, keepdim=True)
                    arrays['probe_centered_displacement'][ti, ki, :, ai] = (zc-rzc).square().sum(-1).mean(1).cpu().numpy()
                for ri, r in enumerate(ranks):
                    if r <= u.shape[-1] and r <= ref['u'].shape[-1]:
                        ru = torch.as_tensor(ref['u'][:, :, :r], device=fit_dev, dtype=dtype)
                        q = (u[:, :, :r].transpose(1, 2) @ ru).square().sum((1, 2)) / int(r)
                        good = (supported.cpu().numpy() >= r) & (ref['supported'] >= r)
                        arrays['low_rank_cka'][ti, ki, :, ri] = np.where(good, q.cpu().numpy().clip(0, 1), np.nan)
                if residual is not None:
                    rr = torch.as_tensor(ref['residual'], device=fit_dev, dtype=dtype)
                    ur = torch.as_tensor(ref['update'], device=fit_dev, dtype=dtype)
                    arrays['novelty_cka'][ti, ki] = _stage1_cka(residual, rr).cpu().numpy()
                    arrays['update_cka'][ti, ki] = _stage1_cka(update, ur).cpu().numpy()
            prev_train, prev_valid = train, h
        logits = np.load(temp_dir / f'stage1_logits_{ti}.npy', mmap_mode='r')
        losses = _stage1_loss(logits, valid_y, masks)
        for key, value in losses.items():
            if key not in arrays:
                arrays[key] = np.full((nt,) + value.shape, np.nan)
            arrays[key][ti] = value
        del logits
        (temp_dir / f'stage1_logits_{ti}.npy').unlink()
        del train, valid, prev_train, prev_valid
        remove_checkpoint_states(temp_dir, ti)
        cleanup_cuda()
        # A small partial output survives interruptions; completed indices are explicit.
        arrays['completed_indices'] = np.asarray(sorted(i for i,m in enumerate(metadata_list) if m is not None))
        np.savez_compressed(output_dir / 'stage1_partial.npz', **arrays)
    steps = [int(m.get('step') if m.get('step') is not None else m['step_from_name']) for m in metadata_list]
    arrays.update(selected_steps=np.asarray(steps), layer_names=np.asarray(names_ref),
                  target_token_positions_1based=np.arange(2, np_+2),
                  alphas=alphas, ranks=ranks, reference_index=np.asarray(reference_index),
                  reference_step=np.asarray(steps[reference_index]),
                  rank_rtol=np.asarray(cli.stage1_rank_rtol),
                  selected_checkpoint_indices=np.asarray(selected_indices),
                  schema_version=np.asarray(1))
    metadata = dict(reference=refmeta, config=vars(config), checkpoints=[str(p) for p in checkpoints],
                    reference_step=steps[reference_index],
                    description='Train-pooled affine ridge; position-wise held-out diagnostics; fixed final reference.',
                    ridge_objective='mean over rows of squared vector error + alpha * squared Frobenius norm',
                    rank_policy='Exact SVD; undefined overlap if either numerical rank is below requested rank.',
                    alphas=alphas, ranks=ranks, novelty_alpha=cli.novelty_ridge_alpha,
                    rhm_loss_computed=bool(cli.stage1_loss),
                    axes='time,layer,target_position; optional last axis alpha or rank',
                    probe_outputs='Affine least-squares scores, not probabilities; do not interpret their MSE as NTP loss.')
    arrays['metadata_json'] = np.asarray(json.dumps(base.jsonify(metadata), sort_keys=True))
    path = output_dir / 'stage1_diagnostics.npz'
    np.savez_compressed(path, **arrays)
    (output_dir / 'stage1_partial.npz').unlink(missing_ok=True)
    (output_dir / 'stage1_metadata.json').write_text(json.dumps(base.jsonify(metadata), indent=2))
    print(f'[STAGE1] Saved {path}', flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # Paths / run selection.
    parser.add_argument("--repo_dir", type=Path, default=Path.cwd())
    parser.add_argument("--source_dir", type=Path, default=None)
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument(
        "--collected_results_root", type=Path, default=Path("collected_results")
    )
    parser.add_argument("--run_dir", type=Path, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--output_name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")

    # Unified definition selector.
    parser.add_argument(
        "--overlap_definition",
        choices=ALL_DEFINITIONS,
        default="cka",
    )

    # Shared checkpoint/reference settings.
    parser.add_argument("--reference_source", choices=("auto", "saved", "generate"), default="auto")
    parser.add_argument("--subset_train_size", type=int, default=512)
    parser.add_argument("--subset_valid_size", type=int, default=512)
    parser.add_argument("--subset_seed", type=int, default=12345)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_checkpoints", type=int, default=0)
    parser.add_argument("--max_step", type=int, default=None)
    parser.add_argument("--temp_root", type=Path, default=None)
    parser.add_argument("--keep_temp", action=argparse.BooleanOptionalAction, default=False)

    # State CKA / low-rank / task state extraction.
    parser.add_argument("--cka_metric", choices=("gram", "feature"), default="gram")
    parser.add_argument("--include_embedding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include_final_norm", action=argparse.BooleanOptionalAction, default=True)

    # Low-rank method.
    parser.add_argument("--low_rank_rank", type=int, default=8)
    parser.add_argument("--low_rank_oversample", type=int, default=8)
    parser.add_argument("--low_rank_power_iters", type=int, default=1)
    parser.add_argument("--low_rank_position_chunk", type=int, default=8)

    # Novelty / task ridge fits.  Alphas refer to mean-MSE + alpha*||W||^2.
    parser.add_argument("--novelty_ridge_alpha", type=float, default=1e-4)
    parser.add_argument("--task_ridge_alpha", type=float, default=1e-4)
    parser.add_argument("--linear_fit_device", type=str, default="auto")
    parser.add_argument("--linear_fit_dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--linear_fit_chunk_rows", type=int, default=65536)

    parser.add_argument("--stage1_alphas", default="1e-5,1e-4,1e-3",
                        help="Three fixed ridge values; no test-set alpha selection.")
    parser.add_argument("--stage1_ranks", default="8,16")
    parser.add_argument("--stage1_rank_rtol", type=float, default=1e-5)
    parser.add_argument("--stage1_loss", action=argparse.BooleanOptionalAction, default=True,
                        help="Compute exact RHM loss decomposition on the same subset; --no-stage1_loss skips masks.")
    return parser.parse_args()


# =============================================================================
# Main dispatch
# =============================================================================


def main() -> None:
    cli = parse_args()
    torch.set_default_dtype(torch.float32)

    if cli.subset_train_size <= 0 or cli.subset_valid_size <= 0:
        raise ValueError("subset sizes must be positive")
    if cli.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if cli.novelty_ridge_alpha < 0 or cli.task_ridge_alpha < 0:
        raise ValueError("ridge alphas must be non-negative")
    if cli.low_rank_rank <= 0:
        raise ValueError("low_rank_rank must be positive")

    repo_dir = cli.repo_dir.expanduser().resolve()
    source_dir = base.resolve_source_dir(repo_dir, cli.source_dir)
    data_root = base.resolve_relative(cli.data_root, repo_dir)
    collected_root = base.resolve_relative(cli.collected_results_root, repo_dir)
    collected_root.mkdir(parents=True, exist_ok=True)

    if str(cli.device).startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; using CPU", flush=True)
        cli.device = "cpu"

    run_dir = base.resolve_run_dir(
        repo_dir,
        data_root,
        cli.run_dir,
        cli.run_name,
        cli.run_id,
    )
    config = base.load_config(run_dir, cli.device)
    require_transformer(config)
    init_module = base.import_init(source_dir)
    checkpoints, selected_indices, all_checkpoints = selected_checkpoints(
        run_dir, cli.max_step, cli.num_checkpoints
    )

    output_name = cli.output_name or default_output_name(
        run_dir, cli.overlap_definition
    )
    output_dir = collected_root / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] definition={cli.overlap_definition}", flush=True)
    print(f"[INFO] run_dir={run_dir}", flush=True)
    print(f"[INFO] source_dir={source_dir}", flush=True)
    print(
        f"[INFO] checkpoints={len(checkpoints)}/{len(all_checkpoints)} "
        f"indices={selected_indices.tolist()}",
        flush=True,
    )
    print(f"[INFO] output_dir={output_dir}", flush=True)

    if cli.overlap_definition == "stage1":
        import tempfile
        root = output_dir if cli.temp_root is None else base.resolve_relative(cli.temp_root, repo_dir)
        root.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix="_stage1_", dir=root))
        try:
            compute_stage1(checkpoints, config, init_module, cli, repo_dir, source_dir,
                           run_dir, output_dir, temp_dir, selected_indices)
        finally:
            if cli.keep_temp:
                print(f"[STAGE1] Temporary files kept: {temp_dir}", flush=True)
            else:
                shutil.rmtree(temp_dir, ignore_errors=True)
        return

    # ------------------------------------------------------------------
    # Existing definitions: reuse the mature implementation verbatim.
    # ------------------------------------------------------------------
    if cli.overlap_definition == "weights":
        result, metadata, arch = base.compute_weight_overlaps(
            checkpoints, config, init_module
        )
        if arch != "transformer":
            raise ValueError("weights mode is restricted to Transformer NTP runs.")
        base.save_weights(
            output_dir,
            result,
            checkpoints,
            metadata,
            config,
            run_dir,
            arch,
        )
        return

    if cli.overlap_definition in {"cka", "block_update_cka"}:
        train_reference, valid_reference, reference_metadata = base.load_references(
            config, cli, repo_dir, source_dir, run_dir
        )
        cli.metric = str(cli.cka_metric)
        cli.representation_mode = (
            "state" if cli.overlap_definition == "cka" else "block_update"
        )
        if cli.representation_mode == "block_update":
            # These flags are ignored by the base implementation in update mode,
            # but setting them false keeps metadata unambiguous.
            cli.include_embedding = False
            cli.include_final_norm = False

        temp_dir = resolved_temp_dir(
            repo_dir, output_dir, output_name, cli.temp_root
        )
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True)
        started = time.time()
        try:
            (
                train_cka,
                valid_cka,
                layer_names,
                checkpoint_metadata,
                self_norms,
                arch,
            ) = base.compute_representation_overlaps(
                checkpoints,
                config,
                init_module,
                train_reference,
                valid_reference,
                cli,
                temp_dir,
            )
            if arch != "transformer":
                raise ValueError("CKA modes are restricted to Transformer NTP runs.")
            base.save_representations(
                output_dir,
                train_cka,
                valid_cka,
                layer_names,
                self_norms,
                reference_metadata,
                checkpoints,
                checkpoint_metadata,
                config,
                run_dir,
                arch,
                cli,
            )
            print(
                f"[INFO] total {cli.overlap_definition} time="
                f"{time.time() - started:.2f}s",
                flush=True,
            )
        finally:
            if cli.keep_temp:
                print(f"[INFO] temporary files kept in {temp_dir}", flush=True)
            else:
                shutil.rmtree(temp_dir, ignore_errors=True)
        return

    # ------------------------------------------------------------------
    # New finite-rank / novelty / task definitions.
    # ------------------------------------------------------------------
    temp_dir = resolved_temp_dir(repo_dir, output_dir, output_name, cli.temp_root)
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)
    started = time.time()

    try:
        if cli.overlap_definition == "task":
            (
                train_inputs,
                train_targets,
                valid_inputs,
                _valid_targets,
                reference_metadata,
            ) = load_ntp_task_references(
                config, cli, repo_dir, source_dir, run_dir
            )
            overlap, layer_names, metadata, arch = compute_task(
                checkpoints,
                config,
                init_module,
                train_inputs,
                train_targets,
                valid_inputs,
                cli,
                temp_dir,
            )
            save_new_overlap(
                output_dir=output_dir,
                definition="task",
                overlap=overlap,
                layer_names=layer_names,
                checkpoints=checkpoints,
                checkpoint_metadata=metadata,
                selected_indices=selected_indices,
                config=config,
                run_dir=run_dir,
                model_architecture=arch,
                reference_metadata=reference_metadata,
                cli=cli,
            )

        else:
            train_reference, valid_reference, reference_metadata = base.load_references(
                config, cli, repo_dir, source_dir, run_dir
            )

            if cli.overlap_definition == "low_rank":
                overlap, layer_names, metadata, arch = compute_low_rank(
                    checkpoints,
                    config,
                    init_module,
                    valid_reference,
                    cli,
                    temp_dir,
                )
                save_new_overlap(
                    output_dir=output_dir,
                    definition="low_rank",
                    overlap=overlap,
                    layer_names=layer_names,
                    checkpoints=checkpoints,
                    checkpoint_metadata=metadata,
                    selected_indices=selected_indices,
                    config=config,
                    run_dir=run_dir,
                    model_architecture=arch,
                    reference_metadata=reference_metadata,
                    cli=cli,
                )

            elif cli.overlap_definition == "novelty":
                (
                    overlap,
                    layer_names,
                    metadata,
                    arch,
                    novelty_energy,
                    novelty_dynamics,
                ) = compute_novelty(
                    checkpoints,
                    config,
                    init_module,
                    train_reference,
                    valid_reference,
                    cli,
                    temp_dir,
                )
                reference_metadata = dict(reference_metadata)
                reference_metadata.update(
                    {
                        "fit_split": "train",
                        "evaluation_split": "valid",
                        "map_rule": "one A_l(t) per checkpoint/layer, pooled over train positions",
                    }
                )
                save_new_overlap(
                    output_dir=output_dir,
                    definition="novelty",
                    overlap=overlap,
                    layer_names=layer_names,
                    checkpoints=checkpoints,
                    checkpoint_metadata=metadata,
                    selected_indices=selected_indices,
                    config=config,
                    run_dir=run_dir,
                    model_architecture=arch,
                    reference_metadata=reference_metadata,
                    cli=cli,
                    novelty_energy_fraction=novelty_energy,
                    novelty_dynamics=novelty_dynamics,
                )
            else:
                raise RuntimeError(f"Unhandled definition {cli.overlap_definition}")

        print(
            f"[INFO] total {cli.overlap_definition} time={time.time() - started:.2f}s",
            flush=True,
        )
    finally:
        if cli.keep_temp:
            print(f"[INFO] temporary files kept in {temp_dir}", flush=True)
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
