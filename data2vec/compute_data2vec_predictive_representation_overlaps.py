#!/usr/bin/env python3
"""Compute King-style predictive representation overlaps for data2vec/RHM.

The script is the data2vec counterpart of ``compute_slm_representation_overlaps.py``.
For every selected source checkpoint t and representation layer l, it fits a
cross-validated ridge map

    H_source(t, l, position) -> H_target(t', l', position)

on one fixed set of RHM sequences.  The score is the un-clipped normalized
explained variance on held-out folds.  By default l'=l, t' is the final saved
checkpoint, and both source and target representations come from the student
encoder evaluated on unmasked inputs.

Expected checkpoint layout
--------------------------

<data2vec_dir>/<run_name>/
    data2vec_model_step1_L4_s2_v16_m5_P8192_nH32_nE2048_mp0.15_ema0.99_online.pt
    data2vec_model_step2_...pt
    ...

The checkpoint files are plain ``model.state_dict()`` dictionaries, matching
``train.py`` in the data2vec folder.  Architecture dimensions are inferred from
the state dict and the filename.  RHM parameters and the grammar seed are also
inferred from the filename/run-directory name, with explicit CLI overrides.

Outputs
-------

<output_root>/<output_name>/predictive_overlaps.npz
<output_root>/<output_name>/metadata.json

The main tensor has shape

    predictive_q_by_position[position, source_layer, target_layer, source_time].

Unlike causal next-token prediction, data2vec predicts the representation at the
same masked position.  Therefore source and target token-position indices are
identical and run from 1 to s**L.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from numpy.lib.format import open_memmap
from torch.utils.data import DataLoader, TensorDataset


EPS64 = 1e-12


# =============================================================================
# Generic helpers
# =============================================================================


def safe_torch_load(path: Path, map_location: str | torch.device = "cpu") -> Any:
    """Load a PyTorch file across old and new PyTorch versions."""

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def jsonify(value: Any) -> Any:
    """Convert common scientific-Python objects to JSON-compatible values."""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonify(item) for item in value]
    if hasattr(value, "__dict__"):
        return jsonify(vars(value))
    return repr(value)


def resolve_path(path: Path, base: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value {value!r}")


# =============================================================================
# Checkpoint discovery and configuration inference
# =============================================================================


_STEP_RE = re.compile(r"data2vec_model_step(?P<step>\d+)_.*\.pt$")
_FLOAT_RE = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_SUFFIX_RE = re.compile(
    r"L(?P<rhm_layers>\d+)"
    r"_s(?P<tuple_size>\d+)"
    r"_v(?P<vocab_size>\d+)"
    r"_m(?P<num_synonyms>\d+)"
    r"_P(?P<train_size>\d+)"
    r"_nH(?P<n_heads>\d+)"
    r"_nE(?P<d_model>\d+)"
    rf"_mp(?P<mask_prob>{_FLOAT_RE})"
    rf"_ema(?P<ema_decay>{_FLOAT_RE})"
    r"(?P<online>_online)?"
)
_SEED_RE = re.compile(r"(?:^|_)seed(?P<seed>\d+)(?:_|$)")


def checkpoint_step(path: Path) -> Optional[int]:
    match = _STEP_RE.search(path.name)
    return int(match.group("step")) if match else None


def find_checkpoints(run_dir: Path, max_step: Optional[int]) -> List[Path]:
    found: List[Tuple[int, Path]] = []
    for path in run_dir.glob("data2vec_model_step*.pt"):
        step = checkpoint_step(path)
        if step is None:
            continue
        if max_step is None or step <= int(max_step):
            found.append((step, path.resolve()))
    found.sort(key=lambda item: (item[0], item[1].name))
    if not found:
        raise FileNotFoundError(
            f"No data2vec_model_step*.pt checkpoints found in {run_dir}"
        )
    return [path for _, path in found]


def extract_state_dict(payload: Any) -> Dict[str, torch.Tensor]:
    """Extract and normalize a model state dict."""

    state: Optional[Mapping[str, Any]] = None
    if isinstance(payload, Mapping):
        for key in ("model", "model_state_dict", "state_dict"):
            value = payload.get(key)
            if isinstance(value, Mapping):
                state = value
                break
        if state is None and payload and all(torch.is_tensor(v) for v in payload.values()):
            state = payload
    if state is None:
        raise RuntimeError("Checkpoint does not contain a recognizable state_dict")

    output = {str(key): value for key, value in state.items()}
    if output and all(key.startswith("module.") for key in output):
        output = {key[len("module.") :]: value for key, value in output.items()}
    return output


def parse_filename_config(path: Path) -> Dict[str, Any]:
    match = _SUFFIX_RE.search(path.name)
    if not match:
        raise ValueError(
            "Could not parse data2vec parameters from checkpoint filename: "
            f"{path.name}. Supply the RHM/model overrides explicitly."
        )
    groups = match.groupdict()
    return {
        "rhm_layers": int(groups["rhm_layers"]),
        "tuple_size": int(groups["tuple_size"]),
        "vocab_size": int(groups["vocab_size"]),
        "num_synonyms": int(groups["num_synonyms"]),
        "train_size": int(groups["train_size"]),
        "n_heads": int(groups["n_heads"]),
        "d_model": int(groups["d_model"]),
        "mask_prob": float(groups["mask_prob"]),
        "ema_decay": float(groups["ema_decay"]),
        "online": groups["online"] is not None,
    }


def infer_seed_from_run_dir(run_dir: Path) -> Optional[int]:
    match = _SEED_RE.search(run_dir.name)
    return int(match.group("seed")) if match else None


def infer_state_dimensions(state: Mapping[str, torch.Tensor]) -> Dict[str, int]:
    embed_key = "encoder.embed_tokens.weight"
    pos_key = "encoder.embed_positions.weight"
    fc1_key = "encoder.layers.0.fc1.weight"
    if embed_key not in state or pos_key not in state or fc1_key not in state:
        missing = [key for key in (embed_key, pos_key, fc1_key) if key not in state]
        raise RuntimeError(f"Checkpoint is not a compatible Data2Vec state dict; missing {missing}")

    layer_indices = []
    for key in state:
        match = re.match(r"encoder\.layers\.(\d+)\.", key)
        if match:
            layer_indices.append(int(match.group(1)))
    if not layer_indices:
        raise RuntimeError("Could not infer encoder depth from checkpoint state dict")

    regression_weights = []
    for key, value in state.items():
        match = re.match(r"regression_head\.(\d+)\.weight$", key)
        if match and value.ndim == 2:
            regression_weights.append(int(match.group(1)))

    embedding_shape = tuple(state[embed_key].shape)
    position_shape = tuple(state[pos_key].shape)
    return {
        "model_vocab_size": int(embedding_shape[0]),
        "vocab_size": int(embedding_shape[0]) - 2,
        "d_model": int(embedding_shape[1]),
        "max_seq_len": int(position_shape[0]),
        "n_layers": max(layer_indices) + 1,
        "d_ff": int(state[fc1_key].shape[0]),
        "head_layers": max(1, len(regression_weights)),
    }


def optional_override(value: Optional[Any], fallback: Any) -> Any:
    return fallback if value is None else value


def infer_config(
    checkpoint: Path,
    run_dir: Path,
    cli: argparse.Namespace,
) -> argparse.Namespace:
    """Infer all dimensions required to reconstruct the checkpoint exactly."""

    payload = safe_torch_load(checkpoint, map_location="cpu")
    state = extract_state_dict(payload)
    filename = parse_filename_config(checkpoint)
    dimensions = infer_state_dimensions(state)

    seed_from_name = infer_seed_from_run_dir(run_dir)
    seed_rules = optional_override(cli.seed_rules, seed_from_name)
    if seed_rules is None:
        raise ValueError(
            "Could not infer the RHM grammar seed from the run-directory name. "
            "Pass --seed_rules explicitly."
        )

    config = SimpleNamespace(
        vocab_size=int(optional_override(cli.vocab_size, filename["vocab_size"])),
        num_synonyms=int(optional_override(cli.num_synonyms, filename["num_synonyms"])),
        num_layers=int(optional_override(cli.rhm_layers, filename["rhm_layers"])),
        tuple_size=int(optional_override(cli.tuple_size, filename["tuple_size"])),
        seed_rules=int(seed_rules),
        train_size=int(filename["train_size"]),
        online=bool(filename["online"]),
        d_model=int(dimensions["d_model"]),
        n_heads=int(optional_override(cli.n_heads, filename["n_heads"])),
        n_layers=int(dimensions["n_layers"]),
        d_ff=int(dimensions["d_ff"]),
        dropout=float(cli.dropout),
        mask_prob=float(filename["mask_prob"]),
        mask_length=int(cli.mask_length),
        average_top_k_layers=int(
            optional_override(
                cli.average_top_k_layers,
                min(int(dimensions["n_layers"]), int(filename["rhm_layers"])),
            )
        ),
        loss_beta=float(cli.loss_beta),
        head_layers=int(dimensions["head_layers"]),
        ema_decay=float(filename["ema_decay"]),
        ema_end_decay=float(optional_override(cli.ema_end_decay, filename["ema_decay"])),
        ema_anneal_end_step=int(cli.ema_anneal_end_step),
        layer_norm_target_layer=bool(cli.layer_norm_target_layer),
        layer_norm_targets=bool(cli.layer_norm_targets),
        max_seq_len=int(dimensions["max_seq_len"]),
        model_vocab_size=int(dimensions["model_vocab_size"]),
    )

    expected_seq_len = config.tuple_size ** config.num_layers
    if expected_seq_len != config.max_seq_len:
        raise RuntimeError(
            "RHM/checkpoint sequence-length mismatch: "
            f"tuple_size**num_layers={expected_seq_len}, "
            f"encoder positional table length={config.max_seq_len}."
        )
    if config.vocab_size != dimensions["vocab_size"]:
        raise RuntimeError(
            "Vocabulary mismatch between filename/override and checkpoint: "
            f"{config.vocab_size} vs {dimensions['vocab_size']}"
        )
    if config.d_model != filename["d_model"]:
        raise RuntimeError(
            "d_model mismatch between checkpoint tensor and filename: "
            f"{config.d_model} vs {filename['d_model']}"
        )
    if config.d_model % config.n_heads != 0:
        raise ValueError(
            f"d_model={config.d_model} is not divisible by n_heads={config.n_heads}"
        )
    return argparse.Namespace(**vars(config))


def import_data2vec_modules(data2vec_dir: Path):
    if not (data2vec_dir / "data2vec.py").exists():
        raise FileNotFoundError(f"Missing data2vec.py in {data2vec_dir}")
    if not (data2vec_dir / "random_hierarchy_model.py").exists():
        raise FileNotFoundError(f"Missing random_hierarchy_model.py in {data2vec_dir}")
    if str(data2vec_dir) not in sys.path:
        sys.path.insert(0, str(data2vec_dir))

    from data2vec import Data2Vec  # type: ignore
    from random_hierarchy_model import (  # type: ignore
        sample_data_from_generator_classes,
        sample_rules,
    )

    return Data2Vec, sample_rules, sample_data_from_generator_classes


def build_model(config: argparse.Namespace, Data2Vec: Any) -> torch.nn.Module:
    return Data2Vec(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        dropout=config.dropout,
        max_seq_len=config.max_seq_len,
        average_top_k_layers=config.average_top_k_layers,
        loss_beta=config.loss_beta,
        mask_prob=config.mask_prob,
        mask_length=config.mask_length,
        ema_decay=config.ema_decay,
        ema_end_decay=config.ema_end_decay,
        ema_anneal_end_step=config.ema_anneal_end_step,
        layer_norm_target_layer=config.layer_norm_target_layer,
        layer_norm_targets=config.layer_norm_targets,
        head_layers=config.head_layers,
    )


def reconstruct_model(
    checkpoint: Path,
    config: argparse.Namespace,
    Data2Vec: Any,
    device: str,
    bf16: bool,
) -> torch.nn.Module:
    payload = safe_torch_load(checkpoint, map_location="cpu")
    state = extract_state_dict(payload)
    model = build_model(config, Data2Vec)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/model mismatch for {checkpoint.name}: "
            f"missing={missing}, unexpected={unexpected}"
        )
    model.to(device).eval()
    model._amp_enabled = bool(bf16 and device.startswith("cuda"))
    return model


def validate_checkpoint_compatibility(
    checkpoint: Path,
    config: argparse.Namespace,
) -> None:
    state = extract_state_dict(safe_torch_load(checkpoint, map_location="cpu"))
    dimensions = infer_state_dimensions(state)
    expected = {
        "vocab_size": config.vocab_size,
        "d_model": config.d_model,
        "n_layers": config.n_layers,
        "d_ff": config.d_ff,
        "max_seq_len": config.max_seq_len,
    }
    for key, value in expected.items():
        if int(dimensions[key]) != int(value):
            raise RuntimeError(
                f"Checkpoint {checkpoint.name} changed {key}: "
                f"{dimensions[key]} != {value}"
            )


# =============================================================================
# Checkpoint selection
# =============================================================================


def log_select_indices(number_items: int, number_selected: int) -> np.ndarray:
    if number_selected <= 0 or number_selected >= number_items:
        return np.arange(number_items, dtype=np.int64)
    raw = np.rint(
        np.logspace(0, math.log10(number_items), num=number_selected)
    ).astype(int) - 1
    selected: List[int] = []
    for index in np.clip(raw, 0, number_items - 1).tolist():
        if index not in selected:
            selected.append(index)
    for index in np.rint(
        np.linspace(0, number_items - 1, max(4 * number_selected, number_selected))
    ).astype(int):
        if len(selected) == number_selected:
            break
        if int(index) not in selected:
            selected.append(int(index))
    return np.asarray(sorted(selected[:number_selected]), dtype=np.int64)


def resolve_target_index(checkpoints: Sequence[Path], cli: argparse.Namespace) -> int:
    if cli.target_step is not None:
        matches = [
            index
            for index, path in enumerate(checkpoints)
            if checkpoint_step(path) == int(cli.target_step)
        ]
        if not matches:
            available = [checkpoint_step(path) for path in checkpoints]
            raise ValueError(
                f"target_step={cli.target_step} not found; available={available}"
            )
        return matches[-1]

    target_index = int(cli.target_index)
    if target_index < 0:
        target_index += len(checkpoints)
    if target_index < 0 or target_index >= len(checkpoints):
        raise ValueError(
            f"target_index={cli.target_index} outside [0,{len(checkpoints) - 1}]"
        )
    return target_index


def predictive_source_indices(
    number_checkpoints: int,
    target_index: int,
    take_every: int,
    max_sources: int,
) -> np.ndarray:
    if take_every <= 0:
        raise ValueError("take_every must be positive")
    selected = sorted(set(range(0, number_checkpoints, take_every)) | {target_index})
    if max_sources > 0 and len(selected) > max_sources:
        keep = log_select_indices(len(selected), max_sources)
        selected = [selected[index] for index in keep]
        if target_index not in selected:
            selected[-1] = target_index
        selected = sorted(set(selected))
    return np.asarray(selected, dtype=np.int64)


# =============================================================================
# Fixed RHM reference data
# =============================================================================


def normalize_reference_array(
    value: Any,
    *,
    vocab_size: int,
    seq_len: int,
) -> torch.Tensor:
    array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    if array.ndim == 3:
        if array.shape[1] == vocab_size:
            array = array.argmax(axis=1)
        elif array.shape[2] == vocab_size:
            array = array.argmax(axis=2)
        else:
            raise ValueError(f"Cannot infer one-hot axis for reference shape {array.shape}")
    if array.ndim != 2:
        raise ValueError(f"Expected reference sequences [N,T], got {array.shape}")
    if array.shape[1] != seq_len:
        raise ValueError(
            f"Reference sequence length {array.shape[1]} != expected {seq_len}"
        )
    array = array.astype(np.int64, copy=False)
    if array.size:
        minimum = int(array.min())
        maximum = int(array.max())
        if minimum >= 0 and maximum <= vocab_size - 1:
            array = array + 1
        elif minimum >= 1 and maximum <= vocab_size:
            pass
        else:
            raise ValueError(
                f"Reference token ids outside 0..{vocab_size - 1} or 1..{vocab_size}: "
                f"min={minimum}, max={maximum}"
            )
    return torch.from_numpy(np.ascontiguousarray(array)).long()


def _first_present(mapping: Mapping[str, Any], names: Sequence[str]) -> Optional[Any]:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def load_reference_file(
    path: Path,
    config: argparse.Namespace,
    train_size: int,
    valid_size: int,
    subset_seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Load optional exact/fixed reference sequences from NPZ or PyTorch."""

    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=True) as payload:
            mapping = {name: payload[name] for name in payload.files}
    else:
        payload = safe_torch_load(path, map_location="cpu")
        if torch.is_tensor(payload) or isinstance(payload, np.ndarray):
            mapping = {"sequences": payload}
        elif isinstance(payload, Mapping):
            mapping = dict(payload)
        else:
            raise TypeError(f"Unsupported reference payload type: {type(payload)}")

    train_raw = _first_present(
        mapping,
        ("train_sequences", "train_tokens", "train_inputs", "train", "x_train"),
    )
    valid_raw = _first_present(
        mapping,
        (
            "valid_sequences",
            "val_sequences",
            "test_sequences",
            "valid_tokens",
            "test_tokens",
            "valid",
            "val",
            "test",
            "x_valid",
            "x_test",
        ),
    )

    generator = np.random.default_rng(subset_seed)
    if train_raw is None or valid_raw is None:
        all_raw = _first_present(mapping, ("sequences", "tokens", "inputs", "data", "x"))
        if all_raw is None:
            raise KeyError(
                f"Could not find train/valid or combined sequence arrays in {path}"
            )
        all_sequences = normalize_reference_array(
            all_raw,
            vocab_size=config.vocab_size,
            seq_len=config.max_seq_len,
        )
        needed = train_size + valid_size
        if len(all_sequences) < needed:
            raise ValueError(
                f"Reference file has {len(all_sequences)} sequences, need {needed}"
            )
        indices = generator.choice(len(all_sequences), size=needed, replace=False)
        train = all_sequences[indices[:train_size]]
        valid = all_sequences[indices[train_size:]]
    else:
        train_all = normalize_reference_array(
            train_raw,
            vocab_size=config.vocab_size,
            seq_len=config.max_seq_len,
        )
        valid_all = normalize_reference_array(
            valid_raw,
            vocab_size=config.vocab_size,
            seq_len=config.max_seq_len,
        )
        if len(train_all) < train_size or len(valid_all) < valid_size:
            raise ValueError(
                f"Reference file too small: train={len(train_all)}, valid={len(valid_all)}, "
                f"requested train={train_size}, valid={valid_size}"
            )
        train_indices = generator.choice(len(train_all), size=train_size, replace=False)
        valid_indices = generator.choice(len(valid_all), size=valid_size, replace=False)
        train = train_all[np.sort(train_indices)]
        valid = valid_all[np.sort(valid_indices)]

    metadata = {
        "reference_source": "file",
        "reference_file": str(path),
        "subset_seed": subset_seed,
        "train_size": int(len(train)),
        "valid_size": int(len(valid)),
    }
    return train.contiguous(), valid.contiguous(), metadata


def generate_rhm_references(
    config: argparse.Namespace,
    train_size: int,
    valid_size: int,
    subset_seed: int,
    sample_rules: Any,
    sample_data_from_generator_classes: Any,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Generate fixed, independent samples from the run's exact RHM grammar."""

    rules = sample_rules(
        v=config.vocab_size,
        n=config.vocab_size,
        m=config.num_synonyms,
        s=config.tuple_size,
        L=config.num_layers,
        seed=config.seed_rules,
    )

    def draw(number_samples: int, seed: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(int(seed))
        labels = torch.randint(
            low=0,
            high=config.vocab_size,
            size=(number_samples,),
            generator=generator,
        )
        leaves, _ = sample_data_from_generator_classes(generator, labels, rules)
        # Data2Vec reserves token id 0 for padding; visible RHM symbols are 1..v.
        return leaves.long().add(1).contiguous()

    train = draw(train_size, subset_seed)
    valid = draw(valid_size, subset_seed + 1)
    metadata = {
        "reference_source": "generated_from_rhm_rules",
        "seed_rules": config.seed_rules,
        "train_sample_seed": subset_seed,
        "valid_sample_seed": subset_seed + 1,
        "train_size": train_size,
        "valid_size": valid_size,
        "online_run": bool(config.online),
        "note": (
            "Online pretraining has no finite reusable train set; these are fixed "
            "evaluation samples from the same grammar instance."
        ),
    }
    return train, valid, metadata


def load_references(
    cli: argparse.Namespace,
    config: argparse.Namespace,
    sample_rules: Any,
    sample_data_from_generator_classes: Any,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    train_size = int(cli.subset_train_size)
    valid_size = int(cli.subset_valid_size)
    if train_size <= 0 or valid_size <= 0:
        raise ValueError("subset_train_size and subset_valid_size must be positive")
    if cli.reference_file is not None:
        return load_reference_file(
            cli.reference_file,
            config,
            train_size,
            valid_size,
            int(cli.subset_seed),
        )
    return generate_rhm_references(
        config,
        train_size,
        valid_size,
        int(cli.subset_seed),
        sample_rules,
        sample_data_from_generator_classes,
    )


# =============================================================================
# Representation extraction
# =============================================================================


def representation_names(
    config: argparse.Namespace,
    representation_mode: str,
    include_embedding: bool,
    include_final_norm: bool,
    include_teacher_target: bool,
) -> List[str]:
    if representation_mode == "block_update":
        names = [f"block_update_{index + 1}" for index in range(config.n_layers)]
    else:
        names: List[str] = []
        if include_embedding:
            names.append("embedding")
        names.extend(f"block_{index + 1}" for index in range(config.n_layers))
        if include_final_norm:
            names.append("final_norm")
    if include_teacher_target:
        names.append("teacher_target")
    return names


def autocast_context(device: str, enabled: bool):
    if enabled and device.startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    from contextlib import nullcontext

    return nullcontext()


@torch.no_grad()
def forward_representations(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    encoder_source: str,
    representation_mode: str,
    include_embedding: bool,
    include_final_norm: bool,
    include_teacher_target: bool,
    bf16: bool,
    device: str,
) -> List[torch.Tensor]:
    """Extract unmasked student/teacher states in deterministic eval mode."""

    if encoder_source == "student":
        encoder = model.encoder
    elif encoder_source == "teacher":
        encoder = model.teacher
    else:
        raise ValueError(f"Unknown encoder_source={encoder_source}")

    with autocast_context(device, bf16):
        hidden, key_padding_mask = encoder._embed(input_ids)
        outputs: List[torch.Tensor] = []

        if representation_mode == "state":
            if include_embedding:
                outputs.append(hidden)
            for layer in encoder.layers:
                hidden, _ = layer(hidden, key_padding_mask=key_padding_mask)
                outputs.append(hidden)
            if include_final_norm:
                outputs.append(encoder.final_norm(hidden))
        elif representation_mode == "block_update":
            for layer in encoder.layers:
                next_hidden, _ = layer(hidden, key_padding_mask=key_padding_mask)
                outputs.append(next_hidden - hidden)
                hidden = next_hidden
        else:
            raise ValueError(f"Unknown representation_mode={representation_mode}")

        if include_teacher_target:
            outputs.append(model._teacher_targets(input_ids))

    return [output.detach().float().cpu() for output in outputs]


def representation_path(
    temp_dir: Path,
    role: str,
    checkpoint_index: int,
    layer_index: int,
    position_index: int,
) -> Path:
    return (
        temp_dir
        / role
        / f"repr_k{checkpoint_index:05d}_l{layer_index:03d}_p{position_index:04d}.npy"
    )


@torch.no_grad()
def store_checkpoint_representations(
    checkpoint: Path,
    checkpoint_index: int,
    role: str,
    reference: torch.Tensor,
    config: argparse.Namespace,
    Data2Vec: Any,
    cli: argparse.Namespace,
    temp_dir: Path,
    encoder_source: str,
) -> Tuple[List[str], int]:
    """Stream one checkpoint's [sample,position,feature] tensors to NPY memmaps."""

    model = reconstruct_model(
        checkpoint,
        config,
        Data2Vec,
        cli.device,
        cli.bf16,
    )
    names = representation_names(
        config,
        cli.representation_mode,
        cli.include_embedding,
        cli.include_final_norm,
        cli.include_teacher_target,
    )
    loader = DataLoader(
        TensorDataset(reference),
        batch_size=cli.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=cli.device.startswith("cuda"),
    )

    writers: Optional[List[List[np.memmap]]] = None
    offset = 0
    number_positions = -1
    for (batch,) in loader:
        batch = batch.to(cli.device, non_blocking=True)
        representations = forward_representations(
            model,
            batch,
            encoder_source=encoder_source,
            representation_mode=cli.representation_mode,
            include_embedding=cli.include_embedding,
            include_final_norm=cli.include_final_norm,
            include_teacher_target=cli.include_teacher_target,
            bf16=cli.bf16,
            device=cli.device,
        )
        if len(representations) != len(names):
            raise RuntimeError(
                f"Representation count {len(representations)} != name count {len(names)}"
            )
        if writers is None:
            number_positions = int(representations[0].shape[1])
            writers = []
            for layer_index, representation in enumerate(representations):
                if representation.shape[1] != number_positions:
                    raise RuntimeError("Number of positions differs across representations")
                layer_writers: List[np.memmap] = []
                feature_dim = int(representation.shape[2])
                for position_index in range(number_positions):
                    path = representation_path(
                        temp_dir,
                        role,
                        checkpoint_index,
                        layer_index,
                        position_index,
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    layer_writers.append(
                        open_memmap(
                            path,
                            mode="w+",
                            dtype=np.float32,
                            shape=(len(reference), feature_dim),
                        )
                    )
                writers.append(layer_writers)

        assert writers is not None
        batch_size = int(batch.shape[0])
        for layer_index, representation in enumerate(representations):
            array = representation.numpy()
            for position_index in range(number_positions):
                writers[layer_index][position_index][offset : offset + batch_size] = array[
                    :, position_index, :
                ]
        offset += batch_size

    if writers is None or offset != len(reference):
        raise RuntimeError(
            f"Representation extraction wrote {offset}/{len(reference)} samples"
        )
    for layer_writers in writers:
        for writer in layer_writers:
            writer.flush()
    del writers
    del model
    if cli.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return names, number_positions


# =============================================================================
# Predictive ridge overlap
# =============================================================================


def parse_ridge_alphas(value: str) -> np.ndarray:
    alphas = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        alpha = float(item)
        if alpha < 0:
            raise ValueError(f"Ridge alpha must be non-negative, got {alpha}")
        alphas.append(alpha)
    if not alphas:
        raise ValueError("Empty ridge-alpha grid")
    return np.asarray(alphas, dtype=np.float64)


def make_cv_folds(number_samples: int, number_folds: int, seed: int) -> List[np.ndarray]:
    if number_folds < 2:
        raise ValueError("predictive_num_folds must be >=2")
    if number_samples < number_folds:
        raise ValueError(
            f"Need at least one sample per fold: N={number_samples}, K={number_folds}"
        )
    permutation = np.random.default_rng(seed).permutation(number_samples)
    return [fold.astype(np.int64, copy=False) for fold in np.array_split(permutation, number_folds)]


def _torch_dtype(name: str) -> torch.dtype:
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported ridge_dtype={name}")


def _inverse_shifted_eigenvalues(
    eigenvalues: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    if alpha > 0:
        return 1.0 / (eigenvalues + float(alpha))
    maximum = float(eigenvalues.max().detach().cpu()) if eigenvalues.numel() else 0.0
    tolerance = max(torch.finfo(eigenvalues.dtype).eps * max(1, eigenvalues.numel()) * maximum, 0.0)
    return torch.where(
        eigenvalues > tolerance,
        1.0 / eigenvalues.clamp_min(torch.finfo(eigenvalues.dtype).tiny),
        torch.zeros_like(eigenvalues),
    )


def ridge_predictive_ev_cv(
    x: np.ndarray,
    y: np.ndarray,
    folds: Sequence[np.ndarray],
    alphas: np.ndarray,
    *,
    standardize_x: bool,
    device: str,
    dtype_name: str,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Exact K-fold ridge CV with a primal/dual eigendecomposition per fold.

    The returned explained variance is not clipped.  Negative values therefore
    retain the interpretation "worse than predicting the training-fold mean".
    """

    x_np = np.asarray(x)
    y_np = np.asarray(y)
    if x_np.ndim != 2 or y_np.ndim != 2 or x_np.shape[0] != y_np.shape[0]:
        raise ValueError(f"Expected aligned 2D X,Y; got {x_np.shape}, {y_np.shape}")

    dtype = _torch_dtype(dtype_name)
    number_samples = x_np.shape[0]
    all_indices = np.arange(number_samples, dtype=np.int64)
    fold_scores = np.full((len(alphas), len(folds)), np.nan, dtype=np.float64)
    total_numerators = np.zeros(len(alphas), dtype=np.float64)
    total_denominators = np.zeros(len(alphas), dtype=np.float64)

    for fold_index, test_index in enumerate(folds):
        test_mask = np.zeros(number_samples, dtype=bool)
        test_mask[test_index] = True
        train_index = all_indices[~test_mask]

        # Explicit copies avoid read-only memmap warnings and release files promptly.
        x_train = torch.as_tensor(
            np.array(x_np[train_index], dtype=np.float64 if dtype == torch.float64 else np.float32, copy=True),
            dtype=dtype,
            device=device,
        )
        x_test = torch.as_tensor(
            np.array(x_np[test_index], dtype=np.float64 if dtype == torch.float64 else np.float32, copy=True),
            dtype=dtype,
            device=device,
        )
        y_train = torch.as_tensor(
            np.array(y_np[train_index], dtype=np.float64 if dtype == torch.float64 else np.float32, copy=True),
            dtype=dtype,
            device=device,
        )
        y_test = torch.as_tensor(
            np.array(y_np[test_index], dtype=np.float64 if dtype == torch.float64 else np.float32, copy=True),
            dtype=dtype,
            device=device,
        )

        x_mean = x_train.mean(dim=0, keepdim=True)
        x_train = x_train - x_mean
        x_test = x_test - x_mean
        if standardize_x:
            x_std = x_train.std(dim=0, unbiased=False, keepdim=True)
            x_std = torch.where(x_std > torch.finfo(dtype).eps, x_std, torch.ones_like(x_std))
            x_train = x_train / x_std
            x_test = x_test / x_std

        y_mean = y_train.mean(dim=0, keepdim=True)
        y_train = y_train - y_mean
        y_test = y_test - y_mean
        denominator_tensor = torch.sum(y_test * y_test)
        denominator = float(denominator_tensor.detach().cpu())
        if denominator <= EPS64:
            continue

        # Choose the smaller exact ridge system.  The dual form is normally used
        # for the 2048-dimensional data2vec states with a smaller CV sample count.
        if x_train.shape[0] <= x_train.shape[1]:
            gram = x_train @ x_train.T
            gram = 0.5 * (gram + gram.T)
            eigenvalues, eigenvectors = torch.linalg.eigh(gram)
            eigenvalues = eigenvalues.clamp_min(0)
            left = (x_test @ x_train.T) @ eigenvectors
            right = eigenvectors.T @ y_train
        else:
            gram = x_train.T @ x_train
            gram = 0.5 * (gram + gram.T)
            eigenvalues, eigenvectors = torch.linalg.eigh(gram)
            eigenvalues = eigenvalues.clamp_min(0)
            left = x_test @ eigenvectors
            right = eigenvectors.T @ (x_train.T @ y_train)

        for alpha_index, alpha in enumerate(alphas):
            inverse = _inverse_shifted_eigenvalues(eigenvalues, float(alpha))
            prediction = (left * inverse.unsqueeze(0)) @ right
            residual = y_test - prediction
            numerator = float(torch.sum(residual * residual).detach().cpu())
            score = 1.0 - numerator / denominator
            fold_scores[alpha_index, fold_index] = score
            total_numerators[alpha_index] += numerator
            total_denominators[alpha_index] += denominator

        del x_train, x_test, y_train, y_test, gram, eigenvalues, eigenvectors, left, right
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    scores_by_alpha = np.full(len(alphas), np.nan, dtype=np.float64)
    valid = total_denominators > EPS64
    scores_by_alpha[valid] = 1.0 - total_numerators[valid] / total_denominators[valid]
    if not np.any(np.isfinite(scores_by_alpha)):
        return (
            float("nan"),
            float("nan"),
            np.full(len(folds), np.nan, dtype=np.float64),
            scores_by_alpha,
        )
    best_index = int(np.nanargmax(scores_by_alpha))
    return (
        float(scores_by_alpha[best_index]),
        float(alphas[best_index]),
        fold_scores[best_index].copy(),
        scores_by_alpha,
    )


# =============================================================================
# Output and resume
# =============================================================================


def save_partial(
    path: Path,
    *,
    q: np.ndarray,
    best_alpha: np.ndarray,
    fold_q: np.ndarray,
    alpha_scores: np.ndarray,
    completed_sources: np.ndarray,
    source_steps: np.ndarray,
    target_step: int,
) -> None:
    np.savez_compressed(
        path,
        predictive_q_by_position=q,
        predictive_best_alpha_by_position=best_alpha,
        predictive_fold_q_by_position=fold_q,
        predictive_alpha_scores_by_position=alpha_scores,
        completed_sources=completed_sources,
        source_steps=source_steps,
        target_step=np.asarray(target_step, dtype=np.int64),
    )


def load_partial(
    path: Path,
    *,
    expected_shape: Tuple[int, int, int, int],
    source_steps: np.ndarray,
    target_step: int,
    number_folds: int,
    number_alphas: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as payload:
        q = payload["predictive_q_by_position"]
        best = payload["predictive_best_alpha_by_position"]
        fold = payload["predictive_fold_q_by_position"]
        alpha = payload["predictive_alpha_scores_by_position"]
        completed = payload["completed_sources"]
        saved_steps = payload["source_steps"]
        saved_target = int(payload["target_step"])
    if tuple(q.shape) != expected_shape:
        print(f"[WARN] Ignoring incompatible partial shape {q.shape} != {expected_shape}")
        return None
    if tuple(fold.shape) != expected_shape + (number_folds,):
        print("[WARN] Ignoring partial with incompatible fold dimension")
        return None
    if tuple(alpha.shape) != expected_shape + (number_alphas,):
        print("[WARN] Ignoring partial with incompatible alpha dimension")
        return None
    if not np.array_equal(saved_steps, source_steps) or saved_target != target_step:
        print("[WARN] Ignoring partial from a different checkpoint selection")
        return None
    return q, best, fold, alpha, completed.astype(bool, copy=False)


def save_final(
    output_dir: Path,
    *,
    q: np.ndarray,
    best_alpha: np.ndarray,
    fold_q: np.ndarray,
    alpha_scores: np.ndarray,
    alphas: np.ndarray,
    folds: Sequence[np.ndarray],
    source_checkpoints: Sequence[Path],
    target_checkpoint: Path,
    source_layer_names: Sequence[str],
    target_layer_names: Sequence[str],
    reference_metadata: Mapping[str, Any],
    config: argparse.Namespace,
    cli: argparse.Namespace,
    elapsed_seconds: float,
) -> None:
    source_steps = np.asarray([checkpoint_step(path) for path in source_checkpoints], dtype=np.int64)
    target_step = int(checkpoint_step(target_checkpoint) or -1)
    samples_seen = source_steps * int(cli.training_batch_size)
    target_samples_seen = target_step * int(cli.training_batch_size)

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        q_mean = np.nanmean(q, axis=0)
        alpha_median = np.nanmedian(best_alpha, axis=0)

    output_path = output_dir / "predictive_overlaps.npz"
    np.savez_compressed(
        output_path,
        predictive_q_by_position=q,
        predictive_q_position_mean=q_mean,
        predictive_best_alpha_by_position=best_alpha,
        predictive_best_alpha_position_median=alpha_median,
        predictive_fold_q_by_position=fold_q,
        predictive_alpha_scores_by_position=alpha_scores,
        ridge_alphas=alphas,
        fold_sizes=np.asarray([len(fold) for fold in folds], dtype=np.int64),
        source_layer_names=np.asarray(source_layer_names, dtype=object),
        target_layer_names=np.asarray(target_layer_names, dtype=object),
        source_steps=source_steps,
        source_training_samples_seen=samples_seen,
        source_checkpoint_files=np.asarray([str(path) for path in source_checkpoints], dtype=object),
        target_step=np.asarray(target_step, dtype=np.int64),
        target_training_samples_seen=np.asarray(target_samples_seen, dtype=np.int64),
        target_checkpoint_file=np.asarray(str(target_checkpoint)),
        token_positions_1based=np.arange(1, q.shape[0] + 1, dtype=np.int64),
        representation_mode=np.asarray(cli.representation_mode),
        source_encoder=np.asarray(cli.source_encoder),
        target_encoder=np.asarray(cli.target_encoder),
        predictive_layer_mode=np.asarray(cli.predictive_layer_mode),
        predictive_num_folds=np.asarray(cli.predictive_num_folds, dtype=np.int64),
        predictive_standardize_x=np.asarray(cli.predictive_standardize_x),
        training_batch_size=np.asarray(cli.training_batch_size, dtype=np.int64),
        include_embedding=np.asarray(cli.include_embedding),
        include_final_norm=np.asarray(cli.include_final_norm),
        include_teacher_target=np.asarray(cli.include_teacher_target),
        reference_metadata_json=np.asarray(json.dumps(jsonify(reference_metadata), sort_keys=True)),
        config_json=np.asarray(json.dumps(jsonify(vars(config)), sort_keys=True)),
    )

    metadata = {
        "output_path": output_path,
        "run_dir": cli.run_dir,
        "data2vec_dir": cli.data2vec_dir,
        "source_checkpoints": list(source_checkpoints),
        "target_checkpoint": target_checkpoint,
        "source_steps": source_steps,
        "target_step": target_step,
        "source_training_samples_seen": samples_seen,
        "target_training_samples_seen": target_samples_seen,
        "training_batch_size": cli.training_batch_size,
        "representation_mode": cli.representation_mode,
        "source_encoder": cli.source_encoder,
        "target_encoder": cli.target_encoder,
        "predictive_layer_mode": cli.predictive_layer_mode,
        "source_layer_names": list(source_layer_names),
        "target_layer_names": list(target_layer_names),
        "num_positions": int(q.shape[0]),
        "ridge_alphas": alphas,
        "predictive_num_folds": cli.predictive_num_folds,
        "predictive_standardize_x": cli.predictive_standardize_x,
        "ridge_device": cli.ridge_device,
        "ridge_dtype": cli.ridge_dtype,
        "reference": reference_metadata,
        "config": vars(config),
        "elapsed_seconds": elapsed_seconds,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(jsonify(metadata), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"[INFO] saved {output_path}", flush=True)


# =============================================================================
# Main computation
# =============================================================================


def compute_predictive_overlaps(
    source_checkpoints: Sequence[Path],
    target_checkpoint: Path,
    reference: torch.Tensor,
    config: argparse.Namespace,
    Data2Vec: Any,
    cli: argparse.Namespace,
    output_dir: Path,
    temp_dir: Path,
    reference_metadata: Mapping[str, Any],
) -> None:
    alphas = parse_ridge_alphas(cli.ridge_alphas)
    folds = make_cv_folds(
        len(reference),
        int(cli.predictive_num_folds),
        int(cli.subset_seed) + 7919,
    )

    print(f"[INFO] extracting target representations: {target_checkpoint.name}", flush=True)
    target_layer_names, number_positions = store_checkpoint_representations(
        target_checkpoint,
        0,
        "target",
        reference,
        config,
        Data2Vec,
        cli,
        temp_dir,
        cli.target_encoder,
    )
    source_layer_names = representation_names(
        config,
        cli.representation_mode,
        cli.include_embedding,
        cli.include_final_norm,
        cli.include_teacher_target,
    )

    number_source_layers = len(source_layer_names)
    number_target_layers = len(target_layer_names)
    number_sources = len(source_checkpoints)
    shape = (
        number_positions,
        number_source_layers,
        number_target_layers,
        number_sources,
    )
    q = np.full(shape, np.nan, dtype=np.float32)
    best_alpha = np.full(shape, np.nan, dtype=np.float32)
    fold_q = np.full(shape + (len(folds),), np.nan, dtype=np.float32)
    alpha_scores = np.full(shape + (len(alphas),), np.nan, dtype=np.float32)
    completed_sources = np.zeros(number_sources, dtype=bool)

    source_steps = np.asarray([checkpoint_step(path) for path in source_checkpoints], dtype=np.int64)
    target_step = int(checkpoint_step(target_checkpoint) or -1)
    partial_path = output_dir / "predictive_overlaps.partial.npz"
    if cli.resume:
        loaded = load_partial(
            partial_path,
            expected_shape=shape,
            source_steps=source_steps,
            target_step=target_step,
            number_folds=len(folds),
            number_alphas=len(alphas),
        )
        if loaded is not None:
            q, best_alpha, fold_q, alpha_scores, completed_sources = loaded
            print(
                f"[INFO] resumed {int(completed_sources.sum())}/{number_sources} source checkpoints",
                flush=True,
            )

    if cli.predictive_layer_mode == "same":
        layer_pairs = [
            (index, index)
            for index in range(min(number_source_layers, number_target_layers))
        ]
    else:
        layer_pairs = [
            (source_index, target_index)
            for source_index in range(number_source_layers)
            for target_index in range(number_target_layers)
        ]

    started = time.time()
    for source_index, checkpoint in enumerate(source_checkpoints):
        if completed_sources[source_index]:
            print(
                f"[INFO] source {source_index + 1}/{number_sources} already complete: {checkpoint.name}",
                flush=True,
            )
            continue

        validate_checkpoint_compatibility(checkpoint, config)
        print(
            f"[INFO] extracting source {source_index + 1}/{number_sources}: {checkpoint.name}",
            flush=True,
        )
        names, source_positions = store_checkpoint_representations(
            checkpoint,
            source_index,
            "source",
            reference,
            config,
            Data2Vec,
            cli,
            temp_dir,
            cli.source_encoder,
        )
        if names != source_layer_names or source_positions != number_positions:
            raise RuntimeError("Representation layout changed across checkpoints")

        for position_index in range(number_positions):
            source_cache: Dict[int, np.ndarray] = {}
            target_cache: Dict[int, np.ndarray] = {}
            for pair_counter, (source_layer_index, target_layer_index) in enumerate(layer_pairs):
                print(
                    "[INFO] ridge "
                    f"source={source_index + 1}/{number_sources} "
                    f"position={position_index + 1}/{number_positions} "
                    f"layers={source_layer_index}->{target_layer_index} "
                    f"pair={pair_counter + 1}/{len(layer_pairs)}",
                    flush=True,
                )
                if source_layer_index not in source_cache:
                    source_cache[source_layer_index] = np.load(
                        representation_path(
                            temp_dir,
                            "source",
                            source_index,
                            source_layer_index,
                            position_index,
                        ),
                        mmap_mode="r",
                    )
                if target_layer_index not in target_cache:
                    target_cache[target_layer_index] = np.load(
                        representation_path(
                            temp_dir,
                            "target",
                            0,
                            target_layer_index,
                            position_index,
                        ),
                        mmap_mode="r",
                    )

                value, alpha, fold_values, alpha_values = ridge_predictive_ev_cv(
                    source_cache[source_layer_index],
                    target_cache[target_layer_index],
                    folds,
                    alphas,
                    standardize_x=bool(cli.predictive_standardize_x),
                    device=cli.ridge_device,
                    dtype_name=cli.ridge_dtype,
                )
                q[position_index, source_layer_index, target_layer_index, source_index] = value
                best_alpha[
                    position_index, source_layer_index, target_layer_index, source_index
                ] = alpha
                fold_q[
                    position_index,
                    source_layer_index,
                    target_layer_index,
                    source_index,
                    :,
                ] = fold_values.astype(np.float32, copy=False)
                alpha_scores[
                    position_index,
                    source_layer_index,
                    target_layer_index,
                    source_index,
                    :,
                ] = alpha_values.astype(np.float32, copy=False)

            source_cache.clear()
            target_cache.clear()

        completed_sources[source_index] = True
        save_partial(
            partial_path,
            q=q,
            best_alpha=best_alpha,
            fold_q=fold_q,
            alpha_scores=alpha_scores,
            completed_sources=completed_sources,
            source_steps=source_steps,
            target_step=target_step,
        )

        source_temp = temp_dir / "source"
        if source_temp.exists():
            shutil.rmtree(source_temp)
        if cli.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - started
    save_final(
        output_dir,
        q=q,
        best_alpha=best_alpha,
        fold_q=fold_q,
        alpha_scores=alpha_scores,
        alphas=alphas,
        folds=folds,
        source_checkpoints=source_checkpoints,
        target_checkpoint=target_checkpoint,
        source_layer_names=source_layer_names,
        target_layer_names=target_layer_names,
        reference_metadata=reference_metadata,
        config=config,
        cli=cli,
        elapsed_seconds=elapsed,
    )
    partial_path.unlink(missing_ok=True)


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # Paths and run.
    parser.add_argument("--repo_dir", type=Path, default=Path.cwd())
    parser.add_argument("--data2vec_dir", type=Path, default=Path("data2vec"))
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, default=Path("collected_results"))
    parser.add_argument("--output_name", type=str, default=None)
    parser.add_argument("--reference_file", type=Path, default=None)

    # Checkpoint selection.
    parser.add_argument("--max_step", type=int, default=None)
    parser.add_argument("--take_every", type=int, default=4)
    parser.add_argument("--num_checkpoints", type=int, default=0)
    parser.add_argument("--target_step", type=int, default=None)
    parser.add_argument("--target_index", type=int, default=-1)

    # Fixed reference set.
    parser.add_argument("--subset_train_size", type=int, default=128)
    parser.add_argument("--subset_valid_size", type=int, default=2048)
    parser.add_argument("--subset_seed", type=int, default=12345)
    parser.add_argument("--batch_size", type=int, default=128)

    # Representation definition.
    parser.add_argument("--source_encoder", choices=("student", "teacher"), default="student")
    parser.add_argument("--target_encoder", choices=("student", "teacher"), default="student")
    parser.add_argument(
        "--representation_mode",
        choices=("state", "block_update"),
        default="state",
    )
    parser.add_argument("--include_embedding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include_final_norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--include_teacher_target",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Append the averaged top-K teacher target as an additional representation.",
    )

    # Predictive readout.
    parser.add_argument("--predictive_layer_mode", choices=("same", "all"), default="same")
    parser.add_argument("--predictive_num_folds", type=int, default=5)
    parser.add_argument(
        "--ridge_alphas",
        type=str,
        default="1e-6,1e-4,1e-2,1e0,1e2,1e4,1e6",
    )
    parser.add_argument(
        "--predictive_standardize_x",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--ridge_device", type=str, default="auto")
    parser.add_argument("--ridge_dtype", choices=("float64", "float32"), default="float64")

    # Runtime and bookkeeping.
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--training_batch_size", type=int, default=512)
    parser.add_argument("--temp_root", type=Path, default=None)
    parser.add_argument("--keep_temp", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)

    # Optional inference overrides.  Normally only --seed_rules is ever needed.
    parser.add_argument("--seed_rules", type=int, default=None)
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--num_synonyms", type=int, default=None)
    parser.add_argument("--rhm_layers", type=int, default=None)
    parser.add_argument("--tuple_size", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=None)

    # Non-shape hyperparameters needed to reconstruct Data2Vec.  Defaults match
    # Table 1 and the uploaded implementation; they do not affect eval states
    # except average_top_k when --include_teacher_target is requested.
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mask_length", type=int, default=1)
    parser.add_argument("--average_top_k_layers", type=int, default=None)
    parser.add_argument("--loss_beta", type=float, default=4.0)
    parser.add_argument("--ema_end_decay", type=float, default=None)
    parser.add_argument("--ema_anneal_end_step", type=int, default=100_000)
    parser.add_argument(
        "--layer_norm_target_layer",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--layer_norm_targets",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    cli.repo_dir = cli.repo_dir.expanduser().resolve()
    cli.data2vec_dir = resolve_path(cli.data2vec_dir, cli.repo_dir)
    cli.run_dir = resolve_path(cli.run_dir, cli.repo_dir)
    cli.output_root = resolve_path(cli.output_root, cli.repo_dir)
    if cli.reference_file is not None:
        cli.reference_file = resolve_path(cli.reference_file, cli.repo_dir)

    if not cli.run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {cli.run_dir}")
    if cli.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; using CPU", flush=True)
        cli.device = "cpu"
    if cli.ridge_device == "auto":
        cli.ridge_device = cli.device
    if cli.ridge_device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA ridge device unavailable; using CPU", flush=True)
        cli.ridge_device = "cpu"

    Data2Vec, sample_rules, sample_data_from_generator_classes = import_data2vec_modules(
        cli.data2vec_dir
    )
    all_checkpoints = find_checkpoints(cli.run_dir, cli.max_step)
    target_index = resolve_target_index(all_checkpoints, cli)
    selected_indices = predictive_source_indices(
        len(all_checkpoints),
        target_index,
        int(cli.take_every),
        int(cli.num_checkpoints),
    )
    source_checkpoints = [all_checkpoints[index] for index in selected_indices]
    target_checkpoint = all_checkpoints[target_index]

    config = infer_config(target_checkpoint, cli.run_dir, cli)
    for checkpoint in source_checkpoints:
        validate_checkpoint_compatibility(checkpoint, config)

    output_name = cli.output_name or (
        f"data2vec_predictive_{cli.run_dir.name}_"
        f"{cli.source_encoder}_to_{cli.target_encoder}_{cli.representation_mode}"
    )
    output_dir = cli.output_root / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_root = output_dir if cli.temp_root is None else resolve_path(cli.temp_root, cli.repo_dir)
    temp_dir = temp_root / f"_tmp_{output_name}"
    if temp_dir.exists() and not cli.resume:
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    train_reference, valid_reference, reference_metadata = load_references(
        cli,
        config,
        sample_rules,
        sample_data_from_generator_classes,
    )
    reference = torch.cat((train_reference, valid_reference), dim=0).contiguous()
    reference_metadata = dict(reference_metadata)
    reference_metadata.update(
        {
            "readout_reference_source": "train_reference_plus_valid_reference",
            "readout_num_samples": int(len(reference)),
            "readout_num_folds": int(cli.predictive_num_folds),
            "readout_fold_seed": int(cli.subset_seed) + 7919,
        }
    )

    print(f"[INFO] run_dir={cli.run_dir}", flush=True)
    print(f"[INFO] data2vec_dir={cli.data2vec_dir}", flush=True)
    print(
        f"[INFO] source checkpoints={len(source_checkpoints)}/{len(all_checkpoints)} "
        f"steps={[checkpoint_step(path) for path in source_checkpoints]}",
        flush=True,
    )
    print(
        f"[INFO] target checkpoint={target_checkpoint.name} step={checkpoint_step(target_checkpoint)}",
        flush=True,
    )
    print(f"[INFO] output_dir={output_dir}", flush=True)
    print(f"[INFO] reference shape={tuple(reference.shape)}", flush=True)
    print(f"[INFO] inferred config={json.dumps(jsonify(vars(config)), sort_keys=True)}", flush=True)

    total_started = time.time()
    compute_predictive_overlaps(
        source_checkpoints,
        target_checkpoint,
        reference,
        config,
        Data2Vec,
        cli,
        output_dir,
        temp_dir,
        reference_metadata,
    )
    print(f"[INFO] total time={time.time() - total_started:.2f}s", flush=True)

    if cli.keep_temp:
        print(f"[INFO] temporary representations kept in {temp_dir}", flush=True)
    else:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
