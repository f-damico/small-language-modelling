#!/usr/bin/env python3
"""Compute checkpoint-to-checkpoint representation or weight overlaps.

Expected run layout
-------------------

data/<run_name>/<run_id>/
    args.json
    results.pt
    rules.pt
    data/
        tree_rules.pt
        tree_metadata.json
        dataset_full.npz
    checkpoints/
        checkpoint_step_<step>.pt

Outputs are written to:

collected_results/<output_name>/

Representation target
---------------------
Centered linear CKA is computed separately by split, token position and layer.

* metric=gram:
    Sample-space Gram implementation.

* metric=feature:
    Feature-space implementation.

* representation_mode=state:
    Embedding, residual state after each block/layer, and optionally the final
    normalized state.

* representation_mode=block_update:
    Residual write h_out - h_in for each block/layer.

Predictive target
-----------------
A King-style cross-validated linear readout is fitted from source
representations H(t,l) to one fixed target representation H(t',l').  The
reported score is the un-clipped normalized explained variance on held-out
folds, with ridge regularisation selected by five-fold cross-validation by
default.  Source times are selected every --take_every saved checkpoints, and
the target time defaults to the last saved checkpoint.

Weight target
-------------
Cosine overlaps are computed for:

* all trainable parameters together;
* generic parameter groups;
* every individual parameter;
* every residual layer.

Transformer layers additionally have separate:

* global layer overlap;
* attention overlap;
* MLP overlap;
* normalization overlap.

Mamba layers additionally have separate:

* global layer overlap;
* mixer overlap;
* normalization overlap.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import re
import shutil
import sys
import time

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from torch.utils.data import DataLoader, TensorDataset


EPS = 1e-12
DOT_BLOCK = 1_000_000


# =============================================================================
# General helpers
# =============================================================================


def safe_load(
    path: Path,
    map_location: str | torch.device = "cpu",
) -> Any:
    """Load a PyTorch file compatibly across PyTorch versions."""

    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )

    except TypeError:
        return torch.load(
            path,
            map_location=map_location,
        )


def jsonify(value: Any) -> Any:
    """Convert objects to JSON-serializable values."""

    if isinstance(
        value,
        (str, int, float, bool),
    ) or value is None:
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
        return {
            str(key): jsonify(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            jsonify(item)
            for item in value
        ]

    if hasattr(value, "__dict__"):
        return jsonify(vars(value))

    return repr(value)


def as_namespace(
    value: Any,
) -> argparse.Namespace:
    """Convert a saved configuration into argparse.Namespace."""

    if isinstance(value, argparse.Namespace):
        return copy.deepcopy(value)

    if isinstance(value, SimpleNamespace):
        return argparse.Namespace(
            **vars(value)
        )

    if isinstance(value, Mapping):
        return argparse.Namespace(
            **dict(value)
        )

    if hasattr(value, "__dict__"):
        return argparse.Namespace(
            **vars(value)
        )

    raise TypeError(
        f"Cannot convert {type(value)} "
        "to argparse.Namespace"
    )


def resolve_relative(
    path: Path,
    base: Path,
) -> Path:
    """Resolve a possibly relative path against a base folder."""

    path = path.expanduser()

    if path.is_absolute():
        return path.resolve()

    return (
        base
        / path
    ).resolve()


def resolve_source_dir(
    repo_dir: Path,
    source_dir: Optional[Path],
) -> Path:
    """Find the folder containing init.py."""

    if source_dir is not None:
        output = resolve_relative(
            source_dir,
            repo_dir,
        )

        if not (
            output
            / "init.py"
        ).exists():
            raise FileNotFoundError(
                f"No init.py found in {output}"
            )

        return output

    candidates = (
        repo_dir / "SLM",
        repo_dir,
    )

    for candidate in candidates:
        if (
            candidate
            / "init.py"
        ).exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not find SLM/init.py or "
        "init.py in the repository."
    )


def import_init(
    source_dir: Path,
) -> Any:
    """Import the repository init module."""

    if str(source_dir) not in sys.path:
        sys.path.insert(
            0,
            str(source_dir),
        )

    return importlib.import_module(
        "init"
    )


# =============================================================================
# Run and checkpoint discovery
# =============================================================================


def resolve_run_dir(
    repo_dir: Path,
    data_root: Path,
    run_dir: Optional[Path],
    run_name: Optional[str],
    run_id: Optional[str],
) -> Path:
    """Resolve the leaf directory containing one run."""

    if run_dir is not None:
        output = resolve_relative(
            run_dir,
            repo_dir,
        )

    else:
        if not run_name:
            raise ValueError(
                "Provide --run_dir, or --run_name "
                "with optional --run_id."
            )

        output = Path(
            run_name
        ).expanduser()

        if not output.is_absolute():
            output = (
                data_root
                / output
            )

        if run_id:
            output = (
                output
                / run_id
            )

        output = output.resolve()

    if not output.is_dir():
        raise FileNotFoundError(
            "Leaf run directory does not exist: "
            f"{output}"
        )

    return output


def checkpoint_step(
    path: Path,
) -> Optional[int]:
    """Extract the training step from a checkpoint filename."""

    patterns = (
        r"checkpoint_step_(\d+)\.pt$",
        r"checkpoint_epoch_(\d+)\.pt$",
        r"step[_-]?(\d+)\.pt$",
        r"epoch[_-]?(\d+)\.pt$",
    )

    for pattern in patterns:
        match = re.search(
            pattern,
            path.name,
        )

        if match:
            return int(
                match.group(1)
            )

    return None


def find_checkpoints(
    run_dir: Path,
    max_step: Optional[int],
) -> List[Path]:
    """Find checkpoint_step_<step>.pt files."""

    checkpoint_dir = (
        run_dir
        / "checkpoints"
    )

    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            "Missing checkpoint folder: "
            f"{checkpoint_dir}"
        )

    found: List[
        Tuple[int, Path]
    ] = []

    for path in checkpoint_dir.glob(
        "*.pt"
    ):
        step = checkpoint_step(
            path
        )

        if step is None:
            continue

        if (
            max_step is None
            or step <= max_step
        ):
            found.append(
                (
                    step,
                    path.resolve(),
                )
            )

    found.sort(
        key=lambda item: (
            item[0],
            item[1].name,
        )
    )

    if not found:
        raise FileNotFoundError(
            "No checkpoint_step_<step>.pt "
            f"files found in {checkpoint_dir}"
        )

    return [
        path
        for _, path in found
    ]


def log_select_indices(
    number_items: int,
    number_selected: int,
) -> np.ndarray:
    """Select approximately logarithmically spaced checkpoint indices."""

    if (
        number_selected <= 0
        or number_selected >= number_items
    ):
        return np.arange(
            number_items,
            dtype=np.int64,
        )

    raw = (
        np.rint(
            np.logspace(
                0,
                math.log10(number_items),
                num=number_selected,
            )
        ).astype(int)
        - 1
    )

    selected: List[int] = []

    for index in np.clip(
        raw,
        0,
        number_items - 1,
    ).tolist():
        if index not in selected:
            selected.append(index)

    linear_candidates = np.rint(
        np.linspace(
            0,
            number_items - 1,
            max(
                4 * number_selected,
                number_selected,
            ),
        )
    ).astype(int)

    for index in linear_candidates:
        if len(selected) == number_selected:
            break

        index = int(index)

        if index not in selected:
            selected.append(index)

    return np.asarray(
        sorted(
            selected[
                :number_selected
            ]
        ),
        dtype=np.int64,
    )


def merge_missing(
    target: argparse.Namespace,
    source: argparse.Namespace,
) -> argparse.Namespace:
    """Fill fields missing from target using source."""

    for key, value in vars(
        source
    ).items():
        if (
            not hasattr(
                target,
                key,
            )
            or getattr(
                target,
                key,
            ) is None
        ):
            setattr(
                target,
                key,
                value,
            )

    return target


def load_config(
    run_dir: Path,
    device: str,
) -> argparse.Namespace:
    """Load configuration from args.json with results.pt fallback."""

    config: Optional[
        argparse.Namespace
    ] = None

    args_path = (
        run_dir
        / "args.json"
    )

    results_path = (
        run_dir
        / "results.pt"
    )

    if args_path.exists():
        with args_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            config = as_namespace(
                json.load(handle)
            )

    if results_path.exists():
        payload = safe_load(
            results_path
        )

        if (
            isinstance(
                payload,
                Mapping,
            )
            and "config" in payload
        ):
            saved_config = as_namespace(
                payload["config"]
            )

            if config is None:
                config = saved_config

            else:
                config = merge_missing(
                    config,
                    saved_config,
                )

    if config is None:
        raise RuntimeError(
            "Could not load configuration from "
            f"{args_path} or {results_path}"
        )

    config.device = device

    return config


def extract_state_dict(
    payload: Any,
) -> Dict[str, torch.Tensor]:
    """Extract a model state dictionary from a checkpoint."""

    if isinstance(
        payload,
        Mapping,
    ):
        for key in (
            "model",
            "model_state_dict",
            "state_dict",
        ):
            value = payload.get(
                key
            )

            if isinstance(
                value,
                Mapping,
            ):
                return dict(value)

        if (
            payload
            and all(
                torch.is_tensor(value)
                for value in payload.values()
            )
        ):
            return dict(payload)

    raise RuntimeError(
        "Checkpoint does not contain "
        "a model state_dict."
    )


def checkpoint_metadata(
    path: Path,
    payload: Any,
) -> Dict[str, Any]:
    """Extract checkpoint metadata."""

    output: Dict[
        str,
        Any,
    ] = {
        "file": str(path),
        "name": path.name,
        "step_from_name": checkpoint_step(
            path
        ),
    }

    if isinstance(
        payload,
        Mapping,
    ):
        for key in (
            "step",
            "epoch",
            "state",
        ):
            if key in payload:
                output[key] = jsonify(
                    payload[key]
                )

        state = payload.get(
            "state"
        )

        if isinstance(
            state,
            Mapping,
        ):
            output.setdefault(
                "step",
                state.get(
                    "step",
                    state.get("t"),
                ),
            )

            output.setdefault(
                "epoch",
                state.get("epoch"),
            )

    return output


def reconstruct_model(
    checkpoint_path: Path,
    config: argparse.Namespace,
    init_module: Any,
    device: str,
) -> Tuple[
    torch.nn.Module,
    Dict[str, Any],
]:
    """Reconstruct a model and load one checkpoint."""

    payload = safe_load(
        checkpoint_path
    )

    state_dict = extract_state_dict(
        payload
    )

    checkpoint_config = copy.deepcopy(
        config
    )

    checkpoint_config.device = device

    model = init_module.init_model(
        checkpoint_config
    )

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )

    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint/model mismatch for "
            f"{checkpoint_path.name}: "
            f"missing={missing}, "
            f"unexpected={unexpected}"
        )

    model.to(
        device
    ).eval()

    return (
        model,
        checkpoint_metadata(
            checkpoint_path,
            payload,
        ),
    )


def steps_and_epochs(
    checkpoints: Sequence[Path],
    metadata: Sequence[
        Mapping[str, Any]
    ],
    config: argparse.Namespace,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    """Recover checkpoint steps and corresponding epochs."""

    steps_per_epoch = int(
        getattr(
            config,
            "steps_per_epoch",
            0,
        )
        or 0
    )

    if steps_per_epoch <= 0:
        try:
            steps_per_epoch = (
                int(config.train_size)
                // (
                    int(config.batch_size)
                    * int(config.block_size)
                )
            )

        except Exception:
            steps_per_epoch = 0

    steps: List[int] = []
    epochs: List[float] = []

    for path, entry in zip(
        checkpoints,
        metadata,
    ):
        step = entry.get(
            "step",
            entry.get(
                "step_from_name",
                checkpoint_step(path),
            ),
        )

        try:
            step = int(step)

        except (
            TypeError,
            ValueError,
        ):
            step = -1

        steps.append(step)

        try:
            epoch = float(
                entry.get("epoch")
            )

        except (
            TypeError,
            ValueError,
        ):
            if (
                step >= 0
                and steps_per_epoch > 0
            ):
                epoch = (
                    step
                    / steps_per_epoch
                )

            else:
                epoch = np.nan

        epochs.append(epoch)

    return (
        np.asarray(
            steps,
            dtype=np.int64,
        ),
        np.asarray(
            epochs,
            dtype=float,
        ),
    )


# =============================================================================
# Fixed reference sequences
# =============================================================================


def is_rhm(
    config: argparse.Namespace,
    run_dir: Path,
) -> bool:
    """Determine whether the run uses the RHM dataset."""

    dataset_name = str(
        getattr(
            config,
            "dataset",
            "",
        )
    ).lower()

    return (
        "rhm" in dataset_name
        or (
            run_dir
            / "rules.pt"
        ).exists()
    )


def prepare_sequences(
    array: Any,
    config: argparse.Namespace,
) -> np.ndarray:
    """Convert saved/generated sequences to integer model inputs."""

    if torch.is_tensor(array):
        sequences = (
            array.detach()
            .cpu()
            .numpy()
        )

    else:
        sequences = np.asarray(
            array
        )

    if sequences.ndim == 3:
        vocabulary_size = int(
            config.vocab_size
        )

        if (
            sequences.shape[1]
            == vocabulary_size
        ):
            sequences = sequences.argmax(
                axis=1
            )

        elif (
            sequences.shape[2]
            == vocabulary_size
        ):
            sequences = sequences.argmax(
                axis=2
            )

        else:
            raise ValueError(
                "Cannot infer one-hot layout "
                f"for shape {sequences.shape}"
            )

    if sequences.ndim != 2:
        raise ValueError(
            "Expected sequences with shape "
            f"[N,T], got {sequences.shape}"
        )

    sequences = sequences.astype(
        np.int64,
        copy=False,
    )

    block_size = int(
        config.block_size
    )

    if (
        sequences.shape[1]
        < block_size
    ):
        raise ValueError(
            "Sequence length "
            f"{sequences.shape[1]} is smaller "
            f"than block_size={block_size}"
        )

    sequences = sequences[
        :,
        :block_size,
    ]

    if sequences.size:
        vocabulary_size = int(
            config.vocab_size
        )

        minimum = int(
            sequences.min()
        )

        maximum = int(
            sequences.max()
        )

        # Compatibility with older one-based RHM files.
        if (
            minimum >= 1
            and maximum == vocabulary_size
        ):
            sequences = (
                sequences
                - 1
            )

            minimum = int(
                sequences.min()
            )

            maximum = int(
                sequences.max()
            )

        if (
            minimum < 0
            or maximum >= vocabulary_size
        ):
            raise ValueError(
                "Token IDs outside "
                f"[0,{vocabulary_size - 1}]: "
                f"min={minimum}, max={maximum}"
            )

    return sequences


def choose_rows(
    sequences: np.ndarray,
    requested: int,
    generator: np.random.Generator,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    """Select a deterministic subset of rows."""

    number_sequences = len(
        sequences
    )

    if number_sequences == 0:
        raise ValueError(
            "Reference dataset is empty."
        )

    if (
        requested <= 0
        or requested >= number_sequences
    ):
        indices = np.arange(
            number_sequences,
            dtype=np.int64,
        )

    else:
        indices = np.sort(
            generator.choice(
                number_sequences,
                size=requested,
                replace=False,
            ).astype(np.int64)
        )

    return (
        sequences[indices],
        indices,
    )


def npz_first(
    npz: np.lib.npyio.NpzFile,
    names: Sequence[str],
) -> Optional[np.ndarray]:
    """Return the first matching array from an NPZ file."""

    for name in names:
        if name in npz.files:
            return npz[name]

    return None


def load_saved_rhm(
    run_dir: Path,
    config: argparse.Namespace,
    train_size: int,
    valid_size: int,
    subset_seed: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Dict[str, Any],
]:
    """Load fixed train/validation RHM sequences from dataset_full.npz."""

    candidates = (
        run_dir
        / "data"
        / "dataset_full.npz",

        run_dir
        / "dataset_full.npz",

        run_dir
        / "data"
        / "dataset_reference_subset.npz",
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

    for path in candidates:
        if not path.exists():
            continue

        with np.load(
            path,
            allow_pickle=True,
        ) as npz:
            train_raw = npz_first(
                npz,
                train_names,
            )

            valid_raw = npz_first(
                npz,
                valid_names,
            )

            if (
                train_raw is None
                or valid_raw is None
            ):
                continue

            generator = np.random.default_rng(
                subset_seed
            )

            train_sequences, train_indices = choose_rows(
                prepare_sequences(
                    train_raw,
                    config,
                ),
                train_size,
                generator,
            )

            valid_sequences, valid_indices = choose_rows(
                prepare_sequences(
                    valid_raw,
                    config,
                ),
                valid_size,
                generator,
            )

            metadata = {
                "reference_source": (
                    "saved_rhm_dataset"
                ),
                "reference_detail": str(
                    path
                ),
                "subset_seed": subset_seed,
                "train_indices": (
                    train_indices.tolist()
                ),
                "valid_indices": (
                    valid_indices.tolist()
                ),
            }

            return (
                torch.from_numpy(
                    train_sequences
                ).long(),
                torch.from_numpy(
                    valid_sequences
                ).long(),
                metadata,
            )

    raise FileNotFoundError(
        "No saved RHM dataset_full.npz "
        f"found under {run_dir}"
    )


def find_rules(
    run_dir: Path,
) -> Path:
    """Find the saved RHM rules."""

    candidates = (
        run_dir
        / "rules.pt",

        run_dir
        / "data"
        / "tree_rules.pt",
    )

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "No rules.pt or data/tree_rules.pt "
        f"under {run_dir}"
    )


def normalize_rules(
    value: Any,
) -> Dict[int, torch.Tensor]:
    """Convert saved rules to a sorted dictionary of integer tensors."""

    if isinstance(
        value,
        Mapping,
    ):
        output = {
            int(key): (
                torch.as_tensor(item)
                .detach()
                .cpu()
                .long()
            )
            for key, item in value.items()
        }

    elif isinstance(
        value,
        (list, tuple),
    ):
        output = {
            index: (
                torch.as_tensor(item)
                .detach()
                .cpu()
                .long()
            )
            for index, item in enumerate(value)
        }

    else:
        raise TypeError(
            "Unsupported rules type: "
            f"{type(value)}"
        )

    return {
        key: output[key]
        for key in sorted(output)
    }


def generate_rhm(
    run_dir: Path,
    config: argparse.Namespace,
    train_size: int,
    valid_size: int,
    subset_seed: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Dict[str, Any],
]:
    """Generate fixed RHM reference sequences from saved rules."""

    rules_path = find_rules(
        run_dir
    )

    rules = normalize_rules(
        safe_load(
            rules_path
        )
    )

    generator = (
        torch.Generator(
            device="cpu"
        )
        .manual_seed(
            subset_seed
        )
    )

    number_classes = int(
        getattr(
            config,
            "num_classes",
            rules[
                min(rules)
            ].shape[0],
        )
    )

    labels = torch.randint(
        low=0,
        high=number_classes,
        size=(
            train_size
            + valid_size,
        ),
        generator=generator,
    )

    features = labels.view(
        -1,
        1,
    )

    for level in sorted(rules):
        rule = rules[level]

        selected_rules = torch.randint(
            low=0,
            high=rule.shape[1],
            size=features.shape,
            generator=generator,
        )

        features = rule[
            features,
            selected_rules,
        ].flatten(
            start_dim=1
        )

    sequences = prepare_sequences(
        features,
        config,
    )

    metadata = {
        "reference_source": (
            "generated_from_rules"
        ),
        "reference_detail": str(
            rules_path
        ),
        "subset_seed": subset_seed,
        "train_indices": list(
            range(train_size)
        ),
        "valid_indices": list(
            range(valid_size)
        ),
    }

    return (
        torch.from_numpy(
            sequences[:train_size]
        ).long(),
        torch.from_numpy(
            sequences[train_size:]
        ).long(),
        metadata,
    )


def resolve_text_corpus(
    config: argparse.Namespace,
    split: str,
    repo_dir: Path,
    source_dir: Path,
    run_dir: Path,
) -> Path:
    """Find a non-RHM token corpus."""

    filename = (
        f"{config.dataset}."
        f"{split}.npy"
    )

    configured_path = str(
        getattr(
            config,
            "path",
            "",
        )
    )

    raw_candidates = (
        Path(
            configured_path
            + filename
        ),
        Path(
            configured_path
        ) / filename,
        Path(filename),
    )

    tried: List[str] = []

    for raw_path in raw_candidates:
        if raw_path.is_absolute():
            candidates = (
                raw_path,
            )

        else:
            candidates = tuple(
                base
                / raw_path
                for base in (
                    Path.cwd(),
                    repo_dir,
                    source_dir,
                    run_dir,
                )
            )

        for path in candidates:
            path = (
                path.expanduser()
                .resolve()
            )

            tried.append(
                str(path)
            )

            if path.exists():
                return path

    raise FileNotFoundError(
        f"Could not find {split} corpus. "
        f"Tried: {tried}"
    )


def sample_text_windows(
    path: Path,
    number_windows: int,
    block_size: int,
    subset_seed: int,
) -> Tuple[
    torch.Tensor,
    np.ndarray,
]:
    """Sample fixed contiguous windows from a text corpus."""

    corpus = np.load(
        path,
        mmap_mode="r",
    ).reshape(-1)

    maximum_start = (
        len(corpus)
        - block_size
    )

    if maximum_start < 0:
        raise ValueError(
            f"Corpus {path} is shorter "
            f"than block_size={block_size}"
        )

    generator = np.random.default_rng(
        subset_seed
    )

    starts = np.sort(
        generator.choice(
            maximum_start + 1,
            number_windows,
            replace=(
                number_windows
                > maximum_start + 1
            ),
        )
    )

    windows = np.stack(
        [
            np.asarray(
                corpus[
                    start:
                    start + block_size
                ],
                dtype=np.int64,
            )
            for start in starts
        ]
    )

    return (
        torch.from_numpy(
            windows
        ).long(),
        starts.astype(
            np.int64
        ),
    )


def load_references(
    config: argparse.Namespace,
    cli: argparse.Namespace,
    repo_dir: Path,
    source_dir: Path,
    run_dir: Path,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Dict[str, Any],
]:
    """Load or generate the fixed representation reference sequences."""

    train_size = int(
        cli.subset_train_size
    )

    valid_size = int(
        cli.subset_valid_size
    )

    subset_seed = int(
        cli.subset_seed
    )

    if (
        train_size <= 0
        or valid_size <= 0
    ):
        raise ValueError(
            "subset sizes must be positive."
        )

    if is_rhm(
        config,
        run_dir,
    ):
        reference_source = (
            cli.reference_source
        )

        online = bool(
            getattr(
                config,
                "online",
                False,
            )
        )

        if (
            reference_source
            == "generate"
            or (
                reference_source
                == "auto"
                and online
            )
        ):
            return generate_rhm(
                run_dir,
                config,
                train_size,
                valid_size,
                subset_seed,
            )

        try:
            return load_saved_rhm(
                run_dir,
                config,
                train_size,
                valid_size,
                subset_seed,
            )

        except FileNotFoundError:
            if (
                reference_source
                == "saved"
            ):
                raise

            return generate_rhm(
                run_dir,
                config,
                train_size,
                valid_size,
                subset_seed,
            )

    train_path = resolve_text_corpus(
        config,
        "train",
        repo_dir,
        source_dir,
        run_dir,
    )

    valid_path = resolve_text_corpus(
        config,
        "valid",
        repo_dir,
        source_dir,
        run_dir,
    )

    train_sequences, train_starts = sample_text_windows(
        train_path,
        train_size,
        int(config.block_size),
        subset_seed,
    )

    valid_sequences, valid_starts = sample_text_windows(
        valid_path,
        valid_size,
        int(config.block_size),
        subset_seed + 1,
    )

    metadata = {
        "reference_source": (
            "text_corpus_windows"
        ),
        "reference_detail": {
            "train": str(
                train_path
            ),
            "valid": str(
                valid_path
            ),
        },
        "subset_seed": subset_seed,
        "train_indices": (
            train_starts.tolist()
        ),
        "valid_indices": (
            valid_starts.tolist()
        ),
    }

    return (
        train_sequences,
        valid_sequences,
        metadata,
    )


# =============================================================================
# Representations and CKA
# =============================================================================


def architecture(
    model: torch.nn.Module,
) -> str:
    """Identify Transformer or Mamba."""

    if (
        hasattr(
            model,
            "blocks",
        )
        and hasattr(
            model,
            "token_embedding_table",
        )
    ):
        return "transformer"

    if (
        hasattr(
            model,
            "layers",
        )
        and hasattr(
            model,
            "embedding",
        )
    ):
        return "mamba"

    raise TypeError(
        "Expected Transformer CLM "
        "or MambaLM."
    )


def representation_names(
    model: torch.nn.Module,
    representation_mode: str,
    include_embedding: bool,
    include_final_norm: bool,
) -> List[str]:
    """Return representation names in extraction order."""

    model_architecture = architecture(
        model
    )

    if (
        model_architecture
        == "transformer"
    ):
        layers = model.blocks
        stem = "block"

    else:
        layers = model.layers
        stem = "layer"

    if (
        representation_mode
        == "block_update"
    ):
        return [
            f"{stem}_update_{index + 1}"
            for index in range(
                len(layers)
            )
        ]

    names: List[str] = []

    if include_embedding:
        names.append(
            "embedding"
        )

    names.extend(
        f"{stem}_{index + 1}"
        for index in range(
            len(layers)
        )
    )

    if include_final_norm:
        names.append(
            "final_norm"
        )

    return names


@torch.no_grad()
def forward_representations(
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    representation_mode: str,
    include_embedding: bool,
    include_final_norm: bool,
) -> List[torch.Tensor]:
    """Extract layer representations for one batch."""

    model_architecture = architecture(
        model
    )

    if (
        model_architecture
        == "transformer"
    ):
        hidden = model.token_embedding_table(
            token_ids
        )

        if hasattr(
            model,
            "position_embedding_table",
        ):
            positions = torch.arange(
                token_ids.shape[1],
                device=token_ids.device,
            )

            hidden = (
                hidden
                + model.position_embedding_table(
                    positions
                )
            )

        layers = model.blocks
        final_norm = model.ln_f

    else:
        hidden = model.embedding(
            token_ids
        )

        layers = model.layers
        final_norm = model.norm_f

    outputs: List[
        torch.Tensor
    ] = []

    if (
        representation_mode
        == "state"
    ):
        if include_embedding:
            outputs.append(
                hidden.detach()
                .float()
                .cpu()
            )

        for layer in layers:
            hidden = layer(
                hidden
            )

            outputs.append(
                hidden.detach()
                .float()
                .cpu()
            )

        if include_final_norm:
            outputs.append(
                final_norm(
                    hidden
                )
                .detach()
                .float()
                .cpu()
            )

    else:
        for layer in layers:
            next_hidden = layer(
                hidden
            )

            outputs.append(
                (
                    next_hidden
                    - hidden
                )
                .detach()
                .float()
                .cpu()
            )

            hidden = next_hidden

    return outputs


def temporary_path(
    root: Path,
    split: str,
    metric: str,
    checkpoint_index: int,
    layer_index: int,
    position_index: int,
) -> Path:
    """Return a temporary representation filename."""

    return (
        root
        / split
        / (
            f"{metric}_"
            f"k{checkpoint_index:05d}_"
            f"l{layer_index:03d}_"
            f"p{position_index:04d}.npy"
        )
    )


def frobenius_norm(
    array: np.ndarray,
) -> float:
    """Stable Frobenius norm."""

    array64 = array.astype(
        np.float64,
        copy=False,
    )

    value = float(
        np.sqrt(
            np.sum(
                array64
                * array64
            )
        )
    )

    if (
        np.isfinite(value)
        and value > EPS
    ):
        return value

    return 0.0


def save_representation_temp(
    path: Path,
    centered_hidden: np.ndarray,
    metric: str,
) -> float:
    """Save the temporary matrix used for CKA."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if metric == "gram":
        gram = (
            centered_hidden
            @ centered_hidden.T
        )

        norm = frobenius_norm(
            gram
        )

        if norm > 0:
            gram = (
                gram
                / norm
            )

        else:
            gram = np.zeros_like(
                gram
            )

        np.save(
            path,
            gram.astype(
                np.float32
            ),
        )

        return norm

    covariance = (
        centered_hidden.T
        @ centered_hidden
    )

    norm = frobenius_norm(
        covariance
    )

    np.save(
        path,
        centered_hidden.astype(
            np.float32,
            copy=False,
        ),
    )

    return norm


@torch.no_grad()
def store_checkpoint_representations(
    checkpoint_path: Path,
    checkpoint_index: int,
    config: argparse.Namespace,
    init_module: Any,
    references: Mapping[
        str,
        torch.Tensor,
    ],
    cli: argparse.Namespace,
    temp_dir: Path,
) -> Tuple[
    List[str],
    Dict[str, np.ndarray],
    Dict[str, Any],
    str,
]:
    """Extract and store representations for one checkpoint."""

    model, metadata = reconstruct_model(
        checkpoint_path,
        config,
        init_module,
        cli.device,
    )

    layer_names = representation_names(
        model,
        cli.representation_mode,
        cli.include_embedding,
        cli.include_final_norm,
    )

    model_architecture = architecture(
        model
    )

    norms: Dict[
        str,
        np.ndarray,
    ] = {}

    for split, sequences in references.items():
        loader = DataLoader(
            TensorDataset(
                sequences
            ),
            batch_size=cli.batch_size,
            shuffle=False,
            num_workers=0,
        )

        chunks: Optional[
            List[
                List[
                    List[torch.Tensor]
                ]
            ]
        ] = None

        for (batch,) in loader:
            representations = forward_representations(
                model,
                batch.to(
                    cli.device
                ),
                cli.representation_mode,
                cli.include_embedding,
                cli.include_final_norm,
            )

            if chunks is None:
                chunks = [
                    [
                        []
                        for _ in range(
                            representation.shape[1]
                        )
                    ]
                    for representation in representations
                ]

            for layer_index, representation in enumerate(
                representations
            ):
                for position_index in range(
                    representation.shape[1]
                ):
                    chunks[
                        layer_index
                    ][
                        position_index
                    ].append(
                        representation[
                            :,
                            position_index,
                        ].contiguous()
                    )

        if chunks is None:
            raise RuntimeError(
                "No batches generated for "
                f"split={split}"
            )

        split_norms = np.full(
            (
                len(chunks),
                len(chunks[0]),
            ),
            np.nan,
            dtype=np.float64,
        )

        for layer_index, positions in enumerate(
            chunks
        ):
            for position_index, pieces in enumerate(
                positions
            ):
                hidden = torch.cat(
                    pieces
                ).float()

                hidden = (
                    hidden
                    - hidden.mean(
                        dim=0,
                        keepdim=True,
                    )
                )

                split_norms[
                    layer_index,
                    position_index,
                ] = save_representation_temp(
                    temporary_path(
                        temp_dir,
                        split,
                        cli.metric,
                        checkpoint_index,
                        layer_index,
                        position_index,
                    ),
                    hidden.numpy(),
                    cli.metric,
                )

        norms[split] = split_norms

    del model

    if (
        cli.device.startswith("cuda")
        and torch.cuda.is_available()
    ):
        torch.cuda.empty_cache()

    return (
        layer_names,
        norms,
        metadata,
        model_architecture,
    )


def pairwise_cka(
    temp_dir: Path,
    split: str,
    metric: str,
    self_norms: np.ndarray,
    number_checkpoints: int,
    number_layers: int,
    number_positions: int,
) -> np.ndarray:
    """Compute all checkpoint-to-checkpoint CKA matrices."""

    output = np.zeros(
        (
            number_positions,
            number_layers,
            number_checkpoints,
            number_checkpoints,
        ),
        dtype=np.float32,
    )

    for position_index in range(
        number_positions
    ):
        for layer_index in range(
            number_layers
        ):
            print(
                "[INFO] CKA "
                f"split={split} "
                f"position={position_index} "
                f"layer={layer_index}",
                flush=True,
            )

            for first_checkpoint in range(
                number_checkpoints
            ):
                first = np.load(
                    temporary_path(
                        temp_dir,
                        split,
                        metric,
                        first_checkpoint,
                        layer_index,
                        position_index,
                    ),
                    mmap_mode="r",
                )

                for second_checkpoint in range(
                    first_checkpoint,
                    number_checkpoints,
                ):
                    second = np.load(
                        temporary_path(
                            temp_dir,
                            split,
                            metric,
                            second_checkpoint,
                            layer_index,
                            position_index,
                        ),
                        mmap_mode="r",
                    )

                    if metric == "gram":
                        value = float(
                            np.sum(
                                first.astype(
                                    np.float64,
                                    copy=False,
                                )
                                * second.astype(
                                    np.float64,
                                    copy=False,
                                )
                            )
                        )

                    else:
                        first_norm = self_norms[
                            first_checkpoint,
                            layer_index,
                            position_index,
                        ]

                        second_norm = self_norms[
                            second_checkpoint,
                            layer_index,
                            position_index,
                        ]

                        if (
                            first_norm <= EPS
                            or second_norm <= EPS
                        ):
                            value = 0.0

                        else:
                            cross_covariance = (
                                first.T.astype(
                                    np.float64,
                                    copy=False,
                                )
                                @ second.astype(
                                    np.float64,
                                    copy=False,
                                )
                            )

                            value = float(
                                np.sum(
                                    cross_covariance
                                    * cross_covariance
                                )
                                / (
                                    first_norm
                                    * second_norm
                                )
                            )

                    value = float(
                        np.clip(
                            value,
                            0.0,
                            1.0,
                        )
                    )

                    output[
                        position_index,
                        layer_index,
                        first_checkpoint,
                        second_checkpoint,
                    ] = value

                    output[
                        position_index,
                        layer_index,
                        second_checkpoint,
                        first_checkpoint,
                    ] = value

    return output


def compute_representation_overlaps(
    checkpoints: Sequence[Path],
    config: argparse.Namespace,
    init_module: Any,
    train_reference: torch.Tensor,
    valid_reference: torch.Tensor,
    cli: argparse.Namespace,
    temp_dir: Path,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    List[str],
    List[Dict[str, Any]],
    Dict[str, np.ndarray],
    str,
]:
    """Compute representation overlaps for all checkpoints."""

    references = {
        "train": train_reference,
        "valid": valid_reference,
    }

    layer_names_reference: Optional[
        List[str]
    ] = None

    architecture_reference: Optional[
        str
    ] = None

    checkpoint_metadata_list: List[
        Dict[str, Any]
    ] = []

    norm_lists: Dict[
        str,
        List[np.ndarray],
    ] = {
        "train": [],
        "valid": [],
    }

    for checkpoint_index, checkpoint_path in enumerate(
        checkpoints
    ):
        print(
            "[INFO] extracting representations "
            f"{checkpoint_index + 1}/"
            f"{len(checkpoints)}: "
            f"{checkpoint_path.name}",
            flush=True,
        )

        (
            layer_names,
            norms,
            metadata,
            model_architecture,
        ) = store_checkpoint_representations(
            checkpoint_path,
            checkpoint_index,
            config,
            init_module,
            references,
            cli,
            temp_dir,
        )

        if layer_names_reference is None:
            layer_names_reference = (
                layer_names
            )

            architecture_reference = (
                model_architecture
            )

        elif (
            layer_names
            != layer_names_reference
            or model_architecture
            != architecture_reference
        ):
            raise RuntimeError(
                "Architecture or layer names "
                "changed between checkpoints."
            )

        for split in norm_lists:
            norm_lists[
                split
            ].append(
                norms[split]
            )

        checkpoint_metadata_list.append(
            metadata
        )

    if (
        layer_names_reference is None
        or architecture_reference is None
    ):
        raise RuntimeError(
            "No checkpoint was processed."
        )

    self_norms = {
        split: np.stack(values)
        for split, values in norm_lists.items()
    }

    number_positions = int(
        train_reference.shape[1]
    )

    train_cka = pairwise_cka(
        temp_dir,
        "train",
        cli.metric,
        self_norms["train"],
        len(checkpoints),
        len(layer_names_reference),
        number_positions,
    )

    valid_cka = pairwise_cka(
        temp_dir,
        "valid",
        cli.metric,
        self_norms["valid"],
        len(checkpoints),
        len(layer_names_reference),
        number_positions,
    )

    return (
        train_cka,
        valid_cka,
        layer_names_reference,
        checkpoint_metadata_list,
        self_norms,
        architecture_reference,
    )



# =============================================================================
# Predictive linear-readout overlaps
# =============================================================================


def parse_ridge_alphas(value: str) -> np.ndarray:
    """Parse a comma-separated ridge-alpha grid."""

    alphas = []

    for item in str(value).split(","):
        item = item.strip()

        if not item:
            continue

        alpha = float(item)

        if alpha < 0:
            raise ValueError(
                "Ridge alphas must be non-negative, "
                f"got {alpha}."
            )

        alphas.append(alpha)

    if not alphas:
        raise ValueError("Empty ridge-alpha grid.")

    return np.asarray(
        alphas,
        dtype=np.float64,
    )


def predictive_representation_path(
    root: Path,
    role: str,
    checkpoint_index: int,
    layer_index: int,
    position_index: int,
) -> Path:
    """Return a temporary raw-representation filename for predictive overlap."""

    return (
        root
        / "predictive"
        / role
        / (
            f"repr_k{checkpoint_index:05d}_"
            f"l{layer_index:03d}_"
            f"p{position_index:04d}.npy"
        )
    )


@torch.no_grad()
def store_predictive_checkpoint_representations(
    checkpoint_path: Path,
    checkpoint_index: int,
    role: str,
    config: argparse.Namespace,
    init_module: Any,
    reference: torch.Tensor,
    cli: argparse.Namespace,
    temp_dir: Path,
) -> Tuple[
    List[str],
    Dict[str, Any],
    str,
    int,
]:
    """Extract raw representations for one checkpoint and one reference set."""

    model, metadata = reconstruct_model(
        checkpoint_path,
        config,
        init_module,
        cli.device,
    )

    layer_names = representation_names(
        model,
        cli.representation_mode,
        cli.include_embedding,
        cli.include_final_norm,
    )

    model_architecture = architecture(
        model
    )

    loader = DataLoader(
        TensorDataset(
            reference
        ),
        batch_size=cli.batch_size,
        shuffle=False,
        num_workers=0,
    )

    chunks: Optional[
        List[
            List[
                List[torch.Tensor]
            ]
        ]
    ] = None

    for (batch,) in loader:
        representations = forward_representations(
            model,
            batch.to(
                cli.device
            ),
            cli.representation_mode,
            cli.include_embedding,
            cli.include_final_norm,
        )

        if chunks is None:
            chunks = [
                [
                    []
                    for _ in range(
                        representation.shape[1]
                    )
                ]
                for representation in representations
            ]

        for layer_index, representation in enumerate(
            representations
        ):
            for position_index in range(
                representation.shape[1]
            ):
                chunks[
                    layer_index
                ][
                    position_index
                ].append(
                    representation[
                        :,
                        position_index,
                    ].contiguous()
                )

    if chunks is None:
        raise RuntimeError(
            "No batches generated for predictive "
            f"role={role}."
        )

    for layer_index, positions in enumerate(
        chunks
    ):
        for position_index, pieces in enumerate(
            positions
        ):
            hidden = torch.cat(
                pieces
            ).float().cpu().numpy()

            path = predictive_representation_path(
                temp_dir,
                role,
                checkpoint_index,
                layer_index,
                position_index,
            )

            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            np.save(
                path,
                hidden.astype(
                    np.float32,
                    copy=False,
                ),
            )

    number_positions = len(
        chunks[0]
    )

    del model

    if (
        cli.device.startswith("cuda")
        and torch.cuda.is_available()
    ):
        torch.cuda.empty_cache()

    return (
        layer_names,
        metadata,
        model_architecture,
        number_positions,
    )


def make_cv_folds(
    number_samples: int,
    number_folds: int,
    seed: int,
) -> List[np.ndarray]:
    """Create deterministic shuffled folds for readout cross-validation."""

    number_samples = int(
        number_samples
    )

    number_folds = int(
        number_folds
    )

    if number_folds < 2:
        raise ValueError(
            "predictive_num_folds must be at least 2."
        )

    if number_samples < number_folds:
        raise ValueError(
            "Need at least one sample per fold: "
            f"number_samples={number_samples}, "
            f"number_folds={number_folds}."
        )

    generator = np.random.default_rng(
        int(seed)
    )

    permutation = generator.permutation(
        number_samples
    )

    return [
        fold.astype(
            np.int64,
            copy=False,
        )
        for fold in np.array_split(
            permutation,
            number_folds,
        )
    ]


def _standardize_train_test(
    train: np.ndarray,
    test: np.ndarray,
    *,
    standardize: bool,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Center, and optionally z-score, a predictor matrix by train stats."""

    mean = train.mean(
        axis=0,
        keepdims=True,
    )

    train_out = train - mean
    test_out = test - mean

    if standardize:
        std = train_out.std(
            axis=0,
            keepdims=True,
        )

        std = np.where(
            std > EPS,
            std,
            1.0,
        )

        train_out = train_out / std
        test_out = test_out / std

    return (
        train_out,
        test_out,
        mean,
    )


def ridge_predictive_ev_cv(
    x: np.ndarray,
    y: np.ndarray,
    folds: Sequence[np.ndarray],
    alphas: np.ndarray,
    *,
    standardize_x: bool,
) -> Tuple[
    float,
    float,
    np.ndarray,
    np.ndarray,
]:
    """Five-fold RidgeCV-style normalized explained variance.

    For each ridge value, a linear map W is learned on K-1 folds and evaluated
    on the held-out fold.  The selected alpha is the one with the largest
    aggregated held-out normalized explained variance.  The final score is not
    clipped; values below zero mean worse than the train-mean baseline.
    """

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(
            f"Expected 2D X and Y, got {x.shape} and {y.shape}."
        )

    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and Y must have the same number of samples, "
            f"got {x.shape[0]} and {y.shape[0]}."
        )

    number_samples = x.shape[0]
    all_indices = np.arange(
        number_samples,
        dtype=np.int64,
    )

    fold_scores = np.full(
        (
            len(alphas),
            len(folds),
        ),
        np.nan,
        dtype=np.float64,
    )

    total_numerators = np.zeros(
        len(alphas),
        dtype=np.float64,
    )

    total_denominators = np.zeros(
        len(alphas),
        dtype=np.float64,
    )

    for fold_index, test_index in enumerate(
        folds
    ):
        test_mask = np.zeros(
            number_samples,
            dtype=bool,
        )

        test_mask[
            test_index
        ] = True

        train_index = all_indices[
            ~test_mask
        ]

        x_train = x[
            train_index
        ]

        x_test = x[
            test_index
        ]

        y_train = y[
            train_index
        ]

        y_test = y[
            test_index
        ]

        x_train, x_test, _ = _standardize_train_test(
            x_train,
            x_test,
            standardize=standardize_x,
        )

        y_mean = y_train.mean(
            axis=0,
            keepdims=True,
        )

        y_train_centered = y_train - y_mean
        y_test_centered = y_test - y_mean

        denominator = float(
            np.sum(
                y_test_centered
                * y_test_centered
            )
        )

        if denominator <= EPS:
            continue

        xtx = (
            x_train.T
            @ x_train
        )

        xty = (
            x_train.T
            @ y_train_centered
        )

        identity = np.eye(
            xtx.shape[0],
            dtype=np.float64,
        )

        for alpha_index, alpha in enumerate(
            alphas
        ):
            system = xtx + float(alpha) * identity

            try:
                weights = np.linalg.solve(
                    system,
                    xty,
                )

            except np.linalg.LinAlgError:
                weights = np.linalg.pinv(
                    system,
                    rcond=1e-10,
                ) @ xty

            prediction_centered = (
                x_test
                @ weights
            )

            residual = (
                y_test_centered
                - prediction_centered
            )

            numerator = float(
                np.sum(
                    residual
                    * residual
                )
            )

            fold_score = 1.0 - numerator / denominator

            fold_scores[
                alpha_index,
                fold_index,
            ] = fold_score

            total_numerators[
                alpha_index
            ] += numerator

            total_denominators[
                alpha_index
            ] += denominator

    scores_by_alpha = np.full(
        len(alphas),
        np.nan,
        dtype=np.float64,
    )

    valid = total_denominators > EPS

    scores_by_alpha[
        valid
    ] = 1.0 - (
        total_numerators[
            valid
        ]
        / total_denominators[
            valid
        ]
    )

    if not np.any(
        np.isfinite(
            scores_by_alpha
        )
    ):
        return (
            float("nan"),
            float("nan"),
            np.full(
                len(folds),
                np.nan,
                dtype=np.float64,
            ),
            scores_by_alpha,
        )

    best_alpha_index = int(
        np.nanargmax(
            scores_by_alpha
        )
    )

    return (
        float(
            scores_by_alpha[
                best_alpha_index
            ]
        ),
        float(
            alphas[
                best_alpha_index
            ]
        ),
        fold_scores[
            best_alpha_index
        ].copy(),
        scores_by_alpha,
    )


def resolve_predictive_target_index(
    checkpoints: Sequence[Path],
    cli: argparse.Namespace,
) -> int:
    """Resolve the single target checkpoint t'."""

    if not checkpoints:
        raise ValueError("No checkpoints available.")

    if cli.target_step is not None:
        matches = [
            index
            for index, path in enumerate(
                checkpoints
            )
            if checkpoint_step(path) == int(
                cli.target_step
            )
        ]

        if not matches:
            available = [
                checkpoint_step(path)
                for path in checkpoints
            ]

            raise ValueError(
                f"target_step={cli.target_step} not found. "
                f"Available steps include {available[:10]}..."
            )

        return matches[
            -1
        ]

    target_index = int(
        cli.target_index
    )

    if target_index < 0:
        target_index = len(
            checkpoints
        ) + target_index

    if target_index < 0 or target_index >= len(
        checkpoints
    ):
        raise ValueError(
            f"target_index={cli.target_index} is outside "
            f"[0,{len(checkpoints)-1}]."
        )

    return target_index


def predictive_source_indices(
    number_checkpoints: int,
    target_index: int,
    take_every: int,
    max_sources: int,
) -> np.ndarray:
    """Select source times t every take_every checkpoints, always including t'."""

    take_every = int(
        take_every
    )

    if take_every <= 0:
        raise ValueError(
            f"take_every must be positive, got {take_every}."
        )

    selected = list(
        range(
            0,
            number_checkpoints,
            take_every,
        )
    )

    selected.append(
        int(target_index)
    )

    selected = sorted(
        set(selected)
    )

    max_sources = int(
        max_sources
    )

    if (
        max_sources > 0
        and len(selected) > max_sources
    ):
        sub = log_select_indices(
            len(selected),
            max_sources,
        )

        selected = [
            selected[index]
            for index in sub
        ]

        if target_index not in selected:
            selected[-1] = int(
                target_index
            )

        selected = sorted(
            set(selected)
        )

    return np.asarray(
        selected,
        dtype=np.int64,
    )


def compute_predictive_overlaps(
    source_checkpoints: Sequence[Path],
    target_checkpoint: Path,
    config: argparse.Namespace,
    init_module: Any,
    readout_reference: torch.Tensor,
    cli: argparse.Namespace,
    temp_dir: Path,
) -> Tuple[
    Dict[str, Any],
    List[Dict[str, Any]],
    Dict[str, Any],
    str,
]:
    """Compute directed predictive overlaps H(t,l) -> H(t',l')."""

    alphas = parse_ridge_alphas(
        cli.ridge_alphas
    )

    folds = make_cv_folds(
        int(
            readout_reference.shape[0]
        ),
        int(
            cli.predictive_num_folds
        ),
        int(
            cli.subset_seed
        )
        + 7919,
    )

    print(
        "[INFO] extracting predictive target representations "
        f"t'={target_checkpoint.name}",
        flush=True,
    )

    (
        target_layer_names,
        target_metadata,
        architecture_reference,
        number_positions,
    ) = store_predictive_checkpoint_representations(
        target_checkpoint,
        0,
        "target",
        config,
        init_module,
        readout_reference,
        cli,
        temp_dir,
    )

    source_metadata_list: List[
        Dict[str, Any]
    ] = []

    source_layer_names_reference: Optional[
        List[str]
    ] = None

    for source_index, checkpoint_path in enumerate(
        source_checkpoints
    ):
        print(
            "[INFO] extracting predictive source representations "
            f"{source_index + 1}/{len(source_checkpoints)}: "
            f"{checkpoint_path.name}",
            flush=True,
        )

        (
            source_layer_names,
            source_metadata,
            source_architecture,
            source_number_positions,
        ) = store_predictive_checkpoint_representations(
            checkpoint_path,
            source_index,
            "source",
            config,
            init_module,
            readout_reference,
            cli,
            temp_dir,
        )

        if source_layer_names_reference is None:
            source_layer_names_reference = source_layer_names

        elif source_layer_names != source_layer_names_reference:
            raise RuntimeError(
                "Source layer names changed between checkpoints."
            )

        if (
            source_architecture != architecture_reference
            or source_number_positions != number_positions
        ):
            raise RuntimeError(
                "Architecture or number of positions changed "
                "between source and target checkpoints."
            )

        source_metadata_list.append(
            source_metadata
        )

    if source_layer_names_reference is None:
        raise RuntimeError(
            "No source checkpoint was processed."
        )

    number_source_layers = len(
        source_layer_names_reference
    )

    number_target_layers = len(
        target_layer_names
    )

    number_sources = len(
        source_checkpoints
    )

    q = np.full(
        (
            number_positions,
            number_source_layers,
            number_target_layers,
            number_sources,
        ),
        np.nan,
        dtype=np.float32,
    )

    best_alpha = np.full_like(
        q,
        np.nan,
        dtype=np.float32,
    )

    fold_q = np.full(
        (
            number_positions,
            number_source_layers,
            number_target_layers,
            number_sources,
            len(folds),
        ),
        np.nan,
        dtype=np.float32,
    )

    alpha_scores = np.full(
        (
            number_positions,
            number_source_layers,
            number_target_layers,
            number_sources,
            len(alphas),
        ),
        np.nan,
        dtype=np.float32,
    )

    if cli.predictive_layer_mode == "same":
        layer_pairs = [
            (
                layer_index,
                layer_index,
            )
            for layer_index in range(
                min(
                    number_source_layers,
                    number_target_layers,
                )
            )
        ]

    elif cli.predictive_layer_mode == "all":
        layer_pairs = [
            (
                source_layer_index,
                target_layer_index,
            )
            for source_layer_index in range(
                number_source_layers
            )
            for target_layer_index in range(
                number_target_layers
            )
        ]

    else:
        raise ValueError(
            f"Unknown predictive_layer_mode={cli.predictive_layer_mode}."
        )

    for position_index in range(
        number_positions
    ):
        for source_index in range(
            number_sources
        ):
            source_cache: Dict[
                int,
                np.ndarray,
            ] = {}

            target_cache: Dict[
                int,
                np.ndarray,
            ] = {}

            for pair_counter, (source_layer_index, target_layer_index) in enumerate(
                layer_pairs
            ):
                print(
                    "[INFO] predictive ridge "
                    f"position={position_index} "
                    f"source_time={source_index + 1}/{number_sources} "
                    f"source_layer={source_layer_index} "
                    f"target_layer={target_layer_index} "
                    f"pair={pair_counter + 1}/{len(layer_pairs)}",
                    flush=True,
                )

                if source_layer_index not in source_cache:
                    source_cache[
                        source_layer_index
                    ] = np.load(
                        predictive_representation_path(
                            temp_dir,
                            "source",
                            source_index,
                            source_layer_index,
                            position_index,
                        ),
                        mmap_mode="r",
                    ).astype(
                        np.float64,
                        copy=False,
                    )

                if target_layer_index not in target_cache:
                    target_cache[
                        target_layer_index
                    ] = np.load(
                        predictive_representation_path(
                            temp_dir,
                            "target",
                            0,
                            target_layer_index,
                            position_index,
                        ),
                        mmap_mode="r",
                    ).astype(
                        np.float64,
                        copy=False,
                    )

                value, alpha, fold_values, alpha_values = ridge_predictive_ev_cv(
                    source_cache[
                        source_layer_index
                    ],
                    target_cache[
                        target_layer_index
                    ],
                    folds,
                    alphas,
                    standardize_x=bool(
                        cli.predictive_standardize_x
                    ),
                )

                q[
                    position_index,
                    source_layer_index,
                    target_layer_index,
                    source_index,
                ] = value

                best_alpha[
                    position_index,
                    source_layer_index,
                    target_layer_index,
                    source_index,
                ] = alpha

                fold_q[
                    position_index,
                    source_layer_index,
                    target_layer_index,
                    source_index,
                    :,
                ] = fold_values.astype(
                    np.float32,
                    copy=False,
                )

                alpha_scores[
                    position_index,
                    source_layer_index,
                    target_layer_index,
                    source_index,
                    :,
                ] = alpha_values.astype(
                    np.float32,
                    copy=False,
                )

    result = {
        "predictive_q_by_position": q,
        "predictive_q_position_mean": np.nanmean(
            q,
            axis=0,
        ),
        "predictive_best_alpha_by_position": best_alpha,
        "predictive_best_alpha_position_median": np.nanmedian(
            best_alpha,
            axis=0,
        ),
        "predictive_fold_q_by_position": fold_q,
        "predictive_alpha_scores_by_position": alpha_scores,
        "ridge_alphas": alphas.astype(
            np.float64,
            copy=False,
        ),
        "source_layer_names": np.asarray(
            source_layer_names_reference,
            dtype=object,
        ),
        "target_layer_names": np.asarray(
            target_layer_names,
            dtype=object,
        ),
        "fold_sizes": np.asarray(
            [
                len(fold)
                for fold in folds
            ],
            dtype=np.int64,
        ),
    }

    return (
        result,
        source_metadata_list,
        target_metadata,
        architecture_reference,
    )


# =============================================================================
# Weight overlaps
# =============================================================================


def blocked_dot(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """Compute a float64 dot product in memory-sized blocks."""

    total = 0.0

    for start in range(
        0,
        first.size,
        DOT_BLOCK,
    ):
        stop = min(
            start + DOT_BLOCK,
            first.size,
        )

        total += float(
            np.dot(
                first[
                    start:stop
                ].astype(
                    np.float64,
                    copy=False,
                ),
                second[
                    start:stop
                ].astype(
                    np.float64,
                    copy=False,
                ),
            )
        )

    return total


def raw_dot_matrix(
    vectors: Sequence[np.ndarray],
) -> np.ndarray:
    """Compute the pairwise raw dot matrix."""

    number_vectors = len(
        vectors
    )

    output = np.zeros(
        (
            number_vectors,
            number_vectors,
        ),
        dtype=np.float64,
    )

    for first_index in range(
        number_vectors
    ):
        for second_index in range(
            first_index,
            number_vectors,
        ):
            value = blocked_dot(
                vectors[first_index],
                vectors[second_index],
            )

            output[
                first_index,
                second_index,
            ] = value

            output[
                second_index,
                first_index,
            ] = value

    return output


def cosine_from_raw(
    raw_dot: np.ndarray,
) -> np.ndarray:
    """Normalize a raw dot matrix into cosine overlaps."""

    diagonal = np.maximum(
        np.diag(raw_dot),
        0.0,
    )

    denominator = np.sqrt(
        diagonal[:, None]
        * diagonal[None, :]
    )

    output = np.zeros_like(
        raw_dot
    )

    valid = denominator > EPS

    output[valid] = (
        raw_dot[valid]
        / denominator[valid]
    )

    return np.clip(
        output,
        -1.0,
        1.0,
    ).astype(
        np.float32
    )


def classify_parameter(
    name: str,
    model_architecture: str,
) -> Tuple[
    str,
    Optional[int],
    Optional[str],
]:
    """Assign a parameter to generic, layer and component groups."""

    if (
        model_architecture
        == "transformer"
    ):
        top_level_groups = (
            (
                "token_embedding_table.",
                "token_embedding",
            ),
            (
                "position_embedding_table.",
                "position_embedding",
            ),
            (
                "ln_f.",
                "final_norm",
            ),
            (
                "lm_head.",
                "lm_head",
            ),
        )

        for prefix, group in top_level_groups:
            if name.startswith(prefix):
                return (
                    group,
                    None,
                    None,
                )

        match = re.match(
            r"blocks\.(\d+)\.(.+)",
            name,
        )

        if match:
            layer_index = int(
                match.group(1)
            )

            remainder = match.group(2)

            if remainder.startswith(
                "attn."
            ):
                component = "attention"

            elif remainder.startswith(
                "ffwd."
            ):
                component = "mlp"

            elif remainder.startswith(
                (
                    "ln1.",
                    "ln2.",
                )
            ):
                component = "norm"

            else:
                component = "other"

            return (
                f"layer_{layer_index + 1}",
                layer_index,
                component,
            )

    else:
        top_level_groups = (
            (
                "embedding.",
                "token_embedding",
            ),
            (
                "norm_f.",
                "final_norm",
            ),
            (
                "lm_head.",
                "lm_head",
            ),
        )

        for prefix, group in top_level_groups:
            if name.startswith(prefix):
                return (
                    group,
                    None,
                    None,
                )

        match = re.match(
            r"layers\.(\d+)\.(.+)",
            name,
        )

        if match:
            layer_index = int(
                match.group(1)
            )

            remainder = match.group(2)

            if remainder.startswith(
                "mamba."
            ):
                component = "mixer"

            elif remainder.startswith(
                "norm."
            ):
                component = "norm"

            else:
                component = "other"

            return (
                f"layer_{layer_index + 1}",
                layer_index,
                component,
            )

    return (
        "other",
        None,
        None,
    )


def stack_cosines(
    raw_matrices: Mapping[
        str,
        np.ndarray,
    ],
    names: Sequence[str],
    number_checkpoints: int,
) -> np.ndarray:
    """Normalize and stack a collection of raw dot matrices."""

    if not names:
        return np.empty(
            (
                0,
                number_checkpoints,
                number_checkpoints,
            ),
            dtype=np.float32,
        )

    return np.stack(
        [
            cosine_from_raw(
                raw_matrices[name]
            )
            for name in names
        ]
    )


def compute_weight_overlaps(
    checkpoints: Sequence[Path],
    config: argparse.Namespace,
    init_module: Any,
) -> Tuple[
    Dict[str, Any],
    List[Dict[str, Any]],
    str,
]:
    """Compute all requested weight-overlap matrices."""

    cpu_config = copy.deepcopy(
        config
    )

    cpu_config.device = "cpu"

    reference_model = init_module.init_model(
        cpu_config
    ).cpu()

    model_architecture = architecture(
        reference_model
    )

    parameter_names = [
        name
        for name, _ in reference_model.named_parameters()
    ]

    if (
        model_architecture
        == "transformer"
    ):
        depth = len(
            reference_model.blocks
        )

    else:
        depth = len(
            reference_model.layers
        )

    del reference_model

    checkpoint_states: List[
        Dict[str, np.ndarray]
    ] = []

    checkpoint_metadata_list: List[
        Dict[str, Any]
    ] = []

    for checkpoint_index, checkpoint_path in enumerate(
        checkpoints
    ):
        print(
            "[INFO] loading weights "
            f"{checkpoint_index + 1}/"
            f"{len(checkpoints)}: "
            f"{checkpoint_path.name}",
            flush=True,
        )

        payload = safe_load(
            checkpoint_path
        )

        state_dict = extract_state_dict(
            payload
        )

        missing = [
            name
            for name in parameter_names
            if name not in state_dict
        ]

        if missing:
            raise RuntimeError(
                "Missing trainable parameters in "
                f"{checkpoint_path.name}: "
                f"{missing[:10]}"
            )

        checkpoint_states.append(
            {
                name: (
                    state_dict[name]
                    .detach()
                    .cpu()
                    .float()
                    .reshape(-1)
                    .numpy()
                    .copy()
                )
                for name in parameter_names
            }
        )

        checkpoint_metadata_list.append(
            checkpoint_metadata(
                checkpoint_path,
                payload,
            )
        )

    number_checkpoints = len(
        checkpoints
    )

    def zero_matrix() -> np.ndarray:
        return np.zeros(
            (
                number_checkpoints,
                number_checkpoints,
            ),
            dtype=np.float64,
        )

    raw_all = zero_matrix()

    raw_groups: Dict[
        str,
        np.ndarray,
    ] = {}

    layer_names = [
        f"layer_{index + 1}"
        for index in range(depth)
    ]

    raw_layers = {
        name: zero_matrix()
        for name in layer_names
    }

    if (
        model_architecture
        == "transformer"
    ):
        component_names = (
            "attention",
            "mlp",
            "norm",
            "other",
        )

    else:
        component_names = (
            "mixer",
            "norm",
            "other",
        )

    raw_components = {
        component: {
            name: zero_matrix()
            for name in layer_names
        }
        for component in component_names
    }

    parameter_groups: Dict[
        str,
        List[str],
    ] = {}

    layer_parameter_groups = {
        name: []
        for name in layer_names
    }

    component_parameter_groups = {
        component: {
            name: []
            for name in layer_names
        }
        for component in component_names
    }

    per_parameter: List[
        np.ndarray
    ] = []

    for parameter_index, parameter_name in enumerate(
        parameter_names
    ):
        print(
            "[INFO] parameter overlap "
            f"{parameter_index + 1}/"
            f"{len(parameter_names)}: "
            f"{parameter_name}",
            flush=True,
        )

        raw = raw_dot_matrix(
            [
                checkpoint_state[
                    parameter_name
                ]
                for checkpoint_state in checkpoint_states
            ]
        )

        per_parameter.append(
            cosine_from_raw(
                raw
            )
        )

        raw_all += raw

        (
            group,
            layer_index,
            component,
        ) = classify_parameter(
            parameter_name,
            model_architecture,
        )

        raw_groups.setdefault(
            group,
            zero_matrix(),
        )

        raw_groups[group] += raw

        parameter_groups.setdefault(
            group,
            [],
        ).append(
            parameter_name
        )

        if layer_index is not None:
            layer_name = (
                f"layer_{layer_index + 1}"
            )

            raw_layers[
                layer_name
            ] += raw

            layer_parameter_groups[
                layer_name
            ].append(
                parameter_name
            )

            if component in raw_components:
                raw_components[
                    component
                ][
                    layer_name
                ] += raw

                component_parameter_groups[
                    component
                ][
                    layer_name
                ].append(
                    parameter_name
                )

    group_names = sorted(
        raw_groups
    )

    result: Dict[
        str,
        Any,
    ] = {
        "weight_overlap_all": (
            cosine_from_raw(
                raw_all
            )
        ),

        "weight_overlap_by_group": (
            stack_cosines(
                raw_groups,
                group_names,
                number_checkpoints,
            )
        ),

        "weight_group_names": np.asarray(
            group_names,
            dtype=object,
        ),

        "weight_overlap_by_parameter": np.stack(
            per_parameter
        ),

        "weight_parameter_names": np.asarray(
            parameter_names,
            dtype=object,
        ),

        "weight_overlap_by_layer": stack_cosines(
            raw_layers,
            layer_names,
            number_checkpoints,
        ),

        "weight_layer_names": np.asarray(
            layer_names,
            dtype=object,
        ),

        "parameter_groups": (
            parameter_groups
        ),

        "layer_parameter_groups": (
            layer_parameter_groups
        ),

        "component_parameter_groups": (
            component_parameter_groups
        ),
    }

    if (
        model_architecture
        == "transformer"
    ):
        result.update(
            {
                "transformer_weight_overlap_global_by_layer": (
                    result[
                        "weight_overlap_by_layer"
                    ]
                ),

                "transformer_weight_overlap_attention_by_layer": (
                    stack_cosines(
                        raw_components[
                            "attention"
                        ],
                        layer_names,
                        number_checkpoints,
                    )
                ),

                "transformer_weight_overlap_mlp_by_layer": (
                    stack_cosines(
                        raw_components[
                            "mlp"
                        ],
                        layer_names,
                        number_checkpoints,
                    )
                ),

                "transformer_weight_overlap_norm_by_layer": (
                    stack_cosines(
                        raw_components[
                            "norm"
                        ],
                        layer_names,
                        number_checkpoints,
                    )
                ),
            }
        )

    else:
        result.update(
            {
                "mamba_weight_overlap_global_by_layer": (
                    result[
                        "weight_overlap_by_layer"
                    ]
                ),

                "mamba_weight_overlap_mixer_by_layer": (
                    stack_cosines(
                        raw_components[
                            "mixer"
                        ],
                        layer_names,
                        number_checkpoints,
                    )
                ),

                "mamba_weight_overlap_norm_by_layer": (
                    stack_cosines(
                        raw_components[
                            "norm"
                        ],
                        layer_names,
                        number_checkpoints,
                    )
                ),
            }
        )

    return (
        result,
        checkpoint_metadata_list,
        model_architecture,
    )


# =============================================================================
# Saving
# =============================================================================


def default_output_name(
    run_dir: Path,
    cli: argparse.Namespace,
) -> str:
    """Construct the automatic output-folder name."""

    base_name = (
        f"{run_dir.parent.name}_"
        f"{run_dir.name}"
    )

    if (
        cli.overlap_target
        == "weights"
    ):
        return (
            f"{base_name}_"
            "weight_overlaps"
        )

    if (
        cli.overlap_target
        == "predictive"
    ):
        return (
            f"{base_name}_"
            "predictive_overlaps_"
            f"{cli.representation_mode}_"
            f"{cli.predictive_layer_mode}_"
            f"every{cli.take_every}"
        )

    return (
        f"{base_name}_"
        "representation_overlaps_"
        f"{cli.metric}_"
        f"{cli.representation_mode}"
    )


def common_arrays(
    checkpoints: Sequence[Path],
    metadata: Sequence[
        Mapping[str, Any]
    ],
    config: argparse.Namespace,
    run_dir: Path,
    model_architecture: str,
) -> Dict[str, Any]:
    """Build arrays shared by both output types."""

    steps, epochs = steps_and_epochs(
        checkpoints,
        metadata,
        config,
    )

    return {
        "selected_checkpoint_files": np.asarray(
            [
                str(path)
                for path in checkpoints
            ],
            dtype=object,
        ),

        "selected_checkpoint_names": np.asarray(
            [
                path.name
                for path in checkpoints
            ],
            dtype=object,
        ),

        "selected_steps": steps,
        "selected_epochs": epochs,

        "architecture": np.asarray(
            model_architecture
        ),

        "run_dir": np.asarray(
            str(run_dir)
        ),

        "args_json": np.asarray(
            json.dumps(
                jsonify(
                    vars(config)
                ),
                sort_keys=True,
            )
        ),

        "checkpoint_metadata_json": np.asarray(
            json.dumps(
                jsonify(metadata),
                sort_keys=True,
            )
        ),
    }


def save_weights(
    output_dir: Path,
    result: Dict[str, Any],
    checkpoints: Sequence[Path],
    metadata: Sequence[
        Mapping[str, Any]
    ],
    config: argparse.Namespace,
    run_dir: Path,
    model_architecture: str,
) -> None:
    """Save weight-overlap outputs."""

    arrays = {
        key: value
        for key, value in result.items()
        if isinstance(
            value,
            np.ndarray,
        )
    }

    arrays.update(
        common_arrays(
            checkpoints,
            metadata,
            config,
            run_dir,
            model_architecture,
        )
    )

    arrays.update(
        {
            "overlap_target": np.asarray(
                "weights"
            ),

            "parameter_groups_json": np.asarray(
                json.dumps(
                    jsonify(
                        result[
                            "parameter_groups"
                        ]
                    ),
                    sort_keys=True,
                )
            ),

            "layer_parameter_groups_json": np.asarray(
                json.dumps(
                    jsonify(
                        result[
                            "layer_parameter_groups"
                        ]
                    ),
                    sort_keys=True,
                )
            ),

            "component_parameter_groups_json": np.asarray(
                json.dumps(
                    jsonify(
                        result[
                            "component_parameter_groups"
                        ]
                    ),
                    sort_keys=True,
                )
            ),
        }
    )

    output_path = (
        output_dir
        / "weight_overlaps.npz"
    )

    np.savez_compressed(
        output_path,
        **arrays,
    )

    metadata_path = (
        output_dir
        / "metadata.json"
    )

    metadata_payload = {
        "run_dir": run_dir,
        "output_path": output_path,
        "overlap_target": "weights",
        "architecture": (
            model_architecture
        ),
        "num_checkpoints": len(
            checkpoints
        ),
        "weight_group_names": (
            result[
                "weight_group_names"
            ]
        ),
        "weight_layer_names": (
            result[
                "weight_layer_names"
            ]
        ),
        "parameter_groups": (
            result[
                "parameter_groups"
            ]
        ),
        "layer_parameter_groups": (
            result[
                "layer_parameter_groups"
            ]
        ),
        "component_parameter_groups": (
            result[
                "component_parameter_groups"
            ]
        ),
    }

    metadata_path.write_text(
        json.dumps(
            jsonify(
                metadata_payload
            ),
            indent=2,
            sort_keys=True,
        )
    )

    print(
        f"[INFO] saved {output_path}",
        flush=True,
    )



def save_predictive(
    output_dir: Path,
    result: Mapping[str, Any],
    source_checkpoints: Sequence[Path],
    target_checkpoint: Path,
    source_metadata: Sequence[
        Mapping[str, Any]
    ],
    target_metadata: Mapping[str, Any],
    reference_metadata: Mapping[
        str,
        Any,
    ],
    config: argparse.Namespace,
    run_dir: Path,
    model_architecture: str,
    cli: argparse.Namespace,
) -> None:
    """Save predictive linear-readout overlap outputs."""

    arrays = common_arrays(
        source_checkpoints,
        source_metadata,
        config,
        run_dir,
        model_architecture,
    )

    arrays.update(
        {
            key: value
            for key, value in result.items()
            if isinstance(
                value,
                np.ndarray,
            )
        }
    )

    target_step_value, target_epoch_value = steps_and_epochs(
        [
            target_checkpoint
        ],
        [
            target_metadata
        ],
        config,
    )

    q = result[
        "predictive_q_by_position"
    ]

    arrays.update(
        {
            "overlap_target": np.asarray(
                "predictive"
            ),

            "representation_mode": np.asarray(
                cli.representation_mode
            ),

            "predictive_layer_mode": np.asarray(
                cli.predictive_layer_mode
            ),

            "predictive_num_folds": np.asarray(
                cli.predictive_num_folds
            ),

            "predictive_standardize_x": np.asarray(
                cli.predictive_standardize_x
            ),

            "take_every": np.asarray(
                cli.take_every
            ),

            "target_checkpoint_file": np.asarray(
                str(
                    target_checkpoint
                )
            ),

            "target_checkpoint_name": np.asarray(
                target_checkpoint.name
            ),

            "target_step": target_step_value,

            "target_epoch": target_epoch_value,

            "target_metadata_json": np.asarray(
                json.dumps(
                    jsonify(
                        target_metadata
                    ),
                    sort_keys=True,
                )
            ),

            "input_token_positions_1based": np.arange(
                1,
                q.shape[0] + 1,
            ),

            "target_token_positions_1based": np.arange(
                2,
                q.shape[0] + 2,
            ),

            "reference_source": np.asarray(
                reference_metadata[
                    "reference_source"
                ]
            ),

            "reference_metadata_json": np.asarray(
                json.dumps(
                    jsonify(
                        reference_metadata
                    ),
                    sort_keys=True,
                )
            ),

            "include_embedding": np.asarray(
                cli.include_embedding
            ),

            "include_final_norm": np.asarray(
                cli.include_final_norm
            ),
        }
    )

    output_path = (
        output_dir
        / "predictive_overlaps.npz"
    )

    np.savez_compressed(
        output_path,
        **arrays,
    )

    metadata_path = (
        output_dir
        / "metadata.json"
    )

    metadata_payload = {
        "run_dir": run_dir,
        "output_path": output_path,
        "overlap_target": "predictive",
        "architecture": model_architecture,
        "representation_mode": cli.representation_mode,
        "predictive_layer_mode": cli.predictive_layer_mode,
        "predictive_num_folds": cli.predictive_num_folds,
        "ridge_alphas": result[
            "ridge_alphas"
        ],
        "take_every": cli.take_every,
        "source_num_checkpoints": len(
            source_checkpoints
        ),
        "target_checkpoint": target_checkpoint,
        "source_layer_names": result[
            "source_layer_names"
        ],
        "target_layer_names": result[
            "target_layer_names"
        ],
        "num_positions": q.shape[0],
        "reference": reference_metadata,
    }

    metadata_path.write_text(
        json.dumps(
            jsonify(
                metadata_payload
            ),
            indent=2,
            sort_keys=True,
        )
    )

    print(
        f"[INFO] saved {output_path}",
        flush=True,
    )


def save_representations(
    output_dir: Path,
    train_cka: np.ndarray,
    valid_cka: np.ndarray,
    layer_names: Sequence[str],
    self_norms: Mapping[
        str,
        np.ndarray,
    ],
    reference_metadata: Mapping[
        str,
        Any,
    ],
    checkpoints: Sequence[Path],
    metadata: Sequence[
        Mapping[str, Any]
    ],
    config: argparse.Namespace,
    run_dir: Path,
    model_architecture: str,
    cli: argparse.Namespace,
) -> None:
    """Save representation-overlap outputs."""

    arrays = common_arrays(
        checkpoints,
        metadata,
        config,
        run_dir,
        model_architecture,
    )

    arrays.update(
        {
            "overlap_target": np.asarray(
                "representations"
            ),

            "metric": np.asarray(
                cli.metric
            ),

            "representation_mode": np.asarray(
                cli.representation_mode
            ),

            "train_cka_by_position": (
                train_cka
            ),

            "valid_cka_by_position": (
                valid_cka
            ),

            # Compatibility aliases.
            "test_cka_by_position": (
                valid_cka
            ),

            "train_cka_position_mean": np.nanmean(
                train_cka,
                axis=0,
            ),

            "valid_cka_position_mean": np.nanmean(
                valid_cka,
                axis=0,
            ),

            "test_cka_position_mean": np.nanmean(
                valid_cka,
                axis=0,
            ),

            "layer_names": np.asarray(
                layer_names,
                dtype=object,
            ),

            "input_token_positions_1based": np.arange(
                1,
                train_cka.shape[0] + 1,
            ),

            "target_token_positions_1based": np.arange(
                2,
                train_cka.shape[0] + 2,
            ),

            "train_self_norms": (
                self_norms[
                    "train"
                ]
            ),

            "valid_self_norms": (
                self_norms[
                    "valid"
                ]
            ),

            "test_self_norms": (
                self_norms[
                    "valid"
                ]
            ),

            "reference_source": np.asarray(
                reference_metadata[
                    "reference_source"
                ]
            ),

            "reference_metadata_json": np.asarray(
                json.dumps(
                    jsonify(
                        reference_metadata
                    ),
                    sort_keys=True,
                )
            ),

            "include_embedding": np.asarray(
                cli.include_embedding
            ),

            "include_final_norm": np.asarray(
                cli.include_final_norm
            ),
        }
    )

    output_path = (
        output_dir
        / "representation_overlaps_per_position.npz"
    )

    np.savez_compressed(
        output_path,
        **arrays,
    )

    metadata_path = (
        output_dir
        / "metadata.json"
    )

    metadata_payload = {
        "run_dir": run_dir,
        "output_path": output_path,
        "overlap_target": (
            "representations"
        ),
        "architecture": (
            model_architecture
        ),
        "metric": cli.metric,
        "representation_mode": (
            cli.representation_mode
        ),
        "num_checkpoints": len(
            checkpoints
        ),
        "layer_names": list(
            layer_names
        ),
        "num_positions": (
            train_cka.shape[0]
        ),
        "reference": (
            reference_metadata
        ),
    }

    metadata_path.write_text(
        json.dumps(
            jsonify(
                metadata_payload
            ),
            indent=2,
            sort_keys=True,
        )
    )

    print(
        f"[INFO] saved {output_path}",
        flush=True,
    )


# =============================================================================
# Command line and main
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--repo_dir",
        type=Path,
        default=Path.cwd(),
    )

    parser.add_argument(
        "--source_dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("data"),
    )

    parser.add_argument(
        "--collected_results_root",
        type=Path,
        default=Path(
            "collected_results"
        ),
    )

    parser.add_argument(
        "--run_dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )

    parser.add_argument(
        "--overlap_target",
        choices=(
            "representations",
            "weights",
            "predictive",
        ),
        default="representations",
    )

    parser.add_argument(
        "--metric",
        choices=(
            "gram",
            "feature",
        ),
        default="gram",
    )

    parser.add_argument(
        "--representation_mode",
        choices=(
            "state",
            "block_update",
        ),
        default="block_update",
    )

    parser.add_argument(
        "--reference_source",
        choices=(
            "auto",
            "saved",
            "generate",
        ),
        default="auto",
    )

    parser.add_argument(
        "--subset_train_size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--subset_valid_size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--subset_seed",
        type=int,
        default=12345,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--num_checkpoints",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--take_every",
        type=int,
        default=4,
        help="Predictive target: use one source checkpoint every this many saved checkpoints.",
    )

    parser.add_argument(
        "--target_step",
        type=int,
        default=None,
        help="Predictive target: checkpoint step used as t'. Default: last checkpoint.",
    )

    parser.add_argument(
        "--target_index",
        type=int,
        default=-1,
        help="Predictive target: checkpoint index used as t' when target_step is absent.",
    )

    parser.add_argument(
        "--predictive_layer_mode",
        choices=(
            "same",
            "all",
        ),
        default="same",
        help="Predictive target: same computes only l'=l; all computes every layer pair.",
    )

    parser.add_argument(
        "--predictive_num_folds",
        type=int,
        default=5,
        help="Predictive target: number of readout CV folds.",
    )

    parser.add_argument(
        "--ridge_alphas",
        type=str,
        default="1e-6,1e-4,1e-2,1e0,1e2,1e4,1e6",
        help="Predictive target: comma-separated ridge-alpha grid.",
    )

    parser.add_argument(
        "--predictive_standardize_x",
        action="store_true",
        default=True,
        help="Predictive target: z-score source features inside each train fold.",
    )

    parser.add_argument(
        "--no_predictive_standardize_x",
        dest="predictive_standardize_x",
        action="store_false",
    )

    parser.add_argument(
        "--max_step",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--temp_root",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--keep_temp",
        action="store_true",
    )

    parser.add_argument(
        "--include_embedding",
        action="store_true",
        default=True,
    )

    parser.add_argument(
        "--no_include_embedding",
        dest="include_embedding",
        action="store_false",
    )

    parser.add_argument(
        "--include_final_norm",
        action="store_true",
        default=True,
    )

    parser.add_argument(
        "--no_include_final_norm",
        dest="include_final_norm",
        action="store_false",
    )

    return parser.parse_args()


def main() -> None:
    cli = parse_args()

    repo_dir = (
        cli.repo_dir
        .expanduser()
        .resolve()
    )

    source_dir = resolve_source_dir(
        repo_dir,
        cli.source_dir,
    )

    data_root = resolve_relative(
        cli.data_root,
        repo_dir,
    )

    collected_results_root = resolve_relative(
        cli.collected_results_root,
        repo_dir,
    )

    collected_results_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        cli.device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        print(
            "[WARN] CUDA requested but unavailable; "
            "falling back to CPU.",
            flush=True,
        )

        cli.device = "cpu"

    run_dir = resolve_run_dir(
        repo_dir,
        data_root,
        cli.run_dir,
        cli.run_name,
        cli.run_id,
    )

    config = load_config(
        run_dir,
        cli.device,
    )

    init_module = import_init(
        source_dir
    )

    all_checkpoints = find_checkpoints(
        run_dir,
        cli.max_step,
    )

    predictive_target_checkpoint: Optional[Path] = None

    if cli.overlap_target == "predictive":
        predictive_target_index = resolve_predictive_target_index(
            all_checkpoints,
            cli,
        )

        selected_indices = predictive_source_indices(
            len(all_checkpoints),
            predictive_target_index,
            cli.take_every,
            cli.num_checkpoints,
        )

        predictive_target_checkpoint = all_checkpoints[
            predictive_target_index
        ]

    else:
        selected_indices = log_select_indices(
            len(all_checkpoints),
            cli.num_checkpoints,
        )

    checkpoints = [
        all_checkpoints[index]
        for index in selected_indices
    ]

    output_name = (
        cli.output_name
        or default_output_name(
            run_dir,
            cli,
        )
    )

    output_dir = (
        collected_results_root
        / output_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"[INFO] run_dir={run_dir}",
        flush=True,
    )

    print(
        "[INFO] checkpoints="
        f"{len(checkpoints)}/"
        f"{len(all_checkpoints)}",
        flush=True,
    )

    if predictive_target_checkpoint is not None:
        print(
            "[INFO] predictive_target_checkpoint="
            f"{predictive_target_checkpoint.name}",
            flush=True,
        )

    print(
        f"[INFO] output_dir={output_dir}",
        flush=True,
    )

    if (
        cli.overlap_target
        == "weights"
    ):
        (
            result,
            checkpoint_metadata_list,
            model_architecture,
        ) = compute_weight_overlaps(
            checkpoints,
            config,
            init_module,
        )

        save_weights(
            output_dir,
            result,
            checkpoints,
            checkpoint_metadata_list,
            config,
            run_dir,
            model_architecture,
        )

        return

    (
        train_reference,
        valid_reference,
        reference_metadata,
    ) = load_references(
        config,
        cli,
        repo_dir,
        source_dir,
        run_dir,
    )

    if cli.temp_root is None:
        temp_root = output_dir

    else:
        temp_root = resolve_relative(
            cli.temp_root,
            repo_dir,
        )

    temp_dir = (
        temp_root
        / f"_tmp_{output_name}"
    )

    if temp_dir.exists():
        shutil.rmtree(
            temp_dir
        )

    temp_dir.mkdir(
        parents=True
    )

    started = time.time()

    if cli.overlap_target == "predictive":
        if predictive_target_checkpoint is None:
            raise RuntimeError(
                "Internal error: predictive target checkpoint was not resolved."
            )

        readout_reference = torch.cat(
            [
                train_reference,
                valid_reference,
            ],
            dim=0,
        )

        predictive_reference_metadata = dict(
            reference_metadata
        )

        predictive_reference_metadata.update(
            {
                "readout_reference_source": "train_reference_plus_valid_reference",
                "readout_num_samples": int(
                    readout_reference.shape[0]
                ),
                "readout_num_folds": int(
                    cli.predictive_num_folds
                ),
                "readout_fold_seed": int(
                    cli.subset_seed
                )
                + 7919,
            }
        )

        (
            predictive_result,
            source_metadata_list,
            target_metadata,
            model_architecture,
        ) = compute_predictive_overlaps(
            checkpoints,
            predictive_target_checkpoint,
            config,
            init_module,
            readout_reference,
            cli,
            temp_dir,
        )

        save_predictive(
            output_dir,
            predictive_result,
            checkpoints,
            predictive_target_checkpoint,
            source_metadata_list,
            target_metadata,
            predictive_reference_metadata,
            config,
            run_dir,
            model_architecture,
            cli,
        )

        print(
            "[INFO] total predictive-overlap time="
            f"{time.time() - started:.2f}s",
            flush=True,
        )

        if cli.keep_temp:
            print(
                "[INFO] temporary files kept in "
                f"{temp_dir}",
                flush=True,
            )

        else:
            shutil.rmtree(
                temp_dir,
                ignore_errors=True,
            )

        return

    (
        train_cka,
        valid_cka,
        layer_names,
        checkpoint_metadata_list,
        self_norms,
        model_architecture,
    ) = compute_representation_overlaps(
        checkpoints,
        config,
        init_module,
        train_reference,
        valid_reference,
        cli,
        temp_dir,
    )

    save_representations(
        output_dir,
        train_cka,
        valid_cka,
        layer_names,
        self_norms,
        reference_metadata,
        checkpoints,
        checkpoint_metadata_list,
        config,
        run_dir,
        model_architecture,
        cli,
    )

    print(
        "[INFO] total representation time="
        f"{time.time() - started:.2f}s",
        flush=True,
    )

    if cli.keep_temp:
        print(
            "[INFO] temporary files kept in "
            f"{temp_dir}",
            flush=True,
        )

    else:
        shutil.rmtree(
            temp_dir,
            ignore_errors=True,
        )


if __name__ == "__main__":
    main()