#!/usr/bin/env python3
"""Layer-wise attention/MLP relative linearization diagnostics for SLM checkpoints.

VERSION: layerwise_attention_mlp_first_baseline_v1

The script selects K saved checkpoints approximately logarithmically in their
optimizer steps. The earliest selected checkpoint is the *single common
baseline*. For every selected target checkpoint, and for every Transformer
layer, it evaluates attention and MLP parameter groups separately.

For a group g (for example, attention in layer 3), define

    delta_theta_g(t) = theta_g(t) - theta_g(t0),

with every parameter outside g fixed to zero displacement. The actual isolated
functional change is measured with a hybrid model in which only group g is
moved from its baseline value to its target value:

    delta_z_actual_g(t)
      = f(X; theta_t0 with group g replaced by theta_g(t)) - f(X; theta_t0).

The first-order prediction at the common baseline is

    delta_z_linear_g(t) = J_g(X; theta_t0) delta_theta_g(t).

The reported relative linearization error is

    epsilon_g(t)
      = ||delta_z_actual_g(t) - delta_z_linear_g(t)||_F
        / ||delta_z_actual_g(t)||_F.

This group-isolated definition is essential: comparing a group-only tangent
contribution against the full network's output change would incorrectly count
all other parameter groups as "linearization error".

Expected run layout
-------------------

data/<run_name>/<run_id>/
    args.json
    results.pt
    rules.pt
    checkpoints/checkpoint_step_<step>.pt

Output layout
-------------

collected_results/<output_name>/
    component_linearization_diagnostics.npz
    component_linearization_summary.csv
    component_linearization_metadata.json

Main NPZ arrays have shape [K, 2, L] for scalar metrics and [K, 2, L, T]
for position-resolved metrics. Component axis 0 is attention and axis 1 is MLP.
The first target is the baseline itself, so its change is zero and ratio-based
metrics are NaN by construction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from compute_slm_representation_overlaps import (
    checkpoint_step,
    extract_state_dict,
    find_checkpoints,
    import_init,
    load_config,
    load_references,
    resolve_relative,
    resolve_run_dir,
    resolve_source_dir,
    safe_load,
)

try:
    from torch.func import functional_call as torch_functional_call
    from torch.func import jvp as torch_jvp

    HAS_TORCH_FUNC = True
except ImportError:  # pragma: no cover
    HAS_TORCH_FUNC = False
    torch_functional_call = None
    torch_jvp = None

try:
    from torch.nn.utils.stateless import functional_call as stateless_functional_call
except ImportError:  # pragma: no cover
    stateless_functional_call = None


DIAGNOSTIC_VERSION = "layerwise_attention_mlp_first_baseline_v1"
COMPONENTS = ("attention", "mlp")
EPS = 1e-30


# =============================================================================
# Checkpoint selection
# =============================================================================


def select_log_spaced_checkpoints(
    checkpoints: Sequence[Path],
    number_selected: int,
) -> Tuple[List[Path], np.ndarray]:
    """Select exactly K distinct checkpoints approximately log-spaced in step.

    The earliest and latest available checkpoints are always retained.
    """

    number_checkpoints = len(checkpoints)
    if number_checkpoints == 0:
        raise ValueError("No checkpoints were provided.")

    count = min(max(int(number_selected), 1), number_checkpoints)
    if count == number_checkpoints:
        indices = np.arange(number_checkpoints, dtype=np.int64)
        return list(checkpoints), indices

    steps = np.asarray(
        [checkpoint_step(path) for path in checkpoints],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(steps)) or np.any(steps < 0):
        raise ValueError(
            "Every checkpoint filename must contain a non-negative step, for "
            "example checkpoint_step_100.pt."
        )
    if np.any(np.diff(steps) <= 0):
        raise ValueError(
            f"Checkpoint steps must be strictly increasing, got {steps.tolist()}"
        )

    if count == 1:
        indices = np.asarray([0], dtype=np.int64)
        return [checkpoints[0]], indices

    target_steps = np.geomspace(
        max(float(steps[0]), 1.0),
        max(float(steps[-1]), 1.0),
        num=count,
    )
    log_steps = np.log(np.maximum(steps, 1.0))

    selected: List[int] = []
    for target_index, target in enumerate(target_steps):
        remaining = count - target_index - 1
        lower = 0 if not selected else selected[-1] + 1
        upper = number_checkpoints - remaining - 1

        if target_index == 0:
            chosen = 0
        elif target_index == count - 1:
            chosen = number_checkpoints - 1
        else:
            allowed = np.arange(lower, upper + 1, dtype=np.int64)
            distances = np.abs(
                log_steps[allowed] - math.log(max(float(target), 1.0))
            )
            chosen = int(allowed[int(np.argmin(distances))])

        selected.append(chosen)

    indices = np.asarray(selected, dtype=np.int64)
    if len(np.unique(indices)) != count:
        raise RuntimeError(f"Checkpoint selection produced duplicates: {indices}")
    return [checkpoints[int(index)] for index in indices], indices


# =============================================================================
# Model state and component groups
# =============================================================================


def load_state_dict_cpu(path: Path) -> Dict[str, torch.Tensor]:
    state = extract_state_dict(safe_load(path, map_location="cpu"))
    return {
        str(name): torch.as_tensor(value).detach().cpu()
        for name, value in state.items()
    }


def state_parameters(
    state_dict: Mapping[str, torch.Tensor],
    model_parameters: Mapping[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    output: Dict[str, torch.Tensor] = {}
    missing: List[str] = []

    for name, model_parameter in model_parameters.items():
        value = state_dict.get(name)
        if value is None:
            missing.append(name)
            continue
        output[name] = value.to(
            device=device,
            dtype=model_parameter.dtype,
            non_blocking=True,
        )

    if missing:
        raise RuntimeError(f"Checkpoint is missing trainable parameters: {missing}")
    return output


def state_buffers(
    state_dict: Mapping[str, torch.Tensor],
    model_buffers: Mapping[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    output: Dict[str, torch.Tensor] = {}
    for name, model_buffer in model_buffers.items():
        value = state_dict.get(name, model_buffer)
        output[name] = torch.as_tensor(value).to(
            device=device,
            dtype=model_buffer.dtype,
            non_blocking=True,
        )
    return output


def discover_transformer_component_groups(
    parameter_names: Sequence[str],
) -> Tuple[List[str], Dict[str, Dict[str, List[str]]]]:
    """Return blocks.<i>.attn.* and blocks.<i>.ffwd.* groups by layer."""

    layer_indices: set[int] = set()
    grouped_by_index: Dict[str, Dict[int, List[str]]] = {
        component: {} for component in COMPONENTS
    }

    pattern = re.compile(r"^blocks\.(\d+)\.(attn|ffwd)\.(.+)$")
    for name in parameter_names:
        match = pattern.match(name)
        if match is None:
            continue

        layer_index = int(match.group(1))
        raw_component = match.group(2)
        component = "attention" if raw_component == "attn" else "mlp"
        layer_indices.add(layer_index)
        grouped_by_index[component].setdefault(layer_index, []).append(name)

    if not layer_indices:
        raise TypeError(
            "No Transformer parameters matching blocks.<layer>.attn.* or "
            "blocks.<layer>.ffwd.* were found. This diagnostic is only for "
            "the GPT-2-style Transformer in transformer.py."
        )

    ordered_indices = sorted(layer_indices)
    expected = list(range(ordered_indices[-1] + 1))
    if ordered_indices != expected:
        raise RuntimeError(
            f"Transformer layer indices are not contiguous: {ordered_indices}"
        )

    layer_names = [f"layer_{index + 1}" for index in ordered_indices]
    groups: Dict[str, Dict[str, List[str]]] = {
        component: {} for component in COMPONENTS
    }

    for component in COMPONENTS:
        for layer_index, layer_name in zip(ordered_indices, layer_names):
            names = sorted(grouped_by_index[component].get(layer_index, []))
            if not names:
                raise RuntimeError(
                    f"No {component} parameters found for {layer_name}."
                )
            groups[component][layer_name] = names

    return layer_names, groups


# =============================================================================
# Functional model and directional derivatives
# =============================================================================


def center_logits(logits: torch.Tensor, enabled: bool) -> torch.Tensor:
    if not enabled:
        return logits
    return logits - logits.mean(dim=-1, keepdim=True)


def functional_logits(
    model: torch.nn.Module,
    parameters: Mapping[str, torch.Tensor],
    buffers: Mapping[str, torch.Tensor],
    inputs: torch.Tensor,
    center: bool,
) -> torch.Tensor:
    if HAS_TORCH_FUNC:
        logits = torch_functional_call(
            model,
            (parameters, buffers),
            (inputs,),
            strict=True,
        )
    else:
        if stateless_functional_call is None:
            raise RuntimeError(
                "No functional-call implementation is available. Install PyTorch 2.x."
            )
        complete_state = dict(buffers)
        complete_state.update(parameters)
        logits = stateless_functional_call(
            model,
            complete_state,
            (inputs,),
            strict=True,
        )
    return center_logits(logits, center)


def group_direction(
    baseline_parameters: Mapping[str, torch.Tensor],
    target_parameters: Mapping[str, torch.Tensor],
    group_names: Sequence[str],
) -> Dict[str, torch.Tensor]:
    """Full parameter pytree with nonzero displacement only for one group."""

    selected = set(group_names)
    return {
        name: (
            target_parameters[name] - baseline_value
            if name in selected
            else torch.zeros_like(baseline_value)
        )
        for name, baseline_value in baseline_parameters.items()
    }


def hybrid_parameters(
    baseline_parameters: Mapping[str, torch.Tensor],
    target_parameters: Mapping[str, torch.Tensor],
    group_names: Sequence[str],
) -> Dict[str, torch.Tensor]:
    """Parameters with exactly one group replaced by its target values."""

    selected = set(group_names)
    return {
        name: (target_parameters[name] if name in selected else baseline_value)
        for name, baseline_value in baseline_parameters.items()
    }


def autodiff_jvp(
    model: torch.nn.Module,
    baseline_parameters: Mapping[str, torch.Tensor],
    baseline_buffers: Mapping[str, torch.Tensor],
    direction: Mapping[str, torch.Tensor],
    inputs: torch.Tensor,
    center: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not HAS_TORCH_FUNC:
        raise RuntimeError("torch.func is unavailable in this PyTorch installation.")

    def model_function(parameter_tree):
        return functional_logits(
            model,
            parameter_tree,
            baseline_buffers,
            inputs,
            center,
        )

    return torch_jvp(
        model_function,
        (baseline_parameters,),
        (direction,),
    )


@torch.no_grad()
def finite_difference_jvp(
    model: torch.nn.Module,
    baseline_parameters: Mapping[str, torch.Tensor],
    baseline_buffers: Mapping[str, torch.Tensor],
    direction: Mapping[str, torch.Tensor],
    inputs: torch.Tensor,
    center: bool,
    epsilon: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    plus = {
        name: value + float(epsilon) * direction[name]
        for name, value in baseline_parameters.items()
    }
    minus = {
        name: value - float(epsilon) * direction[name]
        for name, value in baseline_parameters.items()
    }

    baseline_logits = functional_logits(
        model, baseline_parameters, baseline_buffers, inputs, center
    )
    forward_plus = functional_logits(model, plus, baseline_buffers, inputs, center)
    forward_minus = functional_logits(model, minus, baseline_buffers, inputs, center)
    tangent = (forward_plus - forward_minus) / (2.0 * float(epsilon))
    return baseline_logits, tangent


class DirectionalDerivative:
    def __init__(self, method: str, finite_difference_epsilon: float):
        self.requested_method = str(method)
        self.resolved_method: Optional[str] = None
        self.finite_difference_epsilon = float(finite_difference_epsilon)

    def __call__(
        self,
        model: torch.nn.Module,
        baseline_parameters: Mapping[str, torch.Tensor],
        baseline_buffers: Mapping[str, torch.Tensor],
        direction: Mapping[str, torch.Tensor],
        inputs: torch.Tensor,
        center: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.resolved_method == "autodiff":
            return autodiff_jvp(
                model,
                baseline_parameters,
                baseline_buffers,
                direction,
                inputs,
                center,
            )

        if self.resolved_method == "finite_difference":
            return finite_difference_jvp(
                model,
                baseline_parameters,
                baseline_buffers,
                direction,
                inputs,
                center,
                self.finite_difference_epsilon,
            )

        if self.requested_method == "autodiff":
            result = autodiff_jvp(
                model,
                baseline_parameters,
                baseline_buffers,
                direction,
                inputs,
                center,
            )
            self.resolved_method = "autodiff"
            print("[INFO] JVP method resolved to autodiff", flush=True)
            return result

        if self.requested_method == "finite_difference":
            result = finite_difference_jvp(
                model,
                baseline_parameters,
                baseline_buffers,
                direction,
                inputs,
                center,
                self.finite_difference_epsilon,
            )
            self.resolved_method = "finite_difference"
            print("[INFO] JVP method resolved to finite_difference", flush=True)
            return result

        try:
            result = autodiff_jvp(
                model,
                baseline_parameters,
                baseline_buffers,
                direction,
                inputs,
                center,
            )
            self.resolved_method = "autodiff"
            print("[INFO] JVP method resolved to autodiff", flush=True)
            return result
        except (RuntimeError, NotImplementedError) as error:
            print(
                "[WARN] Autodiff JVP failed; falling back to central finite "
                f"differences. Reason: {type(error).__name__}: {error}",
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.resolved_method = "finite_difference"
            return finite_difference_jvp(
                model,
                baseline_parameters,
                baseline_buffers,
                direction,
                inputs,
                center,
                self.finite_difference_epsilon,
            )


# =============================================================================
# Metric accumulation
# =============================================================================


def empty_accumulator(number_positions: int) -> Dict[str, Any]:
    return {
        "actual_sq": 0.0,
        "linear_sq": 0.0,
        "residual_sq": 0.0,
        "dot": 0.0,
        "actual_sq_by_position": np.zeros(number_positions, dtype=np.float64),
        "linear_sq_by_position": np.zeros(number_positions, dtype=np.float64),
        "residual_sq_by_position": np.zeros(number_positions, dtype=np.float64),
        "dot_by_position": np.zeros(number_positions, dtype=np.float64),
        "num_sequences": 0,
        "num_logits": 0,
    }


def accumulate(
    accumulator: Dict[str, Any],
    actual_change: torch.Tensor,
    linear_change: torch.Tensor,
) -> None:
    residual = actual_change - linear_change

    actual64 = actual_change.detach().to(torch.float64)
    linear64 = linear_change.detach().to(torch.float64)
    residual64 = residual.detach().to(torch.float64)

    accumulator["actual_sq"] += float(torch.sum(actual64 * actual64).item())
    accumulator["linear_sq"] += float(torch.sum(linear64 * linear64).item())
    accumulator["residual_sq"] += float(torch.sum(residual64 * residual64).item())
    accumulator["dot"] += float(torch.sum(actual64 * linear64).item())

    accumulator["actual_sq_by_position"] += (
        torch.sum(actual64 * actual64, dim=(0, 2)).cpu().numpy()
    )
    accumulator["linear_sq_by_position"] += (
        torch.sum(linear64 * linear64, dim=(0, 2)).cpu().numpy()
    )
    accumulator["residual_sq_by_position"] += (
        torch.sum(residual64 * residual64, dim=(0, 2)).cpu().numpy()
    )
    accumulator["dot_by_position"] += (
        torch.sum(actual64 * linear64, dim=(0, 2)).cpu().numpy()
    )

    accumulator["num_sequences"] += int(actual_change.shape[0])
    accumulator["num_logits"] += int(actual_change.numel())


def safe_ratio(numerator: float, denominator: float) -> float:
    if (
        not np.isfinite(numerator)
        or not np.isfinite(denominator)
        or denominator <= EPS
    ):
        return float("nan")
    return float(numerator / denominator)


def safe_array_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    output = np.full_like(numerator, np.nan)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > EPS)
    output[valid] = numerator[valid] / denominator[valid]
    return output


def finalize(accumulator: Mapping[str, Any]) -> Dict[str, Any]:
    actual_norm = math.sqrt(max(float(accumulator["actual_sq"]), 0.0))
    linear_norm = math.sqrt(max(float(accumulator["linear_sq"]), 0.0))
    residual_norm = math.sqrt(max(float(accumulator["residual_sq"]), 0.0))

    actual_norm_pos = np.sqrt(
        np.maximum(accumulator["actual_sq_by_position"], 0.0)
    )
    linear_norm_pos = np.sqrt(
        np.maximum(accumulator["linear_sq_by_position"], 0.0)
    )
    residual_norm_pos = np.sqrt(
        np.maximum(accumulator["residual_sq_by_position"], 0.0)
    )

    return {
        "relative_error": safe_ratio(residual_norm, actual_norm),
        "alignment": safe_ratio(
            float(accumulator["dot"]), actual_norm * linear_norm
        ),
        "amplitude_ratio": safe_ratio(linear_norm, actual_norm),
        "explained_fraction": (
            float("nan")
            if float(accumulator["actual_sq"]) <= EPS
            else 1.0
            - float(accumulator["residual_sq"])
            / float(accumulator["actual_sq"])
        ),
        "actual_norm": actual_norm,
        "linear_norm": linear_norm,
        "residual_norm": residual_norm,
        "relative_error_by_position": safe_array_ratio(
            residual_norm_pos, actual_norm_pos
        ),
        "alignment_by_position": safe_array_ratio(
            accumulator["dot_by_position"],
            actual_norm_pos * linear_norm_pos,
        ),
        "amplitude_ratio_by_position": safe_array_ratio(
            linear_norm_pos, actual_norm_pos
        ),
        "explained_fraction_by_position": np.where(
            accumulator["actual_sq_by_position"] > EPS,
            1.0
            - accumulator["residual_sq_by_position"]
            / np.maximum(accumulator["actual_sq_by_position"], EPS),
            np.nan,
        ),
        "actual_norm_by_position": actual_norm_pos,
        "linear_norm_by_position": linear_norm_pos,
        "residual_norm_by_position": residual_norm_pos,
        "num_sequences": int(accumulator["num_sequences"]),
        "num_logits": int(accumulator["num_logits"]),
    }


def relative_parameter_displacement(
    baseline_parameters: Mapping[str, torch.Tensor],
    target_parameters: Mapping[str, torch.Tensor],
    group_names: Sequence[str],
) -> float:
    delta_sq = 0.0
    baseline_sq = 0.0
    for name in group_names:
        baseline = baseline_parameters[name].detach().to(torch.float64)
        target = target_parameters[name].detach().to(torch.float64)
        delta_sq += float(torch.sum((target - baseline) ** 2).item())
        baseline_sq += float(torch.sum(baseline ** 2).item())

    return safe_ratio(
        math.sqrt(max(delta_sq, 0.0)),
        math.sqrt(max(baseline_sq, 0.0)),
    )


# =============================================================================
# One target and one layer-component group
# =============================================================================


def evaluate_group(
    *,
    model: torch.nn.Module,
    baseline_parameters: Mapping[str, torch.Tensor],
    baseline_buffers: Mapping[str, torch.Tensor],
    target_parameters: Mapping[str, torch.Tensor],
    group_names: Sequence[str],
    sequences: torch.Tensor,
    batch_size: int,
    device: torch.device,
    center: bool,
    derivative: DirectionalDerivative,
) -> Dict[str, Any]:
    direction = group_direction(
        baseline_parameters,
        target_parameters,
        group_names,
    )
    hybrid = hybrid_parameters(
        baseline_parameters,
        target_parameters,
        group_names,
    )

    number_positions = int(sequences.shape[1])
    accumulator = empty_accumulator(number_positions)

    for start in range(0, len(sequences), int(batch_size)):
        stop = min(start + int(batch_size), len(sequences))
        inputs = sequences[start:stop].to(
            device=device,
            non_blocking=True,
        ).long()

        baseline_logits, linear_change = derivative(
            model,
            baseline_parameters,
            baseline_buffers,
            direction,
            inputs,
            center,
        )

        with torch.no_grad():
            hybrid_logits = functional_logits(
                model,
                hybrid,
                baseline_buffers,
                inputs,
                center,
            )

        actual_change = hybrid_logits - baseline_logits.detach()
        accumulate(accumulator, actual_change, linear_change.detach())

        del inputs, baseline_logits, hybrid_logits, actual_change, linear_change

    result = finalize(accumulator)
    result["relative_parameter_displacement"] = relative_parameter_displacement(
        baseline_parameters,
        target_parameters,
        group_names,
    )
    result["num_group_parameters"] = int(
        sum(baseline_parameters[name].numel() for name in group_names)
    )

    del direction, hybrid
    return result


# =============================================================================
# Output
# =============================================================================


SCALAR_METRICS = (
    "relative_error",
    "alignment",
    "amplitude_ratio",
    "explained_fraction",
    "actual_norm",
    "linear_norm",
    "residual_norm",
    "relative_parameter_displacement",
    "num_group_parameters",
    "num_sequences",
    "num_logits",
)

POSITION_METRICS = (
    "relative_error_by_position",
    "alignment_by_position",
    "amplitude_ratio_by_position",
    "explained_fraction_by_position",
    "actual_norm_by_position",
    "linear_norm_by_position",
    "residual_norm_by_position",
)


def allocate_result_arrays(
    number_checkpoints: int,
    number_layers: int,
    number_positions: int,
) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {}
    for name in SCALAR_METRICS:
        arrays[name] = np.full(
            (number_checkpoints, len(COMPONENTS), number_layers),
            np.nan,
            dtype=np.float64,
        )
    for name in POSITION_METRICS:
        arrays[name] = np.full(
            (
                number_checkpoints,
                len(COMPONENTS),
                number_layers,
                number_positions,
            ),
            np.nan,
            dtype=np.float64,
        )
    return arrays


def save_outputs(
    *,
    output_dir: Path,
    selected_checkpoints: Sequence[Path],
    selected_indices: np.ndarray,
    selected_steps: np.ndarray,
    selected_epochs: np.ndarray,
    layer_names: Sequence[str],
    component_groups: Mapping[str, Mapping[str, Sequence[str]]],
    split_arrays: Mapping[str, Mapping[str, np.ndarray]],
    metadata: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "diagnostic_version": np.asarray(DIAGNOSTIC_VERSION),
        "selected_checkpoint_indices": selected_indices.astype(np.int64),
        "selected_steps": selected_steps.astype(np.int64),
        "selected_epochs": selected_epochs.astype(np.float64),
        "baseline_selected_index": np.asarray(0, dtype=np.int64),
        "baseline_step": np.asarray(selected_steps[0], dtype=np.int64),
        "baseline_epoch": np.asarray(selected_epochs[0], dtype=np.float64),
        "checkpoint_files": np.asarray(
            [str(path) for path in selected_checkpoints], dtype=object
        ),
        "component_names": np.asarray(COMPONENTS, dtype=object),
        "layer_names": np.asarray(layer_names, dtype=object),
    }

    for split, arrays in split_arrays.items():
        for name, values in arrays.items():
            payload[f"{split}_{name}"] = values

    np.savez_compressed(
        output_dir / "component_linearization_diagnostics.npz",
        **payload,
    )

    with (
        output_dir / "component_linearization_metadata.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    fieldnames = [
        "diagnostic_version",
        "split",
        "target_selected_index",
        "target_checkpoint_index",
        "baseline_step",
        "target_step",
        "baseline_epoch",
        "target_epoch",
        "component",
        "layer_index",
        "layer_name",
        "relative_error",
        "alignment",
        "amplitude_ratio",
        "explained_fraction",
        "actual_norm",
        "linear_norm",
        "residual_norm",
        "relative_parameter_displacement",
        "num_group_parameters",
        "baseline_checkpoint_file",
        "target_checkpoint_file",
        "parameter_names",
    ]

    with (
        output_dir / "component_linearization_summary.csv"
    ).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for split, arrays in split_arrays.items():
            for target_index in range(len(selected_checkpoints)):
                for component_index, component in enumerate(COMPONENTS):
                    for layer_index, layer_name in enumerate(layer_names):
                        writer.writerow(
                            {
                                "diagnostic_version": DIAGNOSTIC_VERSION,
                                "split": split,
                                "target_selected_index": target_index,
                                "target_checkpoint_index": int(
                                    selected_indices[target_index]
                                ),
                                "baseline_step": int(selected_steps[0]),
                                "target_step": int(selected_steps[target_index]),
                                "baseline_epoch": float(selected_epochs[0]),
                                "target_epoch": float(selected_epochs[target_index]),
                                "component": component,
                                "layer_index": layer_index + 1,
                                "layer_name": layer_name,
                                "relative_error": float(
                                    arrays["relative_error"][
                                        target_index, component_index, layer_index
                                    ]
                                ),
                                "alignment": float(
                                    arrays["alignment"][
                                        target_index, component_index, layer_index
                                    ]
                                ),
                                "amplitude_ratio": float(
                                    arrays["amplitude_ratio"][
                                        target_index, component_index, layer_index
                                    ]
                                ),
                                "explained_fraction": float(
                                    arrays["explained_fraction"][
                                        target_index, component_index, layer_index
                                    ]
                                ),
                                "actual_norm": float(
                                    arrays["actual_norm"][
                                        target_index, component_index, layer_index
                                    ]
                                ),
                                "linear_norm": float(
                                    arrays["linear_norm"][
                                        target_index, component_index, layer_index
                                    ]
                                ),
                                "residual_norm": float(
                                    arrays["residual_norm"][
                                        target_index, component_index, layer_index
                                    ]
                                ),
                                "relative_parameter_displacement": float(
                                    arrays["relative_parameter_displacement"][
                                        target_index, component_index, layer_index
                                    ]
                                ),
                                "num_group_parameters": int(
                                    arrays["num_group_parameters"][
                                        target_index, component_index, layer_index
                                    ]
                                ) if np.isfinite(
                                    arrays["num_group_parameters"][
                                        target_index, component_index, layer_index
                                    ]
                                ) else 0,
                                "baseline_checkpoint_file": str(
                                    selected_checkpoints[0]
                                ),
                                "target_checkpoint_file": str(
                                    selected_checkpoints[target_index]
                                ),
                                "parameter_names": json.dumps(
                                    component_groups[component][layer_name]
                                ),
                            }
                        )


# =============================================================================
# CLI and main
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute first-baseline relative linearization errors separately "
            "for Transformer attention and MLP parameters, layer by layer."
        )
    )

    parser.add_argument("--repo_dir", type=Path, default=Path.cwd())
    parser.add_argument("--source_dir", type=Path, default=None)
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument(
        "--collected_results_root",
        type=Path,
        default=Path("collected_results"),
    )

    parser.add_argument("--run_dir", type=Path, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--output_name", type=str, default=None)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_checkpoints", type=int, default=5)
    parser.add_argument("--max_step", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--subset_size", type=int, default=512)
    parser.add_argument("--subset_seed", type=int, default=12345)
    parser.add_argument("--split", choices=("train", "test", "both"), default="test")

    parser.add_argument(
        "--reference_source",
        choices=("auto", "saved", "generate"),
        default="auto",
        help="How fixed RHM reference sequences are obtained.",
    )
    parser.add_argument(
        "--jvp_method",
        choices=("auto", "autodiff", "finite_difference"),
        default="auto",
    )
    parser.add_argument("--finite_difference_epsilon", type=float, default=1e-3)

    center_group = parser.add_mutually_exclusive_group()
    center_group.add_argument(
        "--center_logits",
        dest="center_logits",
        action="store_true",
    )
    center_group.add_argument(
        "--no_center_logits",
        dest="center_logits",
        action="store_false",
    )
    parser.set_defaults(center_logits=True)

    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    torch.set_default_dtype(torch.float32)

    if cli.num_checkpoints < 2:
        raise ValueError("--num_checkpoints must be at least 2.")
    if cli.subset_size <= 0:
        raise ValueError("--subset_size must be positive.")
    if cli.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if cli.finite_difference_epsilon <= 0:
        raise ValueError("--finite_difference_epsilon must be positive.")

    repo_dir = cli.repo_dir.expanduser().resolve()
    source_dir = resolve_source_dir(repo_dir, cli.source_dir)
    data_root = resolve_relative(cli.data_root, repo_dir)
    collected_results_root = resolve_relative(
        cli.collected_results_root,
        repo_dir,
    )

    run_dir = resolve_run_dir(
        repo_dir,
        data_root,
        cli.run_dir,
        cli.run_name,
        cli.run_id,
    )
    config = load_config(run_dir, cli.device)
    init_module = import_init(source_dir)

    if str(getattr(config, "model", "gpt2")).lower() != "gpt2":
        raise ValueError(
            "This diagnostic requires --model gpt2 because it separates "
            "Transformer attention and MLP parameter groups."
        )

    if cli.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False."
        )
    device = torch.device(cli.device)

    all_checkpoints = find_checkpoints(run_dir, cli.max_step)
    selected_checkpoints, selected_indices = select_log_spaced_checkpoints(
        all_checkpoints,
        cli.num_checkpoints,
    )
    selected_steps = np.asarray(
        [int(checkpoint_step(path)) for path in selected_checkpoints],
        dtype=np.int64,
    )

    steps_per_epoch = int(getattr(config, "steps_per_epoch", 0) or 0)
    if steps_per_epoch <= 0:
        train_size = int(getattr(config, "train_size", 0) or 0)
        training_batch_size = int(getattr(config, "batch_size", 1) or 1)
        if str(getattr(config, "dataset", "")).lower() == "rhm":
            steps_per_epoch = max(1, math.ceil(train_size / training_batch_size))
        else:
            block_size = int(getattr(config, "block_size", 1) or 1)
            steps_per_epoch = max(
                1,
                train_size // max(1, training_batch_size * block_size),
            )
    selected_epochs = selected_steps.astype(np.float64) / float(steps_per_epoch)

    output_name = (
        cli.output_name
        or f"component_linearization_{run_dir.parent.name}_{run_dir.name}"
    )
    output_dir = collected_results_root / output_name

    print(f"[INFO] diagnostic_version={DIAGNOSTIC_VERSION}", flush=True)
    print(f"[INFO] run_dir={run_dir}", flush=True)
    print(f"[INFO] source_dir={source_dir}", flush=True)
    print(f"[INFO] output_dir={output_dir}", flush=True)
    print(f"[INFO] available checkpoints={len(all_checkpoints)}", flush=True)
    print(
        f"[INFO] selected checkpoint indices={selected_indices.tolist()}",
        flush=True,
    )
    print(f"[INFO] selected steps={selected_steps.tolist()}", flush=True)
    print(
        f"[INFO] common baseline selected_index=0 step={int(selected_steps[0])}",
        flush=True,
    )

    model = init_module.init_model(config)
    model.to(device).eval()
    model_parameter_templates = {
        name: parameter.detach()
        for name, parameter in model.named_parameters()
    }
    model_buffer_templates = {
        name: buffer.detach()
        for name, buffer in model.named_buffers()
    }

    layer_names, component_groups = discover_transformer_component_groups(
        list(model_parameter_templates)
    )
    print(f"[INFO] layer_names={layer_names}", flush=True)
    for component in COMPONENTS:
        for layer_name in layer_names:
            print(
                f"[INFO] group component={component} layer={layer_name} "
                f"parameters={component_groups[component][layer_name]}",
                flush=True,
            )

    baseline_state = load_state_dict_cpu(selected_checkpoints[0])
    baseline_parameters = state_parameters(
        baseline_state,
        model_parameter_templates,
        device,
    )
    baseline_buffers = state_buffers(
        baseline_state,
        model_buffer_templates,
        device,
    )

    reference_cli = SimpleNamespace(
        subset_train_size=int(cli.subset_size),
        subset_valid_size=int(cli.subset_size),
        subset_seed=int(cli.subset_seed),
        reference_source=str(cli.reference_source),
    )
    train_reference, test_reference, data_metadata = load_references(
        config,
        reference_cli,
        repo_dir,
        source_dir,
        run_dir,
    )

    split_sequences: Dict[str, torch.Tensor] = {}
    if cli.split in ("train", "both"):
        split_sequences["train"] = train_reference.contiguous()
    if cli.split in ("test", "both"):
        split_sequences["test"] = test_reference.contiguous()

    number_positions = int(next(iter(split_sequences.values())).shape[1])
    split_arrays = {
        split: allocate_result_arrays(
            len(selected_checkpoints),
            len(layer_names),
            number_positions,
        )
        for split in split_sequences
    }

    derivative = DirectionalDerivative(
        cli.jvp_method,
        cli.finite_difference_epsilon,
    )
    started = time.time()

    for target_index, target_checkpoint in enumerate(selected_checkpoints):
        target_step = int(selected_steps[target_index])
        print(
            f"[TARGET {target_index + 1}/{len(selected_checkpoints)}] "
            f"baseline_step={int(selected_steps[0])} target_step={target_step} "
            f"file={target_checkpoint.name}",
            flush=True,
        )

        target_state = (
            baseline_state
            if target_index == 0
            else load_state_dict_cpu(target_checkpoint)
        )
        target_parameters = (
            baseline_parameters
            if target_index == 0
            else state_parameters(
                target_state,
                model_parameter_templates,
                device,
            )
        )

        for component_index, component in enumerate(COMPONENTS):
            for layer_index, layer_name in enumerate(layer_names):
                group_names = component_groups[component][layer_name]

                for split, sequences in split_sequences.items():
                    group_started = time.time()
                    result = evaluate_group(
                        model=model,
                        baseline_parameters=baseline_parameters,
                        baseline_buffers=baseline_buffers,
                        target_parameters=target_parameters,
                        group_names=group_names,
                        sequences=sequences,
                        batch_size=cli.batch_size,
                        device=device,
                        center=bool(cli.center_logits),
                        derivative=derivative,
                    )

                    for name in SCALAR_METRICS:
                        split_arrays[split][name][
                            target_index, component_index, layer_index
                        ] = result[name]
                    for name in POSITION_METRICS:
                        split_arrays[split][name][
                            target_index, component_index, layer_index
                        ] = result[name]

                    print(
                        f"[RESULT] split={split} baseline_step={int(selected_steps[0])} "
                        f"target_step={target_step} component={component} "
                        f"layer={layer_index + 1} "
                        f"relative_error={result['relative_error']:.8g} "
                        f"alignment={result['alignment']:.8g} "
                        f"amplitude_ratio={result['amplitude_ratio']:.8g} "
                        f"relative_parameter_displacement="
                        f"{result['relative_parameter_displacement']:.8g} "
                        f"time={time.time() - group_started:.2f}s",
                        flush=True,
                    )

                if device.type == "cuda":
                    torch.cuda.empty_cache()

        if target_index != 0:
            del target_parameters
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metadata = {
        "diagnostic": "layerwise_attention_mlp_linearization_error",
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "definition": (
            "For each layer-component group g: "
            "||[f(theta0 with g replaced by g_t)-f(theta0)]-J_g(theta0)(g_t-g_0)||_F "
            "/ ||f(theta0 with g replaced by g_t)-f(theta0)||_F"
        ),
        "baseline_rule": (
            "The earliest of the log-spaced selected checkpoints is the single "
            "common baseline for every selected target."
        ),
        "actual_change_rule": (
            "Group-isolated hybrid model: only the selected attention or MLP "
            "group in one layer is moved to its target checkpoint value; all "
            "other trainable parameters and buffers remain at baseline."
        ),
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "source_dir": str(source_dir),
        "device": str(device),
        "split": cli.split,
        "subset_size": int(cli.subset_size),
        "subset_seed": int(cli.subset_seed),
        "batch_size": int(cli.batch_size),
        "center_logits": bool(cli.center_logits),
        "num_available_checkpoints": len(all_checkpoints),
        "num_selected_checkpoints": len(selected_checkpoints),
        "selected_checkpoint_indices": selected_indices.tolist(),
        "selected_steps": selected_steps.tolist(),
        "selected_epochs": selected_epochs.tolist(),
        "baseline_selected_index": 0,
        "baseline_step": int(selected_steps[0]),
        "baseline_epoch": float(selected_epochs[0]),
        "checkpoint_files": [str(path) for path in selected_checkpoints],
        "steps_per_epoch": int(steps_per_epoch),
        "components": list(COMPONENTS),
        "layer_names": list(layer_names),
        "component_parameter_groups": component_groups,
        "reference_data": data_metadata,
        "requested_jvp_method": cli.jvp_method,
        "resolved_jvp_method": derivative.resolved_method,
        "finite_difference_epsilon": float(cli.finite_difference_epsilon),
        "total_seconds": float(time.time() - started),
        "config": {
            key: value
            for key, value in vars(config).items()
            if isinstance(value, (str, int, float, bool)) or value is None
        },
    }

    save_outputs(
        output_dir=output_dir,
        selected_checkpoints=selected_checkpoints,
        selected_indices=selected_indices,
        selected_steps=selected_steps,
        selected_epochs=selected_epochs,
        layer_names=layer_names,
        component_groups=component_groups,
        split_arrays=split_arrays,
        metadata=metadata,
    )

    print(f"[DONE] diagnostics saved in {output_dir}", flush=True)
    print(f"[DONE] total time={time.time() - started:.2f}s", flush=True)


if __name__ == "__main__":
    main()
