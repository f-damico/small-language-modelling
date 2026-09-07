"""Plot predictive representation overlaps produced by
compute_data2vec_predictive_representation_overlaps.py.

The main public function is

    plot_data2vec_predictive_same_layer_overlap_vs_time(...)

The loader understands the data2vec output tensor

    predictive_q_by_position[
        token_position,
        source_layer,
        target_layer,
        source_checkpoint,
    ].
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _as_python_scalar(value: Any) -> Any:
    """Convert a zero-dimensional NumPy value to a Python scalar."""
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    return value


def _string_list(value: np.ndarray) -> list[str]:
    """Decode a NumPy string/object array into a list of strings."""
    return [str(item) for item in np.asarray(value).reshape(-1).tolist()]


def _resolve_data2vec_predictive_npz(
    results_dir: str | Path,
    output_name: Optional[str] = None,
) -> Path:
    """Resolve the data2vec ``predictive_overlaps.npz`` file.

    Accepted forms
    --------------
    1. results_dir is the .npz file itself.
    2. results_dir is one overlap-output directory.
    3. results_dir is the output root and output_name is the output directory.
    """
    root = Path(results_dir).expanduser()

    if root.suffix == ".npz":
        path = root
    elif output_name is None:
        path = root / "predictive_overlaps.npz"
    else:
        path = root / str(output_name) / "predictive_overlaps.npz"

    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            "Could not find data2vec predictive-overlap results at\n"
            f"    {path}\n"
            "Pass either the output directory itself, or the output root together "
            "with output_name."
        )
    return path


def _safe_nanmean(
    values: np.ndarray,
    axis: int | tuple[int, ...],
) -> np.ndarray:
    """NaN-aware mean without RuntimeWarning for entirely missing slices."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    count = finite.sum(axis=axis)
    total = np.where(finite, values, 0.0).sum(axis=axis)
    return np.divide(
        total,
        count,
        out=np.full(np.shape(total), np.nan, dtype=float),
        where=count > 0,
    )


def _safe_nanargmax_last(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Argmax over the last axis, returning -1 and NaN for all-NaN slices."""
    values = np.asarray(values, dtype=float)
    finite_any = np.any(np.isfinite(values), axis=-1)
    filled = np.where(np.isfinite(values), values, -np.inf)
    indices = np.argmax(filled, axis=-1)
    indices = np.where(finite_any, indices, -1)

    chosen = np.full(indices.shape, np.nan, dtype=float)
    valid = indices >= 0
    if np.any(valid):
        chosen[valid] = np.take_along_axis(
            values[valid],
            indices[valid, None],
            axis=-1,
        )[:, 0]
    return indices, chosen


def _resolve_position_indices(
    position: str | int | np.integer | Sequence[int],
    token_positions_1based: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Resolve physical, one-based token positions.

    ``position="mean"`` or ``"all"`` selects all token positions.
    Integer positions are interpreted as the one-based token labels saved in
    ``token_positions_1based``.
    """
    labels = np.asarray(token_positions_1based, dtype=int).reshape(-1)

    if isinstance(position, str):
        key = position.strip().lower()
        if key in {"mean", "all"}:
            return np.arange(len(labels), dtype=int), "mean"
        if key.startswith("position_"):
            position = int(key.split("_", 1)[1])
        else:
            try:
                position = int(key)
            except ValueError as exc:
                raise ValueError(
                    "position must be 'mean', 'all', an integer one-based token "
                    "position, or a sequence of such integers."
                ) from exc

    if isinstance(position, (int, np.integer)):
        requested = [int(position)]
    else:
        requested = [int(item) for item in position]

    indices = []
    for item in requested:
        matches = np.where(labels == item)[0]
        if len(matches) != 1:
            raise ValueError(
                f"Token position {item} is not available. "
                f"Available one-based positions: {labels.tolist()}"
            )
        indices.append(int(matches[0]))

    return np.asarray(indices, dtype=int), ",".join(str(labels[i]) for i in indices)


def load_data2vec_predictive_results(
    results_dir: str | Path,
    output_name: Optional[str] = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load one data2vec predictive-overlap output."""
    path = _resolve_data2vec_predictive_npz(results_dir, output_name)

    with np.load(path, allow_pickle=True) as payload:
        data = {key: payload[key] for key in payload.files}

    metadata_path = path.with_name("metadata.json")
    metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            metadata = {}

    required = {
        "predictive_q_by_position",
        "source_layer_names",
        "target_layer_names",
        "source_steps",
        "target_step",
    }
    missing = sorted(required.difference(data))
    if missing:
        raise KeyError(
            f"{path} is missing required data2vec overlap arrays: {missing}"
        )

    q = np.asarray(data["predictive_q_by_position"])
    if q.ndim != 4:
        raise ValueError(
            "Expected predictive_q_by_position with shape "
            "[position,source_layer,target_layer,time], "
            f"got {q.shape}"
        )

    info = {
        "npz_path": str(path),
        "metadata_path": str(metadata_path) if metadata_path.is_file() else None,
        "metadata": metadata,
        "representation_mode": str(
            _as_python_scalar(data.get("representation_mode", "unknown"))
        ),
        "source_encoder": str(
            _as_python_scalar(data.get("source_encoder", "unknown"))
        ),
        "target_encoder": str(
            _as_python_scalar(data.get("target_encoder", "unknown"))
        ),
        "predictive_layer_mode": str(
            _as_python_scalar(data.get("predictive_layer_mode", "unknown"))
        ),
    }
    return data, info


def get_data2vec_predictive_all_layers_q(
    results_dir: str | Path,
    output_name: Optional[str] = None,
    *,
    position: str | int | np.integer | Sequence[int] = "mean",
    alpha_mode: str = "position_mean",
    use_saved_best: bool = False,
) -> tuple[np.ndarray, list[str], list[str], dict[str, np.ndarray], dict[str, Any]]:
    """Return q[source_layer,target_layer,time] for data2vec.

    Parameters
    ----------
    position
        ``"mean"``/``"all"`` averages over all token positions. An integer is a
        physical one-based token position. A sequence averages over the selected
        one-based positions.

    alpha_mode
        Used only when ``use_saved_best=False``.

        ``"position_mean"``
            For every source-layer/target-layer/time triple, first average the
            validation score across the selected token positions for each ridge
            alpha, then choose the alpha with the largest position-mean score.
            This applies one common alpha to all selected positions.

        ``"per_position"``
            Select the best alpha independently at every token position, then
            average the resulting best scores across positions.

    use_saved_best
        Use ``predictive_q_by_position`` exactly as saved by the computation.
        The saved tensor uses an independently selected best alpha at every
        token position. It is then averaged over the selected positions.
    """
    data, info = load_data2vec_predictive_results(
        results_dir,
        output_name=output_name,
    )

    q_saved = np.asarray(data["predictive_q_by_position"], dtype=float)
    source_names = _string_list(data["source_layer_names"])
    target_names = _string_list(data["target_layer_names"])

    if len(source_names) != q_saved.shape[1]:
        raise ValueError(
            f"{len(source_names)} source names for source-layer axis {q_saved.shape[1]}"
        )
    if len(target_names) != q_saved.shape[2]:
        raise ValueError(
            f"{len(target_names)} target names for target-layer axis {q_saved.shape[2]}"
        )

    token_positions = np.asarray(
        data.get(
            "token_positions_1based",
            np.arange(1, q_saved.shape[0] + 1, dtype=int),
        ),
        dtype=int,
    ).reshape(-1)

    position_indices, position_label = _resolve_position_indices(
        position,
        token_positions,
    )

    if use_saved_best:
        q = _safe_nanmean(q_saved[position_indices], axis=0)
        info.update(
            {
                "q_source": "predictive_q_by_position",
                "alpha_mode_used": "saved_per_position_best",
                "selected_alpha_indices": None,
                "selected_alphas": None,
            }
        )
    else:
        if "predictive_alpha_scores_by_position" not in data:
            raise KeyError(
                "The result file has no predictive_alpha_scores_by_position. "
                "Set use_saved_best=True."
            )

        alpha_scores = np.asarray(
            data["predictive_alpha_scores_by_position"],
            dtype=float,
        )
        expected = q_saved.shape + (
            len(np.asarray(data.get("ridge_alphas", []))),
        )
        if alpha_scores.shape[:4] != q_saved.shape:
            raise ValueError(
                "predictive_alpha_scores_by_position has incompatible shape "
                f"{alpha_scores.shape}; expected first four axes {q_saved.shape}"
            )

        selected_scores = alpha_scores[position_indices]
        mode = alpha_mode.strip().lower()

        if mode in {"position_mean", "mean_position", "common"}:
            # [source_layer, target_layer, time, alpha]
            mean_scores = _safe_nanmean(selected_scores, axis=0)
            best_indices, q = _safe_nanargmax_last(mean_scores)
            mode_used = "position_mean_common_alpha"

        elif mode in {"per_position", "position", "independent"}:
            # [selected_position, source_layer, target_layer, time]
            _, best_by_position = _safe_nanargmax_last(selected_scores)
            q = _safe_nanmean(best_by_position, axis=0)
            best_indices = None
            mode_used = "independent_alpha_per_position"

        else:
            raise ValueError(
                "alpha_mode must be 'position_mean' or 'per_position'"
            )

        ridge_alphas = np.asarray(data.get("ridge_alphas", []), dtype=float)
        selected_alphas = None
        if best_indices is not None and ridge_alphas.size:
            selected_alphas = np.full(best_indices.shape, np.nan, dtype=float)
            valid = best_indices >= 0
            selected_alphas[valid] = ridge_alphas[best_indices[valid]]

        info.update(
            {
                "q_source": "predictive_alpha_scores_by_position",
                "alpha_mode_used": mode_used,
                "selected_alpha_indices": best_indices,
                "selected_alphas": selected_alphas,
            }
        )

    if q.ndim != 3:
        raise RuntimeError(
            f"Internal error: expected q[source_layer,target_layer,time], got {q.shape}"
        )

    info.update(
        {
            "position_request": position,
            "position_label": position_label,
            "position_indices": position_indices,
            "token_positions_1based": token_positions,
            "selected_token_positions_1based": token_positions[position_indices],
            "use_saved_best": bool(use_saved_best),
        }
    )
    return q, source_names, target_names, data, info


def _choose_data2vec_predictive_time_axis(
    data: dict[str, np.ndarray],
    *,
    time_key: str = "steps",
) -> tuple[np.ndarray, str]:
    """Choose a time axis from the arrays saved by the data2vec computation."""
    aliases = {
        "steps": "source_steps",
        "step": "source_steps",
        "source_steps": "source_steps",
        "training_samples_seen": "source_training_samples_seen",
        "samples_seen": "source_training_samples_seen",
        "source_training_samples_seen": "source_training_samples_seen",
        "p": "source_training_samples_seen",
        "index": "source_checkpoint_index",
        "checkpoint_index": "source_checkpoint_index",
    }

    normalized = str(time_key).strip().lower()
    resolved = aliases.get(normalized, time_key)

    if resolved == "source_checkpoint_index":
        n_times = np.asarray(data["predictive_q_by_position"]).shape[-1]
        return np.arange(n_times, dtype=int), "checkpoint_index"

    if resolved not in data:
        available = [
            key
            for key in (
                "source_steps",
                "source_training_samples_seen",
            )
            if key in data
        ]
        raise KeyError(
            f"Time key {time_key!r} is not available. "
            f"Available data2vec time axes: {available + ['checkpoint_index']}"
        )

    times = np.asarray(data[resolved]).reshape(-1)
    label = {
        "source_steps": "steps",
        "source_training_samples_seen": "training samples seen",
    }.get(resolved, resolved)
    return times, label


def _data2vec_target_time_label(
    data: dict[str, np.ndarray],
    actual_time_key: str,
) -> Optional[float]:
    """Return the target checkpoint coordinate on the chosen time axis."""
    if actual_time_key == "steps":
        key = "target_step"
    elif actual_time_key == "training samples seen":
        key = "target_training_samples_seen"
    else:
        return None

    if key not in data:
        return None

    value = np.asarray(data[key])
    if value.size != 1:
        return None

    target = float(value.reshape(()))
    return target if np.isfinite(target) else None


def plot_data2vec_predictive_same_layer_overlap_vs_time(
    results_dir,
    output_name=None,
    *,
    position="mean",
    alpha_mode="position_mean",
    use_saved_best=False,
    time_key="steps",
    layers=None,
    x_mode="absolute",
    xscale="log",
    yscale="linear",
    ylim=None,
    figsize=(7.2, 5.0),
    linewidth=2.6,
    alpha=0.95,
    marker=None,
    cmap="turbo",
    cmap_min=0.05,
    cmap_max=0.85,
    legend_fontsize=8,
    grid=True,
    add_zero_line=True,
    add_one_line=True,
    title=None,
):
    """Plot the same-layer data2vec predictive overlap q_{l -> l}(t,t').

    Each curve corresponds to one data2vec representation layer. The target
    checkpoint t' is the single checkpoint selected when the overlap job was
    run.

    The data2vec result tensor is position-resolved, and both source and target
    representations refer to the same physical token position.
    """
    import matplotlib as mpl

    q, source_names, target_names, data, info = (
        get_data2vec_predictive_all_layers_q(
            results_dir,
            output_name=output_name,
            position=position,
            alpha_mode=alpha_mode,
            use_saved_best=use_saved_best,
        )
    )

    if q.ndim != 3:
        raise ValueError(
            f"Expected q[source_layer,target_layer,time], got shape {q.shape}"
        )

    if q.shape[0] != q.shape[1]:
        raise ValueError(
            "Same-layer plotting requires the same number of source and target "
            f"layers. Got source={q.shape[0]}, target={q.shape[1]}."
        )

    if source_names != target_names:
        raise ValueError(
            "Source and target representation names differ, so index-wise "
            "same-layer curves are ambiguous.\n"
            f"source={source_names}\n"
            f"target={target_names}"
        )

    times, actual_time_key = _choose_data2vec_predictive_time_axis(
        data,
        time_key=time_key,
    )
    if len(times) != q.shape[2]:
        raise ValueError(
            f"Time axis has length {len(times)} but q has {q.shape[2]} source times"
        )

    target_time = _data2vec_target_time_label(data, actual_time_key)

    def resolve_layers(selection, names):
        if selection is None:
            return np.arange(len(names), dtype=int)

        if isinstance(selection, (str, int, np.integer)):
            selection = [selection]

        out = []
        for item in selection:
            if isinstance(item, str):
                if item not in names:
                    raise ValueError(
                        f"Layer {item!r} not found. Available: {names}"
                    )
                out.append(names.index(item))
            else:
                j = int(item)
                if j < 0:
                    j += len(names)
                if j < 0 or j >= len(names):
                    raise ValueError(
                        f"Layer index {item} outside "
                        f"[-{len(names)},{len(names)-1}]"
                    )
                out.append(j)

        # Preserve order while removing accidental duplicates.
        return np.asarray(list(dict.fromkeys(out)), dtype=int)

    layer_idx = resolve_layers(layers, source_names)

    if x_mode == "absolute":
        x = times.astype(float, copy=True)
        xlabel = rf"$t$ ({actual_time_key})"

    elif x_mode == "delta_to_target":
        if target_time is None:
            raise ValueError(
                "x_mode='delta_to_target' requires target_step or "
                "target_training_samples_seen in the npz."
            )
        x = times.astype(float) - float(target_time)
        xlabel = rf"$t-t'$ ({actual_time_key})"

    elif x_mode == "distance_to_target":
        if target_time is None:
            raise ValueError(
                "x_mode='distance_to_target' requires target_step or "
                "target_training_samples_seen in the npz."
            )
        x = float(target_time) - times.astype(float)
        xlabel = rf"$t'-t$ ({actual_time_key})"

    else:
        raise ValueError(
            "x_mode must be 'absolute', 'delta_to_target', or "
            "'distance_to_target'"
        )

    mask = np.isfinite(x)
    if xscale == "log":
        mask &= x > 0

    if not np.any(mask):
        raise ValueError(
            "No source times remain after applying the x-axis mask. "
            "For a final target checkpoint, x_mode='delta_to_target' gives "
            "non-positive values and cannot be shown on a logarithmic axis. "
            "Use x_mode='absolute', xscale='linear', or "
            "x_mode='distance_to_target'."
        )

    x_plot = x[mask]
    time_idx = np.where(mask)[0]

    colormap = mpl.colormaps.get_cmap(cmap)
    color_values = np.linspace(
        float(cmap_min),
        float(cmap_max),
        len(layer_idx),
    )
    layer_colors = {
        int(j): colormap(color_values[k])
        for k, j in enumerate(layer_idx)
    }

    fig, ax = plt.subplots(figsize=figsize)

    for j in layer_idx:
        y = q[j, j, time_idx]
        ax.plot(
            x_plot,
            y,
            marker=marker,
            linewidth=linewidth,
            alpha=alpha,
            color=layer_colors[int(j)],
            label=source_names[j],
        )

    if add_zero_line:
        ax.axhline(0.0, color="black", lw=0.8, alpha=0.35)

    if add_one_line:
        ax.axhline(
            1.0,
            color="black",
            lw=0.8,
            alpha=0.25,
            linestyle="--",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(
        r"same-layer predictive overlap $q_{l\to l}(t,t')$"
    )
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    if ylim is not None:
        ax.set_ylim(*ylim)

    if grid:
        ax.grid(True, alpha=0.3)

    if title is None:
        source_encoder = info.get("source_encoder", "source")
        target_encoder = info.get("target_encoder", "target")
        representation_mode = info.get("representation_mode", "representation")

        title_eff = (
            "data2vec same-layer predictive representation overlap"
            "\n"
            f"{source_encoder} → {target_encoder}; "
            f"mode={representation_mode}; "
            f"positions={info['position_label']}"
        )
        if target_time is not None:
            title_eff += rf"; fixed $t'={target_time:g}$ {actual_time_key}"
    else:
        title_eff = title

    ax.set_title(title_eff)
    ax.legend(fontsize=legend_fontsize, frameon=False)
    fig.tight_layout()

    info.update(
        {
            "time_key_used": actual_time_key,
            "source_times": times,
            "source_time_plot_mask": mask,
            "target_time": target_time,
            "layer_indices": layer_idx,
            "plotted_layers": [source_names[i] for i in layer_idx],
            "same_layer_only": True,
            "x_mode": x_mode,
            "xscale": xscale,
            "yscale": yscale,
            "cmap": cmap,
            "cmap_min": cmap_min,
            "cmap_max": cmap_max,
        }
    )

    return fig, ax, info


# Convenient shorter alias.
plot_predictive_same_layer_overlap_vs_time_data2vec = (
    plot_data2vec_predictive_same_layer_overlap_vs_time
)
