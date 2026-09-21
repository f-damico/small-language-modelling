from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt


def _find_ntp_overlap_npz(results_dir, output_name=None, overlap_target="auto"):
    """Resolve one NTP overlap result file.

    Supported ``overlap_target`` values are

        auto
        representations   # backward-compatible umbrella for representation-like outputs
        cka
        block_update_cka
        weights
        low_rank
        novelty
        task

    ``representations`` first looks for the historical CKA file.  If that file
    is absent, it also accepts exactly one of the new representation-derived
    files (low-rank, novelty, task).  This keeps old notebook calls working for
    result directories that contain a single new overlap file.
    """
    results_dir = Path(results_dir).expanduser()
    target = str(overlap_target).strip().lower()

    aliases = {
        "representation": "representations",
        "rep": "representations",
        "state_cka": "cka",
        "block_update": "block_update_cka",
        "block_cka": "block_update_cka",
        "lowrank": "low_rank",
        "task_relevant": "task",
    }
    target = aliases.get(target, target)

    allowed = {
        "auto",
        "representations",
        "cka",
        "block_update_cka",
        "weights",
        "low_rank",
        "novelty",
        "task",
    }
    if target not in allowed:
        raise ValueError(
            "overlap_target must be one of "
            f"{sorted(allowed)}, got {overlap_target!r}."
        )

    filenames = {
        "cka": "representation_overlaps_per_position.npz",
        "block_update_cka": "representation_overlaps_per_position.npz",
        "weights": "weight_overlaps.npz",
        "low_rank": "low_rank_overlaps.npz",
        "novelty": "novelty_overlaps.npz",
        "task": "task_overlaps.npz",
    }
    representation_files = [
        filenames["cka"],
        filenames["low_rank"],
        filenames["novelty"],
        filenames["task"],
    ]
    all_files = [filenames["weights"], *representation_files]

    if results_dir.is_file():
        if results_dir.name in all_files:
            return results_dir.resolve()
        raise FileNotFoundError(
            "Expected one of the NTP overlap result files\n    "
            + "\n    ".join(all_files)
            + f"\nbut got {results_dir}"
        )

    base = results_dir / str(output_name) if output_name is not None else results_dir

    if target == "auto":
        wanted = all_files
    elif target == "representations":
        wanted = representation_files
    else:
        wanted = [filenames[target]]

    # Prefer files directly in the requested output directory.
    direct = [base / name for name in wanted if (base / name).is_file()]
    if len(direct) == 1:
        return direct[0].resolve()
    if len(direct) > 1:
        # Backward-compatible preference: the historical representation file
        # wins for overlap_target='representations'. Otherwise ambiguity is real.
        old_rep = base / filenames["cka"]
        if target == "representations" and old_rep in direct:
            return old_rep.resolve()
        raise RuntimeError(
            "Found more than one compatible overlap file in the output directory. "
            "Set overlap_target explicitly.\n"
            + "\n".join(str(path) for path in direct)
        )

    # Recursive fallback is useful when results_dir is the collected-results root.
    found = []
    for name in wanted:
        found.extend(sorted(base.glob(f"**/{name}")))
    found = sorted(set(path.resolve() for path in found))

    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        raise RuntimeError(
            "Found more than one compatible overlap .npz file. "
            "Specify output_name, pass the exact output folder, or set "
            "overlap_target explicitly.\n"
            + "\n".join(str(path) for path in found[:30])
        )

    raise FileNotFoundError(
        f"Could not find a compatible overlap file under {base}. "
        f"Expected one of: {wanted}"
    )

def _find_ntp_representation_overlap_npz(results_dir, output_name=None):
    """
    Backward-compatible alias for old notebooks.
    """
    return _find_ntp_overlap_npz(
        results_dir,
        output_name=output_name,
        overlap_target="representations",
    )


def _decode_np_string_array(arr):
    arr = np.asarray(arr)

    out = []
    for x in arr.tolist():
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))

    return out


def _np_scalar_to_str(x):
    arr = np.asarray(x)
    if arr.shape == ():
        val = arr.item()
    else:
        val = arr.tolist()

    if isinstance(val, bytes):
        return val.decode("utf-8")

    return str(val)


def _infer_overlap_target(data, npz_path=None):
    """Infer the precise overlap definition stored in an NTP result file.

    Returns one of
        cka, block_update_cka, weights, low_rank, novelty, task.
    """
    if "overlap_definition" in data.files:
        target = _np_scalar_to_str(data["overlap_definition"]).lower()
        if target in {"low_rank", "novelty", "task", "cka", "block_update_cka"}:
            return target

    if "overlap_target" in data.files:
        target = _np_scalar_to_str(data["overlap_target"]).lower()
        if target == "weights":
            return "weights"
        if target == "representations":
            mode = (
                _np_scalar_to_str(data["representation_mode"]).lower()
                if "representation_mode" in data.files
                else "state"
            )
            return "block_update_cka" if mode == "block_update" else "cka"

    if "weight_overlap_all" in data.files or "weight_overlap_by_group" in data.files:
        return "weights"

    if "low_rank_overlap_by_position" in data.files:
        return "low_rank"
    if "task_target" in data.files:
        return "task"
    if "novelty_ridge_alpha" in data.files:
        return "novelty"

    if "test_cka_position_mean" in data.files or "train_cka_position_mean" in data.files:
        mode = (
            _np_scalar_to_str(data["representation_mode"]).lower()
            if "representation_mode" in data.files
            else "state"
        )
        return "block_update_cka" if mode == "block_update" else "cka"

    if npz_path is not None:
        name = Path(npz_path).name
        by_name = {
            "weight_overlaps.npz": "weights",
            "representation_overlaps_per_position.npz": "cka",
            "low_rank_overlaps.npz": "low_rank",
            "novelty_overlaps.npz": "novelty",
            "task_overlaps.npz": "task",
        }
        if name in by_name:
            inferred = by_name[name]
            if inferred == "cka" and "representation_mode" in data.files:
                mode = _np_scalar_to_str(data["representation_mode"]).lower()
                if mode == "block_update":
                    return "block_update_cka"
            return inferred

    raise ValueError(
        "Could not infer overlap definition from the npz contents."
    )

def _choose_time_axis(data, time_key="steps"):
    """
    Use selected_steps by default.
    Fallback: selected_epochs.
    """
    if time_key in ("steps", "step", "global_updates", "global_update"):
        if "selected_steps" in data.files:
            t = np.asarray(data["selected_steps"], dtype=float)
            if t.size > 0 and np.any(np.isfinite(t)) and np.any(t >= 0):
                return t, "steps"

    if time_key in ("epochs", "epoch"):
        if "selected_epochs" in data.files:
            return np.asarray(data["selected_epochs"], dtype=float), "epochs"

    if "selected_steps" in data.files:
        t = np.asarray(data["selected_steps"], dtype=float)
        if t.size > 0 and np.any(np.isfinite(t)) and np.any(t >= 0):
            return t, "steps"

    if "selected_epochs" in data.files:
        return np.asarray(data["selected_epochs"], dtype=float), "epochs"

    raise KeyError("Could not find selected_steps or selected_epochs in the npz file.")


def _resolve_reference_indices(times, reference_times=None, reference_indices=None):
    """
    Convert selected t' values into checkpoint indices.
    """
    times = np.asarray(times, dtype=float)

    if reference_indices is not None:
        ref_idx = np.asarray(reference_indices, dtype=int)
        if np.any(ref_idx < 0) or np.any(ref_idx >= len(times)):
            raise ValueError(f"reference_indices must be in [0, {len(times)-1}]")
        return np.unique(ref_idx)

    if reference_times is None:
        return np.unique(np.asarray([0, len(times) // 2, len(times) - 1], dtype=int))

    ref_idx = []
    for t0 in reference_times:
        j = int(np.argmin(np.abs(times - float(t0))))
        ref_idx.append(j)

    return np.unique(np.asarray(ref_idx, dtype=int))


def _resolve_layer_indices(layer_names, layers=None):
    """
    layers can be None, list of int, or list of layer names.
    """
    if layers is None:
        return np.arange(len(layer_names), dtype=int)

    out = []

    for layer in layers:
        if isinstance(layer, str):
            if layer not in layer_names:
                raise ValueError(f"Layer {layer!r} not found. Available: {layer_names}")
            out.append(layer_names.index(layer))
        else:
            j = int(layer)
            if j < 0 or j >= len(layer_names):
                raise ValueError(f"Layer index {j} outside [0, {len(layer_names)-1}]")
            out.append(j)

    return np.asarray(out, dtype=int)


def _resolve_weight_indices(names, selection=None, kind="weight group"):
    """
    Resolve selected weight groups/parameters.

    selection can be:
        None
        list of int
        list of names
        single int
        single name
    """
    if selection is None:
        return np.arange(len(names), dtype=int)

    if isinstance(selection, (str, int, np.integer)):
        selection = [selection]

    out = []

    for item in selection:
        if isinstance(item, str):
            if item not in names:
                raise ValueError(f"{kind} {item!r} not found. Available: {names}")
            out.append(names.index(item))
        else:
            j = int(item)
            if j < 0 or j >= len(names):
                raise ValueError(f"{kind} index {j} outside [0, {len(names)-1}]")
            out.append(j)

    return np.asarray(out, dtype=int)


def _select_representation_q_array(
    data,
    split="test",
    position="mean",
    overlap_definition=None,
):
    """Select a representation-like overlap array.

    Returns
    -------
    q : ndarray, shape [num_layers, K, K]
        Position-selected overlap matrix.

    The historical CKA outputs support train/test.  The new low-rank, novelty,
    and task definitions are evaluated on the held-out validation reference;
    their files expose both ``valid_*`` and ``test_*`` aliases for convenience.
    """
    split = str(split).lower()
    if split not in ("train", "valid", "test"):
        raise ValueError("split must be 'train', 'valid', or 'test'.")

    definition = (
        str(overlap_definition).lower()
        if overlap_definition is not None
        else _infer_overlap_target(data)
    )

    if definition in {"low_rank", "novelty", "task"}:
        # New methods all use the same generic keys.  'test' is an alias of
        # the held-out validation set written by compute_slm_ntp_overlaps.py.
        if split == "train":
            raise ValueError(
                f"{definition!r} overlap is only evaluated on the held-out "
                "validation/test reference. Use split='test' or split='valid'."
            )
        prefix = "valid" if split == "valid" else "test"
        mean_key = f"{prefix}_overlap_position_mean"
        bypos_key = f"{prefix}_overlap_by_position"
    else:
        # Historical representation CKA files expose train/test keys.
        if split == "valid":
            split = "test"
        mean_key = f"{split}_cka_position_mean"
        bypos_key = f"{split}_cka_by_position"

    if position == "mean" or position is None:
        if mean_key not in data.files:
            raise KeyError(
                f"{mean_key!r} not found. Available keys: {data.files}"
            )
        q = np.asarray(data[mean_key], dtype=float)
        return q, "position average", {
            "position_mode": "mean",
            "position_indices": None,
            "target_positions_1based": None,
            "source_array": mean_key,
            "overlap_definition": definition,
        }

    if bypos_key not in data.files:
        raise KeyError(f"{bypos_key!r} not found. Available keys: {data.files}")

    q_bypos = np.asarray(data[bypos_key], dtype=float)
    num_pos = q_bypos.shape[0]

    if "target_token_positions_1based" in data.files:
        target_positions = np.asarray(data["target_token_positions_1based"], dtype=int)
    else:
        target_positions = np.arange(2, 2 + num_pos, dtype=int)

    def resolve_one_position(pos):
        p = int(pos)
        matches = np.where(target_positions == p)[0]
        if len(matches) > 0:
            return int(matches[0])
        if 0 <= p < num_pos:
            return p
        raise ValueError(
            f"Invalid position={p}. Use one of target positions "
            f"{target_positions.tolist()} or zero-based indices 0..{num_pos-1}."
        )

    if isinstance(position, (list, tuple, np.ndarray)):
        pos_idx = np.asarray([resolve_one_position(p) for p in position], dtype=int)
        pos_idx = np.unique(pos_idx)
        q = np.nanmean(q_bypos[pos_idx], axis=0)
        selected_targets = target_positions[pos_idx]
        label = "positions " + ",".join(str(x) for x in selected_targets.tolist())
        mode = "selected_mean"
    else:
        pos_idx = np.asarray([resolve_one_position(position)], dtype=int)
        q = q_bypos[pos_idx[0]]
        selected_targets = target_positions[pos_idx]
        label = f"target position i={int(selected_targets[0])}"
        mode = "single"

    return q, label, {
        "position_mode": mode,
        "position_indices": pos_idx,
        "target_positions_1based": selected_targets,
        "source_array": bypos_key,
        "overlap_definition": definition,
    }

def _select_weight_q_array(
    data,
    weight_level="group",
    weight_selection=None,
):
    """
    Select a weight-overlap array.

    Supported weight_level values
    -----------------------------
    Generic:
        "all"
        "group"
        "parameter"
        "layer" or "global_by_layer"

    Transformer:
        "attention"
        "mlp"
        "norm"

    Mamba:
        "mixer"
        "norm"
    """
    weight_level = str(weight_level).lower()

    # ------------------------------------------------------------------
    # All model parameters together
    # ------------------------------------------------------------------
    if weight_level in ("all", "global", "full", "model"):
        if "weight_overlap_all" not in data.files:
            raise KeyError(
                "'weight_overlap_all' not found. "
                f"Available keys: {data.files}"
            )

        q = np.asarray(
            data["weight_overlap_all"],
            dtype=float,
        )[None, :, :]

        item_names = ["all_weights"]

        return q, "all trainable weights", {
            "weight_level": "all",
            "weight_indices": np.array([0], dtype=int),
            "weight_names": item_names,
        }, item_names

    # ------------------------------------------------------------------
    # Generic groups
    # ------------------------------------------------------------------
    if weight_level in ("group", "groups", "by_group"):
        q_key = "weight_overlap_by_group"
        names_key = "weight_group_names"
        label_prefix = "weight groups"
        kind = "weight group"

    # ------------------------------------------------------------------
    # Individual parameters
    # ------------------------------------------------------------------
    elif weight_level in (
        "parameter",
        "parameters",
        "param",
        "params",
        "by_parameter",
    ):
        q_key = "weight_overlap_by_parameter"
        names_key = "weight_parameter_names"
        label_prefix = "weight parameters"
        kind = "weight parameter"

    # ------------------------------------------------------------------
    # Global overlap separately for every residual layer
    # ------------------------------------------------------------------
    elif weight_level in (
        "layer",
        "layers",
        "global_by_layer",
        "layer_global",
    ):
        q_key = "weight_overlap_by_layer"
        names_key = "weight_layer_names"
        label_prefix = "global layer weights"
        kind = "layer"

    # ------------------------------------------------------------------
    # Transformer attention separately for every layer
    # ------------------------------------------------------------------
    elif weight_level in (
        "attention",
        "attn",
        "transformer_attention",
    ):
        q_key = "transformer_weight_overlap_attention_by_layer"
        names_key = "weight_layer_names"
        label_prefix = "attention weights"
        kind = "attention layer"

    # ------------------------------------------------------------------
    # Transformer MLP separately for every layer
    # ------------------------------------------------------------------
    elif weight_level in (
        "mlp",
        "ffwd",
        "feedforward",
        "feed_forward",
        "transformer_mlp",
    ):
        q_key = "transformer_weight_overlap_mlp_by_layer"
        names_key = "weight_layer_names"
        label_prefix = "MLP weights"
        kind = "MLP layer"

    # ------------------------------------------------------------------
    # Normalization parameters separately for every layer
    # ------------------------------------------------------------------
    elif weight_level in (
        "norm",
        "normalization",
        "layer_norm",
    ):
        if "transformer_weight_overlap_norm_by_layer" in data.files:
            q_key = "transformer_weight_overlap_norm_by_layer"

        elif "mamba_weight_overlap_norm_by_layer" in data.files:
            q_key = "mamba_weight_overlap_norm_by_layer"

        else:
            raise KeyError(
                "No layer-wise normalization overlap array found. "
                f"Available keys: {data.files}"
            )

        names_key = "weight_layer_names"
        label_prefix = "normalization weights"
        kind = "normalization layer"

    # ------------------------------------------------------------------
    # Mamba mixer separately for every layer
    # ------------------------------------------------------------------
    elif weight_level in (
        "mixer",
        "mamba",
        "mamba_mixer",
    ):
        q_key = "mamba_weight_overlap_mixer_by_layer"
        names_key = "weight_layer_names"
        label_prefix = "Mamba mixer weights"
        kind = "Mamba layer"

    else:
        raise ValueError(
            "weight_level must be one of: "
            "'all', 'group', 'parameter', 'layer', "
            "'attention', 'mlp', 'norm', or 'mixer'."
        )

    if q_key not in data.files:
        raise KeyError(
            f"{q_key!r} not found. "
            f"Available keys: {data.files}"
        )

    q_all = np.asarray(
        data[q_key],
        dtype=float,
    )

    if names_key in data.files:
        names = _decode_np_string_array(
            data[names_key]
        )
    else:
        names = [
            f"layer_{i + 1}"
            for i in range(q_all.shape[0])
        ]

    indices = _resolve_weight_indices(
        names,
        selection=weight_selection,
        kind=kind,
    )

    q = q_all[indices]
    selected_names = [
        names[index]
        for index in indices
    ]

    if weight_selection is None:
        label = label_prefix
    else:
        label = (
            label_prefix
            + ": "
            + ", ".join(selected_names)
        )

    return q, label, {
        "weight_level": weight_level,
        "weight_indices": indices,
        "weight_names": selected_names,
        "source_array": q_key,
    }, selected_names

def _select_q_array(
    data,
    split="test",
    position="mean",
    overlap_target="auto",
    weight_level="group",
    weight_selection=None,
):
    """General selector for all NTP overlap definitions.

    Representation-like definitions return q[layer,K,K].
    Weight definitions return q[group-or-parameter,K,K].
    """
    actual = _infer_overlap_target(data)
    requested = str(overlap_target).lower()

    # ``representations`` is intentionally an umbrella here so that old calls
    # also work on low_rank/novelty/task output directories.
    if requested not in {"auto", "representations"}:
        aliases = {"lowrank": "low_rank", "block_update": "block_update_cka"}
        requested = aliases.get(requested, requested)
        if requested != actual:
            # cka/block_update share one historical filename, so trust the file
            # metadata rather than making the user restate representation_mode.
            same_old_family = {requested, actual} <= {"cka", "block_update_cka"}
            if not same_old_family:
                raise ValueError(
                    f"Requested overlap_target={requested!r}, but the file contains "
                    f"{actual!r}."
                )

    if actual == "weights":
        q, label, info, names = _select_weight_q_array(
            data,
            weight_level=weight_level,
            weight_selection=weight_selection,
        )
        info["overlap_target"] = "weights"
        info["overlap_definition"] = "weights"
        return q, label, info, names

    q, label, info = _select_representation_q_array(
        data,
        split=split,
        position=position,
        overlap_definition=actual,
    )

    if "layer_names" in data.files:
        names = _decode_np_string_array(data["layer_names"])
    else:
        names = [f"layer_{i}" for i in range(q.shape[0])]

    info["overlap_target"] = "representations"
    info["overlap_definition"] = actual
    return q, label, info, names

def plot_ntp_representation_self_overlap_vs_time(
    results_dir,
    output_name=None,
    split="test",
    position="mean",
    reference_times=None,
    reference_indices=None,
    layers=None,
    time_key="steps",
    x_mode="absolute",
    only_t_ge_tprime=True,
    xscale="log",
    yscale="linear",
    ylim=(0, 1.05),
    figsize_per_panel=(6.2, 4.3),
    ncols=3,
    marker=None,
    linewidth=3.0,
    alpha=0.95,
    title_prefix=None,
    legend_fontsize=8,
    grid=True,
    overlap_target="auto",
    weight_level="group",
    weight_selection=None,
):
    """Plot q(t,t') at fixed reference times for any NTP overlap definition.

    Supported outputs
    -----------------
    ``cka`` / ``block_update_cka``
        representation_overlaps_per_position.npz
    ``weights``
        weight_overlaps.npz
    ``low_rank``
        low_rank_overlaps.npz
    ``novelty``
        novelty_overlaps.npz
    ``task``
        task_overlaps.npz

    ``overlap_target='auto'`` is recommended.  For backward compatibility,
    ``overlap_target='representations'`` also accepts low-rank, novelty, and
    task files when the supplied output directory contains only one compatible
    representation-like result.
    """
    npz_path = _find_ntp_overlap_npz(
        results_dir,
        output_name=output_name,
        overlap_target=overlap_target,
    )

    data = np.load(npz_path, allow_pickle=True)
    actual_definition = _infer_overlap_target(data, npz_path=npz_path)

    q, object_label, object_info, object_names = _select_q_array(
        data,
        split=split,
        position=position,
        overlap_target=overlap_target,
        weight_level=weight_level,
        weight_selection=weight_selection,
    )

    times, actual_time_key = _choose_time_axis(data, time_key=time_key)

    if q.ndim != 3:
        raise ValueError(f"Expected q with shape [items,K,K], got {q.shape}")
    if q.shape[1] != len(times) or q.shape[2] != len(times):
        raise ValueError(
            f"Mismatch between q shape {q.shape} and time axis length {len(times)}."
        )

    is_weights = actual_definition == "weights"
    item_idx = _resolve_layer_indices(object_names, layers=layers)

    if is_weights:
        item_axis_label = "weight"
        ylabel = r"$q_W(t,t')$"
        ylim_eff = (-1.05, 1.05) if ylim is None else ylim
        definition_label = "weights"
    else:
        item_axis_label = "layer"
        ylim_eff = (0.0, 1.05) if ylim is None else ylim
        labels = {
            "cka": "representation CKA",
            "block_update_cka": "block-update CKA",
            "low_rank": "low-rank subspace overlap",
            "novelty": "novelty CKA",
            "task": "task-relevant CKA",
        }
        definition_label = labels.get(actual_definition, actual_definition)

        if actual_definition == "low_rank" and "low_rank_rank" in data.files:
            rank = int(np.asarray(data["low_rank_rank"]).reshape(()))
            ylabel = rf"$q_{{\ell,r}}(t,t')$  ($r={rank}$)"
            definition_label += f" (r={rank})"
        elif actual_definition == "novelty":
            ylabel = r"$q^{\rm nov}_\ell(t,t')$"
        elif actual_definition == "task":
            ylabel = r"$q^{\rm task}_\ell(t,t')$"
        else:
            ylabel = r"$q_\ell(t,t')$"

    ref_idx = _resolve_reference_indices(
        times,
        reference_times=reference_times,
        reference_indices=reference_indices,
    )

    nrefs = len(ref_idx)
    ncols_eff = min(int(ncols), nrefs)
    nrows = math.ceil(nrefs / ncols_eff)

    fig, axs = plt.subplots(
        nrows,
        ncols_eff,
        figsize=(figsize_per_panel[0] * ncols_eff, figsize_per_panel[1] * nrows),
        squeeze=False,
    )
    axs_flat = axs.ravel()

    for ax, j_ref in zip(axs_flat, ref_idx):
        t_ref = times[j_ref]

        if only_t_ge_tprime:
            mask = times >= t_ref
        else:
            mask = np.ones_like(times, dtype=bool)

        x = times[mask]

        if x_mode == "delta":
            x = x - t_ref
            xlabel = rf"$t - t'$ ({actual_time_key})"
        elif x_mode == "absolute":
            xlabel = rf"$t$ ({actual_time_key})"
        else:
            raise ValueError("x_mode must be 'absolute' or 'delta'.")

        if xscale == "log":
            finite_positive = np.isfinite(x) & (x > 0)
            mask_indices = np.where(mask)[0]
            keep_indices = mask_indices[finite_positive]
            x_plot = x[finite_positive]
        else:
            keep_indices = np.where(mask)[0]
            x_plot = x

        for item in item_idx:
            y = q[item, keep_indices, j_ref]
            ax.plot(
                x_plot,
                y,
                marker=marker,
                linewidth=linewidth,
                alpha=alpha,
                label=object_names[item],
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if ylim_eff is not None:
            ax.set_ylim(*ylim_eff)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        if grid:
            ax.grid(True, alpha=0.3)

        if title_prefix is None:
            if is_weights:
                first_line = f"weights, {object_label}"
            else:
                first_line = f"{definition_label}: {split}, {object_label}"
        else:
            if is_weights:
                first_line = f"{title_prefix}: weights, {object_label}"
            else:
                first_line = (
                    f"{title_prefix}: {definition_label}, {split}, {object_label}"
                )

        ax.set_title(
            first_line + "\n" + rf"reference $t'={t_ref:g}$ {actual_time_key}"
        )
        ax.legend(fontsize=legend_fontsize, frameon=False)

    for ax in axs_flat[nrefs:]:
        ax.axis("off")

    fig.tight_layout()

    info = {
        "npz_path": str(npz_path),
        "overlap_target": "weights" if is_weights else "representations",
        "overlap_definition": actual_definition,
        "split": None if is_weights else split,
        "position": None if is_weights else position,
        "object_label": object_label,
        "object_info": object_info,
        "time_key_used": actual_time_key,
        "times": times,
        "reference_indices": ref_idx,
        "reference_times_available": times[ref_idx],
        "item_axis_label": item_axis_label,
        "item_indices": item_idx,
        "item_names": [object_names[i] for i in item_idx],
        "q_shape": q.shape,
    }

    if not is_weights:
        info["layer_indices"] = item_idx
        info["layer_names"] = [object_names[i] for i in item_idx]
        info["position_label"] = object_label
        info["position_info"] = object_info
        if actual_definition == "low_rank" and "low_rank_rank" in data.files:
            info["low_rank_rank"] = int(np.asarray(data["low_rank_rank"]).reshape(()))
        if actual_definition == "novelty" and "novelty_ridge_alpha" in data.files:
            info["novelty_ridge_alpha"] = float(
                np.asarray(data["novelty_ridge_alpha"]).reshape(())
            )
        if actual_definition == "task" and "task_ridge_alpha" in data.files:
            info["task_ridge_alpha"] = float(
                np.asarray(data["task_ridge_alpha"]).reshape(())
            )
    else:
        info["weight_indices"] = item_idx
        info["weight_names"] = [object_names[i] for i in item_idx]

    data.close()
    return fig, axs, info

def plot_ntp_weight_self_overlap_vs_time(
    results_dir,
    output_name=None,
    weight_level="group",
    weight_selection=None,
    reference_times=None,
    reference_indices=None,
    layers=None,
    time_key="steps",
    x_mode="absolute",
    only_t_ge_tprime=True,
    xscale="log",
    yscale="linear",
    ylim=(-1.05, 1.05),
    figsize_per_panel=(6.2, 4.3),
    ncols=3,
    marker=None,
    linewidth=2.0,
    alpha=0.95,
    title_prefix=None,
    legend_fontsize=8,
    grid=True,
):
    """
    Convenience wrapper for weight-overlap files.

    Examples
    --------
    Global all-weight overlap:
        plot_ntp_weight_self_overlap_vs_time(
            results_dir="results",
            output_name="L3_weights_v1",
            weight_level="all",
        )

    Grouped weight overlap:
        plot_ntp_weight_self_overlap_vs_time(
            results_dir="results",
            output_name="L3_weights_v1",
            weight_level="group",
        )

    Selected groups:
        plot_ntp_weight_self_overlap_vs_time(
            results_dir="results",
            output_name="L3_weights_v1",
            weight_level="group",
            weight_selection=["block_1", "block_2", "block_3"],
        )

    Per-parameter overlap:
        plot_ntp_weight_self_overlap_vs_time(
            results_dir="results",
            output_name="L3_weights_v1",
            weight_level="parameter",
            weight_selection=["blocks.0.sa.key", "blocks.0.sa.value"],
        )
    """
    return plot_ntp_representation_self_overlap_vs_time(
        results_dir=results_dir,
        output_name=output_name,
        split="test",
        position="mean",
        reference_times=reference_times,
        reference_indices=reference_indices,
        layers=layers,
        time_key=time_key,
        x_mode=x_mode,
        only_t_ge_tprime=only_t_ge_tprime,
        xscale=xscale,
        yscale=yscale,
        ylim=ylim,
        figsize_per_panel=figsize_per_panel,
        ncols=ncols,
        marker=marker,
        linewidth=linewidth,
        alpha=alpha,
        title_prefix=title_prefix,
        legend_fontsize=legend_fontsize,
        grid=grid,
        overlap_target="weights",
        weight_level=weight_level,
        weight_selection=weight_selection,
    )

def plot_ntp_novelty_dynamics(results_dir, *, layers=('block_1', 'block_4', 'block_8'),
                              position='mean', save_path=None):
    """One figure for the affine predictor h_l ~= (h_{l-1}-mx) A + my.

    Uses the last selected checkpoint as reference. Raw-coordinate comparisons,
    not invariants under changes of representation basis. Errors include mean
    offsets; their denominators are centered target energies. Position pooling
    uses ratios of summed energies, not averages of unstable per-position ratios.
    Requires a novelty run computed with the predictor-dynamics extension.
    """
    path = Path(results_dir).expanduser()
    if path.suffix != '.npz':
        path = path / 'novelty_overlaps.npz'
    with np.load(path, allow_pickle=True) as f:
        if 'novelty_prediction_change' not in f.files:
            raise ValueError('Rerun OVERLAP_DEFINITION=novelty with the updated computation; '
                             'old overlaps do not contain predictor dynamics.')
        names = f['layer_names'].tolist()
        keys = ('prediction_change', 'fixed_input_change', 'prediction_cka',
                'reference_target_energy', 'target_energy', 'residual_mse', 'map_relative_change')
        data = {k: f['novelty_' + k] for k in keys}
        t = f['selected_steps']
        reference = int(f['novelty_dynamics_reference_index'])
    positions = np.arange(2, data['prediction_change'].shape[-1] + 2)
    pi = np.arange(len(positions)) if position == 'mean' else np.flatnonzero(positions == int(position))
    if not len(pi):
        raise ValueError(f'Available target positions: {positions.tolist()}')
    def ratio(num, den):
        return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 1e-20)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for color, layer in zip(plt.get_cmap('tab10')(np.arange(len(layers))), layers):
        name = layer.replace('block_', 'block_novelty_') if layer.startswith('block_') and not layer.startswith('block_novelty_') else layer
        if name not in names:
            raise ValueError(f'Unknown layer {layer}; available: {names}')
        ki = names.index(name)
        pooled = {k: data[k][:, ki, pi].mean(-1) for k in keys if k != 'map_relative_change'}
        axes[0,0].plot(t, data['map_relative_change'][:,ki], color=color, label=layer)
        axes[0,1].plot(t, ratio(pooled['prediction_change'], pooled['reference_target_energy']), color=color)
        axes[0,1].plot(t, ratio(pooled['fixed_input_change'], pooled['reference_target_energy']), color=color, ls='--')
        axes[1,0].plot(t, pooled['prediction_cka'], color=color)
        axes[1,1].plot(t, ratio(pooled['residual_mse'], pooled['target_energy']), color=color)
    axes[0,0].set_title('Map drift (all training positions pooled)')
    axes[0,0].set_ylabel(r'$\|A_t-A_*\|_F^2/\|A_*\|_F^2$')
    axes[0,0].legend()
    axes[0,1].set_title('Prediction drift: own inputs (solid), fixed inputs (dashed)')
    axes[0,1].set_ylabel('Squared change / reference target variance')
    axes[1,0].set_title('Prediction geometry on matched sequences')
    axes[1,0].set_ylabel('Prediction CKA with reference')
    axes[1,0].set_ylim(-.02, 1.02)
    axes[1,1].set_title('Held-out reconstruction error')
    axes[1,1].set_ylabel('Residual MSE / current target variance')
    axes[1,1].axhline(1, color='gray', ls=':', lw=1)
    for ax in axes.flat:
        ax.set_xscale('log' if np.all(t > 0) else 'symlog')
        ax.set_xlabel('Training step')
        ax.grid(alpha=.22)
    fig.suptitle(f'Novelty predictor dynamics | reference step {t[reference]} | position={position}')
    if save_path is not None:
        out = Path(save_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=160)
    return fig


def plot_ntp_stage1(results_dir, *, layers=('block_1', 'block_4', 'block_8'),
                    alpha=1e-4, position='mean', save_dir=None):
    """Three assessment figures. No test-set alpha selection or time interpolation.

    Returns a dict of figures. ``position`` is 'mean' or a physical target-token
    position (2..sequence_length). Bands show ridge sensitivity, NOT uncertainty.
    """
    import json
    import warnings
    path = Path(results_dir).expanduser()
    if path.suffix != '.npz':
        path = path / 'stage1_diagnostics.npz'
    with np.load(path, allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    names = data['layer_names'].tolist()
    missing = [x for x in layers if x not in names]
    if missing:
        raise ValueError(f'Unknown layers {missing}; available: {names}')
    indices = [names.index(x) for x in layers]
    positions = data['target_token_positions_1based']
    if position == 'mean':
        pi = np.arange(len(positions))
    else:
        pi = np.flatnonzero(positions == int(position))
        if not len(pi):
            raise ValueError(f'Available target positions: {positions.tolist()}')
    def mean(a, axis):
        a = np.asarray(a, dtype=float)
        count = np.isfinite(a).sum(axis)
        return np.divide(np.nansum(a, axis), count,
                         out=np.full(np.shape(count), np.nan, dtype=float), where=count>0)
    def curve(key):
        return mean(data[key][:, :, pi], 2)
    def ratio(num, den):
        return np.divide(num, den, out=np.full_like(num, np.nan), where=den>1e-20)
    t = data['selected_steps']
    ai = int(np.argmin(abs(np.log(data['alphas'])-np.log(alpha))))
    used_alpha = data['alphas'][ai]
    reference = int(data['reference_step'])
    colors = plt.get_cmap('tab10')(np.arange(len(indices)))
    def style(ax, ylabel):
        ax.set_xscale('log' if np.all(t>0) else 'symlog')
        ax.set_xlabel('Training step')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.22)
    baseline = float(mean(data['probe_position_baseline_error'][pi], 0))
    if baseline <= 1e-20:
        raise ValueError('Position baseline has zero error; use raw arrays instead of normalized plots.')
    figures = {}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for ki, color, label in zip(indices, colors, layers):
        axes[0,0].plot(t, curve('state_cka')[:,ki], color=color, ls='--', lw=1.4)
        probe_q = curve('probe_cka')[:,ki]
        axes[0,0].plot(t, probe_q[:,ai], color=color, label=label)
        axes[0,0].fill_between(t, probe_q.min(-1), probe_q.max(-1), color=color, alpha=.15)
        errors = curve('probe_error')[:,ki] / baseline
        axes[0,1].plot(t, errors[:,ai], color=color)
        axes[0,1].fill_between(t, errors.min(-1), errors.max(-1), color=color, alpha=.15)
        displacements = curve('probe_displacement')[:,ki] / baseline
        axes[1,0].plot(t, displacements[:,ai], color=color)
        axes[1,0].fill_between(t, displacements.min(-1), displacements.max(-1), color=color, alpha=.15)
        amplitude = np.sqrt(np.maximum(curve('probe_energy')[:,ki] / baseline, 0))
        axes[1,1].plot(t, amplitude[:,ai], color=color)
        axes[1,1].fill_between(t, amplitude.min(-1), amplitude.max(-1), color=color, alpha=.15)
    style(axes[0,0], 'Overlap with reference'); axes[0,0].set_ylim(-.02,1.02)
    axes[0,0].set_title('Solid: probe CKA; dashed: state CKA'); axes[0,0].legend()
    style(axes[0,1], 'Probe error / position baseline error')
    axes[0,1].axhline(1, color='gray', ls=':'); axes[0,1].set_title('Below 1: better than token-frequency baseline')
    style(axes[1,0], 'Prediction change / position baseline error')
    axes[1,0].set_title('Direct squared difference from reference')
    style(axes[1,1], 'Centered probe RMS / baseline RMS')
    axes[1,1].set_title('Prediction amplitude hidden by CKA normalization')
    fig.suptitle(f'Probe assessment | reference step {reference} | alpha={used_alpha:g}\n'
                 'Shading: fixed ridge values, not a confidence interval')
    figures['01_probes'] = fig

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for ki, color, label in zip(indices, colors, layers):
        axes[0,0].plot(t, curve('novelty_cka')[:,ki], color=color, label=label)
        eta = ratio(curve('novelty_energy')[:,ki], curve('state_energy')[:,ki])
        axes[0,1].plot(t, eta, color=color)
    style(axes[0,0], 'Novelty CKA with reference'); axes[0,0].set_ylim(-.02,1.02)
    axes[0,0].legend(); axes[0,0].set_title('How much does residual geometry change?')
    style(axes[0,1], 'Centered residual energy / state energy')
    axes[0,1].set_title('How large is that residual?')
    axes[0,1].set_yscale('symlog', linthresh=1e-6)
    rank_indices = [names.index('embedding')] + indices
    rank_colors = ['black'] + list(colors)
    r = int(max(data['ranks']))
    for ki, color in zip(rank_indices, rank_colors):
        support = (data['numerical_rank'][:,ki,pi] >= r).mean(-1)
        axes[1,0].plot(t, support, color=color, label=names[ki])
    style(axes[1,0], f'Fraction of positions supporting rank {r}')
    axes[1,0].set_ylim(-.03,1.03); axes[1,0].legend(fontsize=8)
    axes[1,0].set_title('Unsupported low-rank overlaps are saved as NaN')
    emb = names.index('embedding')
    for ti in sorted(set((0, len(t)//2, len(t)-1))):
        s = data['singular_values'][ti,emb,pi]
        relative = ratio(s, s[:, :1])
        axes[1,1].semilogy(np.arange(1,s.shape[-1]+1), np.maximum(mean(relative, 0), 1e-8), label=f't={t[ti]}')
    axes[1,1].axhline(float(data['rank_rtol']), color='gray', ls=':', label='Rank threshold')
    for rank in data['ranks']:
        axes[1,1].axvline(rank, color='gray', alpha=.3)
    axes[1,1].set_xlabel('Singular-value index'); axes[1,1].set_ylabel('Embedding singular value / largest')
    axes[1,1].set_title('Embedding spectrum (display floor: 1e-8)')
    axes[1,1].set_ylim(5e-9, 1.5)
    axes[1,1].legend(fontsize=8); axes[1,1].grid(alpha=.22)
    fig.suptitle('Novelty amplitude and low-rank validity')
    figures['02_structure'] = fig

    if 'within_loss' in data:
        fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
        total = mean(data['ntp_loss'][:,pi], 1)
        within = mean(data['within_loss'][:,pi], 1)
        contribution = mean(data['peeled_contribution'][:,pi], 1)
        ax.plot(t,total,color='black',lw=2.2,label='Total NTP')
        ax.plot(t,within,color='gray',ls='--',lw=2,label='Within compatible set')
        for ell in range(contribution.shape[-1]):
            ax.plot(t,contribution[:,ell],label=f'RHM level {ell+1}')
        style(ax,'Loss contribution [nats / token]')
        ax.set_yscale('symlog',linthresh=1e-6)
        ax.set_title('Matched-data decomposition: total = within + four levels')
        ax.legend(fontsize=9)
        figures['03_loss'] = fig
    else:
        warnings.warn('RHM mask computation was disabled; loss-decomposition figure is unavailable.')
    if save_dir is not None:
        out = Path(save_dir); out.mkdir(parents=True, exist_ok=True)
        for name, fig in figures.items():
            fig.savefig(out / f'{name}.png', dpi=160)
    return figures


def load_rhm_latent_probes(results_dir):
    """Read atomic per-checkpoint files; absent jobs remain NaN on the time grid."""
    files = sorted(Path(results_dir).expanduser().glob('latent_step_*.npz'))
    if not files:
        return {'completed_steps': np.array([], dtype=int), 'files': []}
    rows = []
    for path in files:
        with np.load(path, allow_pickle=False) as f:
            rows.append({key: f[key] for key in f.files})
    first = rows[0]
    if any(str(r['signature']) != str(first['signature']) for r in rows):
        raise ValueError('Mixed configurations in result folder.')
    rows.sort(key=lambda r: int(r['step']))
    expected = first['expected_steps'].astype(int)
    if len(np.unique(expected)) != len(expected):
        raise ValueError('Duplicate expected checkpoint steps')
    order = np.argsort(expected)
    times = expected[order]
    out = {key: first[key] for key in ('layer_names','levels','positions','mode',
                                     'probe_levels','score_names','fit_diagnostic_names')}
    out.update(steps=times, completed_steps=np.array([int(r['step']) for r in rows]),
               files=[str(p) for p in files])
    for key in ('train_scores','valid_scores','fit_diagnostics'):
        out[key] = np.full((len(times),)+first[key].shape, np.nan)
        for r in rows:
            out[key][np.flatnonzero(times == int(r['step']))[0]] = r[key]
    out['missing_steps'] = np.setdiff1d(times, out['completed_steps'])
    return out


def plot_rhm_latent_probes(results_dir, *, metric='kl', split='test',
                           levels=None, layers=None, position='mean',
                           xscale='log', yscale='linear', figsize=None):
    """One panel per RHM level, turbo curves for transformer layers.

    metric: kl | error | excess_error. position: mean or one-based token index.
    Test is the existing held-out valid split. Missing jobs make gaps in curves.
    Complete error is posterior-expected hard error, not MAP-label disagreement.
    """
    data = load_rhm_latent_probes(results_dir)
    if not len(data['completed_steps']):
        print('No completed latent checkpoints yet.')
        return None, None, data
    if metric not in ('kl','error','excess_error'):
        raise ValueError('metric must be kl, error or excess_error')
    split = 'valid' if split == 'test' else split
    if split not in ('train','valid'):
        raise ValueError('split must be train, test or valid')
    ks = np.unique(data['levels']) if levels is None else np.atleast_1d(levels)
    if not len(ks) or not set(ks).issubset(set(data['levels'])):
        raise ValueError('Requested RHM level not available')
    names = data['layer_names'].tolist()
    selected = names if layers is None else ([layers] if isinstance(layers, str) else list(layers))
    li = [names.index(name) for name in selected]
    cols = min(2, len(ks)); nr = int(np.ceil(len(ks)/cols))
    fig, axes = plt.subplots(nr, cols, squeeze=False, figsize=figsize or (6*cols, 4*nr))
    colors = plt.get_cmap('turbo')(np.linspace(.05,.9,len(names)))
    score = data[split+'_scores']
    y = score[..., 0 if metric == 'kl' else 1].copy()
    if metric == 'excess_error': y -= score[..., 2]
    mask = data['steps'] > 0 if xscale == 'log' else np.ones(len(data['steps']), dtype=bool)
    for ax, k in zip(axes.flat, ks):
        ix = data['levels'] == k
        if position != 'mean': ix &= data['positions'] == int(position)
        if not ix.any():
            raise ValueError(f'No targets at k={k}, position={position}')
        for ell in li:
            # Equal-weight position average; each position has the same sequence count.
            ax.plot(data['steps'][mask], y[:,ell,ix].mean(-1)[mask],
                    color=colors[ell], label=names[ell], marker='.', lw=2)
        if metric == 'error':
            ax.plot(data['steps'][mask], score[:,0,ix,2].mean(-1)[mask],
                    'k--', lw=1, label='BP Bayes error')
        ax.set(xscale=xscale, yscale=yscale, xlabel='training steps',
               ylabel=metric.replace('_',' '), title=f'RHM level k={k} ({str(data["mode"])})')
        ax.grid(alpha=.25)
    for ax in list(axes.flat)[len(ks):]: ax.set_visible(False)
    axes.flat[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f'{len(data["completed_steps"])}/{len(data["steps"])} checkpoints complete')
    fig.tight_layout()
    return fig, axes, data
