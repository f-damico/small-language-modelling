"""Load and visualize latent-probe results across training checkpoints.

Notebook usage:

    %load_ext autoreload
    %autoreload 2

    from plot_probe_latents import load_runs, plot_accuracy, plot_gain, plot_heatmap

    runs = load_runs("./results", V=16, M=4, L=4, EMA=0.99)
    plot_accuracy(runs, source="teacher")
    plot_gain(runs, source="teacher")
    plot_heatmap(runs, source="teacher")
"""

import glob
import os
import re
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch


def _extract_step(path: str):
    m = re.search(r"latent_probe_step(\d+)_", os.path.basename(path))
    return int(m.group(1)) if m else None


def load_runs(
    results_root: str,
    V: int,
    M: int,
    L: int,
    EMA: float,
    pattern_suffix: str = "P8192_nH16_nE1024_mp0.15",
    online: bool = True,
) -> List[Dict]:
    """Load all latent_probe step files for one (V, M, L, EMA) run.

    Returns a list of dicts sorted by step, each holding the saved payload
    plus a "step" key.
    """
    subdir = os.path.join(results_root, f"online_v{V}_m{M}_L{L}")
    suffix = f"L{L}_s2_v{V}_m{M}_{pattern_suffix}_ema{EMA}"
    if online:
        suffix += "_online"
    glob_pattern = os.path.join(subdir, f"latent_probe_step*_{suffix}.pt")
    paths = sorted(glob.glob(glob_pattern), key=lambda p: _extract_step(p) or -1)
    runs = []
    for p in paths:
        step = _extract_step(p)
        if step is None:
            continue
        payload = torch.load(p, map_location="cpu", weights_only=False)
        payload["step"] = step
        runs.append(payload)
    return runs


def _level_means_array(runs, model_key: str, source: str):
    """Returns (steps array, level_means matrix of shape [n_steps, L])."""
    steps = np.array([r["step"] for r in runs])
    lm = [r[model_key][source]["level_means"] for r in runs]
    L = len(lm[0])
    mat = np.array([[d[level] for level in range(L)] for d in lm])
    return steps, mat


def plot_accuracy(
    runs,
    source: str = "teacher",
    ax=None,
    title_suffix: str = "",
    colors=None,
):
    """Accuracy vs step, one line per level. Trained solid, random_init dashed."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    steps, trained = _level_means_array(runs, "trained", source)
    _, randinit = _level_means_array(runs, "random_init", source)
    L = trained.shape[1]
    if colors is None:
        colors = plt.cm.viridis(np.linspace(0, 0.9, L))
    for level in range(L):
        ax.plot(steps, trained[:, level], "-o", color=colors[level],
                label=f"level {level}", markersize=4)
        ax.plot(steps, randinit[:, level], "--", color=colors[level], alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("mean probe accuracy")
    ax.set_title(f"Probe accuracy ({source}){title_suffix}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


def plot_gain(
    runs,
    source: str = "teacher",
    ax=None,
    title_suffix: str = "",
    colors=None,
):
    """Trained minus random_init per level — isolates the learning signal."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    steps, trained = _level_means_array(runs, "trained", source)
    _, randinit = _level_means_array(runs, "random_init", source)
    gain = trained - randinit
    L = gain.shape[1]
    if colors is None:
        colors = plt.cm.viridis(np.linspace(0, 0.9, L))
    for level in range(L):
        ax.plot(steps, gain[:, level], "-o", color=colors[level],
                label=f"level {level}", markersize=4)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("accuracy gain over random init")
    ax.set_title(f"Probe gain ({source}){title_suffix}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


def plot_heatmap(
    runs,
    source: str = "teacher",
    metric: str = "accuracy",
    ax=None,
    title_suffix: str = "",
    cmap="viridis",
):
    """Heatmap with x=step, y=level. metric in {'accuracy', 'gain'}."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3.5))
    steps, trained = _level_means_array(runs, "trained", source)
    if metric == "accuracy":
        data = trained
        vmin, vmax = 0, 1
    elif metric == "gain":
        _, randinit = _level_means_array(runs, "random_init", source)
        data = trained - randinit
        vmax = float(np.max(np.abs(data)))
        vmin = -vmax
        cmap = "RdBu_r"
    else:
        raise ValueError(metric)

    im = ax.pcolormesh(
        steps, np.arange(data.shape[1]), data.T,
        shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax,
    )
    ax.set_xscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("level")
    ax.set_yticks(range(data.shape[1]))
    ax.set_title(f"Probe {metric} ({source}){title_suffix}")
    plt.colorbar(im, ax=ax)
    return ax


def plot_across_m(
    ms: List[int],
    *,
    results_root: str,
    V: int,
    L: int,
    EMA: float,
    source: str = "teacher",
    metric: str = "gain",
    xscale: float = 1.0,
    xlabel: str = "step",
    **load_kwargs,
):
    """One panel per level, overlay lines across m. metric in {'accuracy','gain'}.

    `xscale` lets you rescale step by some m-dependent factor (e.g. pass a dict
    via a custom loop) for collapse attempts.
    """
    per_m = {}
    for m in ms:
        runs = load_runs(results_root, V=V, M=m, L=L, EMA=EMA, **load_kwargs)
        if not runs:
            continue
        steps, trained = _level_means_array(runs, "trained", source)
        _, randinit = _level_means_array(runs, "random_init", source)
        per_m[m] = (steps, trained, randinit)

    any_mat = next(iter(per_m.values()))[1]
    Lv = any_mat.shape[1]
    fig, axes = plt.subplots(1, Lv, figsize=(4.2 * Lv, 4), squeeze=False, sharey=True)
    colors = plt.cm.plasma(np.linspace(0, 0.85, len(per_m)))
    for level in range(Lv):
        ax = axes[0][level]
        for color, (m, (steps, trained, randinit)) in zip(colors, per_m.items()):
            y = trained[:, level] if metric == "accuracy" else trained[:, level] - randinit[:, level]
            ax.plot(steps * xscale, y, "-o", color=color, markersize=3.5, label=f"m={m}")
        if metric == "gain":
            ax.axhline(0, color="k", linewidth=0.5)
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_title(f"level {level}")
        ax.grid(True, alpha=0.3)
    axes[0][0].set_ylabel(f"probe {metric}")
    axes[0][-1].legend(fontsize=8)
    fig.suptitle(f"Across m — {source}")
    fig.tight_layout()
    return fig


def plot_grid(
    ms: List[int],
    *,
    results_root: str,
    V: int,
    L: int,
    EMA: float,
    source: str = "teacher",
    kind: str = "accuracy",
    **load_kwargs,
):
    """Convenience: one panel per m in a row. kind in {'accuracy','gain','heatmap'}."""
    fig, axes = plt.subplots(1, len(ms), figsize=(6 * len(ms), 4.5), squeeze=False)
    for ax, m in zip(axes[0], ms):
        runs = load_runs(results_root, V=V, M=m, L=L, EMA=EMA, **load_kwargs)
        if not runs:
            ax.text(0.5, 0.5, f"no runs for m={m}", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        title = f" — m={m}"
        if kind == "accuracy":
            plot_accuracy(runs, source=source, ax=ax, title_suffix=title)
        elif kind == "gain":
            plot_gain(runs, source=source, ax=ax, title_suffix=title)
        elif kind == "heatmap":
            plot_heatmap(runs, source=source, ax=ax, title_suffix=title)
        else:
            raise ValueError(kind)
    fig.tight_layout()
    return fig
