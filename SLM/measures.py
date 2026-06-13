"""Evaluation utilities for text and RHM next-token training."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F


def _empty_detailed() -> Dict[str, object]:
    return {
        "loss": float("nan"),
        "acc": float("nan"),
        "err": float("nan"),
        "num_samples": 0,
        "num_tokens": 0,
        "loss_by_position": np.array([], dtype=np.float64),
        "acc_by_position": np.array([], dtype=np.float64),
        "err_by_position": np.array([], dtype=np.float64),
        "count_by_position": np.array([], dtype=np.float64),
        "logit_effdim_entropy_by_position": np.array([], dtype=np.float64),
        "logit_effdim_pr_by_position": np.array([], dtype=np.float64),
        "logit_effdim_entropy_norm_by_position": np.array([], dtype=np.float64),
        "logit_effdim_pr_norm_by_position": np.array([], dtype=np.float64),
        "logit_cov_eigvals_by_position": np.empty((0, 0), dtype=np.float64),
    }


def _effective_dimensions(cov: torch.Tensor, eps: float = 1e-12) -> Dict[str, np.ndarray]:
    cov = 0.5 * (cov + cov.transpose(-1, -2))
    eigvals = torch.linalg.eigvalsh(cov).to(torch.float64).clamp_min(0.0)
    total = eigvals.sum(dim=-1)
    valid = torch.isfinite(total) & (total > eps)

    weights = torch.zeros_like(eigvals)
    weights[valid] = eigvals[valid] / total[valid, None]
    entropy = -torch.where(
        weights > eps,
        weights * torch.log(torch.clamp(weights, min=eps)),
        torch.zeros_like(weights),
    ).sum(dim=-1)
    entropy_dim = torch.where(valid, torch.exp(entropy), torch.zeros_like(entropy))
    pr_dim = torch.where(
        valid,
        1.0 / torch.clamp((weights * weights).sum(dim=-1), min=eps),
        torch.zeros_like(total),
    )
    # Subtract one because subtracting the vocabulary mean removes the all-ones direction.
    max_dim = max(int(eigvals.shape[-1]) - 1, 1)
    return {
        "logit_effdim_entropy_by_position": entropy_dim.cpu().numpy(),
        "logit_effdim_pr_by_position": pr_dim.cpu().numpy(),
        "logit_effdim_entropy_norm_by_position": (entropy_dim / max_dim).cpu().numpy(),
        "logit_effdim_pr_norm_by_position": (pr_dim / max_dim).cpu().numpy(),
        "logit_cov_eigvals_by_position": eigvals.cpu().numpy(),
    }


@torch.no_grad()
def evaluate_detailed(
    model,
    dataset,
    device,
    *,
    iters: Optional[int] = None,
    max_samples: Optional[int] = None,
    compute_logit_effdim: bool = False,
) -> Dict[str, object]:
    """Evaluate mean and position-resolved next-token loss and accuracy.

    When ``compute_logit_effdim`` is enabled, the covariance of gauge-fixed
    logits is accumulated independently at every target position and converted
    to entropy and participation-ratio effective dimensions.
    """
    if dataset is None:
        return _empty_detailed()

    dataset.reset()
    model.eval()
    n_batches = int(dataset.num_batches if iters is None else min(iters, dataset.num_batches))

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    total_samples = 0
    loss_by_pos = correct_by_pos = count_by_pos = None
    sum_z = sum_zz = None

    for _ in range(n_batches):
        inputs, targets = dataset.next_batch()
        if max_samples is not None and max_samples > 0:
            remaining = int(max_samples) - total_samples
            if remaining <= 0:
                break
            inputs = inputs[:remaining]
            targets = targets[:remaining]
        inputs = inputs.to(device, non_blocking=True).long()
        targets = targets.to(device, non_blocking=True).long()

        logits = model(inputs)
        B, T = targets.shape
        V = logits.shape[-1]
        token_losses = F.cross_entropy(
            logits.reshape(-1, V),
            targets.reshape(-1),
            reduction="none",
        ).view(B, T)
        correct = logits.argmax(dim=-1).eq(targets)

        if loss_by_pos is None:
            loss_by_pos = torch.zeros(T, dtype=torch.float64, device=device)
            correct_by_pos = torch.zeros(T, dtype=torch.float64, device=device)
            count_by_pos = torch.zeros(T, dtype=torch.float64, device=device)
            if compute_logit_effdim:
                sum_z = torch.zeros((T, V), dtype=torch.float64, device=device)
                sum_zz = torch.zeros((T, V, V), dtype=torch.float64, device=device)

        loss_by_pos += token_losses.to(torch.float64).sum(dim=0)
        correct_by_pos += correct.to(torch.float64).sum(dim=0)
        count_by_pos += float(B)
        total_loss += float(token_losses.sum().item())
        total_correct += int(correct.sum().item())
        total_tokens += int(targets.numel())
        total_samples += int(B)

        if compute_logit_effdim:
            z = logits.detach().to(torch.float64)
            z = z - z.mean(dim=-1, keepdim=True)
            sum_z += z.sum(dim=0)
            sum_zz += torch.einsum("btv,btw->tvw", z, z)

        if max_samples is not None and max_samples > 0 and total_samples >= max_samples:
            break

    if total_tokens == 0 or loss_by_pos is None:
        return _empty_detailed()

    denom = torch.clamp(count_by_pos, min=1.0)
    loss_pos = (loss_by_pos / denom).cpu().numpy()
    acc_pos = (correct_by_pos / denom).cpu().numpy()
    result = {
        "loss": total_loss / total_tokens,
        "acc": total_correct / total_tokens,
        "err": 1.0 - total_correct / total_tokens,
        "num_samples": total_samples,
        "num_tokens": total_tokens,
        "loss_by_position": loss_pos,
        "acc_by_position": acc_pos,
        "err_by_position": 1.0 - acc_pos,
        "count_by_position": count_by_pos.cpu().numpy(),
        "logit_effdim_entropy_by_position": np.array([], dtype=np.float64),
        "logit_effdim_pr_by_position": np.array([], dtype=np.float64),
        "logit_effdim_entropy_norm_by_position": np.array([], dtype=np.float64),
        "logit_effdim_pr_norm_by_position": np.array([], dtype=np.float64),
        "logit_cov_eigvals_by_position": np.empty((0, 0), dtype=np.float64),
    }
    if compute_logit_effdim:
        count = torch.clamp(count_by_pos, min=1.0)
        mean_z = sum_z / count[:, None]
        second = sum_zz / count[:, None, None]
        cov = second - mean_z[:, :, None] * mean_z[:, None, :]
        result.update(_effective_dimensions(cov))
    return result


@torch.no_grad()
def evaluate(model, criterion, dataset, iters, device):
    """Backward-compatible scalar evaluator used by older scripts."""
    del criterion
    return evaluate_detailed(model, dataset, device, iters=iters)["loss"]


@torch.no_grad()
def loss_by_token(model, criterion, dataset, block_size, iters, device):
    """Backward-compatible n-gram/position loss vector."""
    del criterion, block_size
    return torch.as_tensor(
        evaluate_detailed(model, dataset, device, iters=iters)["loss_by_position"],
        dtype=torch.float32,
    )
