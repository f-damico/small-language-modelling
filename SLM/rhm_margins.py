"""
Level-wise RHM compatibility masks and M_{l,i} diagnostics for next-token prediction.

Put this file in the same folder as main.py/train.py.  It is intentionally
standalone: it only needs the fixed RHM rules and the token sequences returned by
RandomHierarchyModel.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RHMParamsLite:
    num_features: int
    num_classes: int
    num_synonyms: int
    tuple_size: int
    num_layers: int

    @property
    def sequence_length(self) -> int:
        return self.tuple_size ** self.num_layers


class CompatibilityComputer:
    """Exact compatible candidate-set computer for one fixed RHM rule instance."""

    def __init__(self, params: RHMParamsLite, rules: Sequence):
        self.params = params
        if isinstance(rules, dict):
            self.rules = [self._to_numpy_rule(rules[k]) for k in sorted(rules.keys())]
        else:
            self.rules = [self._to_numpy_rule(r) for r in rules]
        self.v = params.num_features
        self.n = params.num_classes
        self.s = params.tuple_size
        self.L = params.num_layers
        self._token_universe = frozenset(range(self.v))

    @staticmethod
    def _to_numpy_rule(rule):
        if isinstance(rule, torch.Tensor):
            return rule.detach().cpu().numpy().astype(np.int64)
        return np.asarray(rule, dtype=np.int64)

    def _block_start_0based(self, i0: int, level: int) -> int:
        block = self.s ** level
        return (i0 // block) * block

    def _build_pattern(self, sequence: np.ndarray, i0: int, level: int) -> Tuple[int, ...]:
        """
        Pattern over the level-l block containing i0.
        Leaves strictly left of the target inside that block are observed.
        The target and all future leaves are unknown (-1).
        """
        start = self._block_start_0based(i0, level)
        block = self.s ** level
        pattern = np.full(block, -1, dtype=np.int64)
        target_local = i0 - start
        for local in range(target_local):
            pattern[local] = int(sequence[start + local])
        return tuple(int(x) for x in pattern)

    @lru_cache(maxsize=200000)
    def _any_completion_under_symbol(self, depth: int, rule_idx: int, pattern: Tuple[int, ...], root_symbol: int) -> bool:
        if depth == 0:
            obs = pattern[0]
            return obs == -1 or obs == root_symbol

        child_size = self.s ** (depth - 1)
        rules_here = self.rules[rule_idx][root_symbol]
        for children in rules_here:
            ok = True
            for child_idx in range(self.s):
                start = child_idx * child_size
                child_pattern = pattern[start:start + child_size]
                child_symbol = int(children[child_idx])
                if not self._any_completion_under_symbol(depth - 1, rule_idx + 1, child_pattern, child_symbol):
                    ok = False
                    break
            if ok:
                return True
        return False

    @lru_cache(maxsize=200000)
    def _target_tokens_under_symbol(
        self,
        depth: int,
        rule_idx: int,
        pattern: Tuple[int, ...],
        target_local_pos: int,
        root_symbol: int,
    ) -> frozenset[int]:
        if depth == 0:
            obs = pattern[0]
            if obs != -1 and obs != root_symbol:
                return frozenset()
            return frozenset((root_symbol,))

        child_size = self.s ** (depth - 1)
        target_child = target_local_pos // child_size
        target_child_local = target_local_pos % child_size

        out: set[int] = set()
        rules_here = self.rules[rule_idx][root_symbol]
        for children in rules_here:
            consistent = True
            for child_idx in range(self.s):
                if child_idx == target_child:
                    continue
                start = child_idx * child_size
                child_pattern = pattern[start:start + child_size]
                child_symbol = int(children[child_idx])
                if not self._any_completion_under_symbol(depth - 1, rule_idx + 1, child_pattern, child_symbol):
                    consistent = False
                    break
            if not consistent:
                continue

            start = target_child * child_size
            child_pattern = pattern[start:start + child_size]
            child_symbol = int(children[target_child])
            out.update(
                self._target_tokens_under_symbol(
                    depth - 1,
                    rule_idx + 1,
                    child_pattern,
                    target_child_local,
                    child_symbol,
                )
            )
        return frozenset(out)

    def compatible_token_set(self, sequence: np.ndarray, position_1based: int, level: int) -> frozenset[int]:
        """Return A_{i,l}: candidate target tokens compatible up to level l.
        
        sequence: A complete RHM leaf sequence, sequence = [x_1, x_2, ..., x_d]
        position_1based: the position whose possible token values are being computed

        """
        if level == 0:
            return self._token_universe                        # at level zero all tokens compatible   
        if level < 0 or level > self.L:
            raise ValueError(f"level must be in [0, {self.L}], got {level}")

        i0 = position_1based - 1                               # the position. Simply to change i0=i−1
        pattern = self._build_pattern(sequence, i0, level)     # A tuple describing the observed part of the relevant hierarchy block.
                                                               # Is the partial leaf configuration that the candidate target token must be compatible with.
                                                               # It contains:
                                                               # actual token IDs for positions before the target that are considered observed
                                                               # -1 for the target and unobserved positions 
        start = self._block_start_0based(i0, level)            # start is the zero-based index of the first leaf in the "level" block containing the target 
                                                               # start tells where the target’s block begins
        target_local = i0 - start                              # The target position measured relative to the beginning of that block
        rule_idx = self.L - level                              # The index of the first rule array corresponding to this level-specific block

        root_domain = self.n if level == self.L else self.v    # number of possible root symbols that the rest of the code will test
        out: set[int] = set()                                  # start with an empty set of compatible tokens
        for root_symbol in range(root_domain):                 # loops over every allowed root symbol and collects the target tokens that are possible under each one
            out.update(
                self._target_tokens_under_symbol(              # returns a set of candidate token IDs that can occur at the target position under this specific root symbol. Can be empty
                    level,
                    rule_idx,
                    pattern,
                    target_local,
                    root_symbol,
                )
            )
        return frozenset(out)                                  # out is the union of candidate tokens obtained from all possible root symbols

    def masks_for_sequences(self, sequences: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return A and B masks for next-token positions i=2..d.

        Shapes are [num_samples, d-1, L, v].  The level axis contains l=1..L.
        B_{i,l}=A_{i,l-1} minus A_{i,l}.
        """
        seqs = np.asarray(sequences, dtype=np.int64)
        if seqs.ndim != 2:
            raise ValueError(f"Expected sequences with shape [B,d], got {seqs.shape}")

        Bsz, d = seqs.shape
        if d != self.params.sequence_length:
            raise ValueError(f"Expected sequence length {self.params.sequence_length}, got {d}")

        A_masks = np.zeros((Bsz, d - 1, self.L, self.v), dtype=bool)
        B_masks = np.zeros((Bsz, d - 1, self.L, self.v), dtype=bool)

        for b in range(Bsz):                                                           # loop over datapoints mu
            seq = seqs[b]                                                              # full sequence
            for pos0 in range(1, d):                                                   # target positions i=2..d, zero-based pos0=1..d-1
                prev_A = set(range(self.v))                                            # A at level 0 has all tokens
                for ell in range(1, self.L + 1):                                       
                    A = set(self.compatible_token_set(seq, pos0 + 1, ell))             # A_l^{mu,i} computed here
                    Bset = prev_A.difference(A)                                        # the set of just excluded tokens
                    if A:
                        A_masks[b, pos0 - 1, ell - 1, list(A)] = True                  # sets True for all tokens in A_l^{mu,i}
                    if Bset:
                        B_masks[b, pos0 - 1, ell - 1, list(Bset)] = True
                    prev_A = A
        return A_masks, B_masks

    def clear_caches(self) -> None:
        self._any_completion_under_symbol.cache_clear()
        self._target_tokens_under_symbol.cache_clear()


def _strip_long_offset_and_validate(seq: torch.Tensor, args) -> torch.Tensor:
    """Normalize integer RHM tokens to [0, num_features-1] and fail early on bad batches."""
    seq = seq.long()
    v = int(args.num_features)
    if seq.numel() == 0:
        return seq

    mn = int(seq.min().item())
    mx = int(seq.max().item())

    # The original RHM dataset with input_format='long' stores x+1.  The
    # small-language-modeling integration normalises all batches to zero-based
    # ids before they reach this function; the explicit flag prevents an
    # all-nonzero mini-batch from being shifted a second time.
    already_zero_based = bool(getattr(args, "rhm_tokens_are_zero_based", False))
    if (not already_zero_based) and mn >= 1 and mx <= v:
        seq = seq - 1
        mn = int(seq.min().item())
        mx = int(seq.max().item())

    if mn < 0 or mx >= v:
        raise ValueError(
            "Token ids outside the valid range [0, num_features-1]. "
            f"Got min={mn}, max={mx}, num_features={v}, shape={tuple(seq.shape)}. "
            "This usually means an online mini-batch was accidentally collated as "
            "[outer_batch, online_batch, d] and interpreted as one-hot data."
        )
    return seq


def batch_to_token_sequences(batch, args) -> torch.Tensor:
    """
    Return integer token sequences with values in [0, vocab_size-1].

    Handles all layouts used by this repo:
      offline long:        [B, d]
      offline one-hot:     [B, V, d]
      online long item:    [online_B, d]
      online one-hot item: [online_B, V, d]
      accidentally nested batches from DataLoader auto-collation:
                           [outer_B, online_B, d] or [outer_B, online_B, V, d]
    """
    x = batch[0] if isinstance(batch, (tuple, list)) else batch
    v = int(args.num_features)
    d = int(args.num_tokens)

    if x.ndim == 4:
        # Nested online one-hot: [outer_B, online_B, V, d] or [outer_B, online_B, d, V].
        if x.shape[-2] == v and x.shape[-1] == d:
            x = x.reshape(-1, v, d)
            seq = torch.argmax(x, dim=1)
        elif x.shape[-1] == v and x.shape[-2] == d:
            x = x.reshape(-1, d, v)
            seq = torch.argmax(x, dim=-1)
        else:
            raise ValueError(f"Cannot interpret 4D batch with shape {tuple(x.shape)} as RHM tokens.")
        return _strip_long_offset_and_validate(seq, args).to(args.device, non_blocking=True).long()

    if x.ndim == 3:
        # Important: online long batches can arrive as [outer_B, online_B, d].
        # Do NOT argmax over dim=1 unless this is truly [B, V, d].
        if (not torch.is_floating_point(x)) and x.shape[-1] == d:
            seq = x.reshape(-1, d)
        elif x.shape[1] == v and x.shape[2] == d:
            seq = torch.argmax(x, dim=1)
        elif x.shape[1] == d and x.shape[2] == v:
            seq = torch.argmax(x, dim=-1)
        else:
            raise ValueError(f"Cannot interpret 3D batch with shape {tuple(x.shape)} as RHM tokens.")
        return _strip_long_offset_and_validate(seq, args).to(args.device, non_blocking=True).long()

    if x.ndim == 2:
        seq = _strip_long_offset_and_validate(x, args)
        return seq.to(args.device, non_blocking=True).long()

    raise ValueError(f"Cannot interpret batch with shape {tuple(x.shape)} as RHM token sequences.")

def _nan_array(L: int) -> np.ndarray:
    return np.full((L,), np.nan, dtype=np.float64)


@torch.no_grad()
def compute_rhm_m_l_metrics(
    model,
    loader,
    args,
    rules,
    max_samples: int | None = None,
    eps: float = 1e-12,
) -> Dict[str, object]:
    """
    Compute Transformer-logit M_{l,i} diagnostics.

    Returned level arrays have length L and correspond to levels l=1..L.
    Returned by-position arrays have shape [d-1, L] and correspond to target
    positions i=2..d.
    """
    L = int(args.num_layers)
    d = int(args.num_tokens)
    v = int(args.num_features)
    if loader is None:
        return {
            "rhm_M_mean": _nan_array(L),
            "rhm_M_pos_frac": _nan_array(L),
            "rhm_peeled_loss": _nan_array(L),
            "rhm_valid_frac": _nan_array(L),
            "rhm_M_mean_by_position": np.full((d - 1, L), np.nan),
            "rhm_M_pos_frac_by_position": np.full((d - 1, L), np.nan),
            "rhm_peeled_loss_by_position": np.full((d - 1, L), np.nan),
            "rhm_valid_count_by_position": np.zeros((d - 1, L), dtype=np.int64),
            "rhm_num_samples": 0,
        }

    params = RHMParamsLite(
        num_features=args.num_features,
        num_classes=args.num_classes,
        num_synonyms=args.num_synonyms,
        tuple_size=args.tuple_size,
        num_layers=args.num_layers,
    )
    compatibility = CompatibilityComputer(params, rules)

    try:
        model.eval()
        device = args.device

        sum_M = torch.zeros((d - 1, L), dtype=torch.float64, device=device)
        sum_pos = torch.zeros((d - 1, L), dtype=torch.float64, device=device)
        sum_peel = torch.zeros((d - 1, L), dtype=torch.float64, device=device)
        count = torch.zeros((d - 1, L), dtype=torch.float64, device=device)
        total_samples = 0

        for batch in loader:
            seq_cpu = batch_to_token_sequences(batch, args)
            if max_samples is not None and max_samples > 0:
                remaining = max_samples - total_samples
                if remaining <= 0:
                    break
                seq_cpu = seq_cpu[:remaining]
            if seq_cpu.numel() == 0:
                continue

            seq_np = seq_cpu.detach().cpu().numpy().astype(np.int64)
            A_np, B_np = compatibility.masks_for_sequences(seq_np)
            valid_np = B_np.any(axis=-1) & A_np.any(axis=-1)

            seq = seq_cpu.to(device, non_blocking=True)
            logits = model(seq[:, :-1])
            probs = F.softmax(logits, dim=-1).to(torch.float64)

            A = torch.as_tensor(A_np, dtype=torch.bool, device=device)
            Bmask = torch.as_tensor(B_np, dtype=torch.bool, device=device)
            valid = torch.as_tensor(valid_np, dtype=torch.bool, device=device)

            pA = (probs.unsqueeze(2) * A.to(torch.float64)).sum(dim=-1)
            pB = (probs.unsqueeze(2) * Bmask.to(torch.float64)).sum(dim=-1)
            M = torch.log(pA.clamp_min(eps)) - torch.log(pB.clamp_min(eps))
            peeled = F.softplus(-M)

            valid_f = valid.to(torch.float64)
            sum_M += (M * valid_f).sum(dim=0)
            sum_pos += ((M > 0).to(torch.float64) * valid_f).sum(dim=0)
            sum_peel += (peeled * valid_f).sum(dim=0)
            count += valid_f.sum(dim=0)
            total_samples += seq_cpu.size(0)

            if max_samples is not None and max_samples > 0 and total_samples >= max_samples:
                break

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
        valid_frac = (count.sum(dim=0) / max(1, total_samples * (d - 1))).detach().cpu().numpy()

        zero_level = count.sum(dim=0).detach().cpu().numpy() == 0
        M_mean[zero_level] = np.nan
        M_pos[zero_level] = np.nan
        peeled_level[zero_level] = np.nan

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

    finally:
        compatibility.clear_caches()
