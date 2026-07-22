"""Probe data2vec masked-token features against exact BP token posteriors.

The target here is the leaf-token posterior

    p(x_i | x_observed)

for masked leaf positions i.  BP supplies the exact posterior over the RHM
vocabulary at each masked token; a linear probe is trained from encoder features
at those same masked positions to the BP distribution.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from BP_utils import BpRhm, BP_countcorrect_upward
from data2vec import Data2Vec, amp_context
from random_hierarchy_model import sample_data_from_generator_classes


LayerSpec = Union[str, int]


@dataclass
class PosteriorProbeBatch:
    leaves: torch.Tensor
    mask: torch.Tensor
    features: torch.Tensor
    sample_indices: torch.Tensor
    positions: torch.Tensor
    targets: torch.Tensor
    hard_targets: torch.Tensor


@dataclass
class PosteriorProbeMetrics:
    soft_ce: float
    kl_to_bp: float
    top1_agreement: float
    mean_bp_entropy: float
    mean_probe_entropy: float
    num_masked: int


@dataclass
class BpPosteriorComparisonMetrics:
    known_leaf_depth: int
    cross_entropy_exact_to_filtered: float
    kl_exact_to_filtered: float
    reverse_kl_filtered_to_exact: float
    js_divergence: float
    top1_agreement: float
    mean_exact_entropy: float
    mean_filtered_entropy: float
    mean_l1_distance: float
    mean_squared_error: float
    num_masked: int


def infer_rhm_shape(rules: Dict[int, torch.Tensor]) -> Tuple[int, int, int, int]:
    """Return (vocab_size, tuple_size, num_synonyms, num_layers)."""
    first = rules[0]
    vocab_size = int(first.shape[0])
    num_synonyms = int(first.shape[1])
    tuple_size = int(first.shape[2])
    num_layers = len(rules)
    return vocab_size, tuple_size, num_synonyms, num_layers


def rule_levels_from_leaf_depth(
    rules: Dict[int, torch.Tensor],
    known_leaf_depth: int,
) -> List[int]:
    """Return true-rule levels retained when counting known rules from leaves up.

    RHM rules are indexed top-down: rule 0 expands the root/class, and
    rule L-1 expands the parents of leaf tokens.  `known_leaf_depth=1` therefore
    keeps only rule L-1; `known_leaf_depth=L` keeps the exact full RHM.
    """
    num_layers = len(rules)
    if known_leaf_depth < 0 or known_leaf_depth > num_layers:
        raise ValueError(
            f"known_leaf_depth must be in [0, {num_layers}], got {known_leaf_depth}"
        )
    return list(range(num_layers - known_leaf_depth, num_layers))


def branch_transition_matrix(
    rule: torch.Tensor,
    branch: int,
    vocab_size: int,
) -> torch.Tensor:
    """Return P(child_at_branch | parent) induced by uniform synonym choice."""
    num_parents, num_synonyms, tuple_size = rule.shape
    if branch < 0 or branch >= tuple_size:
        raise ValueError(f"branch must be in [0, {tuple_size}), got {branch}")
    if num_parents != vocab_size:
        raise ValueError(
            "Filtered BP currently expects num_classes == vocab_size; "
            f"got {num_parents=} and {vocab_size=}"
        )

    transition = torch.zeros(vocab_size, vocab_size, dtype=torch.float32, device=rule.device)
    parents = torch.arange(num_parents, device=rule.device).repeat_interleave(num_synonyms)
    children = rule[:, :, branch].reshape(-1).long()
    weights = torch.full_like(children, 1.0 / num_synonyms, dtype=torch.float32)
    transition.index_put_((parents, children), weights, accumulate=True)
    return transition


def path_digits(position: int, base: int, length: int) -> List[int]:
    """Return the root-to-node branch path for a level-`length` tree position."""
    digits = [0] * length
    for idx in range(length - 1, -1, -1):
        digits[idx] = position % base
        position //= base
    return digits


def cut_path_transitions(
    rules: Dict[int, torch.Tensor],
    cut_level: int,
    *,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """Return path-specific P(x_cut_position | x_root).

    Shape is (num_cut_positions, vocab_size, vocab_size), with axes
    (position, root_symbol, cut_symbol).  This is the RHM analogue of the
    filtered-model factors P(x_j | x0) in the hierarchical filtering paper.
    """
    vocab_size, tuple_size, _, num_layers = infer_rhm_shape(rules)
    if cut_level < 0 or cut_level > num_layers:
        raise ValueError(f"cut_level must be in [0, {num_layers}], got {cut_level}")
    if cut_level == 0:
        return torch.empty(0, vocab_size, vocab_size, device=device)

    branch_transitions = []
    for level in range(cut_level):
        rule = rules[level].to(device)
        branch_transitions.append(
            [
                branch_transition_matrix(rule, branch, vocab_size)
                for branch in range(tuple_size)
            ]
        )

    transitions = []
    for position in range(tuple_size**cut_level):
        path = path_digits(position, tuple_size, cut_level)
        transition = torch.eye(vocab_size, device=device)
        for level, branch in enumerate(path):
            transition = transition @ branch_transitions[level][branch]
        transitions.append(transition)
    return torch.stack(transitions, dim=0)


def normalize_message(message: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    """Normalize a BP message over the vocabulary axis."""
    return message / message.sum(dim=0, keepdim=True).clamp_min(eps)


class FilteredBpRhm(BpRhm):
    """BP with only a subset of RHM rule tables known.

    Unknown rule factors are marginalized under the random-rule prior.  Since an
    unknown rule table maps each parent to uniformly random child tuples, the
    factor sends uniform upward and downward messages and disconnects information
    across that level.
    """

    def __init__(self, *args, known_rule_levels: Sequence[int], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.known_rule_levels = set(int(l) for l in known_rule_levels)

    def _uniform_message(self, num_variables: int) -> torch.Tensor:
        return torch.ones((self.v, num_variables), device=self.device) / self.v

    def BP_upward_iteration(self):
        for l in range(self.L - 1, -1, -1):
            if l in self.known_rule_levels:
                proba_rules = self.upward_rule_to_proba(l)
                self.nu_up[l] = proba_rules.sum(1)
                self.nu_up[l] = self.nu_up[l] / self.nu_up[l].sum(axis=0, keepdims=True)
            else:
                child_variables = self.nu_up[l + 1].reshape(self.v, -1).shape[1]
                self.nu_up[l] = self._uniform_message(child_variables // self.s)
        return self.nu_up

    def BP_downward_iteration(self):
        for l in range(0, self.L):
            if l in self.known_rule_levels:
                self.nu_down[l + 1] = self.compute_downward_messages(l)
                self.nu_down[l + 1] = self.nu_down[l + 1] / self.nu_down[l + 1].sum(
                    axis=0, keepdims=True
                )
            else:
                child_variables = self.nu_up[l + 1].reshape(self.v, -1).shape[1]
                self.nu_down[l + 1] = self._uniform_message(child_variables)
        return self.nu_down


class PathMarginalFilteredBpRhm(BpRhm):
    """BP on the paper-style filtered hierarchy.

    The upper tree is replaced by conditionally independent factors
    P(x_cut_position | x_root), while the lower tree below `cut_level` keeps the
    original RHM rules.  This preserves the path-specific root-conditioned
    marginals at the cut and removes correlations between cut nodes except
    through the root.
    """

    def __init__(self, *args, cut_level: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if cut_level <= 0 or cut_level > self.L:
            raise ValueError(f"cut_level must be in [1, {self.L}], got {cut_level}")
        self.cut_level = int(cut_level)
        self.num_cut_positions = self.s**self.cut_level
        self.path_transitions = cut_path_transitions(
            self.rules,
            self.cut_level,
            device=self.device,
        )
        self._root_factor_messages: Optional[torch.Tensor] = None
        self._batch_size: Optional[int] = None

    def BP_upward_iteration(self):
        for l in range(self.L - 1, self.cut_level - 1, -1):
            proba_rules = self.upward_rule_to_proba(l)
            self.nu_up[l] = proba_rules.sum(1)
            self.nu_up[l] = normalize_message(self.nu_up[l])

        cut_messages = self.nu_up[self.cut_level].reshape(
            self.v,
            -1,
            self.num_cut_positions,
        )
        self._batch_size = cut_messages.shape[1]

        # root_factor_messages[a, b, j] = sum_x P_j(x | a) u_j^b(x)
        self._root_factor_messages = torch.einsum(
            "jax,xbj->abj",
            self.path_transitions,
            cut_messages,
        )
        self._root_factor_messages = normalize_message(self._root_factor_messages)
        self.nu_up[0] = normalize_message(self._root_factor_messages.prod(dim=2))
        return self.nu_up

    def BP_downward_iteration(self):
        if self._root_factor_messages is None or self._batch_size is None:
            raise RuntimeError("Run BP_upward_iteration before BP_downward_iteration")

        root_prior = self.nu_down[0].reshape(self.v, -1)
        if root_prior.shape[1] == 1:
            root_prior = root_prior.expand(self.v, self._batch_size)
        elif root_prior.shape[1] != self._batch_size:
            raise ValueError(
                f"root prior has {root_prior.shape[1]} columns, expected "
                f"1 or {self._batch_size}"
            )

        cut_down = torch.empty(
            self.v,
            self._batch_size,
            self.num_cut_positions,
            device=self.device,
        )
        for position in range(self.num_cut_positions):
            root_to_factor = root_prior.clone()
            for other in range(self.num_cut_positions):
                if other == position:
                    continue
                root_to_factor = root_to_factor * self._root_factor_messages[:, :, other]
            root_to_factor = normalize_message(root_to_factor)
            cut_down[:, :, position] = torch.einsum(
                "ax,ab->xb",
                self.path_transitions[position],
                root_to_factor,
            )
        self.nu_down[self.cut_level] = normalize_message(
            cut_down.reshape(self.v, self._batch_size * self.num_cut_positions)
        )

        for l in range(self.cut_level, self.L):
            self.nu_down[l + 1] = self.compute_downward_messages(l)
            self.nu_down[l + 1] = normalize_message(self.nu_down[l + 1])
        return self.nu_down

    def compute_all_marginals(self):
        marginals = {0: self.compute_variable_marginals(0)}
        for l in range(self.cut_level, self.L + 1):
            marginals[l] = self.compute_variable_marginals(l)
        return marginals


def sample_rhm_leaves(
    rules: Dict[int, torch.Tensor],
    batch_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample RHM leaves as unshifted token ids in [0, vocab_size)."""
    vocab_size = int(rules[0].shape[0])
    labels = torch.randint(0, vocab_size, size=(batch_size,), generator=generator)
    leaves, _ = sample_data_from_generator_classes(generator, labels, rules)
    return leaves.long()


def make_mask(
    batch_size: int,
    seq_len: int,
    *,
    generator: torch.Generator,
    mask_prob: float = 0.15,
    fixed_positions: Optional[Sequence[int]] = None,
    force_at_least_one: bool = True,
) -> torch.Tensor:
    """Create a boolean mask over leaf positions.

    If `fixed_positions` is given, the same positions are masked in every sample.
    Otherwise positions are sampled independently with probability `mask_prob`.
    """
    if fixed_positions is not None:
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        for pos in fixed_positions:
            if pos < 0 or pos >= seq_len:
                raise ValueError(f"fixed mask position {pos} is outside seq_len={seq_len}")
            mask[:, int(pos)] = True
        return mask

    mask = torch.rand(batch_size, seq_len, generator=generator) < mask_prob
    if force_at_least_one:
        empty = ~mask.any(dim=1)
        if empty.any():
            rows = torch.where(empty)[0]
            fill = torch.randint(0, seq_len, size=(rows.numel(),), generator=generator)
            mask[rows, fill] = True
    return mask


@torch.no_grad()
def bp_leaf_posteriors(
    rules: Dict[int, torch.Tensor],
    leaves: torch.Tensor,
    mask: torch.Tensor,
    *,
    device: Union[str, torch.device] = "cpu",
    known_rule_levels: Optional[Sequence[int]] = None,
    known_leaf_depth: Optional[int] = None,
    filter_mode: str = "path_marginal",
) -> torch.Tensor:
    """Compute exact BP leaf-token marginals under masked evidence.

    Args:
        rules: RHM rules.
        leaves: Unshifted leaf tokens, shape (batch, seq_len).
        mask: Boolean mask, shape (batch, seq_len). True positions are treated as
            unobserved/uniform evidence.
        device: Device for BP computation.
        known_rule_levels: Optional top-down true-rule indices to retain. Unknown
            levels are marginalized as uniform random rule factors.
        known_leaf_depth: Optional convenience argument counting known rule tables
            from leaves upward. For L=4, depth 1 keeps rule 3 only, depth 4 is
            exact BP. Mutually exclusive with `known_rule_levels`.
        filter_mode: For `known_leaf_depth`, "path_marginal" uses the filtered
            hierarchy from arXiv:2408.15138v3: root-to-cut factors are
            P(x_cut | x_root). "uniform" keeps the older baseline where removed
            levels send uniform messages.

    Returns:
        Tensor of shape (batch, seq_len, vocab_size), where each row is a
        posterior distribution over the leaf token vocabulary.
    """
    vocab_size, tuple_size, num_synonyms, num_layers = infer_rhm_shape(rules)
    leaves = leaves.long().to(device)
    mask = mask.bool().to(device)

    if filter_mode not in {"path_marginal", "uniform"}:
        raise ValueError(
            f"filter_mode must be 'path_marginal' or 'uniform', got {filter_mode!r}"
        )
    if known_rule_levels is not None and known_leaf_depth is not None:
        raise ValueError("Pass only one of known_rule_levels or known_leaf_depth")
    bp_kwargs = dict(
        v=vocab_size,
        s=tuple_size,
        m=num_synonyms,
        L=num_layers,
        rules=rules,
        device=device,
    )

    if known_leaf_depth is not None:
        cut_level = num_layers - known_leaf_depth
        if cut_level == 0:
            bp = BpRhm(**bp_kwargs)
        elif filter_mode == "path_marginal":
            bp = PathMarginalFilteredBpRhm(**bp_kwargs, cut_level=cut_level)
        else:
            known_rule_levels = rule_levels_from_leaf_depth(rules, known_leaf_depth)
            bp = FilteredBpRhm(**bp_kwargs, known_rule_levels=known_rule_levels)
    else:
        if known_rule_levels is None:
            known_rule_levels = list(range(num_layers))
        elif set(known_rule_levels) != set(range(num_layers)) and filter_mode != "uniform":
            raise ValueError("Explicit known_rule_levels require filter_mode='uniform'")
        if set(known_rule_levels) == set(range(num_layers)):
            bp = BpRhm(**bp_kwargs)
        else:
            bp = FilteredBpRhm(**bp_kwargs, known_rule_levels=known_rule_levels)

    nu_up_l = bp.set_masking_to_leaf_messages(leaves, mask.flatten())
    marginals = bp.run_BP_from_upward_messages(nu_up_l)
    leaf_marginal = marginals[num_layers].reshape(vocab_size, leaves.shape[0], leaves.shape[1])
    return leaf_marginal.permute(1, 2, 0).contiguous().cpu()


def bp_posterior_comparison_metrics(
    exact_posteriors: torch.Tensor,
    filtered_posteriors: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Compare filtered BP token posteriors to exact BP on masked rows.

    The exact posterior is treated as the target distribution q, and the filtered
    posterior as p.  Metrics are averaged over masked token positions.
    """
    mask = mask.bool()
    q = exact_posteriors[mask].float()
    p = filtered_posteriors[mask].float()
    if q.numel() == 0:
        raise ValueError("No masked tokens were provided for BP posterior comparison")

    log_q = q.clamp_min(eps).log()
    log_p = p.clamp_min(eps).log()
    m = 0.5 * (q + p)
    log_m = m.clamp_min(eps).log()

    cross_entropy = -(q * log_p).sum(dim=-1)
    exact_entropy = -(q * log_q).sum(dim=-1)
    filtered_entropy = -(p * log_p).sum(dim=-1)
    kl_exact_to_filtered = cross_entropy - exact_entropy
    reverse_kl = -(p * log_q).sum(dim=-1) - filtered_entropy
    js_divergence = 0.5 * (q * (log_q - log_m)).sum(dim=-1)
    js_divergence = js_divergence + 0.5 * (p * (log_p - log_m)).sum(dim=-1)
    top1_agreement = (q.argmax(dim=-1) == p.argmax(dim=-1)).float()
    l1_distance = (q - p).abs().sum(dim=-1)
    squared_error = (q - p).square().sum(dim=-1)

    return {
        "cross_entropy_exact_to_filtered": float(cross_entropy.mean().item()),
        "kl_exact_to_filtered": float(kl_exact_to_filtered.mean().item()),
        "reverse_kl_filtered_to_exact": float(reverse_kl.mean().item()),
        "js_divergence": float(js_divergence.mean().item()),
        "top1_agreement": float(top1_agreement.mean().item()),
        "mean_exact_entropy": float(exact_entropy.mean().item()),
        "mean_filtered_entropy": float(filtered_entropy.mean().item()),
        "mean_l1_distance": float(l1_distance.mean().item()),
        "mean_squared_error": float(squared_error.mean().item()),
        "num_masked": int(q.shape[0]),
    }


@torch.no_grad()
def compare_filtered_bp_to_exact(
    rules: Dict[int, torch.Tensor],
    *,
    depths: Optional[Sequence[int]] = None,
    batch_size: int = 512,
    num_batches: int = 8,
    mask_prob: float = 0.15,
    fixed_positions: Optional[Sequence[int]] = None,
    bp_device: Union[str, torch.device] = "cpu",
    bp_filter_mode: str = "path_marginal",
    seed: int = 0,
    eps: float = 1e-12,
) -> List[Dict[str, float]]:
    """Compare exact BP posteriors against filtered BP posteriors.

    `depths` counts known true rule tables from leaves upward.  For L=4,
    depth=0 knows no true rules, depth=1 knows only the bottom rule table, and
    depth=4 is exact BP.  Each depth is evaluated on the same sampled leaves and
    masks, so differences are not due to different evidence.
    """
    _, tuple_size, _, num_layers = infer_rhm_shape(rules)
    seq_len = tuple_size**num_layers
    if depths is None:
        depths = list(range(num_layers + 1))
    depths = [int(depth) for depth in depths]

    generator = torch.Generator()
    generator.manual_seed(seed)
    totals: Dict[int, Dict[str, float]] = {depth: {} for depth in depths}
    counts = {depth: 0 for depth in depths}

    for _ in range(num_batches):
        leaves = sample_rhm_leaves(rules, batch_size=batch_size, generator=generator)
        mask = make_mask(
            batch_size,
            seq_len,
            generator=generator,
            mask_prob=mask_prob,
            fixed_positions=fixed_positions,
        )
        exact = bp_leaf_posteriors(
            rules,
            leaves,
            mask,
            device=bp_device,
            known_leaf_depth=None,
        )

        for depth in depths:
            filtered = bp_leaf_posteriors(
                rules,
                leaves,
                mask,
                device=bp_device,
                known_leaf_depth=depth,
                filter_mode=bp_filter_mode,
            )
            metrics = bp_posterior_comparison_metrics(
                exact,
                filtered,
                mask,
                eps=eps,
            )
            n = metrics["num_masked"]
            counts[depth] += n
            for key, value in metrics.items():
                if key == "num_masked":
                    continue
                totals[depth][key] = totals[depth].get(key, 0.0) + value * n

    rows: List[Dict[str, float]] = []
    for depth in depths:
        count = counts[depth]
        if count == 0:
            raise ValueError("No masked tokens were sampled for BP posterior comparison")
        row = {
            "known_leaf_depth": float(depth),
            "num_masked": float(count),
        }
        for key, value in totals[depth].items():
            row[key] = value / count
        rows.append(row)
    return rows


@torch.no_grad()
def extract_masked_encoder_features(
    model: Data2Vec,
    leaves: torch.Tensor,
    mask: torch.Tensor,
    *,
    device: Union[str, torch.device],
    layer: LayerSpec = "final",
) -> torch.Tensor:
    """Run the student encoder on masked leaves and return per-token features.

    `leaves` are unshifted RHM tokens.  The model input uses +1 shifted token ids,
    with masked positions replaced by `model.mask_idx`.
    """
    model.eval()
    input_ids = leaves.long().to(device) + 1
    input_ids = input_ids.clone()
    input_ids[mask.to(device).bool()] = model.mask_idx

    with amp_context(model):
        if layer == "final":
            hidden = model.encoder(input_ids=input_ids)
        else:
            _, layer_results = model.encoder(input_ids=input_ids, return_all_layers=True)
            hidden = layer_results[int(layer)].hidden
    return hidden.float().cpu()


def flatten_masked(
    features: torch.Tensor,
    posteriors: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return feature/target rows only at masked token positions."""
    mask = mask.bool()
    x = features[mask]
    sample_grid = torch.arange(mask.shape[0]).unsqueeze(1).expand(-1, mask.shape[1])
    sample_indices = sample_grid[mask]
    position_grid = torch.arange(mask.shape[1]).unsqueeze(0).expand(mask.shape[0], -1)
    positions = position_grid[mask]
    q = posteriors[mask]
    y = q.argmax(dim=-1)
    return x, sample_indices, positions, q, y


def make_probe_batch(
    model: Data2Vec,
    rules: Dict[int, torch.Tensor],
    *,
    batch_size: int,
    generator: torch.Generator,
    model_device: Union[str, torch.device],
    bp_device: Union[str, torch.device] = "cpu",
    layer: LayerSpec = "final",
    mask_prob: float = 0.15,
    fixed_positions: Optional[Sequence[int]] = None,
    bp_known_rule_levels: Optional[Sequence[int]] = None,
    bp_known_leaf_depth: Optional[int] = None,
    bp_filter_mode: str = "path_marginal",
) -> PosteriorProbeBatch:
    leaves = sample_rhm_leaves(rules, batch_size=batch_size, generator=generator)
    mask = make_mask(
        batch_size,
        leaves.shape[1],
        generator=generator,
        mask_prob=mask_prob,
        fixed_positions=fixed_positions,
    )
    posteriors = bp_leaf_posteriors(
        rules,
        leaves,
        mask,
        device=bp_device,
        known_rule_levels=bp_known_rule_levels,
        known_leaf_depth=bp_known_leaf_depth,
        filter_mode=bp_filter_mode,
    )
    token_features = extract_masked_encoder_features(
        model,
        leaves,
        mask,
        device=model_device,
        layer=layer,
    )
    features, sample_indices, positions, targets, hard_targets = flatten_masked(
        token_features,
        posteriors,
        mask,
    )
    return PosteriorProbeBatch(
        leaves=leaves,
        mask=mask,
        features=features,
        sample_indices=sample_indices,
        positions=positions,
        targets=targets,
        hard_targets=hard_targets,
    )


class TokenPosteriorProbe(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        features: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.linear(features)


class PositionwiseTokenPosteriorProbe(nn.Module):
    """One independent linear posterior probe per leaf position."""

    def __init__(self, d_model: int, vocab_size: int, seq_len: int):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(d_model, vocab_size) for _ in range(seq_len)])

    def forward(self, features: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        logits = features.new_empty(features.shape[0], self.heads[0].out_features)
        for pos in positions.unique(sorted=True).tolist():
            rows = positions == pos
            logits[rows] = self.heads[int(pos)](features[rows])
        return logits


def soft_cross_entropy(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    return -(target_probs * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


@torch.no_grad()
def evaluate_probe_on_batches(
    probe: nn.Module,
    batches: Iterable[PosteriorProbeBatch],
    *,
    device: Union[str, torch.device],
) -> PosteriorProbeMetrics:
    probe.eval()
    total_ce = 0.0
    total_kl = 0.0
    total_correct = 0
    total_entropy = 0.0
    total_probe_entropy = 0.0
    total = 0
    eps = 1e-12

    for batch in batches:
        x = batch.features.to(device)
        positions = batch.positions.to(device)
        q = batch.targets.to(device)
        logits = probe(x, positions)
        log_p = F.log_softmax(logits, dim=-1)
        p = log_p.exp()
        ce = -(q * log_p).sum(dim=-1)
        entropy = -(q * (q + eps).log()).sum(dim=-1)
        probe_entropy = -(p * log_p).sum(dim=-1)
        kl = ce - entropy
        pred = logits.argmax(dim=-1)
        hard = batch.hard_targets.to(device)

        n = x.shape[0]
        total_ce += ce.sum().item()
        total_kl += kl.sum().item()
        total_correct += (pred == hard).sum().item()
        total_entropy += entropy.sum().item()
        total_probe_entropy += probe_entropy.sum().item()
        total += n

    denom = max(1, total)
    return PosteriorProbeMetrics(
        soft_ce=total_ce / denom,
        kl_to_bp=total_kl / denom,
        top1_agreement=total_correct / denom,
        mean_bp_entropy=total_entropy / denom,
        mean_probe_entropy=total_probe_entropy / denom,
        num_masked=total,
    )


@torch.no_grad()
def rule_validity_metrics(
    leaves: torch.Tensor,
    bp: BpRhm,
    *,
    path_position: Optional[int] = None,
    prefix: str = "sample_rule_valid",
) -> Dict[str, float]:
    """Return compact rule-validity metrics for sampled leaf sequences.

    `BP_countcorrect_upward` returns one tensor per top-down rule level.  This
    wrapper stores scalar summaries suitable for probe histories:
    whole-sequence validity, per-layer means, and optionally the single factor
    on the path from the root to `path_position`.
    """
    if leaves.dim() == 3:
        if leaves.shape[1] == bp.v:
            leaves = leaves.argmax(dim=1)
        elif leaves.shape[-1] == bp.v:
            leaves = leaves.argmax(dim=-1)
        else:
            raise ValueError(
                "3D leaves must be one-hot with vocabulary on dim 1 or dim -1"
            )

    leaves = leaves.long().to(bp.device)
    if leaves.dim() != 2:
        raise ValueError(f"leaves must have shape (batch, seq_len), got {tuple(leaves.shape)}")
    if path_position is not None and not (0 <= int(path_position) < leaves.shape[1]):
        raise ValueError(
            f"path_position must be in [0, {leaves.shape[1]}), got {path_position}"
        )

    frac_correct, _ = BP_countcorrect_upward(leaves, bp)
    metrics: Dict[str, float] = {
        f"{prefix}_full": float(frac_correct[0].float().mean().item())
    }

    for rule_level in sorted(frac_correct):
        leaf_counted_layer = bp.L - int(rule_level)
        values = frac_correct[rule_level].float().reshape(-1)
        metrics[f"{prefix}_layer_{leaf_counted_layer}"] = float(values.mean().item())
        if path_position is not None:
            factor_index = int(path_position) // (bp.s ** (bp.L - int(rule_level)))
            metrics[f"{prefix}_path_layer_{leaf_counted_layer}"] = float(
                values[factor_index].item()
            )

    return metrics


@torch.no_grad()
def sample_probe_rule_validity(
    probe: nn.Module,
    batches: Iterable[PosteriorProbeBatch],
    rules: Dict[int, torch.Tensor],
    *,
    device: Union[str, torch.device],
    bp_device: Union[str, torch.device] = "cpu",
    num_samples: int = 1,
    generator: Optional[torch.Generator] = None,
    path_position: Optional[int] = None,
) -> Dict[str, float]:
    """Sample masked tokens from a probe and check RHM rule consistency."""
    vocab_size, tuple_size, num_synonyms, num_layers = infer_rhm_shape(rules)
    bp = BpRhm(
        v=vocab_size,
        s=tuple_size,
        m=num_synonyms,
        L=num_layers,
        rules=rules,
        device=bp_device,
    )

    probe.eval()
    totals: Dict[str, float] = {}
    total_sequences = 0
    num_samples = int(num_samples)
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")

    for batch in batches:
        x = batch.features.to(device)
        positions_device = batch.positions.to(device)
        logits = probe(x, positions_device)
        probs = logits.softmax(dim=-1).cpu()

        for _ in range(num_samples):
            sampled_tokens = torch.multinomial(
                probs,
                num_samples=1,
                replacement=True,
                generator=generator,
            ).squeeze(-1)
            sampled_leaves = batch.leaves.clone()
            sampled_leaves[batch.sample_indices, batch.positions] = sampled_tokens.long()
            metrics = rule_validity_metrics(
                sampled_leaves,
                bp,
                path_position=path_position,
            )
            n = int(sampled_leaves.shape[0])
            total_sequences += n
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value) * n

    if total_sequences == 0:
        raise ValueError("No sequences were provided for probe rule-validity evaluation")

    out = {key: value / total_sequences for key, value in totals.items()}
    out["sample_rule_check_num_sequences"] = float(total_sequences)
    return out


def train_token_posterior_probe(
    model: Data2Vec,
    rules: Dict[int, torch.Tensor],
    *,
    train_steps: int = 1000,
    batch_size: int = 512,
    lr: float = 1e-3,
    mask_prob: float = 0.15,
    fixed_positions: Optional[Sequence[int]] = None,
    layer: LayerSpec = "final",
    model_device: Union[str, torch.device] = "cuda",
    probe_device: Union[str, torch.device] = "cuda",
    bp_device: Union[str, torch.device] = "cpu",
    seed: int = 0,
    eval_every: int = 100,
    eval_batches: int = 8,
    separate_positions: bool = True,
    bp_known_rule_levels: Optional[Sequence[int]] = None,
    bp_known_leaf_depth: Optional[int] = None,
    bp_filter_mode: str = "path_marginal",
    online_batches: bool = True,
    rule_check: bool = False,
    rule_check_samples: int = 1,
    rule_check_position: Optional[int] = None,
    rule_check_seed: Optional[int] = None,
) -> Tuple[nn.Module, List[Dict[str, float]]]:
    """Train a linear probe from masked-token features to BP posteriors.

    With `online_batches=True`, fresh RHM leaves and masks are sampled for every
    train step and evaluation point.  With `online_batches=False`, one train
    batch and one evaluation set are generated once and reused.
    """
    vocab_size = int(rules[0].shape[0])
    seq_len = int(rules[0].shape[2]) ** len(rules)
    if separate_positions:
        probe = PositionwiseTokenPosteriorProbe(
            d_model=model.d_model,
            vocab_size=vocab_size,
            seq_len=seq_len,
        ).to(probe_device)
    else:
        probe = TokenPosteriorProbe(d_model=model.d_model, vocab_size=vocab_size).to(probe_device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    generator = torch.Generator()
    generator.manual_seed(seed)
    history: List[Dict[str, float]] = []

    model.to(model_device).eval()

    def new_probe_batch(batch_generator: torch.Generator) -> PosteriorProbeBatch:
        return make_probe_batch(
            model,
            rules,
            batch_size=batch_size,
            generator=batch_generator,
            model_device=model_device,
            bp_device=bp_device,
            layer=layer,
            mask_prob=mask_prob,
            fixed_positions=fixed_positions,
            bp_known_rule_levels=bp_known_rule_levels,
            bp_known_leaf_depth=bp_known_leaf_depth,
            bp_filter_mode=bp_filter_mode,
        )

    fixed_train_batch = None
    fixed_eval_data = None
    if not online_batches:
        fixed_train_batch = new_probe_batch(generator)
        eval_generator = torch.Generator()
        eval_generator.manual_seed(seed + 1_000_000)
        fixed_eval_data = [new_probe_batch(eval_generator) for _ in range(eval_batches)]

    for step in range(1, train_steps + 1):
        batch = new_probe_batch(generator) if online_batches else fixed_train_batch
        x = batch.features.to(probe_device)
        positions = batch.positions.to(probe_device)
        q = batch.targets.to(probe_device)

        probe.train()
        logits = probe(x, positions)
        loss = soft_cross_entropy(logits, q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % eval_every == 0 or step == train_steps:
            if online_batches:
                eval_seed = seed + 1_000_000 + step
                eval_generator = torch.Generator()
                eval_generator.manual_seed(eval_seed)
                eval_data = [new_probe_batch(eval_generator) for _ in range(eval_batches)]
            else:
                eval_data = fixed_eval_data
            metrics = evaluate_probe_on_batches(probe, eval_data, device=probe_device)
            rule_metrics: Dict[str, float] = {}
            if rule_check:
                path_position = rule_check_position
                if path_position is None and fixed_positions is not None:
                    fixed_positions_list = list(fixed_positions)
                    if len(fixed_positions_list) == 1:
                        path_position = int(fixed_positions_list[0])
                rule_generator = torch.Generator()
                base_rule_seed = seed + 2_000_000 if rule_check_seed is None else rule_check_seed
                rule_generator.manual_seed(int(base_rule_seed) + int(step))
                rule_metrics = sample_probe_rule_validity(
                    probe,
                    eval_data,
                    rules,
                    device=probe_device,
                    bp_device=bp_device,
                    num_samples=rule_check_samples,
                    generator=rule_generator,
                    path_position=path_position,
                )
            row = {
                "step": float(step),
                "train_soft_ce": float(loss.detach().cpu().item()),
                "eval_soft_ce": metrics.soft_ce,
                "eval_kl_to_bp": metrics.kl_to_bp,
                "eval_top1_agreement": metrics.top1_agreement,
                "eval_mean_bp_entropy": metrics.mean_bp_entropy,
                "eval_mean_probe_entropy": metrics.mean_probe_entropy,
                "eval_num_masked": float(metrics.num_masked),
                "separate_positions": float(separate_positions),
                "online_batches": float(online_batches),
                "bp_known_leaf_depth": (
                    float("nan") if bp_known_leaf_depth is None else float(bp_known_leaf_depth)
                ),
            }
            row.update(rule_metrics)
            history.append(row)
            message = (
                f"step {step:>6}: ce={row['eval_soft_ce']:.4f} "
                f"kl={row['eval_kl_to_bp']:.4f} "
                f"top1={row['eval_top1_agreement']:.4f} "
                f"H_bp={row['eval_mean_bp_entropy']:.4f} "
                f"H_probe={row['eval_mean_probe_entropy']:.4f}"
            )
            if "sample_rule_valid_full" in row:
                message += f" sample_valid={row['sample_rule_valid_full']:.4f}"
            print(message, flush=True)

    return probe, history
