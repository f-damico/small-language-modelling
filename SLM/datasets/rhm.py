from __future__ import annotations

import random
from itertools import product

import torch
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class rhm_config:
    v: int
    n: int
    m: int
    s: int
    L: int
    seed: int

    def __post_init__(self):
        self.rules = sample_rules(
            self.v, self.n, self.m, self.s, self.L, seed=self.seed
        )
        self.inv_rules = build_inverse_rules(self.rules, self.s, self.v)
        self.pmax = compute_pmax(self.n, self.m, self.s, self.L)
        self.inv_rules_unclustered = build_inverse_rules_unclustered(
            self.rules, self.s, self.v, self.m
        )

    def __str__(self):
        return f"v={self.v}, n={self.n}, m={self.m}, s={self.s}, L={self.L}, seed={self.seed}"

    def to_dict(self):
        return {
            "v": self.v,
            "n": self.n,
            "m": self.m,
            "s": self.s,
            "L": self.L,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            v=d["v"],
            n=d["n"],
            m=d["m"],
            s=d["s"],
            L=d["L"],
            seed=d["seed"],
        )


def dec2bin(n: torch.Tensor, bits: int | None = None) -> torch.Tensor:
    """
    Convert integers to binary.

    Args:
        n: Tensor of integers.
        bits: Optional length of the representation.
    """
    x = n.int()
    if bits is None:
        if x.numel() == 0:
            bits = 1
        else:
            bits = int((x.max() + 1).log2().ceil().item())
            bits = max(bits, 1)
    mask = 2 ** torch.arange(bits - 1, -1, -1, device=x.device, dtype=x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()


def dec2base(n: torch.Tensor, base: int, length: int | None = None) -> torch.Tensor:
    """
    Convert integers into a different base.

    Args:
        n: Tensor of integers.
        base: The base.
        length: Optional length of the representation.
    """
    x = n.int().clone()
    if x.numel() == 0:
        return x
    if x.sum() == 0:
        if length is None:
            return torch.zeros((x.numel(), 1), dtype=x.dtype, device=x.device)
        return torch.zeros((x.numel(), length), dtype=x.dtype, device=x.device)

    digits = []
    while x.sum():
        digits.append(x % base)
        x = x.div(base, rounding_mode="floor")
    if length is not None:
        if len(digits) > length:
            raise ValueError("length too small to represent input numbers")
        pad = length - len(digits)
        if pad:
            zeros = torch.zeros_like(digits[0])
            digits += [zeros] * pad

    out = torch.stack(digits[::-1], dim=1)
    return out


def sample_rules(
    v: int, n: int, m: int, s: int, L: int, seed: int = 42
) -> dict[int, torch.Tensor]:
    """
    Sample random rules for a random hierarchy model.
    """
    random.seed(seed)
    tuples = list(product(*[range(v) for _ in range(s)]))

    rules: dict[int, torch.Tensor] = {}
    rules[0] = torch.tensor(random.sample(tuples, n * m)).reshape(n, m, -1)
    for i in range(1, L):
        rules[i] = torch.tensor(random.sample(tuples, v * m)).reshape(v, m, -1)
    return rules


def sample_data_from_generator_classes(
    g: torch.Generator, y: torch.Tensor, rules: dict[int, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample data from class labels using the sampled rules.
    """
    L = len(rules)
    labels = y.clone()
    x = y
    for i in range(L):
        chosen_rule = torch.randint(
            low=0, high=rules[i].shape[1], size=x.shape, generator=g
        )
        x = rules[i][x, chosen_rule].flatten(start_dim=1)
    return x, labels


def sample_with_replacement(
    train_size: int, test_size: int, seed_sample: int, rules: dict[int, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    n = rules[0].shape[0]
    if train_size == -1:
        train_size = 1_000_000
    g = torch.Generator().manual_seed(seed_sample)
    y = torch.randint(low=0, high=n, size=(train_size + test_size,), generator=g)
    return sample_data_from_generator_classes(g, y, rules)


def sample_data_from_indices(
    samples: torch.Tensor,
    rules: dict[int, torch.Tensor],
    n: int,
    m: int,
    s: int,
    L: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_data = n * m ** ((s**L - 1) // (s - 1))
    data_per_hl = max_data // n

    high_level = samples.div(data_per_hl, rounding_mode="floor")
    low_level = samples % data_per_hl

    labels = high_level
    features = labels
    size = 1

    for ell in range(L):
        choices = m ** (size)
        data_per_hl = data_per_hl // choices
        high_level = low_level.div(data_per_hl, rounding_mode="floor")
        high_level = dec2base(high_level, m, length=size).squeeze()
        features = rules[ell][features, high_level]
        features = features.flatten(start_dim=1)
        size *= s
        low_level = low_level % data_per_hl

    return features, labels


def sample_without_replacement(
    max_data: int,
    train_size: int,
    test_size: int,
    seed_sample: int,
    rules: dict[int, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    L = len(rules)
    n = rules[0].shape[0]
    m = rules[0].shape[1]
    s = rules[0].shape[2]

    if train_size == -1:
        samples = torch.arange(max_data)
    else:
        test_size = min(test_size, max_data - train_size)
        random.seed(seed_sample)
        samples = torch.tensor(random.sample(range(max_data), train_size + test_size))

    return sample_data_from_indices(samples, rules, n, m, s, L)


def infer_model_parameters_from_rules(rules: dict[int, torch.Tensor]) -> dict[str, int]:
    return {
        "m": rules[0].shape[1],
        "s": rules[0].shape[2],
        "L": len(rules),
        "n": rules[0].shape[0],
        "v": 1 + int(max(torch.max(rules[i]).item() for i in range(1, len(rules)))),
    }


def convert_features_to_one_hot(features: torch.Tensor, v: int) -> torch.Tensor:
    return F.one_hot(features, v).transpose(-2, -1).float()


def compute_pmax(n: int, m: int, s: int, L: int) -> int:
    return n * m ** ((s**L - 1) // (s - 1))


def build_inverse_rules(rules, s=None, v=None):
    """
    Builds inverse rules mapping from child configurations to parent symbols.
    This function takes a dictionary of rules where each entry maps a parent symbol to its
    child configuration, and creates an inverse mapping from child configurations back to
    their parent symbols.
    Parameters
    ----------
    rules : dict
        Dictionary where keys are layer indices and values are tensors representing
        production rules. Each rule maps a parent symbol to its child configuration.
    s : int, optional
        The branching factor (number of children per parent). If None, it's inferred from the rules.
    v : int, optional
        The number of states/symbols in the model. If None, it's inferred from the rules.
    Returns
    -------
    dict
        Dictionary where keys are layer indices and values are tensors of shape (v, v, ..., v)
        with s dimensions. Each entry maps a child configuration (represented as indices)
        to its parent symbol, with -1 for invalid configurations.
    Notes
    -----
    The inverse rules tensor for each layer has shape (v, v, ..., v) with s dimensions,
    where each position stores the parent symbol that would generate the child configuration
    represented by the indices. Invalid configurations are marked with -1.
    """

    if v is None:
        v = infer_model_parameters_from_rules(rules)["v"]
    if s is None:
        s = infer_model_parameters_from_rules(rules)["s"]
    inverse_rules = {}
    for ell, layer_rules in rules.items():
        inverse_rules[ell] = -torch.ones(size=(s * (v,)), dtype=torch.long)
        for symbol, rule in enumerate(layer_rules):
            inverse_rules[ell][tuple(rule.unbind(1))] = symbol
    return inverse_rules


def build_inverse_rules_unclustered(rules, s=None, v=None, m=None):
    """
    Builds inverse rules mapping from child configurations to parent symbols, where parents symbols
    are in the vocabulary of size v*m. That is, it takes v^s tuples to v*m symbols.
    Parameters:
    ----------
    rules : dict
        Dictionary where keys are layer indices and values are tensors representing
        production rules. Each rule maps a parent symbol to its child configuration.
    s : int, optional
        The branching factor (number of children per parent). If None, it's inferred from the rules.
    v : int, optional
        The number of states/symbols in the model. If None, it's inferred from the rules.
    m : int, optional
        The number of synonyms per parent symbol. If None, it's inferred from the rules.
    Returns
    -------
    dict
        Dictionary where keys are layer indices and values are tensors of shape (v, v, ..., v)
        with s dimensions. Each entry maps a child configuration (represented as indices)
        to its "parent" symbol, with -1 for invalid configurations.
    Notes
    -----
    The inverse rules tensor for each layer has shape (v, v, ..., v) with s dimensions,
    where each position stores the parent symbol that would generate the child configuration
    represented by the indices. Invalid configurations are marked with -1.
    """
    if v is None:
        v = infer_model_parameters_from_rules(rules)["v"]
    if s is None:
        s = infer_model_parameters_from_rules(rules)["s"]
    if m is None:
        m = infer_model_parameters_from_rules(rules)["m"]
    inverse_rules = {}
    for ell, layer_rules in rules.items():
        inverse_rules[ell] = -torch.ones(size=(s * (v,)), dtype=torch.long)
        for symbol, symbol_rules in enumerate(layer_rules):
            for syn, rule in enumerate(symbol_rules):
                inverse_rules[ell][tuple(rule.unbind())] = syn + m * symbol
    return inverse_rules


def build_latents_from_inv_rules(
    x: torch.Tensor,
    inv_rules: dict,
    s: int,
    L: int,
    copy_L: bool = True,
    max_level: int = 0,
) -> dict:
    """
    Build latent representations for each layer by applying inverse rules.

    This function constructs latent representations for each layer in a hierarchical model
    by applying inverse rules starting from the highest layer and working downwards.

    Parameters
    ----------
    x : torch.tensor
        Input tensor representing the highest layer latent variables.

    inv_rules : dict
        Dictionary mapping layer indices to inverse rule functions.
        Each rule transforms variables from layer L to layer L-1.

    s : int
        Size of grouping for applying the inverse rules (usually branching factor).

    L : int
        Index of the highest layer in the hierarchy.

    copy_L : bool, default=True
        If True, creates a copy of input tensor x for the highest layer latent.
        If False, uses the original tensor x without copying.

    Returns
    -------
    dict
        Dictionary mapping layer indices to their corresponding latent representations.
        The keys range from 0 to L, where L is the highest layer.
    """
    latents = {}
    latents[L] = x.clone() if copy_L else x
    for layer in range(L, max_level, -1):
        ell = layer - 1
        x_s_tuples = latents[layer].reshape(-1, s)
        latents[ell] = inv_rules[ell][tuple(x_s_tuples.unbind(1))].reshape(x.shape[0], -1)
    return latents


def build_latents_from_inv_rules_unclustered(
    x: torch.Tensor,
    inv_rules_unclustered: dict,
    s: int,
    L: int,
    *,
    inv_rules: dict | None = None,
    latents: dict | None = None,
    copy_L: bool = True,
) -> dict:
    """
    Build unclustered latent representations for each layer by applying inverse rules.
    """
    if latents is None:
        assert inv_rules is not None, (
            "Must provide inv_rules if latents is not provided"
        )
        latents = build_latents_from_inv_rules(
            x=x,
            inv_rules=inv_rules,
            s=s,
            L=L,
            copy_L=False,
        )
    # Latents is now not none.
    assert latents is not None
    unclustered_latents = {}
    unclustered_latents[L] = x.clone() if copy_L else x

    # TODO: proceed layer-wise.
    for layer in range(L, 0, -1):
        ell = layer - 1
        x_s_tuples = latents[layer].reshape(-1, s)
        unclustered_latents[ell] = inv_rules_unclustered[ell][
            tuple(x_s_tuples.unbind(1))
        ].reshape(x.shape[0], -1)
    return unclustered_latents
