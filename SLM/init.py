"""Data, model and optimizer initialisation for small-language-modeling.

This version preserves the original text-corpus path and adds the Random
Hierarchy Model (RHM) as a first-class dataset.  The model always receives
zero-based integer token ids, independently of the RHM storage format.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import models
from models.transformer_v2 import CLMv2

try:
    from datasets.rhm import (
        rhm_config,
        sample_data_from_generator_classes,
        sample_with_replacement,
        sample_without_replacement,
    )
except (ImportError, ModuleNotFoundError):
    # Avoid a collision with the external HuggingFace ``datasets`` package when
    # this repository's datasets directory is not an explicit Python package.
    local_datasets = Path(__file__).resolve().parent / "datasets"
    if str(local_datasets) not in sys.path:
        sys.path.insert(0, str(local_datasets))
    from rhm import (
        rhm_config,
        sample_data_from_generator_classes,
        sample_with_replacement,
        sample_without_replacement,
    )


class CharacterLevelTokenizer:
    """Character-level tokenizer retained for backward compatibility."""

    def __init__(self, data):
        self.data = data
        self.vocab = sorted(list(set(self.data)))
        self.vocab_size = len(self.vocab)
        self.i_to_s = {i: ch for i, ch in enumerate(self.vocab)}
        self.s_to_i = {ch: i for i, ch in self.i_to_s.items()}

    def encode(self, s):
        return [self.s_to_i[c] for c in s]

    def decode(self, s):
        return "".join([self.i_to_s[i] for i in s])


class MyTextDataLoader:
    """Original contiguous-token mini-batch loader, with a clone method."""

    def __init__(self, B: int, T: int, tokens):
        self.B = int(B)
        self.T = int(T)
        self.tokens = torch.as_tensor(tokens, dtype=torch.long).contiguous()
        self.num_batches = max(1, (len(self.tokens) - 1) // max(1, self.B * self.T))
        needed = self.B * self.T + 1
        if len(self.tokens) < needed:
            raise ValueError(
                f"Text slice has {len(self.tokens)} tokens, but one batch needs {needed}. "
                "Increase train_size/val_size or reduce batch_size/block_size."
            )
        print(f"loaded {len(self.tokens)} tokens, split into {self.num_batches} batches")
        self.current_batch = 0
        self.is_rhm = False

    def __len__(self) -> int:
        return self.num_batches

    def reset(self):
        self.current_batch = 0

    def clone(self):
        return MyTextDataLoader(self.B, self.T, self.tokens.clone())

    def next_batch(self):
        B, T = self.B, self.T
        start = self.current_batch
        end = start + B * T + 1
        if end > len(self.tokens):
            self.reset()
            start = 0
            end = B * T + 1
        chunk = self.tokens[start:end]
        targets = chunk[1:].view(B, T)
        inputs = chunk[:-1].view(B, T)
        self.current_batch += B * T
        return inputs, targets


class RHMDataLoader:
    """Mini-batch loader for fixed or freshly sampled RHM sequences.

    In online mode, every training call draws fresh sequences from the same
    fixed rules.  Deterministic online loaders are used for validation and RHM
    diagnostics so that all checkpoints are compared on the same finite sample.
    """

    is_rhm = True

    def __init__(
        self,
        *,
        rules,
        batch_size: int,
        num_samples: int,
        seed: int,
        num_classes: int,
        sequences: Optional[torch.Tensor] = None,
        online: bool = False,
        deterministic: bool = False,
        shuffle: bool = False,
        name: str = "rhm",
    ):
        self.rules = rules
        self.B = int(batch_size)
        self.batch_size = int(batch_size)
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.num_classes = int(num_classes)
        self.online = bool(online)
        self.deterministic = bool(deterministic)
        self.shuffle = bool(shuffle)
        self.name = str(name)
        self.sequences = None if sequences is None else torch.as_tensor(sequences, dtype=torch.long).contiguous()
        if not self.online and self.sequences is None:
            raise ValueError("Offline RHMDataLoader requires fixed sequences.")
        if self.sequences is not None:
            self.num_samples = int(len(self.sequences))
        if self.num_samples <= 0:
            raise ValueError(f"{self.name}: num_samples must be positive, got {self.num_samples}")
        self.num_batches = max(1, math.ceil(self.num_samples / max(1, self.B)))
        self._generator = torch.Generator().manual_seed(self.seed)
        self._cursor = 0
        self._order = torch.arange(self.num_samples)
        self.reset()

    def __len__(self) -> int:
        return self.num_batches

    def reset(self):
        self._cursor = 0
        if self.online and self.deterministic:
            self._generator = torch.Generator().manual_seed(self.seed)
        elif not self.online and self.shuffle:
            self._order = torch.randperm(self.num_samples, generator=self._generator)
        else:
            self._order = torch.arange(self.num_samples)

    def clone(self, *, batch_size: Optional[int] = None):
        return RHMDataLoader(
            rules=self.rules,
            batch_size=self.B if batch_size is None else int(batch_size),
            num_samples=self.num_samples,
            seed=self.seed,
            num_classes=self.num_classes,
            sequences=self.sequences,
            online=self.online,
            deterministic=self.deterministic,
            shuffle=False,
            name=f"{self.name}_clone",
        )

    def _next_batch_size(self) -> int:
        remaining = self.num_samples - self._cursor
        if remaining <= 0:
            self.reset()
            remaining = self.num_samples
        return min(self.B, remaining)

    def next_sequence_batch(self) -> torch.Tensor:
        bs = self._next_batch_size()
        if self.online:
            labels = torch.randint(
                low=0,
                high=self.num_classes,
                size=(bs,),
                generator=self._generator,
            )
            sequences, _ = sample_data_from_generator_classes(self._generator, labels, self.rules)
            sequences = sequences.long()
        else:
            idx = self._order[self._cursor : self._cursor + bs]
            sequences = self.sequences[idx]
        self._cursor += bs
        return sequences.contiguous()

    def next_batch(self):
        sequences = self.next_sequence_batch()
        if sequences.ndim != 2 or sequences.shape[1] < 2:
            raise ValueError(f"Expected [B,d] RHM sequences with d>=2, got {tuple(sequences.shape)}")
        return sequences[:, :-1], sequences[:, 1:]

    def __iter__(self):
        self.reset()
        for _ in range(self.num_batches):
            # rhm_margins.py expects the full sequence, not shifted inputs/targets.
            yield self.next_sequence_batch()


def _normalise_rhm_features(features, num_features: int, num_tokens: int) -> torch.Tensor:
    x = torch.as_tensor(features)
    if x.ndim == 2:
        x = x.long()
        if x.numel() and int(x.min()) >= 1 and int(x.max()) <= int(num_features):
            x = x - 1
        return x
    if x.ndim == 3:
        if x.shape[1] == num_features and x.shape[2] == num_tokens:
            return x.argmax(dim=1).long()
        if x.shape[1] == num_tokens and x.shape[2] == num_features:
            return x.argmax(dim=-1).long()
    raise ValueError(f"Cannot convert RHM features with shape {tuple(x.shape)} to [N,d] tokens.")


def _random_slice(corpus: np.ndarray, requested: int, rng: random.Random) -> np.ndarray:
    requested = int(requested)
    if requested <= 0:
        return corpus
    if requested > len(corpus):
        raise ValueError(f"Requested {requested} tokens from a corpus of length {len(corpus)}")
    max_start = len(corpus) - requested
    start = 0 if max_start == 0 else rng.randint(0, max_start)
    return corpus[start : start + requested]


def _init_text_data(config):
    if config.tokenizer is None:
        raise ValueError("--tokenizer is required for text datasets.")
    base = Path(config.path)
    with open(base / config.tokenizer, "r", encoding="utf-8") as handle:
        tokenizer = json.load(handle)

    config.vocab_size = int(tokenizer["vocab_size"])
    config.eos_token_id = tokenizer.get("eos_token_id")
    config.unk_token_id = tokenizer.get("unk_token_id")
    print("vocabulary size:", config.vocab_size)

    rank = int(getattr(config, "rank", 0))
    world_size = int(getattr(config, "world_size", 1))
    rng = random.Random(int(getattr(config, "seed_sample", 0)) + 1000003 * rank)
    train_corpus = np.load(base / f"{config.dataset}.train.npy")
    valid_corpus = np.load(base / f"{config.dataset}.valid.npy")
    print("number of training tokens:", len(train_corpus))
    print("number of validation tokens:", len(valid_corpus))

    train_tokens = _random_slice(train_corpus, int(config.train_size) + 1, rng)
    valid_tokens = _random_slice(valid_corpus, int(config.val_size) + 1, rng)
    train_loader = MyTextDataLoader(config.batch_size, config.block_size, train_tokens)
    train_eval_loader = train_loader.clone()
    val_loader = MyTextDataLoader(config.eval_batch_size, config.block_size, valid_tokens)

    info = SimpleNamespace(
        dataset=str(config.dataset),
        is_rhm=False,
        online=False,
        rules=None,
        train_sequences=None,
        val_sequences=None,
    )
    return tokenizer, train_loader, val_loader, train_eval_loader, info


def _init_rhm_data(config):
    """Initialise RHM data directly through ``datasets/rhm.py``."""
    config.num_classes = int(config.num_features if config.num_classes is None else config.num_classes)
    config.num_tokens = int(config.tuple_size) ** int(config.num_layers)
    expected_block = config.num_tokens - 1
    if config.block_size is None:
        config.block_size = expected_block
    elif int(config.block_size) != expected_block:
        raise ValueError(
            f"For RHM next-token training block_size must be s^L-1={expected_block}, "
            f"got {config.block_size}."
        )

    config.vocab_size = int(config.num_features)
    config.eos_token_id = None
    config.unk_token_id = None
    config.rhm_tokens_are_zero_based = True

    hierarchy = rhm_config(
        v=int(config.num_features),
        n=int(config.num_classes),
        m=int(config.num_synonyms),
        s=int(config.tuple_size),
        L=int(config.num_layers),
        seed=int(config.seed_rules),
    )
    rules = hierarchy.rules
    config.pmax = int(hierarchy.pmax)

    eval_train_size = int(config.eval_train_size or min(config.train_size, 16384))
    eval_val_size = int(config.eval_val_size or config.val_size)
    if eval_train_size <= 0 or eval_val_size <= 0:
        raise ValueError("RHM evaluation reference sizes must be positive.")

    if config.online:
        train_loader = RHMDataLoader(
            rules=rules,
            batch_size=config.batch_size,
            num_samples=config.train_size,
            seed=int(config.seed_sample) + 1000003 * int(getattr(config, "rank", 0)),
            num_classes=config.num_classes,
            online=True,
            deterministic=False,
            name="online_train_stream",
        )
        train_eval_loader = RHMDataLoader(
            rules=rules,
            batch_size=config.eval_batch_size,
            num_samples=eval_train_size,
            seed=config.seed_sample + 104729,
            num_classes=config.num_classes,
            online=True,
            deterministic=True,
            name="online_train_reference",
        )
        val_loader = RHMDataLoader(
            rules=rules,
            batch_size=config.eval_batch_size,
            num_samples=eval_val_size,
            seed=config.seed_sample + 2097593,
            num_classes=config.num_classes,
            online=True,
            deterministic=True,
            name="online_validation_reference",
        )
        train_sequences = None
        val_sequences = None
        sampling = "fresh_with_replacement_from_fixed_tree"
    else:
        train_size = int(config.train_size)
        val_size = int(config.val_size)
        use_replacement = bool(config.replacement)

        # ``random.sample(range(pmax), ...)`` cannot represent ranges whose
        # Python length exceeds sys.maxsize.  Preserve the previous repository's
        # safe fallback while still using the new rhm.py sampling functions.
        if (not use_replacement) and hierarchy.pmax > sys.maxsize:
            print(
                "RHM pmax exceeds sys.maxsize; switching offline sampling to replacement.",
                flush=True,
            )
            use_replacement = True
            config.replacement = True

        if use_replacement:
            all_sequences, _ = sample_with_replacement(
                train_size, val_size, int(config.seed_sample), rules
            )
            actual_train_size = 1_000_000 if train_size == -1 else train_size
        else:
            if train_size == -1:
                actual_train_size = int(hierarchy.pmax)
            else:
                actual_train_size = min(train_size, int(hierarchy.pmax))
            available_val = max(0, int(hierarchy.pmax) - actual_train_size)
            actual_val_size = min(val_size, available_val)
            all_sequences, _ = sample_without_replacement(
                int(hierarchy.pmax),
                actual_train_size,
                actual_val_size,
                int(config.seed_sample),
                rules,
            )

        all_sequences = _normalise_rhm_features(
            all_sequences, config.num_features, config.num_tokens
        )
        train_sequences = all_sequences[:actual_train_size]
        val_sequences = all_sequences[actual_train_size:]
        if len(val_sequences) == 0:
            raise ValueError(
                "Offline RHM sampling produced no validation sequences. "
                "Reduce train_size or enable --replacement."
            )

        train_loader = RHMDataLoader(
            rules=rules,
            batch_size=config.batch_size,
            num_samples=len(train_sequences),
            seed=int(config.seed_sample) + 1000003 * int(getattr(config, "rank", 0)),
            num_classes=config.num_classes,
            sequences=train_sequences,
            online=False,
            shuffle=True,
            name="offline_train",
        )
        train_eval_loader = RHMDataLoader(
            rules=rules,
            batch_size=config.eval_batch_size,
            num_samples=min(eval_train_size, len(train_sequences)),
            seed=config.seed_sample + 104729,
            num_classes=config.num_classes,
            sequences=train_sequences[:eval_train_size],
            online=False,
            shuffle=False,
            name="offline_train_reference",
        )
        val_loader = RHMDataLoader(
            rules=rules,
            batch_size=config.eval_batch_size,
            num_samples=min(eval_val_size, len(val_sequences)),
            seed=config.seed_sample + 2097593,
            num_classes=config.num_classes,
            sequences=val_sequences[:eval_val_size],
            online=False,
            shuffle=False,
            name="offline_validation",
        )
        sampling = "fixed_with_replacement" if use_replacement else "fixed_without_replacement"

    info = SimpleNamespace(
        dataset="rhm",
        is_rhm=True,
        online=bool(config.online),
        rules=rules,
        rhm_config=hierarchy,
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        sampling=sampling,
        num_tokens=config.num_tokens,
        pmax=hierarchy.pmax,
    )
    print(
        "RHM:",
        f"v={config.num_features}, n={config.num_classes}, m={config.num_synonyms}, "
        f"s={config.tuple_size}, L={config.num_layers}, d={config.num_tokens}, "
        f"pmax={hierarchy.pmax}, online={config.online}",
    )
    return None, train_loader, val_loader, train_eval_loader, info


def init_data(config):
    """Return tokenizer, train loader, validation loader, train-eval loader and metadata."""
    if str(config.dataset).lower() == "rhm":
        result = _init_rhm_data(config)
    else:
        result = _init_text_data(config)

    _, train_loader, _, _, _ = result
    if getattr(config, "max_steps", None) is None or int(config.max_steps) <= 0:
        if bool(config.online) and str(config.dataset).lower() != "rhm":
            config.max_steps = int(train_loader.num_batches)
        else:
            config.max_steps = int(config.max_epochs) * int(train_loader.num_batches)
    else:
        config.max_steps = int(config.max_steps)
    print(f"Training for {config.max_steps} optimizer steps")
    return result


def init_model(config, seed=None):
    """Initialise either the GPT-2-style transformer or the Mamba language model."""
    seed = config.seed_model if seed is None else seed
    torch.manual_seed(int(seed))

    if config.model == "mamba":
        model = models.MambaLM(
            vocab_size=config.vocab_size,
            d_model=config.d_embedding,
            depth=config.depth,
            d_state=getattr(config, "d_state", 16),
            d_conv=getattr(config, "d_conv", 4),
            expand=getattr(config, "mamba_expand", 2),
            dropout=config.dropout,
            share_emb=False,
        )
    elif config.model == "gpt2":
        model = models.CLM(
            vocab_size=config.vocab_size,
            block_size=config.block_size,
            embedding_dim=config.d_embedding,
            num_heads=config.n_heads,
            ffwd_size=config.ffwd_size,
            num_layers=config.depth,
            dropout=config.dropout,
            rope=config.rope,
            share_emb=False,
        )
    elif config.model == "transformer_v2":
        if bool(getattr(config, "rope", False)):
            raise ValueError("transformer_v2 follows the notebook architecture and uses learned absolute positions; disable --rope.")
        model = CLMv2(
            vocab_size=config.vocab_size,
            block_size=config.block_size,
            embedding_dim=config.d_embedding,
            num_heads=config.n_heads,
            ffwd_size=config.ffwd_size,
            num_layers=config.depth,
            dropout=config.dropout,
            muP=bool(getattr(config, "mup", True)),
            embedding_scale=float(getattr(config, "embedding_scale", 0.05)),
            share_emb=False,
        )
    else:
        raise ValueError(
            f"Unknown model {config.model!r}; use 'gpt2', 'transformer_v2' or 'mamba'."
        )

    model.to(config.device)
    print("# parameters:", sum(p.numel() for p in model.parameters()))
    return model


def init_training(model, config):
    """Initialise cross entropy, AdamW and the selected learning-rate schedule."""
    raw_model = model.module if hasattr(model, "module") else model
    criterion = nn.CrossEntropyLoss(reduction="mean")
    if config.optim != "adam":
        raise ValueError("Only AdamW is implemented in this repository.")
    if config.model == "transformer_v2":
        optimizer = raw_model.configure_optimizers(lr=config.lr, wd=config.l2)
    else:
        optimizer = optim.AdamW(raw_model.parameters(), lr=config.lr, weight_decay=config.l2)

    if config.scheduler == "cosine":
        scheduler = CosineWarmupLR(
            optimizer,
            config.warmup_time,
            config.decay_time,
            config.decay_factor,
        )
    elif config.scheduler == "none":
        scheduler = NoOpScheduler()
    else:
        raise ValueError("scheduler must be 'cosine' or 'none'")
    return criterion, optimizer, scheduler


class NoOpScheduler:
    def step(self):
        return None


class CosineWarmupLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_time, decay_time, min_lr_factor):
        self.warmup = max(0, int(warmup_time))
        self.decay = max(self.warmup + 1, int(decay_time))
        self.min_lr_factor = float(min_lr_factor)
        super().__init__(optimizer)

    def get_lr(self):
        factor = self.get_lr_factor(step=self.last_epoch)
        return [base_lr * factor for base_lr in self.base_lrs]

    def get_lr_factor(self, step):
        if self.warmup > 0 and step < self.warmup:
            return max(0.0, float(step) / float(self.warmup))
        if step < self.decay:
            decay_step = step - self.warmup
            total_decay = max(1, self.decay - self.warmup)
            cosine_decay = 0.5 * (1.0 + np.cos(np.pi * decay_step / total_decay))
            return self.min_lr_factor + (1.0 - self.min_lr_factor) * cosine_decay
        return self.min_lr_factor
