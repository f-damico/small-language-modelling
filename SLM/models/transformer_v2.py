"""Alternative causal Transformer adapted from the attached ``ntp.ipynb``.

The original ``models/transformer.py`` is intentionally untouched.  This module
keeps the repository's public language-model API (integer token ids in,
``[B, T, vocab_size]`` logits out), while using the notebook architecture:

* one-hot tokens followed by a learned linear embedding;
* learned absolute positional embeddings;
* explicit per-head Q/K/V tensors with no attention output projection;
* pre-LayerNorm residual blocks;
* ReLU feed-forward networks;
* optional muP-style attention/MLP/embedding initialisation;
* a zero-initialised output-head weight.

The repository-facing attribute names (``token_embedding_table``, ``blocks``,
``attn``, ``ffwd``, ``ln1``, ``ln2``, ``ln_f`` and ``lm_head``) are retained so
that the existing diagnostics and checkpoint-overlap code work without changes.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def _matrix_spectral_and_21_norm(parameter: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    matrix = parameter.detach().to(dtype=torch.float64).reshape(parameter.shape[0], -1)
    spectral = torch.linalg.matrix_norm(matrix, ord=2)
    transpose_21 = torch.linalg.vector_norm(matrix, ord=2, dim=0).sum()
    return spectral, transpose_21


@torch.no_grad()
def _bartlett_log_complexity(named_parameters, *, exclude_qk: bool = False) -> torch.Tensor:
    log_product = None
    correction = None
    first_parameter = None
    eps = torch.finfo(torch.float64).tiny

    for name, parameter in named_parameters:
        if first_parameter is None:
            first_parameter = parameter
        if not parameter.requires_grad or parameter.ndim < 2:
            continue

        lname = name.lower()
        if exclude_qk and (
            lname.endswith(".query")
            or lname.endswith(".key")
            or lname.endswith(".wq")
            or lname.endswith(".wk")
            or ".query." in lname
            or ".key." in lname
            or ".wq." in lname
            or ".wk." in lname
            or "q_proj" in lname
            or "k_proj" in lname
        ):
            continue

        spectral, transpose_21 = _matrix_spectral_and_21_norm(parameter)
        spectral = torch.clamp(spectral, min=eps)
        term = torch.pow(torch.clamp(transpose_21 / spectral, min=eps), 2.0 / 3.0)
        log_product = torch.log(spectral) if log_product is None else log_product + torch.log(spectral)
        correction = term if correction is None else correction + term

    if log_product is None or correction is None:
        device = first_parameter.device if first_parameter is not None else torch.device("cpu")
        return torch.tensor(float("-inf"), dtype=torch.float64, device=device)

    return log_product + 1.5 * torch.log(torch.clamp(correction, min=eps))


@torch.no_grad()
def _l2_log_norm(parameters) -> torch.Tensor:
    total = None
    first_parameter = None
    for parameter in parameters:
        if first_parameter is None:
            first_parameter = parameter
        if not parameter.requires_grad:
            continue
        value = torch.sum(parameter.detach().to(torch.float64) ** 2)
        total = value if total is None else total + value

    if total is None:
        device = first_parameter.device if first_parameter is not None else torch.device("cpu")
        return torch.tensor(float("-inf"), dtype=torch.float64, device=device)
    return 0.5 * torch.log(torch.clamp(total, min=torch.finfo(torch.float64).tiny))


class OneHotLinearEmbedding(nn.Linear):
    """Apply an ``nn.Linear`` embedding to integer ids through one-hot encoding."""

    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__(vocab_size, embedding_dim, bias=True)
        self.vocab_size = int(vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim >= 1 and token_ids.dtype in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            x = F.one_hot(token_ids.long(), num_classes=self.vocab_size).to(dtype=self.weight.dtype)
        else:
            x = token_ids.to(dtype=self.weight.dtype)
            if x.shape[-1] != self.vocab_size:
                raise ValueError(
                    f"Expected integer token ids or one-hot inputs with last dimension "
                    f"{self.vocab_size}, got shape {tuple(x.shape)}."
                )
        return F.linear(x, self.weight, self.bias)


class CausalMultiHeadAttention(nn.Module):
    """Notebook-style causal multi-head attention with explicit Q/K/V tensors."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_k: int,
        num_heads: int,
        *,
        muP: bool = True,
        dropout: float = 0.0,
        max_seq_len: int,
    ):
        super().__init__()
        self.d_in = int(d_in)
        self.d_out = int(d_out)
        self.d_k = int(d_k)
        self.num_heads = int(num_heads)
        self.muP = bool(muP)

        # Shapes are the same as WQ, WK and WV in the notebook.
        self.WQ = nn.Parameter(torch.randn(self.d_in, self.d_k, self.num_heads))
        self.WK = nn.Parameter(torch.randn(self.d_k, self.num_heads, self.d_in))
        self.WV = nn.Parameter(torch.randn(self.d_out, self.num_heads, self.d_in))

        self.dropout = nn.Dropout(float(dropout))
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(int(max_seq_len), int(max_seq_len), dtype=torch.bool)),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, input_dim = x.shape
        if input_dim != self.d_in:
            raise ValueError(f"Expected input dimension {self.d_in}, got {input_dim}.")
        if seq_len > self.tril.shape[0]:
            raise ValueError(
                f"Sequence length {seq_len} exceeds configured maximum {self.tril.shape[0]}."
            )

        # Q, K: [B, T, H, d_k]; V: [B, T, H, d_out].
        q = torch.einsum("ikh,bti->bthk", self.WQ, x)
        k = torch.einsum("khi,bti->bthk", self.WK, x)
        v = torch.einsum("ohi,bti->btho", self.WV, x)

        scores = torch.einsum("bthk,bshk->bhts", q, k)
        scores = scores / (float(self.d_k) if self.muP else math.sqrt(float(self.d_k)))
        scores = scores.masked_fill(~self.tril[:seq_len, :seq_len][None, None], float("-inf"))
        attention = self.dropout(F.softmax(scores, dim=-1))

        output = torch.einsum("bsho,bhts->btho", v, attention)
        return output.reshape(batch_size, seq_len, self.num_heads * self.d_out)


class FeedForward(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlockV2(nn.Module):
    """Pre-LayerNorm residual block from the notebook architecture."""

    def __init__(
        self,
        embedding_dim: int,
        input_size: int,
        num_heads: int,
        ffwd_size: int = 4,
        *,
        muP: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        head_dim = embedding_dim // num_heads
        self.attn = CausalMultiHeadAttention(
            d_in=embedding_dim,
            d_out=head_dim,
            d_k=head_dim,
            num_heads=num_heads,
            muP=muP,
            dropout=dropout,
            max_seq_len=input_size,
        )
        self.ffwd = FeedForward(
            embedding_dim,
            int(ffwd_size) * embedding_dim,
            embedding_dim,
            dropout=dropout,
        )
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ffwd(self.ln2(x)))
        return x


class CLMv2(nn.Module):
    """Alternative causal language model selectable as ``--model transformer_v2``."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        embedding_dim: int,
        num_heads: int,
        ffwd_size: int,
        num_layers: int,
        dropout: float,
        *,
        muP: bool = True,
        embedding_scale: float = 0.05,
        share_emb: bool = False,
    ):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if share_emb:
            raise ValueError("transformer_v2 follows the notebook and does not share input/output weights")

        self.vocab_size = int(vocab_size)
        self.block_size = int(block_size)
        self.embedding_dim = int(embedding_dim)
        self.num_heads = int(num_heads)
        self.ffwd_size = int(ffwd_size)
        self.num_layers = int(num_layers)
        self.muP = bool(muP)
        self.embedding_scale = float(embedding_scale)
        self.rope = False

        self.token_embedding_table = OneHotLinearEmbedding(self.vocab_size, self.embedding_dim)
        self.position_embedding_table = nn.Embedding(self.block_size, self.embedding_dim)
        self.blocks = nn.Sequential(
            *[
                TransformerBlockV2(
                    embedding_dim=self.embedding_dim,
                    input_size=self.block_size,
                    num_heads=self.num_heads,
                    ffwd_size=self.ffwd_size,
                    muP=self.muP,
                    dropout=dropout,
                )
                for _ in range(self.num_layers)
            ]
        )

        # The notebook has no final normalisation.  Identity preserves that
        # architecture while retaining the repository's common model interface.
        self.ln_f = nn.Identity()
        self.lm_head = nn.Linear(self.embedding_dim, self.vocab_size)

        with torch.no_grad():
            self.lm_head.weight.zero_()
            if self.muP:
                self.apply_muP_initialization()

    @torch.no_grad()
    def apply_muP_initialization(self) -> None:
        d = float(self.embedding_dim)
        self.token_embedding_table.weight.normal_(0.0, self.embedding_scale)
        self.position_embedding_table.weight.normal_(0.0, self.embedding_scale)

        for block in self.blocks:
            block.attn.WK.normal_(0.0, 1.0 / math.sqrt(d))
            block.attn.WQ.normal_(0.0, 1.0 / math.sqrt(d))
            block.attn.WV.normal_(0.0, 1.0 / math.sqrt(d))
            block.ffwd.net[0].weight.normal_(0.0, 1.0 / math.sqrt(d))
            block.ffwd.net[3].weight.normal_(0.0, 0.5 / math.sqrt(d))

    def forward(self, idx: torch.Tensor, targets=None) -> torch.Tensor:
        del targets
        if idx.ndim != 2:
            raise ValueError(f"Expected token ids with shape [B,T], got {tuple(idx.shape)}")
        _, seq_len = idx.shape
        if seq_len > self.block_size:
            raise ValueError(
                f"Sequence length {seq_len} exceeds block_size={self.block_size}."
            )

        positions = torch.arange(seq_len, device=idx.device)
        x = self.token_embedding_table(idx) + self.position_embedding_table(positions)[None, :, :]
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    @torch.no_grad()
    def compute_model_log_norm(self):
        return _bartlett_log_complexity(self.named_parameters(), exclude_qk=False)

    @torch.no_grad()
    def compute_model_norm(self):
        return torch.exp(self.compute_model_log_norm())

    @torch.no_grad()
    def compute_model_log_norm_no_qk(self):
        return _bartlett_log_complexity(self.named_parameters(), exclude_qk=True)

    @torch.no_grad()
    def compute_model_norm_no_qk(self):
        return torch.exp(self.compute_model_log_norm_no_qk())

    @torch.no_grad()
    def compute_l2_log_norm(self):
        return _l2_log_norm(self.parameters())

    @torch.no_grad()
    def compute_l2_norm(self):
        return torch.exp(self.compute_l2_log_norm())

    def configure_optimizers(self, lr: float, wd: float) -> torch.optim.Optimizer:
        """Use the parameter grouping from the notebook implementation."""
        decay = []
        no_decay = []
        output_head = []
        embeddings = []

        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("lm_head."):
                output_head.append(parameter)
            elif name.startswith("token_embedding_table.") or name.startswith(
                "position_embedding_table."
            ):
                embeddings.append(parameter)
            elif parameter.ndim >= 2:
                decay.append(parameter)
            else:
                no_decay.append(parameter)

        groups = [
            {"params": decay, "weight_decay": wd, "lr": lr},
            {"params": no_decay, "weight_decay": 0.0, "lr": lr},
            {"params": output_head, "weight_decay": wd, "lr": lr},
            {
                "params": embeddings,
                "weight_decay": 0.0,
                "lr": lr * math.sqrt(float(self.embedding_dim)),
            },
        ]
        return torch.optim.AdamW(groups, lr=lr)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, num_tokens: int) -> torch.Tensor:
        for _ in range(int(num_tokens)):
            idx_cond = idx[:, -self.block_size :]
            logits = self(idx_cond)[:, -1, :]
            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx


# A short alias for callers that prefer the notebook's generic GPT name.
GPT = CLMv2
