"""
Mamba language model.

The SSM block is sourced from the official mamba-ssm package when available,
falling back to the pure-PyTorch reference in mamba_scratch.py otherwise.
This means the file runs on CPU / macOS (fallback) and gets the Triton-accelerated
kernels automatically on a CUDA machine with mamba-ssm installed.

Installation (Linux + CUDA only):
    pip install mamba-ssm causal-conv1d
See INSTALL.md at the bottom of this file for the full guide.
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def _matrix_spectral_and_21_norm(parameter: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact operator norm and ||W^T||_{2,1} in float64.

    Convolution kernels and higher-rank tensors are viewed as matrices with the
    first dimension as output channels and all remaining dimensions flattened.
    """
    matrix = parameter.detach().to(dtype=torch.float64).reshape(parameter.shape[0], -1)
    spectral = torch.linalg.matrix_norm(matrix, ord=2)
    # ||W^T||_{2,1}: sum of Euclidean norms of the columns of W.
    transpose_21 = torch.linalg.vector_norm(matrix, ord=2, dim=0).sum()
    return spectral, transpose_21


@torch.no_grad()
def _bartlett_log_complexity(named_parameters, *, exclude_qk: bool = False) -> torch.Tensor:
    """Bartlett-style spectral complexity in the numerically stable log domain.

    N(W) = prod_i ||W_i||_2 * [sum_i (||W_i^T||_{2,1}/||W_i||_2)^(2/3)]^(3/2).
    Only matrix-like trainable parameters enter.  The no-QK variant removes the
    Transformer query and key matrices, matching the previous RHM repository.
    """
    log_product = None
    correction = None
    eps = torch.finfo(torch.float64).tiny
    for name, parameter in named_parameters:
        if not parameter.requires_grad or parameter.ndim < 2:
            continue
        lname = name.lower()
        if exclude_qk and (
            lname.endswith(".query")
            or lname.endswith(".key")
            or ".query." in lname
            or ".key." in lname
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
        # Keep the return device aligned with the model when no matrix exists.
        first = next((p for _, p in named_parameters), None)
        device = first.device if first is not None else torch.device("cpu")
        return torch.tensor(float("-inf"), dtype=torch.float64, device=device)
    return log_product + 1.5 * torch.log(torch.clamp(correction, min=eps))


@torch.no_grad()
def _l2_log_norm(parameters) -> torch.Tensor:
    total = None
    for parameter in parameters:
        if not parameter.requires_grad:
            continue
        value = torch.sum(parameter.detach().to(torch.float64) ** 2)
        total = value if total is None else total + value
    if total is None:
        return torch.tensor(float("-inf"), dtype=torch.float64)
    return 0.5 * torch.log(torch.clamp(total, min=torch.finfo(torch.float64).tiny))


# ── SSM block ────────────────────────────────────────────────────────────────
# Use the official Mamba block (Triton kernels, step(), inference cache) when
# mamba-ssm is installed; fall back to the pure-PyTorch version otherwise.

try:
    from mamba_ssm import Mamba as MambaBlock          # official CUDA/Triton block
    _MAMBA_SSM_AVAILABLE = True
except ImportError:
    from .mamba_scratch import MambaBlock               # pure-PyTorch fallback
    _MAMBA_SSM_AVAILABLE = False


# ── RMSNorm ───────────────────────────────────────────────────────────────────
# Use the Triton-fused kernel when available, otherwise plain PyTorch.

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm
except ImportError:
    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-8, **kwargs):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(dim))
            self.eps   = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            rms = x.norm(dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
            return self.scale * x / (rms + self.eps)


# ── Weight initialisation (mirrors the official mixer_seq_simple.py) ─────────

def _init_weights(
    module: nn.Module,
    n_layer: int,
    initializer_range: float = 0.02,
    rescale_prenorm_residual: bool = True,
    n_residuals_per_layer: int = 1,
) -> None:
    """
    GPT-2 / Mamba-style weight initialisation.

    Key differences from a naive std=0.02 blanket init:
      1. dt_proj.bias is protected by a _no_reinit flag set inside MambaBlock —
         that bias was carefully initialised to inv_softplus(uniform(dt_min, dt_max))
         and must not be zeroed out.
      2. out_proj.weight is depth-scaled by 1/sqrt(n_layer) to prevent the
         residual stream variance from growing with depth.
    """
    if isinstance(module, nn.Linear):
        # Zero all Linear biases, but skip any marked _no_reinit
        # (dt_proj.bias in the official Mamba block carries this flag).
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Scale down the output projection of each residual block so that the
        # residual-stream variance at initialisation is O(1) regardless of depth.
        for name, p in module.named_parameters():
            if name == "out_proj.weight":
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


# ── Residual block ────────────────────────────────────────────────────────────

class MambaResidualBlock(nn.Module):
    """
    Pre-RMSNorm + Mamba SSM + residual connection.

    When mamba-ssm is installed the inner block is the official Mamba class
    (with Triton kernels and step() for fast inference).
    When it is not, MambaBlock from mamba_scratch.py is used instead.

    Args:
        d_model:    residual-stream width
        d_state:    SSM state dimension N (default 16)
        d_conv:     depthwise-conv kernel size (default 4)
        expand:     inner-dim expansion factor (default 2)
        dropout:    dropout applied after the SSM output (before residual add)
        layer_idx:  layer index forwarded to the official Mamba block for
                    inference-cache bookkeeping (ignored by the fallback)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.norm = RMSNorm(d_model)

        # Build the SSM block with the right kwargs for each backend
        if _MAMBA_SSM_AVAILABLE:
            # Official block: layer_idx is needed for inference-cache lookups;
            # it does not accept a dropout kwarg.
            self.mamba = MambaBlock(
                d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                layer_idx=layer_idx,
            )
        else:
            # Pure-PyTorch fallback: no layer_idx, but accepts dropout.
            self.mamba = MambaBlock(
                d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.mamba(self.norm(x)))


# ── Language model ────────────────────────────────────────────────────────────

class MambaLM(nn.Module):
    """
    Mamba causal language model.

    Drop-in replacement for CLM: forward(idx) → (B, T, vocab_size).

    Args:
        vocab_size:  vocabulary size
        d_model:     residual-stream width (= d_embedding in the CLM interface)
        depth:       number of Mamba residual blocks
        d_state:     SSM state dimension N (default 16)
        d_conv:      depthwise-conv kernel size (default 4)
        expand:      inner-dim expansion factor (default 2)
        dropout:     dropout rate
        share_emb:   tie token embedding ↔ LM-head weights
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        depth: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        share_emb: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.depth   = depth

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            MambaResidualBlock(
                d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
                layer_idx=i,
            )
            for i in range(depth)
        ])
        self.norm_f  = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if share_emb:
            self.lm_head.weight = self.embedding.weight

        # Apply weight init; pass depth so out_proj.weight is depth-scaled.
        self.apply(partial(_init_weights, n_layer=depth))


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


    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        idx: (B, L) long tensor of token ids
        returns: (B, L, vocab_size) logits
        """
        x = self.embedding(idx)       # (B, L, d_model)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)            # (B, L, d_model)
        return self.lm_head(x)        # (B, L, vocab_size)

    def generate(self, idx: torch.Tensor, num_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        """
        Autoregressively sample num_tokens new tokens.

        idx:         (B, T) prompt token ids
        num_tokens:  how many tokens to generate
        temperature: softmax temperature (lower = more peaked)
        returns:     (B, T + num_tokens) token ids
        """
        for _ in range(num_tokens):
            logits = self(idx)                          # (B, T, vocab_size)
            logits = logits[:, -1, :] / temperature    # (B, vocab_size)
            probs  = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

    def backend(self) -> str:
        """Returns which SSM backend is active."""
        return "mamba-ssm (Triton)" if _MAMBA_SSM_AVAILABLE else "mamba_scratch (pure PyTorch)"
