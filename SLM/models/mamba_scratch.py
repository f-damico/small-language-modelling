"""
Pure-PyTorch implementation of the Mamba SSM block.

Exported: MambaBlock  — used as a fallback by mamba.py when mamba-ssm is not installed.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _parallel_scan(dA: torch.Tensor, dBu: torch.Tensor) -> torch.Tensor:
    """
    Hillis-Steele parallel prefix scan solving h_t = dA_t * h_{t-1} + dBu_t
    with h_{-1} = 0, in O(log L) parallel steps.

    Replaces the O(L) Python for-loop that launches L sequential CUDA kernels
    (one per timestep), which is the dominant bottleneck on GPU at typical
    sequence lengths (L = 128–2048).

    Each step doubles the stride, composing adjacent (A, B) pairs via the
    associative rule  (A2, B2) ∘ (A1, B1) = (A2*A1, A2*B1 + B2).
    After ceil(log2 L) steps every position holds the cumulative h_t.

    dA:  (B, L, d_inner, d_state)   — discretised A coefficients (Ā_t)
    dBu: (B, L, d_inner, d_state)   — discretised B * u terms (B̄_t u_t)
    returns h: (B, L, d_inner, d_state)
    """
    L = dA.shape[1]
    a, b = dA, dBu
    stride = 1
    while stride < L:
        # Positions t >= stride get composed with position t - stride.
        new_a = a[:, stride:] * a[:, :-stride]
        new_b = a[:, stride:] * b[:, :-stride] + b[:, stride:]
        a = torch.cat([a[:, :stride], new_a], dim=1)
        b = torch.cat([b[:, :stride], new_b], dim=1)
        stride <<= 1
    return b  # h_t for every t


class MambaBlock(nn.Module):
    """
    Selective State Space block (Mamba).

    Args:
        d_model:  residual-stream width
        d_state:  SSM state dimension (N in the paper), default 16
        d_conv:   depthwise-conv kernel width, default 4
        expand:   inner-dim expansion factor (d_inner = expand * d_model), default 2
        dropout:  dropout rate applied before the output projection
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)

        # Project d_model → 2*d_inner (x-branch + z-gate)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # Causal depthwise conv1d on x-branch
        # padding = d_conv-1 on the left; we trim the right after convolution.
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            bias=True,
            padding=d_conv - 1,
        )

        # SSM projections
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialise dt_proj bias so that softplus(bias) spans [dt_min, dt_max]
        dt_min, dt_max, dt_floor = 0.001, 0.1, 1e-4
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))   # softplus inverse
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Prevent any outer _init_weights call from zeroing this bias.
        self.dt_proj.bias._no_reinit = True

        # A: stored as log for numerical stability; (d_inner, d_state)
        # Initialised to the HiPPO diagonal values 1, 2, …, d_state.
        A = torch.arange(1, d_state + 1, dtype=torch.float).unsqueeze(0).expand(
            self.d_inner, -1
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # D: per-channel skip-connection scalar
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, d_model)
        returns: (B, L, d_model)
        """
        B, L, _ = x.shape

        # Project and split into x-branch (SSM input) and z (gate)
        xz = self.in_proj(x)                         # (B, L, 2*d_inner)
        x_branch, z = xz.chunk(2, dim=-1)            # each: (B, L, d_inner)

        # Causal depthwise conv — trim right-padding to preserve causality
        x_branch = x_branch.transpose(1, 2)          # (B, d_inner, L)
        x_branch = self.conv1d(x_branch)[..., :L]    # (B, d_inner, L)
        x_branch = x_branch.transpose(1, 2)          # (B, L, d_inner)
        x_branch = F.silu(x_branch)

        # Selective SSM
        y = self._ssm(x_branch)                      # (B, L, d_inner)

        # SiLU gating
        y = y * F.silu(z)
        y = self.dropout(y)
        return self.out_proj(y)                       # (B, L, d_model)

    def _ssm(self, u: torch.Tensor) -> torch.Tensor:
        """
        Selective state-space scan (parallel prefix, O(log L) steps).

        u: (B, L, d_inner)
        returns: (B, L, d_inner)
        """
        B, L, d_inner = u.shape
        d_state = self.d_state

        # Fixed A (forced negative-definite for stability)
        A = -torch.exp(self.A_log.float())            # (d_inner, d_state)
        D = self.D.float()                            # (d_inner,)

        # Input-dependent Δ, B, C  (the "selective" mechanism)
        x_dbc = self.x_proj(u)                       # (B, L, dt_rank + 2*d_state)
        delta, B_in, C_in = x_dbc.split(
            [self.dt_rank, d_state, d_state], dim=-1
        )
        # delta: (B, L, dt_rank) → (B, L, d_inner) after projection + softplus
        delta = F.softplus(self.dt_proj(delta))       # (B, L, d_inner)

        # Zero-order-hold discretisation
        #   Ā   = exp(Δ ⊗ A)          shape: (B, L, d_inner, d_state)
        #   B̄u  = Δ ⊗ B_in ⊗ u       shape: (B, L, d_inner, d_state)
        dA  = torch.exp(delta.unsqueeze(-1) * A)     # (B, L, d_inner, d_state)
        dBu = (
            delta.unsqueeze(-1)
            * B_in.unsqueeze(2)
            * u.unsqueeze(-1)
        )                                             # (B, L, d_inner, d_state)

        # Parallel prefix scan: h_t = Ā_t h_{t-1} + B̄_t u_t  (O(log L) steps)
        h = _parallel_scan(dA, dBu)                  # (B, L, d_inner, d_state)

        # y_t = C_t h_t + D u_t
        y = (h * C_in.unsqueeze(2)).sum(-1)          # (B, L, d_inner)
        y = y + u * D                                 # D skip connection
        return y
