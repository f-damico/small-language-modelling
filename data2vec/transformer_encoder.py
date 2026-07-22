import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class LayerResult:
    hidden: torch.Tensor
    ffn_output: torch.Tensor


class Data2VecTransformerLayer(nn.Module):
    """Transformer block exposing the FFN output before the final residual add."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.final_layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = x
        x_norm = self.self_attn_layer_norm(x)
        attn_out, _ = self.self_attn(
            x_norm,
            x_norm,
            x_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = residual + self.dropout(attn_out)

        residual = x
        x_norm = self.final_layer_norm(x)
        ffn_output = self.fc2(self.dropout(self.activation(self.fc1(x_norm))))
        x = residual + self.dropout(ffn_output)
        return x, ffn_output


class Data2VecTransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        max_seq_len: int,
        padding_idx: int,
        mask_idx: int,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.padding_idx = padding_idx
        self.mask_idx = mask_idx

        self.embed_tokens = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.embed_positions = nn.Embedding(max_seq_len, d_model)
        self.emb_layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                Data2VecTransformerLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=self.d_model ** -0.5)
        nn.init.normal_(self.embed_positions.weight, mean=0.0, std=self.d_model ** -0.5)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.embed_tokens.weight[self.padding_idx].zero_()

    def _embed(
        self,
        input_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        positions = positions.expand(batch_size, -1)
        x = self.embed_tokens(input_ids) + self.embed_positions(positions)
        x = self.emb_layer_norm(x)
        x = self.dropout(x)
        key_padding_mask = input_ids.eq(self.padding_idx)
        if not key_padding_mask.any():
            key_padding_mask = None
        return x, key_padding_mask

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_all_layers: bool = False,
    ):
        if features is None:
            if input_ids is None:
                raise ValueError("Either input_ids or features must be provided")
            x, key_padding_mask = self._embed(input_ids)
        else:
            x = features

        layer_results: List[LayerResult] = []
        for layer in self.layers:
            x, ffn_output = layer(x, key_padding_mask=key_padding_mask)
            layer_results.append(LayerResult(hidden=x, ffn_output=ffn_output))

        x = self.final_norm(x)
        if return_all_layers:
            return x, layer_results
        return x

