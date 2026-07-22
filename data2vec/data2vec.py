import contextlib
import copy
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_encoder import Data2VecTransformerEncoder


def amp_context(model):
    """autocast(bfloat16) if the model was flagged for amp, else no-op."""
    if getattr(model, "_amp_enabled", False):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def build_data2vec_file_name_suffix(args) -> str:
    suffix = (
        f"L{args.num_layers}_s{args.tuple_size}_v{args.vocab_size}_m{args.num_synonyms}"
        f"_P{args.train_size}_nH{args.n_heads}_nE{args.d_model}_mp{args.mask_prob}"
        f"_ema{args.ema_decay}"
    )
    if getattr(args, "online", False):
        suffix += "_online"
    return suffix


def build_data2vec_model(args):
    average_top_k_layers = getattr(args, "average_top_k_layers", None)
    if average_top_k_layers is None:
        average_top_k_layers = min(args.n_layers, args.num_layers)

    ema_anneal_end_step = getattr(args, "ema_anneal_end_step", None)
    if ema_anneal_end_step is None:
        ema_anneal_end_step = 100_000

    return Data2Vec(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=getattr(args, "d_ff", args.d_model * 4),
        dropout=getattr(args, "dropout", 0.1),
        max_seq_len=args.tuple_size ** args.num_layers,
        average_top_k_layers=average_top_k_layers,
        loss_beta=getattr(args, "loss_beta", 4.0),
        mask_prob=args.mask_prob,
        mask_length=getattr(args, "mask_length", 1),
        ema_decay=args.ema_decay,
        ema_end_decay=getattr(args, "ema_end_decay", 0.9999),
        ema_anneal_end_step=ema_anneal_end_step,
        layer_norm_target_layer=getattr(args, "layer_norm_target_layer", True),
        layer_norm_targets=getattr(args, "layer_norm_targets", False),
        head_layers=getattr(args, "head_layers", 1),
    )


class Data2Vec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        max_seq_len: int,
        average_top_k_layers: int,
        loss_beta: float,
        mask_prob: float,
        mask_length: int,
        ema_decay: float,
        ema_end_decay: float,
        ema_anneal_end_step: int,
        layer_norm_target_layer: bool,
        layer_norm_targets: bool,
        head_layers: int,
    ):
        super().__init__()
        self.base_vocab_size = vocab_size
        self.padding_idx = 0
        self.mask_idx = vocab_size + 1
        self.vocab_size = vocab_size + 2
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.average_top_k_layers = average_top_k_layers
        self.loss_beta = loss_beta
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.ema_decay = ema_decay
        self.ema_end_decay = ema_end_decay
        self.ema_anneal_end_step = ema_anneal_end_step
        self.layer_norm_target_layer = layer_norm_target_layer
        self.layer_norm_targets = layer_norm_targets
        self._step = 0
        self._ema_decay_current = ema_decay

        self.encoder = Data2VecTransformerEncoder(
            vocab_size=self.vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=max_seq_len,
            padding_idx=self.padding_idx,
            mask_idx=self.mask_idx,
        )
        self.teacher = copy.deepcopy(self.encoder).float()
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        # Share the embedding frontend weights with the student, as in fairseq.
        # The teacher should not own the student's dropout module; the EMA teacher
        # runs in eval mode and target extraction already bypasses teacher._embed().
        self.teacher.embed_tokens = self.encoder.embed_tokens
        self.teacher.embed_positions = self.encoder.embed_positions
        self.teacher.emb_layer_norm = self.encoder.emb_layer_norm
        self.teacher.dropout = nn.Identity()

        head = []
        for _ in range(max(0, head_layers - 1)):
            head.extend(
                [
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                ]
            )
        head.append(nn.Linear(d_model, d_model))
        self.regression_head = nn.Sequential(*head)
        self._last_loss_breakdown: Dict[str, float] = {}

    def _to_input_ids(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.long:
            return x
        if x.ndim != 3:
            raise TypeError("Expected one-hot input of shape (batch, vocab, seq_len)")
        # Dataset one-hot uses class ids in [0, base_vocab_size-1]; shift by +1
        # so 0 remains reserved as padding, matching the fairseq-style setup.
        input_ids = x.argmax(dim=1).long() + 1
        return input_ids

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    def get_last_loss_breakdown(self) -> Dict[str, float]:
        return dict(self._last_loss_breakdown)

    def _get_ema_decay(self) -> float:
        if self._step >= self.ema_anneal_end_step:
            return self.ema_end_decay
        pct = self._step / max(1, self.ema_anneal_end_step)
        return self.ema_decay + (self.ema_end_decay - self.ema_decay) * pct

    @torch.no_grad()
    def update_teacher(self) -> None:
        decay = self._get_ema_decay()
        self._ema_decay_current = decay
        self._last_loss_breakdown["ema_decay"] = decay

        if decay >= 1.0:
            # Frozen teacher: skip all parameter updates (incl. shared embeddings).
            self._step += 1
            return

        student_state = self.encoder.state_dict()
        teacher_state = self.teacher.state_dict()
        shared_prefixes = (
            "embed_tokens.",
            "embed_positions.",
            "emb_layer_norm.",
        )

        for key, teacher_value in teacher_state.items():
            if key.startswith(shared_prefixes):
                teacher_value.copy_(student_state[key])
                continue

            student_value = student_state[key]
            if teacher_value.dtype.is_floating_point:
                teacher_value.mul_(decay)
                teacher_value.add_(student_value.detach().float(), alpha=1.0 - decay)
            else:
                teacher_value.copy_(student_value)

        self._step += 1

    def _sample_mask_starts(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        start_prob = min(1.0, self.mask_prob / max(1, self.mask_length))
        return torch.rand(batch_size, seq_len, device=device) < start_prob

    def generate_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        non_padding = input_ids.ne(self.padding_idx)

        if self.mask_length == 1:
            mask = torch.rand(batch_size, seq_len, device=device) < self.mask_prob
        else:
            starts = self._sample_mask_starts(batch_size, seq_len, device)
            mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
            for offset in range(self.mask_length):
                shifted = torch.zeros_like(mask)
                if offset == 0:
                    shifted |= starts
                else:
                    shifted[:, offset:] = starts[:, :-offset]
                mask |= shifted

        mask &= non_padding
        empty_rows = non_padding.any(dim=1) & ~mask.any(dim=1)
        if empty_rows.any():
            rows = torch.where(empty_rows)[0]
            for row in rows.tolist():
                valid_positions = torch.where(non_padding[row])[0]
                random_pos = valid_positions[torch.randint(0, valid_positions.numel(), (1,), device=device)]
                mask[row, random_pos] = True
        return mask

    def _teacher_targets(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, layer_results = self.teacher(
                input_ids=input_ids,
                return_all_layers=True,
            )
            top_k = layer_results[-self.average_top_k_layers :]
            target_layers = [result.ffn_output for result in top_k]
            if self.layer_norm_target_layer:
                target_layers = [
                    F.layer_norm(layer.float(), layer.shape[-1:])
                    for layer in target_layers
                ]
            targets = torch.stack(target_layers, dim=0).mean(dim=0)
            if self.layer_norm_targets:
                targets = F.layer_norm(targets.float(), targets.shape[-1:])
            return targets

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        input_ids = self._to_input_ids(input_ids)

        if mask is None:
            mask = self.generate_mask(input_ids)

        teacher_targets = self._teacher_targets(input_ids)
        student_input_ids = input_ids.clone()
        student_input_ids[mask] = self.mask_idx

        student_hidden = self.encoder(student_input_ids)
        student_pred = self.regression_head(student_hidden[mask])
        target = teacher_targets[mask]

        if self.loss_beta == 0:
            loss_values = F.mse_loss(
                student_pred.float(),
                target.float(),
                reduction="none",
            ).sum(dim=-1)
        else:
            loss_values = F.smooth_l1_loss(
                student_pred.float(),
                target.float(),
                reduction="none",
                beta=self.loss_beta,
            ).sum(dim=-1)

        loss = loss_values.mean() * (1.0 / math.sqrt(student_pred.size(-1)))
        self._last_loss_breakdown = {
            "regression": loss.detach().item(),
            "ema_decay": self._ema_decay_current,
        }
        return loss, student_hidden, teacher_targets

    @torch.no_grad()
    def get_representations(
        self,
        input_ids: torch.Tensor,
        return_all_layers: bool = False,
    ):
        self.eval()
        input_ids = self._to_input_ids(input_ids)
        if return_all_layers:
            _, layer_results = self.encoder(input_ids=input_ids, return_all_layers=True)
            return [layer.hidden for layer in layer_results]
        return self.encoder(input_ids)

    @torch.no_grad()
    def get_sequence_representation(self, input_ids: torch.Tensor, pooling: str = "mean") -> torch.Tensor:
        hidden = self.get_representations(input_ids)
        if pooling == "mean":
            return hidden.mean(dim=1)
        if pooling == "max":
            return hidden.max(dim=1).values
        if pooling == "first":
            return hidden[:, 0]
        raise ValueError(f"Unknown pooling: {pooling}")
