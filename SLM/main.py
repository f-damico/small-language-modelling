#!/usr/bin/env python3
"""Train GPT-2-style or Mamba small language models on text or RHM data."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import init
import train


def _git_info(repo_dir: Path):
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(repo_dir), "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training a small Transformer/Mamba language model on text or Random Hierarchy Model data"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ddp", action="store_true", help="enable DistributedDataParallel; launch with torchrun")
    parser.add_argument("--ddp_timeout_minutes", type=int, default=7200, help="DDP/NCCL collective timeout in minutes; keep large when rank 0 runs slow diagnostics")
    parser.add_argument("--local_rank", type=int, default=None, help="DDP local rank; normally set by torchrun through LOCAL_RANK")
    tf32_group = parser.add_mutually_exclusive_group()
    tf32_group.add_argument("--tf32", dest="tf32", action="store_true", help="allow TF32 matmul/cuDNN on CUDA GPUs")
    tf32_group.add_argument("--no_tf32", dest="tf32", action="store_false", help="disable TF32")
    parser.set_defaults(tf32=True)
    parser.add_argument("--dataset", type=str, default="rhm", help="rhm or the basename of a tokenized text corpus")
    parser.add_argument("--path", type=str, default="datasets/", help="text dataset/tokenizer directory")
    parser.add_argument("--tokenizer", type=str, default=None, help="tokenizer JSON for text datasets")

    # Common data sizes.
    parser.add_argument("--block_size", metavar="T", type=int, default=None)
    parser.add_argument("--batch_size", metavar="B", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=1024)
    parser.add_argument("--train_size", metavar="P", type=int, default=65536)
    parser.add_argument("--val_size", metavar="Val", type=int, default=32768)
    parser.add_argument("--eval_train_size", type=int, default=16384)
    parser.add_argument("--eval_val_size", type=int, default=16384)

    online_group = parser.add_mutually_exclusive_group()
    online_group.add_argument("--online", dest="online", action="store_true", help="fresh RHM samples at every batch")
    online_group.add_argument("--offline", dest="online", action="store_false", help="fixed finite RHM dataset")
    parser.set_defaults(online=True)

    # RHM parameters.  num_layers is deliberately distinct from model depth.
    parser.add_argument("--num_features", metavar="v", type=int, default=32)
    parser.add_argument("--num_classes", metavar="n", type=int, default=None)
    parser.add_argument("--num_synonyms", metavar="m", type=int, default=8)
    parser.add_argument("--tuple_size", metavar="s", type=int, default=2)
    parser.add_argument("--num_layers", metavar="L", type=int, default=3, help="RHM hierarchy depth")
    parser.add_argument("--num_tokens", type=int, default=None, help="derived as tuple_size**num_layers for RHM")
    parser.add_argument("--seed_rules", type=int, default=0)
    parser.add_argument("--seed_sample", type=int, default=0)
    parser.add_argument("--replacement", action="store_true", help="offline RHM sampling with replacement")
    parser.add_argument("--input_format", type=str, default="long", choices=["long", "onehot"])
    parser.add_argument("--whitening", type=int, default=0)

    # Architecture. The original GPT-2 model remains unchanged; transformer_v2 is optional.
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        choices=["gpt2", "transformer_v2", "mamba"],
    )
    parser.add_argument("--d_embedding", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6, help="number of Transformer/Mamba blocks")
    parser.add_argument("--seed_model", type=int, default=0)
    parser.add_argument("--n_heads", type=int, default=8, help="Transformer attention heads")
    parser.add_argument("--ffwd_size", type=int, default=4, help="Transformer MLP width multiplier")
    parser.add_argument("--rope", action="store_true")
    mup_group = parser.add_mutually_exclusive_group()
    mup_group.add_argument(
        "--mup",
        dest="mup",
        action="store_true",
        help="transformer_v2: use notebook muP-style initialisation",
    )
    mup_group.add_argument(
        "--no_mup",
        dest="mup",
        action="store_false",
        help="transformer_v2: disable notebook muP-style initialisation",
    )
    parser.set_defaults(mup=True)
    parser.add_argument(
        "--embedding_scale",
        type=float,
        default=0.05,
        help="transformer_v2 input/position embedding initialisation std",
    )
    parser.add_argument("--d_state", type=int, default=16)
    parser.add_argument("--d_conv", type=int, default=4)
    parser.add_argument("--mamba_expand", type=int, default=2)

    # Optimisation.
    parser.add_argument("--optim", type=str, default="adam", choices=["adam"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "none"])
    parser.add_argument("--warmup_time", type=int, default=100)
    parser.add_argument("--decay_time", type=int, default=100000)
    parser.add_argument("--decay_factor", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=None, help="explicit optimizer-step budget; overrides max_epochs")
    parser.add_argument("--loss_threshold", type=float, default=None)

    # Validation, diagnostics and saving.
    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument("--save_freq", type=int, default=1000, help="fallback linear validation cadence if num_validations=0")
    parser.add_argument("--num_validations", type=int, default=100, help="number of log-spaced validation/result saves")
    parser.add_argument("--num_weight_saves", type=int, default=100, help="number of log-spaced model-weight checkpoints")
    parser.add_argument("--measure_train", action="store_true")
    parser.add_argument("--loss_by_token", action="store_true", help="retained for compatibility; position losses are always saved")
    parser.add_argument("--checkpoints", action="store_true", help="retained for compatibility; use num_weight_saves")

    parser.add_argument(
        "--compute_rhm_diagnostics",
        "--compute_M_l",
        "--compute_rhm_margins",
        dest="compute_rhm_diagnostics",
        action="store_true",
        help="compute exact level-wise RHM margins, peeled losses and logit effective dimensions",
    )
    parser.add_argument("--rhm_margins_max_train_samples", "--M_l_max_train_samples", type=int, default=16384)
    parser.add_argument("--rhm_margins_max_val_samples", "--M_l_max_test_samples", type=int, default=16384)
    parser.add_argument("--rhm_margins_batch_size", "--M_l_batch_size", type=int, default=1024)

    save_group = parser.add_mutually_exclusive_group()
    save_group.add_argument("--save_run_data", dest="save_run_data", action="store_true")
    save_group.add_argument("--no_save_run_data", dest="save_run_data", action="store_false")
    parser.set_defaults(save_run_data=True)

    # Output layout.
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--run_name", type=str, default="small_lm_rhm")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--outname", type=str, default=None, help="legacy alias for output_dir")
    return parser.parse_args()



def _setup_distributed(config):
    """Initialise single-node/multi-node DDP when requested by torchrun."""
    env_world = int(os.environ.get("WORLD_SIZE", "1"))
    config.ddp = bool(getattr(config, "ddp", False) or env_world > 1)

    if not config.ddp:
        config.rank = 0
        config.local_rank = 0
        config.world_size = 1
        config.is_main_process = True
        if str(config.device).startswith("cuda") and torch.cuda.is_available():
            config.device = "cuda:0"
        return config

    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this PyTorch build")
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() and str(config.device).startswith("cuda") else "gloo"
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(minutes=int(getattr(config, "ddp_timeout_minutes", 7200))),
        )

    config.rank = dist.get_rank()
    config.world_size = dist.get_world_size()
    env_local = os.environ.get("LOCAL_RANK")
    if config.local_rank is None:
        config.local_rank = int(env_local) if env_local is not None else config.rank

    if str(config.device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(config.local_rank)
        config.device = f"cuda:{config.local_rank}"
    else:
        config.device = "cpu"
    config.is_main_process = config.rank == 0
    return config


def _cleanup_distributed(config):
    if bool(getattr(config, "ddp", False)) and dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _configure_cuda_math(config):
    if bool(getattr(config, "tf32", True)) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def _prepare_config(config):
    if config.num_classes is None:
        config.num_classes = config.num_features
    if config.dataset.lower() == "rhm":
        expected_tokens = int(config.tuple_size) ** int(config.num_layers)
        if config.num_tokens is not None and int(config.num_tokens) != expected_tokens:
            raise ValueError(f"num_tokens must equal tuple_size**num_layers={expected_tokens}")
        config.num_tokens = expected_tokens
        if config.block_size is None:
            config.block_size = expected_tokens - 1
    elif config.block_size is None:
        raise ValueError("--block_size is required for text datasets")

    if (
        config.model in {"gpt2", "transformer_v2"}
        and config.d_embedding % config.n_heads != 0
    ):
        raise ValueError("d_embedding must be divisible by n_heads for the Transformer")
    for name in (
        "eval_train_size",
        "eval_val_size",
        "rhm_margins_max_train_samples",
        "rhm_margins_max_val_samples",
    ):
        value = getattr(config, name)
        if value is not None and int(value) <= 0:
            setattr(config, name, None)

    if config.outname is not None and config.output_dir is None:
        config.output_dir = config.outname[:-3] if config.outname.endswith(".pt") else config.outname
    if config.output_dir is None:
        if config.dataset.lower() == "rhm":
            run_id = (
                f"{config.model}_P_{config.train_size}_seed_rules_{config.seed_rules}_"
                f"seed_sample_{config.seed_sample}_seed_model_{config.seed_model}"
            )
        else:
            run_id = f"{config.dataset}_{config.model}_P_{config.train_size}_seed_model_{config.seed_model}"
        config.output_dir = str(Path(config.data_root) / config.run_name / run_id)
    config.output_dir = str(Path(config.output_dir).expanduser())
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    config.git = _git_info(Path(__file__).resolve().parents[1])
    return config


def run(config):
    config = _setup_distributed(config)
    _configure_cuda_math(config)
    config = _prepare_config(config)
    random.seed(config.seed_sample)
    np.random.seed(config.seed_sample)
    torch.manual_seed(config.seed_model)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed_model)

    if config.is_main_process:
        print("=" * 80, flush=True)
        print(
            f"distributed={config.ddp} rank={config.rank}/{config.world_size} "
            f"local_rank={config.local_rank} device={config.device} tf32={config.tf32}",
            flush=True,
        )
        print(f"dataset={config.dataset} model={config.model} online={config.online}", flush=True)
        print(f"output_dir={config.output_dir}", flush=True)
        if config.dataset.lower() == "rhm":
            print(
                f"RHM v={config.num_features} n={config.num_classes} m={config.num_synonyms} "
                f"s={config.tuple_size} L={config.num_layers} d={config.num_tokens}",
                flush=True,
            )
        print(
            f"architecture depth={config.depth} d_embedding={config.d_embedding} "
            f"n_heads={config.n_heads if config.model in {'gpt2', 'transformer_v2'} else 'n/a'}",
            flush=True,
        )
        print(
            f"log saves: validations={config.num_validations}, weights={config.num_weight_saves}; "
            f"RHM diagnostics={config.compute_rhm_diagnostics}",
            flush=True,
        )
        print("=" * 80, flush=True)
    tokenizer, train_loader, val_loader, train_eval_loader, data_info = init.init_data(config)
    del tokenizer
    model = init.init_model(config)
    if bool(getattr(config, "ddp", False)):
        ddp_kwargs = {}
        if str(config.device).startswith("cuda"):
            ddp_kwargs.update(device_ids=[config.local_rank], output_device=config.local_rank)
        model = DDP(model, **ddp_kwargs)
    criterion, optimizer, scheduler = init.init_training(model, config)
    try:
        return train.train_model(
            model,
            train_loader,
            val_loader,
            train_eval_loader,
            criterion,
            optimizer,
            scheduler,
            config,
            data_info,
        )
    finally:
        _cleanup_distributed(config)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    run(parse_args())
