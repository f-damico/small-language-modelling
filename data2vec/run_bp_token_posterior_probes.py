"""Run BP posterior baselines and online linear probes from the command line.

This script systematizes the notebook workflow:

1. build the RHM rules and compute exact BP-vs-filtered BP posterior metrics;
2. load a data2vec checkpoint;
3. train online linear probes against exact or filtered BP token posteriors.

Probe data is always sampled online when `--probe_online` is true, so
`--probe_steps 100000 --probe_batch_size 256` means 100k fresh probe batches.
"""

import argparse
import json
import os
import re
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Union

import torch

from bp_token_posterior_probe import (
    compare_filtered_bp_to_exact,
    train_token_posterior_probe,
)
from data2vec import build_data2vec_model
from random_hierarchy_model import RandomHierarchyModel


Depth = Optional[int]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute exact BP-vs-filtered BP metrics and train online linear "
            "masked-token posterior probes."
        )
    )

    # Model / RHM config. These must match the checkpoint.
    parser.add_argument("--vocab_size", type=int, default=16)
    parser.add_argument("--num_synonyms", type=int, default=5)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--tuple_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1, help="RHM rule seed.")
    parser.add_argument("--d_model", type=int, default=2048)
    parser.add_argument("--d_ff", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=32)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mask_prob", type=float, default=0.15)
    parser.add_argument("--mask_length", type=int, default=1)
    parser.add_argument("--average_top_k_layers", type=int, default=None)
    parser.add_argument("--loss_beta", type=float, default=4.0)
    parser.add_argument("--head_layers", type=int, default=1)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--ema_end_decay", type=float, default=None)
    parser.add_argument("--ema_anneal_end_step", type=int, default=100_000)
    parser.add_argument(
        "--layer_norm_target_layer",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--layer_norm_targets", action="store_true")
    parser.add_argument("--online", action=argparse.BooleanOptionalAction, default=True)

    # Checkpoint / output.
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="data2vec checkpoint to probe. If omitted, probes a random-init model.",
    )
    parser.add_argument("--output_dir", type=str, default="bp_probe_runs")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--bp_device", type=str, default="cpu")
    parser.add_argument("--bf16", action="store_true")

    # Masking / BP targets.
    parser.add_argument(
        "--fixed_positions",
        type=str,
        default=None,
        help='Comma-separated masked positions, e.g. "4,9". Use "none" for Bernoulli masks.',
    )
    parser.add_argument(
        "--bp_filter_mode",
        type=str,
        default="path_marginal",
        choices=["path_marginal", "uniform"],
        help="path_marginal matches arXiv:2408.15138v3; uniform is the older baseline.",
    )
    parser.add_argument(
        "--compare_depths",
        type=str,
        default="all",
        help='Depths for exact BP-vs-filtered BP metrics: "all", "filtered", or comma list.',
    )
    parser.add_argument("--compare_batch_size", type=int, default=512)
    parser.add_argument("--compare_batches", type=int, default=16)
    parser.add_argument("--compare_seed", type=int, default=31415)
    parser.add_argument(
        "--skip_bp_compare",
        action="store_true",
        help="Skip exact BP-vs-filtered BP metric computation.",
    )

    # Probe training.
    parser.add_argument(
        "--probe_depths",
        type=str,
        default="exact",
        help=(
            'Probe target depths: "exact", "all", "filtered", or comma list like '
            '"0,1,2,3,4". An integer d means known_leaf_depth=d.'
        ),
    )
    parser.add_argument("--probe_steps", type=int, default=100_000)
    parser.add_argument("--probe_batch_size", type=int, default=256)
    parser.add_argument("--probe_lr", type=float, default=1e-3)
    parser.add_argument("--probe_seed", type=int, default=2024)
    parser.add_argument("--eval_every", type=int, default=1_000)
    parser.add_argument("--eval_batches", type=int, default=8)
    parser.add_argument("--probe_layer", type=str, default="final")
    parser.add_argument(
        "--separate_positions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use one independent linear head per token position.",
    )
    parser.add_argument(
        "--probe_online",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample fresh RHM leaves/masks for every probe step and eval point.",
    )
    parser.add_argument(
        "--rule_check",
        action="store_true",
        help=(
            "During probe eval, sample masked tokens from the probe distribution "
            "and check RHM rule consistency with BP_countcorrect_upward."
        ),
    )
    parser.add_argument(
        "--rule_check_samples",
        type=int,
        default=1,
        help="Number of probe samples per eval batch for rule consistency checks.",
    )
    parser.add_argument(
        "--rule_check_position",
        type=int,
        default=None,
        help=(
            "Optional masked position used for path-specific rule checks. If omitted "
            "and exactly one fixed position is used, that position is inferred."
        ),
    )
    parser.add_argument(
        "--rule_check_seed",
        type=int,
        default=None,
        help="Optional base seed for probe-sampling rule checks.",
    )
    return parser


def parse_fixed_positions(spec: Optional[str]) -> Optional[List[int]]:
    if spec is None or spec.strip().lower() in {"", "none", "null"}:
        return None
    return [int(part) for part in spec.split(",") if part.strip()]


def parse_layer(spec: str) -> Union[str, int]:
    return "final" if spec == "final" else int(spec)


def parse_depths(spec: str, num_layers: int, *, allow_exact: bool) -> List[Depth]:
    text = spec.strip().lower()
    if allow_exact and text == "exact":
        return [None]
    if text == "all":
        return list(range(num_layers + 1))
    if text == "filtered":
        return list(range(num_layers))
    if text in {"", "none", "null"}:
        return []

    depths: List[Depth] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if allow_exact and part == "exact":
            depths.append(None)
            continue
        depth = int(part)
        if depth < 0 or depth > num_layers:
            raise ValueError(f"Depth {depth} outside [0, {num_layers}]")
        depths.append(depth)
    return depths


def depth_label(depth: Depth) -> str:
    return "exact" if depth is None else f"depth{depth}"


def sanitize_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "run"


def checkpoint_stem(path: Optional[str]) -> str:
    if path is None:
        return "random_init"
    return os.path.splitext(os.path.basename(path))[0]


def state_dict_to_cpu(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_model_args(args: argparse.Namespace) -> SimpleNamespace:
    d_ff = args.d_ff if args.d_ff is not None else args.d_model * 4
    ema_end_decay = args.ema_end_decay if args.ema_end_decay is not None else args.ema_decay
    return SimpleNamespace(
        vocab_size=args.vocab_size,
        num_synonyms=args.num_synonyms,
        num_layers=args.num_layers,
        tuple_size=args.tuple_size,
        d_model=args.d_model,
        d_ff=d_ff,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        mask_prob=args.mask_prob,
        mask_length=args.mask_length,
        average_top_k_layers=args.average_top_k_layers,
        loss_beta=args.loss_beta,
        head_layers=args.head_layers,
        ema_decay=args.ema_decay,
        ema_end_decay=ema_end_decay,
        ema_anneal_end_step=args.ema_anneal_end_step,
        layer_norm_target_layer=args.layer_norm_target_layer,
        layer_norm_targets=args.layer_norm_targets,
        online=args.online,
        train_size=None,
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    fixed_positions = parse_fixed_positions(args.fixed_positions)
    probe_layer = parse_layer(args.probe_layer)
    compare_depths = parse_depths(args.compare_depths, args.num_layers, allow_exact=False)
    probe_depths = parse_depths(args.probe_depths, args.num_layers, allow_exact=True)

    run_name = args.run_name or checkpoint_stem(args.checkpoint_path)
    run_name = sanitize_name(run_name)
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    model_args = build_model_args(args)
    config = {
        "args": vars(args),
        "model_args": vars(model_args),
        "device": device,
        "fixed_positions": fixed_positions,
        "compare_depths": compare_depths,
        "probe_depths": [depth_label(depth) for depth in probe_depths],
    }
    write_json(os.path.join(output_dir, "config.json"), config)

    dataset = RandomHierarchyModel(
        num_features=args.vocab_size,
        num_synonyms=args.num_synonyms,
        num_layers=args.num_layers,
        num_classes=args.vocab_size,
        tuple_size=args.tuple_size,
        seed_rules=args.seed,
        train_size=1,
        test_size=0,
        seed_sample=42,
        input_format="long",
        replacement=1,
    )
    rules = dataset.rules

    if not args.skip_bp_compare:
        print("Computing exact BP-vs-filtered BP metrics", flush=True)
        bp_metrics = compare_filtered_bp_to_exact(
            rules,
            depths=compare_depths,
            batch_size=args.compare_batch_size,
            num_batches=args.compare_batches,
            mask_prob=args.mask_prob,
            fixed_positions=fixed_positions,
            bp_device=args.bp_device,
            bp_filter_mode=args.bp_filter_mode,
            seed=args.compare_seed,
        )
        for row in bp_metrics:
            print(
                f"depth={int(row['known_leaf_depth'])}: "
                f"ce={row['cross_entropy_exact_to_filtered']:.4f} "
                f"kl={row['kl_exact_to_filtered']:.4f} "
                f"js={row['js_divergence']:.4f} "
                f"top1={row['top1_agreement']:.4f} "
                f"H_exact={row['mean_exact_entropy']:.4f} "
                f"H_filtered={row['mean_filtered_entropy']:.4f}",
                flush=True,
            )
        write_json(os.path.join(output_dir, "bp_filter_metrics.json"), {"rows": bp_metrics})

    print("Building model", flush=True)
    model = build_data2vec_model(model_args).to(device).eval()
    model._amp_enabled = bool(args.bf16 and device.startswith("cuda"))
    if args.checkpoint_path is not None:
        print(f"Loading checkpoint: {args.checkpoint_path}", flush=True)
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    else:
        print("No checkpoint_path provided; probing random-init model", flush=True)

    for idx, depth in enumerate(probe_depths):
        label = depth_label(depth)
        print("=" * 80, flush=True)
        print(
            f"Training probe target={label} "
            f"steps={args.probe_steps} batch={args.probe_batch_size} "
            f"online={args.probe_online}",
            flush=True,
        )
        probe_seed = args.probe_seed + idx * 10_000
        probe, history = train_token_posterior_probe(
            model,
            rules,
            train_steps=args.probe_steps,
            batch_size=args.probe_batch_size,
            lr=args.probe_lr,
            mask_prob=args.mask_prob,
            fixed_positions=fixed_positions,
            layer=probe_layer,
            model_device=device,
            probe_device=device,
            bp_device=args.bp_device,
            seed=probe_seed,
            eval_every=args.eval_every,
            eval_batches=args.eval_batches,
            separate_positions=args.separate_positions,
            bp_known_leaf_depth=depth,
            bp_filter_mode=args.bp_filter_mode,
            online_batches=args.probe_online,
            rule_check=args.rule_check,
            rule_check_samples=args.rule_check_samples,
            rule_check_position=args.rule_check_position,
            rule_check_seed=args.rule_check_seed,
        )

        payload = {
            "probe_state_dict": state_dict_to_cpu(probe),
            "history": history,
            "target_depth": depth,
            "target_label": label,
            "bp_filter_mode": args.bp_filter_mode,
            "checkpoint_path": args.checkpoint_path,
            "config": config,
        }
        pt_path = os.path.join(output_dir, f"probe_{label}.pt")
        json_path = os.path.join(output_dir, f"probe_{label}_history.json")
        torch.save(payload, pt_path)
        write_json(
            json_path,
            {
                "target_depth": depth,
                "target_label": label,
                "bp_filter_mode": args.bp_filter_mode,
                "history": history,
            },
        )
        print(f"Saved {pt_path}", flush=True)
        print(f"Saved {json_path}", flush=True)
        del probe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
