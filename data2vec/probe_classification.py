"""Post-hoc linear and MLP probes on data2vec features (mean-pooled class readout).

Mirrors `probe_latents.py` in spirit but targets the class label rather than
per-position latents. Uses a fresh OnlineRHMLoader for probe training so the
probe sample budget is decoupled from the encoder's training set size — useful
for offline runs where the encoder saw very few unique examples.
"""

import argparse
import glob
import os
import re

import torch

from data2vec import build_data2vec_file_name_suffix, build_data2vec_model
from random_hierarchy_model import OnlineRHMLoader, RandomHierarchyModel
from train_probe import run_probe_suite


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Post-hoc linear / MLP classification probes on data2vec checkpoints."
    )

    # Model / data config — must match the checkpoint.
    parser.add_argument("--vocab_size", type=int, default=16)
    parser.add_argument("--num_synonyms", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--tuple_size", type=int, default=2)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=16)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--d_ff", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mask_prob", type=float, default=0.15)
    parser.add_argument("--mask_length", type=int, default=1)
    parser.add_argument("--average_top_k_layers", type=int, default=None)
    parser.add_argument("--loss_beta", type=float, default=4.0)
    parser.add_argument("--head_layers", type=int, default=1)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--ema_end_decay", type=float, default=None)
    parser.add_argument("--ema_anneal_end_step", type=int, default=None)
    parser.add_argument(
        "--layer_norm_target_layer",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--layer_norm_targets", action="store_true")
    parser.add_argument("--train_size", type=int, default=8192)
    parser.add_argument("--online", action="store_true",
                        help="Kept for compat with build_data2vec_model / suffix; probe data is always online.")

    # Probe-specific.
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory to scan for data2vec_model_step*_<suffix>.pt; "
                             "required unless --checkpoint_path is given.")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="If set, probe only this single checkpoint.")
    parser.add_argument("--latest_only", action="store_true",
                        help="If set with --checkpoint_dir, probe only the checkpoint with the "
                             "largest step (ignores earlier checkpoints).")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to write probe results (defaults to --checkpoint_dir).")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=0,
                        help="Must match the --seed used for the training run (drives RHM rules).")
    parser.add_argument("--probe_steps", type=int, default=2_000)
    parser.add_argument("--probe_lr", type=float, default=1e-3)
    parser.add_argument("--probe_batch_size", type=int, default=512)
    parser.add_argument("--probe_online_seed", type=int, default=2024)
    parser.add_argument("--probe_test_size", type=int, default=4096)
    parser.add_argument("--pooling", type=str, default="mean",
                        choices=["mean", "max", "first"])
    return parser


def _extract_step(path: str) -> int:
    m = re.search(r"step(\d+)_", os.path.basename(path))
    return int(m.group(1)) if m else -1


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.n_layers is None:
        args.n_layers = 2 * args.num_layers
    if args.d_model is None:
        args.d_model = args.n_heads * 64
    if args.d_ff is None:
        args.d_ff = args.d_model * 4
    if args.ema_end_decay is None:
        args.ema_end_decay = args.ema_decay
    if args.ema_anneal_end_step is None:
        args.ema_anneal_end_step = 100_000

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve checkpoints.
    if args.checkpoint_path is not None:
        ckpts = [args.checkpoint_path]
        default_out = os.path.dirname(args.checkpoint_path)
    elif args.checkpoint_dir is not None:
        suffix = build_data2vec_file_name_suffix(args)
        pattern = os.path.join(args.checkpoint_dir, f"data2vec_model_step*_{suffix}.pt")
        ckpts = sorted(glob.glob(pattern), key=_extract_step)
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints match {pattern}")
        if args.latest_only:
            ckpts = [ckpts[-1]]
        default_out = args.checkpoint_dir
    else:
        raise ValueError("Pass either --checkpoint_path or --checkpoint_dir.")

    output_dir = args.output_dir or default_out
    os.makedirs(output_dir, exist_ok=True)

    # RHM rules: deterministic from --seed; must match training.
    dataset = RandomHierarchyModel(
        num_features=args.vocab_size,
        num_synonyms=args.num_synonyms,
        num_layers=args.num_layers,
        num_classes=args.vocab_size,
        tuple_size=args.tuple_size,
        seed_rules=args.seed,
        train_size=0,
        test_size=args.probe_test_size,
        seed_sample=42,
        input_format="onehot",
        whitening=0,
        replacement=1,
    )

    # Probe train: online, fresh samples every step.
    probe_train_loader = OnlineRHMLoader(
        rules=dataset.rules,
        num_classes=args.vocab_size,
        num_features=args.vocab_size,
        batch_size=args.probe_batch_size,
        seed=args.probe_online_seed,
        input_format="onehot",
    )
    # Probe test: fixed subset from the RHM dataset (different sample_seed from training).
    probe_test_set = torch.utils.data.Subset(dataset, range(args.probe_test_size))
    probe_test_loader = torch.utils.data.DataLoader(
        probe_test_set, batch_size=args.probe_batch_size, shuffle=False, num_workers=0,
    )

    print(f"Probing {len(ckpts)} checkpoint(s) in {output_dir}")
    print(f"  probe_steps={args.probe_steps}  batch={args.probe_batch_size}  "
          f"total_per_probe={args.probe_steps * args.probe_batch_size} samples")
    print(f"  probe_test_size={args.probe_test_size}")

    for ckpt_path in ckpts:
        step = _extract_step(ckpt_path)
        print(f"\n=== step {step}: {os.path.basename(ckpt_path)} ===", flush=True)

        model = build_data2vec_model(args).to(device).eval()
        model._amp_enabled = args.bf16
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        results = run_probe_suite(
            data2vec_model=model,
            train_loader=probe_train_loader,
            test_loader=probe_test_loader,
            num_classes=args.vocab_size,
            num_steps=args.probe_steps,
            lr=args.probe_lr,
            device=device,
            pooling=args.pooling,
        )
        print(f"  linear: best {results['linear']['best_test_acc']:.4f}  "
              f"mlp: best {results['mlp']['best_test_acc']:.4f}", flush=True)

        suffix = build_data2vec_file_name_suffix(args)
        out_path = os.path.join(
            output_dir,
            f"probe_classification_step{step}_{suffix}.pt",
        )
        torch.save(
            {
                "step": step,
                "results": results,
                "probe_steps": args.probe_steps,
                "probe_batch_size": args.probe_batch_size,
                "probe_test_size": args.probe_test_size,
                "args": vars(args),
            },
            out_path,
        )
        del model


if __name__ == "__main__":
    main()
