"""Train linear probes on data2vec teacher targets to recover RHM latents.

For each chosen leaf position i and each RHM level L in {0..num_layers-1},
train a linear probe that maps the teacher target at position i to the value
of i's ancestor latent at level L. The probe feature is what data2vec
regresses during pretraining: the average of the top-K teacher FFN
pre-residual outputs (with optional per-layer LayerNorm), taken from the
unmasked input.
"""

import argparse
import os
import warnings

import torch
import torch.nn as nn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from data2vec import amp_context, build_data2vec_file_name_suffix, build_data2vec_model
from random_hierarchy_model import sample_data_from_generator_classes, sample_rules


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe RHM latents from data2vec teacher targets."
    )

    # Must match the checkpoint's training config.
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
    parser.add_argument("--online", action="store_true")

    # Probe-specific.
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--bf16", action="store_true",
                        help="Run encoder forwards under torch.autocast(bfloat16).")
    parser.add_argument(
        "--probe_train_size", type=int, default=None,
        help="Defaults to 10 * vocab_size * num_synonyms ** (num_layers + 1).",
    )
    parser.add_argument("--probe_test_size", type=int, default=1024)
    parser.add_argument("--probe_steps", type=int, default=10_000)
    parser.add_argument("--probe_lr", type=float, default=1e-3)
    parser.add_argument("--probe_batch_size", type=int, default=512)
    parser.add_argument(
        "--positions", type=str, default="two_random",
        choices=["two_random", "all"],
    )
    parser.add_argument("--position_seed", type=int, default=0)
    parser.add_argument("--sample_seed", type=int, default=1234)
    parser.add_argument("--random_init_seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0,
                        help="Must match the --seed used for the training run being probed "
                             "(drives RHM rules).")
    parser.add_argument("--probe_online", action="store_true",
                        help="Sample fresh RHM batches each step; one final test pass.")
    parser.add_argument("--probe_online_seed", type=int, default=2024)
    parser.add_argument(
        "--classifier", type=str, default="logreg", choices=["logreg", "sgd"],
        help="Offline classifier: sklearn LogisticRegression (stable) or Adam-trained linear head.",
    )
    parser.add_argument("--logreg_C", type=float, default=1e6)
    parser.add_argument("--logreg_max_iter", type=int, default=1000)
    parser.add_argument("--mean_pooling", action="store_true",
                        help="Also probe mean-pooled features across all positions.")
    parser.add_argument("--output_dir", type=str, default="./results/probe_latents")
    return parser


def sample_rhm_with_tree(rules, num_samples: int, seed: int):
    g = torch.Generator()
    g.manual_seed(seed)
    n = rules[0].shape[0]
    y = torch.randint(0, n, size=(num_samples,), generator=g)
    x_st, labels = sample_data_from_generator_classes(
        g, y, rules, return_tree_structure=True
    )
    return x_st, labels


def _extract_features(model, input_ids: torch.Tensor, source: str) -> torch.Tensor:
    with amp_context(model):
        if source == "teacher":
            return model._teacher_targets(input_ids).float()
        if source == "student":
            return model.encoder(input_ids).float()
    raise ValueError(f"Unknown source: {source}")


@torch.no_grad()
def extract_teacher_targets(
    model,
    tokens: torch.Tensor,
    device: str,
    batch_size: int = 512,
    source: str = "teacher",
) -> torch.Tensor:
    # tokens: (N, seq_len) long in [0, vocab_size). Shift by +1 to match the
    # padding-reserved id convention (padding_idx=0, classes at [1, vocab_size]).
    model.eval()
    input_ids_all = tokens.long() + 1
    chunks = []
    for i in range(0, input_ids_all.shape[0], batch_size):
        input_ids = input_ids_all[i : i + batch_size].to(device)
        chunks.append(_extract_features(model, input_ids, source).cpu())
    return torch.cat(chunks, dim=0)


def ancestor_index(position: int, level: int, L: int, s: int) -> int:
    return position // (s ** (L - level))


def sklearn_logreg_probe(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    C: float = 1.0,
    max_iter: int = 1000,
):
    clf = LogisticRegression(
        solver="lbfgs",
        C=C,
        max_iter=max_iter,
        n_jobs=-1,
    )
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(X_train.numpy(), y_train.numpy())
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in wlist)
    acc = float(clf.score(X_test.numpy(), y_test.numpy()))
    n_iter = int(clf.n_iter_.max())
    return acc, converged, n_iter


def train_linear_probe(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    num_classes: int,
    num_steps: int,
    lr: float,
    device: str,
    batch_size: int = 512,
) -> float:
    d = X_train.shape[-1]
    probe = nn.Linear(d, num_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    N = X_train.shape[0]
    g = torch.Generator()
    g.manual_seed(0)

    X_test_dev = X_test.to(device)
    y_test_dev = y_test.to(device)

    step = 0
    best_test_acc = 0.0
    while step < num_steps:
        idx = torch.randperm(N, generator=g)
        for j in range(0, N, batch_size):
            if step >= num_steps:
                break
            ids = idx[j : j + batch_size]
            xb = X_train[ids].to(device)
            yb = y_train[ids].to(device)
            probe.train()
            logits = probe(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

        probe.eval()
        with torch.no_grad():
            preds = probe(X_test_dev).argmax(dim=-1)
            acc = (preds == y_test_dev).float().mean().item()
            best_test_acc = max(best_test_acc, acc)

    return best_test_acc


@torch.no_grad()
def _teacher_targets_batch(
    model, tokens: torch.Tensor, device: str, source: str = "teacher"
) -> torch.Tensor:
    input_ids = (tokens.long() + 1).to(device)
    return _extract_features(model, input_ids, source)


def probe_model_online(model, args, rules, positions, L, s, device, tag: str, source: str):
    print("=" * 80)
    print(f"PROBING (online, source={source}): {tag}")
    print("=" * 80, flush=True)

    n = rules[0].shape[0]
    g = torch.Generator()
    g.manual_seed(args.probe_online_seed)

    keys = list(positions) + (["mean"] if args.mean_pooling else [])
    probes = {
        (key, level): nn.Linear(args.d_model, args.vocab_size).to(device)
        for key in keys
        for level in range(L)
    }
    params = [p for probe in probes.values() for p in probe.parameters()]
    optimizer = torch.optim.Adam(params, lr=args.probe_lr)
    criterion = nn.CrossEntropyLoss()

    def feature_slice(targets, key):
        if key == "mean":
            return targets.mean(dim=1)
        return targets[:, key, :]

    def label_for(key, level, x_st):
        if key == "mean":
            anc = 0
        else:
            anc = ancestor_index(key, level, L, s)
        labels_full = x_st[level] if level == 0 else x_st[level][:, anc]
        return labels_full.long(), anc

    for probe in probes.values():
        probe.train()
    for step in range(args.probe_steps):
        y = torch.randint(0, n, size=(args.probe_batch_size,), generator=g)
        x_st, _ = sample_data_from_generator_classes(
            g, y, rules, return_tree_structure=True
        )
        targets = _teacher_targets_batch(model, x_st[L], device, source=source)

        total_loss = 0.0
        for key in keys:
            X = feature_slice(targets, key)
            for level in range(L):
                y_lbl, _ = label_for(key, level, x_st)
                logits = probes[(key, level)](X)
                total_loss = total_loss + criterion(logits, y_lbl.to(device))
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    # One large test pass.
    g_test = torch.Generator()
    g_test.manual_seed(args.probe_online_seed + 10**6)
    y = torch.randint(0, n, size=(args.probe_test_size,), generator=g_test)
    x_st, _ = sample_data_from_generator_classes(
        g_test, y, rules, return_tree_structure=True
    )
    targets = _teacher_targets_batch(model, x_st[L], device, source=args.source)

    results = {key: {} for key in keys}
    for probe in probes.values():
        probe.eval()
    with torch.no_grad():
        for key in keys:
            X = feature_slice(targets, key)
            for level in range(L):
                y_lbl, anc = label_for(key, level, x_st)
                preds = probes[(key, level)](X).argmax(dim=-1)
                acc = (preds == y_lbl.to(device)).float().mean().item()
                results[key][level] = acc
                pos_str = "mean" if key == "mean" else f"{key:>3}"
                print(
                    f"[{tag}] position {pos_str}, level {level}, ancestor {anc:>3}: "
                    f"test acc {acc:.4f}",
                    flush=True,
                )

    level_means = {
        level: sum(results[p][level] for p in positions) / len(positions)
        for level in range(L)
    }
    print(f"[{tag}] Mean test accuracy across positions:")
    for level, mean_acc in level_means.items():
        print(f"  level {level}: {mean_acc:.4f}", flush=True)
    return results, level_means


def probe_model(model, args, x_st, positions, L, s, device, tag: str, source: str):
    print("=" * 80)
    print(f"PROBING (source={source}): {tag}")
    print("=" * 80, flush=True)

    tokens = x_st[L]
    targets = extract_teacher_targets(
        model, tokens, device, batch_size=args.probe_batch_size, source=source,
    )
    targets_train = targets[: args.probe_train_size]
    targets_test = targets[args.probe_train_size :]
    print(f"Teacher target dim: {targets.shape[-1]}")

    keys = list(positions) + (["mean"] if args.mean_pooling else [])
    if args.mean_pooling:
        mean_train = targets_train.mean(dim=1)
        mean_test = targets_test.mean(dim=1)

    results = {}
    for key in keys:
        results[key] = {}
        if key == "mean":
            X_tr, X_te = mean_train, mean_test
        else:
            X_tr = targets_train[:, key, :]
            X_te = targets_test[:, key, :]
        for level in range(L):
            anc = 0 if key == "mean" else ancestor_index(key, level, L, s)
            labels_full = x_st[level] if level == 0 else x_st[level][:, anc]
            y_tr = labels_full[: args.probe_train_size].long()
            y_te = labels_full[args.probe_train_size :].long()
            conv_info = ""
            if args.classifier == "logreg":
                acc, converged, n_iter = sklearn_logreg_probe(
                    X_tr, y_tr, X_te, y_te,
                    C=args.logreg_C,
                    max_iter=args.logreg_max_iter,
                )
                conv_info = (
                    f" [n_iter={n_iter}{'' if converged else ' NO-CONV'}]"
                )
            else:
                acc = train_linear_probe(
                    X_tr, y_tr, X_te, y_te,
                    num_classes=args.vocab_size,
                    num_steps=args.probe_steps,
                    lr=args.probe_lr,
                    device=device,
                    batch_size=args.probe_batch_size,
                )
            results[key][level] = acc
            pos_str = "mean" if key == "mean" else f"{key:>3}"
            print(
                f"[{tag}] position {pos_str}, level {level}, ancestor {anc:>3}: "
                f"test acc {acc:.4f}{conv_info}",
                flush=True,
            )

    level_means = {
        level: sum(results[p][level] for p in positions) / len(positions)
        for level in range(L)
    }
    print(f"[{tag}] Mean test accuracy across positions:")
    for level, mean_acc in level_means.items():
        print(f"  level {level}: {mean_acc:.4f}", flush=True)
    return results, level_means


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
    if args.probe_train_size is None:
        args.probe_train_size = (
            10 * args.vocab_size * args.num_synonyms ** (args.num_layers + 1)
        )

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("ARGS")
    print("=" * 80)
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")
    print("=" * 80)

    rules = sample_rules(
        v=args.vocab_size,
        n=args.vocab_size,
        m=args.num_synonyms,
        s=args.tuple_size,
        L=args.num_layers,
        seed=args.seed,
    )

    L = args.num_layers
    s = args.tuple_size
    seq_len = s ** L

    if args.positions == "all":
        positions = list(range(seq_len))
    else:
        pg = torch.Generator()
        pg.manual_seed(args.position_seed)
        positions = sorted(
            set(int(p) for p in torch.randperm(seq_len, generator=pg)[:2].tolist())
        )
    print(f"Probing positions: {positions}, seq_len: {seq_len}")

    if args.probe_online:
        def run_probe(model, tag, source):
            return probe_model_online(
                model, args, rules, positions, L, s, device, tag, source
            )
    else:
        total = args.probe_train_size + args.probe_test_size
        x_st, _ = sample_rhm_with_tree(rules, total, seed=args.sample_seed)

        def run_probe(model, tag, source):
            return probe_model(
                model, args, x_st, positions, L, s, device, tag, source
            )

    sources = ["teacher", "student"]

    torch.manual_seed(args.random_init_seed)
    random_model = build_data2vec_model(args).to(device).eval()
    random_model._amp_enabled = args.bf16
    random_out = {
        src: run_probe(random_model, tag="random_init", source=src)
        for src in sources
    }
    del random_model

    trained_model = build_data2vec_model(args)
    state = torch.load(args.checkpoint_path, map_location="cpu")
    trained_model.load_state_dict(state)
    trained_model = trained_model.to(device).eval()
    trained_model._amp_enabled = args.bf16
    trained_out = {
        src: run_probe(trained_model, tag="trained", source=src)
        for src in sources
    }

    print("=" * 80)
    print("SUMMARY (mean test accuracy across positions)")
    print("=" * 80)
    header = (
        f"{'level':>6} {'rand/teach':>12} {'rand/stud':>12} "
        f"{'train/teach':>12} {'train/stud':>12}"
    )
    print(header)
    for level in range(L):
        rt = random_out["teacher"][1][level]
        rs = random_out["student"][1][level]
        tt = trained_out["teacher"][1][level]
        ts = trained_out["student"][1][level]
        print(
            f"{level:>6} {rt:>12.4f} {rs:>12.4f} {tt:>12.4f} {ts:>12.4f}",
            flush=True,
        )
    if args.mean_pooling:
        print("-" * 80)
        print("MEAN-POOLED FEATURES (predicting ancestor at position 0)")
        print(header)
        for level in range(L):
            rt = random_out["teacher"][0]["mean"][level]
            rs = random_out["student"][0]["mean"][level]
            tt = trained_out["teacher"][0]["mean"][level]
            ts = trained_out["student"][0]["mean"][level]
            print(
                f"{level:>6} {rt:>12.4f} {rs:>12.4f} {tt:>12.4f} {ts:>12.4f}",
                flush=True,
            )

    import re
    m = re.search(r"data2vec_model_step(\d+)_", os.path.basename(args.checkpoint_path))
    step_tag = f"_step{m.group(1)}" if m else ""
    save_path = os.path.join(
        args.output_dir,
        f"latent_probe{step_tag}_{build_data2vec_file_name_suffix(args)}.pt",
    )
    torch.save(
        {
            "positions": positions,
            "random_init": {
                src: {"results": r, "level_means": m}
                for src, (r, m) in random_out.items()
            },
            "trained": {
                src: {"results": r, "level_means": m}
                for src, (r, m) in trained_out.items()
            },
            "args": vars(args),
        },
        save_path,
    )
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
