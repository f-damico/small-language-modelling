from typing import Iterable, Optional, Tuple

import torch

from data2vec import Data2Vec, amp_context, build_data2vec_file_name_suffix
from sensitivity_data2vec import data2vec_sensitivity, print_sensitivity_results
from train_probe import run_probe_suite


def _infinite_batches(loader: Iterable):
    # Yields batches forever without caching (unlike itertools.cycle).
    # Works for finite DataLoaders (outer `while` reshuffles each pass)
    # and for infinite iterators (inner `for` never exits).
    while True:
        for batch in loader:
            yield batch


def create_data2vec_optimizer(
    model: Data2Vec,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.98),
) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=betas,
    )


def train_step(
    model: Data2Vec,
    batch: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
):
    model.train()
    optimizer.zero_grad()
    with amp_context(model):
        loss, _, _ = model(batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    model.update_teacher()
    if scheduler is not None:
        scheduler.step()
    return loss.item(), model.get_last_loss_breakdown()


def train_data2vec(
    model: Data2Vec,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    num_steps: int = 32_768,
    lr: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    log_spacing_subdivs: int = 1,
    log_spacing_min_step: int = 1,
    lr_schedule: str = "none",
    warmup_steps: Optional[int] = None,
    args=None,
    dataset=None,
    compute_sensitivity: bool = False,
    num_sensitivity_samples: int = 256,
    probe_train_loader: Optional[torch.utils.data.DataLoader] = None,
    compute_probe: bool = False,
    probe_steps: int = 0,
    probe_lr: float = 1e-3,
    probe_pooling: str = "mean",
    probe_linear: bool = False,
    probe_num_classes: Optional[int] = None,
    resume_state: Optional[dict] = None,
):
    if args is None:
        output_dir = "./"
        file_name_suffix = ""
    else:
        output_dir = args.output_dir
        file_name_suffix = build_data2vec_file_name_suffix(args)

    model = model.to(device)
    optimizer = create_data2vec_optimizer(model, lr=lr)
    if resume_state is not None and resume_state.get("optimizer") is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
    scheduler = None
    if lr_schedule == "tri_stage":
        if warmup_steps is None:
            warmup_steps = max(1, int(round(num_steps * 0.05))) if num_steps > 1 else 0

        decay_steps = max(1, int(round(num_steps * 0.15))) if num_steps > 1 else 1
        hold_end_step = max(warmup_steps, num_steps - decay_steps)

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return max(1e-6, step / warmup_steps)
            if step < hold_end_step:
                return 1.0
            progress = (step - hold_end_step) / max(1, num_steps - hold_end_step)
            return max(0.0, 1.0 - progress)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if resume_state is not None and resume_state.get("logs") is not None:
        logs = resume_state["logs"]
        steps = list(logs.get("steps", []))
        train_losses = list(logs.get("train_losses", []))
        test_losses = list(logs.get("test_losses", []))
        sensitivity_results_list = list(logs.get("sensitivity_results", []))
        probe_results_list = list(logs.get("probe_results", []))
    else:
        steps = []
        train_losses = []
        test_losses = []
        sensitivity_results_list = []
        probe_results_list = []

    start_step = 1 if resume_state is None else int(resume_state.get("start_step", 1))
    step = start_step

    def _build_log_steps(n: int, subdivs: int, min_step: int) -> set:
        # s_i = round(2 ** (i / subdivs)); exponent is exact in fp64 at every
        # multiple of subdivs, so powers of 2 are always hit without drift.
        out = [1]
        i = 1
        while True:
            s = int(round(2.0 ** (i / subdivs)))
            s = max(s, out[-1] + min_step)
            if s > n:
                break
            out.append(s)
            i += 1
        if n >= 1 and out[-1] != n:
            out.append(n)
        return set(out)

    log_steps_set = _build_log_steps(num_steps, log_spacing_subdivs, log_spacing_min_step)
    # Skip log steps already recorded before the resume point.
    log_steps_set = {s for s in log_steps_set if s >= start_step}
    train_iter = _infinite_batches(train_loader)

    if resume_state is not None:
        print(f"Resuming at step {start_step} with {len(steps)} log entries preserved", flush=True)

    # Step 0: evaluate the random-init model as a baseline (before any optimizer step).
    if resume_state is None:
        init_test_loss, init_sensitivity_results, init_probe_results = evaluate_data2vec(
            model,
            test_loader,
            device=device,
            compute_sensitivity=compute_sensitivity,
            dataset=dataset,
            num_sensitivity_samples=num_sensitivity_samples,
            seed_sample=999,
            seed_synonyms=456,
            probe_train_loader=probe_train_loader,
            compute_probe=compute_probe,
            probe_steps=probe_steps,
            probe_lr=probe_lr,
            probe_pooling=probe_pooling,
            probe_linear=probe_linear,
            probe_num_classes=probe_num_classes,
        )
        print(f"Step 0 (init), Test Loss: {init_test_loss:.2e}", flush=True)
        if init_probe_results is not None:
            print(
                "  Probe snapshot:"
                f" linear best {init_probe_results['linear']['best_test_acc']:.4f},"
                f" mlp best {init_probe_results['mlp']['best_test_acc']:.4f}",
                flush=True,
            )
        steps.append(0)
        train_losses.append(float("nan"))
        test_losses.append(init_test_loss)
        sensitivity_results_list.append(init_sensitivity_results)
        probe_results_list.append(init_probe_results)
        torch.save(
            {
                "steps": steps,
                "train_losses": train_losses,
                "test_losses": test_losses,
                "sensitivity_results": sensitivity_results_list,
                "probe_results": probe_results_list,
            },
            f"{output_dir}/data2vec_training_log_{file_name_suffix}.pt",
        )

    while step <= num_steps:
        batch = next(train_iter)
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x = x.to(device)
        loss, train_breakdown = train_step(model, x, optimizer, scheduler)

        if step in log_steps_set:
            test_loss, sensitivity_results, probe_results = evaluate_data2vec(
                model,
                test_loader,
                device=device,
                compute_sensitivity=compute_sensitivity,
                dataset=dataset,
                num_sensitivity_samples=num_sensitivity_samples,
                seed_sample=999,
                seed_synonyms=456,
                probe_train_loader=probe_train_loader,
                compute_probe=compute_probe,
                probe_steps=probe_steps,
                probe_lr=probe_lr,
                probe_pooling=probe_pooling,
                probe_linear=probe_linear,
                probe_num_classes=probe_num_classes,
            )
            print(
                f"Step {step}, Train Loss: {loss:.2e}, Test Loss: {test_loss:.2e}, "
                f"EMA: {train_breakdown.get('ema_decay', 0.0):.6f}",
                flush=True,
            )
            if probe_results is not None:
                print(
                    "  Probe snapshot:"
                    f" linear best {probe_results['linear']['best_test_acc']:.4f},"
                    f" mlp best {probe_results['mlp']['best_test_acc']:.4f}",
                    flush=True,
                )

            steps.append(step)
            train_losses.append(loss)
            test_losses.append(test_loss)
            sensitivity_results_list.append(sensitivity_results)
            probe_results_list.append(probe_results)

            save_dict = {
                "steps": steps,
                "train_losses": train_losses,
                "test_losses": test_losses,
                "sensitivity_results": sensitivity_results_list,
                "probe_results": probe_results_list,
            }
            torch.save(save_dict, f"{output_dir}/data2vec_training_log_{file_name_suffix}.pt")

            if args is not None and args.save_checkpoints:
                torch.save(model.state_dict(), f"{output_dir}/data2vec_model_step{step}_{file_name_suffix}.pt")
            else:
                torch.save(model.state_dict(), f"{output_dir}/data2vec_model_{file_name_suffix}.pt")
            torch.save(optimizer.state_dict(), f"{output_dir}/data2vec_optimizer_{file_name_suffix}.pt")

        step += 1


def evaluate_data2vec(
    model: Data2Vec,
    test_loader: torch.utils.data.DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    compute_sensitivity: bool = False,
    dataset=None,
    num_sensitivity_samples: int = 256,
    seed_sample: int = 0,
    seed_synonyms: int = 42,
    probe_train_loader: Optional[torch.utils.data.DataLoader] = None,
    compute_probe: bool = False,
    probe_steps: int = 0,
    probe_lr: float = 1e-3,
    probe_pooling: str = "mean",
    probe_linear: bool = False,
    probe_num_classes: Optional[int] = None,
):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad(), amp_context(model):
        for batch in test_loader:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.to(device)
            loss, _, _ = model(x)
            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    avg_loss = total_loss / max(1, total_samples)
    sensitivity_results = None
    if compute_sensitivity:
        if dataset is None:
            raise ValueError("dataset must be provided when compute_sensitivity=True")
        sensitivity_results = data2vec_sensitivity(
            model=model,
            dataset=dataset,
            num_samples=num_sensitivity_samples,
            seed_sample=seed_sample,
            seed_synonyms=seed_synonyms,
            device=device,
        )
        print_sensitivity_results(sensitivity_results, title="DATA2VEC ENCODER (during training)")

    probe_results = None
    if compute_probe:
        if probe_train_loader is None:
            raise ValueError("probe_train_loader must be provided when compute_probe=True")
        if probe_num_classes is None:
            raise ValueError("probe_num_classes must be provided when compute_probe=True")
        probe_results = run_probe_suite(
            data2vec_model=model,
            train_loader=probe_train_loader,
            test_loader=test_loader,
            num_classes=probe_num_classes,
            num_steps=probe_steps,
            lr=probe_lr,
            device=device,
            pooling=probe_pooling,
        )

    return avg_loss, sensitivity_results, probe_results
