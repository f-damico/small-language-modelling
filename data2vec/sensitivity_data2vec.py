"""Synonym sensitivity analysis for the token-based data2vec encoder."""

from typing import Dict, List

import torch

from data2vec import amp_context


def get_encoder_activations(model, input_ids: torch.Tensor) -> List[torch.Tensor]:
    with torch.no_grad(), amp_context(model):
        model.eval()
        acts = model.get_representations(input_ids, return_all_layers=True)
    return [a.float() for a in acts]


def compute_typical_difference(activations: torch.Tensor) -> float:
    flat = activations.reshape(activations.shape[0], -1)
    if flat.shape[0] < 2:
        return 1.0
    return torch.nn.functional.pdist(flat, p=2).pow(2).mean().item()


def compute_sensitivity(
    original_acts: torch.Tensor,
    synonym_acts: torch.Tensor,
    typical_diff: float,
) -> float:
    if typical_diff < 1e-10:
        return 0.0
    batch_size = original_acts.shape[0]
    flat_orig = original_acts.reshape(batch_size, -1)
    flat_syn = synonym_acts.reshape(batch_size, -1)
    diff_sq = (flat_orig - flat_syn).pow(2).sum()
    return (diff_sq / (batch_size * typical_diff)).item()


def data2vec_sensitivity(
    model,
    dataset,
    num_samples: int = 256,
    seed_sample: int = 0,
    seed_synonyms: int = 42,
    device: str = "cuda",
) -> Dict:
    from random_hierarchy_model import sample_data_and_synonyms_with_replacement

    rules = dataset.rules
    num_layers_rhm = len(rules)
    num_layers_model = len(model.encoder.layers)

    data_syn, _ = sample_data_and_synonyms_with_replacement(
        sample_size=num_samples,
        seed_sample=seed_sample,
        rules=rules,
        synonims_layer=[i for i in range(num_layers_rhm)],
        synonims_seed=seed_synonyms,
    )

    x_orig = data_syn["data"].long().to(device) + 1
    orig_residual = get_encoder_activations(model, x_orig)
    typical_diff_residual = [compute_typical_difference(act) for act in orig_residual]

    residual_sens = {}
    for rhm_level in range(num_layers_rhm):
        x_syn = data_syn[rhm_level].long().to(device) + 1
        syn_residual = get_encoder_activations(model, x_syn)
        residual_sens[rhm_level] = {}
        for layer_idx in range(num_layers_model):
            residual_sens[rhm_level][layer_idx] = compute_sensitivity(
                orig_residual[layer_idx],
                syn_residual[layer_idx],
                typical_diff_residual[layer_idx],
            )

    return {"residual": residual_sens}


def print_sensitivity_results(results: Dict, title: str = "DATA2VEC ENCODER") -> None:
    residual_sens = results["residual"]
    num_layers_model = len(list(residual_sens.values())[0])

    print("\n" + "=" * 80)
    print(f"{title} - RESIDUAL STREAM sensitivity (lower = more invariant)")
    print("=" * 80)
    print(f"{'RHM Level':<12}", end="")
    for layer_idx in range(num_layers_model):
        print(f"Layer {layer_idx:<8}", end="")
    print()
    print("-" * 80)
    for rhm_level in sorted(residual_sens.keys()):
        label = f"(class) {rhm_level:<3}" if rhm_level == 0 else f"{rhm_level:<12}"
        print(label, end="")
        for layer_idx in range(num_layers_model):
            print(f"{residual_sens[rhm_level][layer_idx]:<14.4f}", end="")
        print()
    print("=" * 80)

