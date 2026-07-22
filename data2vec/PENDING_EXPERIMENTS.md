# Pending RHM experiments

## [ ] Latent probes — online big_bf16 runs (m=3..7 × seeds {0,1})

```bash
sbatch slurm/run_probe_all_steps_big_bf16.sh
```

Runs `probe_latents.py` on every step checkpoint in `online_v16_m{M}_L4_big_bf16_seed{SEED}/` for m∈{3,4,5,6,7} and seeds∈{0,1} (10 array tasks). Outputs `latent_probe_step*_{suffix}.pt` per checkpoint — the per-level RHM-latent recovery accuracy used for the hierarchy-cascade plots. No online big_bf16 dir currently has any `latent_probe_*` files. 10 h wall.
