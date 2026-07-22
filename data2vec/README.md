# data2vec_rhm_v2

This directory contains a cleaner, more paper-faithful `data2vec` implementation
for the Random Hierarchy Model.

Design choices:
- Reuses `random_hierarchy_model.py` from the original repo.
- Drops the repo-specific hierarchical/multi-scale pretraining branch.
- Keeps the existing one-hot RHM interface, but internally converts one-hot
  inputs back to token ids before the encoder.
- Keeps the original checkpoint/log naming convention:
  - `data2vec_training_log_<suffix>.pt`
  - `data2vec_model_step<step>_<suffix>.pt`
  - `data2vec_model_<suffix>.pt`

Key implementation points borrowed from the fairseq design:
- teacher/student self-distillation with EMA teacher
- fp32 EMA teacher weights
- shared token/position frontend between student and teacher
- top-`K` teacher-layer target averaging
- regression on masked positions only
- configurable Smooth-L1 beta and EMA annealing

Current defaults are chosen to be closer to the published NLP setup on top of
RHM:
- transformer depth defaults to `2 * num_layers`
- teacher averages the top `num_layers` transformer blocks by default
- masking defaults to span masking with `mask_prob=0.35`, `mask_length=4`
- loss defaults to Smooth-L1 with `beta=4.0`

Main entry point:

```bash
python main_with_probe.py --train_size 8192 --num_steps 32768 --save_checkpoints
```
