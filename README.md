# Small Language Modelling

Code to train small language models (transformers and Mamba SSMs) on text datasets and study their scaling behaviour. Accompanies the paper:

> **Deriving neural scaling laws from the statistics of natural language**  
> Francesco Cagnetta, Allan Raventós, Surya Ganguli, Matthieu Wyart  
> https://arxiv.org/pdf/2602.07488

## Repository structure

```
SLM/                  # Training code
  main.py             # Entry point — parses arguments and runs training
  models/             # Transformer and Mamba architectures
  train.py            # Training step
  measures.py         # Evaluation utilities
  init.py             # Data and model initialisation
  datasets/           # Dataset loading utilities

running_scripts/      # SLURM job scripts

results/              # Data analysis
  figure_2.ipynb      # Reproduces Figure 2
  figure_5.ipynb      # Reproduces Figure 5
  proc_*.ipynb        # Processing notebooks for scaling, hyperparameters, etc.
```

## Requirements

- PyTorch

**Mamba models** require two additional packages with CUDA/Triton kernels (Linux + CUDA only):

```bash
pip install mamba-ssm causal-conv1d
```

Without these, the code falls back to a pure-PyTorch Mamba implementation that runs on CPU or macOS but is significantly slower. The Triton-accelerated path was tested on H100 GPUs.

## Usage

```bash
python SLM/main.py \
  --dataset <name> \
  --block_size 128 \
  --batch_size 64 \
  --d_embedding 256 \
  --depth 12 \
  --n_heads 4 \
  --lr 1e-3 \
  --max_epochs 1 \
  --outname results/my_run
```

Run `python SLM/main.py --help` for the full list of options.

## License

MIT — see [LICENSE](LICENSE).
