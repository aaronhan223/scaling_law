# Scaling Law Experiments for Language Models

This repository contains code to reproduce and extend the scaling law experiments from the paper "Scaling Laws for Neural Language Models" (Kaplan et al., 2020).

## Overview

The experiments test how language model performance scales with:
- **Model size** (number of parameters)
- **Dataset size** (number of training tokens)  
- **Compute budget** (FLOPs)

Additionally, this implementation includes a novel experiment setting where random linear and non-linear transformations are applied to the data to test scaling laws under distribution shift.

## Installation

1. Install dependencies:
```bash
cd src
pip install -r requirements.txt
```

2. Set up Weights & Biases (optional but recommended):
```bash
wandb login
```

## Project Structure

```
src/
├── config.py               # Experiment configurations
├── data/                   # Data loading and transformations
│   ├── dataset.py         # WikiText-103 dataset loader
│   └── transforms.py      # Linear and non-linear transformations
├── models/                 # Model implementations
│   └── transformer.py     # Transformer language model
├── training/              # Training utilities
│   ├── trainer.py        # Training loop
│   └── utils.py          # Helper functions
├── experiments/           # Experiment runners
│   ├── scaling_law_experiment.py  # Main experiment logic
│   └── compute_tracker.py         # Compute usage tracking
├── visualization/         # Plotting functions
│   └── plots.py          # Scaling law visualizations
└── main.py               # Entry point
```

## Running Experiments

### Basic Usage

Run a small-scale experiment:
```bash
python main.py --model-sizes small --dataset-sizes 1M,10M --no-wandb
```

### Full Scaling Law Experiments

Run standard scaling law experiments:
```bash
python main.py \
    --experiment-name scaling_laws_standard \
    --model-sizes small,medium,large \
    --dataset-sizes 1M,10M,100M \
    --skip-transformed
```

Run experiments with data transformations:
```bash
python main.py \
    --experiment-name scaling_laws_transformed \
    --model-sizes small,medium,large \
    --dataset-sizes 1M,10M,100M \
    --skip-standard
```

Run both standard and transformed experiments:
```bash
python main.py \
    --experiment-name scaling_laws_full \
    --model-sizes small,medium,large \
    --dataset-sizes 1M,10M,100M
```

### Command Line Options

- `--model-sizes`: Comma-separated list of model sizes (tiny, small, medium, large, xlarge, or indices 0-6)
- `--dataset-sizes`: Comma-separated list of dataset sizes (e.g., 1M, 10M, 100M, 1B)
- `--batch-size`: Training batch size (default: 32)
- `--learning-rate`: Learning rate (default: 5e-4)
- `--warmup-steps`: Number of warmup steps (default: 1000)
- `--eval-steps`: Evaluation frequency (default: 500)
- `--output-dir`: Output directory (default: ./outputs)
- `--wandb-project`: W&B project name (default: scaling-laws)
- `--no-wandb`: Disable W&B logging
- `--device`: Device to use (cuda/cpu)
- `--fp16`: Use mixed precision training (default: True)
- `--skip-standard`: Skip standard experiments
- `--skip-transformed`: Skip transformed experiments
- `--plot-only`: Only generate plots from existing results

### Model Sizes

The predefined model configurations are:

| Size   | Parameters | Hidden Size | Layers | Heads |
|--------|------------|-------------|--------|-------|
| 0 (tiny) | ~7M      | 128         | 2      | 4     |
| 1      | ~13M       | 256         | 4      | 8     |
| 2      | ~39M       | 512         | 6      | 8     |
| 3 (medium) | ~125M  | 768         | 12     | 12    |
| 4      | ~355M      | 1024        | 24     | 16    |
| 5 (large) | ~775M   | 1536        | 36     | 16    |
| 6 (xlarge) | ~1.5B  | 2048        | 48     | 32    |

## Outputs

The experiments generate:

1. **Model checkpoints**: Saved in `outputs/<experiment_name>/`
2. **Results CSV/JSON**: Scaling law measurements in `outputs/<experiment_name>/results/`
3. **Plots**: Visualizations similar to the paper figures in `outputs/<experiment_name>/plots/`
   - Loss vs Parameters (Figure 1)
   - Loss vs Dataset Size (Figure 2)
   - Loss vs Compute (Figure 4)
   - Optimal Model Size vs Compute (Figure 6)
   - Comparison plots for standard vs transformed

4. **W&B logs**: Real-time training metrics and plots (if enabled)

## Expected Results

The experiments should reproduce the key findings from the paper:

1. **Power law relationships**:
   - Loss ∝ N^(-α) where N is model parameters (α ≈ 0.076)
   - Loss ∝ D^(-β) where D is dataset size (β ≈ 0.095)
   - Loss ∝ C^(-γ) where C is compute (γ ≈ 0.050)

2. **Optimal allocation**: For a fixed compute budget, optimal model size scales as N_opt ∝ C^0.73

3. **Transformed data**: The novel transformation experiments show how scaling laws change under distribution shift.

## Extending the Experiments

To add new experiments:

1. **New model architectures**: Modify `models/transformer.py`
2. **New datasets**: Add loaders in `data/dataset.py`
3. **New transformations**: Implement in `data/transforms.py`
4. **New metrics**: Update `training/trainer.py`

## Citation

If you use this code, please cite:

```bibtex
@article{kaplan2020scaling,
  title={Scaling laws for neural language models},
  author={Kaplan, Jared and McCandlish, Sam and Henighan, Tom and Brown, Tom B and Chess, Benjamin and Child, Rewon and Gray, Scott and Radford, Alec and Wu, Jeffrey and Amodei, Dario},
  journal={arXiv preprint arXiv:2001.08361},
  year={2020}
}
```

## Notes

- The experiments are compute-intensive. Start with small-scale tests.
- Use GPU acceleration and mixed precision training for faster experiments.
- The full experiments in the paper used much larger models and datasets than the defaults here.
- Adjust batch sizes based on your GPU memory. 