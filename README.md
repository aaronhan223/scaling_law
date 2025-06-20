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
├── main.py               # Entry point
├── visualize_results.py  # Standalone plotting script
└── run_*.sh             # Experiment scripts
```

## Running Experiments

### Quick Start

For a comprehensive test that validates the experimental setup:
```bash
./run_quick_test.sh
```
This runs 72 experiments (6 model sizes × 6 dataset sizes × 2 experiment types) with smaller datasets for faster execution (~2-4 hours).

### Comprehensive Experiments

#### Standard Scaling Laws Only
```bash
./run_standard.sh
```
Runs 36 experiments testing standard scaling laws (6 model sizes × 6 dataset sizes).

#### Transformed Data Experiments Only
```bash
./run_transform.sh
```
Runs 36 experiments testing scaling laws under data transformations (6 model sizes × 6 dataset sizes).

#### Full Experiments (Recommended)
```bash
./run_full.sh
```
Runs 72 experiments covering both standard and transformed data scaling laws (~6-12 hours).

#### Extensive Experiments (Maximum Coverage)
```bash
./run_extensive.sh
```
Runs 98 experiments with all 7 model sizes and 7 dataset sizes (~12-24 hours).

### Custom Experiments

Run specific configurations:
```bash
python main.py \
    --model-sizes 0,1,2,3,4,5 \
    --dataset-sizes 1M,3M,10M,30M,100M,300M \
    --wandb-project my-scaling-laws
```

### Command Line Options

- `--model-sizes`: Comma-separated list of model indices (0-6) or names (tiny, small, medium, large, xlarge)
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

### Model Configurations

The framework includes 7 predefined model configurations spanning multiple orders of magnitude:

| Index | Size | Parameters | Hidden Size | Layers | Heads | Description |
|-------|------|------------|-------------|--------|-------|-------------|
| 0 | tiny | ~7M | 128 | 2 | 4 | Minimal model |
| 1 | small-1 | ~13M | 256 | 4 | 8 | Small model |
| 2 | small-2 | ~39M | 512 | 6 | 8 | Medium-small |
| 3 | medium | ~125M | 768 | 12 | 12 | Medium model |
| 4 | large-1 | ~355M | 1024 | 24 | 16 | Large model |
| 5 | large-2 | ~775M | 1536 | 36 | 16 | Very large |
| 6 | xlarge | ~1.5B | 2048 | 48 | 32 | Extra large |

### Dataset Configurations

Standard dataset sizes for comprehensive scaling:
- **Quick test**: 100K, 300K, 1M, 3M, 10M, 30M tokens
- **Standard**: 1M, 3M, 10M, 30M, 100M, 300M tokens  
- **Extensive**: 1M, 3M, 10M, 30M, 100M, 300M, 1B tokens

Each size provides approximately 3× scaling between adjacent points for robust power law fitting.

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

### Generating Plots

Create plots from existing results:
```bash
python visualize_results.py outputs/experiment/results/scaling_law_results.csv
```

## Expected Results

The experiments should reproduce the key findings from the paper:

1. **Power law relationships**:
   - Loss ∝ N^(-α) where N is model parameters (α ≈ 0.076)
   - Loss ∝ D^(-β) where D is dataset size (β ≈ 0.095)
   - Loss ∝ C^(-γ) where C is compute (γ ≈ 0.050)

2. **Optimal allocation**: For a fixed compute budget, optimal model size scales as N_opt ∝ C^0.73

3. **Transformed data**: The novel transformation experiments show how scaling laws change under distribution shift, providing insights into:
   - Robustness of scaling relationships
   - Impact of data quality vs quantity
   - Generalization across different data distributions

## Computational Requirements

### Resource Estimates

| Experiment Type | Models | Datasets | Total Experiments | Est. Time | GPU Memory |
|----------------|--------|----------|-------------------|-----------|------------|
| Quick Test | 6 | 6 | 72 | 2-4 hours | 8-16 GB |
| Standard | 6 | 6 | 36 | 3-6 hours | 8-16 GB |
| Full | 6 | 6 | 72 | 6-12 hours | 8-16 GB |
| Extensive | 7 | 7 | 98 | 12-24 hours | 16-32 GB |

### Optimization Tips

- Use mixed precision training (`--fp16`) to reduce memory usage
- Adjust batch size based on available GPU memory
- Start with quick test to validate setup before running full experiments
- Use gradient accumulation for larger effective batch sizes
- Monitor W&B logs for training progress

## Extending the Experiments

To add new experiments:

1. **New model architectures**: Modify `models/transformer.py`
2. **New datasets**: Add loaders in `data/dataset.py`
3. **New transformations**: Implement in `data/transforms.py`
4. **New metrics**: Update `training/trainer.py`
5. **Custom scaling ranges**: Modify configurations in `config.py`

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

- The experiments are compute-intensive. Start with `run_quick_test.sh` to validate your setup.
- Use GPU acceleration and mixed precision training for optimal performance.
- The largest models require significant GPU memory (16+ GB recommended).
- Results improve with more scaling points - use at least 6 points per dimension for robust power law fitting.
- Monitor training progress through W&B or local logs to catch issues early. 