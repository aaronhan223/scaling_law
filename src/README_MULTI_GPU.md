# Multi-GPU Training Guide for Scaling Law Experiments

This guide explains how to use the multi-GPU training capabilities in the scaling law experiments codebase.

## Overview

The codebase now supports both single-GPU and multi-GPU training using PyTorch's DataParallel. This allows for faster training and the ability to use larger batch sizes for better efficiency.

## Experiment Types

The codebase supports four different experiment types:

1. **Standard**: No transformations applied to the data
2. **Linear**: Only linear token transformations applied
3. **Nonlinear**: Only nonlinear token transformations applied  
4. **Transformed**: Both linear and nonlinear transformations applied (combined)

## Available Scripts

- `run_test_transforms.sh` - Quick test of all transform types (12 experiments, ~15-30 minutes)
- `run_full_multi_gpu.sh` - Standard, linear, and nonlinear experiments using multiple GPUs (108 experiments)
- `run_all_experiments.sh` - Comprehensive experiments with all four types (120 experiments)

### Script Details

#### Quick Test Script (`run_test_transforms.sh`)
```bash
./src/run_test_transforms.sh
```
- 2 model sizes × 2 dataset sizes × 3 experiment types = 12 experiments
- Runtime: 15-30 minutes
- Good for testing the setup and new functionality

#### Multi-GPU Script (`run_full_multi_gpu.sh`)  
```bash
./src/run_full_multi_gpu.sh
```
- 5 model sizes × 6 dataset sizes × 3 experiment types = 90 experiments
- Runs: standard, linear, nonlinear (skips combined transformed)
- Runtime: 3-6 hours with 2 GPUs

#### Comprehensive Script (`run_all_experiments.sh`)
```bash
./src/run_all_experiments.sh
```
- 5 model sizes × 6 dataset sizes × 4 experiment types = 120 experiments
- Runs: standard, linear, nonlinear, and combined transformed
- Runtime: 4-8 hours (depending on GPU configuration)

## GPU Selection

### For Single-GPU Scripts

By default, single-GPU scripts use GPU 0. To use a different GPU:

```bash
# Use GPU 2
GPU_DEVICE=2 ./run_full.sh

# Use GPU 3
GPU_DEVICE=3 ./run_quick_test.sh
```

### For Multi-GPU Scripts

By default, multi-GPU scripts use GPUs 0 and 1. To use different GPUs:

```bash
# Use GPUs 2 and 3
GPU_DEVICES=2,3 ./run_full_multi_gpu.sh

# Use all 4 GPUs
GPU_DEVICES=0,1,2,3 ./run_full_multi_gpu.sh

# Use only GPU 1 and 3
GPU_DEVICES=1,3 ./run_quick_test_multi_gpu.sh
```

## Batch Size Recommendations

When using multiple GPUs, you can use larger batch sizes for better efficiency:

- **Single GPU**: batch size 4-32 (depending on model size and GPU memory)
- **2 GPUs**: batch size 16-64
- **4 GPUs**: batch size 32-128

The multi-GPU scripts are pre-configured with appropriate batch sizes.

## Checking Available GPUs

To see what GPUs are available on your system:

```bash
# List all GPUs
nvidia-smi

# Show GPU details
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
```

## Performance Expectations

Multi-GPU training provides near-linear speedup for data-parallel workloads:

- **2 GPUs**: ~1.8-1.9x speedup
- **4 GPUs**: ~3.5-3.7x speedup

Actual speedup depends on:
- Model size (larger models scale better)
- Batch size (larger batches scale better)
- GPU interconnect speed
- Data loading efficiency

## Example Usage

### Quick Test on Specific GPUs

```bash
# Test on GPUs 1 and 2
GPU_DEVICES=1,2 ./run_quick_test_multi_gpu.sh
```

### Full Experiment on Single GPU

```bash
# Run on GPU 3
GPU_DEVICE=3 ./run_full.sh
```

### Standard Experiments on Multiple GPUs

```bash
# Run on GPUs 0,1,2,3
GPU_DEVICES=0,1,2,3 ./run_standard_multi_gpu.sh
```

## Monitoring GPU Usage

During training, you can monitor GPU usage:

```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi

# Or use nvidia-ml-py for more detailed monitoring
nvidia-smi dmon -s um
```

## Troubleshooting

### Out of Memory Errors

If you encounter CUDA out of memory errors:

1. Reduce batch size
2. Use gradient accumulation
3. Use fewer GPUs with smaller batch size per GPU

### RuntimeError: grad can be implicitly created only for scalar outputs

This error can occur with DataParallel when the loss is not properly reduced to a scalar. The codebase handles this automatically by ensuring the loss is averaged across GPUs before calling backward(). If you encounter this error with custom models, ensure your model returns a scalar loss or apply `.mean()` to the loss before calling `.backward()`.

### Uneven GPU Usage

DataParallel may show uneven GPU usage (GPU 0 typically has higher usage). This is normal as GPU 0 handles the loss computation and gradient aggregation.

### Slow Data Loading

If GPUs are waiting for data:

1. Increase number of data loader workers
2. Use a faster storage system
3. Consider data prefetching

## Command Line Options

The main script supports these GPU-related options:

```bash
python main.py \
    --multi-gpu          # Enable multi-GPU training (default: True)
    --no-multi-gpu       # Disable multi-GPU training
    --device cuda        # Device to use (default: cuda)
```

### Experiment Type Options

Control which experiment types to run:

```bash
python main.py \
    --run-standard       # Run standard experiments (default: True)
    --run-transformed    # Run combined transform experiments (default: True)
    --run-linear         # Run linear transform experiments (default: True)
    --run-nonlinear      # Run nonlinear transform experiments (default: True)
    --skip-standard      # Skip standard experiments
    --skip-transformed   # Skip combined transform experiments  
    --skip-linear        # Skip linear transform experiments
    --skip-nonlinear     # Skip nonlinear transform experiments
```

### Example Usage

```bash
# Run only standard and linear experiments
python main.py --run-standard --run-linear --skip-transformed --skip-nonlinear

# Run only nonlinear experiments
python main.py --run-nonlinear --skip-standard --skip-transformed --skip-linear

# Run all experiment types (default behavior)
python main.py --run-standard --run-transformed --run-linear --run-nonlinear
```

## Output Structure

After running experiments, you'll find the following in your output directory:

### Results Files
- `results/scaling_law_results.json` - Raw results in JSON format
- `results/scaling_law_results.csv` - Results in CSV format for analysis

### Generated Plots

The system generates separate plots for each experiment type:

#### Individual Experiment Type Plots
- `standard_loss_vs_parameters.png` - Standard experiments: loss vs model parameters
- `standard_loss_vs_dataset_size.png` - Standard experiments: loss vs dataset size  
- `standard_loss_vs_compute.png` - Standard experiments: loss vs compute
- `standard_optimal_model_size.png` - Standard experiments: optimal model size vs compute

- `linear_*.png` - Same plots for linear transform experiments
- `nonlinear_*.png` - Same plots for nonlinear transform experiments  
- `transformed_*.png` - Same plots for combined transform experiments

#### Comparison Plots
- `comparison.png` - Standard vs Combined transforms comparison
- `comparison_linear.png` - Standard vs Linear transforms comparison
- `comparison_nonlinear.png` - Standard vs Nonlinear transforms comparison

Each comparison plot includes:
1. Loss by model size comparison
2. Loss by dataset size comparison  
3. Perplexity distribution comparison
4. Relative performance impact analysis

## Best Practices

1. **Start with Quick Test**: Always run the quick test first to ensure your setup works
2. **Monitor First Few Steps**: Watch GPU usage and memory in the first few training steps
3. **Use Appropriate Batch Sizes**: Larger batches with more GPUs, but stay within memory limits
4. **Check Load Balancing**: Ensure all GPUs are being utilized effectively
5. **Save Checkpoints**: Multi-GPU training can be interrupted; regular checkpoints help resume 