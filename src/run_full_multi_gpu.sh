#!/bin/bash

# Full scaling law experiments with multi-GPU support
# Uses multiple GPUs to accelerate training
# This will run experiments with standard, linear transform, and nonlinear transform settings

echo "=========================================="
echo "MULTI-GPU COMPREHENSIVE SCALING LAW EXPERIMENTS"
echo "=========================================="
echo ""
echo "This script will run experiments using multiple GPUs:"
echo "- Standard experiments: 6 model sizes × 6 dataset sizes = 36 experiments"
echo "- Linear transform experiments: 6 model sizes × 6 dataset sizes = 36 experiments"
echo "- Nonlinear transform experiments: 6 model sizes × 6 dataset sizes = 36 experiments"
echo "- Total: 108 experiments"
echo ""
echo "Model configurations (with reduced dataset sizes for memory efficiency):"
echo "  0: ~700K params  (128 hidden, 2 layers, 4 heads)"
echo "  1: ~1.3M params  (256 hidden, 4 layers, 8 heads)"
echo "  2: ~3.9M params  (512 hidden, 6 layers, 8 heads)"
echo "  3: ~10M params   (512 hidden, 8 layers, 8 heads)"
echo "  4: ~30M params   (640 hidden, 10 layers, 10 heads)"
echo "  5: ~125M params  (768 hidden, 12 layers, 12 heads)"
echo ""
echo "Dataset sizes (very large scale): 10M, 30M, 100M, 300M, 1B, 3B tokens"
echo ""
echo "Experiment types:"
echo "  - Standard: No transformations applied"
echo "  - Linear: Only linear token transformations applied"
echo "  - Nonlinear: Only nonlinear token transformations applied"
echo ""

# Check available GPUs
echo "Checking available GPUs..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# Default GPU configuration - modify based on your setup
# Examples:
# - Use GPUs 0 and 1: CUDA_VISIBLE_DEVICES=0,1
# - Use GPUs 2 and 3: CUDA_VISIBLE_DEVICES=2,3
# - Use all 4 GPUs: CUDA_VISIBLE_DEVICES=0,1,2,3
GPU_DEVICES="${GPU_DEVICES:-0,1}"  # Default to GPUs 0 and 1, override with environment variable

echo "Using GPUs: $GPU_DEVICES"
echo ""
echo "To use different GPUs, run: GPU_DEVICES=2,3 ./run_full_multi_gpu.sh"
echo ""
echo "Expected total runtime: 24-48 hours with 2 GPUs (5 model sizes, 3 experiment types, very large datasets)"
echo "=========================================="
echo ""

echo "Starting multi-GPU comprehensive scaling law experiments..."

# Run with specified GPUs
# Note: With multi-GPU, we can use larger batch sizes for better efficiency
# Running standard, linear, and nonlinear experiments (skipping combined transformed)
CUDA_VISIBLE_DEVICES=$GPU_DEVICES python main.py \
    --experiment-name scaling_laws_full_multi_gpu \
    --model-sizes 0,1,2,3,4 \
    --dataset-sizes 3M,10M,30M,100M,300M,1B,3B \
    --dataset-name openwebtext \
    --wandb-project scaling-laws-r42 \
    --batch-size 4 \
    --eval-steps 500 \
    --warmup-steps 1000 \
    --learning-rate 5e-4 \
    --multi-gpu \
    --run-standard \
    --run-linear \
    --run-nonlinear \
    --run-clustering \
    --run-modular \
    --run-quantization \
    --skip-transformed 

echo ""
echo "=========================================="
echo "MULTI-GPU EXPERIMENTS COMPLETE!"
echo "=========================================="
echo "Check the outputs directory for results and plots."
echo "Generated plots will include:"
echo "  - standard_*.png: Standard experiment plots"
echo "  - linear_*.png: Linear transform experiment plots" 
echo "  - nonlinear_*.png: Nonlinear transform experiment plots"
echo "  - comparison_linear.png: Standard vs Linear comparison"
echo "  - comparison_nonlinear.png: Standard vs Nonlinear comparison" 