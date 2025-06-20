#!/bin/bash

# Full scaling law experiments with multi-GPU support
# Uses multiple GPUs to accelerate training
# This will run 72 total experiments (36 standard + 36 transformed)

echo "=========================================="
echo "MULTI-GPU COMPREHENSIVE SCALING LAW EXPERIMENTS"
echo "=========================================="
echo ""
echo "This script will run experiments using multiple GPUs:"
echo "- Standard experiments: 6 model sizes × 6 dataset sizes = 36 experiments"
echo "- Transformed experiments: 6 model sizes × 6 dataset sizes = 36 experiments"
echo "- Total: 72 experiments"
echo ""
echo "Model configurations:"
echo "  0: ~7M params   (128 hidden, 2 layers, 4 heads)"
echo "  1: ~13M params  (256 hidden, 4 layers, 8 heads)"
echo "  2: ~39M params  (512 hidden, 6 layers, 8 heads)"
echo "  3: ~125M params (768 hidden, 12 layers, 12 heads)"
echo "  4: ~355M params (1024 hidden, 24 layers, 16 heads)"
echo "  5: ~775M params (1536 hidden, 36 layers, 16 heads)"
echo ""
echo "Dataset sizes: 1M, 3M, 10M, 30M, 100M, 300M tokens"
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
echo "Expected total runtime: 3-6 hours with 2 GPUs (depending on hardware)"
echo "=========================================="
echo ""

echo "Starting multi-GPU comprehensive scaling law experiments..."

# Run with specified GPUs
# Note: With multi-GPU, we can use larger batch sizes for better efficiency
CUDA_VISIBLE_DEVICES=$GPU_DEVICES python main.py \
    --experiment-name scaling_laws_full_multi_gpu \
    --model-sizes 0,1,2,3,4,5 \
    --dataset-sizes 1M,3M,10M,30M,100M,300M \
    --wandb-project scaling-laws-comprehensive \
    --batch-size 4 \
    --eval-steps 500 \
    --warmup-steps 1000 \
    --learning-rate 5e-4 \
    --multi-gpu \
    --no-fp16

echo ""
echo "=========================================="
echo "MULTI-GPU EXPERIMENTS COMPLETE!"
echo "=========================================="
echo "Check the outputs directory for results and plots." 