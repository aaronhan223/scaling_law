#!/bin/bash

# Standard scaling law experiments with multi-GPU support
# Using 6 model sizes and 6 dataset sizes for robust power law fitting

echo "=========================================="
echo "MULTI-GPU STANDARD SCALING LAW EXPERIMENTS"
echo "=========================================="
echo ""
echo "Running comprehensive standard scaling law experiments using multiple GPUs..."
echo "Model sizes: 6 configurations (tiny to large)"
echo "Dataset sizes: 6 configurations (1M to 300M tokens)"
echo ""

# Check available GPUs
echo "Checking available GPUs..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# Default GPU configuration
GPU_DEVICES="${GPU_DEVICES:-0,1}"  # Default to GPUs 0 and 1

echo "Using GPUs: $GPU_DEVICES"
echo "Expected runtime: 2-4 hours with 2 GPUs depending on hardware"
echo "=========================================="
echo ""

CUDA_VISIBLE_DEVICES=$GPU_DEVICES python main.py \
    --experiment-name scaling_laws_standard_multi_gpu \
    --model-sizes 0,1,2,3,4,5 \
    --dataset-sizes 1M,3M,10M,30M,100M,300M \
    --skip-transformed \
    --wandb-project scaling-laws-standard \
    --batch-size 64 \
    --eval-steps 500 \
    --warmup-steps 1000 \
    --multi-gpu

echo ""
echo "=========================================="
echo "MULTI-GPU STANDARD EXPERIMENTS COMPLETE!"
echo "==========================================" 