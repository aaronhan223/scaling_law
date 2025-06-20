#!/bin/bash

# Quick test of scaling law experiments with multi-GPU support
# Uses multiple GPUs to validate the experimental setup quickly

echo "=========================================="
echo "MULTI-GPU QUICK SCALING LAW TEST"
echo "=========================================="
echo ""
echo "This is a reduced-scope test using multiple GPUs:"
echo "- Standard experiments: 6 model sizes × 6 dataset sizes = 36 experiments"
echo "- Transformed experiments: 6 model sizes × 6 dataset sizes = 36 experiments"
echo "- Total: 72 experiments (but with smaller datasets for speed)"
echo ""
echo "Model configurations:"
echo "  0: ~7M params   (128 hidden, 2 layers, 4 heads)"
echo "  1: ~13M params  (256 hidden, 4 layers, 8 heads)"
echo "  2: ~39M params  (512 hidden, 6 layers, 8 heads)"
echo "  3: ~125M params (768 hidden, 12 layers, 12 heads)"
echo "  4: ~355M params (1024 hidden, 24 layers, 16 heads)"
echo "  5: ~775M params (1536 hidden, 36 layers, 16 heads)"
echo ""
echo "Dataset sizes (reduced): 100K, 300K, 1M, 3M, 10M, 30M tokens"
echo ""

# Check available GPUs
echo "Checking available GPUs..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# Default GPU configuration
GPU_DEVICES="${GPU_DEVICES:-0,1}"  # Default to GPUs 0 and 1

echo "Using GPUs: $GPU_DEVICES"
echo ""
echo "Expected total runtime: 1-2 hours with 2 GPUs (depending on hardware)"
echo "=========================================="
echo ""

# Run with specified GPUs
CUDA_VISIBLE_DEVICES=$GPU_DEVICES python main.py \
    --experiment-name scaling_laws_quick_test_multi_gpu \
    --model-sizes 0,1,2,3,4,5 \
    --dataset-sizes 100K,300K,1M,3M,10M,30M \
    --wandb-project scaling-laws-test \
    --batch-size 64 \
    --eval-steps 200 \
    --warmup-steps 500 \
    --learning-rate 5e-4 \
    --multi-gpu

echo ""
echo "=========================================="
echo "MULTI-GPU QUICK TEST COMPLETE!"
echo "=========================================="
echo "Check the outputs directory for results and plots."
echo "If results look good, run the full experiment with run_full_multi_gpu.sh" 