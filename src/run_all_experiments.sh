#!/bin/bash

# Comprehensive scaling law experiments with all transformation types
# Runs standard, linear, nonlinear, and combined transformed experiments

echo "=========================================="
echo "COMPREHENSIVE SCALING LAW EXPERIMENTS"
echo "=========================================="
echo ""
echo "This script will run all experiment types:"
echo "- Standard experiments: 5 model sizes × 6 dataset sizes = 30 experiments"
echo "- Linear transform experiments: 5 model sizes × 6 dataset sizes = 30 experiments"
echo "- Nonlinear transform experiments: 5 model sizes × 6 dataset sizes = 30 experiments"
echo "- Combined transform experiments: 5 model sizes × 6 dataset sizes = 30 experiments"
echo "- Total: 120 experiments"
echo ""
echo "Model configurations:"
echo "  0: ~700K params  (128 hidden, 2 layers, 4 heads)"
echo "  1: ~1.3M params  (256 hidden, 4 layers, 8 heads)"
echo "  2: ~3.9M params  (512 hidden, 6 layers, 8 heads)"
echo "  3: ~10M params   (512 hidden, 8 layers, 8 heads)"
echo "  4: ~30M params   (640 hidden, 10 layers, 10 heads)"
echo ""
echo "Dataset sizes: 100K, 300K, 1M, 3M, 10M, 30M tokens"
echo ""
echo "Experiment types:"
echo "  - Standard: No transformations applied"
echo "  - Linear: Only linear token transformations applied"
echo "  - Nonlinear: Only nonlinear token transformations applied"
echo "  - Transformed: Both linear and nonlinear transformations applied"
echo ""

# Check available GPUs
echo "Checking available GPUs..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null || echo "No NVIDIA GPUs detected"
echo ""

# GPU configuration
GPU_DEVICES="${GPU_DEVICES:-0}"  # Default to GPU 0, override with environment variable

echo "Using GPUs: $GPU_DEVICES"
echo ""
echo "To use different GPUs, run: GPU_DEVICES=0,1 ./run_all_experiments.sh"
echo ""
echo "Expected total runtime: 4-8 hours (depending on GPU configuration)"
echo "=========================================="
echo ""

echo "Starting comprehensive scaling law experiments..."

# Run all experiment types
CUDA_VISIBLE_DEVICES=$GPU_DEVICES python main.py \
    --experiment-name comprehensive_scaling_laws \
    --model-sizes 0,1,2,3,4 \
    --dataset-sizes 100K,300K,1M,3M,10M,30M \
    --wandb-project scaling-laws-comprehensive \
    --batch-size 4 \
    --eval-steps 500 \
    --warmup-steps 1000 \
    --learning-rate 5e-4 \
    --multi-gpu \
    --run-standard \
    --run-linear \
    --run-nonlinear \
    --run-transformed

echo ""
echo "=========================================="
echo "COMPREHENSIVE EXPERIMENTS COMPLETE!"
echo "=========================================="
echo "Check the outputs directory for results and plots."
echo ""
echo "Generated plots will include:"
echo "  - standard_*.png: Standard experiment plots"
echo "  - linear_*.png: Linear transform experiment plots" 
echo "  - nonlinear_*.png: Nonlinear transform experiment plots"
echo "  - transformed_*.png: Combined transform experiment plots"
echo "  - comparison.png: Standard vs Combined comparison"
echo "  - comparison_linear.png: Standard vs Linear comparison"
echo "  - comparison_nonlinear.png: Standard vs Nonlinear comparison"
echo ""
echo "Results saved in CSV and JSON format for further analysis." 