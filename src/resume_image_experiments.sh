#!/bin/bash

# Resume Image Scaling Law Experiments
# Continues from where the previous experiment run left off in image_scaling_laws_20250916_074801

echo "=========================================="
echo "RESUMING IMAGE SCALING LAW EXPERIMENTS"
echo "=========================================="
echo ""
echo "Original experiment directory: image_scaling_laws_20250916_074801"
echo "This script will continue the incomplete experiments:"
echo ""
echo "Missing experiments identified:"
echo "1. Standard experiments:"
echo "   - vit_large/data_300000 (1 experiment)"
echo ""
echo "2. Nonlinear experiments:"
echo "   - vit_small/data_300000 (1 experiment)"
echo "   - vit_base/* (6 experiments)"
echo "   - vit_large/* (6 experiments)"
echo ""
echo "3. Quantization experiments:"
echo "   - All models, all data sizes (24 experiments)"
echo ""
echo "Total missing: 38 experiments"
echo "=========================================="
echo ""

# Check available GPUs
echo "Checking available GPUs..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null || echo "No NVIDIA GPUs detected"
echo ""

# Check ImageNet data availability
IMAGENET_PATH="/cis/home/xhan56/imagenet"
if [ ! -d "$IMAGENET_PATH" ]; then
    echo "Error: ImageNet data not found at $IMAGENET_PATH"
    echo "Please ensure ImageNet data is properly mounted/available."
    exit 1
fi
echo "ImageNet data found at: $IMAGENET_PATH"
echo ""

# Default GPU configuration
GPU_DEVICES="${GPU_DEVICES:-0,1,2,3}"
echo "Using GPUs: $GPU_DEVICES"
echo ""

# Set the output directory to continue in the same location
OUTPUT_DIR="./outputs/image_scaling_laws_20250916_074801"
echo "Continuing experiments in: $OUTPUT_DIR"
echo ""

# Function to run specific experiments
run_experiment() {
    local exp_type=$1
    local model_sizes=$2
    local dataset_sizes=$3
    local description=$4

    echo "=========================================="
    echo "Running $description"
    echo "Model sizes: $model_sizes"
    echo "Dataset sizes: $dataset_sizes"
    echo "=========================================="

    # Activate conda environment and build the command
    local cmd="CUDA_VISIBLE_DEVICES=$GPU_DEVICES python main_image.py \
        --experiment-name image_scaling_laws \
        --output-dir $OUTPUT_DIR \
        --model-sizes $model_sizes \
        --dataset-sizes $dataset_sizes \
        --batch-size 128 \
        --eval-batch-size 128 \
        --num-epochs 100 \
        --learning-rate 1e-4 \
        --img-size 224 \
        --num-workers 8 \
        --seed 42 \
        --wandb-project image-scaling-laws"

    # Add experiment type flags
    case $exp_type in
        "standard")
            cmd="$cmd --run-standard --skip-linear --skip-nonlinear --skip-quantization --skip-lowrank"
            ;;
        "nonlinear")
            cmd="$cmd --skip-standard --skip-linear --run-nonlinear --skip-quantization --skip-lowrank"
            ;;
        "quantization")
            cmd="$cmd --skip-standard --skip-linear --skip-nonlinear --run-quantization --skip-lowrank"
            ;;
    esac

    echo "Command: $cmd"
    echo ""

    # Execute the command
    eval $cmd

    if [ $? -eq 0 ]; then
        echo "$description completed successfully!"
    else
        echo "Error: $description failed!"
        return 1
    fi
    echo ""
}

# Phase 1: Complete remaining standard experiments
echo "PHASE 1: Completing standard experiments..."
echo "Missing: vit_large with 300K dataset"
run_experiment "standard" "3" "300K" "Standard Experiments - VIT Large 300K"

if [ $? -ne 0 ]; then
    echo "Phase 1 failed. Stopping execution."
    exit 1
fi

# Phase 2: Complete remaining nonlinear experiments
echo "PHASE 2: Completing nonlinear experiments..."

echo "Running nonlinear - vit_small with 300K dataset..."
run_experiment "nonlinear" "1" "100K,300K" "Nonlinear Experiments - VIT Small 300K"

if [ $? -ne 0 ]; then
    echo "Phase 2a failed. Stopping execution."
    exit 1
fi

echo "Running nonlinear - vit_base with all dataset sizes..."
run_experiment "nonlinear" "2" "1K,3K,10K,30K,100K,300K" "Nonlinear Experiments - VIT Base All Sizes"

if [ $? -ne 0 ]; then
    echo "Phase 2b failed. Stopping execution."
    exit 1
fi

echo "Running nonlinear - vit_large with all dataset sizes..."
run_experiment "nonlinear" "3" "1K,3K,10K,30K,100K,300K" "Nonlinear Experiments - VIT Large All Sizes"

if [ $? -ne 0 ]; then
    echo "Phase 2c failed. Stopping execution."
    exit 1
fi

# Phase 3: Run all quantization experiments
echo "PHASE 3: Running all quantization experiments..."
echo "Running quantization experiments for all models and dataset sizes..."
run_experiment "quantization" "0,1,2,3" "1K,3K,10K,30K,100K,300K" "Quantization Experiments - All Models All Sizes"

if [ $? -ne 0 ]; then
    echo "Phase 3 failed. Stopping execution."
    exit 1
fi

echo ""
echo "=========================================="
echo "RESUME EXPERIMENTS COMPLETED SUCCESSFULLY!"
echo "=========================================="
echo ""
echo "All missing experiments have been completed."
echo "Results and plots are saved in: $OUTPUT_DIR"
echo ""
echo "You can now run the final aggregation and plotting script:"
echo "python aggregate_and_plot_results.py --input-dir $OUTPUT_DIR"
echo ""
echo "=========================================="