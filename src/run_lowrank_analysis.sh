#!/bin/bash

# Low-rank Analysis Script
# Tests the relationship between low-rank ratio and scaling law parameters

echo "=========================================="
echo "LOW-RANK TRANSFORMATION ANALYSIS"
echo "=========================================="
echo ""
echo "This script analyzes the relationship between low-rank transformation"
echo "significance (low-rank ratio) and the fitted scaling law parameters"
echo "(alpha, beta, gamma)."
echo ""
echo "Analysis details:"
echo "  - Tests low-rank ratios from 0.01 to 0.5"
echo "  - Uses 3 small model sizes for efficiency"
echo "  - Dataset sizes: 100K, 1M, 10M tokens"
echo "  - Generates comprehensive plots and analysis"
echo ""

# Check available GPUs
echo "Checking available GPUs..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null || echo "No NVIDIA GPUs detected"
echo ""

# GPU configuration
GPU_DEVICES="${GPU_DEVICES:-0,1,2,3}"  # Default to GPU 0, override with environment variable

echo "Using GPUs: $GPU_DEVICES"
echo ""
echo "To use different GPUs, run: GPU_DEVICES=1 ./run_lowrank_analysis.sh"
echo ""

# Configuration options
DATASET_NAME="${DATASET_NAME:-wikitext}"
DATASET_CONFIG="${DATASET_CONFIG:-wikitext-103-v1}"
MIN_RATIO="${MIN_RATIO:-0.01}"
MAX_RATIO="${MAX_RATIO:-0.3}"
NUM_RATIOS="${NUM_RATIOS:-8}"
MODEL_SIZES="${MODEL_SIZES:-0,1,2,3,4}"
DATASET_SIZES="${DATASET_SIZES:-100K,300K,1M,3M,10M,30M,100M}"
BATCH_SIZE="${BATCH_SIZE:-4}"

echo "Configuration:"
echo "  Dataset: $DATASET_NAME ($DATASET_CONFIG)"
echo "  Low-rank ratios: $MIN_RATIO to $MAX_RATIO ($NUM_RATIOS ratios)"
echo "  Model sizes: $MODEL_SIZES"
echo "  Dataset sizes: $DATASET_SIZES"
echo "  Batch size: $BATCH_SIZE"
echo ""
echo "Expected runtime: 2-4 hours (depending on GPU and configuration)"
echo "=========================================="
echo ""

echo "Starting low-rank analysis..."

# Set memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Memory optimization settings:"
echo "  PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo ""

# Run the low-rank analysis
CUDA_VISIBLE_DEVICES=$GPU_DEVICES python lowrank_analysis.py \
    --min-ratio $MIN_RATIO \
    --max-ratio $MAX_RATIO \
    --num-ratios $NUM_RATIOS \
    --model-sizes $MODEL_SIZES \
    --dataset-sizes $DATASET_SIZES \
    --dataset-name $DATASET_NAME \
    --dataset-config $DATASET_CONFIG \
    --batch-size $BATCH_SIZE \
    --learning-rate 5e-4 \
    --wandb-project lowrank-analysis \
    --experiment-name lowrank_analysis

echo ""
echo "=========================================="
echo "LOW-RANK ANALYSIS COMPLETE!"
echo "=========================================="
echo "Check the outputs/lowrank_analysis directory for:"
echo "  - lowrank_analysis_results.csv: Raw results data"
echo "  - lowrank_analysis_results.json: Results in JSON format"
echo "  - plots/: Generated visualization plots"
echo ""
echo "Generated plots include:"
echo "  - scaling_parameters_vs_lowrank.png: Main analysis plot"
echo "  - loss_distribution_vs_lowrank.png: Loss distribution analysis"
echo "  - lowrank_heatmap.png: Correlation analysis"
echo "  - lowrank_summary.png: Comprehensive summary"
echo ""
echo "Analysis shows how low-rank ratio affects:"
echo "  - α (alpha): Model size scaling exponent"
echo "  - β (beta): Dataset size scaling exponent" 
echo "  - γ (gamma): Compute scaling exponent"
echo "=========================================="