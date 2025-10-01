#!/bin/bash

# Cross-Domain Scaling Law Prediction Experiment Runner
# 
# This script runs the cross-domain scaling law prediction experiment.
# It establishes scaling laws on WikiText-103 (source domain) and predicts
# scaling laws for PubMed OA abstracts (target domain) based on information
# content analysis.
#
# Usage: ./run_cross_domain_experiment.sh [OPTIONS]
#
# Environment variables:
# - GPU_DEVICES: Comma-separated list of GPU device IDs (default: 0)
# - DATASET_SIZES: Dataset sizes to test (default: 1K,10K,100K)
# - MODEL_SIZES: Model size indices to test (default: 0,1,2)
# - BATCH_SIZE: Training batch size (default: 4)
# - NO_WANDB: Set to 1 to disable WandB logging

set -e  # Exit on any error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default configuration
DEFAULT_GPU_DEVICES="0,1,2,3"
DEFAULT_DATASET_SIZES="1K,10K,100K"
DEFAULT_MODEL_SIZES="0,1,2"  
DEFAULT_BATCH_SIZE="4"
DEFAULT_LEARNING_RATE="5e-4"
DEFAULT_SEED="42"

# Read from environment or use defaults
GPU_DEVICES="${GPU_DEVICES:-$DEFAULT_GPU_DEVICES}"
DATASET_SIZES="${DATASET_SIZES:-$DEFAULT_DATASET_SIZES}"
MODEL_SIZES="${MODEL_SIZES:-$DEFAULT_MODEL_SIZES}"
BATCH_SIZE="${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}"
LEARNING_RATE="${LEARNING_RATE:-$DEFAULT_LEARNING_RATE}"
SEED="${SEED:-$DEFAULT_SEED}"
NO_WANDB="${NO_WANDB:-0}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu-devices)
            GPU_DEVICES="$2"
            shift 2
            ;;
        --dataset-sizes)
            DATASET_SIZES="$2"
            shift 2
            ;;
        --model-sizes)
            MODEL_SIZES="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --no-wandb)
            NO_WANDB=1
            shift
            ;;
        --help|-h)
            echo "Cross-Domain Scaling Law Prediction Experiment"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --gpu-devices DEVICES     Comma-separated GPU device IDs (default: $DEFAULT_GPU_DEVICES)"
            echo "  --dataset-sizes SIZES     Dataset sizes to test (default: $DEFAULT_DATASET_SIZES)"
            echo "  --model-sizes INDICES     Model size indices (default: $DEFAULT_MODEL_SIZES)"
            echo "  --batch-size SIZE         Training batch size (default: $DEFAULT_BATCH_SIZE)"
            echo "  --learning-rate RATE      Learning rate (default: $DEFAULT_LEARNING_RATE)"
            echo "  --seed SEED               Random seed (default: $DEFAULT_SEED)"
            echo "  --no-wandb                Disable WandB logging"
            echo "  --help, -h                Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  GPU_DEVICES, DATASET_SIZES, MODEL_SIZES, BATCH_SIZE,"
            echo "  LEARNING_RATE, SEED, NO_WANDB"
            echo ""
            echo "Examples:"
            echo "  # Quick test with small datasets"
            echo "  $0 --dataset-sizes 1K,10K --model-sizes 0,1 --batch-size 2"
            echo ""
            echo "  # Full experiment with multiple GPUs"
            echo "  GPU_DEVICES=0,1,2,3 $0 --dataset-sizes 100K,1M,10M --model-sizes 0,1,2,3"
            echo ""
            echo "  # Disable WandB logging"
            echo "  $0 --no-wandb"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set CUDA devices
export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"

# Print configuration
echo "=================================="
echo "CROSS-DOMAIN SCALING LAW EXPERIMENT"
echo "=================================="
echo "Configuration:"
echo "  GPU devices: $GPU_DEVICES"
echo "  Dataset sizes: $DATASET_SIZES" 
echo "  Model sizes: $MODEL_SIZES"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Seed: $SEED"
echo "  WandB logging: $([ "$NO_WANDB" = "1" ] && echo "disabled" || echo "enabled")"
echo "=================================="

# Check if Python script exists
if [ ! -f "cross_domain_scaling_experiment.py" ]; then
    echo "Error: cross_domain_scaling_experiment.py not found in current directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Check Python environment
echo "Checking Python environment..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python3 -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python3 -c "import datasets; print(f'Datasets version: {datasets.__version__}')"

echo "Available GPUs:"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

echo ""
echo "Starting cross-domain scaling law experiment..."
echo "This will:"
echo "1. Establish scaling laws on WikiText-103 (source domain)"
echo "2. Measure information content of WikiText-103 and PubMed"
echo "3. Predict PubMed scaling laws based on information distortion"
echo "4. Validate predictions by training on PubMed"
echo ""

# Build command
CMD="python3 cross_domain_scaling_experiment.py"
CMD="$CMD --dataset-sizes $DATASET_SIZES"
CMD="$CMD --model-sizes $MODEL_SIZES"
CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --learning-rate $LEARNING_RATE"
CMD="$CMD --seed $SEED"

if [ "$NO_WANDB" = "1" ]; then
    CMD="$CMD --no-wandb"
fi

# Create timestamp for this run
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
OUTPUT_DIR="./outputs/cross_domain_experiment/cross_domain_scaling_$TIMESTAMP"

CMD="$CMD --output-dir ./outputs/cross_domain_experiment"
CMD="$CMD --experiment-name cross_domain_scaling"

echo "Running command: $CMD"
echo ""

# Run the experiment
start_time=$(date +%s)

if ! eval $CMD; then
    echo ""
    echo "❌ Cross-domain experiment failed!"
    exit 1
fi

end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo "=================================="
echo "CROSS-DOMAIN EXPERIMENT COMPLETE"
echo "=================================="
echo "Duration: ${duration}s ($((duration/60))m $((duration%60))s)"
echo "Results saved to: $OUTPUT_DIR"

# Try to show brief summary if results exist
if [ -d "$OUTPUT_DIR" ] && [ -f "$OUTPUT_DIR/cross_domain_results.csv" ]; then
    echo ""
    echo "Brief Results Summary:"
    python3 -c "
import pandas as pd
import sys
try:
    df = pd.read_csv('$OUTPUT_DIR/cross_domain_results.csv')
    if len(df) > 0:
        result = df.iloc[0]
        print(f'  Source Domain: {result.get(\"source_domain\", \"N/A\")}')
        print(f'  Target Domain: {result.get(\"target_domain\", \"N/A\")}')
        print(f'  Prediction Accuracy: {result.get(\"prediction_accuracy\", 0)*100:.1f}%')
        print(f'  Alpha Error: {result.get(\"alpha_rel_error\", 0)*100:.1f}%')
        print(f'  Beta Error: {result.get(\"beta_rel_error\", 0)*100:.1f}%')
        print(f'  Gamma Error: {result.get(\"gamma_rel_error\", 0)*100:.1f}%')
    else:
        print('  No results found in CSV')
except Exception as e:
    print(f'  Could not load results: {e}')
" 2>/dev/null || echo "  Could not load results summary"
fi

echo ""
echo "✅ Cross-domain scaling law prediction experiment completed successfully!"

# Check if we should show next steps
echo ""
echo "Next Steps:"
echo "  1. Review detailed results in: $OUTPUT_DIR/cross_domain_results.json"
echo "  2. Analyze prediction accuracy and information distortion metrics"
echo "  3. Consider running with larger datasets or different domains"
echo "  4. Visualize results with plotting scripts"
echo ""

# Show disk usage
if command -v du >/dev/null 2>&1; then
    total_size=$(du -sh ./outputs 2>/dev/null | cut -f1 || echo "unknown")
    echo "Total output directory size: $total_size"
fi

echo "Experiment completed at: $(date)"