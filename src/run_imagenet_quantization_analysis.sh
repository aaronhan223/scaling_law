#!/bin/bash

# ImageNet Quantization Analysis Script
# Tests the relationship between quantization levels and scaling law parameters on ImageNet data

echo "=================================================="
echo "IMAGENET QUANTIZATION TRANSFORMATION ANALYSIS"
echo "=================================================="
echo ""
echo "This script analyzes the relationship between quantization transformation"
echo "significance (quantization levels) and the fitted scaling law parameters"
echo "(alpha, beta, gamma) on ImageNet data."
echo ""
echo "Analysis details:"
echo "  - Tests quantization levels from 2 (1-bit) to 256 (8-bit)"
echo "  - Uses smaller vision models for computational efficiency"
echo "  - Dataset sizes: 1K, 10K, 50K samples"
echo "  - Generates comprehensive plots and analysis"
echo ""

# Check available GPUs
echo "Checking available GPUs..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null || echo "No NVIDIA GPUs detected"
echo ""

# GPU configuration
GPU_DEVICES="${GPU_DEVICES:-0,1,2,3}"  # Default to GPUs 0,1, override with environment variable

echo "Using GPUs: $GPU_DEVICES"
echo ""
echo "To use different GPUs, run: GPU_DEVICES=1,2 ./run_imagenet_quantization_analysis.sh"
echo ""

# Configuration options
MIN_LEVELS="${MIN_LEVELS:-2}"
MAX_LEVELS="${MAX_LEVELS:-256}"
NUM_LEVELS="${NUM_LEVELS:-4}"
MODEL_SIZES="${MODEL_SIZES:-0,1,2,3}"
DATASET_SIZES="${DATASET_SIZES:-1K,5K,10K,50K,100K}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
IMG_SIZE="${IMG_SIZE:-224}"
NUM_CLASSES="${NUM_CLASSES:-1000}"

echo "Configuration:"
echo "  Quantization levels: $MIN_LEVELS to $MAX_LEVELS ($NUM_LEVELS levels)"
echo "  Model sizes: $MODEL_SIZES"
echo "  Dataset sizes: $DATASET_SIZES"
echo "  Batch size: $BATCH_SIZE (eval: $EVAL_BATCH_SIZE)"
echo "  Number of epochs: $NUM_EPOCHS"
echo "  Image size: ${IMG_SIZE}x${IMG_SIZE}"
echo "  Number of classes: $NUM_CLASSES"
echo ""
echo "Expected runtime: 4-8 hours (depending on GPU and configuration)"
echo "=================================================="
echo ""

# Check for existing processes to prevent resource conflicts
EXISTING_PROCESSES=$(ps aux | grep -v grep | grep "imagenet_quantization_analysis.py" | wc -l)
if [ "$EXISTING_PROCESSES" -gt 0 ]; then
    echo "ERROR: Found $EXISTING_PROCESSES existing ImageNet quantization analysis processes running."
    echo "Please terminate existing processes before starting a new analysis:"
    echo "  pkill -f imagenet_quantization_analysis.py"
    echo ""
    echo "Current processes:"
    ps aux | grep -v grep | grep "imagenet_quantization_analysis.py" | head -5
    echo ""
    echo "Exiting to prevent resource conflicts..."
    exit 1
fi

echo "Starting ImageNet quantization analysis..."

# Set memory optimization environment variables (if supported)
# Note: expandable_segments may not be available in all PyTorch versions
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo "Memory optimization settings:"
echo "  PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo ""

# Check if ImageNet data exists
IMAGENET_PATH="/cis/home/xhan56/imagenet"
if [ ! -d "$IMAGENET_PATH" ]; then
    echo "ERROR: ImageNet data not found at $IMAGENET_PATH"
    echo "Please ensure ImageNet data is available at the expected location."
    exit 1
fi

echo "ImageNet data found at: $IMAGENET_PATH"
echo "Checking available data splits..."
ls -la "$IMAGENET_PATH"
echo ""

# Run the ImageNet quantization analysis
CUDA_VISIBLE_DEVICES=$GPU_DEVICES python imagenet_quantization_analysis.py \
    --min-levels $MIN_LEVELS \
    --max-levels $MAX_LEVELS \
    --num-levels $NUM_LEVELS \
    --model-sizes $MODEL_SIZES \
    --dataset-sizes $DATASET_SIZES \
    --batch-size $BATCH_SIZE \
    --eval-batch-size $EVAL_BATCH_SIZE \
    --learning-rate 1e-4 \
    --num-epochs $NUM_EPOCHS \
    --eval-steps 200 \
    --img-size $IMG_SIZE \
    --num-classes $NUM_CLASSES \
    --num-workers 4 \
    --wandb-project imagenet-quantization-analysis-r43 \
    --experiment-name imagenet_quantization_analysis_r43

echo ""
echo "=================================================="
echo "IMAGENET QUANTIZATION ANALYSIS COMPLETE!"
echo "=================================================="
echo "Check the outputs/imagenet_quantization_analysis directory for:"
echo "  - imagenet_quantization_analysis_results.csv: Raw results data"
echo "  - imagenet_quantization_analysis_results.json: Results in JSON format"
echo "  - plots/: Generated visualization plots"
echo ""
echo "Generated plots include:"
echo "  - scaling_parameters_vs_quantization.png: Main analysis plot"
echo "  - loss_distribution_vs_quantization.png: Loss distribution analysis"
echo "  - quantization_heatmap.png: Correlation analysis"
echo "  - quantization_summary.png: Comprehensive summary"
echo ""
echo "Analysis shows how quantization levels affect:"
echo "  - α (alpha): Model size scaling exponent"
echo "  - β (beta): Dataset size scaling exponent"
echo "  - γ (gamma): Compute/time scaling exponent"
echo ""
echo "Key insights:"
echo "  - Compare scaling behavior at different quantization levels"
echo "  - Identify optimal quantization for different model/data scales"
echo "  - Understand trade-offs between quantization and scaling laws"
echo "=================================================="