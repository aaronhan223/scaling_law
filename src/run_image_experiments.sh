#!/bin/bash

# Image Scaling Law Experiments on ImageNet using Vision Transformers
# Uses Vision Transformer models of varying sizes to study scaling laws on ImageNet

echo "=========================================="
echo "IMAGE SCALING LAW EXPERIMENTS ON IMAGENET"
echo "=========================================="
echo ""
echo "This script runs scaling law experiments on ImageNet using Vision Transformers:"
echo "- Standard experiments: 3 model sizes × 3 dataset sizes = 9 experiments"
echo "- Linear transform experiments: 3 model sizes × 3 dataset sizes = 9 experiments"
echo "- Nonlinear transform experiments: 3 model sizes × 3 dataset sizes = 9 experiments"
echo "- Quantization transform experiments: 3 model sizes × 3 dataset sizes = 9 experiments"
echo "- Low-rank transform experiments: 3 model sizes × 3 dataset sizes = 9 experiments"
echo "- Total: 45 experiments"
echo ""
echo "Vision Transformer model configurations:"
echo "  0: Tiny ViT  (~5.5M params, 192 embed_dim, 12 layers, 3 heads)"
echo "  1: Small ViT (~22M params, 384 embed_dim, 12 layers, 6 heads)"
echo "  2: Base ViT  (~86M params, 768 embed_dim, 12 layers, 12 heads)"
echo "  3: Large ViT (~307M params, 1024 embed_dim, 24 layers, 16 heads)"
echo ""
echo "Dataset sizes: 1K, 10K, 50K samples from ImageNet"
echo ""
echo "Transform types:"
echo "  - Standard: No transformations applied to images"
echo "  - Linear: Linear transformations applied to image patches"
echo "  - Nonlinear: Non-linear neural network transformations"
echo "  - Quantization: Pixel-level quantization for information loss"
echo "  - Low-rank: Low-rank matrix decomposition of image features"
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
echo "Available splits: $(ls -1 $IMAGENET_PATH | tr '\n' ' ')"
echo ""

# Default GPU configuration - modify based on your setup
# Examples:
# - Use GPUs 0 and 1: GPU_DEVICES=0,1
# - Use GPUs 2 and 3: GPU_DEVICES=2,3
# - Use all 4 GPUs: GPU_DEVICES=0,1,2,3
GPU_DEVICES="${GPU_DEVICES:-0,1,2,3}"  # Default to all GPUs, override with environment variable

echo "Using GPUs: $GPU_DEVICES"
echo ""
echo "To use different GPUs, run: GPU_DEVICES=0,1 ./run_image_experiments.sh"
echo ""
echo "Expected total runtime: 2-8 hours with multiple GPUs (3 model sizes, 5 experiment types)"
echo "=========================================="
echo ""

echo "Starting image scaling law experiments..."

# Set memory optimization environment variables (if supported by PyTorch version)
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting image experiments with GPU optimization..."
echo ""

# Run with specified GPUs
# Note: With multi-GPU, we can use larger batch sizes for better efficiency
# Running all transform experiments for comprehensive analysis
CUDA_VISIBLE_DEVICES=$GPU_DEVICES python main_image.py \
    --experiment-name image_scaling_laws \
    --model-sizes 0,1,2,3 \
    --dataset-sizes 1K,3K,10K,30K,100K,300K \
    --batch-size 128 \
    --eval-batch-size 128 \
    --num-epochs 100 \
    --learning-rate 1e-4 \
    --img-size 224 \
    --num-workers 8 \
    --seed 42 \
    --wandb-project image-scaling-laws \
    --run-standard \
    --skip-linear \
    --run-nonlinear \
    --run-quantization \
    --skip-lowrank

echo ""
echo "=========================================="
echo "IMAGE SCALING LAW EXPERIMENTS COMPLETE!"
echo "=========================================="
echo "Check the outputs directory for results and plots."
echo "Generated plots will include:"
echo "  - standard_*.png: Standard experiment plots"
echo "  - linear_*.png: Linear transform experiment plots"
echo "  - nonlinear_*.png: Nonlinear transform experiment plots"
echo "  - quantization_*.png: Quantization transform experiment plots"
echo "  - lowrank_*.png: Low-rank transform experiment plots"
echo "  - image_comparison_*.png: Comparison plots between experiment types"
echo ""
echo "Analysis shows how model performance scales with:"
echo "  - Model size (parameters): Vision Transformer scaling"
echo "  - Dataset size (samples): ImageNet subset scaling"
echo "  - Transform effects: Impact of information-preserving/losing transforms"
echo "=========================================="