#!/bin/bash

# Full scaling law experiments - both standard and transformed data
# Comprehensive scaling with 6 model sizes and 6 dataset sizes each
# This will run 72 total experiments (36 standard + 36 transformed)

echo "=========================================="
echo "COMPREHENSIVE SCALING LAW EXPERIMENTS"
echo "=========================================="
echo ""
echo "This script will run:"
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
echo "Expected total runtime: 6-12 hours (depending on hardware)"
echo "=========================================="
echo ""

echo "Starting comprehensive scaling law experiments..."

python main.py \
    --experiment-name scaling_laws_full_comprehensive \
    --model-sizes 0,1,2,3,4,5 \
    --dataset-sizes 1M,3M,10M,30M,100M,300M \
    --wandb-project scaling-laws-comprehensive \
    --batch-size 32 \
    --eval-steps 500 \
    --warmup-steps 1000 \
    --learning-rate 5e-4

echo ""
echo "=========================================="
echo "EXPERIMENTS COMPLETE!"
echo "=========================================="
echo "Check the outputs directory for results and plots." 