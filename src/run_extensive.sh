#!/bin/bash

# Extensive scaling law experiments - maximum coverage
# Uses all 7 model sizes and 7 dataset sizes for the most comprehensive results
# This will run 98 total experiments (49 standard + 49 transformed)

echo "=========================================="
echo "EXTENSIVE SCALING LAW EXPERIMENTS"
echo "=========================================="
echo ""
echo "This script will run the most comprehensive experiments:"
echo "- Standard experiments: 7 model sizes × 7 dataset sizes = 49 experiments"
echo "- Transformed experiments: 7 model sizes × 7 dataset sizes = 49 experiments"
echo "- Total: 98 experiments"
echo ""
echo "Model configurations (all 7):"
echo "  0: ~7M params   (128 hidden, 2 layers, 4 heads)"
echo "  1: ~13M params  (256 hidden, 4 layers, 8 heads)"
echo "  2: ~39M params  (512 hidden, 6 layers, 8 heads)"
echo "  3: ~125M params (768 hidden, 12 layers, 12 heads)"
echo "  4: ~355M params (1024 hidden, 24 layers, 16 heads)"
echo "  5: ~775M params (1536 hidden, 36 layers, 16 heads)"
echo "  6: ~1.5B params (2048 hidden, 48 layers, 32 heads)"
echo ""
echo "Dataset sizes: 1M, 3M, 10M, 30M, 100M, 300M, 1B tokens"
echo ""
echo "Expected total runtime: 12-24 hours (depending on hardware)"
echo "WARNING: This will require significant compute resources!"
echo "=========================================="
echo ""

read -p "Do you want to continue with extensive experiments? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Experiment cancelled."
    echo "Consider running run_full.sh for a more manageable experiment."
    exit 1
fi

echo "Starting extensive scaling law experiments..."
echo "This may take 12-24 hours to complete."

python main.py \
    --experiment-name scaling_laws_extensive \
    --model-sizes 0,1,2,3,4,5,6 \
    --dataset-sizes 1M,3M,10M,30M,100M,300M,1B \
    --wandb-project scaling-laws-extensive \
    --batch-size 32 \
    --eval-steps 500 \
    --warmup-steps 1000 \
    --learning-rate 5e-4

echo ""
echo "=========================================="
echo "EXTENSIVE EXPERIMENTS COMPLETE!"
echo "=========================================="
echo "Check the outputs directory for results and plots."
echo "This should provide the most comprehensive scaling law analysis." 