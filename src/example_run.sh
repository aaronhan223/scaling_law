#!/bin/bash

# Example commands for running scaling law experiments

echo "Scaling Law Experiments - Example Commands"
echo "=========================================="
echo ""

echo "1. Quick test (smallest models and datasets):"
echo "python main.py --model-sizes 0 --dataset-sizes 1M --no-wandb --eval-steps 100"
echo ""

echo "2. Small-scale experiment:"
echo "python main.py --model-sizes tiny,small --dataset-sizes 1M,10M --no-wandb"
echo ""

echo "3. Medium-scale standard experiment:"
echo "python main.py --model-sizes small,medium --dataset-sizes 1M,10M,100M --skip-transformed"
echo ""

echo "4. Transformed data experiment:"
echo "python main.py --model-sizes small --dataset-sizes 1M,10M --skip-standard"
echo ""

echo "5. Full experiment with W&B logging:"
echo "python main.py --model-sizes small,medium,large --dataset-sizes 1M,10M,100M --wandb-project my-scaling-laws"
echo ""

echo "6. Generate plots from existing results:"
echo "python main.py --plot-only --output-dir ./outputs/scaling_laws_20240101_120000"
echo ""

echo "Choose a command above or customize your own!"

# Uncomment one of the following to run:

# Quick test
# python main.py --model-sizes 0 --dataset-sizes 1M --no-wandb --eval-steps 100

# Small experiment  
# python main.py --model-sizes tiny,small --dataset-sizes 1M,10M --no-wandb 