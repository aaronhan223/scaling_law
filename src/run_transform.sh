#!/bin/bash

# Transformed data scaling law experiments with comprehensive scaling
# Using 6 model sizes and 6 dataset sizes to test scaling laws under distribution shift

echo "Running comprehensive transformed data scaling law experiments..."
echo "Model sizes: 6 configurations (tiny to large)"
echo "Dataset sizes: 6 configurations (1M to 300M tokens)"
echo "Transformations: Random linear + non-linear (tanh)"
echo "Expected runtime: Several hours depending on hardware"

python main.py \
    --experiment-name scaling_laws_transform_comprehensive \
    --model-sizes 0,1,2,3,4,5 \
    --dataset-sizes 1M,3M,10M,30M,100M,300M \
    --skip-standard \
    --wandb-project scaling-laws-transform \
    --batch-size 32 \
    --eval-steps 500 \
    --warmup-steps 1000