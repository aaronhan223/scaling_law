#!/bin/bash

# LibriSpeech scaling law experiments with multi-GPU support
# Uses wav2vec 2.0 backbone for speech recognition
# This will run experiments with standard, linear transform, nonlinear transform, quantization, and low-rank transform settings

echo "=========================================="
echo "MULTI-GPU LIBRISPEECH SCALING LAW EXPERIMENTS"
echo "=========================================="
echo ""
echo "This script will run speech recognition scaling law experiments using LibriSpeech:"
echo "- Standard experiments: 5 model sizes × 4 dataset sizes = 20 experiments"
echo "- Linear transform experiments: 5 model sizes × 4 dataset sizes = 20 experiments"
echo "- Nonlinear transform experiments: 5 model sizes × 4 dataset sizes = 20 experiments"
echo "- Quantization transform experiments: 5 model sizes × 4 dataset sizes = 20 experiments"
echo "- Low-rank transform experiments: 5 model sizes × 4 dataset sizes = 20 experiments"
echo "- Total: 100 experiments"
echo ""
echo "Model configurations (Wav2Vec2-based):"
echo "  0: ~1.5M params   (256 hidden, 4 layers, 4 heads)"
echo "  1: ~5M params     (384 hidden, 6 layers, 6 heads)"
echo "  2: ~15M params    (512 hidden, 8 layers, 8 heads)"
echo "  3: ~30M params    (640 hidden, 10 layers, 10 heads)"
echo "  4: ~95M params    (768 hidden, 12 layers, 12 heads)"
echo ""
echo "Dataset sizes (LibriSpeech): 10h, 50h, 100h, 500h"
echo "Full 960h dataset used for final evaluation on test-clean and test-other"
echo ""
echo "Experiment types:"
echo "  - Standard: No transformations applied to CTC labels"
echo "  - Linear: Linear transformations applied to CTC labels"
echo "  - Nonlinear: Nonlinear transformations applied to CTC labels"
echo "  - Quantization: Quantization transformations applied to CTC labels"
echo "  - Low-rank: Low-rank square matrix transformations applied to CTC labels"
echo ""
echo "Evaluation metrics:"
echo "  - CTC Loss on dev-clean and test-clean"
echo "  - Word Error Rate (WER) on dev-clean and test-clean"
echo "  - Word Error Rate (WER) on test-other (noisy test set)"
echo ""

# Check available GPUs
echo "Checking available GPUs..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# Check if LibriSpeech data exists
LIBRISPEECH_ROOT="${LIBRISPEECH_ROOT:-/cis/home/xhan56/LibriSpeech}"
echo "Checking LibriSpeech data at: $LIBRISPEECH_ROOT"

if [ ! -d "$LIBRISPEECH_ROOT" ]; then
    echo "Error: LibriSpeech directory not found at $LIBRISPEECH_ROOT"
    echo "Please set LIBRISPEECH_ROOT environment variable or ensure data is at the default location"
    echo "Expected structure:"
    echo "  $LIBRISPEECH_ROOT/"
    echo "    train-clean-100/"
    echo "    train-clean-360/"
    echo "    train-other-500/"
    echo "    dev-clean/"
    echo "    dev-other/"
    echo "    test-clean/"
    echo "    test-other/"
    exit 1
fi

# Check required subdirectories
required_dirs=("train-clean-100" "train-clean-360" "train-other-500" "dev-clean" "test-clean")
for dir in "${required_dirs[@]}"; do
    if [ ! -d "$LIBRISPEECH_ROOT/$dir" ]; then
        echo "Warning: Required directory $LIBRISPEECH_ROOT/$dir not found"
    else
        echo "Found: $LIBRISPEECH_ROOT/$dir"
    fi
done
echo ""

# Default GPU configuration
GPU_DEVICES="${GPU_DEVICES:-0,1,2,3}"  # Default to GPUs 0 and 1

echo "Using GPUs: $GPU_DEVICES"
echo ""
echo "To use different GPUs, run: GPU_DEVICES=2,3 ./run_librispeech_experiments.sh"
echo ""

# Check if required Python packages are available
echo "Checking required packages..."
python -c "import torch; import torchaudio; import transformers; import jiwer; print('Core packages found')" || {
    echo "Error: Missing core packages. Please install:"
    echo "  pip install torch torchaudio transformers jiwer datasets"
    exit 1
}

# Check for audio backend support
echo "Checking audio backend support..."
python -c "
try:
    import soundfile
    print('soundfile found - FLAC support available')
except ImportError:
    print('Warning: soundfile not found. Installing for better FLAC support...')
    import subprocess
    subprocess.run(['pip', 'install', 'soundfile'], check=True)
    print('soundfile installed successfully')
" || {
    echo "Warning: Could not install soundfile. FLAC support may be limited."
    echo "You can manually install with: pip install soundfile"
}
echo ""

echo "Expected total runtime: 12-24 hours with 2 GPUs (speech experiments are computationally intensive)"
echo "Note: Speech experiments require more memory than text experiments due to audio processing"
echo "Note: LibriSpeech uses FLAC audio format which requires soundfile backend for torchaudio"
echo "Note: Datasets will be cached to disk after first run for faster subsequent experiments"
echo "=========================================="
echo ""

echo "Starting multi-GPU LibriSpeech scaling law experiments..."

# Check if we should run all experiments or just a subset
RUN_ALL="${RUN_ALL:-true}"
if [ "$RUN_ALL" = "true" ]; then
    echo "Running ALL experiment types (standard + all transforms)..."

    # Run comprehensive experiments with all transform types
    CUDA_VISIBLE_DEVICES=$GPU_DEVICES python main_speech.py \
        --experiment-name librispeech_scaling_laws_comprehensive \
        --librispeech-root "$LIBRISPEECH_ROOT" \
        --model-sizes 0,1,2,3 \
        --dataset-hours 10,50,100,500 \
        --wandb-project librispeech-scaling-laws \
        --batch-size 32 \
        --eval-steps 500 \
        --warmup-steps 1000 \
        --learning-rate 5e-4 \
        --max-audio-length 20.0 \
        --multi-gpu \
        --run-standard \
        --skip-linear \
        --run-nonlinear \
        --run-quantization \
        --skip-lowrank \
        --quantization-levels 8 \
        --lowrank-rank 16
else
    echo "Running STANDARD experiments only (use RUN_ALL=true for all transforms)..."

    # Run standard experiments only (faster for initial testing)
    CUDA_VISIBLE_DEVICES=$GPU_DEVICES python main_speech.py \
        --experiment-name librispeech_scaling_laws_standard \
        --librispeech-root "$LIBRISPEECH_ROOT" \
        --model-sizes 0,1,2 \
        --dataset-hours 10,50,100 \
        --wandb-project librispeech-scaling-laws \
        --batch-size 8 \
        --eval-steps 500 \
        --warmup-steps 1000 \
        --learning-rate 5e-4 \
        --max-audio-length 20.0 \
        --multi-gpu \
        --run-standard \
        --skip-linear \
        --skip-nonlinear \
        --skip-quantization \
        --skip-lowrank
fi

echo ""
echo "=========================================="
echo "LIBRISPEECH EXPERIMENTS COMPLETE!"
echo "=========================================="
echo "Check the outputs_speech directory for results and plots."
echo "Generated plots will include:"
echo "  - standard_*.png: Standard experiment plots"
if [ "$RUN_ALL" = "true" ]; then
    echo "  - linear_*.png: Linear transform experiment plots"
    echo "  - nonlinear_*.png: Nonlinear transform experiment plots"
    echo "  - quantization_*.png: Quantization transform experiment plots"
    echo "  - lowrank_*.png: Low-rank transform experiment plots"
    echo "  - comparison_linear.png: Standard vs Linear comparison"
    echo "  - comparison_nonlinear.png: Standard vs Nonlinear comparison"
    echo "  - comparison_quantization.png: Standard vs Quantization comparison"
    echo "  - comparison_lowrank.png: Standard vs Low-rank comparison"
fi
echo ""
echo "Key findings to look for:"
echo "  - How WER scales with model size and dataset size"
echo "  - Impact of different transformations on speech recognition performance"
echo "  - Comparison of scaling laws between text and speech modalities"
echo "  - Performance on clean vs noisy test sets"
echo ""
echo "To run comprehensive experiments with all transforms:"
echo "  RUN_ALL=true ./run_librispeech_experiments.sh"
echo ""
echo "To use specific LibriSpeech location:"
echo "  LIBRISPEECH_ROOT=/path/to/LibriSpeech ./run_librispeech_experiments.sh"
echo ""
echo "Dataset caching information:"
echo "  - First run will be slower as datasets are processed and cached"
echo "  - Subsequent runs will load cached datasets much faster"
echo "  - Cache is stored in ./dataset_cache by default"
echo "  - Cache can be cleared manually by removing the cache directory"