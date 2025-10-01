#!/bin/bash

# LibriSpeech Quantization Analysis Script
# Tests the relationship between quantization levels and scaling law parameters on LibriSpeech data

echo "=========================================="
echo "LIBRISPEECH QUANTIZATION ANALYSIS"
echo "=========================================="
echo ""
echo "This script analyzes the relationship between quantization transformation"
echo "significance (quantization levels) and the fitted scaling law parameters"
echo "(alpha, beta, gamma) for speech recognition on LibriSpeech."
echo ""
echo "Analysis details:"
echo "  - Tests quantization levels from 2 (1-bit) to 32 (5-bit)"
echo "  - Uses 3 small model sizes for efficiency (~1.5M, ~5M, ~15M params)"
echo "  - Dataset sizes: 10h, 50h, 100h of LibriSpeech"
echo "  - Generates comprehensive plots and analysis"
echo "  - Evaluates Word Error Rate (WER) scaling behavior"
echo ""

# Check available GPUs
echo "Checking available GPUs..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null || echo "No NVIDIA GPUs detected"
echo ""

# Check if LibriSpeech data exists
LIBRISPEECH_ROOT="${LIBRISPEECH_ROOT:-/cis/home/xhan56/LibriSpeech}"
echo "Checking LibriSpeech data at: $LIBRISPEECH_ROOT"

if [ ! -d "$LIBRISPEECH_ROOT" ]; then
    echo "Error: LibriSpeech directory not found at $LIBRISPEECH_ROOT"
    echo "Please set LIBRISPEECH_ROOT environment variable or ensure data is at the default location"
    exit 1
fi

# Check required subdirectories
required_dirs=("train-clean-100" "dev-clean")
for dir in "${required_dirs[@]}"; do
    if [ ! -d "$LIBRISPEECH_ROOT/$dir" ]; then
        echo "Warning: Required directory $LIBRISPEECH_ROOT/$dir not found"
    else
        echo "Found: $LIBRISPEECH_ROOT/$dir"
    fi
done
echo ""

# GPU configuration
GPU_DEVICES="${GPU_DEVICES:-0,1}"

echo "Using GPUs: $GPU_DEVICES"
echo ""

# Check if required Python packages are available
echo "Checking required packages..."
python -c "import torch; import torchaudio; import transformers; import jiwer; import scipy; print('Core packages found')" || {
    echo "Error: Missing core packages. Please install:"
    echo "  pip install torch torchaudio transformers jiwer scipy"
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
}
echo ""

echo "Expected total runtime: 6-12 hours (quantization analysis is computationally intensive)"
echo "Note: This analysis tests different levels of information compression in speech recognition"
echo "Note: Results will show how quantization affects scaling law parameters (alpha, beta, gamma)"
echo "Note: Datasets and results are cached automatically - subsequent runs will be much faster!"
echo "Note: Cache is stored in ./dataset_cache/quantization/ directory"
echo "=========================================="
echo ""

echo "Starting LibriSpeech quantization analysis..."

# Default settings - can be overridden with environment variables
BATCH_SIZE="${BATCH_SIZE:-16}"
MIN_LEVELS="${MIN_LEVELS:-2}"
MAX_LEVELS="${MAX_LEVELS:-32}"
NUM_LEVELS="${NUM_LEVELS:-6}"

# Check if specific levels are provided
if [ -n "$QUANTIZATION_LEVELS" ]; then
    echo "Using custom quantization levels: $QUANTIZATION_LEVELS"
    CUDA_VISIBLE_DEVICES=$GPU_DEVICES python librispeech_quantization_analysis.py \
        --librispeech-root "$LIBRISPEECH_ROOT" \
        --levels "$QUANTIZATION_LEVELS" \
        --batch-size $BATCH_SIZE \
        --num-workers 4 \
        --experiment-name "librispeech_quantization_custom" \
        --wandb-project "librispeech-quantization-analysis" \
        --output-dir "./outputs_speech_quantization"
else
    echo "Using automatic quantization level range: $MIN_LEVELS to $MAX_LEVELS levels"
    CUDA_VISIBLE_DEVICES=$GPU_DEVICES python librispeech_quantization_analysis.py \
        --librispeech-root "$LIBRISPEECH_ROOT" \
        --min-levels $MIN_LEVELS \
        --max-levels $MAX_LEVELS \
        --num-levels $NUM_LEVELS \
        --batch-size $BATCH_SIZE \
        --num-workers 0 \
        --experiment-name "librispeech_quantization_analysis" \
        --wandb-project "librispeech-quantization-analysis" \
        --output-dir "./outputs_speech_quantization"
fi

echo ""
echo "=========================================="
echo "LIBRISPEECH QUANTIZATION ANALYSIS COMPLETE!"
echo "=========================================="
echo "Check the outputs_speech_quantization directory for results and plots."
echo ""
echo "Analysis shows how quantization levels affect:"
echo "  - Word Error Rate (WER) scaling with model size"
echo "  - WER scaling with dataset size"
echo "  - WER scaling with compute (training FLOPs)"
echo "  - Scaling law parameter stability across compression levels"
echo ""
echo "Key findings to look for:"
echo "  - How compression (fewer bits) affects scaling law exponents"
echo "  - Critical quantization levels where performance degrades"
echo "  - Compare scaling behavior at different compression levels"
echo "  - Information-theoretic limits of speech recognition scaling"
echo ""
echo "To run with custom quantization levels:"
echo "  QUANTIZATION_LEVELS='2,4,8,16,32' ./run_librispeech_quantization_analysis.sh"
echo ""
echo "To run with custom parameters:"
echo "  BATCH_SIZE=16 MIN_LEVELS=4 MAX_LEVELS=64 ./run_librispeech_quantization_analysis.sh"
echo ""
echo "To use specific LibriSpeech location:"
echo "  LIBRISPEECH_ROOT=/path/to/LibriSpeech ./run_librispeech_quantization_analysis.sh"
echo ""
echo "Cache management:"
echo "  - Cache directory: ./dataset_cache/quantization/"
echo "  - First run will cache datasets and results for subsequent runs"
echo "  - To clear cache: rm -rf ./dataset_cache/quantization/"
echo "  - To see cache status: python -c \"from data.quantization_cache import get_quantization_cache; get_quantization_cache().print_cache_status()\""