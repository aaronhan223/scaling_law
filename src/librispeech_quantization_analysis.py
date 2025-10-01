#!/usr/bin/env python3
"""
LibriSpeech Quantization Analysis Script

This script analyzes the relationship between quantization transformation significance
(quantization levels) and the fitted scaling law parameters for speech recognition tasks.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import json
import time
from datetime import datetime

from config import ExperimentConfig
from experiments.speech_scaling_law_experiment import SpeechScalingLawExperiment
from training import set_seed
from data.quantization_cache import get_quantization_cache
from typing import Any


def fit_scaling_law(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit power law y = A * x^(-alpha) and return alpha, intercept, and R².

    Args:
        x: Independent variable (parameters, dataset size, or compute)
        y: Dependent variable (WER or loss)

    Returns:
        Tuple of (alpha, log_intercept, r_squared)
    """
    if len(x) < 3 or len(y) < 3:
        return np.nan, np.nan, np.nan

    # Remove any invalid values
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 3:
        return np.nan, np.nan, np.nan

    x_clean = x[mask]
    y_clean = y[mask]

    # Fit in log space: log(y) = log(A) - alpha * log(x)
    log_x = np.log10(x_clean)
    log_y = np.log10(y_clean)

    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
        alpha = -slope  # Convert slope to positive scaling exponent
        r_squared = r_value ** 2
        return alpha, intercept, r_squared
    except Exception:
        return np.nan, np.nan, np.nan


def run_single_quantization_experiment_cached(
    levels: int,
    config: ExperimentConfig,
    librispeech_root: str,
    cache_dir: str,
    model_indices: List[int],
    quantization_cache,
    experiment_output_dir: str
) -> Optional[Dict[str, Any]]:
    """
    Run a single quantization experiment with caching support.

    Returns:
        Experiment results dictionary or None if failed
    """

    # Check if this specific experiment is already cached
    cache_key = quantization_cache._generate_dataset_key(
        librispeech_root=librispeech_root,
        dataset_hours=config.data_config.dataset_sizes,
        model_indices=model_indices,
        quantization_levels=levels,
        batch_size=config.data_config.train_batch_size,
        max_audio_length=config.data_config.max_length,
        processor_name="facebook/wav2vec2-base-960h",
        seed=config.training_config.seed
    )

    # Try to load cached results first
    cached_results = quantization_cache.load_experiment_results(cache_key)
    if cached_results:
        print(f"✓ Found cached results for {levels} levels, skipping computation")
        return cached_results['results']

    print(f"Running new experiment for {levels} levels...")

    # Update config for this quantization level
    analysis_config = ExperimentConfig()
    analysis_config.__dict__.update(config.__dict__)

    # Configure for quantization analysis
    analysis_config.experiment_name = f"librispeech_quantization_analysis_{levels}_levels"
    analysis_config.run_standard = False
    analysis_config.run_linear_transform = False
    analysis_config.run_nonlinear_transform = False
    analysis_config.run_quantization_transform = True
    analysis_config.run_lowrank_transform = False

    # Set quantization parameters
    analysis_config.transform_config.quantization_num_levels = levels
    analysis_config.training_config.output_dir = experiment_output_dir

    try:
        # Create and run experiment
        experiment = SpeechScalingLawExperiment(
            config=analysis_config,
            librispeech_root=librispeech_root,
            model_indices=model_indices,
            cache_dir=cache_dir
        )

        experiment_results = experiment.run_all_experiments()

        if experiment_results:
            # Cache the results for future use
            experiment_config = {
                'quantization_levels': levels,
                'dataset_hours': config.data_config.dataset_sizes,
                'model_indices': model_indices,
                'batch_size': config.data_config.train_batch_size,
            }

            quantization_cache.cache_experiment_results(
                experiment_results, experiment_config, cache_key
            )

            return experiment_results
        else:
            print(f"Warning: Experiment failed for {levels} levels")
            return None

    except Exception as e:
        print(f"Error for quantization levels {levels}: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_quantization_scaling_laws(
    quantization_levels: List[int],
    config: ExperimentConfig,
    librispeech_root: str,
    cache_dir: str,
    output_dir: str
) -> pd.DataFrame:
    """
    Run scaling law experiments for different quantization levels on LibriSpeech.

    Args:
        quantization_levels: List of quantization levels to test (number of discrete values)
        config: Experiment configuration
        librispeech_root: Path to LibriSpeech data
        cache_dir: Directory for dataset caching
        output_dir: Output directory for results

    Returns:
        DataFrame with scaling law analysis results
    """
    results = []

    # Model indices to test (smaller models for efficiency)
    model_indices = [0, 1, 2]  # ~1.5M, ~5M, ~15M parameters

    # Initialize quantization cache
    quantization_cache = get_quantization_cache(os.path.join(cache_dir, "quantization"))
    quantization_cache.print_cache_status()

    # Use smaller dataset sizes for efficiency
    config.data_config.dataset_sizes = [10, 50, 100]  # hours

    print(f"\nStarting quantization analysis for {len(quantization_levels)} levels...")
    print(f"Dataset hours: {config.data_config.dataset_sizes}")
    print(f"Model indices: {model_indices}")
    print(f"Cache directory: {quantization_cache.cache_dir}")

    for i, levels in enumerate(quantization_levels):
        print(f"\nLIBRISPEECH QUANTIZATION ANALYSIS: {levels} levels ({i+1}/{len(quantization_levels)})")
        print("=" * 80)

        experiment_output_dir = os.path.join(output_dir, f"quantization_{levels}_levels")
        os.makedirs(experiment_output_dir, exist_ok=True)

        # Try to run experiment with caching
        experiment_results = run_single_quantization_experiment_cached(
            levels=levels,
            config=config,
            librispeech_root=librispeech_root,
            cache_dir=cache_dir,
            model_indices=model_indices,
            quantization_cache=quantization_cache,
            experiment_output_dir=experiment_output_dir
        )

        if experiment_results:
            # If we got cached results, they might already be processed
            if isinstance(experiment_results, dict) and 'df_all' in experiment_results:
                df_all = experiment_results['df_all']
            else:
                # Process raw experiment results
                from experiments.speech_scaling_law_experiment import SpeechScalingLawExperiment

                # Create temporary experiment instance for analysis
                temp_config = ExperimentConfig()
                temp_config.__dict__.update(config.__dict__)
                temp_config.transform_config.quantization_num_levels = levels

                temp_experiment = SpeechScalingLawExperiment(
                    config=temp_config,
                    librispeech_root=librispeech_root,
                    model_indices=model_indices,
                    cache_dir=cache_dir
                )

                # Set the results and analyze
                temp_experiment.results = experiment_results
                df_results = temp_experiment.analyze_results()
                df_all = df_results["all"]

            if df_all is not None and len(df_all) > 0:
                # Extract quantization experiments
                quant_df = df_all[df_all["experiment_type"] == "quantization"]

                if len(quant_df) > 0:
                    # Calculate scaling law parameters for each scaling dimension
                    scaling_params = {}

                    # Model size scaling (fixed dataset size)
                    for dataset_size in quant_df["dataset_size"].unique():
                        subset = quant_df[quant_df["dataset_size"] == dataset_size]
                        if len(subset) >= 3:
                            params = subset["model_params"].values
                            wer = subset["dev_wer"].values

                            alpha, intercept, r2 = fit_scaling_law(params, wer)
                            scaling_params[f"model_alpha_data{dataset_size}"] = alpha
                            scaling_params[f"model_r2_data{dataset_size}"] = r2

                    # Dataset size scaling (fixed model size)
                    for model_name in quant_df["model_name"].unique():
                        subset = quant_df[quant_df["model_name"] == model_name]
                        if len(subset) >= 3:
                            dataset_sizes = subset["dataset_size"].values
                            wer = subset["dev_wer"].values

                            alpha, intercept, r2 = fit_scaling_law(dataset_sizes, wer)
                            scaling_params[f"data_alpha_model{model_name}"] = alpha
                            scaling_params[f"data_r2_model{model_name}"] = r2

                    # Compute size scaling (using training FLOPS)
                    if "train_flops" in quant_df.columns:
                        for dataset_size in quant_df["dataset_size"].unique():
                            subset = quant_df[quant_df["dataset_size"] == dataset_size]
                            if len(subset) >= 3:
                                flops = subset["train_flops"].values
                                wer = subset["dev_wer"].values

                                alpha, intercept, r2 = fit_scaling_law(flops, wer)
                                scaling_params[f"compute_alpha_data{dataset_size}"] = alpha
                                scaling_params[f"compute_r2_data{dataset_size}"] = r2

                    # Calculate average scaling parameters
                    avg_scaling_params = {}
                    for param_type in ["model_alpha", "model_r2", "data_alpha", "data_r2", "compute_alpha", "compute_r2"]:
                        values = [v for k, v in scaling_params.items() if param_type in k and not np.isnan(v)]
                        if values:
                            avg_scaling_params[f"avg_{param_type}"] = np.mean(values)
                            avg_scaling_params[f"std_{param_type}"] = np.std(values)
                        else:
                            avg_scaling_params[f"avg_{param_type}"] = np.nan
                            avg_scaling_params[f"std_{param_type}"] = np.nan

                    # Calculate summary statistics
                    mean_wer = quant_df["dev_wer"].mean()
                    min_wer = quant_df["dev_wer"].min()
                    max_wer = quant_df["dev_wer"].max()

                    # Store results
                    result = {
                        'quantization_levels': levels,
                        'quantization_bits': np.log2(levels),
                        'mean_wer': mean_wer,
                        'min_wer': min_wer,
                        'max_wer': max_wer,
                        'num_experiments': len(quant_df),
                        **scaling_params,
                        **avg_scaling_params,
                    }

                    results.append(result)

                    print(f"Completed quantization levels {levels}: Mean WER = {mean_wer:.4f}")
                else:
                    print(f"Warning: No quantization results found for {levels} levels")
            else:
                print(f"Warning: No results found for {levels} levels")
        else:
            print(f"Warning: Experiment failed for {levels} levels")

    # Show final cache status
    print(f"\nFinal cache status:")
    quantization_cache.print_cache_status()

    # Convert to DataFrame
    if results:
        df = pd.DataFrame(results)

        # Save results
        results_file = os.path.join(output_dir, "librispeech_quantization_analysis_results.csv")
        df.to_csv(results_file, index=False)
        print(f"\nResults saved to: {results_file}")

        return df
    else:
        print("No results obtained!")
        return pd.DataFrame()


def parse_quantization_levels(levels_str: str) -> List[int]:
    """Parse quantization levels from string."""
    if "," in levels_str:
        return [int(x.strip()) for x in levels_str.split(",")]
    else:
        return [int(levels_str)]


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="LibriSpeech quantization analysis")

    # Data settings
    parser.add_argument(
        "--librispeech-root",
        type=str,
        default="/cis/home/xhan56/LibriSpeech",
        help="Root directory containing LibriSpeech data"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./dataset_cache",
        help="Directory to store cached datasets"
    )

    # Quantization settings
    parser.add_argument(
        "--levels",
        type=str,
        default=None,
        help="Comma-separated list of quantization levels (e.g., '2,4,8,16,32')"
    )
    parser.add_argument(
        "--min-levels",
        type=int,
        default=2,
        help="Minimum quantization levels (default: 2 for 1-bit)",
    )
    parser.add_argument(
        "--max-levels",
        type=int,
        default=32,
        help="Maximum quantization levels (default: 32 for 5-bit)",
    )
    parser.add_argument(
        "--num-levels",
        type=int,
        default=6,
        help="Number of quantization levels to test (default: 6)",
    )

    # Experiment settings
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="librispeech_quantization_analysis",
        help="Name of the experiment"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )

    # Output settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs_speech_quantization",
        help="Output directory"
    )

    # WandB settings
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="librispeech-quantization-analysis",
        help="WandB project name"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging"
    )

    # Other settings
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Check LibriSpeech root directory
    if not os.path.exists(args.librispeech_root):
        print(f"Error: LibriSpeech root directory not found: {args.librispeech_root}")
        print("Please provide the correct path using --librispeech-root")
        return

    # Set up quantization levels
    if args.levels:
        quantization_levels = [int(x.strip()) for x in args.levels.split(",")]
    else:
        # Use powers of 2 spacing for quantization levels
        quantization_levels = [2**i for i in range(
            int(np.log2(args.min_levels)),
            int(np.log2(args.max_levels)) + 1
        )]

        # Add some intermediate values if needed
        if args.min_levels <= 3 <= args.max_levels and 3 not in quantization_levels:
            quantization_levels.append(3)
        if args.min_levels <= 6 <= args.max_levels and 6 not in quantization_levels:
            quantization_levels.append(6)
        if args.min_levels <= 12 <= args.max_levels and 12 not in quantization_levels:
            quantization_levels.append(12)
        if args.min_levels <= 24 <= args.max_levels and 24 not in quantization_levels:
            quantization_levels.append(24)
        quantization_levels = sorted(set(quantization_levels))

    print(f"Testing {len(quantization_levels)} quantization levels:")
    for levels in quantization_levels:
        bits = np.log2(levels)
        print(f"  {levels} levels ({bits:.1f} bits)")

    # Create experiment config
    config = ExperimentConfig()
    config.experiment_name = args.experiment_name
    config.use_wandb = not args.no_wandb
    config.wandb_project = args.wandb_project

    # Configure for quantization analysis
    config.data_config.train_batch_size = args.batch_size
    config.data_config.eval_batch_size = args.batch_size
    config.data_config.num_workers = args.num_workers

    # Use shorter training for analysis
    config.training_config.eval_steps = 200
    config.training_config.warmup_steps = 500
    config.training_config.learning_rate = 5e-4
    config.training_config.seed = args.seed

    # Set num_workers to 0 to avoid multiprocessing tensor issues
    config.data_config.num_workers = 0

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{config.experiment_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Save experiment parameters
    params = {
        'quantization_levels': quantization_levels,
        'librispeech_root': args.librispeech_root,
        'experiment_name': args.experiment_name,
        'batch_size': args.batch_size,
        'timestamp': timestamp,
    }

    params_file = os.path.join(output_dir, "experiment_params.json")
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=2)

    print("=" * 80)
    print("LIBRISPEECH QUANTIZATION ANALYSIS")
    print("=" * 80)
    print(f"Experiment name: {config.experiment_name}")
    print(f"LibriSpeech root: {args.librispeech_root}")
    print(f"Output directory: {output_dir}")
    print(f"Quantization levels: {quantization_levels}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 80)

    # Run analysis
    results_df = analyze_quantization_scaling_laws(
        quantization_levels, config, args.librispeech_root, args.cache_dir, output_dir
    )

    # Generate summary and visualizations
    if len(results_df) > 0:
        from visualization.librispeech_quantization_plots import (
            plot_quantization_summary, print_quantization_summary
        )

        # Print summary
        print_quantization_summary(results_df)

        # Generate plots
        plots_dir = os.path.join(output_dir, "plots")
        plot_quantization_summary(results_df, plots_dir, "librispeech_quantization_analysis")

        print(f"\nResults and plots saved to: {output_dir}")
        print("=" * 80)
    else:
        print("No results obtained!")


if __name__ == "__main__":
    main()