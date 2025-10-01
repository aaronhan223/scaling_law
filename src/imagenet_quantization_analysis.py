#!/usr/bin/env python3
"""
ImageNet Quantization Analysis Script

This script analyzes the relationship between quantization transformation significance
(quantization ratio) and the fitted scaling law parameters (alpha, beta, gamma) on ImageNet data.
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
from experiments.image_scaling_law_experiment import ImageScalingLawExperiment
from training import set_seed


def fit_scaling_law(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit power law y = A * x^(-alpha) and return alpha, intercept, and R².

    Args:
        x: Independent variable (parameters, dataset size, or compute)
        y: Dependent variable (loss)

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

    slope, intercept, r_value, _, _ = stats.linregress(log_x, log_y)

    # Return -slope as alpha (since we expect negative slopes)
    alpha = -slope
    r_squared = r_value ** 2

    return alpha, intercept, r_squared


def extract_scaling_parameters(df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract scaling law parameters (alpha, beta, gamma) from experiment results.

    Args:
        df: DataFrame with experiment results

    Returns:
        Dictionary with scaling parameters and fit quality metrics
    """
    results = {}

    # Alpha: Loss vs Model Parameters
    if len(df) > 0 and 'model_params' in df.columns and 'final_loss' in df.columns:
        # Group by model size and take average loss
        model_loss = df.groupby('model_params')['final_loss'].mean()
        if len(model_loss) >= 3:
            alpha, alpha_intercept, alpha_r2 = fit_scaling_law(
                model_loss.index.values, model_loss.values
            )
            results.update({
                'alpha': alpha,
                'alpha_intercept': alpha_intercept,
                'alpha_r2': alpha_r2
            })

    # Beta: Loss vs Dataset Size
    if len(df) > 0 and 'dataset_size' in df.columns and 'final_loss' in df.columns:
        # Group by dataset size and take average loss
        data_loss = df.groupby('dataset_size')['final_loss'].mean()
        if len(data_loss) >= 3:
            beta, beta_intercept, beta_r2 = fit_scaling_law(
                data_loss.index.values, data_loss.values
            )
            results.update({
                'beta': beta,
                'beta_intercept': beta_intercept,
                'beta_r2': beta_r2
            })

    # Gamma: Loss vs Compute (for image experiments, may not be available)
    if len(df) > 0 and 'total_time' in df.columns and 'final_loss' in df.columns:
        # Use training time as proxy for compute
        compute_df = df[df['total_time'].notna()]
        if len(compute_df) >= 3:
            gamma, gamma_intercept, gamma_r2 = fit_scaling_law(
                compute_df['total_time'].values, compute_df['final_loss'].values
            )
            results.update({
                'gamma': gamma,
                'gamma_intercept': gamma_intercept,
                'gamma_r2': gamma_r2
            })

    return results


def run_imagenet_quantization_analysis(
    quantization_levels: List[int],
    config: ExperimentConfig,
    output_dir: str,
) -> pd.DataFrame:
    """
    Run scaling law experiments for different quantization levels on ImageNet.

    Args:
        quantization_levels: List of quantization levels to test (number of discrete values)
        config: Base experiment configuration
        output_dir: Output directory for results

    Returns:
        DataFrame with quantization analysis results
    """
    results = []

    for i, levels in enumerate(quantization_levels):
        print(f"\n{'='*80}")
        print(f"IMAGENET QUANTIZATION ANALYSIS: {levels} levels ({i+1}/{len(quantization_levels)})")
        print(f"{'='*80}")

        # Create modified config for this quantization level
        analysis_config = ExperimentConfig()

        # Copy base config settings
        analysis_config.data_config = config.data_config
        analysis_config.training_config = config.training_config
        analysis_config.model_config = config.model_config
        analysis_config.experiment_name = f"imagenet_quantization_analysis_{levels}_levels"
        analysis_config.use_wandb = config.use_wandb
        analysis_config.wandb_project = f"{config.wandb_project}_imagenet_quantization"

        # Image-specific settings
        analysis_config.img_size = getattr(config, 'img_size', 224)
        analysis_config.num_classes = getattr(config, 'num_classes', 1000)
        if hasattr(config, 'model_indices'):
            analysis_config.model_indices = config.model_indices

        # Set quantization parameters for images
        analysis_config.transform_config.quantization_num_levels = levels

        # Enable only quantization transform
        analysis_config.run_standard = False
        analysis_config.run_linear_transform = False
        analysis_config.run_nonlinear_transform = False
        analysis_config.run_clustering_transform = False
        analysis_config.run_modular_transform = False
        analysis_config.run_lowrank_transform = False
        analysis_config.run_quantization_transform = True

        # Create subdirectory for this quantization level
        levels_output_dir = os.path.join(output_dir, f"levels_{levels}")
        analysis_config.training_config.output_dir = levels_output_dir

        try:
            # Clear memory before each experiment
            import torch
            import gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # Run experiments with memory optimization
            experiment = ImageScalingLawExperiment(analysis_config)
            experiment_results = experiment.run_all_experiments()

            # Clear memory after each experiment
            del experiment
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM Error for quantization levels {levels}: {e}")
                print("Attempting recovery with aggressive memory optimization...")

                # Clear any cached data
                import torch
                import gc
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                # Reduce batch size and try again
                original_batch_size = analysis_config.data_config.train_batch_size
                analysis_config.data_config.train_batch_size = max(1, original_batch_size // 2)
                analysis_config.data_config.eval_batch_size = max(1, analysis_config.data_config.eval_batch_size // 2)

                print(f"Retrying with reduced batch size: {analysis_config.data_config.train_batch_size}")

                try:
                    experiment = ImageScalingLawExperiment(analysis_config)
                    experiment_results = experiment.run_all_experiments()
                    print(f"Successfully completed with reduced batch size")
                except Exception as retry_error:
                    print(f"Failed even with reduced batch size: {retry_error}")
                    continue
            else:
                raise e

        if experiment_results:
            # Convert to DataFrame
            df = pd.DataFrame(experiment_results)

            # Extract scaling parameters
            scaling_params = extract_scaling_parameters(df)

            # Store results
            result = {
                'quantization_levels': levels,
                'quantization_bits': np.log2(levels),
                'num_experiments': len(df),
                **scaling_params
            }

            # Add some aggregate metrics
            if len(df) > 0:
                result.update({
                    'mean_loss': df['final_loss'].mean(),
                    'std_loss': df['final_loss'].std(),
                    'min_loss': df['final_loss'].min(),
                    'max_loss': df['final_loss'].max(),
                    'mean_accuracy': df['final_accuracy'].mean() if 'final_accuracy' in df.columns else None,
                    'std_accuracy': df['final_accuracy'].std() if 'final_accuracy' in df.columns else None,
                })

            results.append(result)

            print(f"Completed quantization levels {levels}")

            # Format scaling parameters safely
            alpha_val = scaling_params.get('alpha', None)
            beta_val = scaling_params.get('beta', None)
            gamma_val = scaling_params.get('gamma', None)

            alpha_str = f"{alpha_val:.3f}" if alpha_val is not None else "N/A"
            beta_str = f"{beta_val:.3f}" if beta_val is not None else "N/A"
            gamma_str = f"{gamma_val:.3f}" if gamma_val is not None else "N/A"

            print(f"  Scaling parameters: α={alpha_str}, β={beta_str}, γ={gamma_str}")

            # Save intermediate results
            results_df = pd.DataFrame(results)
            results_file = os.path.join(output_dir, "imagenet_quantization_analysis_results.csv")
            results_df.to_csv(results_file, index=False)

    return pd.DataFrame(results)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ImageNet Quantization Transformation Analysis")

    # Quantization settings
    parser.add_argument(
        "--min-levels",
        type=int,
        default=2,
        help="Minimum quantization levels (default: 2 for 1-bit)",
    )
    parser.add_argument(
        "--max-levels",
        type=int,
        default=256,
        help="Maximum quantization levels (default: 256 for 8-bit)",
    )
    parser.add_argument(
        "--num-levels",
        type=int,
        default=8,
        help="Number of quantization levels to test (default: 8)",
    )
    parser.add_argument(
        "--levels",
        type=str,
        default=None,
        help="Comma-separated list of specific levels to test (overrides min/max/num)",
    )

    # Model and data settings
    parser.add_argument(
        "--model-sizes",
        type=str,
        default="0,1,2",
        help="Comma-separated model size indices (default: 0,1,2 for small models)",
    )
    parser.add_argument(
        "--dataset-sizes",
        type=str,
        default="1K,10K,50K",
        help="Comma-separated dataset sizes (default: 1K,10K,50K)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Image size (default: 224)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=1000,
        help="Number of classes (default: 1000)",
    )

    # Training settings
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (default: 32 for memory efficiency)",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=64,
        help="Evaluation batch size (default: 64)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5 for efficiency)",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=200,
        help="Evaluation frequency (default: 200)",
    )

    # Output settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/imagenet_quantization_analysis",
        help="Output directory (default: ./outputs/imagenet_quantization_analysis)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="imagenet_quantization_analysis",
        help="Experiment name (default: imagenet_quantization_analysis)",
    )

    # WandB settings
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="imagenet-quantization-analysis",
        help="WandB project name (default: imagenet-quantization-analysis)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging",
    )

    # Other settings
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only generate plots from existing results",
    )

    return parser.parse_args()


def parse_dataset_sizes(dataset_sizes_str: str) -> List[int]:
    """Parse dataset sizes from string."""
    sizes = []
    for size in dataset_sizes_str.split(","):
        size = size.strip().upper()
        if size.endswith("M"):
            sizes.append(int(float(size[:-1]) * 1e6))
        elif size.endswith("B"):
            sizes.append(int(float(size[:-1]) * 1e9))
        elif size.endswith("K"):
            sizes.append(int(float(size[:-1]) * 1e3))
        else:
            try:
                sizes.append(int(size))
            except ValueError:
                print(f"Warning: Cannot parse dataset size '{size}', skipping")
    return sorted(sizes)


def parse_model_sizes(model_sizes_str: str) -> List[int]:
    """Parse model size indices from string."""
    indices = []
    for idx in model_sizes_str.split(","):
        try:
            indices.append(int(idx.strip()))
        except ValueError:
            print(f"Warning: Cannot parse model size index '{idx}', skipping")
    return sorted(indices)


def main():
    """Main function."""
    args = parse_args()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.experiment_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("IMAGENET QUANTIZATION TRANSFORMATION ANALYSIS")
    print("=" * 80)
    print(f"Output directory: {output_dir}")

    if args.plot_only:
        # Load existing results and generate plots
        results_file = os.path.join(args.output_dir, "imagenet_quantization_analysis_results.csv")
        if not os.path.exists(results_file):
            print(f"Error: Results file not found at {results_file}")
            return

        print(f"Loading results from: {results_file}")
        df = pd.read_csv(results_file)

        # Generate plots
        try:
            from visualization.quantization_plots import plot_imagenet_quantization_analysis
            plot_dir = os.path.join(os.path.dirname(results_file), "plots")
            plot_imagenet_quantization_analysis(df, plot_dir)
            print(f"Plots saved to: {plot_dir}")
        except ImportError:
            print("Warning: Could not import quantization plotting functions")
        return

    # Set up experiment configuration
    config = ExperimentConfig()
    config.experiment_name = args.experiment_name
    config.data_config.train_batch_size = args.batch_size
    config.data_config.eval_batch_size = args.eval_batch_size
    config.data_config.dataset_sizes = parse_dataset_sizes(args.dataset_sizes)
    config.data_config.num_workers = args.num_workers
    config.training_config.learning_rate = args.learning_rate
    config.training_config.seed = args.seed
    config.training_config.num_epochs = args.num_epochs
    config.training_config.eval_steps = args.eval_steps
    config.use_wandb = not args.no_wandb
    config.wandb_project = args.wandb_project

    # Image-specific config
    config.img_size = args.img_size
    config.num_classes = args.num_classes
    config.model_indices = parse_model_sizes(args.model_sizes)

    # Set up quantization levels using min-levels, max-levels, and num-levels
    print(f'Using quantization range: {args.min_levels} to {args.max_levels} with {args.num_levels} levels')

    # Use logarithmic spacing for quantization levels (2^1, 2^2, ..., 2^8)
    quantization_levels = [2**i for i in range(
        int(np.log2(args.min_levels)),
        int(np.log2(args.max_levels)) + 1,
        max(1, (int(np.log2(args.max_levels)) + 1 - int(np.log2(args.min_levels))) // args.num_levels)
    )]
    print('quantization_levels:', quantization_levels)
    # Add specific intermediate values if needed
    # additional_levels = [4, 8, 16, 32, 64, 128, 256]
    # for level in additional_levels:
    #     if args.min_levels <= level <= args.max_levels and level not in quantization_levels:
    #         quantization_levels.append(level)
    # quantization_levels = sorted(set(quantization_levels))

    print(f"Testing {len(quantization_levels)} quantization levels:")
    for levels in quantization_levels:
        bits = np.log2(levels)
        print(f"  {levels} levels -> {bits:.1f}-bit quantization")
    print(f"Model indices: {config.model_indices}")
    print(f"Dataset sizes: {config.data_config.dataset_sizes}")
    print("=" * 80)

    # Save configuration
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump({
            'quantization_levels': quantization_levels,
            'args': vars(args),
            'config': config.__dict__
        }, f, indent=2, default=str)

    # Set seed
    set_seed(args.seed, deterministic=True)

    # Run analysis
    start_time = time.time()
    results_df = run_imagenet_quantization_analysis(quantization_levels, config, output_dir)

    # Save final results
    if len(results_df) > 0:
        results_file = os.path.join(output_dir, "imagenet_quantization_analysis_results.csv")
        results_df.to_csv(results_file, index=False)

        # Also save as JSON
        results_json = os.path.join(output_dir, "imagenet_quantization_analysis_results.json")
        results_df.to_json(results_json, indent=2)

        print("\n" + "=" * 80)
        print("IMAGENET QUANTIZATION ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"Total time: {time.time() - start_time:.1f} seconds")
        print(f"Results saved to: {results_file}")
        print(f"Tested {len(results_df)} quantization levels")

        # Generate plots
        try:
            from visualization.quantization_plots import plot_imagenet_quantization_analysis
            plot_dir = os.path.join(output_dir, "plots")
            plot_imagenet_quantization_analysis(results_df, plot_dir)
            print(f"Plots saved to: {plot_dir}")
        except ImportError:
            print("Warning: Could not import quantization plotting functions")
            print("Run the visualization manually after creating the plot functions")

        # Print summary statistics
        print("\nSummary Statistics:")
        for param in ['alpha', 'beta', 'gamma']:
            if param in results_df.columns:
                values = results_df[param].dropna()
                if len(values) > 0:
                    print(f"  {param}: mean={values.mean():.3f}, std={values.std():.3f}, "
                          f"range=[{values.min():.3f}, {values.max():.3f}]")

        print("=" * 80)
    else:
        print("ERROR: No results collected!")


if __name__ == "__main__":
    main()