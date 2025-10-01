#!/usr/bin/env python3
"""
Low-rank Analysis Script

This script analyzes the relationship between low-rank transformation significance
(low-rank ratio) and the fitted scaling law parameters (alpha, beta, gamma).
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
from experiments import ScalingLawExperiment
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
    
    # Gamma: Loss vs Compute
    if len(df) > 0 and 'petaflop_days' in df.columns and 'final_loss' in df.columns:
        compute_df = df[df['petaflop_days'].notna()]
        if len(compute_df) >= 3:
            gamma, gamma_intercept, gamma_r2 = fit_scaling_law(
                compute_df['petaflop_days'].values, compute_df['final_loss'].values
            )
            results.update({
                'gamma': gamma,
                'gamma_intercept': gamma_intercept,
                'gamma_r2': gamma_r2
            })
    
    return results


def run_lowrank_analysis(
    lowrank_ratios: List[float],
    config: ExperimentConfig,
    output_dir: str,
) -> pd.DataFrame:
    """
    Run scaling law experiments for different low-rank ratios.
    
    Args:
        lowrank_ratios: List of low-rank ratios to test
        config: Base experiment configuration
        output_dir: Output directory for results
        
    Returns:
        DataFrame with low-rank analysis results
    """
    results = []
    
    for i, ratio in enumerate(lowrank_ratios):
        print(f"\n{'='*80}")
        print(f"LOW-RANK ANALYSIS: Ratio {ratio:.4f} ({i+1}/{len(lowrank_ratios)})")
        print(f"{'='*80}")
        
        # Create modified config for this ratio
        analysis_config = ExperimentConfig()
        
        # Copy base config settings
        analysis_config.data_config = config.data_config
        analysis_config.training_config = config.training_config
        analysis_config.model_config = config.model_config
        analysis_config.experiment_name = f"lowrank_analysis_{ratio:.4f}"
        analysis_config.use_wandb = config.use_wandb
        analysis_config.wandb_project = f"{config.wandb_project}_lowrank"
        
        # Ensure no data loader workers to prevent deadlock
        analysis_config.data_config.num_workers = 0
        
        # Set low-rank parameters
        vocab_size = analysis_config.model_config.vocab_size
        analysis_config.transform_config.lowrank_rank = int(vocab_size * ratio)
        
        # Enable only low-rank transform
        analysis_config.run_standard = False
        analysis_config.run_transformed = False
        analysis_config.run_linear_transform = False
        analysis_config.run_nonlinear_transform = False
        analysis_config.run_clustering_transform = False
        analysis_config.run_modular_transform = False
        analysis_config.run_quantization_transform = False
        analysis_config.run_lowrank_transform = True
        
        # Create subdirectory for this ratio
        ratio_output_dir = os.path.join(output_dir, f"ratio_{ratio:.4f}")
        analysis_config.training_config.output_dir = ratio_output_dir
        
        try:
            # Run experiments with memory optimization
            print(f"Starting experiment for ratio {ratio:.4f}...")
            
            # Clear GPU memory before starting
            import torch
            import gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            experiment = ScalingLawExperiment(analysis_config)
            print(f"Experiment object created for ratio {ratio:.4f}")
            
            experiment_results = experiment.run_all_experiments()
            print(f"Experiments completed for ratio {ratio:.4f}")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM Error for low-rank ratio {ratio:.4f}: {e}")
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
                    experiment = ScalingLawExperiment(analysis_config)
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
            
            # Calculate compression ratio
            rank = int(vocab_size * ratio)
            compression_ratio = (2 * rank) / vocab_size  # U and V matrices
            
            # Store results
            result = {
                'lowrank_ratio': ratio,
                'lowrank_rank': rank,
                'compression_ratio': compression_ratio,
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
                })
            
            results.append(result)
            
            print(f"Completed low-rank ratio {ratio:.4f}")
            print(f"  Rank: {rank}, Compression ratio: {compression_ratio:.4f}")
            print(f"  Scaling parameters: α={scaling_params.get('alpha', 'N/A'):.3f}, "
                    f"β={scaling_params.get('beta', 'N/A'):.3f}, "
                    f"γ={scaling_params.get('gamma', 'N/A'):.3f}")
            
            # Save intermediate results
            results_df = pd.DataFrame(results)
            results_file = os.path.join(output_dir, "lowrank_analysis_results.csv")
            results_df.to_csv(results_file, index=False)
                
        # except Exception as e:
        #     print(f"Error running low-rank ratio {ratio:.4f}: {e}")
        #     continue
    
    return pd.DataFrame(results)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Low-rank Transformation Analysis")
    
    # Low-rank settings
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=0.01,
        help="Minimum low-rank ratio (default: 0.01)",
    )
    parser.add_argument(
        "--max-ratio",
        type=float,
        default=0.5,
        help="Maximum low-rank ratio (default: 0.5)",
    )
    parser.add_argument(
        "--num-ratios",
        type=int,
        default=10,
        help="Number of low-rank ratios to test (default: 10)",
    )
    parser.add_argument(
        "--ratios",
        type=str,
        default=None,
        help="Comma-separated list of specific ratios to test (overrides min/max/num)",
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
        default="100K,1M,10M",
        help="Comma-separated dataset sizes (default: 100K,1M,10M)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="wikitext",
        help="Dataset name (default: wikitext)",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="wikitext-103-v1",
        help="Dataset config (default: wikitext-103-v1)",
    )
    
    # Training settings
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size (default: 8)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4,
        help="Learning rate (default: 5e-4)",
    )
    
    # Output settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/lowrank_analysis",
        help="Output directory (default: ./outputs/lowrank_analysis)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="lowrank_analysis",
        help="Experiment name (default: lowrank_analysis)",
    )
    
    # WandB settings
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="lowrank-analysis",
        help="WandB project name (default: lowrank-analysis)",
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


def parse_model_sizes(model_sizes_str: str, config: ExperimentConfig) -> None:
    """Parse and set model sizes in config."""
    indices = [int(x.strip()) for x in model_sizes_str.split(",")]
    
    all_hidden_sizes = config.model_config.hidden_sizes
    all_num_heads = config.model_config.num_attention_heads
    all_num_layers = config.model_config.num_hidden_layers
    
    config.model_config.hidden_sizes = [all_hidden_sizes[i] for i in indices if i < len(all_hidden_sizes)]
    config.model_config.num_attention_heads = [all_num_heads[i] for i in indices if i < len(all_num_heads)]
    config.model_config.num_hidden_layers = [all_num_layers[i] for i in indices if i < len(all_num_layers)]


def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.experiment_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("LOW-RANK TRANSFORMATION ANALYSIS")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    
    if args.plot_only:
        # Load existing results and generate plots
        results_file = os.path.join(args.output_dir, "lowrank_analysis_results.csv")
        if not os.path.exists(results_file):
            print(f"Error: Results file not found at {results_file}")
            return
        
        print(f"Loading results from: {results_file}")
        df = pd.read_csv(results_file)
        
        # Generate plots
        try:
            from visualization.lowrank_plots import plot_lowrank_analysis
            plot_dir = os.path.join(os.path.dirname(results_file), "plots")
            plot_lowrank_analysis(df, plot_dir)
            print(f"Plots saved to: {plot_dir}")
        except ImportError:
            print("Warning: Could not import lowrank plotting functions")
            print("Plots will be generated after the analysis completes")
        return
    
    # Set up experiment configuration
    config = ExperimentConfig()
    config.experiment_name = args.experiment_name
    config.data_config.dataset_name = args.dataset_name
    config.data_config.dataset_config = args.dataset_config
    config.data_config.train_batch_size = args.batch_size
    config.data_config.dataset_sizes = parse_dataset_sizes(args.dataset_sizes)
    config.training_config.learning_rate = args.learning_rate
    config.training_config.seed = args.seed
    config.use_wandb = not args.no_wandb
    config.wandb_project = args.wandb_project
    
    # Fix potential multiprocessing deadlock by disabling data loader workers
    config.data_config.num_workers = 0
    
    # Parse model sizes
    parse_model_sizes(args.model_sizes, config)
    
    # Handle OpenWebText config
    if config.data_config.dataset_name == "openwebtext":
        config.data_config.dataset_config = None
    
    # Set up low-rank ratios
    if args.ratios is not None:
        lowrank_ratios = [float(x.strip()) for x in args.ratios.split(",")]
    else:
        lowrank_ratios = np.logspace(
            np.log10(args.min_ratio),
            np.log10(args.max_ratio),
            args.num_ratios
        ).tolist()
    
    print(f"Testing {len(lowrank_ratios)} low-rank ratios:")
    for ratio in lowrank_ratios:
        vocab_size = config.model_config.vocab_size
        rank = int(vocab_size * ratio)
        compression_ratio = (2 * rank) / vocab_size
        print(f"  {ratio:.4f} -> rank {rank} (compression ratio: {compression_ratio:.4f})")
    print(f"Model sizes: {len(config.model_config.hidden_sizes)}")
    print(f"Dataset sizes: {config.data_config.dataset_sizes}")
    print("=" * 80)
    
    # Save configuration
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump({
            'lowrank_ratios': lowrank_ratios,
            'args': vars(args),
            'config': config.__dict__
        }, f, indent=2, default=str)
    
    # Set seed
    set_seed(args.seed, deterministic=True)
    
    # Run analysis
    start_time = time.time()
    results_df = run_lowrank_analysis(lowrank_ratios, config, output_dir)
    
    # Save final results
    if len(results_df) > 0:
        results_file = os.path.join(output_dir, "lowrank_analysis_results.csv")
        results_df.to_csv(results_file, index=False)
        
        # Also save as JSON
        results_json = os.path.join(output_dir, "lowrank_analysis_results.json")
        results_df.to_json(results_json, indent=2)
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"Total time: {time.time() - start_time:.1f} seconds")
        print(f"Results saved to: {results_file}")
        print(f"Tested {len(results_df)} low-rank ratios")
        
        # Generate plots
        try:
            from visualization.lowrank_plots import plot_lowrank_analysis
            plot_dir = os.path.join(output_dir, "plots")
            plot_lowrank_analysis(results_df, plot_dir)
            print(f"Plots saved to: {plot_dir}")
        except ImportError:
            print("Warning: Could not import lowrank plotting functions")
            print("Run the visualization manually after creating the plot functions")
        
        # Print summary statistics
        print("\nSummary Statistics:")
        for param in ['alpha', 'beta', 'gamma']:
            if param in results_df.columns:
                values = results_df[param].dropna()
                if len(values) > 0:
                    print(f"  {param}: mean={values.mean():.3f}, std={values.std():.3f}, "
                          f"range=[{values.min():.3f}, {values.max():.3f}]")
        
        # Print correlation analysis
        print("\nCorrelation with low-rank ratio:")
        for param in ['alpha', 'beta', 'gamma']:
            if param in results_df.columns and 'lowrank_ratio' in results_df.columns:
                corr = results_df['lowrank_ratio'].corr(results_df[param])
                if not np.isnan(corr):
                    print(f"  {param}: {corr:.3f}")
        
        print("=" * 80)
    else:
        print("ERROR: No results collected!")


if __name__ == "__main__":
    main()