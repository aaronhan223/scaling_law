"""Main entry point for scaling law experiments."""

import argparse
import os
import sys
import torch
import wandb
from datetime import datetime

from config import ExperimentConfig
from experiments import ScalingLawExperiment
from visualization import plot_all_scaling_laws, compare_standard_vs_transformed
from visualization.plots import compare_experiment_types
from training import set_seed
import pdb


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run scaling law experiments")
    
    # Experiment settings
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="scaling_laws",
        help="Name of the experiment",
    )
    parser.add_argument(
        "--run-standard",
        action="store_true",
        default=True,
        help="Run standard experiments",
    )
    parser.add_argument(
        "--run-transformed",
        action="store_true",
        default=True,
        help="Run transformed experiments",
    )
    parser.add_argument(
        "--skip-standard",
        action="store_true",
        help="Skip standard experiments",
    )
    parser.add_argument(
        "--skip-transformed",
        action="store_true",
        help="Skip transformed experiments",
    )
    parser.add_argument(
        "--run-linear",
        action="store_true",
        default=True,
        help="Run linear transform experiments",
    )
    parser.add_argument(
        "--run-nonlinear",
        action="store_true",
        default=True,
        help="Run nonlinear transform experiments",
    )
    parser.add_argument(
        "--skip-linear",
        action="store_true",
        help="Skip linear transform experiments",
    )
    parser.add_argument(
        "--skip-nonlinear",
        action="store_true",
        help="Skip nonlinear transform experiments",
    )
    
    # Model settings
    parser.add_argument(
        "--model-sizes",
        type=str,
        default="small,medium,large",
        help="Comma-separated list of model sizes to test",
    )
    
    # Data settings
    parser.add_argument(
        "--dataset-sizes",
        type=str,
        default="1M,10M,100M",
        help="Comma-separated list of dataset sizes to test (e.g., 1M, 10M, 100M)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size",
    )
    
    # Training settings
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1000,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=500,
        help="Evaluation frequency",
    )
    
    # Output settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output directory",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only generate plots from existing results",
    )
    
    # WandB settings
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="scaling-laws",
        help="WandB project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="WandB entity name",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging",
    )
    
    # Device settings
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Use mixed precision training",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable mixed precision training",
    )
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        default=True,
        help="Use multiple GPUs if available",
    )
    parser.add_argument(
        "--no-multi-gpu",
        action="store_true",
        help="Disable multi-GPU training even if multiple GPUs are available",
    )
    
    # Other settings
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    
    return parser.parse_args()


def parse_model_sizes(model_sizes_str):
    """Parse model sizes from string."""
    size_map = {
        "tiny": [0],      # ~1M params
        "small": [0, 1],  # ~1-10M params
        "medium": [2, 3], # ~10-100M params
        "large": [4, 5],  # ~100M-1B params
        "xlarge": [6],    # >1B params
    }
    
    model_indices = []
    for size in model_sizes_str.split(","):
        size = size.strip().lower()
        if size in size_map:
            model_indices.extend(size_map[size])
        elif size == "all":
            model_indices = list(range(7))
            break
        else:
            try:
                # Try to parse as index
                idx = int(size)
                model_indices.append(idx)
            except ValueError:
                print(f"Warning: Unknown model size '{size}', skipping")
    
    return sorted(set(model_indices))


def parse_dataset_sizes(dataset_sizes_str):
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


def main():
    """Main function."""
    args = parse_args()
    
    # Create experiment config
    config = ExperimentConfig()
    
    # Update config from arguments
    config.experiment_name = args.experiment_name
    config.run_standard = args.run_standard and not args.skip_standard
    config.run_transformed = args.run_transformed and not args.skip_transformed
    config.run_linear_transform = args.run_linear and not args.skip_linear
    config.run_nonlinear_transform = args.run_nonlinear and not args.skip_nonlinear
    
    # Parse model and dataset sizes
    if args.model_sizes != "all":
        model_indices = parse_model_sizes(args.model_sizes)
        all_hidden_sizes = config.model_config.hidden_sizes
        all_num_heads = config.model_config.num_attention_heads
        all_num_layers = config.model_config.num_hidden_layers
        
        config.model_config.hidden_sizes = [all_hidden_sizes[i] for i in model_indices if i < len(all_hidden_sizes)]
        config.model_config.num_attention_heads = [all_num_heads[i] for i in model_indices if i < len(all_num_heads)]
        config.model_config.num_hidden_layers = [all_num_layers[i] for i in model_indices if i < len(all_num_layers)]
    
    config.data_config.dataset_sizes = parse_dataset_sizes(args.dataset_sizes)
    config.data_config.train_batch_size = args.batch_size
    config.data_config.num_workers = args.num_workers
    
    config.training_config.learning_rate = args.learning_rate
    config.training_config.warmup_steps = args.warmup_steps
    config.training_config.eval_steps = args.eval_steps
    config.training_config.output_dir = args.output_dir
    config.training_config.fp16 = args.fp16 and not args.no_fp16
    config.training_config.seed = args.seed
    
    config.use_wandb = not args.no_wandb
    config.wandb_project = args.wandb_project
    config.wandb_entity = args.wandb_entity
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.training_config.output_dir = os.path.join(
        args.output_dir,
        f"{config.experiment_name}_{timestamp}"
    )
    os.makedirs(config.training_config.output_dir, exist_ok=True)
    
    # Save config
    import json
    config_path = os.path.join(config.training_config.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config.__dict__, f, indent=2, default=str)
    
    print("=" * 80)
    print("SCALING LAW EXPERIMENTS")
    print("=" * 80)
    print(f"Experiment name: {config.experiment_name}")
    print(f"Output directory: {config.training_config.output_dir}")
    print(f"Device: {config.device}")
    print(f"Model sizes: {len(config.model_config.hidden_sizes)}")
    print(f"Dataset sizes: {config.data_config.dataset_sizes}")
    print(f"Run standard: {config.run_standard}")
    print(f"Run transformed: {config.run_transformed}")
    print(f"Run linear: {config.run_linear_transform}")
    print(f"Run nonlinear: {config.run_nonlinear_transform}")
    print("=" * 80)
    
    if args.plot_only:
        # Load existing results and generate plots
        import pandas as pd
        results_file = os.path.join(config.training_config.output_dir, "results", "scaling_law_results.csv")
        if not os.path.exists(results_file):
            print(f"Error: Results file not found at {results_file}")
            return
        
        df = pd.read_csv(results_file)
        plot_dir = os.path.join(config.training_config.output_dir, "plots")
        
        # Generate standard plots
        if "standard" in df["experiment_type"].values:
            plot_all_scaling_laws(df, plot_dir, experiment_type="standard")
        
        # Generate transformed plots
        if "transformed" in df["experiment_type"].values:
            plot_all_scaling_laws(df, plot_dir, experiment_type="transformed")
        
        # Compare standard vs transformed
        if len(df["experiment_type"].unique()) > 1:
            compare_standard_vs_transformed(
                df,
                save_path=os.path.join(plot_dir, "comparison.png")
            )
    else:
        # Run experiments
        experiment = ScalingLawExperiment(config)
        results = experiment.run_all_experiments()
        
        # Generate plots
        if results:
            df_results = experiment.analyze_results()
            plot_dir = os.path.join(config.training_config.output_dir, "plots")
            
            # Generate standard plots
            if config.run_standard:
                plot_all_scaling_laws(df_results["all"], plot_dir, experiment_type="standard")
            
            # Generate transformed plots  
            if config.run_transformed:
                plot_all_scaling_laws(df_results["all"], plot_dir, experiment_type="transformed")
            
            # Generate linear transform plots
            if config.run_linear_transform:
                plot_all_scaling_laws(df_results["all"], plot_dir, experiment_type="linear")
            
            # Generate nonlinear transform plots
            if config.run_nonlinear_transform:
                plot_all_scaling_laws(df_results["all"], plot_dir, experiment_type="nonlinear")
            
            # Compare standard vs transformed
            if config.run_standard and config.run_transformed:
                compare_standard_vs_transformed(
                    df_results["all"],
                    save_path=os.path.join(plot_dir, "comparison.png")
                )
            
            # Compare standard vs linear
            if config.run_standard and config.run_linear_transform:
                compare_experiment_types(
                    df_results["all"],
                    type1='standard',
                    type2='linear',
                    save_path=os.path.join(plot_dir, "comparison_linear.png")
                )
            
            # Compare standard vs nonlinear  
            if config.run_standard and config.run_nonlinear_transform:
                compare_experiment_types(
                    df_results["all"],
                    type1='standard',
                    type2='nonlinear',
                    save_path=os.path.join(plot_dir, "comparison_nonlinear.png")
                )
            
            print("\n" + "=" * 80)
            print("EXPERIMENT COMPLETE")
            print("=" * 80)
            print(f"Results saved to: {config.training_config.output_dir}")
            print(f"Plots saved to: {plot_dir}")
            print("=" * 80)


if __name__ == "__main__":
    main() 