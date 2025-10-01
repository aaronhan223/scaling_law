"""Main entry point for image scaling law experiments on ImageNet."""

import argparse
import os
import sys
import torch
import wandb
from datetime import datetime

from config import ExperimentConfig
from experiments.image_scaling_law_experiment import ImageScalingLawExperiment
from visualization import plot_all_scaling_laws, compare_standard_vs_transformed
from visualization.plots import compare_experiment_types
from training import set_seed


def parse_args():
    """Parse command line arguments for image experiments."""
    parser = argparse.ArgumentParser(description="Run image scaling law experiments on ImageNet")

    # Experiment settings
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="image_scaling_laws",
        help="Name of the experiment",
    )
    parser.add_argument(
        "--run-standard",
        action="store_true",
        default=True,
        help="Run standard experiments",
    )
    parser.add_argument(
        "--skip-standard",
        action="store_true",
        help="Skip standard experiments",
    )

    # Transform experiment flags
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
        "--run-quantization",
        action="store_true",
        default=True,
        help="Run quantization transform experiments",
    )
    parser.add_argument(
        "--run-lowrank",
        action="store_true",
        default=True,
        help="Run low-rank transform experiments",
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
    parser.add_argument(
        "--skip-quantization",
        action="store_true",
        help="Skip quantization transform experiments",
    )
    parser.add_argument(
        "--skip-lowrank",
        action="store_true",
        help="Skip low-rank transform experiments",
    )

    # Quantization analysis specific flags
    parser.add_argument(
        "--run-quantization-analysis",
        action="store_true",
        help="Run quantization analysis across multiple quantization levels",
    )
    parser.add_argument(
        "--quantization-levels",
        type=str,
        default="2,4,8,16,32,64,128,256",
        help="Comma-separated list of quantization levels to test (default: 2,4,8,16,32,64,128,256)",
    )

    # Model settings
    parser.add_argument(
        "--model-sizes",
        type=str,
        default="tiny,small,base",
        help="Comma-separated list of vision model sizes to test (tiny, small, base, large)",
    )

    # Data settings
    parser.add_argument(
        "--dataset-sizes",
        type=str,
        default="1K,10K,50K",
        help="Comma-separated list of dataset sizes to test (e.g., 1K, 10K, 50K)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Training batch size",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=256,
        help="Evaluation batch size",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Input image size",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=1000,
        help="Number of classes (subset of ImageNet)",
    )

    # Training settings
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of training epochs",
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
        default="image-scaling-laws",
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
        help="Disable multi-GPU training",
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
        default=8,
        help="Number of data loading workers",
    )

    return parser.parse_args()


def parse_model_sizes(model_sizes_str):
    """Parse vision model sizes from string."""
    size_map = {
        "tiny": 0,
        "small": 1,
        "base": 2,
        "large": 3,
    }

    model_indices = []
    for size in model_sizes_str.split(","):
        size = size.strip().lower()
        if size in size_map:
            model_indices.append(size_map[size])
        elif size == "all":
            model_indices = list(range(4))
            break
        else:
            try:
                # Try to parse as index
                idx = int(size)
                if 0 <= idx <= 3:
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
        elif size.endswith("K"):
            sizes.append(int(float(size[:-1]) * 1e3))
        else:
            try:
                sizes.append(int(size))
            except ValueError:
                print(f"Warning: Cannot parse dataset size '{size}', skipping")

    return sorted(sizes)


def main():
    """Main function for image experiments."""
    args = parse_args()

    # Create experiment config adapted for image experiments
    config = ExperimentConfig()

    # Update config from arguments
    config.experiment_name = args.experiment_name
    config.run_standard = args.run_standard and not args.skip_standard
    config.run_linear_transform = args.run_linear and not args.skip_linear
    config.run_nonlinear_transform = args.run_nonlinear and not args.skip_nonlinear
    config.run_quantization_transform = args.run_quantization and not args.skip_quantization
    config.run_lowrank_transform = args.run_lowrank and not args.skip_lowrank

    # Image-specific config
    config.img_size = args.img_size
    config.num_classes = args.num_classes

    # Model configuration for images
    model_indices = parse_model_sizes(args.model_sizes)
    config.model_indices = model_indices

    # Data configuration
    config.data_config.dataset_sizes = parse_dataset_sizes(args.dataset_sizes)
    config.data_config.train_batch_size = args.batch_size
    config.data_config.eval_batch_size = args.eval_batch_size
    config.data_config.num_workers = args.num_workers

    # Training configuration
    config.training_config.learning_rate = args.learning_rate
    config.training_config.warmup_steps = args.warmup_steps
    config.training_config.eval_steps = args.eval_steps
    config.training_config.fp16 = args.fp16 and not args.no_fp16
    config.training_config.seed = args.seed
    config.training_config.num_epochs = args.num_epochs
    config.training_config.output_dir = args.output_dir

    # Device settings - store in a way that can be accessed later
    config._device = args.device
    config._multi_gpu = args.multi_gpu and not args.no_multi_gpu

    # WandB settings
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
    config_dict = {
        "experiment_name": config.experiment_name,
        "img_size": config.img_size,
        "num_classes": config.num_classes,
        "model_indices": getattr(config, 'model_indices', []),
        "dataset_sizes": config.data_config.dataset_sizes,
        "batch_size": config.data_config.train_batch_size,
        "learning_rate": config.training_config.learning_rate,
        "num_epochs": config.training_config.num_epochs,
        "run_standard": config.run_standard,
        "run_linear": config.run_linear_transform,
        "run_nonlinear": config.run_nonlinear_transform,
        "run_quantization": config.run_quantization_transform,
        "run_lowrank": config.run_lowrank_transform,
    }

    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    print("=" * 80)
    print("IMAGE SCALING LAW EXPERIMENTS")
    print("=" * 80)
    print(f"Experiment name: {config.experiment_name}")
    print(f"Output directory: {config.training_config.output_dir}")
    print(f"Device: {config.device}")
    print(f"Image size: {config.img_size}")
    print(f"Model indices: {getattr(config, 'model_indices', 'all')}")
    print(f"Dataset sizes: {config.data_config.dataset_sizes}")
    print(f"Batch size: {config.data_config.train_batch_size}")
    print(f"Number of epochs: {config.training_config.num_epochs}")
    print(f"Run standard: {config.run_standard}")
    print(f"Run linear: {config.run_linear_transform}")
    print(f"Run nonlinear: {config.run_nonlinear_transform}")
    print(f"Run quantization: {config.run_quantization_transform}")
    print(f"Run low-rank: {config.run_lowrank_transform}")
    print("=" * 80)

    if args.plot_only:
        # Load existing results and generate plots
        import pandas as pd
        results_file = os.path.join(config.training_config.output_dir, "results", "image_scaling_law_results.csv")
        if not os.path.exists(results_file):
            print(f"Error: Results file not found at {results_file}")
            return

        df = pd.read_csv(results_file)
        plot_dir = os.path.join(config.training_config.output_dir, "plots")

        # Generate plots for each experiment type
        for exp_type in df["experiment_type"].unique():
            plot_all_scaling_laws(df, plot_dir, experiment_type=exp_type)

        # Generate comparison plots
        if len(df["experiment_type"].unique()) > 1:
            for exp_type in ["linear", "nonlinear", "quantization", "lowrank"]:
                if exp_type in df["experiment_type"].values and "standard" in df["experiment_type"].values:
                    compare_experiment_types(
                        df,
                        type1='standard',
                        type2=exp_type,
                        save_path=os.path.join(plot_dir, f"comparison_{exp_type}.png")
                    )
    else:
        # Check if quantization analysis is requested
        if args.run_quantization_analysis:
            # Run quantization analysis
            from imagenet_quantization_analysis import run_imagenet_quantization_analysis, extract_scaling_parameters

            print("\n" + "="*80)
            print("RUNNING IMAGENET QUANTIZATION ANALYSIS")
            print("="*80)

            # Parse quantization levels
            quantization_levels = [int(x.strip()) for x in args.quantization_levels.split(",")]

            print(f"Testing {len(quantization_levels)} quantization levels: {quantization_levels}")

            # Run quantization analysis
            results_df = run_imagenet_quantization_analysis(quantization_levels, config, config.training_config.output_dir)

            # Generate quantization-specific plots
            if len(results_df) > 0:
                try:
                    from visualization.quantization_plots import plot_imagenet_quantization_analysis
                    plot_dir = os.path.join(config.training_config.output_dir, "plots")
                    plot_imagenet_quantization_analysis(results_df, plot_dir)
                    print(f"Quantization analysis plots saved to: {plot_dir}")
                except ImportError:
                    print("Warning: Could not import quantization plotting functions")
        else:
            # Run experiments
            experiment = ImageScalingLawExperiment(config)
            results = experiment.run_all_experiments()

            # Generate plots
            plot_dir = os.path.join(config.training_config.output_dir, "plots")
            if results:
                df_results = experiment.analyze_results()

                # Generate plots for each experiment type
                for exp_type, df in df_results.items():
                    if exp_type != "all" and not df.empty:
                        plot_all_scaling_laws(df, plot_dir, experiment_type=exp_type)

                # Generate comparison plots
                if "standard" in df_results and not df_results["standard"].empty:
                    for exp_type in ["linear", "nonlinear", "quantization", "lowrank"]:
                        if exp_type in df_results and not df_results[exp_type].empty:
                            compare_experiment_types(
                                df_results["all"],
                                type1='standard',
                                type2=exp_type,
                                save_path=os.path.join(plot_dir, f"image_comparison_{exp_type}.png")
                            )

            print("\n" + "=" * 80)
            print("IMAGE EXPERIMENT COMPLETE")
            print("=" * 80)
            print(f"Results saved to: {config.training_config.output_dir}")
            print(f"Plots saved to: {plot_dir}")
            print("=" * 80)


if __name__ == "__main__":
    main()