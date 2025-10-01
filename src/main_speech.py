"""Main entry point for speech scaling law experiments on LibriSpeech."""

import argparse
import os
import sys
import torch
import wandb
from datetime import datetime

from config import ExperimentConfig
from experiments.speech_scaling_law_experiment import SpeechScalingLawExperiment
from visualization import plot_all_scaling_laws, compare_standard_vs_transformed
from visualization.plots import compare_experiment_types
from training import set_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run speech scaling law experiments on LibriSpeech")

    # Experiment settings
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="speech_scaling_laws",
        help="Name of the experiment",
    )
    parser.add_argument(
        "--librispeech-root",
        type=str,
        default="/cis/home/xhan56/LibriSpeech",
        help="Root directory containing LibriSpeech data",
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
    parser.add_argument(
        "--run-linear",
        action="store_true",
        default=False,
        help="Run linear transform experiments",
    )
    parser.add_argument(
        "--run-nonlinear",
        action="store_true",
        default=False,
        help="Run nonlinear transform experiments",
    )
    parser.add_argument(
        "--run-quantization",
        action="store_true",
        default=False,
        help="Run quantization transform experiments",
    )
    parser.add_argument(
        "--run-lowrank",
        action="store_true",
        default=False,
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

    # Model settings
    parser.add_argument(
        "--model-sizes",
        type=str,
        default="0,1,2",
        help="Comma-separated list of model sizes to test (0,1,2,3,4)",
    )

    # Data settings
    parser.add_argument(
        "--dataset-hours",
        type=str,
        default="10,50,100,500",
        help="Comma-separated list of dataset hours to test (e.g., 10,50,100,500,960)",
    )
    parser.add_argument(
        "--max-audio-length",
        type=float,
        default=20.0,
        help="Maximum audio length in seconds",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
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
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum training steps (if None, calculated based on data)",
    )

    # Output settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs_speech",
        help="Output directory",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only generate plots from existing results",
    )
    parser.add_argument(
        "--run-quantization-analysis",
        action="store_true",
        help="Run quantization analysis across multiple quantization levels",
    )
    parser.add_argument(
        "--quantization-analysis-levels",
        type=str,
        default="2,4,8,16,32",
        help="Comma-separated list of quantization levels to test (default: 2,4,8,16,32)",
    )

    # WandB settings
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="speech-scaling-laws",
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

    # Transform settings
    parser.add_argument(
        "--quantization-levels",
        type=int,
        default=16,
        help="Number of quantization levels",
    )
    parser.add_argument(
        "--lowrank-rank",
        type=int,
        default=16,
        help="Rank for low-rank transform",
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
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./dataset_cache",
        help="Directory to store cached datasets",
    )

    return parser.parse_args()


def parse_model_sizes(model_sizes_str):
    """Parse model sizes from string."""
    # Support both numerical indices and legacy named sizes
    size_map = {
        "tiny": 0,
        "small": 1,
        "medium": 2,
        "large": 3,
        "xlarge": 4,
    }

    model_indices = []
    for size in model_sizes_str.split(","):
        size = size.strip()

        # First try to parse as integer
        try:
            idx = int(size)
            if 0 <= idx <= 4:  # Valid range for speech models
                model_indices.append(idx)
            else:
                print(f"Warning: Model size index '{idx}' out of range (0-4), skipping")
        except ValueError:
            # If not an integer, try named sizes
            size_lower = size.lower()
            if size_lower in size_map:
                model_indices.append(size_map[size_lower])
            elif size_lower == "all":
                model_indices = list(range(5))
                break
            else:
                print(f"Warning: Unknown model size '{size}', skipping")

    return sorted(set(model_indices))


def parse_dataset_hours(dataset_hours_str):
    """Parse dataset hours from string."""
    hours = []
    for hour_str in dataset_hours_str.split(","):
        hour_str = hour_str.strip()
        if hour_str.endswith("h"):
            hour_str = hour_str[:-1]
        try:
            hours.append(int(hour_str))
        except ValueError:
            print(f"Warning: Cannot parse dataset hours '{hour_str}', skipping")

    return sorted(hours)


def main():
    """Main function."""
    args = parse_args()

    # Check LibriSpeech root directory
    if not os.path.exists(args.librispeech_root):
        print(f"Error: LibriSpeech root directory not found: {args.librispeech_root}")
        print("Please provide the correct path using --librispeech-root")
        return

    # Create experiment config
    config = ExperimentConfig()

    # Update config from arguments
    config.experiment_name = args.experiment_name
    config.run_standard = args.run_standard and not args.skip_standard
    config.run_linear_transform = args.run_linear and not args.skip_linear
    config.run_nonlinear_transform = args.run_nonlinear and not args.skip_nonlinear
    config.run_quantization_transform = args.run_quantization and not args.skip_quantization
    config.run_lowrank_transform = args.run_lowrank and not args.skip_lowrank

    # Set dataset sizes from hours
    dataset_hours = parse_dataset_hours(args.dataset_hours)
    config.data_config.dataset_sizes = dataset_hours  # Will be converted to samples later

    # Set model sizes
    model_indices = parse_model_sizes(args.model_sizes)
    print(f"Using model indices: {model_indices}")

    # Audio and batch settings
    config.data_config.max_length = args.max_audio_length
    config.data_config.train_batch_size = args.batch_size
    config.data_config.eval_batch_size = args.batch_size
    config.data_config.num_workers = args.num_workers

    # Training settings
    config.training_config.learning_rate = args.learning_rate
    config.training_config.warmup_steps = args.warmup_steps
    config.training_config.eval_steps = args.eval_steps
    if args.max_steps:
        config.training_config.max_steps = args.max_steps
    config.training_config.output_dir = args.output_dir
    config.training_config.fp16 = args.fp16 and not args.no_fp16
    config.training_config.seed = args.seed

    # Device settings
    config._device = args.device
    config._multi_gpu = args.multi_gpu and not args.no_multi_gpu

    # WandB settings
    config.use_wandb = not args.no_wandb
    config.wandb_project = args.wandb_project
    config.wandb_entity = args.wandb_entity

    # Transform settings
    config.transform_config.quantization_num_levels = args.quantization_levels
    config.transform_config.lowrank_rank = args.lowrank_rank

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
    print("SPEECH SCALING LAW EXPERIMENTS")
    print("=" * 80)
    print(f"Experiment name: {config.experiment_name}")
    print(f"LibriSpeech root: {args.librispeech_root}")
    print(f"Output directory: {config.training_config.output_dir}")
    print(f"Device: {config.device}")
    print(f"Model indices: {model_indices}")
    print(f"Dataset hours: {dataset_hours}")
    print(f"Max audio length: {args.max_audio_length} seconds")
    print(f"Batch size: {args.batch_size}")
    print(f"Run standard: {config.run_standard}")
    print(f"Run linear: {config.run_linear_transform}")
    print(f"Run nonlinear: {config.run_nonlinear_transform}")
    print(f"Run quantization: {config.run_quantization_transform}")
    print(f"Run low-rank: {config.run_lowrank_transform}")
    if config.run_quantization_transform:
        print(f"  Quantization levels: {config.transform_config.quantization_num_levels}")
    if config.run_lowrank_transform:
        print(f"  Low-rank rank: {config.transform_config.lowrank_rank}")
    print("=" * 80)

    if args.plot_only:
        # Load existing results and generate plots
        import pandas as pd
        results_file = os.path.join(config.training_config.output_dir, "results", "speech_scaling_law_results.csv")
        if not os.path.exists(results_file):
            print(f"Error: Results file not found at {results_file}")
            return

        df = pd.read_csv(results_file)
        plot_dir = os.path.join(config.training_config.output_dir, "plots")

        # Generate plots for each experiment type
        if "standard" in df["experiment_type"].values:
            plot_all_scaling_laws(df, plot_dir, experiment_type="standard")

        if "linear" in df["experiment_type"].values:
            plot_all_scaling_laws(df, plot_dir, experiment_type="linear")

        if "nonlinear" in df["experiment_type"].values:
            plot_all_scaling_laws(df, plot_dir, experiment_type="nonlinear")

        if "quantization" in df["experiment_type"].values:
            plot_all_scaling_laws(df, plot_dir, experiment_type="quantization")

        if "lowrank" in df["experiment_type"].values:
            plot_all_scaling_laws(df, plot_dir, experiment_type="lowrank")

        # Generate comparison plots
        experiment_types = df["experiment_type"].unique()
        if len(experiment_types) > 1:
            for i, exp_type in enumerate(experiment_types):
                if exp_type != "standard":
                    compare_experiment_types(
                        df,
                        type1='standard',
                        type2=exp_type,
                        save_path=os.path.join(plot_dir, f"comparison_{exp_type}.png")
                    )

    elif args.run_quantization_analysis:
        # Run quantization analysis
        from librispeech_quantization_analysis import analyze_quantization_scaling_laws

        # Parse quantization levels
        quantization_levels = [int(x.strip()) for x in args.quantization_analysis_levels.split(",")]
        print(f"Running quantization analysis with levels: {quantization_levels}")

        # Create output directory for quantization analysis
        quant_output_dir = os.path.join(config.training_config.output_dir, "quantization_analysis")
        os.makedirs(quant_output_dir, exist_ok=True)

        # Run quantization analysis
        results_df = analyze_quantization_scaling_laws(
            quantization_levels, config, args.librispeech_root, args.cache_dir, quant_output_dir
        )

        if len(results_df) > 0:
            print(f"\nQuantization analysis completed with {len(results_df)} levels tested")
            print(f"Results saved to: {quant_output_dir}")
        else:
            print("No quantization analysis results obtained!")

    else:
        # Run experiments
        experiment = SpeechScalingLawExperiment(config, args.librispeech_root, model_indices, args.cache_dir)
        results = experiment.run_all_experiments()

        # Generate plots
        if results:
            df_results = experiment.analyze_results()
            plot_dir = os.path.join(config.training_config.output_dir, "plots")

            # Generate plots for each experiment type
            if config.run_standard:
                plot_all_scaling_laws(df_results["all"], plot_dir, experiment_type="standard")

            if config.run_linear_transform:
                plot_all_scaling_laws(df_results["all"], plot_dir, experiment_type="linear")

            if config.run_nonlinear_transform:
                plot_all_scaling_laws(df_results["all"], plot_dir, experiment_type="nonlinear")

            if config.run_quantization_transform:
                plot_all_scaling_laws(df_results["all"], plot_dir, experiment_type="quantization")

            if config.run_lowrank_transform:
                plot_all_scaling_laws(df_results["all"], plot_dir, experiment_type="lowrank")

            # Generate comparison plots
            experiment_types = []
            if config.run_standard:
                experiment_types.append("standard")
            if config.run_linear_transform:
                experiment_types.append("linear")
            if config.run_nonlinear_transform:
                experiment_types.append("nonlinear")
            if config.run_quantization_transform:
                experiment_types.append("quantization")
            if config.run_lowrank_transform:
                experiment_types.append("lowrank")

            # Compare standard vs each transform type
            for exp_type in experiment_types:
                if exp_type != "standard" and config.run_standard:
                    compare_experiment_types(
                        df_results["all"],
                        type1='standard',
                        type2=exp_type,
                        save_path=os.path.join(plot_dir, f"comparison_{exp_type}.png")
                    )

            print("\n" + "=" * 80)
            print("SPEECH EXPERIMENT COMPLETE")
            print("=" * 80)
            print(f"Results saved to: {config.training_config.output_dir}")
            print(f"Plots saved to: {plot_dir}")
            print("=" * 80)

            # Print summary of results
            if df_results["all"] is not None and len(df_results["all"]) > 0:
                print("\nExperiment Summary:")
                for exp_type in experiment_types:
                    exp_df = df_results["all"][df_results["all"]["experiment_type"] == exp_type]
                    if len(exp_df) > 0:
                        best_wer = exp_df["dev_wer"].min() if "dev_wer" in exp_df.columns else "N/A"
                        print(f"  {exp_type}: Best WER = {best_wer}")


if __name__ == "__main__":
    main()