#!/usr/bin/env python3
"""
Aggregate and plot results for Image Scaling Law Experiments.

This script aggregates results from the image_scaling_laws_20250916_074801 experiment directory
and generates comprehensive scaling law plots for all completed experiments.

Usage:
    python aggregate_and_plot_image_results.py --input-dir ./outputs/image_scaling_laws_20250916_074801
"""

import argparse
import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import glob

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from visualization.plots import (
    setup_plot_style,
    plot_loss_vs_parameters,
    plot_loss_vs_dataset_size,
    compare_experiment_types
)


def extract_experiment_info(file_path: str) -> Dict:
    """Extract experiment information from file path."""
    parts = Path(file_path).parts

    # Expected structure: .../image_scaling_laws/[experiment_type]/[model_name]/[data_size]/...
    experiment_type = None
    model_name = None
    data_size = None

    for i, part in enumerate(parts):
        if part == 'image_scaling_laws' and i + 3 < len(parts):
            experiment_type = parts[i + 1]
            model_name = parts[i + 2]
            data_size = parts[i + 3]
            break

    return {
        'experiment_type': experiment_type,
        'model_name': model_name,
        'data_size': data_size
    }


def load_checkpoint_results(checkpoint_dir: str) -> Optional[Dict]:
    """Load results from a checkpoint directory."""
    checkpoint_path = Path(checkpoint_dir)

    # Look for results files
    result_files = list(checkpoint_path.glob("*.json"))

    if not result_files:
        # Look in subdirectories for results
        for subdir in ["best_model", "final_model"]:
            subdir_path = checkpoint_path / subdir
            if subdir_path.exists():
                result_files.extend(list(subdir_path.glob("*.json")))

    if not result_files:
        return None

    # Load the most recent results file
    result_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    try:
        with open(result_files[0], 'r') as f:
            results = json.load(f)
        return results
    except (json.JSONDecodeError, FileNotFoundError):
        return None


def collect_experiment_results(input_dir: str) -> pd.DataFrame:
    """Collect all experiment results from the input directory."""

    base_path = Path(input_dir) / "image_scaling_laws"

    if not base_path.exists():
        print(f"Error: {base_path} does not exist")
        return pd.DataFrame()

    all_results = []

    # Model configurations (matching main_image.py)
    model_configs = {
        'vit_tiny': {'params': 5_500_000, 'embed_dim': 192, 'layers': 12, 'heads': 3},
        'vit_small': {'params': 22_000_000, 'embed_dim': 384, 'layers': 12, 'heads': 6},
        'vit_base': {'params': 86_000_000, 'embed_dim': 768, 'layers': 12, 'heads': 12},
        'vit_large': {'params': 307_000_000, 'embed_dim': 1024, 'layers': 24, 'heads': 16}
    }

    # Scan for experiment directories
    for exp_type_dir in base_path.iterdir():
        if not exp_type_dir.is_dir():
            continue

        experiment_type = exp_type_dir.name
        print(f"Processing {experiment_type} experiments...")

        for model_dir in exp_type_dir.iterdir():
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name

            for data_dir in model_dir.iterdir():
                if not data_dir.is_dir() or not data_dir.name.startswith('data_'):
                    continue

                data_size_str = data_dir.name.replace('data_', '')
                try:
                    data_size = int(data_size_str)
                except ValueError:
                    continue

                print(f"  {model_name}/{data_dir.name}")

                # Load results from this experiment
                results = load_checkpoint_results(str(data_dir))

                if results is None:
                    print(f"    No results found for {experiment_type}/{model_name}/{data_dir.name}")
                    continue

                # Extract relevant metrics
                experiment_result = {
                    'experiment_type': experiment_type,
                    'model_name': model_name,
                    'model_params': model_configs.get(model_name, {}).get('params', 0),
                    'data_size': data_size,
                    'final_loss': results.get('final_loss', results.get('best_loss', np.nan)),
                    'best_loss': results.get('best_loss', results.get('final_loss', np.nan)),
                    'final_accuracy': results.get('final_accuracy', results.get('best_accuracy', np.nan)),
                    'best_accuracy': results.get('best_accuracy', results.get('final_accuracy', np.nan)),
                    'training_time': results.get('training_time', np.nan),
                    'total_epochs': results.get('total_epochs', results.get('epochs_trained', np.nan))
                }

                all_results.append(experiment_result)

    df = pd.DataFrame(all_results)
    print(f"\nLoaded {len(df)} experiment results")

    # Sort for consistent ordering
    if len(df) > 0:
        df = df.sort_values(['experiment_type', 'model_name', 'data_size']).reset_index(drop=True)

    return df


def generate_summary_stats(df: pd.DataFrame) -> Dict:
    """Generate summary statistics from the results."""

    if len(df) == 0:
        return {}

    summary = {
        'total_experiments': len(df),
        'experiment_types': df['experiment_type'].unique().tolist(),
        'model_types': df['model_name'].unique().tolist(),
        'data_sizes': sorted(df['data_size'].unique().tolist()),
        'experiments_by_type': df['experiment_type'].value_counts().to_dict(),
        'best_overall_accuracy': df['best_accuracy'].max() if 'best_accuracy' in df.columns else None,
        'best_overall_loss': df['best_loss'].min() if 'best_loss' in df.columns else None
    }

    return summary


def plot_scaling_laws(df: pd.DataFrame, output_dir: str):
    """Generate comprehensive scaling law plots."""

    if len(df) == 0:
        print("No data to plot")
        return

    output_path = Path(output_dir)
    plots_dir = output_path / "plots"
    plots_dir.mkdir(exist_ok=True)

    colors = setup_plot_style()

    # 1. Loss vs Model Parameters (for each experiment type)
    for exp_type in df['experiment_type'].unique():
        subset = df[df['experiment_type'] == exp_type]

        fig = plot_loss_vs_parameters(
            subset,
            title=f"Test Loss vs Model Parameters - {exp_type.capitalize()}"
        )

        plt.savefig(plots_dir / f"{exp_type}_loss_vs_parameters.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 2. Loss vs Dataset Size (for each experiment type)
    for exp_type in df['experiment_type'].unique():
        subset = df[df['experiment_type'] == exp_type]

        fig = plot_loss_vs_dataset_size(
            subset,
            title=f"Test Loss vs Dataset Size - {exp_type.capitalize()}"
        )

        plt.savefig(plots_dir / f"{exp_type}_loss_vs_dataset_size.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 3. Comparison between experiment types
    if len(df['experiment_type'].unique()) > 1:
        fig = compare_experiment_types(df)
        plt.savefig(plots_dir / "experiment_type_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 4. Accuracy vs Model Parameters
    fig, ax = plt.subplots(figsize=(12, 8))

    for i, exp_type in enumerate(df['experiment_type'].unique()):
        subset = df[df['experiment_type'] == exp_type]
        ax.loglog(
            subset['model_params'],
            subset['best_accuracy'],
            'o-',
            color=colors[i % len(colors)],
            label=exp_type.capitalize(),
            markersize=6
        )

    ax.set_xlabel('Model Parameters')
    ax.set_ylabel('Best Accuracy')
    ax.set_title('Best Accuracy vs Model Parameters')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(plots_dir / "accuracy_vs_parameters.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Training efficiency (accuracy per parameter)
    fig, ax = plt.subplots(figsize=(12, 8))

    df['accuracy_per_param'] = df['best_accuracy'] / df['model_params'] * 1e6  # per million params

    for i, exp_type in enumerate(df['experiment_type'].unique()):
        subset = df[df['experiment_type'] == exp_type]
        ax.semilogx(
            subset['model_params'],
            subset['accuracy_per_param'],
            'o-',
            color=colors[i % len(colors)],
            label=exp_type.capitalize(),
            markersize=6
        )

    ax.set_xlabel('Model Parameters')
    ax.set_ylabel('Accuracy per Million Parameters')
    ax.set_title('Training Efficiency: Accuracy per Million Parameters')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(plots_dir / "training_efficiency.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plots saved to {plots_dir}")


def save_aggregated_results(df: pd.DataFrame, summary: Dict, output_dir: str):
    """Save aggregated results and summary."""

    output_path = Path(output_dir)
    results_dir = output_path / "results"
    results_dir.mkdir(exist_ok=True)

    # Save CSV
    csv_path = results_dir / "image_scaling_law_results_aggregated.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    # Save JSON
    json_path = results_dir / "image_scaling_law_results_aggregated.json"

    # Convert DataFrame to records for JSON serialization
    results_dict = {
        'summary': summary,
        'experiments': df.to_dict('records')
    }

    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)

    print(f"Results saved to {json_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Aggregate and plot image scaling law results")

    parser.add_argument(
        "--input-dir",
        type=str,
        default="./outputs/image_scaling_laws_20250916_074801",
        help="Input directory containing experiment results"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return 1

    print(f"Aggregating results from: {args.input_dir}")

    # Collect results
    df = collect_experiment_results(args.input_dir)

    if len(df) == 0:
        print("No experiment results found!")
        return 1

    # Generate summary
    summary = generate_summary_stats(df)

    print("\n" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)
    print(f"Total experiments: {summary['total_experiments']}")
    print(f"Experiment types: {', '.join(summary['experiment_types'])}")
    print(f"Model types: {', '.join(summary['model_types'])}")
    print(f"Dataset sizes: {summary['data_sizes']}")
    print("\nExperiments by type:")
    for exp_type, count in summary['experiments_by_type'].items():
        print(f"  {exp_type}: {count}")

    if summary['best_overall_accuracy']:
        print(f"\nBest overall accuracy: {summary['best_overall_accuracy']:.4f}")
    if summary['best_overall_loss']:
        print(f"Best overall loss: {summary['best_overall_loss']:.4f}")

    # Save results
    save_aggregated_results(df, summary, args.input_dir)

    # Generate plots
    print(f"\nGenerating plots...")
    plot_scaling_laws(df, args.input_dir)

    print("\n" + "="*50)
    print("AGGREGATION COMPLETE!")
    print("="*50)
    print(f"Results saved in: {args.input_dir}/results/")
    print(f"Plots saved in: {args.input_dir}/plots/")

    return 0


if __name__ == "__main__":
    sys.exit(main())