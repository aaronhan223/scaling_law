#!/usr/bin/env python3
"""Standalone script to visualize scaling law experiment results."""

import argparse
import os
import pandas as pd
from visualization import plot_all_scaling_laws, compare_standard_vs_transformed


def main():
    parser = argparse.ArgumentParser(description="Visualize scaling law results")
    parser.add_argument(
        "results_file",
        type=str,
        help="Path to the CSV results file (e.g., outputs/experiment/results/scaling_law_results.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots (default: same directory as results)",
    )
    parser.add_argument(
        "--experiment-type",
        type=str,
        default="all",
        choices=["all", "standard", "transformed"],
        help="Which experiment type to plot",
    )
    
    args = parser.parse_args()
    
    # Load results
    if not os.path.exists(args.results_file):
        print(f"Error: Results file not found: {args.results_file}")
        return
    
    print(f"Loading results from: {args.results_file}")
    df = pd.read_csv(args.results_file)
    
    # Determine output directory
    if args.output_dir is None:
        output_dir = os.path.join(os.path.dirname(args.results_file), "..", "plots")
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving plots to: {output_dir}")
    
    # Generate plots based on experiment type
    experiment_types = df["experiment_type"].unique()
    
    if args.experiment_type == "all":
        # Plot all experiment types
        for exp_type in experiment_types:
            if exp_type == "standard" or exp_type == "transformed":
                print(f"\nGenerating {exp_type} plots...")
                plot_all_scaling_laws(df, output_dir, experiment_type=exp_type)
        
        # Generate comparison if both types exist
        if "standard" in experiment_types and "transformed" in experiment_types:
            print("\nGenerating comparison plots...")
            compare_standard_vs_transformed(
                df,
                save_path=os.path.join(output_dir, "comparison.png")
            )
    else:
        # Plot specific experiment type
        if args.experiment_type in experiment_types:
            print(f"\nGenerating {args.experiment_type} plots...")
            plot_all_scaling_laws(df, output_dir, experiment_type=args.experiment_type)
        else:
            print(f"Error: No {args.experiment_type} experiments found in results")
    
    print("\nPlotting complete!")


if __name__ == "__main__":
    main() 