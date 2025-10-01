#!/usr/bin/env python3
"""
Script to generate scaling law plots from experiment results.

This script plots scaling laws for:
1. Standard, linear, and nonlinear settings from scaling_laws_full_multi_gpu_20250711_113852
2. Quantization vs standard comparison using results from scaling_laws_full_multi_gpu_20250726_102744
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from visualization.plots import (
    plot_all_scaling_laws,
    plot_loss_vs_parameters,
    plot_loss_vs_dataset_size,
    plot_loss_vs_compute,
    plot_optimal_model_size,
    compare_experiment_types
)


def main():
    """Main function to generate all scaling law plots."""
    # Define paths
    base_dir = Path(__file__).parent
    
    # Experiment 1: Standard/linear/nonlinear (incomplete experiment)
    exp1_dir = base_dir / "outputs" / "scaling_laws_full_multi_gpu_20250711_113852"
    exp1_results = exp1_dir / "results" / "scaling_law_results.csv"
    exp1_plots_dir = exp1_dir / "plots"
    
    # Experiment 2: Quantization (complete experiment)
    exp2_dir = base_dir / "outputs" / "scaling_laws_full_multi_gpu_20250726_102744"
    exp2_results = exp2_dir / "results" / "scaling_law_results.csv"
    exp2_plots_dir = exp2_dir / "plots"
    
    # Create output directories
    exp1_plots_dir.mkdir(exist_ok=True)
    exp2_plots_dir.mkdir(exist_ok=True)
    
    print("Generating scaling law plots...")
    
    # Part 1: Process experiment 1 (standard, linear, nonlinear)
    print(f"\n1. Processing experiment 1: {exp1_dir.name}")
    if exp1_results.exists():
        df1 = pd.read_csv(exp1_results)
        print(f"   Loaded {len(df1)} experiments")
        print(f"   Experiment types: {df1['experiment_type'].unique().tolist()}")
        
        # Generate plots for each available experiment type
        available_types = df1['experiment_type'].unique()
        for exp_type in ['standard', 'linear', 'nonlinear']:
            if exp_type in available_types:
                print(f"   Generating plots for {exp_type}...")
                df_filtered = df1[df1['experiment_type'] == exp_type]
                
                # Generate individual plots
                plots = [
                    ("loss_vs_parameters", plot_loss_vs_parameters),
                    ("loss_vs_dataset_size", plot_loss_vs_dataset_size), 
                    ("loss_vs_compute", plot_loss_vs_compute),
                    ("optimal_model_size", plot_optimal_model_size),
                ]
                
                for name, plot_func in plots:
                    save_path = exp1_plots_dir / f"{exp_type}_{name}.png"
                    fig = plot_func(
                        df_filtered,
                        save_path=str(save_path),
                        title=f"{exp_type.capitalize()} - {name.replace('_', ' ').title()}",
                    )
                    plt.close(fig)
                    print(f"     Saved: {save_path.name}")
            else:
                print(f"   Warning: {exp_type} data not found in experiment results")
        
        # Generate comparison plots if multiple types are available
        if len(available_types) > 1:
            print("   Generating comparison plots...")
            comparisons = [
                ('standard', 'linear'),
                ('standard', 'nonlinear'),
                ('linear', 'nonlinear')
            ]
            
            for type1, type2 in comparisons:
                if type1 in available_types and type2 in available_types:
                    save_path = exp1_plots_dir / f"comparison_{type1}_vs_{type2}.png"
                    fig = compare_experiment_types(
                        df1,
                        type1=type1,
                        type2=type2,
                        save_path=str(save_path)
                    )
                    plt.close(fig)
                    print(f"     Saved: {save_path.name}")
    else:
        print(f"   Error: Results file not found: {exp1_results}")
    
    # Part 2: Process experiment 2 (quantization) and comparison
    print(f"\n2. Processing experiment 2: {exp2_dir.name}")
    if exp2_results.exists():
        df2 = pd.read_csv(exp2_results)
        print(f"   Loaded {len(df2)} experiments")
        print(f"   Experiment types: {df2['experiment_type'].unique().tolist()}")
        
        # Generate plots for quantization
        available_types = df2['experiment_type'].unique()
        for exp_type in available_types:
            print(f"   Generating plots for {exp_type}...")
            df_filtered = df2[df2['experiment_type'] == exp_type]
            
            # Generate individual plots
            plots = [
                ("loss_vs_parameters", plot_loss_vs_parameters),
                ("loss_vs_dataset_size", plot_loss_vs_dataset_size),
                ("loss_vs_compute", plot_loss_vs_compute),
                ("optimal_model_size", plot_optimal_model_size),
            ]
            
            for name, plot_func in plots:
                save_path = exp2_plots_dir / f"{exp_type}_{name}.png"
                fig = plot_func(
                    df_filtered,
                    save_path=str(save_path),
                    title=f"{exp_type.capitalize()} - {name.replace('_', ' ').title()}",
                )
                plt.close(fig)
                print(f"     Saved: {save_path.name}")
    else:
        print(f"   Error: Results file not found: {exp2_results}")
    
    # Part 3: Compare quantization vs standard using data from both experiments
    print("\n3. Generating quantization vs standard comparison...")
    if exp1_results.exists() and exp2_results.exists():
        # Load both datasets
        df1 = pd.read_csv(exp1_results)
        df2 = pd.read_csv(exp2_results)
        
        # Filter for standard from exp1 and quantization from exp2
        df_standard = df1[df1['experiment_type'] == 'standard'].copy()
        df_quantization = df2[df2['experiment_type'] == 'quantization'].copy()
        
        if len(df_standard) > 0 and len(df_quantization) > 0:
            # Combine datasets for comparison
            # Rename quantization to transformed for the comparison function
            df_quantization_renamed = df_quantization.copy()
            df_quantization_renamed['experiment_type'] = 'transformed'
            
            # Combine datasets
            df_combined = pd.concat([df_standard, df_quantization_renamed], ignore_index=True)
            
            # Generate comparison plot
            comparison_path = base_dir / "quantization_vs_standard_comparison.png"
            fig = compare_experiment_types(
                df_combined,
                type1='standard',
                type2='transformed',
                save_path=str(comparison_path)
            )
            plt.close(fig)
            print(f"   Saved: {comparison_path.name}")
            
            # Also generate individual comparison plots
            print("   Generating detailed comparison plots...")
            
            # Create comparison plots directory
            comparison_dir = base_dir / "comparison_plots"
            comparison_dir.mkdir(exist_ok=True)
            
            # Loss vs parameters comparison
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot standard results
            if len(df_standard) > 0:
                model_loss_std = df_standard.groupby('model_params')['final_loss'].mean()
                ax.loglog(
                    model_loss_std.index,
                    model_loss_std.values,
                    'o-',
                    label='Standard',
                    markersize=8,
                    linewidth=2,
                )
            
            # Plot quantization results
            if len(df_quantization) > 0:
                model_loss_quant = df_quantization.groupby('model_params')['final_loss'].mean()
                ax.loglog(
                    model_loss_quant.index,
                    model_loss_quant.values,
                    's-',
                    label='Quantization',
                    markersize=8,
                    linewidth=2,
                )
            
            ax.set_xlabel('Model Parameters')
            ax.set_ylabel('Average Test Loss')
            ax.set_title('Scaling Law Comparison: Standard vs Quantization')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            detailed_comparison_path = comparison_dir / "detailed_scaling_comparison.png"
            plt.savefig(detailed_comparison_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"   Saved: {detailed_comparison_path}")
            
        else:
            print("   Warning: Insufficient data for quantization vs standard comparison")
            if len(df_standard) == 0:
                print("     No standard experiments found in experiment 1")
            if len(df_quantization) == 0:
                print("     No quantization experiments found in experiment 2")
    else:
        print("   Error: Cannot perform comparison - missing results files")
    
    print("\n✅ All plots generated successfully!")
    print(f"\nGenerated plots locations:")
    print(f"  Experiment 1 plots: {exp1_plots_dir}")
    print(f"  Experiment 2 plots: {exp2_plots_dir}")
    print(f"  Comparison plots: {base_dir}/comparison_plots")
    print(f"  Main comparison: {base_dir}/quantization_vs_standard_comparison.png")


if __name__ == "__main__":
    main()