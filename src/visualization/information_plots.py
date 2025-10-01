"""
Plotting functions for information density analysis results.

This module provides comprehensive visualization capabilities for analyzing the relationship
between dataset information density and scaling law parameters.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import os
from pathlib import Path


# Set style for consistent plots
plt.style.use('default')
sns.set_palette("husl")


def plot_information_vs_scaling_params(
    df: pd.DataFrame, 
    output_dir: str,
    figsize: Tuple[int, int] = (15, 12)
) -> None:
    """
    Create comprehensive plots showing information metrics vs scaling parameters.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define metric groups (only use columns that exist)
    all_info_metrics = [
        'information_density', 'token_entropy', 'ttr', 'perplexity_complexity',
        'compression_ratio', 'mutual_information', 'semantic_density', 'info_per_token'
    ]
    all_scaling_params = ['alpha', 'beta', 'gamma']
    
    info_metrics = [m for m in all_info_metrics if m in df.columns]
    scaling_params = [s for s in all_scaling_params if s in df.columns]
    
    if not info_metrics or not scaling_params:
        print("Warning: No valid metric combinations found for plotting")
        return
    
    # Create a comprehensive correlation heatmap
    fig, axes = plt.subplots(len(scaling_params), len(info_metrics), 
                            figsize=(20, 12), sharex=False, sharey=False)
    fig.suptitle('Information Metrics vs Scaling Parameters', fontsize=16, fontweight='bold')
    
    for i, scaling_param in enumerate(scaling_params):
        for j, info_metric in enumerate(info_metrics):
            ax = axes[i, j] if len(scaling_params) > 1 else axes[j]
            
            # Get valid data points
            valid_data = df[[info_metric, scaling_param]].dropna()
            
            if len(valid_data) > 2:
                x = valid_data[info_metric]
                y = valid_data[scaling_param]
                
                # Scatter plot
                ax.scatter(x, y, alpha=0.7, s=50)
                
                # Fit and plot trend line
                try:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    line_x = np.linspace(x.min(), x.max(), 100)
                    line_y = slope * line_x + intercept
                    ax.plot(line_x, line_y, 'r--', alpha=0.8, linewidth=2)
                    
                    # Add correlation info
                    ax.text(0.05, 0.95, f'r={r_value:.3f}\np={p_value:.3f}', 
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                except Exception:
                    pass
                
                ax.set_xlabel(info_metric.replace('_', ' ').title())
                ax.set_ylabel(scaling_param.upper())
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'Insufficient\nData', transform=ax.transAxes,
                       ha='center', va='center', fontsize=12, alpha=0.5)
                ax.set_xlabel(info_metric.replace('_', ' ').title())
                ax.set_ylabel(scaling_param.upper())
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'information_vs_scaling_comprehensive.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_correlation_matrix(df: pd.DataFrame, output_dir: str) -> None:
    """Create correlation matrix heatmap."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Select relevant columns
    info_cols = ['information_density', 'token_entropy', 'ttr', 'perplexity_complexity',
                'compression_ratio', 'mutual_information', 'semantic_density', 'info_per_token']
    scaling_cols = ['alpha', 'beta', 'gamma']
    performance_cols = ['mean_loss', 'min_loss']
    
    all_cols = info_cols + scaling_cols + performance_cols
    available_cols = [col for col in all_cols if col in df.columns]
    
    if len(available_cols) < 3:
        print("Warning: Insufficient columns for correlation matrix")
        return
    
    # Compute correlation matrix
    corr_matrix = df[available_cols].corr()
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                square=True, mask=mask, cbar_kws={"shrink": .8},
                fmt='.3f', linewidths=0.5)
    
    plt.title('Information Metrics and Scaling Parameters Correlation Matrix', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_information_density_distribution(df: pd.DataFrame, output_dir: str) -> None:
    """Plot distribution of information density metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    info_metrics = ['information_density', 'token_entropy', 'ttr', 'compression_ratio', 
                   'mutual_information', 'semantic_density']
    available_metrics = [m for m in info_metrics if m in df.columns]
    
    if not available_metrics:
        return
    
    n_metrics = len(available_metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, metric in enumerate(available_metrics):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]
        
        data = df[metric].dropna()
        if len(data) > 0:
            # Histogram
            ax.hist(data, bins=20, alpha=0.7, edgecolor='black')
            ax.set_xlabel(metric.replace('_', ' ').title())
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of {metric.replace("_", " ").title()}')
            ax.grid(True, alpha=0.3)
            
            # Add statistics text
            mean_val = data.mean()
            std_val = data.std()
            ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.3f}')
            ax.legend()
    
    # Hide unused subplots
    for i in range(n_metrics, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].set_visible(False)
    
    plt.suptitle('Information Density Metrics Distributions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'information_density_distributions.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_scaling_law_fits(df: pd.DataFrame, output_dir: str) -> None:
    """Plot scaling law parameter distributions and quality."""
    os.makedirs(output_dir, exist_ok=True)
    
    scaling_params = ['alpha', 'beta', 'gamma']
    r2_cols = ['alpha_r2', 'beta_r2', 'gamma_r2']
    
    available_params = [p for p in scaling_params if p in df.columns]
    available_r2 = [r for r in r2_cols if r in df.columns]
    
    if not available_params:
        return
    
    # Plot parameter distributions
    fig, axes = plt.subplots(2, len(available_params), figsize=(5*len(available_params), 8))
    if len(available_params) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, param in enumerate(available_params):
        # Parameter distribution
        ax1 = axes[0, i]
        data = df[param].dropna()
        if len(data) > 0:
            ax1.hist(data, bins=15, alpha=0.7, edgecolor='black')
            ax1.set_xlabel(param.upper())
            ax1.set_ylabel('Frequency')
            ax1.set_title(f'{param.upper()} Distribution')
            ax1.grid(True, alpha=0.3)
            
            mean_val = data.mean()
            ax1.axvline(mean_val, color='red', linestyle='--', alpha=0.8, 
                       label=f'Mean: {mean_val:.3f}')
            ax1.legend()
        
        # R² distribution if available
        ax2 = axes[1, i]
        r2_col = f'{param}_r2'
        if r2_col in df.columns:
            r2_data = df[r2_col].dropna()
            if len(r2_data) > 0:
                ax2.hist(r2_data, bins=15, alpha=0.7, edgecolor='black')
                ax2.set_xlabel(f'{param.upper()} R²')
                ax2.set_ylabel('Frequency')
                ax2.set_title(f'{param.upper()} Fit Quality (R²)')
                ax2.grid(True, alpha=0.3)
                
                mean_r2 = r2_data.mean()
                ax2.axvline(mean_r2, color='red', linestyle='--', alpha=0.8,
                           label=f'Mean R²: {mean_r2:.3f}')
                ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'R² data\nnot available', transform=ax2.transAxes,
                    ha='center', va='center', fontsize=12, alpha=0.5)
    
    plt.suptitle('Scaling Law Parameters and Fit Quality', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scaling_parameters_distributions.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_transform_type_comparison(df: pd.DataFrame, output_dir: str) -> None:
    """Compare different transformation types and their effects."""
    os.makedirs(output_dir, exist_ok=True)
    
    if 'transform_type' not in df.columns:
        return
    
    # Get unique transform types
    transform_types = df['transform_type'].unique()
    if len(transform_types) < 2:
        return
    
    # Key metrics to compare
    metrics = ['information_density', 'alpha', 'beta', 'mean_loss']
    available_metrics = [m for m in metrics if m in df.columns]
    
    if not available_metrics:
        return
    
    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(available_metrics[:4]):  # Limit to 4
        ax = axes[i]
        
        # Box plot comparing transform types
        data_by_type = [df[df['transform_type'] == t][metric].dropna() 
                       for t in transform_types]
        valid_data = [(data, t) for data, t in zip(data_by_type, transform_types) 
                     if len(data) > 0]
        
        if valid_data:
            data_list, labels = zip(*valid_data)
            ax.boxplot(data_list, labels=labels)
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} by Transform Type')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(available_metrics), 4):
        axes[i].set_visible(False)
    
    plt.suptitle('Comparison Across Transformation Types', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'transform_type_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_information_ratio_effects(df: pd.DataFrame, output_dir: str) -> None:
    """Plot effects of different information ratios."""
    os.makedirs(output_dir, exist_ok=True)
    
    if 'info_ratio' not in df.columns:
        return
    
    # Filter out standard (ratio = 1.0) entries for cleaner visualization
    transform_df = df[df['info_ratio'] != 1.0]
    if len(transform_df) < 3:
        return
    
    # Metrics to analyze
    y_metrics = ['alpha', 'beta', 'information_density', 'mean_loss']
    available_metrics = [m for m in y_metrics if m in transform_df.columns]
    
    if not available_metrics:
        return
    
    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(available_metrics[:4]):
        ax = axes[i]
        
        # Scatter plot with trend line
        valid_data = transform_df[['info_ratio', metric]].dropna()
        if len(valid_data) > 2:
            x = valid_data['info_ratio']
            y = valid_data[metric]
            
            ax.scatter(x, y, alpha=0.7, s=50)
            
            # Fit trend line
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                line_x = np.linspace(x.min(), x.max(), 100)
                line_y = slope * line_x + intercept
                ax.plot(line_x, line_y, 'r--', alpha=0.8, linewidth=2)
                
                # Add correlation info
                ax.text(0.05, 0.95, f'r={r_value:.3f}\np={p_value:.3f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            except Exception:
                pass
            
            ax.set_xlabel('Information Ratio')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} vs Information Ratio')
            ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(available_metrics), 4):
        axes[i].set_visible(False)
    
    plt.suptitle('Effects of Information Loss (Ratio) on Scaling', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'information_ratio_effects.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_performance_vs_information(df: pd.DataFrame, output_dir: str) -> None:
    """Plot model performance vs information content."""
    os.makedirs(output_dir, exist_ok=True)
    
    if 'mean_loss' not in df.columns:
        return
    
    info_metrics = ['information_density', 'token_entropy', 'ttr', 'compression_ratio']
    available_metrics = [m for m in info_metrics if m in df.columns]
    
    if not available_metrics:
        return
    
    n_metrics = len(available_metrics)
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, metric in enumerate(available_metrics):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]
        
        valid_data = df[[metric, 'mean_loss']].dropna()
        if len(valid_data) > 2:
            x = valid_data[metric]
            y = valid_data['mean_loss']
            
            ax.scatter(x, y, alpha=0.7, s=50)
            
            # Fit trend line
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                line_x = np.linspace(x.min(), x.max(), 100)
                line_y = slope * line_x + intercept
                ax.plot(line_x, line_y, 'r--', alpha=0.8, linewidth=2)
                
                ax.text(0.05, 0.95, f'r={r_value:.3f}\np={p_value:.3f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            except Exception:
                pass
            
            ax.set_xlabel(metric.replace('_', ' ').title())
            ax.set_ylabel('Mean Loss')
            ax.set_title(f'Loss vs {metric.replace("_", " ").title()}')
            ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_metrics, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].set_visible(False)
    
    plt.suptitle('Model Performance vs Information Content', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_vs_information.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def create_summary_report(df: pd.DataFrame, output_dir: str) -> None:
    """Create a summary report of key findings."""
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, 'analysis_summary.txt')
    
    with open(report_path, 'w') as f:
        f.write("INFORMATION DENSITY ANALYSIS SUMMARY REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Basic statistics
        f.write("DATASET OVERVIEW\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total configurations analyzed: {len(df)}\n")
        
        if 'transform_type' in df.columns:
            transform_counts = df['transform_type'].value_counts()
            f.write("Transform types:\n")
            for transform_type, count in transform_counts.items():
                f.write(f"  {transform_type}: {count}\n")
        
        f.write("\n")
        
        # Information metrics summary
        f.write("INFORMATION METRICS SUMMARY\n")
        f.write("-" * 30 + "\n")
        
        info_metrics = ['information_density', 'token_entropy', 'ttr', 'compression_ratio',
                       'mutual_information', 'semantic_density']
        
        for metric in info_metrics:
            if metric in df.columns:
                data = df[metric].dropna()
                if len(data) > 0:
                    f.write(f"{metric.replace('_', ' ').title()}:\n")
                    f.write(f"  Mean: {data.mean():.4f}\n")
                    f.write(f"  Std:  {data.std():.4f}\n")
                    f.write(f"  Range: [{data.min():.4f}, {data.max():.4f}]\n\n")
        
        # Scaling parameters summary
        f.write("SCALING PARAMETERS SUMMARY\n")
        f.write("-" * 30 + "\n")
        
        scaling_params = ['alpha', 'beta', 'gamma']
        for param in scaling_params:
            if param in df.columns:
                data = df[param].dropna()
                if len(data) > 0:
                    f.write(f"{param.upper()}:\n")
                    f.write(f"  Mean: {data.mean():.4f}\n")
                    f.write(f"  Std:  {data.std():.4f}\n")
                    f.write(f"  Range: [{data.min():.4f}, {data.max():.4f}]\n")
                    
                    # R² if available
                    r2_col = f'{param}_r2'
                    if r2_col in df.columns:
                        r2_data = df[r2_col].dropna()
                        if len(r2_data) > 0:
                            f.write(f"  Mean R²: {r2_data.mean():.4f}\n")
                    f.write("\n")
        
        # Correlation analysis
        f.write("KEY CORRELATIONS\n")
        f.write("-" * 20 + "\n")
        
        # Information vs scaling correlations
        info_cols = ['information_density', 'token_entropy', 'ttr', 'compression_ratio']
        scaling_cols = ['alpha', 'beta', 'gamma']
        
        strong_correlations = []
        for info_col in info_cols:
            if info_col in df.columns:
                for scaling_col in scaling_cols:
                    if scaling_col in df.columns:
                        valid_data = df[[info_col, scaling_col]].dropna()
                        if len(valid_data) > 2:
                            corr = valid_data[info_col].corr(valid_data[scaling_col])
                            if abs(corr) > 0.3:  # Strong correlation threshold
                                strong_correlations.append((info_col, scaling_col, corr))
        
        strong_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        
        if strong_correlations:
            f.write("Strong correlations (|r| > 0.3):\n")
            for info_col, scaling_col, corr in strong_correlations[:10]:  # Top 10
                f.write(f"  {info_col} vs {scaling_col}: r = {corr:.3f}\n")
        else:
            f.write("No strong correlations found (|r| > 0.3)\n")
        
        f.write("\n")
        
        # Performance insights
        if 'mean_loss' in df.columns:
            f.write("PERFORMANCE INSIGHTS\n")
            f.write("-" * 20 + "\n")
            
            loss_data = df['mean_loss'].dropna()
            if len(loss_data) > 0:
                f.write(f"Mean Loss across all configurations: {loss_data.mean():.4f}\n")
                f.write(f"Best performing configuration loss: {loss_data.min():.4f}\n")
                f.write(f"Worst performing configuration loss: {loss_data.max():.4f}\n")
                
                # Best configuration details
                best_idx = df['mean_loss'].idxmin()
                if best_idx in df.index:
                    best_config = df.loc[best_idx]
                    f.write("\nBest performing configuration:\n")
                    if 'config_name' in df.columns:
                        f.write(f"  Configuration: {best_config['config_name']}\n")
                    if 'transform_type' in df.columns:
                        f.write(f"  Transform type: {best_config['transform_type']}\n")
                    if 'information_density' in df.columns:
                        f.write(f"  Information density: {best_config['information_density']:.4f}\n")
        
        f.write("\nAnalysis completed successfully.\n")
    
    print(f"Summary report saved to: {report_path}")


def plot_information_analysis(df: pd.DataFrame, output_dir: str) -> None:
    """
    Main function to generate all information analysis plots.
    """
    print(f"Generating information analysis plots in {output_dir}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all plots
    plot_information_vs_scaling_params(df, output_dir)
    plot_correlation_matrix(df, output_dir)
    plot_information_density_distribution(df, output_dir)
    plot_scaling_law_fits(df, output_dir)
    plot_transform_type_comparison(df, output_dir)
    plot_information_ratio_effects(df, output_dir)
    plot_performance_vs_information(df, output_dir)
    
    # Create summary report
    create_summary_report(df, output_dir)
    
    print("All plots generated successfully!")
    print(f"  - Information vs Scaling Parameters: information_vs_scaling_comprehensive.png")
    print(f"  - Correlation Matrix: correlation_matrix.png")
    print(f"  - Information Distributions: information_density_distributions.png")
    print(f"  - Scaling Parameters: scaling_parameters_distributions.png")
    print(f"  - Transform Comparison: transform_type_comparison.png")
    print(f"  - Information Ratio Effects: information_ratio_effects.png")
    print(f"  - Performance vs Information: performance_vs_information.png")
    print(f"  - Summary Report: analysis_summary.txt")