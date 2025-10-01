"""Plotting functions for scaling law visualizations."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
import os


def setup_plot_style():
    """Set up matplotlib style for publication-quality plots."""
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        try:
            plt.style.use('seaborn-whitegrid')
        except:
            plt.style.use('ggplot')
    
    plt.rcParams['figure.figsize'] = (10, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    
    # Set color palette
    colors = sns.color_palette("husl", 8)
    return colors


def plot_loss_vs_parameters(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    title: str = "Test Loss vs Model Parameters",
) -> plt.Figure:
    """
    Plot test loss vs model parameters (similar to Figure 1 in the paper).
    
    Args:
        df: DataFrame with experiment results
        save_path: Path to save the figure
        title: Plot title
    """
    colors = setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Check if data is available
    if len(df) == 0:
        ax.text(0.5, 0.5, 'No data available', 
                horizontalalignment='center', verticalalignment='center', 
                transform=ax.transAxes, fontsize=14)
        ax.set_xlabel('Model Parameters')
        ax.set_ylabel('Test Loss')
        ax.set_title(title)
        return fig
    
    # Get unique experiment types
    experiment_types = df['experiment_type'].unique() if 'experiment_type' in df.columns else ['all']
    
    for i, exp_type in enumerate(experiment_types):
        if exp_type == 'all':
            subset = df
            label = 'All experiments'
        else:
            subset = df[df['experiment_type'] == exp_type]
            label = exp_type.capitalize()
        
        if len(subset) == 0:
            continue
            
        # Plot all points
        ax.loglog(
            subset['model_params'],
            subset['final_loss'],
            'o',
            color=colors[i % len(colors)],
            markersize=6,
            alpha=0.6,
            label=f'{label} (individual)',
        )
        
        # Calculate average loss for each model size
        avg_loss = subset.groupby('model_params')['final_loss'].mean()
        
        if len(avg_loss) > 1:
            # Plot average trend
            ax.loglog(
                avg_loss.index,
                avg_loss.values,
                'o-',
                color=colors[i % len(colors)],
                markersize=10,
                linewidth=2,
                label=f'{label} (average)',
            )
            
            # Fit power law to averages
            if len(avg_loss) > 2:
                log_params = np.log10(avg_loss.index.values)
                log_loss = np.log10(avg_loss.values)
                slope, intercept, r_value, _, _ = stats.linregress(log_params, log_loss)
                
                # Plot fit
                fit_params = np.logspace(
                    np.log10(avg_loss.index.min()),
                    np.log10(avg_loss.index.max()),
                    100
                )
                fit_loss = 10**(slope * np.log10(fit_params) + intercept)
                ax.loglog(
                    fit_params,
                    fit_loss,
                    '--',
                    color=colors[i % len(colors)],
                    linewidth=2,
                    alpha=0.8,
                    label=f'{label} fit (α={-slope:.3f})',
                )
    
    ax.set_xlabel('Model Parameters')
    ax.set_ylabel('Test Loss')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_loss_vs_dataset_size(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    title: str = "Test Loss vs Dataset Size",
) -> plt.Figure:
    """
    Plot test loss vs dataset size (similar to Figure 2 in the paper).
    
    Args:
        df: DataFrame with experiment results
        save_path: Path to save the figure
        title: Plot title
    """
    colors = setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Check if data is available
    if len(df) == 0:
        ax.text(0.5, 0.5, 'No data available', 
                horizontalalignment='center', verticalalignment='center', 
                transform=ax.transAxes, fontsize=14)
        ax.set_xlabel('Dataset Size (tokens)')
        ax.set_ylabel('Test Loss')
        ax.set_title(title)
        return fig
    
    # Get unique model sizes (parameters)
    model_sizes = sorted(df['model_params'].unique()) if 'model_params' in df.columns else []
    
    # Plot dashed lines for each model size
    for i, model_size in enumerate(model_sizes):
        subset = df[df['model_params'] == model_size]
        
        if len(subset) == 0:
            continue
            
        # Sort by dataset size for proper line plotting
        subset = subset.sort_values('dataset_size')
        
        # Plot dashed line for this model size
        ax.loglog(
            subset['dataset_size'],
            subset['final_loss'],
            '--',
            color=colors[i % len(colors)],
            linewidth=2,
            alpha=0.8,
            label=f'{model_size:,} params',
        )
        
        # Plot individual points
        ax.loglog(
            subset['dataset_size'],
            subset['final_loss'],
            'o',
            color=colors[i % len(colors)],
            markersize=6,
            alpha=0.6,
        )
    
    # Fit power law to all data and plot as solid line
    if len(df) > 2:
        log_data = np.log10(df['dataset_size'].values)
        log_loss = np.log10(df['final_loss'].values)
        slope, intercept, r_value, _, _ = stats.linregress(log_data, log_loss)
        
        # Plot overall fit as solid line
        fit_data = np.logspace(
            np.log10(df['dataset_size'].min()),
            np.log10(df['dataset_size'].max()),
            100
        )
        fit_loss = 10**(slope * np.log10(fit_data) + intercept)
        ax.loglog(
            fit_data,
            fit_loss,
            '-',
            color='black',
            linewidth=3,
            alpha=0.8,
            label=f'Overall fit (β={-slope:.3f})',
        )
    
    ax.set_xlabel('Dataset Size (tokens)')
    ax.set_ylabel('Test Loss')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_loss_vs_compute(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    title: str = "Test Loss vs Compute",
) -> plt.Figure:
    """
    Plot test loss vs compute (similar to Figure 1 in the paper).
    
    Args:
        df: DataFrame with experiment results
        save_path: Path to save the figure
        title: Plot title
    """
    colors = setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Check if compute data is available
    if 'petaflop_days' not in df.columns:
        # Estimate compute from model parameters and dataset size
        print("Warning: No compute tracking data found. Estimating compute from model parameters and dataset size.")
        try:
            from experiments.compute_tracker import estimate_compute_for_model
        except ImportError:
            print("Warning: compute_tracker not available, skipping compute estimation")
            return None
        
        df = df.copy()
        df['petaflop_days'] = df.apply(
            lambda row: estimate_compute_for_model(
                row['model_params'], 
                row['dataset_size']
            )['petaflop_days'], 
            axis=1
        )
    
    # Filter for rows with compute data
    df_compute = df[df['petaflop_days'].notna()].copy()
    
    if len(df_compute) == 0:
        print("Warning: No compute data available")
        # Create an empty plot with a message
        ax.text(0.5, 0.5, 'No compute data available\nEnable compute tracking in config', 
                horizontalalignment='center', verticalalignment='center', 
                transform=ax.transAxes, fontsize=14)
        ax.set_xlabel('Compute (PetaFLOP-days)')
        ax.set_ylabel('Test Loss')
        ax.set_title(title)
        return fig
    
    # Plot all points
    ax.loglog(
        df_compute['petaflop_days'],
        df_compute['final_loss'],
        'o',
        color=colors[0],
        markersize=8,
        alpha=0.6,
        label='All experiments',
    )
    
    # Fit power law to all data
    log_compute = np.log10(df_compute['petaflop_days'].values)
    log_loss = np.log10(df_compute['final_loss'].values)
    slope, intercept, r_value, _, _ = stats.linregress(log_compute, log_loss)
    
    # Plot fit
    fit_compute = np.logspace(
        np.log10(df_compute['petaflop_days'].min()),
        np.log10(df_compute['petaflop_days'].max()),
        100
    )
    fit_loss = 10**(slope * np.log10(fit_compute) + intercept)
    ax.loglog(
        fit_compute,
        fit_loss,
        '--',
        color='red',
        linewidth=2,
        label=f'Power law fit (γ={-slope:.2f})',
    )
    
    # Highlight efficient frontier (Pareto optimal points)
    # Find points that are not dominated by any other point
    pareto_indices = []
    for i in range(len(df_compute)):
        is_pareto = True
        for j in range(len(df_compute)):
            if i != j:
                # Check if point j dominates point i
                if (df_compute.iloc[j]['petaflop_days'] <= df_compute.iloc[i]['petaflop_days'] and
                    df_compute.iloc[j]['final_loss'] < df_compute.iloc[i]['final_loss']):
                    is_pareto = False
                    break
        if is_pareto:
            pareto_indices.append(i)
    
    if pareto_indices:
        pareto_df = df_compute.iloc[pareto_indices].sort_values('petaflop_days')
        ax.loglog(
            pareto_df['petaflop_days'],
            pareto_df['final_loss'],
            'o-',
            color='green',
            markersize=10,
            linewidth=2,
            label='Efficient frontier',
        )
    
    ax.set_xlabel('Compute (PetaFLOP-days)')
    ax.set_ylabel('Test Loss')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_optimal_model_size(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    title: str = "Optimal Model Size vs Compute Budget",
) -> plt.Figure:
    """
    Plot optimal model size vs compute budget (similar to Figure 6 in the paper).
    
    Args:
        df: DataFrame with experiment results
        save_path: Path to save the figure
        title: Plot title
    """
    colors = setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Check if compute data is available
    if 'petaflop_days' not in df.columns:
        # Estimate compute from model parameters and dataset size
        print("Warning: No compute tracking data found. Estimating compute from model parameters and dataset size.")
        try:
            from experiments.compute_tracker import estimate_compute_for_model
        except ImportError:
            print("Warning: compute_tracker not available, skipping compute estimation")
            return None
        
        df = df.copy()
        df['petaflop_days'] = df.apply(
            lambda row: estimate_compute_for_model(
                row['model_params'], 
                row['dataset_size']
            )['petaflop_days'], 
            axis=1
        )
    
    # Group by compute budget bins
    df_compute = df[df['petaflop_days'].notna()].copy()
    
    if len(df_compute) == 0:
        print("Warning: No compute data available")
        # Create an empty plot with a message
        ax.text(0.5, 0.5, 'No compute data available\nEnable compute tracking in config', 
                horizontalalignment='center', verticalalignment='center', 
                transform=ax.transAxes, fontsize=14)
        ax.set_xlabel('Compute Budget (PetaFLOP-days)')
        ax.set_ylabel('Optimal Model Size (parameters)')
        ax.set_title(title)
        return fig
    
    # Create compute bins
    compute_bins = np.logspace(
        np.log10(df_compute['petaflop_days'].min()),
        np.log10(df_compute['petaflop_days'].max()),
        10
    )
    
    optimal_sizes = []
    compute_centers = []
    
    for i in range(len(compute_bins) - 1):
        # Find experiments in this compute bin
        mask = (df_compute['petaflop_days'] >= compute_bins[i]) & \
               (df_compute['petaflop_days'] < compute_bins[i+1])
        bin_df = df_compute[mask]
        
        if len(bin_df) > 0:
            # Find model with lowest loss in this bin
            best_idx = bin_df['final_loss'].idxmin()
            optimal_sizes.append(bin_df.loc[best_idx, 'model_params'])
            compute_centers.append(np.sqrt(compute_bins[i] * compute_bins[i+1]))
    
    if optimal_sizes:
        # Plot optimal model sizes
        ax.loglog(
            compute_centers,
            optimal_sizes,
            'o-',
            color=colors[0],
            markersize=10,
            linewidth=2,
            label='Optimal model size',
        )
        
        # Fit power law
        if len(optimal_sizes) > 2:
            log_compute = np.log10(compute_centers)
            log_size = np.log10(optimal_sizes)
            slope, intercept, r_value, _, _ = stats.linregress(log_compute, log_size)
            
            # Plot fit
            fit_compute = np.logspace(
                np.log10(min(compute_centers)),
                np.log10(max(compute_centers)),
                100
            )
            fit_size = 10**(slope * np.log10(fit_compute) + intercept)
            ax.loglog(
                fit_compute,
                fit_size,
                '--',
                color='red',
                linewidth=2,
                label=f'N_opt ∝ C^{slope:.2f}',
            )
    
    ax.set_xlabel('Compute Budget (PetaFLOP-days)')
    ax.set_ylabel('Optimal Model Size (parameters)')
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_all_scaling_laws(
    df: pd.DataFrame,
    output_dir: str,
    experiment_type: str = "standard",
):
    """Generate all scaling law plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter for experiment type
    df_filtered = df[df['experiment_type'] == experiment_type]
    
    # Generate plots
    plots = [
        ("loss_vs_parameters", plot_loss_vs_parameters),
        ("loss_vs_dataset_size", plot_loss_vs_dataset_size),
        ("loss_vs_compute", plot_loss_vs_compute),
        ("optimal_model_size", plot_optimal_model_size),
    ]
    
    for name, plot_func in plots:
        save_path = os.path.join(output_dir, f"{experiment_type}_{name}.png")
        fig = plot_func(
            df_filtered,
            save_path=save_path,
            title=f"{experiment_type.capitalize()} - {name.replace('_', ' ').title()}",
        )
        plt.close(fig)
    
    print(f"All {experiment_type} plots saved to {output_dir}")


def compare_standard_vs_transformed(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Compare standard vs transformed experiments."""
    return compare_experiment_types(
        df, 
        type1='standard', 
        type2='transformed',
        save_path=save_path
    )


def compare_experiment_types(
    df: pd.DataFrame,
    type1: str,
    type2: str,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Compare two experiment types."""
    colors = setup_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    experiment_types = [type1, type2]
    type_colors = colors[:2]
    
    # 1. Loss comparison by model size
    ax = axes[0]
    for exp_type, color in zip(experiment_types, type_colors):
        subset = df[df['experiment_type'] == exp_type]
        if len(subset) > 0:
            model_loss = subset.groupby('model_params')['final_loss'].mean()
            ax.loglog(
                model_loss.index,
                model_loss.values,
                'o-',
                color=color,
                label=exp_type.capitalize(),
                markersize=8,
                linewidth=2,
            )
    ax.set_xlabel('Model Parameters')
    ax.set_ylabel('Average Test Loss')
    ax.set_title('Loss by Model Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Loss comparison by dataset size
    ax = axes[1]
    for exp_type, color in zip(experiment_types, type_colors):
        subset = df[df['experiment_type'] == exp_type]
        if len(subset) > 0:
            data_loss = subset.groupby('dataset_size')['final_loss'].mean()
            ax.loglog(
                data_loss.index,
                data_loss.values,
                'o-',
                color=color,
                label=exp_type.capitalize(),
                markersize=8,
                linewidth=2,
            )
    ax.set_xlabel('Dataset Size (tokens)')
    ax.set_ylabel('Average Test Loss')
    ax.set_title('Loss by Dataset Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Perplexity comparison
    ax = axes[2]
    for exp_type, color in zip(experiment_types, type_colors):
        subset = df[df['experiment_type'] == exp_type]
        if len(subset) > 0 and 'final_perplexity' in subset.columns:
            perplexities = subset['final_perplexity'].dropna().values
            if len(perplexities) > 0:
                ax.hist(
                    perplexities,
                    bins=20,
                    alpha=0.6,
                    color=color,
                    label=exp_type.capitalize(),
                    density=True,
                )
    ax.set_xlabel('Final Perplexity')
    ax.set_ylabel('Density')
    ax.set_title('Perplexity Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Relative performance
    ax = axes[3]
    # Match experiments by model and dataset size
    df1 = df[df['experiment_type'] == type1].set_index(['model_name', 'dataset_size'])
    df2 = df[df['experiment_type'] == type2].set_index(['model_name', 'dataset_size'])
    
    common_indices = df1.index.intersection(df2.index)
    if len(common_indices) > 0:
        relative_loss = []
        model_params = []
        
        for idx in common_indices:
            loss1 = df1.loc[idx, 'final_loss']
            loss2 = df2.loc[idx, 'final_loss']
            relative = (loss2 - loss1) / loss1 * 100
            relative_loss.append(relative)
            model_params.append(df1.loc[idx, 'model_params'])
        
        ax.semilogx(model_params, relative_loss, 'o', color=colors[2], markersize=8)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Model Parameters')
        ax.set_ylabel(f'Relative Loss Increase (%)\n({type2} vs {type1})')
        ax.set_title('Performance Impact')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No common experiments to compare', 
                horizontalalignment='center', verticalalignment='center', 
                transform=ax.transAxes, fontsize=14)
    
    plt.suptitle(f'Comparison: {type1.capitalize()} vs {type2.capitalize()}', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig 