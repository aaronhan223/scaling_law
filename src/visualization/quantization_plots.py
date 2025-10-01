"""Plotting functions for quantization transformation analysis."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional, Tuple, List
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


def plot_scaling_parameters_vs_quantization(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    title: str = "Scaling Law Parameters vs Quantization Ratio",
) -> plt.Figure:
    """
    Plot scaling law parameters (alpha, beta, gamma) vs quantization ratio.
    
    Args:
        df: DataFrame with quantization analysis results
        save_path: Path to save the figure
        title: Plot title
    """
    colors = setup_plot_style()
    
    # Create subplots for each parameter
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    parameters = [
        ('alpha', 'α (Model Size)', 'Model size scaling exponent'),
        ('beta', 'β (Dataset Size)', 'Dataset size scaling exponent'),
        ('gamma', 'γ (Compute)', 'Compute scaling exponent'),
    ]
    
    for i, (param, ylabel, desc) in enumerate(parameters):
        ax = axes[i]
        
        if param in df.columns:
            # Filter out NaN values
            valid_data = df.dropna(subset=[param, 'quantization_ratio'])
            
            if len(valid_data) > 0:
                # Plot scatter points
                ax.scatter(
                    valid_data['quantization_ratio'],
                    valid_data[param],
                    color=colors[i],
                    s=80,
                    alpha=0.7,
                    label=f'{param.upper()} values'
                )
                
                # Fit trend line if we have enough points
                if len(valid_data) >= 3:
                    x = valid_data['quantization_ratio'].values
                    y = valid_data[param].values
                    
                    # Try polynomial fit
                    try:
                        z = np.polyfit(x, y, 2)  # Quadratic fit
                        p = np.poly1d(z)
                        x_trend = np.linspace(x.min(), x.max(), 100)
                        y_trend = p(x_trend)
                        
                        ax.plot(x_trend, y_trend, '--', color=colors[i], 
                               linewidth=2, alpha=0.8, label=f'Quadratic fit')
                    except:
                        # Fallback to linear fit
                        slope, intercept, r_value, _, _ = stats.linregress(x, y)
                        x_trend = np.linspace(x.min(), x.max(), 100)
                        y_trend = slope * x_trend + intercept
                        
                        ax.plot(x_trend, y_trend, '--', color=colors[i], 
                               linewidth=2, alpha=0.8, 
                               label=f'Linear fit (R²={r_value**2:.3f})')
                
                # Add annotation with current range
                ax.text(0.05, 0.95, f'Range: [{valid_data[param].min():.3f}, {valid_data[param].max():.3f}]',
                       transform=ax.transAxes, fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Quantization Ratio')
        ax.set_ylabel(ylabel)
        ax.set_title(desc)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set reasonable axis limits
        if param in df.columns and not df[param].isna().all():
            valid_values = df[param].dropna()
            if len(valid_values) > 0:
                y_range = valid_values.max() - valid_values.min()
                y_margin = y_range * 0.1
                ax.set_ylim(valid_values.min() - y_margin, valid_values.max() + y_margin)
    
    # Fourth subplot: R² values for fit quality
    ax = axes[3]
    r2_params = ['alpha_r2', 'beta_r2', 'gamma_r2']
    r2_labels = ['α R²', 'β R²', 'γ R²']
    
    for j, (r2_param, label) in enumerate(zip(r2_params, r2_labels)):
        if r2_param in df.columns:
            valid_r2 = df.dropna(subset=[r2_param, 'quantization_ratio'])
            if len(valid_r2) > 0:
                ax.scatter(
                    valid_r2['quantization_ratio'],
                    valid_r2[r2_param],
                    color=colors[j],
                    s=60,
                    alpha=0.7,
                    label=label
                )
    
    ax.set_xlabel('Quantization Ratio')
    ax.set_ylabel('R² (Fit Quality)')
    ax.set_title('Scaling Law Fit Quality')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    plt.suptitle(title, fontsize=18)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_loss_distribution_vs_quantization(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    title: str = "Loss Distribution vs Quantization Ratio",
) -> plt.Figure:
    """
    Plot loss distribution metrics vs quantization ratio.
    
    Args:
        df: DataFrame with quantization analysis results
        save_path: Path to save the figure
        title: Plot title
    """
    colors = setup_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Mean loss vs quantization ratio
    ax = axes[0]
    if 'mean_loss' in df.columns:
        valid_data = df.dropna(subset=['mean_loss', 'quantization_ratio'])
        if len(valid_data) > 0:
            ax.scatter(valid_data['quantization_ratio'], valid_data['mean_loss'], 
                      color=colors[0], s=80, alpha=0.7)
            
            # Fit trend line
            if len(valid_data) >= 3:
                x = valid_data['quantization_ratio'].values
                y = valid_data['mean_loss'].values
                slope, intercept, r_value, _, _ = stats.linregress(x, y)
                x_trend = np.linspace(x.min(), x.max(), 100)
                y_trend = slope * x_trend + intercept
                ax.plot(x_trend, y_trend, '--', color=colors[0], linewidth=2, 
                       label=f'Linear fit (R²={r_value**2:.3f})')
                ax.legend()
    
    ax.set_xlabel('Quantization Ratio')
    ax.set_ylabel('Mean Loss')
    ax.set_title('Mean Loss vs Quantization Ratio')
    ax.grid(True, alpha=0.3)
    
    # Loss variance vs quantization ratio
    ax = axes[1]
    if 'std_loss' in df.columns:
        valid_data = df.dropna(subset=['std_loss', 'quantization_ratio'])
        if len(valid_data) > 0:
            ax.scatter(valid_data['quantization_ratio'], valid_data['std_loss'], 
                      color=colors[1], s=80, alpha=0.7)
            
            # Fit trend line
            if len(valid_data) >= 3:
                x = valid_data['quantization_ratio'].values
                y = valid_data['std_loss'].values
                slope, intercept, r_value, _, _ = stats.linregress(x, y)
                x_trend = np.linspace(x.min(), x.max(), 100)
                y_trend = slope * x_trend + intercept
                ax.plot(x_trend, y_trend, '--', color=colors[1], linewidth=2,
                       label=f'Linear fit (R²={r_value**2:.3f})')
                ax.legend()
    
    ax.set_xlabel('Quantization Ratio')
    ax.set_ylabel('Loss Standard Deviation')
    ax.set_title('Loss Variance vs Quantization Ratio')
    ax.grid(True, alpha=0.3)
    
    # Loss range vs quantization ratio
    ax = axes[2]
    if 'min_loss' in df.columns and 'max_loss' in df.columns:
        valid_data = df.dropna(subset=['min_loss', 'max_loss', 'quantization_ratio'])
        if len(valid_data) > 0:
            loss_range = valid_data['max_loss'] - valid_data['min_loss']
            ax.scatter(valid_data['quantization_ratio'], loss_range, 
                      color=colors[2], s=80, alpha=0.7)
            
            # Fit trend line
            if len(valid_data) >= 3:
                x = valid_data['quantization_ratio'].values
                y = loss_range.values
                slope, intercept, r_value, _, _ = stats.linregress(x, y)
                x_trend = np.linspace(x.min(), x.max(), 100)
                y_trend = slope * x_trend + intercept
                ax.plot(x_trend, y_trend, '--', color=colors[2], linewidth=2,
                       label=f'Linear fit (R²={r_value**2:.3f})')
                ax.legend()
    
    ax.set_xlabel('Quantization Ratio')
    ax.set_ylabel('Loss Range (Max - Min)')
    ax.set_title('Loss Range vs Quantization Ratio')
    ax.grid(True, alpha=0.3)
    
    # Number of quantization levels vs mean loss (log scale)
    ax = axes[3]
    if 'quantization_levels' in df.columns and 'mean_loss' in df.columns:
        valid_data = df.dropna(subset=['quantization_levels', 'mean_loss'])
        if len(valid_data) > 0:
            ax.loglog(valid_data['quantization_levels'], valid_data['mean_loss'], 
                     'o', color=colors[3], markersize=8, alpha=0.7)
            
            # Fit power law
            if len(valid_data) >= 3:
                x = valid_data['quantization_levels'].values
                y = valid_data['mean_loss'].values
                log_x = np.log10(x)
                log_y = np.log10(y)
                slope, intercept, r_value, _, _ = stats.linregress(log_x, log_y)
                
                x_trend = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
                y_trend = 10**(slope * np.log10(x_trend) + intercept)
                ax.loglog(x_trend, y_trend, '--', color=colors[3], linewidth=2,
                         label=f'Power law (exp={slope:.3f}, R²={r_value**2:.3f})')
                ax.legend()
    
    ax.set_xlabel('Number of Quantization Levels')
    ax.set_ylabel('Mean Loss')
    ax.set_title('Mean Loss vs Quantization Levels (Log-Log)')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=18)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_quantization_heatmap(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    title: str = "Quantization Analysis Heatmap",
) -> plt.Figure:
    """
    Create a heatmap showing correlations between quantization metrics.
    
    Args:
        df: DataFrame with quantization analysis results
        save_path: Path to save the figure
        title: Plot title
    """
    setup_plot_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Select numeric columns for correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_cols = [col for col in numeric_cols if col not in ['num_experiments']]
    
    if len(correlation_cols) > 1:
        # Correlation heatmap
        ax = axes[0]
        corr_matrix = df[correlation_cols].corr()
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('Correlation Matrix')
        
        # Parameter evolution heatmap
        ax = axes[1]
        param_cols = ['quantization_ratio', 'alpha', 'beta', 'gamma']
        available_cols = [col for col in param_cols if col in df.columns]
        
        if len(available_cols) > 1:
            # Normalize data for better visualization
            normalized_df = df[available_cols].copy()
            for col in available_cols[1:]:  # Skip quantization_ratio
                if col in normalized_df.columns:
                    col_values = normalized_df[col].dropna()
                    if len(col_values) > 0:
                        col_min, col_max = col_values.min(), col_values.max()
                        if col_max > col_min:
                            normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
            
            # Create heatmap data
            heatmap_data = normalized_df.dropna().set_index('quantization_ratio').T
            
            if not heatmap_data.empty:
                sns.heatmap(heatmap_data, annot=True, cmap='viridis', 
                           cbar_kws={"shrink": 0.8}, ax=ax, fmt='.3f')
                ax.set_title('Normalized Parameter Evolution')
                ax.set_xlabel('Quantization Ratio')
                ax.set_ylabel('Scaling Parameters')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_quantization_summary(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    title: str = "Quantization Analysis Summary",
) -> plt.Figure:
    """
    Create a comprehensive summary plot of quantization analysis.
    
    Args:
        df: DataFrame with quantization analysis results
        save_path: Path to save the figure
        title: Plot title
    """
    colors = setup_plot_style()
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Main scaling parameters plot (spanning 2x2)
    ax_main = fig.add_subplot(gs[0:2, 0:2])
    
    parameters = ['alpha', 'beta', 'gamma']
    param_labels = ['α (Model Size)', 'β (Dataset Size)', 'γ (Compute)']
    
    for i, (param, label) in enumerate(zip(parameters, param_labels)):
        if param in df.columns:
            valid_data = df.dropna(subset=[param, 'quantization_ratio'])
            if len(valid_data) > 0:
                ax_main.plot(valid_data['quantization_ratio'], valid_data[param], 
                           'o-', color=colors[i], linewidth=2, markersize=8,
                           label=label, alpha=0.8)
    
    ax_main.set_xlabel('Quantization Ratio')
    ax_main.set_ylabel('Scaling Exponent')
    ax_main.set_title('Scaling Law Exponents vs Quantization', fontsize=14)
    ax_main.legend()
    ax_main.grid(True, alpha=0.3)
    
    # Loss metrics (top right)
    ax_loss = fig.add_subplot(gs[0, 2:])
    if 'mean_loss' in df.columns:
        valid_data = df.dropna(subset=['mean_loss', 'quantization_ratio'])
        if len(valid_data) > 0:
            ax_loss.plot(valid_data['quantization_ratio'], valid_data['mean_loss'], 
                        'o-', color=colors[3], linewidth=2, markersize=6)
    ax_loss.set_xlabel('Quantization Ratio')
    ax_loss.set_ylabel('Mean Loss')
    ax_loss.set_title('Mean Loss Evolution')
    ax_loss.grid(True, alpha=0.3)
    
    # Fit quality (middle right)
    ax_r2 = fig.add_subplot(gs[1, 2:])
    r2_params = ['alpha_r2', 'beta_r2', 'gamma_r2']
    r2_labels = ['α R²', 'β R²', 'γ R²']
    
    for j, (r2_param, label) in enumerate(zip(r2_params, r2_labels)):
        if r2_param in df.columns:
            valid_r2 = df.dropna(subset=[r2_param, 'quantization_ratio'])
            if len(valid_r2) > 0:
                ax_r2.plot(valid_r2['quantization_ratio'], valid_r2[r2_param], 
                          'o-', color=colors[j], linewidth=2, markersize=6, 
                          label=label, alpha=0.8)
    
    ax_r2.set_xlabel('Quantization Ratio')
    ax_r2.set_ylabel('R² (Fit Quality)')
    ax_r2.set_title('Scaling Law Fit Quality')
    ax_r2.legend()
    ax_r2.grid(True, alpha=0.3)
    ax_r2.set_ylim(0, 1.1)
    
    # Quantization levels effect (bottom left)
    ax_levels = fig.add_subplot(gs[2, 0])
    if 'quantization_levels' in df.columns and 'mean_loss' in df.columns:
        valid_data = df.dropna(subset=['quantization_levels', 'mean_loss'])
        if len(valid_data) > 0:
            ax_levels.loglog(valid_data['quantization_levels'], valid_data['mean_loss'], 
                           'o', color=colors[4], markersize=8, alpha=0.7)
    ax_levels.set_xlabel('Quantization Levels')
    ax_levels.set_ylabel('Mean Loss')
    ax_levels.set_title('Loss vs Levels (Log-Log)')
    ax_levels.grid(True, alpha=0.3)
    
    # Loss variance (bottom center)
    ax_var = fig.add_subplot(gs[2, 1])
    if 'std_loss' in df.columns:
        valid_data = df.dropna(subset=['std_loss', 'quantization_ratio'])
        if len(valid_data) > 0:
            ax_var.plot(valid_data['quantization_ratio'], valid_data['std_loss'], 
                       'o-', color=colors[5], linewidth=2, markersize=6)
    ax_var.set_xlabel('Quantization Ratio')
    ax_var.set_ylabel('Loss Std Dev')
    ax_var.set_title('Loss Variability')
    ax_var.grid(True, alpha=0.3)
    
    # Data summary table (bottom right)
    ax_table = fig.add_subplot(gs[2, 2:])
    ax_table.axis('tight')
    ax_table.axis('off')
    
    # Create summary statistics table
    if len(df) > 0:
        summary_data = []
        for param in ['alpha', 'beta', 'gamma']:
            if param in df.columns:
                values = df[param].dropna()
                if len(values) > 0:
                    summary_data.append([
                        param.upper(),
                        f"{values.mean():.3f}",
                        f"{values.std():.3f}",
                        f"[{values.min():.3f}, {values.max():.3f}]"
                    ])
        
        if summary_data:
            table = ax_table.table(cellText=summary_data,
                                 colLabels=['Parameter', 'Mean', 'Std', 'Range'],
                                 cellLoc='center',
                                 loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            ax_table.set_title('Summary Statistics', fontsize=12)
    
    plt.suptitle(title, fontsize=18)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_quantization_analysis(df: pd.DataFrame, output_dir: str):
    """
    Generate all quantization analysis plots.
    
    Args:
        df: DataFrame with quantization analysis results
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating quantization analysis plots...")
    
    # 1. Main scaling parameters plot
    fig1 = plot_scaling_parameters_vs_quantization(
        df, 
        save_path=os.path.join(output_dir, "scaling_parameters_vs_quantization.png")
    )
    plt.close(fig1)
    
    # 2. Loss distribution plots
    fig2 = plot_loss_distribution_vs_quantization(
        df,
        save_path=os.path.join(output_dir, "loss_distribution_vs_quantization.png")
    )
    plt.close(fig2)
    
    # 3. Correlation heatmap
    fig3 = plot_quantization_heatmap(
        df,
        save_path=os.path.join(output_dir, "quantization_heatmap.png")
    )
    plt.close(fig3)
    
    # 4. Comprehensive summary
    fig4 = plot_quantization_summary(
        df,
        save_path=os.path.join(output_dir, "quantization_summary.png")
    )
    plt.close(fig4)
    
    print(f"All quantization analysis plots saved to {output_dir}")
    print("Generated files:")
    print("  - scaling_parameters_vs_quantization.png")
    print("  - loss_distribution_vs_quantization.png") 
    print("  - quantization_heatmap.png")
    print("  - quantization_summary.png") 