"""
Plotting functions for LibriSpeech quantization analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import os


def plot_quantization_levels_vs_wer(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 8)
) -> None:
    """
    Plot quantization levels vs WER to show compression impact.

    Args:
        df: DataFrame with quantization analysis results
        save_path: Path to save the plot
        figsize: Figure size
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

    # Main plot: Quantization levels vs mean WER
    if 'quantization_levels' in df.columns and 'mean_wer' in df.columns:
        valid_data = df.dropna(subset=['quantization_levels', 'mean_wer'])
        if len(valid_data) > 0:
            ax1.loglog(valid_data['quantization_levels'], valid_data['mean_wer'],
                      'bo-', linewidth=2, markersize=8, alpha=0.8)

            # Add power law fit
            if len(valid_data) >= 3:
                x = valid_data['quantization_levels'].values
                y = valid_data['mean_wer'].values

                # Fit in log space
                log_x = np.log10(x)
                log_y = np.log10(y)
                coeffs = np.polyfit(log_x, log_y, 1)

                # Plot fit
                x_fit = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
                y_fit = 10**(coeffs[1]) * x_fit**coeffs[0]
                ax1.loglog(x_fit, y_fit, 'r--', alpha=0.7,
                          label=f'Fit: WER ∝ levels^{coeffs[0]:.2f}')
                ax1.legend()

    ax1.set_xlabel('Quantization Levels')
    ax1.set_ylabel('Mean Word Error Rate (WER)')
    ax1.set_title('WER vs Quantization Levels')
    ax1.grid(True, alpha=0.3)

    # Quantization bits vs WER
    if 'quantization_bits' in df.columns and 'mean_wer' in df.columns:
        valid_data = df.dropna(subset=['quantization_bits', 'mean_wer'])
        if len(valid_data) > 0:
            ax2.semilogy(valid_data['quantization_bits'], valid_data['mean_wer'],
                        'go-', linewidth=2, markersize=8, alpha=0.8)

    ax2.set_xlabel('Quantization Bits')
    ax2.set_ylabel('Mean Word Error Rate (WER)')
    ax2.set_title('WER vs Quantization Bits')
    ax2.grid(True, alpha=0.3)

    # Model scaling alpha vs quantization levels
    if 'avg_model_alpha' in df.columns:
        valid_data = df.dropna(subset=['quantization_levels', 'avg_model_alpha'])
        if len(valid_data) > 0:
            ax3.semilogx(valid_data['quantization_levels'], valid_data['avg_model_alpha'],
                        'mo-', linewidth=2, markersize=8, alpha=0.8)

            # Add error bars if std is available
            if 'std_model_alpha' in df.columns:
                ax3.errorbar(valid_data['quantization_levels'], valid_data['avg_model_alpha'],
                           yerr=valid_data['std_model_alpha'], fmt='mo-',
                           linewidth=2, markersize=8, alpha=0.8, capsize=5)

    ax3.set_xlabel('Quantization Levels')
    ax3.set_ylabel('Model Scaling Exponent (α)')
    ax3.set_title('Model Scaling α vs Quantization')
    ax3.grid(True, alpha=0.3)

    # Data scaling alpha vs quantization levels
    if 'avg_data_alpha' in df.columns:
        valid_data = df.dropna(subset=['quantization_levels', 'avg_data_alpha'])
        if len(valid_data) > 0:
            ax4.semilogx(valid_data['quantization_levels'], valid_data['avg_data_alpha'],
                        'co-', linewidth=2, markersize=8, alpha=0.8)

            # Add error bars if std is available
            if 'std_data_alpha' in df.columns:
                ax4.errorbar(valid_data['quantization_levels'], valid_data['avg_data_alpha'],
                           yerr=valid_data['std_data_alpha'], fmt='co-',
                           linewidth=2, markersize=8, alpha=0.8, capsize=5)

    ax4.set_xlabel('Quantization Levels')
    ax4.set_ylabel('Data Scaling Exponent (α)')
    ax4.set_title('Data Scaling α vs Quantization')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved quantization levels vs WER plot to: {save_path}")

    plt.show()


def plot_scaling_parameter_stability(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: tuple = (15, 10)
) -> None:
    """
    Plot how scaling parameters change with quantization levels.

    Args:
        df: DataFrame with quantization analysis results
        save_path: Path to save the plot
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    # Parameters to plot
    params = [
        ('avg_model_alpha', 'std_model_alpha', 'Model Scaling α', 'red'),
        ('avg_data_alpha', 'std_data_alpha', 'Data Scaling α', 'blue'),
        ('avg_compute_alpha', 'std_compute_alpha', 'Compute Scaling α', 'green'),
        ('avg_model_r2', 'std_model_r2', 'Model Scaling R²', 'orange'),
        ('avg_data_r2', 'std_data_r2', 'Data Scaling R²', 'purple'),
        ('avg_compute_r2', 'std_compute_r2', 'Compute Scaling R²', 'brown'),
    ]

    for i, (param, std_param, title, color) in enumerate(params):
        ax = axes[i]

        if param in df.columns:
            valid_data = df.dropna(subset=['quantization_levels', param])

            if len(valid_data) > 0:
                x = valid_data['quantization_levels']
                y = valid_data[param]

                ax.semilogx(x, y, 'o-', color=color, linewidth=2, markersize=8, alpha=0.8)

                # Add error bars if available
                if std_param in df.columns:
                    yerr = valid_data[std_param]
                    ax.errorbar(x, y, yerr=yerr, fmt='o-', color=color,
                              linewidth=2, markersize=8, alpha=0.8, capsize=5)

        ax.set_xlabel('Quantization Levels')
        ax.set_ylabel(title)
        ax.set_title(f'{title} vs Quantization')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved scaling parameter stability plot to: {save_path}")

    plt.show()


def plot_compression_vs_performance(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 8)
) -> None:
    """
    Plot compression ratio vs speech recognition performance.

    Args:
        df: DataFrame with quantization analysis results
        save_path: Path to save the plot
        figsize: Figure size
    """
    if 'quantization_bits' not in df.columns or 'mean_wer' not in df.columns:
        print("Missing required columns for compression vs performance plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Calculate compression ratio (assuming baseline is 5 bits for speech tokens)
    baseline_bits = 5  # Typical bits for speech tokens
    compression_ratio = baseline_bits / df['quantization_bits']

    # Compression ratio vs WER
    valid_mask = df['mean_wer'].notna() & df['quantization_bits'].notna()
    if valid_mask.sum() > 0:
        ax1.loglog(compression_ratio[valid_mask], df.loc[valid_mask, 'mean_wer'],
                  'bo-', linewidth=3, markersize=10, alpha=0.8)

        # Annotate points with bit levels
        for i, (ratio, wer, bits) in enumerate(zip(compression_ratio[valid_mask],
                                                   df.loc[valid_mask, 'mean_wer'],
                                                   df.loc[valid_mask, 'quantization_bits'])):
            ax1.annotate(f'{bits:.1f}b', (ratio, wer),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, alpha=0.8)

    ax1.set_xlabel('Compression Ratio (baseline/quantized bits)')
    ax1.set_ylabel('Word Error Rate (WER)')
    ax1.set_title('Speech Recognition Performance vs Compression')
    ax1.grid(True, alpha=0.3)

    # Information rate vs performance
    information_rate = df['quantization_bits'] / baseline_bits
    if valid_mask.sum() > 0:
        ax2.loglog(information_rate[valid_mask], df.loc[valid_mask, 'mean_wer'],
                  'ro-', linewidth=3, markersize=10, alpha=0.8)

        # Annotate points
        for i, (rate, wer, bits) in enumerate(zip(information_rate[valid_mask],
                                                  df.loc[valid_mask, 'mean_wer'],
                                                  df.loc[valid_mask, 'quantization_bits'])):
            ax2.annotate(f'{bits:.1f}b', (rate, wer),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, alpha=0.8)

    ax2.set_xlabel('Information Rate (quantized/baseline bits)')
    ax2.set_ylabel('Word Error Rate (WER)')
    ax2.set_title('Performance vs Information Rate')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved compression vs performance plot to: {save_path}")

    plt.show()


def plot_quantization_summary(
    df: pd.DataFrame,
    output_dir: str,
    experiment_name: str = "librispeech_quantization"
) -> None:
    """
    Generate all quantization analysis plots.

    Args:
        df: DataFrame with quantization analysis results
        output_dir: Directory to save plots
        experiment_name: Name for the experiment (used in filenames)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Quantization levels vs WER
    plot_quantization_levels_vs_wer(
        df,
        save_path=os.path.join(output_dir, f"{experiment_name}_levels_vs_wer.png")
    )

    # Plot 2: Scaling parameter stability
    plot_scaling_parameter_stability(
        df,
        save_path=os.path.join(output_dir, f"{experiment_name}_scaling_stability.png")
    )

    # Plot 3: Compression vs performance
    plot_compression_vs_performance(
        df,
        save_path=os.path.join(output_dir, f"{experiment_name}_compression_performance.png")
    )

    print(f"\nAll quantization analysis plots saved to: {output_dir}")


def print_quantization_summary(df: pd.DataFrame) -> None:
    """
    Print a summary of quantization analysis results.

    Args:
        df: DataFrame with quantization analysis results
    """
    if len(df) == 0:
        print("No quantization analysis results to summarize.")
        return

    print("\n" + "="*80)
    print("LIBRISPEECH QUANTIZATION ANALYSIS SUMMARY")
    print("="*80)

    print(f"Tested {len(df)} quantization levels:")

    for _, row in df.iterrows():
        levels = int(row['quantization_levels'])
        bits = row['quantization_bits']
        wer = row['mean_wer']

        print(f"  {levels:2d} levels ({bits:.1f} bits): WER = {wer:.4f}")

    # Best and worst performance
    if 'mean_wer' in df.columns:
        best_idx = df['mean_wer'].idxmin()
        worst_idx = df['mean_wer'].idxmax()

        print(f"\nBest performance:")
        print(f"  {int(df.loc[best_idx, 'quantization_levels'])} levels "
              f"({df.loc[best_idx, 'quantization_bits']:.1f} bits): "
              f"WER = {df.loc[best_idx, 'mean_wer']:.4f}")

        print(f"\nWorst performance:")
        print(f"  {int(df.loc[worst_idx, 'quantization_levels'])} levels "
              f"({df.loc[worst_idx, 'quantization_bits']:.1f} bits): "
              f"WER = {df.loc[worst_idx, 'mean_wer']:.4f}")

    # Correlation analysis
    if len(df) >= 3 and 'quantization_levels' in df.columns and 'mean_wer' in df.columns:
        from scipy.stats import pearsonr

        levels = df['quantization_levels'].values
        wer = df['mean_wer'].values

        corr, p_value = pearsonr(levels, wer)
        print(f"\nCorrelation between quantization levels and WER:")
        print(f"  Pearson r = {corr:.3f}, p = {p_value:.3f}")

        if 'quantization_bits' in df.columns:
            bits = df['quantization_bits'].values
            corr_bits, p_bits = pearsonr(bits, wer)
            print(f"  Correlation with bits: r = {corr_bits:.3f}, p = {p_bits:.3f}")

    print("="*80)