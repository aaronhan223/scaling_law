#!/usr/bin/env python3
"""
Information Analysis Experiment

This script investigates the relationship between useful information content in datasets
and the fitted scaling law parameters (alpha, beta, gamma). It defines various metrics
to quantify information density and studies how these metrics correlate with scaling
behavior.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import digamma
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score
from typing import Dict, List, Tuple, Optional, Any
import json
import time
from datetime import datetime
from collections import Counter
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

from config import ExperimentConfig
from experiments import ScalingLawExperiment
from training import set_seed
from data import CustomDataset


class InformationMetrics:
    """Class to compute various information density metrics for datasets."""
    
    def __init__(self, tokenizer_name: str = "gpt2"):
        """Initialize with tokenizer for text processing."""
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def compute_token_entropy(self, dataset: CustomDataset) -> float:
        """
        Compute Shannon entropy of token distribution.
        Higher entropy indicates more uniform token distribution.
        """
        all_tokens = []
        for i in range(min(len(dataset), 10000)):  # Sample for efficiency
            example = dataset[i]
            tokens = example['input_ids'].numpy().flatten()
            all_tokens.extend(tokens)
        
        # Count token frequencies
        token_counts = Counter(all_tokens)
        total_tokens = len(all_tokens)
        
        # Compute entropy
        entropy = 0.0
        for count in token_counts.values():
            prob = count / total_tokens
            if prob > 0:
                entropy -= prob * np.log2(prob)
        
        return entropy
    
    def compute_perplexity_based_complexity(self, dataset: CustomDataset) -> float:
        """
        Compute complexity based on token sequence predictability.
        Uses n-gram based perplexity estimation.
        """
        sequences = []
        for i in range(min(len(dataset), 5000)):
            example = dataset[i]
            tokens = example['input_ids'].numpy()
            sequences.append(tokens)
        
        # Compute bigram and trigram statistics
        bigram_counts = Counter()
        unigram_counts = Counter()
        total_bigrams = 0
        
        for seq in sequences:
            for i in range(len(seq) - 1):
                bigram = (seq[i], seq[i+1])
                bigram_counts[bigram] += 1
                unigram_counts[seq[i]] += 1
                total_bigrams += 1
        
        # Compute perplexity approximation
        log_prob_sum = 0.0
        count = 0
        
        for bigram, bigram_count in bigram_counts.items():
            unigram_count = unigram_counts[bigram[0]]
            # Laplace smoothing
            prob = (bigram_count + 1) / (unigram_count + len(unigram_counts))
            log_prob_sum += bigram_count * np.log2(prob)
            count += bigram_count
        
        if count > 0:
            avg_log_prob = log_prob_sum / count
            perplexity = 2 ** (-avg_log_prob)
        else:
            perplexity = float('inf')
        
        return perplexity
    
    def compute_vocabulary_richness(self, dataset: CustomDataset) -> Dict[str, float]:
        """
        Compute vocabulary richness metrics.
        """
        all_tokens = []
        for i in range(min(len(dataset), 10000)):
            example = dataset[i]
            tokens = example['input_ids'].numpy().flatten()
            all_tokens.extend(tokens)
        
        unique_tokens = len(set(all_tokens))
        total_tokens = len(all_tokens)
        
        # Type-Token Ratio (TTR)
        ttr = unique_tokens / total_tokens if total_tokens > 0 else 0
        
        # Root TTR
        rttr = unique_tokens / np.sqrt(total_tokens) if total_tokens > 0 else 0
        
        # Log TTR
        logttr = np.log(unique_tokens) / np.log(total_tokens) if total_tokens > 1 else 0
        
        return {
            'ttr': ttr,
            'rttr': rttr, 
            'logttr': logttr,
            'unique_tokens': unique_tokens,
            'total_tokens': total_tokens
        }
    
    def compute_compression_ratio(self, dataset: CustomDataset) -> float:
        """
        Estimate compression ratio as a measure of information content.
        Higher compression ratio indicates more redundancy.
        """
        # Sample text for compression analysis
        sample_texts = []
        for i in range(min(len(dataset), 1000)):
            example = dataset[i]
            tokens = example['input_ids'].numpy()
            # Convert tokens back to text (approximate)
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            sample_texts.append(text)
        
        # Compute compression using simple run-length encoding approximation
        combined_text = " ".join(sample_texts)
        original_length = len(combined_text)
        
        # Simple compression: count repeated patterns
        compressed_size = original_length
        for pattern_len in [2, 3, 4]:
            pattern_counts = Counter()
            for i in range(len(combined_text) - pattern_len + 1):
                pattern = combined_text[i:i+pattern_len]
                pattern_counts[pattern] += 1
            
            # Estimate savings from repeated patterns
            for pattern, count in pattern_counts.items():
                if count > 1:
                    savings = (count - 1) * (pattern_len - 1)  # Simple approximation
                    compressed_size -= min(savings, compressed_size * 0.1)  # Cap savings
        
        compression_ratio = compressed_size / original_length if original_length > 0 else 1.0
        return compression_ratio
    
    def compute_mutual_information_score(self, dataset: CustomDataset) -> float:
        """
        Compute mutual information between consecutive tokens.
        Higher MI indicates more predictable sequences.
        """
        token_pairs = []
        for i in range(min(len(dataset), 5000)):
            example = dataset[i]
            tokens = example['input_ids'].numpy()
            for j in range(len(tokens) - 1):
                token_pairs.append((tokens[j], tokens[j+1]))
        
        if len(token_pairs) == 0:
            return 0.0
        
        # Extract first and second tokens
        first_tokens = [pair[0] for pair in token_pairs]
        second_tokens = [pair[1] for pair in token_pairs]
        
        # Compute mutual information
        mi = mutual_info_score(first_tokens, second_tokens)
        return mi
    
    def compute_semantic_density(self, dataset: CustomDataset) -> float:
        """
        Compute semantic density using TF-IDF and dimensionality analysis.
        """
        # Sample text for semantic analysis
        sample_texts = []
        for i in range(min(len(dataset), 2000)):
            example = dataset[i]
            tokens = example['input_ids'].numpy()
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            if text.strip():  # Only non-empty texts
                sample_texts.append(text)
        
        if len(sample_texts) < 10:
            return 0.0
        
        try:
            # Compute TF-IDF vectors
            vectorizer = TfidfVectorizer(max_features=1000, stop_words=None)
            tfidf_matrix = vectorizer.fit_transform(sample_texts)
            
            # Use PCA to estimate intrinsic dimensionality
            pca = PCA()
            pca.fit(tfidf_matrix.toarray())
            
            # Find number of components explaining 90% of variance
            cumsum_var = np.cumsum(pca.explained_variance_ratio_)
            n_components_90 = np.argmax(cumsum_var >= 0.9) + 1
            
            # Semantic density as ratio of effective dimensions to max dimensions
            semantic_density = n_components_90 / len(sample_texts)
            
            return semantic_density
            
        except Exception as e:
            print(f"Warning: Could not compute semantic density: {e}")
            return 0.0
    
    def compute_effective_vocabulary_size(self, dataset: CustomDataset) -> int:
        """
        Compute the effective vocabulary size (number of unique tokens actually used).
        """
        all_tokens = set()
        for i in range(min(len(dataset), 10000)):
            example = dataset[i]
            tokens = example['input_ids'].numpy().flatten()
            all_tokens.update(tokens.tolist())
        return len(all_tokens)
    
    def compute_token_distribution_uniformity(self, dataset: CustomDataset) -> float:
        """
        Compute how uniform the token distribution is (0 = completely skewed, 1 = perfectly uniform).
        """
        all_tokens = []
        for i in range(min(len(dataset), 10000)):
            example = dataset[i]
            tokens = example['input_ids'].numpy().flatten()
            all_tokens.extend(tokens)
        
        if not all_tokens:
            return 0.0
        
        # Count token frequencies
        token_counts = Counter(all_tokens)
        total_tokens = len(all_tokens)
        unique_tokens = len(token_counts)
        
        # Compute uniformity using entropy ratio
        actual_entropy = 0.0
        for count in token_counts.values():
            prob = count / total_tokens
            if prob > 0:
                actual_entropy -= prob * np.log2(prob)
        
        # Maximum possible entropy (uniform distribution)
        max_entropy = np.log2(unique_tokens) if unique_tokens > 0 else 0
        
        return actual_entropy / max_entropy if max_entropy > 0 else 0.0
    
    def compute_information_density_suite(self, dataset: CustomDataset) -> Dict[str, float]:
        """
        Compute a comprehensive suite of information density metrics.
        """
        print(f"Computing information metrics for dataset with {len(dataset)} examples...")
        
        metrics = {}
        
        # Core vocabulary metrics
        effective_vocab_size = self.compute_effective_vocabulary_size(dataset)
        metrics['effective_vocab_size'] = effective_vocab_size
        print(f"  Effective vocabulary size: {effective_vocab_size}")
        
        # Token-level metrics
        metrics['token_entropy'] = self.compute_token_entropy(dataset)
        print(f"  Token entropy: {metrics['token_entropy']:.3f}")
        
        # Token distribution uniformity
        metrics['token_uniformity'] = self.compute_token_distribution_uniformity(dataset)
        print(f"  Token distribution uniformity: {metrics['token_uniformity']:.3f}")
        
        # Vocabulary richness
        vocab_metrics = self.compute_vocabulary_richness(dataset)
        metrics.update(vocab_metrics)
        print(f"  TTR: {vocab_metrics['ttr']:.3f}")
        
        # Sequence complexity
        metrics['perplexity_complexity'] = self.compute_perplexity_based_complexity(dataset)
        print(f"  Perplexity complexity: {metrics['perplexity_complexity']:.3f}")
        
        # Information content
        metrics['compression_ratio'] = self.compute_compression_ratio(dataset)
        print(f"  Compression ratio: {metrics['compression_ratio']:.3f}")
        
        # Sequential dependencies
        metrics['mutual_information'] = self.compute_mutual_information_score(dataset)
        print(f"  Mutual information: {metrics['mutual_information']:.3f}")
        
        # Semantic content
        metrics['semantic_density'] = self.compute_semantic_density(dataset)
        print(f"  Semantic density: {metrics['semantic_density']:.3f}")
        
        # Compute derived metrics
        # Information per token (entropy weighted by information content)
        metrics['info_per_token'] = metrics['token_entropy'] * (1 - metrics['compression_ratio'])
        
        # Vocabulary utilization ratio (how much of theoretical vocab is used)
        theoretical_vocab_size = 50257  # GPT-2 vocab size
        metrics['vocab_utilization'] = effective_vocab_size / theoretical_vocab_size
        
        # Information density (composite metric - weighted combination)
        metrics['information_density'] = (
            0.25 * metrics['token_entropy'] +                     # Raw entropy
            0.20 * metrics['token_uniformity'] * 10 +             # Distribution uniformity  
            0.15 * metrics['ttr'] * 10 +                          # Type-token ratio
            0.15 * (1.0 / max(metrics['perplexity_complexity'], 1.0)) * 100 +  # Inverse perplexity
            0.10 * (1 - metrics['compression_ratio']) * 10 +      # Information content
            0.10 * metrics['semantic_density'] * 10 +             # Semantic richness
            0.05 * metrics['vocab_utilization'] * 10              # Vocabulary utilization
        )
        
        print(f"  Information density (composite): {metrics['information_density']:.3f}")
        
        return metrics


def fit_scaling_law(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit power law y = A * x^(-alpha) and return alpha, intercept, and R².
    """
    if len(x) < 3 or len(y) < 3:
        return np.nan, np.nan, np.nan
    
    # Remove any invalid values
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 3:
        return np.nan, np.nan, np.nan
    
    x_clean = x[mask]
    y_clean = y[mask]
    
    # Fit in log space: log(y) = log(A) - alpha * log(x)
    log_x = np.log10(x_clean)
    log_y = np.log10(y_clean)
    
    slope, intercept, r_value, _, _ = stats.linregress(log_x, log_y)
    
    # Return -slope as alpha (since we expect negative slopes)
    alpha = -slope
    r_squared = r_value ** 2
    
    return alpha, intercept, r_squared


def extract_scaling_parameters(df: pd.DataFrame) -> Dict[str, float]:
    """Extract scaling law parameters from experiment results."""
    results = {}
    
    # Alpha: Loss vs Model Parameters
    if len(df) > 0 and 'model_params' in df.columns and 'final_loss' in df.columns:
        model_loss = df.groupby('model_params')['final_loss'].mean()
        if len(model_loss) >= 3:
            alpha, alpha_intercept, alpha_r2 = fit_scaling_law(
                model_loss.index.values, model_loss.values
            )
            results.update({
                'alpha': alpha,
                'alpha_intercept': alpha_intercept,
                'alpha_r2': alpha_r2
            })
    
    # Beta: Loss vs Dataset Size
    if len(df) > 0 and 'dataset_size' in df.columns and 'final_loss' in df.columns:
        data_loss = df.groupby('dataset_size')['final_loss'].mean()
        if len(data_loss) >= 3:
            beta, beta_intercept, beta_r2 = fit_scaling_law(
                data_loss.index.values, data_loss.values
            )
            results.update({
                'beta': beta,
                'beta_intercept': beta_intercept,
                'beta_r2': beta_r2
            })
    
    # Gamma: Loss vs Compute
    if len(df) > 0 and 'petaflop_days' in df.columns and 'final_loss' in df.columns:
        compute_df = df[df['petaflop_days'].notna()]
        if len(compute_df) >= 3:
            gamma, gamma_intercept, gamma_r2 = fit_scaling_law(
                compute_df['petaflop_days'].values, compute_df['final_loss'].values
            )
            results.update({
                'gamma': gamma,
                'gamma_intercept': gamma_intercept,
                'gamma_r2': gamma_r2
            })
    
    return results


def create_transformed_dataset_for_metrics(
    dataset_config: Dict[str, Any],
    base_config: ExperimentConfig,
) -> CustomDataset:
    """
    Create a transformed dataset for computing information metrics.
    """
    from data.transforms import (
        TransformedDataset, QuantizationTransform, ClusteringTransform,
        ModularTransform, LinearTransform, NonLinearTransform
    )
    
    # Create base dataset
    base_dataset = CustomDataset(
        split="train",
        dataset_name=dataset_config['dataset_name'],
        dataset_config=dataset_config.get('dataset_config'),
        tokenizer_name=base_config.data_config.tokenizer_name,
        max_length=base_config.data_config.max_length,
        max_tokens=min(dataset_config['dataset_sizes']),  # Use smallest size for efficiency
        seed=base_config.training_config.seed,
    )
    
    # Apply transformations based on config
    transform_type = dataset_config.get('transform_type', 'standard')
    
    if transform_type == 'standard':
        return base_dataset
    
    # Create appropriate transform
    transform = None
    vocab_size = base_config.model_config.vocab_size
    
    if transform_type == 'quantization':
        num_levels = int(vocab_size * dataset_config.get('info_ratio', 0.1))
        transform = QuantizationTransform(
            vocab_size=vocab_size,
            num_levels=num_levels,
            seed=base_config.training_config.seed
        )
        transformed_dataset = TransformedDataset(
            base_dataset,
            quantization_transform=transform,
            cache_transformed=True,
            batch_size=500
        )
    elif transform_type == 'clustering':
        num_clusters = int(vocab_size * dataset_config.get('info_ratio', 0.25))
        transform = ClusteringTransform(
            vocab_size=vocab_size,
            num_clusters=num_clusters,
            seed=base_config.training_config.seed
        )
        transformed_dataset = TransformedDataset(
            base_dataset,
            clustering_transform=transform,
            cache_transformed=True,
            batch_size=500
        )
    elif transform_type == 'modular':
        modulus = int(vocab_size * dataset_config.get('info_ratio', 0.125))
        transform = ModularTransform(
            vocab_size=vocab_size,
            modulus=modulus,
            seed=base_config.training_config.seed
        )
        transformed_dataset = TransformedDataset(
            base_dataset,
            modular_transform=transform,
            cache_transformed=True,
            batch_size=500
        )
    elif transform_type == 'linear':
        transform = LinearTransform(
            vocab_size=vocab_size,
            seed=base_config.training_config.seed
        )
        transformed_dataset = TransformedDataset(
            base_dataset,
            linear_transform=transform,
            cache_transformed=True,
            batch_size=500
        )
    elif transform_type == 'nonlinear':
        transform = NonLinearTransform(
            vocab_size=vocab_size,
            seed=base_config.training_config.seed
        )
        transformed_dataset = TransformedDataset(
            base_dataset,
            nonlinear_transform=transform,
            cache_transformed=True,
            batch_size=500
        )
    else:
        return base_dataset
    
    return transformed_dataset


def run_information_analysis(
    dataset_configs: List[Dict[str, Any]],
    base_config: ExperimentConfig,
    output_dir: str,
) -> pd.DataFrame:
    """
    Run scaling law experiments for different dataset information densities.
    """
    results = []
    
    for i, dataset_config in enumerate(dataset_configs):
        print(f"\n{'='*80}")
        print(f"INFORMATION ANALYSIS: Dataset {i+1}/{len(dataset_configs)}")
        print(f"Configuration: {dataset_config['name']}")
        print(f"{'='*80}")
        
        # Create modified config for this dataset configuration
        analysis_config = ExperimentConfig()
        
        # Copy base config settings
        analysis_config.model_config = base_config.model_config
        analysis_config.training_config = base_config.training_config
        analysis_config.experiment_name = f"info_analysis_{dataset_config['name']}"
        analysis_config.use_wandb = base_config.use_wandb
        analysis_config.wandb_project = f"{base_config.wandb_project}_information"
        
        # Set data configuration
        analysis_config.data_config.dataset_name = dataset_config['dataset_name']
        analysis_config.data_config.dataset_config = dataset_config.get('dataset_config')
        analysis_config.data_config.dataset_sizes = dataset_config['dataset_sizes']
        
        # Apply any transformations to create different information densities
        if 'transform_type' in dataset_config:
            transform_type = dataset_config['transform_type']
            
            # Reset all transform flags
            analysis_config.run_standard = False
            analysis_config.run_transformed = False
            analysis_config.run_linear_transform = False
            analysis_config.run_nonlinear_transform = False
            analysis_config.run_clustering_transform = False
            analysis_config.run_modular_transform = False
            analysis_config.run_quantization_transform = False
            
            if transform_type == 'standard':
                analysis_config.run_standard = True
            elif transform_type == 'quantization':
                analysis_config.run_quantization_transform = True
                vocab_size = analysis_config.model_config.vocab_size
                analysis_config.transform_config.quantization_num_levels = int(
                    vocab_size * dataset_config.get('info_ratio', 0.1)
                )
            elif transform_type == 'clustering':
                analysis_config.run_clustering_transform = True
                vocab_size = analysis_config.model_config.vocab_size
                analysis_config.transform_config.clustering_num_clusters = int(
                    vocab_size * dataset_config.get('info_ratio', 0.25)
                )
            elif transform_type == 'modular':
                analysis_config.run_modular_transform = True
                vocab_size = analysis_config.model_config.vocab_size
                analysis_config.transform_config.modular_modulus = int(
                    vocab_size * dataset_config.get('info_ratio', 0.125)
                )
            elif transform_type == 'linear':
                analysis_config.run_linear_transform = True
            elif transform_type == 'nonlinear':
                analysis_config.run_nonlinear_transform = True
        else:
            # Default to standard experiments if no transform specified
            analysis_config.run_standard = True
        
        # Create subdirectory for this configuration
        config_output_dir = os.path.join(output_dir, f"config_{dataset_config['name']}")
        analysis_config.training_config.output_dir = config_output_dir
        
        # Compute information metrics for this dataset configuration
        print("Computing information metrics...")
        
        # Create transformed dataset for metric computation
        sample_dataset = create_transformed_dataset_for_metrics(dataset_config, analysis_config)
        
        # Compute information metrics
        info_computer = InformationMetrics(analysis_config.data_config.tokenizer_name)
        info_metrics = info_computer.compute_information_density_suite(sample_dataset)
        
        try:
            # Run scaling experiments
            experiment = ScalingLawExperiment(analysis_config)
            experiment_results = experiment.run_all_experiments()
            
            if experiment_results:
                # Convert to DataFrame
                df = pd.DataFrame(experiment_results)
                
                # Extract scaling parameters
                scaling_params = extract_scaling_parameters(df)
                
                # Store results
                result = {
                    'config_name': dataset_config['name'],
                    'dataset_name': dataset_config['dataset_name'],
                    'dataset_config': dataset_config.get('dataset_config', 'default'),
                    'transform_type': dataset_config.get('transform_type', 'standard'),
                    'info_ratio': dataset_config.get('info_ratio', 1.0),
                    'num_experiments': len(df),
                    **info_metrics,
                    **scaling_params
                }
                
                # Add aggregate performance metrics
                if len(df) > 0:
                    result.update({
                        'mean_loss': df['final_loss'].mean(),
                        'std_loss': df['final_loss'].std(),
                        'min_loss': df['final_loss'].min(),
                        'max_loss': df['final_loss'].max(),
                    })
                
                results.append(result)
                
                print(f"Completed configuration {dataset_config['name']}")
                print(f"  Information density: {info_metrics.get('information_density', 'N/A'):.3f}")
                print(f"  Scaling parameters: α={scaling_params.get('alpha', 'N/A'):.3f}, "
                      f"β={scaling_params.get('beta', 'N/A'):.3f}, "
                      f"γ={scaling_params.get('gamma', 'N/A'):.3f}")
                
                # Save intermediate results
                results_df = pd.DataFrame(results)
                results_file = os.path.join(output_dir, "information_analysis_results.csv")
                results_df.to_csv(results_file, index=False)
                
        except Exception as e:
            print(f"Error running configuration {dataset_config['name']}: {e}")
            continue
    
    return pd.DataFrame(results)


def create_dataset_configurations(args) -> List[Dict[str, Any]]:
    """Create different dataset configurations for information analysis."""
    configs = []
    
    # Parse base dataset sizes
    base_sizes = parse_dataset_sizes(args.dataset_sizes)
    
    # Always include standard dataset (baseline) as first configuration
    configs.append({
        'name': 'standard',
        'dataset_name': args.dataset_name,
        'dataset_config': args.dataset_config if args.dataset_config != 'None' else None,
        'dataset_sizes': base_sizes,
        'transform_type': 'standard',
        'info_ratio': 1.0,
    })
    
    # Different information densities using transformations
    # Always include some transforms for comparison, even if not explicitly requested
    transform_configs = []
    
    if args.include_transforms:
        # Quantization with different levels (information loss)
        for ratio in [0.2, 0.4, 0.6, 0.8]:
            transform_configs.append({
                'name': f'quantized_{ratio:.3f}',
                'dataset_name': args.dataset_name,
                'dataset_config': args.dataset_config if args.dataset_config != 'None' else None,
                'dataset_sizes': base_sizes,
                'transform_type': 'quantization',
                'info_ratio': ratio,
            })
        
        # Add other transforms for comparison
        for ratio in [0.2, 0.5, 0.8]:
            transform_configs.append({
                'name': f'clustered_{ratio:.3f}',
                'dataset_name': args.dataset_name,
                'dataset_config': args.dataset_config if args.dataset_config != 'None' else None,
                'dataset_sizes': base_sizes,
                'transform_type': 'clustering',
                'info_ratio': ratio,
            })
            
        for ratio in [0.1, 0.25, 0.5]:
            transform_configs.append({
                'name': f'modular_{ratio:.3f}',
                'dataset_name': args.dataset_name,
                'dataset_config': args.dataset_config if args.dataset_config != 'None' else None,
                'dataset_sizes': base_sizes,
                'transform_type': 'modular',
                'info_ratio': ratio,
            })
    else:
        # Include at least a few quantization levels for basic comparison
        for ratio in [0.2, 0.8]:
            transform_configs.append({
                'name': f'quantized_{ratio:.3f}',
                'dataset_name': args.dataset_name,
                'dataset_config': args.dataset_config if args.dataset_config != 'None' else None,
                'dataset_sizes': base_sizes,
                'transform_type': 'quantization',
                'info_ratio': ratio,
            })
    
    configs.extend(transform_configs)
        
    # Different datasets (if specified)
    if args.multi_dataset:
        other_datasets = [
            ('wikitext', 'wikitext-2-v1'),
            ('wikitext', 'wikitext-103-v1'),
        ]
        
        if args.dataset_name not in ['wikitext']:
            other_datasets.append(('wikitext', 'wikitext-103-v1'))
        
        for dataset_name, dataset_config in other_datasets:
            configs.append({
                'name': f'{dataset_name}_{dataset_config}',
                'dataset_name': dataset_name,
                'dataset_config': dataset_config,
                'dataset_sizes': base_sizes,
                'transform_type': 'standard',
                'info_ratio': 1.0,
            })
    
    return configs


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Information Density Analysis")
    
    # Dataset settings
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="wikitext",
        help="Dataset name (default: wikitext)",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="wikitext-103-v1",
        help="Dataset config (default: wikitext-103-v1)",
    )
    parser.add_argument(
        "--dataset-sizes",
        type=str,
        default="1M,10M",  # Reduced for faster testing
        help="Comma-separated dataset sizes (default: 1M,10M)",
    )
    
    # Analysis settings
    parser.add_argument(
        "--include-transforms",
        action="store_true",
        default=True,  # Enable by default
        help="Include transformation-based information density variations",
    )
    parser.add_argument(
        "--multi-dataset",
        action="store_true",
        help="Include multiple different datasets for comparison",
    )
    
    # Model settings  
    parser.add_argument(
        "--model-sizes",
        type=str,
        default="0,1",  # Reduced for faster testing
        help="Comma-separated model size indices (default: 0,1 for small models)",
    )
    
    # Training settings
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size (default: 8)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4,
        help="Learning rate (default: 5e-4)",
    )
    
    # Output settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/information_analysis",
        help="Output directory (default: ./outputs/information_analysis)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="information_analysis",
        help="Experiment name (default: information_analysis)",
    )
    
    # WandB settings
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="information-analysis",
        help="WandB project name (default: information-analysis)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging",
    )
    
    # Other settings
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only generate plots from existing results",
    )
    parser.add_argument(
        "--validate-metrics",
        action="store_true",
        help="Run validation test to ensure metrics work correctly",
    )
    
    return parser.parse_args()


def parse_dataset_sizes(dataset_sizes_str: str) -> List[int]:
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


def parse_model_sizes(model_sizes_str: str, config: ExperimentConfig) -> None:
    """Parse and set model sizes in config."""
    indices = [int(x.strip()) for x in model_sizes_str.split(",")]
    
    all_hidden_sizes = config.model_config.hidden_sizes
    all_num_heads = config.model_config.num_attention_heads
    all_num_layers = config.model_config.num_hidden_layers
    
    config.model_config.hidden_sizes = [all_hidden_sizes[i] for i in indices if i < len(all_hidden_sizes)]
    config.model_config.num_attention_heads = [all_num_heads[i] for i in indices if i < len(all_num_heads)]
    config.model_config.num_hidden_layers = [all_num_layers[i] for i in indices if i < len(all_num_layers)]


def validate_information_metrics():
    """
    Validate that information metrics work correctly and show expected behavior.
    """
    from config import ExperimentConfig
    
    print("\n" + "="*80)
    print("INFORMATION METRICS VALIDATION")
    print("="*80)
    
    # Create a small test config
    config = ExperimentConfig()
    
    # Test different dataset configurations
    test_configs = [
        {
            'name': 'standard',
            'dataset_name': 'wikitext',
            'dataset_config': 'wikitext-103-v1',
            'dataset_sizes': [1000],  # Very small for testing
            'transform_type': 'standard',
            'info_ratio': 1.0,
        },
        {
            'name': 'quantized_low',
            'dataset_name': 'wikitext',
            'dataset_config': 'wikitext-103-v1',
            'dataset_sizes': [1000],
            'transform_type': 'quantization',
            'info_ratio': 0.1,  # Very low information
        },
        {
            'name': 'quantized_high',
            'dataset_name': 'wikitext',
            'dataset_config': 'wikitext-103-v1',
            'dataset_sizes': [1000],
            'transform_type': 'quantization',
            'info_ratio': 0.8,  # High information
        },
    ]
    
    info_computer = InformationMetrics()
    results = []
    
    for test_config in test_configs:
        print(f"\nTesting: {test_config['name']}")
        print(f"Transform: {test_config['transform_type']}, Ratio: {test_config['info_ratio']}")
        
        try:
            # Create dataset
            dataset = create_transformed_dataset_for_metrics(test_config, config)
            
            # Compute metrics
            metrics = info_computer.compute_information_density_suite(dataset)
            
            result = {'config': test_config['name'], **metrics}
            results.append(result)
            
        except Exception as e:
            print(f"Error testing {test_config['name']}: {e}")
            continue
    
    # Compare results
    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)
    
    if len(results) >= 2:
        standard = next((r for r in results if r['config'] == 'standard'), None)
        quantized_low = next((r for r in results if r['config'] == 'quantized_low'), None)
        quantized_high = next((r for r in results if r['config'] == 'quantized_high'), None)
        
        print("\nKey Metrics Comparison:")
        metrics_to_compare = ['effective_vocab_size', 'token_entropy', 'token_uniformity', 
                             'information_density', 'ttr']
        
        for metric in metrics_to_compare:
            print(f"\n{metric}:")
            if standard and metric in standard:
                print(f"  Standard:      {standard[metric]:.3f}")
            if quantized_low and metric in quantized_low:
                print(f"  Quantized Low: {quantized_low[metric]:.3f}")
            if quantized_high and metric in quantized_high:
                print(f"  Quantized High:{quantized_high[metric]:.3f}")
        
        # Validate expected behavior
        print("\nValidation Checks:")
        validation_passed = True
        
        if standard and quantized_low:
            if quantized_low['effective_vocab_size'] >= standard['effective_vocab_size']:
                print("  ❌ FAIL: Quantized dataset should have smaller effective vocabulary")
                validation_passed = False
            else:
                print("  ✅ PASS: Quantized dataset has smaller effective vocabulary")
                
            if quantized_low['information_density'] >= standard['information_density']:
                print("  ❌ FAIL: Quantized dataset should have lower information density")
                validation_passed = False
            else:
                print("  ✅ PASS: Quantized dataset has lower information density")
        
        if validation_passed:
            print("\n🎉 All validation checks passed!")
        else:
            print("\n⚠️  Some validation checks failed. Please review the metrics.")
            
    else:
        print("Not enough results for comparison")
    
    print("\n" + "="*80)


def main():
    """Main function."""
    args = parse_args()
    
    # Check if validation mode
    if args.validate_metrics:
        validate_information_metrics()
        return
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.experiment_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("DATASET INFORMATION DENSITY ANALYSIS")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    
    if args.plot_only:
        # Load existing results and generate plots
        results_file = os.path.join(args.output_dir, "information_analysis_results.csv")
        if not os.path.exists(results_file):
            print(f"Error: Results file not found at {results_file}")
            return
        
        print(f"Loading results from: {results_file}")
        df = pd.read_csv(results_file)
        
        # Generate plots
        try:
            from visualization.information_plots import plot_information_analysis
            plot_dir = os.path.join(os.path.dirname(results_file), "plots")
            plot_information_analysis(df, plot_dir)
            print(f"Plots saved to: {plot_dir}")
        except ImportError:
            print("Warning: Could not import information plotting functions")
            print("Run the visualization manually after creating the plot functions")
        
        return
    
    # Set up experiment configuration
    config = ExperimentConfig()
    config.experiment_name = args.experiment_name
    
    # Data config
    config.data_config.dataset_name = args.dataset_name
    config.data_config.dataset_config = args.dataset_config if args.dataset_config != 'None' else None
    config.data_config.train_batch_size = args.batch_size
    config.data_config.dataset_sizes = parse_dataset_sizes(args.dataset_sizes)
    
    # Training config
    config.training_config.learning_rate = args.learning_rate
    config.training_config.seed = args.seed
    
    # WandB config
    config.use_wandb = not args.no_wandb
    config.wandb_project = args.wandb_project
    
    # Parse model sizes
    parse_model_sizes(args.model_sizes, config)
    
    # Handle OpenWebText config
    if config.data_config.dataset_name == "openwebtext":
        config.data_config.dataset_config = None
    
    # Create dataset configurations for analysis
    dataset_configs = create_dataset_configurations(args)
    
    print(f"Testing {len(dataset_configs)} dataset configurations:")
    for config_info in dataset_configs:
        print(f"  {config_info['name']}: {config_info['dataset_name']} "
              f"({config_info.get('transform_type', 'standard')})")
    print(f"Model sizes: {len(config.model_config.hidden_sizes)}")
    print(f"Dataset sizes: {config.data_config.dataset_sizes}")
    print("=" * 80)
    
    # Save configuration
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump({
            'dataset_configs': dataset_configs,
            'args': vars(args),
            'base_config': config.__dict__
        }, f, indent=2, default=str)
    
    # Set seed
    set_seed(args.seed, deterministic=True)
    
    # Run analysis
    start_time = time.time()
    results_df = run_information_analysis(dataset_configs, config, output_dir)
    
    # Save final results
    if len(results_df) > 0:
        results_file = os.path.join(output_dir, "information_analysis_results.csv")
        results_df.to_csv(results_file, index=False)
        
        # Also save as JSON
        results_json = os.path.join(output_dir, "information_analysis_results.json")
        results_df.to_json(results_json, indent=2)
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"Total time: {time.time() - start_time:.1f} seconds")
        print(f"Results saved to: {results_file}")
        print(f"Tested {len(results_df)} dataset configurations")
        
        # Generate plots
        try:
            from visualization.information_plots import plot_information_analysis
            plot_dir = os.path.join(output_dir, "plots")
            plot_information_analysis(results_df, plot_dir)
            print(f"Plots saved to: {plot_dir}")
        except ImportError:
            print("Warning: Could not import information plotting functions")
            print("Run the visualization manually after creating the plot functions")
        
        # Print correlation analysis
        print("\nCorrelation Analysis:")
        info_cols = ['information_density', 'token_entropy', 'ttr', 'perplexity_complexity', 
                    'compression_ratio', 'mutual_information', 'semantic_density']
        scaling_cols = ['alpha', 'beta', 'gamma']
        
        for info_col in info_cols:
            if info_col in results_df.columns:
                for scaling_col in scaling_cols:
                    if scaling_col in results_df.columns:
                        valid_data = results_df[[info_col, scaling_col]].dropna()
                        if len(valid_data) > 2:
                            # Check if there's sufficient variation in both variables
                            info_var = valid_data[info_col].var()
                            scaling_var = valid_data[scaling_col].var()
                            if info_var > 1e-10 and scaling_var > 1e-10:
                                corr = valid_data[info_col].corr(valid_data[scaling_col])
                                print(f"  {info_col} vs {scaling_col}: r={corr:.3f}")
                            else:
                                print(f"  {info_col} vs {scaling_col}: r=N/A (insufficient variation)")
                        else:
                            print(f"  {info_col} vs {scaling_col}: r=N/A (insufficient data)")
        
        print("=" * 80)
    else:
        print("ERROR: No results collected!")


if __name__ == "__main__":
    main()