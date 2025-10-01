#!/usr/bin/env python3
"""
Cross-Domain Scaling Law Prediction Experiment

This experiment tests whether we can predict scaling laws for an unknown domain
based on information content analysis of a known domain. Specifically, we:

1. Train on a source domain (WikiText-103) and establish its scaling law
2. Measure information content of both source and target domains (PubMed)
3. Predict the scaling law for the target domain based on information distortion
4. Validate predictions by training on the target domain

The key hypothesis is that scaling laws are invariant to lossless transformations
but dependent on information content, so we can predict new scaling laws by
measuring information distortion between domains.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from scipy.spatial.distance import jensenshannon
from scipy.special import rel_entr, kl_div
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

import torch
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import ExperimentConfig
from experiments import ScalingLawExperiment
from training import set_seed
from data import CustomDataset
from information_analysis import InformationMetrics, fit_scaling_law, extract_scaling_parameters


class EnhancedCrossDomainMetrics:
    """
    Enhanced metrics for quantifying information content differences between domains.
    Uses more sophisticated information-theoretic and statistical measures.
    """
    
    def __init__(self, tokenizer_name: str = "gpt2"):
        """Initialize with tokenizer for text processing."""
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def compute_token_distribution_kl_divergence(self, source_dataset: CustomDataset, 
                                               target_dataset: CustomDataset, 
                                               sample_size: int = 10000) -> float:
        """
        Compute KL divergence between token distributions of source and target domains.
        KL(P||Q) measures how much information is lost when Q is used to approximate P.
        """
        def get_token_distribution(dataset: CustomDataset) -> np.ndarray:
            """Get normalized token distribution from dataset."""
            all_tokens = []
            for i in range(min(len(dataset), sample_size)):
                example = dataset[i]
                tokens = example['input_ids'].numpy().flatten()
                all_tokens.extend(tokens.tolist())
            
            # Count tokens and normalize
            token_counts = Counter(all_tokens)
            vocab_size = self.tokenizer.vocab_size
            distribution = np.zeros(vocab_size)
            
            total_tokens = len(all_tokens)
            for token_id, count in token_counts.items():
                if 0 <= token_id < vocab_size:
                    distribution[token_id] = count / total_tokens
            
            # Add small epsilon to avoid log(0)
            distribution = distribution + 1e-10
            distribution = distribution / distribution.sum()
            
            return distribution
        
        source_dist = get_token_distribution(source_dataset)
        target_dist = get_token_distribution(target_dataset)
        
        # Compute KL divergence: KL(source || target)
        kl_divergence = np.sum(source_dist * np.log(source_dist / target_dist))
        
        return kl_divergence
    
    def compute_jensen_shannon_divergence(self, source_dataset: CustomDataset, 
                                        target_dataset: CustomDataset,
                                        sample_size: int = 10000) -> float:
        """
        Compute Jensen-Shannon divergence between domains.
        JS divergence is symmetric and always finite, making it ideal for domain comparison.
        """
        def get_token_probabilities(dataset: CustomDataset) -> np.ndarray:
            """Get token probability distribution."""
            all_tokens = []
            for i in range(min(len(dataset), sample_size)):
                example = dataset[i]
                tokens = example['input_ids'].numpy().flatten()
                all_tokens.extend(tokens.tolist())
            
            token_counts = Counter(all_tokens)
            vocab_size = self.tokenizer.vocab_size
            probs = np.zeros(vocab_size)
            
            total_tokens = len(all_tokens)
            for token_id, count in token_counts.items():
                if 0 <= token_id < vocab_size:
                    probs[token_id] = count / total_tokens
            
            return probs
        
        source_probs = get_token_probabilities(source_dataset)
        target_probs = get_token_probabilities(target_dataset)
        
        # Use scipy's jensenshannon function (returns JSD^0.5, so we square it)
        js_distance = jensenshannon(source_probs, target_probs)
        js_divergence = js_distance ** 2
        
        return js_divergence
    
    def compute_vocabulary_overlap_metrics(self, source_dataset: CustomDataset, 
                                         target_dataset: CustomDataset,
                                         sample_size: int = 10000) -> Dict[str, float]:
        """
        Compute vocabulary overlap and divergence metrics between domains.
        """
        def get_vocabulary_set(dataset: CustomDataset) -> set:
            """Get unique vocabulary from dataset."""
            vocab = set()
            for i in range(min(len(dataset), sample_size)):
                example = dataset[i]
                tokens = example['input_ids'].numpy().flatten()
                vocab.update(tokens.tolist())
            return vocab
        
        source_vocab = get_vocabulary_set(source_dataset)
        target_vocab = get_vocabulary_set(target_dataset)
        
        # Compute overlap metrics
        intersection = source_vocab.intersection(target_vocab)
        union = source_vocab.union(target_vocab)
        
        jaccard_similarity = len(intersection) / len(union) if len(union) > 0 else 0.0
        vocabulary_overlap_ratio = len(intersection) / min(len(source_vocab), len(target_vocab)) if min(len(source_vocab), len(target_vocab)) > 0 else 0.0
        
        # Domain-specific vocabulary ratios
        source_only = source_vocab - target_vocab
        target_only = target_vocab - source_vocab
        
        source_specific_ratio = len(source_only) / len(source_vocab) if len(source_vocab) > 0 else 0.0
        target_specific_ratio = len(target_only) / len(target_vocab) if len(target_vocab) > 0 else 0.0
        
        return {
            'jaccard_similarity': jaccard_similarity,
            'vocabulary_overlap_ratio': vocabulary_overlap_ratio,
            'source_specific_ratio': source_specific_ratio,
            'target_specific_ratio': target_specific_ratio,
            'source_vocab_size': len(source_vocab),
            'target_vocab_size': len(target_vocab),
            'shared_vocab_size': len(intersection)
        }
    
    def compute_ngram_distribution_distance(self, source_dataset: CustomDataset, 
                                          target_dataset: CustomDataset,
                                          n: int = 2, sample_size: int = 5000) -> Dict[str, float]:
        """
        Compute n-gram distribution distances between domains.
        Measures structural/syntactic differences in language patterns.
        """
        def get_ngram_distribution(dataset: CustomDataset, n: int) -> Dict[tuple, float]:
            """Get normalized n-gram distribution."""
            ngram_counts = Counter()
            total_ngrams = 0
            
            for i in range(min(len(dataset), sample_size)):
                example = dataset[i]
                tokens = example['input_ids'].numpy().flatten()
                
                # Extract n-grams
                for j in range(len(tokens) - n + 1):
                    ngram = tuple(tokens[j:j+n])
                    ngram_counts[ngram] += 1
                    total_ngrams += 1
            
            # Normalize to probabilities
            ngram_probs = {}
            for ngram, count in ngram_counts.items():
                ngram_probs[ngram] = count / total_ngrams if total_ngrams > 0 else 0.0
            
            return ngram_probs
        
        source_ngrams = get_ngram_distribution(source_dataset, n)
        target_ngrams = get_ngram_distribution(target_dataset, n)
        
        # Get all unique n-grams
        all_ngrams = set(source_ngrams.keys()) | set(target_ngrams.keys())
        
        if len(all_ngrams) == 0:
            return {'ngram_kl_divergence': 0.0, 'ngram_js_divergence': 0.0, 'ngram_cosine_similarity': 1.0}
        
        # Create probability vectors
        source_probs = np.array([source_ngrams.get(ngram, 1e-10) for ngram in all_ngrams])
        target_probs = np.array([target_ngrams.get(ngram, 1e-10) for ngram in all_ngrams])
        
        # Normalize
        source_probs = source_probs / source_probs.sum()
        target_probs = target_probs / target_probs.sum()
        
        # Compute KL divergence
        kl_div = np.sum(source_probs * np.log(source_probs / target_probs))
        
        # Compute JS divergence
        js_div = jensenshannon(source_probs, target_probs) ** 2
        
        # Compute cosine similarity
        cosine_sim = cosine_similarity([source_probs], [target_probs])[0, 0]
        
        return {
            f'{n}gram_kl_divergence': kl_div,
            f'{n}gram_js_divergence': js_div,
            f'{n}gram_cosine_similarity': cosine_sim
        }
    
    def compute_semantic_distribution_distance(self, source_dataset: CustomDataset, 
                                             target_dataset: CustomDataset,
                                             sample_size: int = 2000) -> Dict[str, float]:
        """
        Compute semantic distribution distances using TF-IDF representations.
        Measures content/topic differences between domains.
        """
        def get_text_samples(dataset: CustomDataset) -> List[str]:
            """Extract text samples for semantic analysis."""
            texts = []
            for i in range(min(len(dataset), sample_size)):
                example = dataset[i]
                tokens = example['input_ids'].numpy()
                text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                if text.strip():
                    texts.append(text)
            return texts
        
        source_texts = get_text_samples(source_dataset)
        target_texts = get_text_samples(target_dataset)
        
        if len(source_texts) == 0 or len(target_texts) == 0:
            return {'semantic_cosine_distance': 1.0, 'semantic_kl_divergence': float('inf')}
        
        try:
            # Compute TF-IDF for combined corpus
            all_texts = source_texts + target_texts
            vectorizer = TfidfVectorizer(max_features=5000, stop_words=None, ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            # Split back into source and target
            source_matrix = tfidf_matrix[:len(source_texts)]
            target_matrix = tfidf_matrix[len(source_texts):]
            
            # Compute centroid representations
            source_centroid = source_matrix.mean(axis=0).A1
            target_centroid = target_matrix.mean(axis=0).A1
            
            # Cosine similarity between centroids
            cosine_sim = cosine_similarity([source_centroid], [target_centroid])[0, 0]
            cosine_distance = 1.0 - cosine_sim
            
            # KL divergence between normalized centroids
            source_probs = source_centroid + 1e-10
            target_probs = target_centroid + 1e-10
            source_probs = source_probs / source_probs.sum()
            target_probs = target_probs / target_probs.sum()
            
            kl_div = np.sum(source_probs * np.log(source_probs / target_probs))
            
            return {
                'semantic_cosine_distance': cosine_distance,
                'semantic_kl_divergence': kl_div
            }
            
        except Exception as e:
            print(f"Warning: Could not compute semantic distance: {e}")
            return {'semantic_cosine_distance': 1.0, 'semantic_kl_divergence': float('inf')}
    
    def compute_statistical_complexity_difference(self, source_dataset: CustomDataset, 
                                                target_dataset: CustomDataset,
                                                sample_size: int = 5000) -> Dict[str, float]:
        """
        Compute statistical complexity differences between domains.
        Based on predictive information and effective measure complexity.
        """
        def compute_predictive_information(dataset: CustomDataset) -> float:
            """Compute predictive information (mutual information between past and future)."""
            sequences = []
            for i in range(min(len(dataset), sample_size)):
                example = dataset[i]
                tokens = example['input_ids'].numpy().flatten()
                sequences.append(tokens)
            
            if not sequences:
                return 0.0
            
            # Compute mutual information between past and future tokens
            past_tokens = []
            future_tokens = []
            
            for seq in sequences:
                mid_point = len(seq) // 2
                if mid_point > 0 and mid_point < len(seq):
                    # Use simple past/future split
                    past_context = tuple(seq[:mid_point][-3:])  # Last 3 tokens as context
                    future_token = seq[mid_point]
                    past_tokens.append(past_context)
                    future_tokens.append(future_token)
            
            if len(past_tokens) == 0:
                return 0.0
            
            # Compute conditional entropy H(future|past) and H(future)
            past_counts = Counter(past_tokens)
            future_counts = Counter(future_tokens)
            joint_counts = Counter(zip(past_tokens, future_tokens))
            
            total = len(past_tokens)
            
            # H(future|past)
            conditional_entropy = 0.0
            for (past, future), count in joint_counts.items():
                joint_prob = count / total
                past_prob = past_counts[past] / total
                conditional_prob = count / past_counts[past]
                if conditional_prob > 0:
                    conditional_entropy -= joint_prob * np.log2(conditional_prob)
            
            # H(future)
            marginal_entropy = 0.0
            for future, count in future_counts.items():
                prob = count / total
                if prob > 0:
                    marginal_entropy -= prob * np.log2(prob)
            
            # Predictive information = H(future) - H(future|past)
            predictive_info = marginal_entropy - conditional_entropy
            
            return max(0.0, predictive_info)
        
        source_pred_info = compute_predictive_information(source_dataset)
        target_pred_info = compute_predictive_information(target_dataset)
        
        # Relative difference in predictive information
        if source_pred_info > 0:
            pred_info_ratio = target_pred_info / source_pred_info
            pred_info_diff = abs(target_pred_info - source_pred_info) / source_pred_info
        else:
            pred_info_ratio = 1.0 if target_pred_info == 0 else float('inf')
            pred_info_diff = target_pred_info
        
        return {
            'source_predictive_information': source_pred_info,
            'target_predictive_information': target_pred_info,
            'predictive_information_ratio': pred_info_ratio,
            'predictive_information_difference': pred_info_diff
        }
    
    def compute_comprehensive_domain_distance(self, source_dataset: CustomDataset, 
                                            target_dataset: CustomDataset) -> Dict[str, float]:
        """
        Compute comprehensive information-theoretic distance between domains.
        Combines multiple metrics for robust domain difference quantification.
        """
        print("Computing comprehensive domain distance metrics...")
        
        metrics = {}
        
        # Token-level distributional differences
        print("  Computing token distribution distances...")
        try:
            kl_div = self.compute_token_distribution_kl_divergence(source_dataset, target_dataset)
            js_div = self.compute_jensen_shannon_divergence(source_dataset, target_dataset)
            metrics.update({
                'token_kl_divergence': kl_div,
                'token_js_divergence': js_div
            })
        except Exception as e:
            print(f"    Warning: Could not compute token distances: {e}")
            metrics.update({
                'token_kl_divergence': float('inf'),
                'token_js_divergence': 1.0
            })
        
        # Vocabulary overlap analysis
        print("  Computing vocabulary overlap metrics...")
        try:
            vocab_metrics = self.compute_vocabulary_overlap_metrics(source_dataset, target_dataset)
            metrics.update(vocab_metrics)
        except Exception as e:
            print(f"    Warning: Could not compute vocabulary metrics: {e}")
        
        # N-gram structural differences
        print("  Computing n-gram distribution distances...")
        for n in [2, 3]:  # Bigrams and trigrams
            try:
                ngram_metrics = self.compute_ngram_distribution_distance(source_dataset, target_dataset, n=n)
                metrics.update(ngram_metrics)
            except Exception as e:
                print(f"    Warning: Could not compute {n}-gram metrics: {e}")
        
        # Semantic content differences
        print("  Computing semantic distribution distances...")
        try:
            semantic_metrics = self.compute_semantic_distribution_distance(source_dataset, target_dataset)
            metrics.update(semantic_metrics)
        except Exception as e:
            print(f"    Warning: Could not compute semantic metrics: {e}")
        
        # Statistical complexity differences
        print("  Computing statistical complexity differences...")
        try:
            complexity_metrics = self.compute_statistical_complexity_difference(source_dataset, target_dataset)
            metrics.update(complexity_metrics)
        except Exception as e:
            print(f"    Warning: Could not compute complexity metrics: {e}")
        
        # Compute composite information distance score
        print("  Computing composite information distance...")
        self._compute_composite_distance_score(metrics)
        
        return metrics
    
    def _compute_composite_distance_score(self, metrics: Dict[str, float]) -> None:
        """
        Compute a composite information distance score combining multiple metrics.
        """
        # Normalize and weight different distance components
        components = []
        weights = []
        
        # Token-level differences (high weight - fundamental)
        if 'token_js_divergence' in metrics and np.isfinite(metrics['token_js_divergence']):
            components.append(metrics['token_js_divergence'])
            weights.append(0.25)
        
        # Vocabulary differences (medium weight)
        if 'jaccard_similarity' in metrics:
            vocab_distance = 1.0 - metrics['jaccard_similarity']
            components.append(vocab_distance)
            weights.append(0.20)
        
        # Structural differences via n-grams (medium weight)
        if '2gram_js_divergence' in metrics and np.isfinite(metrics['2gram_js_divergence']):
            components.append(metrics['2gram_js_divergence'])
            weights.append(0.20)
        
        # Semantic differences (medium weight)
        if 'semantic_cosine_distance' in metrics and np.isfinite(metrics['semantic_cosine_distance']):
            components.append(metrics['semantic_cosine_distance'])
            weights.append(0.20)
        
        # Statistical complexity differences (lower weight)
        if 'predictive_information_difference' in metrics and np.isfinite(metrics['predictive_information_difference']):
            # Normalize to [0,1] range
            normalized_pred_diff = min(1.0, metrics['predictive_information_difference'])
            components.append(normalized_pred_diff)
            weights.append(0.15)
        
        if len(components) > 0:
            # Weighted average
            total_weight = sum(weights[:len(components)])
            if total_weight > 0:
                composite_distance = sum(w * c for w, c in zip(weights, components)) / total_weight
            else:
                composite_distance = np.mean(components)
        else:
            composite_distance = 1.0  # Maximum distance if no valid metrics
        
        metrics['composite_information_distance'] = composite_distance
        metrics['information_similarity'] = 1.0 - composite_distance
        
        print(f"    Composite information distance: {composite_distance:.3f}")
        print(f"    Information similarity: {metrics['information_similarity']:.3f}")


class CrossDomainScalingPredictor:
    """
    Predicts scaling laws for target domains based on source domain scaling laws
    and information content analysis.
    """
    
    def __init__(self, tokenizer_name: str = "gpt2"):
        """Initialize the cross-domain predictor."""
        self.tokenizer_name = tokenizer_name
        self.info_metrics = InformationMetrics(tokenizer_name)
        self.enhanced_metrics = EnhancedCrossDomainMetrics(tokenizer_name)
        
        # Store source domain data
        self.source_scaling_params = None
        self.source_info_metrics = None
        self.source_dataset_name = None
        self.source_dataset = None  # Store for cross-domain comparison
        
    def establish_source_scaling_law(
        self, 
        source_config: Dict[str, Any],
        base_config: ExperimentConfig,
        output_dir: str
    ) -> Dict[str, float]:
        """
        Establish scaling law for the source domain through experiments.
        
        Args:
            source_config: Configuration for source domain dataset
            base_config: Base experiment configuration
            output_dir: Output directory for results
            
        Returns:
            Dictionary containing scaling law parameters and information metrics
        """
        print(f"\n{'='*80}")
        print("ESTABLISHING SOURCE DOMAIN SCALING LAW")
        print(f"Source Domain: {source_config['dataset_name']}")
        print(f"{'='*80}")
        
        # Create experiment config for source domain
        source_exp_config = ExperimentConfig()
        
        # Copy base settings
        source_exp_config.model_config = base_config.model_config
        source_exp_config.training_config = base_config.training_config
        source_exp_config.experiment_name = f"source_{source_config['name']}"
        source_exp_config.use_wandb = base_config.use_wandb
        source_exp_config.wandb_project = f"{base_config.wandb_project}_crossdomain"
        
        # Set data configuration
        source_exp_config.data_config.dataset_name = source_config['dataset_name']
        source_exp_config.data_config.dataset_config = source_config.get('dataset_config')
        source_exp_config.data_config.dataset_sizes = source_config['dataset_sizes']
        
        # Enable standard experiments only
        source_exp_config.run_standard = True
        source_exp_config.run_transformed = False
        source_exp_config.run_linear_transform = False
        source_exp_config.run_nonlinear_transform = False
        source_exp_config.run_clustering_transform = False
        source_exp_config.run_modular_transform = False
        source_exp_config.run_quantization_transform = False
        source_exp_config.run_lowrank_transform = False
        
        # Set output directory
        source_output_dir = os.path.join(output_dir, "source_domain")
        source_exp_config.training_config.output_dir = source_output_dir
        
        # Compute information metrics for source domain
        print("Computing source domain information metrics...")
        source_dataset = CustomDataset(
            split="train",
            dataset_name=source_config['dataset_name'],
            dataset_config=source_config.get('dataset_config'),
            tokenizer_name=self.tokenizer_name,
            max_length=base_config.data_config.max_length,
            max_tokens=min(source_config['dataset_sizes']),  # Use smallest size for efficiency
            seed=base_config.training_config.seed,
        )
        
        source_info_metrics = self.info_metrics.compute_information_density_suite(source_dataset)
        
        # Run scaling law experiments on source domain
        print("Running scaling law experiments on source domain...")
        experiment = ScalingLawExperiment(source_exp_config)
        experiment_results = experiment.run_all_experiments()
        
        if not experiment_results:
            raise RuntimeError("Failed to run source domain experiments")
            
        # Extract scaling parameters
        df = pd.DataFrame(experiment_results)
        source_scaling_params = extract_scaling_parameters(df)
        
        # Store source domain data
        self.source_scaling_params = source_scaling_params
        self.source_info_metrics = source_info_metrics
        self.source_dataset_name = source_config['dataset_name']
        self.source_dataset = source_dataset  # Store for cross-domain comparison
        
        # Combine results
        source_results = {
            'dataset_name': source_config['dataset_name'],
            'dataset_config': source_config.get('dataset_config', 'default'),
            'num_experiments': len(df),
            **source_info_metrics,
            **source_scaling_params
        }
        
        # Add aggregate performance metrics
        if len(df) > 0:
            source_results.update({
                'mean_loss': df['final_loss'].mean(),
                'std_loss': df['final_loss'].std(),
                'min_loss': df['final_loss'].min(),
                'max_loss': df['final_loss'].max(),
            })
        
        print(f"Source domain scaling law established:")
        print(f"  α (model scaling): {source_scaling_params.get('alpha', 'N/A'):.3f}")
        print(f"  β (data scaling): {source_scaling_params.get('beta', 'N/A'):.3f}")
        print(f"  γ (compute scaling): {source_scaling_params.get('gamma', 'N/A'):.3f}")
        print(f"  Information density: {source_info_metrics.get('information_density', 'N/A'):.3f}")
        
        return source_results
    
    def compute_information_distortion(
        self, 
        target_config: Dict[str, Any],
        base_config: ExperimentConfig
    ) -> Dict[str, float]:
        """
        Compute information distortion between source and target domains.
        
        Args:
            target_config: Configuration for target domain dataset
            base_config: Base experiment configuration
            
        Returns:
            Dictionary containing information distortion metrics
        """
        print(f"\n{'='*80}")
        print("COMPUTING INFORMATION DISTORTION")
        print(f"Target Domain: {target_config['dataset_name']}")
        print(f"{'='*80}")
        
        # Compute information metrics for target domain
        print("Computing target domain information metrics...")
        target_dataset = CustomDataset(
            split="train",
            dataset_name=target_config['dataset_name'],
            dataset_config=target_config.get('dataset_config'),
            tokenizer_name=self.tokenizer_name,
            max_length=base_config.data_config.max_length,
            max_tokens=min(target_config['dataset_sizes']),  # Use smallest size for efficiency
            seed=base_config.training_config.seed,
        )
        
        target_info_metrics = self.info_metrics.compute_information_density_suite(target_dataset)
        
        if self.source_info_metrics is None:
            raise RuntimeError("Source domain scaling law must be established first")
        
        # Compute distortion metrics (relative changes from source to target)
        distortion_metrics = {}
        
        for key in self.source_info_metrics:
            if key in target_info_metrics:
                source_val = self.source_info_metrics[key]
                target_val = target_info_metrics[key]
                
                # Compute relative change (handle division by zero)
                if abs(source_val) > 1e-10:
                    relative_change = (target_val - source_val) / source_val
                else:
                    relative_change = 0.0
                    
                # Compute absolute ratio
                if abs(source_val) > 1e-10:
                    ratio = target_val / source_val
                else:
                    ratio = 1.0 if abs(target_val) < 1e-10 else float('inf')
                
                distortion_metrics[f'{key}_relative_change'] = relative_change
                distortion_metrics[f'{key}_ratio'] = ratio
                
                # Store absolute values too
                distortion_metrics[f'source_{key}'] = source_val
                distortion_metrics[f'target_{key}'] = target_val
        
        # Compute composite information distortion score
        # Higher distortion means more information loss from source to target
        key_metrics = ['information_density', 'token_entropy', 'ttr', 'effective_vocab_size']
        
        composite_distortion = 0.0
        valid_metrics = 0
        
        for metric in key_metrics:
            ratio_key = f'{metric}_ratio'
            if ratio_key in distortion_metrics:
                ratio = distortion_metrics[ratio_key]
                if np.isfinite(ratio) and ratio > 0:
                    # Convert ratio to distortion score (1 = no distortion, 0 = complete loss)
                    distortion_score = min(ratio, 1/ratio) if ratio != 1 else 1.0
                    composite_distortion += distortion_score
                    valid_metrics += 1
        
        if valid_metrics > 0:
            composite_distortion /= valid_metrics
        else:
            composite_distortion = 1.0  # Default to no distortion if we can't compute
            
        distortion_metrics['composite_information_distortion'] = composite_distortion
        
        print(f"Information distortion analysis complete:")
        print(f"  Composite distortion score: {composite_distortion:.3f}")
        print(f"  Target information density: {target_info_metrics.get('information_density', 'N/A'):.3f}")
        print(f"  Source information density: {self.source_info_metrics.get('information_density', 'N/A'):.3f}")
        
        # Also store target metrics for later use
        distortion_metrics.update({f'target_{k}': v for k, v in target_info_metrics.items()})
        
        return distortion_metrics
    
    def predict_target_scaling_law(
        self, 
        distortion_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Predict scaling law for target domain based on information distortion.
        
        The core hypothesis is that scaling laws are preserved under lossless transformations
        but modified under lossy transformations proportional to information loss.
        
        Args:
            distortion_metrics: Information distortion metrics between domains
            
        Returns:
            Dictionary containing predicted scaling law parameters
        """
        print(f"\n{'='*80}")
        print("PREDICTING TARGET DOMAIN SCALING LAW")
        print(f"{'='*80}")
        
        if self.source_scaling_params is None:
            raise RuntimeError("Source domain scaling law must be established first")
        
        predictions = {}
        
        # Get composite distortion score (1 = no distortion, <1 = information loss)
        distortion_factor = distortion_metrics.get('composite_information_distortion', 1.0)
        
        # Get information density ratio as another predictor
        info_density_ratio = distortion_metrics.get('information_density_ratio', 1.0)
        
        # Predict scaling exponents based on information theory principles
        # The key insight: information loss should affect different scaling dimensions differently
        
        # Alpha (model parameter scaling): Less affected by domain shift,
        # more affected by fundamental information complexity
        source_alpha = self.source_scaling_params.get('alpha', np.nan)
        if not np.isnan(source_alpha):
            # Alpha should be less sensitive to domain shift but affected by information density
            alpha_adjustment = 1.0 + 0.3 * (1/info_density_ratio - 1.0) if info_density_ratio > 0 else 1.0
            predictions['predicted_alpha'] = source_alpha * alpha_adjustment
        else:
            predictions['predicted_alpha'] = np.nan
            
        # Beta (data scaling): More affected by information content changes
        source_beta = self.source_scaling_params.get('beta', np.nan)
        if not np.isnan(source_beta):
            # Beta should be more sensitive to information density changes
            beta_adjustment = 1.0 + 0.5 * (1/info_density_ratio - 1.0) if info_density_ratio > 0 else 1.0
            predictions['predicted_beta'] = source_beta * beta_adjustment
        else:
            predictions['predicted_beta'] = np.nan
            
        # Gamma (compute scaling): Should be similar across domains for same model architectures
        source_gamma = self.source_scaling_params.get('gamma', np.nan)
        if not np.isnan(source_gamma):
            # Gamma should be least affected by domain shift
            gamma_adjustment = 1.0 + 0.1 * (1/info_density_ratio - 1.0) if info_density_ratio > 0 else 1.0
            predictions['predicted_gamma'] = source_gamma * gamma_adjustment
        else:
            predictions['predicted_gamma'] = np.nan
        
        # Alternative prediction based on distortion factor
        predictions['predicted_alpha_distortion'] = source_alpha / (distortion_factor ** 0.2) if not np.isnan(source_alpha) else np.nan
        predictions['predicted_beta_distortion'] = source_beta / (distortion_factor ** 0.4) if not np.isnan(source_beta) else np.nan
        predictions['predicted_gamma_distortion'] = source_gamma / (distortion_factor ** 0.1) if not np.isnan(source_gamma) else np.nan
        
        # Store prediction methodology metadata
        predictions['prediction_method'] = 'information_distortion_based'
        predictions['distortion_factor'] = distortion_factor
        predictions['info_density_ratio'] = info_density_ratio
        predictions['source_alpha'] = source_alpha
        predictions['source_beta'] = source_beta
        predictions['source_gamma'] = source_gamma
        
        print(f"Scaling law predictions:")
        print(f"  Source α: {source_alpha:.3f} → Predicted α: {predictions['predicted_alpha']:.3f}")
        print(f"  Source β: {source_beta:.3f} → Predicted β: {predictions['predicted_beta']:.3f}")
        print(f"  Source γ: {source_gamma:.3f} → Predicted γ: {predictions['predicted_gamma']:.3f}")
        print(f"  Distortion factor: {distortion_factor:.3f}")
        print(f"  Info density ratio: {info_density_ratio:.3f}")
        
        return predictions
    
    def validate_predictions(
        self,
        target_config: Dict[str, Any],
        predictions: Dict[str, float],
        base_config: ExperimentConfig,
        output_dir: str
    ) -> Dict[str, float]:
        """
        Validate predictions by running actual experiments on target domain.
        
        Args:
            target_config: Configuration for target domain dataset
            predictions: Predicted scaling law parameters
            base_config: Base experiment configuration
            output_dir: Output directory for results
            
        Returns:
            Dictionary containing validation results and accuracy metrics
        """
        print(f"\n{'='*80}")
        print("VALIDATING PREDICTIONS ON TARGET DOMAIN")
        print(f"Target Domain: {target_config['dataset_name']}")
        print(f"{'='*80}")
        
        # Create experiment config for target domain
        target_exp_config = ExperimentConfig()
        
        # Copy base settings
        target_exp_config.model_config = base_config.model_config
        target_exp_config.training_config = base_config.training_config
        target_exp_config.experiment_name = f"target_{target_config['name']}"
        target_exp_config.use_wandb = base_config.use_wandb
        target_exp_config.wandb_project = f"{base_config.wandb_project}_crossdomain"
        
        # Set data configuration
        target_exp_config.data_config.dataset_name = target_config['dataset_name']
        target_exp_config.data_config.dataset_config = target_config.get('dataset_config')
        target_exp_config.data_config.dataset_sizes = target_config['dataset_sizes']
        
        # Enable standard experiments only
        target_exp_config.run_standard = True
        target_exp_config.run_transformed = False
        target_exp_config.run_linear_transform = False
        target_exp_config.run_nonlinear_transform = False
        target_exp_config.run_clustering_transform = False
        target_exp_config.run_modular_transform = False
        target_exp_config.run_quantization_transform = False
        target_exp_config.run_lowrank_transform = False
        
        # Set output directory
        target_output_dir = os.path.join(output_dir, "target_domain")
        target_exp_config.training_config.output_dir = target_output_dir
        
        # Run scaling law experiments on target domain
        print("Running scaling law experiments on target domain...")
        experiment = ScalingLawExperiment(target_exp_config)
        experiment_results = experiment.run_all_experiments()
        
        if not experiment_results:
            raise RuntimeError("Failed to run target domain experiments")
        
        # Extract actual scaling parameters
        df = pd.DataFrame(experiment_results)
        actual_scaling_params = extract_scaling_parameters(df)
        
        # Compute prediction accuracy
        validation_results = {}
        
        # Store actual values
        validation_results['actual_alpha'] = actual_scaling_params.get('alpha', np.nan)
        validation_results['actual_beta'] = actual_scaling_params.get('beta', np.nan)
        validation_results['actual_gamma'] = actual_scaling_params.get('gamma', np.nan)
        
        # Store predictions
        validation_results['predicted_alpha'] = predictions.get('predicted_alpha', np.nan)
        validation_results['predicted_beta'] = predictions.get('predicted_beta', np.nan)
        validation_results['predicted_gamma'] = predictions.get('predicted_gamma', np.nan)
        
        # Alternative distortion-based predictions
        validation_results['predicted_alpha_distortion'] = predictions.get('predicted_alpha_distortion', np.nan)
        validation_results['predicted_beta_distortion'] = predictions.get('predicted_beta_distortion', np.nan)
        validation_results['predicted_gamma_distortion'] = predictions.get('predicted_gamma_distortion', np.nan)
        
        # Compute prediction errors
        for param in ['alpha', 'beta', 'gamma']:
            actual_val = validation_results[f'actual_{param}']
            predicted_val = validation_results[f'predicted_{param}']
            predicted_val_dist = validation_results[f'predicted_{param}_distortion']
            
            if not np.isnan(actual_val) and not np.isnan(predicted_val):
                abs_error = abs(actual_val - predicted_val)
                rel_error = abs_error / abs(actual_val) if abs(actual_val) > 1e-10 else float('inf')
                validation_results[f'{param}_abs_error'] = abs_error
                validation_results[f'{param}_rel_error'] = rel_error
            else:
                validation_results[f'{param}_abs_error'] = np.nan
                validation_results[f'{param}_rel_error'] = np.nan
                
            # Distortion-based prediction errors
            if not np.isnan(actual_val) and not np.isnan(predicted_val_dist):
                abs_error_dist = abs(actual_val - predicted_val_dist)
                rel_error_dist = abs_error_dist / abs(actual_val) if abs(actual_val) > 1e-10 else float('inf')
                validation_results[f'{param}_abs_error_distortion'] = abs_error_dist
                validation_results[f'{param}_rel_error_distortion'] = rel_error_dist
            else:
                validation_results[f'{param}_abs_error_distortion'] = np.nan
                validation_results[f'{param}_rel_error_distortion'] = np.nan
        
        # Compute overall prediction accuracy
        valid_errors = []
        for param in ['alpha', 'beta', 'gamma']:
            rel_error = validation_results[f'{param}_rel_error']
            if not np.isnan(rel_error) and np.isfinite(rel_error):
                valid_errors.append(rel_error)
        
        if valid_errors:
            validation_results['mean_relative_error'] = np.mean(valid_errors)
            validation_results['prediction_accuracy'] = max(0.0, 1.0 - np.mean(valid_errors))
        else:
            validation_results['mean_relative_error'] = np.nan
            validation_results['prediction_accuracy'] = np.nan
        
        # Same for distortion-based predictions
        valid_errors_dist = []
        for param in ['alpha', 'beta', 'gamma']:
            rel_error = validation_results[f'{param}_rel_error_distortion']
            if not np.isnan(rel_error) and np.isfinite(rel_error):
                valid_errors_dist.append(rel_error)
        
        if valid_errors_dist:
            validation_results['mean_relative_error_distortion'] = np.mean(valid_errors_dist)
            validation_results['prediction_accuracy_distortion'] = max(0.0, 1.0 - np.mean(valid_errors_dist))
        else:
            validation_results['mean_relative_error_distortion'] = np.nan
            validation_results['prediction_accuracy_distortion'] = np.nan
        
        # Add aggregate performance metrics from target experiments
        if len(df) > 0:
            validation_results.update({
                'target_mean_loss': df['final_loss'].mean(),
                'target_std_loss': df['final_loss'].std(),
                'target_min_loss': df['final_loss'].min(),
                'target_max_loss': df['final_loss'].max(),
                'target_num_experiments': len(df)
            })
        
        print(f"Validation results:")
        print(f"  Actual α: {validation_results['actual_alpha']:.3f}, "
              f"Predicted α: {validation_results['predicted_alpha']:.3f}, "
              f"Error: {validation_results['alpha_rel_error']*100:.1f}%")
        print(f"  Actual β: {validation_results['actual_beta']:.3f}, "
              f"Predicted β: {validation_results['predicted_beta']:.3f}, "
              f"Error: {validation_results['beta_rel_error']*100:.1f}%")
        print(f"  Actual γ: {validation_results['actual_gamma']:.3f}, "
              f"Predicted γ: {validation_results['predicted_gamma']:.3f}, "
              f"Error: {validation_results['gamma_rel_error']*100:.1f}%")
        print(f"  Overall prediction accuracy: {validation_results['prediction_accuracy']*100:.1f}%")
        
        return validation_results


def parse_sizes(sizes_str: str) -> List[int]:
    """Parse dataset/model sizes from string."""
    sizes = []
    for size in sizes_str.split(","):
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
                print(f"Warning: Cannot parse size '{size}', skipping")
    return sorted(sizes)


def parse_model_indices(indices_str: str) -> List[int]:
    """Parse model indices from string."""
    return [int(x.strip()) for x in indices_str.split(",")]


def create_domain_configs(args) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Create source and target domain configurations."""
    
    # Parse dataset sizes
    dataset_sizes = parse_sizes(args.dataset_sizes)
    
    # Source domain configuration (WikiText-103)
    source_config = {
        'name': 'wikitext103',
        'dataset_name': 'wikitext',
        'dataset_config': 'wikitext-103-v1',
        'dataset_sizes': dataset_sizes,
    }
    
    # Target domain configuration (PubMed)
    target_config = {
        'name': 'pubmed',
        'dataset_name': 'ncbi/pubmed',
        'dataset_config': None,  # Default config for ncbi/pubmed
        'dataset_sizes': dataset_sizes,
    }
    
    return source_config, target_config


def run_cross_domain_experiment(
    source_config: Dict[str, Any],
    target_config: Dict[str, Any],
    base_config: ExperimentConfig,
    output_dir: str
) -> pd.DataFrame:
    """
    Run the complete cross-domain scaling law prediction experiment.
    
    Args:
        source_config: Source domain configuration
        target_config: Target domain configuration
        base_config: Base experiment configuration
        output_dir: Output directory for results
        
    Returns:
        DataFrame containing all results
    """
    
    predictor = CrossDomainScalingPredictor(base_config.data_config.tokenizer_name)
    
    results = []
    
    try:
        # Step 1: Establish source domain scaling law
        print("Step 1/4: Establishing source domain scaling law...")
        source_results = predictor.establish_source_scaling_law(source_config, base_config, output_dir)
        
        # Step 2: Compute information distortion
        print("Step 2/4: Computing information distortion...")
        distortion_metrics = predictor.compute_information_distortion(target_config, base_config)
        
        # Step 3: Predict target domain scaling law
        print("Step 3/4: Predicting target domain scaling law...")
        predictions = predictor.predict_target_scaling_law(distortion_metrics)
        
        # Step 4: Validate predictions
        print("Step 4/4: Validating predictions...")
        validation_results = predictor.validate_predictions(target_config, predictions, base_config, output_dir)
        
        # Combine all results
        combined_results = {
            'experiment_type': 'cross_domain_prediction',
            'source_domain': source_config['dataset_name'],
            'target_domain': target_config['dataset_name'],
            'timestamp': datetime.now().isoformat(),
            **source_results,
            **distortion_metrics,
            **predictions,
            **validation_results
        }
        
        results.append(combined_results)
        
    except Exception as e:
        print(f"Error in cross-domain experiment: {e}")
        import traceback
        traceback.print_exc()
        
    return pd.DataFrame(results)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Cross-Domain Scaling Law Prediction Experiment")
    
    # Dataset settings
    parser.add_argument(
        "--dataset-sizes",
        type=str,
        default="1K,10K,100K",
        help="Comma-separated dataset sizes (default: 1K,10K,100K)",
    )
    
    # Model settings  
    parser.add_argument(
        "--model-sizes",
        type=str,
        default="0,1,2",
        help="Comma-separated model size indices (default: 0,1,2)",
    )
    
    # Training settings
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size (default: 4)",
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
        default="./outputs/cross_domain_experiment",
        help="Output directory (default: ./outputs/cross_domain_experiment)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="cross_domain_scaling",
        help="Experiment name (default: cross_domain_scaling)",
    )
    
    # WandB settings
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="cross-domain-scaling",
        help="WandB project name (default: cross-domain-scaling)",
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
    
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.experiment_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("CROSS-DOMAIN SCALING LAW PREDICTION EXPERIMENT")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    
    # Set up experiment configuration
    config = ExperimentConfig()
    config.experiment_name = args.experiment_name
    
    # Data config
    config.data_config.train_batch_size = args.batch_size
    
    # Training config
    config.training_config.learning_rate = args.learning_rate
    config.training_config.seed = args.seed
    
    # WandB config
    config.use_wandb = not args.no_wandb
    config.wandb_project = args.wandb_project
    
    # Parse and set model sizes
    model_indices = parse_model_indices(args.model_sizes)
    all_hidden_sizes = config.model_config.hidden_sizes
    all_num_heads = config.model_config.num_attention_heads
    all_num_layers = config.model_config.num_hidden_layers
    
    config.model_config.hidden_sizes = [all_hidden_sizes[i] for i in model_indices if i < len(all_hidden_sizes)]
    config.model_config.num_attention_heads = [all_num_heads[i] for i in model_indices if i < len(all_num_heads)]
    config.model_config.num_hidden_layers = [all_num_layers[i] for i in model_indices if i < len(all_num_layers)]
    
    # Create domain configurations
    source_config, target_config = create_domain_configs(args)
    
    print(f"Source Domain: {source_config['dataset_name']} ({source_config.get('dataset_config', 'default')})")
    print(f"Target Domain: {target_config['dataset_name']} ({target_config.get('dataset_config', 'default')})")
    print(f"Dataset sizes: {parse_sizes(args.dataset_sizes)}")
    print(f"Model sizes: {len(config.model_config.hidden_sizes)}")
    print("=" * 80)
    
    # Save configuration
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump({
            'source_config': source_config,
            'target_config': target_config,
            'args': vars(args),
            'base_config': config.__dict__
        }, f, indent=2, default=str)
    
    # Set seed
    set_seed(args.seed, deterministic=True)
    
    # Run experiment
    start_time = time.time()
    results_df = run_cross_domain_experiment(source_config, target_config, config, output_dir)
    
    # Save results
    if len(results_df) > 0:
        results_file = os.path.join(output_dir, "cross_domain_results.csv")
        results_df.to_csv(results_file, index=False)
        
        # Also save as JSON for better readability
        results_json = os.path.join(output_dir, "cross_domain_results.json")
        results_df.to_json(results_json, indent=2)
        
        print("\n" + "=" * 80)
        print("CROSS-DOMAIN EXPERIMENT COMPLETE")
        print("=" * 80)
        print(f"Total time: {time.time() - start_time:.1f} seconds")
        print(f"Results saved to: {results_file}")
        
        # Print summary results
        if len(results_df) > 0:
            result = results_df.iloc[0]
            print(f"\nExperiment Summary:")
            print(f"  Source domain: {result.get('source_domain', 'N/A')}")
            print(f"  Target domain: {result.get('target_domain', 'N/A')}")
            
            print(f"\nSource Domain Scaling Law:")
            print(f"  α: {result.get('source_alpha', 'N/A'):.3f}")
            print(f"  β: {result.get('source_beta', 'N/A'):.3f}")  
            print(f"  γ: {result.get('source_gamma', 'N/A'):.3f}")
            
            print(f"\nTarget Domain - Predicted vs Actual:")
            print(f"  α: {result.get('predicted_alpha', 'N/A'):.3f} vs {result.get('actual_alpha', 'N/A'):.3f} "
                  f"({result.get('alpha_rel_error', 0)*100:.1f}% error)")
            print(f"  β: {result.get('predicted_beta', 'N/A'):.3f} vs {result.get('actual_beta', 'N/A'):.3f} "
                  f"({result.get('beta_rel_error', 0)*100:.1f}% error)")
            print(f"  γ: {result.get('predicted_gamma', 'N/A'):.3f} vs {result.get('actual_gamma', 'N/A'):.3f} "
                  f"({result.get('gamma_rel_error', 0)*100:.1f}% error)")
            
            print(f"\nPrediction Accuracy: {result.get('prediction_accuracy', 0)*100:.1f}%")
            print(f"Information Distortion Factor: {result.get('distortion_factor', 'N/A'):.3f}")
            
        print("=" * 80)
    else:
        print("ERROR: No results collected!")


if __name__ == "__main__":
    main()