"""Speech scaling law experiment runner for LibriSpeech."""

import os
import json
import torch
from torch.utils.data import DataLoader
import wandb
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import time

from config import ExperimentConfig
from data.librispeech_dataset import create_librispeech_dataloaders, get_librispeech_hours_to_samples
from data.speech_transforms import create_speech_transformed_dataloaders
from models.wav2vec2_model import create_wav2vec2_model_configs, create_wav2vec2_model
from training import set_seed
from .compute_tracker import ComputeTracker


class SpeechScalingLawExperiment:
    """Run speech scaling law experiments on LibriSpeech."""

    def __init__(self, config: ExperimentConfig, librispeech_root: str, model_indices: Optional[List[int]] = None, cache_dir: str = "./dataset_cache"):
        """
        Initialize speech scaling law experiment.

        Args:
            config: Experiment configuration
            librispeech_root: Root directory containing LibriSpeech data
            model_indices: List of model indices to use (0-4), if None uses all
            cache_dir: Directory to store cached datasets
        """
        self.config = config
        self.librispeech_root = librispeech_root
        self.cache_dir = cache_dir
        self.results = []
        self.dataset_cache = {}

        # Set seeds
        set_seed(
            self.config.training_config.seed,
            deterministic=self.config.deterministic
        )

        # Get model configurations for speech
        all_model_configs = create_wav2vec2_model_configs()

        # Filter model configs based on indices
        if model_indices is not None:
            self.model_configs = [all_model_configs[i] for i in model_indices if 0 <= i < len(all_model_configs)]
        else:
            self.model_configs = all_model_configs

        # Convert dataset sizes from hours to sample counts
        self.dataset_sizes = self._convert_hours_to_samples(config.data_config.dataset_sizes)

        print(f"Speech scaling law experiment initialized")
        print(f"Model configurations: {len(self.model_configs)}")
        if model_indices is not None:
            print(f"Using model indices: {model_indices}")
        for i, config_item in enumerate(self.model_configs):
            print(f"  {i}: {config_item['model_name']} ({config_item['num_parameters']:,} params)")
        print(f"Dataset sizes (samples): {self.dataset_sizes}")

    def _convert_hours_to_samples(self, hours_list: List[int]) -> List[int]:
        """Convert hours to approximate sample counts."""
        sample_counts = []
        for hours in hours_list:
            if isinstance(hours, str):
                # Handle string inputs like "100h"
                if hours.endswith('h'):
                    hours = int(hours[:-1])
                else:
                    hours = int(hours)
            samples = get_librispeech_hours_to_samples(hours)
            sample_counts.append(samples)
        return sample_counts

    def _get_or_create_dataloaders(
        self,
        max_samples: int,
        experiment_type: str = "standard",
        cache_key: Optional[str] = None
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get dataloaders from cache or create new ones."""

        if cache_key is None:
            cache_key = f"librispeech_{max_samples}_{experiment_type}"

        if cache_key not in self.dataset_cache:
            print(f"Creating LibriSpeech dataloaders for {max_samples:,} samples...")

            if experiment_type == "standard":
                # Create standard dataloaders
                train_loader, valid_loader, test_loader = create_librispeech_dataloaders(
                    data_root=self.librispeech_root,
                    train_splits=["train-clean-100", "train-clean-360", "train-other-500"],
                    dev_split="dev-clean",
                    test_split="test-clean",
                    max_audio_length=self.config.data_config.max_length,
                    max_train_samples=max_samples,
                    max_val_samples=min(max_samples // 20, 5000),  # 5% or 5000 samples
                    max_test_samples=min(max_samples // 50, 2000),  # 2% or 2000 samples
                    train_batch_size=self.config.data_config.train_batch_size,
                    eval_batch_size=self.config.data_config.eval_batch_size,
                    num_workers=self.config.data_config.num_workers,
                    seed=self.config.training_config.seed,
                    cache_dir=self.cache_dir,
                    use_cache=True,
                )
            else:
                # First create base dataloaders
                base_train_loader, base_valid_loader, base_test_loader = create_librispeech_dataloaders(
                    data_root=self.librispeech_root,
                    train_splits=["train-clean-100", "train-clean-360", "train-other-500"],
                    dev_split="dev-clean",
                    test_split="test-clean",
                    max_audio_length=self.config.data_config.max_length,
                    max_train_samples=max_samples,
                    max_val_samples=min(max_samples // 20, 5000),
                    max_test_samples=min(max_samples // 50, 2000),
                    train_batch_size=self.config.data_config.train_batch_size,
                    eval_batch_size=self.config.data_config.eval_batch_size,
                    num_workers=self.config.data_config.num_workers,
                    seed=self.config.training_config.seed,
                    cache_dir=self.cache_dir,
                    use_cache=True,
                )

                # Get vocab size from one of the datasets
                vocab_size = base_train_loader.dataset.datasets[0].get_vocab_size()

                # Create transformed dataloaders
                train_loader, valid_loader, test_loader, _ = create_speech_transformed_dataloaders(
                    base_train_dataset=base_train_loader.dataset,
                    base_valid_dataset=base_valid_loader.dataset,
                    base_test_dataset=base_test_loader.dataset,
                    vocab_size=vocab_size,
                    feature_dim=None,  # We work with raw audio
                    train_batch_size=self.config.data_config.train_batch_size,
                    eval_batch_size=self.config.data_config.eval_batch_size,
                    num_workers=self.config.data_config.num_workers,
                    apply_linear=(experiment_type == "linear"),
                    apply_nonlinear=(experiment_type == "nonlinear"),
                    apply_quantization=(experiment_type == "quantization"),
                    apply_lowrank=(experiment_type == "lowrank"),
                    linear_seed=self.config.transform_config.linear_transform_seed,
                    nonlinear_seed=self.config.transform_config.nonlinear_seed,
                    quantization_seed=self.config.transform_config.quantization_seed,
                    lowrank_seed=self.config.transform_config.lowrank_seed,
                    cache_dir=self.cache_dir,
                    use_cache=True,
                )

            self.dataset_cache[cache_key] = (train_loader, valid_loader, test_loader)
        else:
            print(f"Using cached dataloaders for {max_samples:,} samples...")

        return self.dataset_cache[cache_key]

    def run_single_experiment(
        self,
        model_config: Dict[str, Any],
        dataset_size: int,
        experiment_type: str = "standard",
    ) -> Dict[str, Any]:
        """Run a single speech recognition experiment."""
        print(f"\n{'='*60}")
        print(f"Running {experiment_type} speech experiment:")
        print(f"Model: {model_config['model_name']} ({model_config['num_parameters']:,} params)")
        print(f"Dataset size: {dataset_size:,} samples")
        print(f"{'='*60}\n")

        # Get dataloaders
        train_loader, valid_loader, test_loader = self._get_or_create_dataloaders(
            dataset_size, experiment_type
        )

        # Get vocab size from dataset
        if hasattr(train_loader.dataset, 'datasets'):
            # ConcatDataset
            vocab_size = train_loader.dataset.datasets[0].get_vocab_size()
        else:
            # Direct dataset or transformed dataset
            base_dataset = getattr(train_loader.dataset, 'base_dataset', train_loader.dataset)
            if hasattr(base_dataset, 'datasets'):
                vocab_size = base_dataset.datasets[0].get_vocab_size()
            else:
                vocab_size = base_dataset.get_vocab_size()

        # Create model
        model = create_wav2vec2_model(model_config, vocab_size=vocab_size)
        print(f"Created model with {model.get_num_params():,} parameters")

        # Calculate training steps based on samples (approximation)
        estimated_tokens_per_sample = self.config.data_config.max_length * 16000 // 320  # Wav2Vec2 downsampling
        estimated_total_tokens = dataset_size * estimated_tokens_per_sample
        max_steps = int(self.config.training_config.get_max_steps_for_tokens(
            estimated_total_tokens,
            self.config.data_config.train_batch_size,
            estimated_tokens_per_sample,
        ))

        # Create compute tracker
        compute_tracker = None
        if self.config.track_compute:
            compute_tracker = ComputeTracker(
                model_params=model.get_num_params(),
                batch_size=self.config.data_config.train_batch_size,
                sequence_length=estimated_tokens_per_sample,
                gradient_accumulation_steps=self.config.training_config.gradient_accumulation_steps,
            )

        # Create output directory
        output_dir = self.config.get_output_dir(
            experiment_type,
            model_config['model_name'],
            dataset_size
        )

        # Import trainer here to avoid circular imports
        from training.speech_trainer import SpeechTrainer

        # Create trainer
        wandb_run_name = f"speech_{experiment_type}_{model_config['model_name']}_data{dataset_size}"
        trainer = SpeechTrainer(
            model=model,
            train_dataloader=train_loader,
            eval_dataloader=valid_loader,
            test_dataloader=test_loader,
            learning_rate=self.config.training_config.learning_rate,
            weight_decay=self.config.training_config.weight_decay,
            adam_beta1=self.config.training_config.adam_beta1,
            adam_beta2=self.config.training_config.adam_beta2,
            adam_epsilon=self.config.training_config.adam_epsilon,
            max_grad_norm=self.config.training_config.max_grad_norm,
            warmup_steps=self.config.training_config.warmup_steps,
            max_steps=max_steps,
            eval_steps=self.config.training_config.eval_steps,
            save_steps=self.config.training_config.save_steps,
            logging_steps=self.config.training_config.logging_steps,
            gradient_accumulation_steps=self.config.training_config.gradient_accumulation_steps,
            fp16=self.config.training_config.fp16,
            output_dir=output_dir,
            experiment_name=self.config.experiment_name,
            use_wandb=self.config.use_wandb,
            wandb_project=self.config.wandb_project,
            wandb_run_name=wandb_run_name,
            device=self.config.device,
            compute_tracker=compute_tracker,
            use_multi_gpu=self.config.multi_gpu,
        )

        # Train model
        train_results = trainer.train()

        # Get compute stats
        compute_stats = {}
        if compute_tracker:
            compute_stats = compute_tracker.get_stats()

        # Compile results
        result = {
            "experiment_type": experiment_type,
            "model_name": model_config['model_name'],
            "model_params": model_config['num_parameters'],
            "dataset_size": dataset_size,
            "final_loss": train_results["final_eval_metrics"]["loss"],
            "test_loss": train_results["final_eval_metrics"].get("test_loss", None),
            "dev_wer": train_results["final_eval_metrics"].get("wer", None),
            "test_wer": train_results["final_eval_metrics"].get("test_wer", None),
            "total_steps": train_results["total_steps"],
            "total_time": train_results["total_time"],
            **compute_stats,
        }

        # Close wandb run
        if self.config.use_wandb:
            wandb.finish()

        return result

    def run_all_experiments(self) -> List[Dict[str, Any]]:
        """Run all speech scaling law experiments."""
        dataset_sizes = self.dataset_sizes

        # Use the model configs that were already filtered in the constructor
        model_configs = self.model_configs

        print(f"\nRunning experiments with {len(model_configs)} model configurations:")

        # Pre-cache datasets if running multiple experiment types
        num_experiments = sum([
            self.config.run_standard,
            self.config.run_linear_transform,
            self.config.run_nonlinear_transform,
            self.config.run_quantization_transform,
            self.config.run_lowrank_transform
        ])

        if num_experiments > 1:
            print("\n" + "="*80)
            print("PRE-CACHING DATASETS FOR EFFICIENCY")
            print("="*80)
            for dataset_size in dataset_sizes:
                for exp_type in ["standard", "linear", "nonlinear", "quantization", "lowrank"]:
                    if getattr(self.config, f"run_{exp_type}_transform" if exp_type != "standard" else "run_standard"):
                        self._get_or_create_dataloaders(dataset_size, exp_type)

        # Run standard experiments
        if self.config.run_standard:
            print("\n" + "="*80)
            print("RUNNING STANDARD SPEECH EXPERIMENTS")
            print("="*80)

            for model_config in model_configs:
                for dataset_size in dataset_sizes:
                    # Skip very large combinations
                    if model_config['num_parameters'] > 5e7 and dataset_size > 200000:
                        print(f"Skipping large combination: {model_config['model_name']} with {dataset_size} samples")
                        continue

                    result = self.run_single_experiment(
                        model_config=model_config,
                        dataset_size=dataset_size,
                        experiment_type="standard",
                    )
                    self.results.append(result)

                    # Save intermediate results
                    self.save_results()

        # Run linear transform experiments
        if self.config.run_linear_transform:
            print("\n" + "="*80)
            print("RUNNING LINEAR TRANSFORM SPEECH EXPERIMENTS")
            print("="*80)

            for model_config in model_configs:
                for dataset_size in dataset_sizes:
                    if model_config['num_parameters'] > 5e7 and dataset_size > 200000:
                        print(f"Skipping large combination: {model_config['model_name']} with {dataset_size} samples")
                        continue

                    result = self.run_single_experiment(
                        model_config=model_config,
                        dataset_size=dataset_size,
                        experiment_type="linear",
                    )
                    self.results.append(result)
                    self.save_results()

        # Run nonlinear transform experiments
        if self.config.run_nonlinear_transform:
            print("\n" + "="*80)
            print("RUNNING NONLINEAR TRANSFORM SPEECH EXPERIMENTS")
            print("="*80)

            for model_config in model_configs:
                for dataset_size in dataset_sizes:
                    if model_config['num_parameters'] > 5e7 and dataset_size > 200000:
                        print(f"Skipping large combination: {model_config['model_name']} with {dataset_size} samples")
                        continue

                    result = self.run_single_experiment(
                        model_config=model_config,
                        dataset_size=dataset_size,
                        experiment_type="nonlinear",
                    )
                    self.results.append(result)
                    self.save_results()

        # Run quantization transform experiments
        if self.config.run_quantization_transform:
            print("\n" + "="*80)
            print("RUNNING QUANTIZATION TRANSFORM SPEECH EXPERIMENTS")
            print("="*80)

            for model_config in model_configs:
                for dataset_size in dataset_sizes:
                    if model_config['num_parameters'] > 5e7 and dataset_size > 200000:
                        print(f"Skipping large combination: {model_config['model_name']} with {dataset_size} samples")
                        continue

                    result = self.run_single_experiment(
                        model_config=model_config,
                        dataset_size=dataset_size,
                        experiment_type="quantization",
                    )
                    self.results.append(result)
                    self.save_results()

        # Run low-rank transform experiments
        if self.config.run_lowrank_transform:
            print("\n" + "="*80)
            print("RUNNING LOW-RANK TRANSFORM SPEECH EXPERIMENTS")
            print("="*80)

            for model_config in model_configs:
                for dataset_size in dataset_sizes:
                    if model_config['num_parameters'] > 5e7 and dataset_size > 200000:
                        print(f"Skipping large combination: {model_config['model_name']} with {dataset_size} samples")
                        continue

                    result = self.run_single_experiment(
                        model_config=model_config,
                        dataset_size=dataset_size,
                        experiment_type="lowrank",
                    )
                    self.results.append(result)
                    self.save_results()

        # Clear dataset cache to free memory
        self.dataset_cache.clear()

        return self.results

    def save_results(self):
        """Save experiment results to file."""
        results_dir = os.path.join(self.config.training_config.output_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        # Save as JSON
        results_file = os.path.join(results_dir, "speech_scaling_law_results.json")
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)

        # Save as CSV
        if self.results:
            df = pd.DataFrame(self.results)
            csv_file = os.path.join(results_dir, "speech_scaling_law_results.csv")
            df.to_csv(csv_file, index=False)

        print(f"Results saved to {results_dir}")

    def analyze_results(self) -> Dict[str, pd.DataFrame]:
        """Analyze and return results as DataFrames."""
        if not self.results:
            raise ValueError("No results to analyze. Run experiments first.")

        df = pd.DataFrame(self.results)

        # Separate results by experiment type
        standard_df = df[df['experiment_type'] == 'standard'].copy() if 'standard' in df['experiment_type'].values else pd.DataFrame()
        linear_df = df[df['experiment_type'] == 'linear'].copy() if 'linear' in df['experiment_type'].values else pd.DataFrame()
        nonlinear_df = df[df['experiment_type'] == 'nonlinear'].copy() if 'nonlinear' in df['experiment_type'].values else pd.DataFrame()
        quantization_df = df[df['experiment_type'] == 'quantization'].copy() if 'quantization' in df['experiment_type'].values else pd.DataFrame()
        lowrank_df = df[df['experiment_type'] == 'lowrank'].copy() if 'lowrank' in df['experiment_type'].values else pd.DataFrame()

        return {
            "all": df,
            "standard": standard_df,
            "linear": linear_df,
            "nonlinear": nonlinear_df,
            "quantization": quantization_df,
            "lowrank": lowrank_df,
        }