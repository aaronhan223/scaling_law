"""Image scaling law experiment runner for ImageNet."""

import os
import json
from torch.utils.data import DataLoader
import wandb
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn

from config import ExperimentConfig
from models.vision_transformer import create_vision_model, get_vision_model_configs
from data.imagenet_dataset import (
    create_imagenet_dataloaders,
    create_imagenet_dataloaders_with_transforms,
    parse_imagenet_size
)
from data.image_transforms import create_image_transform
from training import Trainer, ImageTrainer, set_seed


class ImageScalingLawExperiment:
    """Run image scaling law experiments on ImageNet."""

    def __init__(self, config: ExperimentConfig):
        """Initialize experiment with configuration."""
        self.config = config
        self.results = []
        self.dataset_cache = {}  # Cache for loaded datasets

        # Set seeds
        set_seed(
            self.config.training_config.seed,
            deterministic=self.config.deterministic
        )

        # ImageNet specific settings
        self.imagenet_path = "/cis/home/xhan56/imagenet"
        self.img_size = getattr(config, 'img_size', 224)
        self.num_classes = getattr(config, 'num_classes', 1000)

    def _get_vision_model_configs(self) -> List[Dict[str, Any]]:
        """Get vision model configurations based on experiment config."""
        import copy
        all_configs = get_vision_model_configs()

        # Make deep copies to avoid mutating the original configs
        all_configs = [copy.deepcopy(config) for config in all_configs]

        # Update num_classes in all configs to match experiment config
        for config in all_configs:
            config['num_classes'] = self.num_classes

        # Filter based on experiment config if specified
        if hasattr(self.config, 'model_indices') and self.config.model_indices:
            selected_configs = [all_configs[i] for i in self.config.model_indices if i < len(all_configs)]
            return selected_configs

        return all_configs

    def _get_or_create_dataloaders(
        self,
        dataset_size: int,
        experiment_type: str = "standard",
        cache_key: Optional[str] = None
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get dataloaders from cache or create new ones."""
        if cache_key is None:
            cache_key = f"imagenet_{dataset_size}_{experiment_type}"

        if cache_key not in self.dataset_cache:
            print(f"Creating ImageNet dataloaders for {dataset_size:,} samples...")

            # Parse dataset sizes for train/val/test
            max_samples_train = dataset_size
            # Ensure validation and test sets have at least some samples
            max_samples_val = max(min(dataset_size // 10, 50000), 100)  # At least 100 samples
            max_samples_test = max(min(dataset_size // 10, 10000), 50)   # At least 50 samples

            print(f"Dataset size configuration: train={max_samples_train}, val={max_samples_val}, test={max_samples_test}")

            if experiment_type == "standard":
                # Create standard dataloaders
                train_loader, val_loader, test_loader = create_imagenet_dataloaders(
                    data_path=self.imagenet_path,
                    img_size=self.img_size,
                    train_batch_size=self.config.data_config.train_batch_size,
                    eval_batch_size=self.config.data_config.eval_batch_size,
                    num_workers=self.config.data_config.num_workers,
                    max_samples_train=max_samples_train,
                    max_samples_val=max_samples_val,
                    max_samples_test=max_samples_test,
                    num_classes=self.num_classes if self.num_classes < 1000 else None,
                    seed=self.config.training_config.seed,
                    normalize=True,
                    pin_memory=True,
                )
            else:
                # Create transforms based on experiment type
                transforms = self._create_transforms(experiment_type)

                # Create dataloaders with transforms
                train_loader, val_loader, test_loader = create_imagenet_dataloaders_with_transforms(
                    data_path=self.imagenet_path,
                    img_size=self.img_size,
                    train_batch_size=self.config.data_config.train_batch_size,
                    eval_batch_size=self.config.data_config.eval_batch_size,
                    num_workers=self.config.data_config.num_workers,
                    max_samples_train=max_samples_train,
                    max_samples_val=max_samples_val,
                    max_samples_test=max_samples_test,
                    num_classes=self.num_classes if self.num_classes < 1000 else None,
                    seed=self.config.training_config.seed,
                    normalize=True,
                    pin_memory=True,
                    **transforms
                )

            self.dataset_cache[cache_key] = (train_loader, val_loader, test_loader)
        else:
            print(f"Using cached dataloaders for {dataset_size:,} samples...")

        return self.dataset_cache[cache_key]

    def _create_transforms(self, experiment_type: str) -> Dict[str, Any]:
        """Create transforms based on experiment type."""
        transforms = {
            "linear_transform": None,
            "nonlinear_transform": None,
            "quantization_transform": None,
            "lowrank_transform": None,
        }

        if experiment_type == "linear":
            # Create linear transform for image channels/pixels
            transforms["linear_transform"] = create_image_transform(
                transform_name="linear",
                input_dim=3,  # RGB channels
                transform_dim=3,  # Keep same channel dimension
                transform_type="channel",
                seed=self.config.transform_config.linear_transform_seed,
            )

        elif experiment_type == "nonlinear":
            # Create non-linear transform for image channels
            transforms["nonlinear_transform"] = create_image_transform(
                transform_name="nonlinear",
                input_dim=3,  # RGB channels
                hidden_dims=[12, 6, 3],  # Expand-maintain-compress for channels
                transform_type="channel",
                nonlinear_type="gelu",
                dropout_rate=0.1,
                noise_scale=0.05,
                seed=self.config.transform_config.nonlinear_seed,
            )

        elif experiment_type == "quantization":
            # Create quantization transform for pixels
            transforms["quantization_transform"] = create_image_transform(
                transform_name="quantization",
                input_dim=None,  # Not used for pixel-level quantization
                num_levels=16,  # 4-bit quantization
                transform_type="pixel",
                quantization_mode="uniform",
                seed=self.config.transform_config.quantization_seed,
            )

        elif experiment_type == "lowrank":
            # Create low-rank transform for image channels
            rank = self.config.transform_config.lowrank_rank if self.config.transform_config.lowrank_rank is not None else 2
            transforms["lowrank_transform"] = create_image_transform(
                transform_name="lowrank",
                input_dim=3,  # RGB channels
                rank=rank,
                transform_type="channel",
                seed=self.config.transform_config.lowrank_seed,
            )

        return transforms

    def run_single_experiment(
        self,
        model_config: Dict[str, Any],
        dataset_size: int,
        experiment_type: str = "standard",
    ) -> Dict[str, Any]:
        """Run a single training experiment."""
        print(f"\n{'='*60}")
        print(f"Running {experiment_type} image experiment:")
        print(f"Model: {model_config['model_name']} ({model_config['num_parameters']:,} params)")
        print(f"Dataset size: {dataset_size:,} samples")
        print(f"{'='*60}\n")

        # Get or create dataloaders
        train_loader, val_loader, test_loader = self._get_or_create_dataloaders(
            dataset_size, experiment_type
        )

        # Create vision model
        model = create_vision_model(model_config)
        print(f"Created vision model with {model.get_num_params():,} parameters")
        print(f"Model num_classes: {model.num_classes}")

        # Validate that dataset labels match model num_classes
        print("\nValidating dataset labels...")
        sample_batch = next(iter(train_loader))
        sample_labels = sample_batch['labels']
        print(f"Sample batch labels - min: {sample_labels.min().item()}, max: {sample_labels.max().item()}")
        print(f"Sample batch has {len(torch.unique(sample_labels))} unique classes")

        if sample_labels.max().item() >= model.num_classes:
            raise ValueError(
                f"Dataset has labels up to {sample_labels.max().item()} but model only has "
                f"{model.num_classes} classes. Labels must be in range [0, {model.num_classes-1}]"
            )

        # Calculate training steps for images (different from text)
        # For images, we typically train for epochs rather than tokens
        samples_per_epoch = len(train_loader.dataset)
        steps_per_epoch = len(train_loader)

        # Default to fewer epochs for image experiments due to computational cost
        num_epochs = getattr(self.config.training_config, 'num_epochs', 10)
        max_steps = num_epochs * steps_per_epoch

        print(f"Training for {num_epochs} epochs ({max_steps:,} steps)")

        # Create compute tracker (optional)
        compute_tracker = None
        if hasattr(self.config, 'track_compute') and self.config.track_compute:
            try:
                from experiments.compute_tracker import ComputeTracker
                compute_tracker = ComputeTracker(
                    model_params=model.get_num_params(),
                    batch_size=self.config.data_config.train_batch_size,
                    sequence_length=224*224,  # Image patches equivalent
                    gradient_accumulation_steps=self.config.training_config.gradient_accumulation_steps,
                )
            except ImportError:
                print("Warning: ComputeTracker not available, skipping compute tracking")

        # Create output directory
        output_dir = self.config.get_output_dir(
            experiment_type,
            model_config['model_name'],
            dataset_size
        )

        # Create trainer with image-specific settings
        wandb_run_name = f"image_{experiment_type}_{model_config['model_name']}_data{dataset_size}"
        trainer = ImageTrainer(
            model=model,
            train_dataloader=train_loader,
            eval_dataloader=val_loader,
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

        # Compile results
        result = {
            "experiment_type": experiment_type,
            "model_name": model_config['model_name'],
            "model_params": model_config['num_parameters'],
            "dataset_size": dataset_size,
            "final_loss": train_results["final_eval_metrics"]["loss"],
            "final_accuracy": train_results["final_eval_metrics"].get("accuracy", None),
            "test_loss": train_results["final_eval_metrics"].get("test_loss", None),
            "test_accuracy": train_results["final_eval_metrics"].get("test_accuracy", None),
            "total_steps": train_results["total_steps"],
            "total_time": train_results["total_time"],
            "samples_per_epoch": samples_per_epoch,
            "num_epochs": num_epochs,
        }

        # Close wandb run
        if self.config.use_wandb:
            wandb.finish()

        return result

    def run_all_experiments(self) -> List[Dict[str, Any]]:
        """Run all image scaling law experiments."""
        model_configs = self._get_vision_model_configs()

        # Parse dataset sizes
        dataset_sizes = []
        if hasattr(self.config.data_config, 'dataset_sizes'):
            for size_str in self.config.data_config.dataset_sizes:
                if isinstance(size_str, str):
                    dataset_sizes.append(parse_imagenet_size(size_str))
                else:
                    dataset_sizes.append(size_str)
        else:
            # Default dataset sizes for image experiments
            dataset_sizes = [1000, 10000, 50000, 100000]

        print(f"Running experiments with dataset sizes: {dataset_sizes}")

        # Run standard experiments
        if self.config.run_standard:
            print("\n" + "="*80)
            print("RUNNING STANDARD IMAGE EXPERIMENTS")
            print("="*80)

            for model_config in model_configs:
                for dataset_size in dataset_sizes:
                    # Skip very large combinations to save compute
                    if model_config['num_parameters'] > 100e6 and dataset_size > 100000:
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

        # Run transformed experiments
        experiment_types = []
        if self.config.run_linear_transform:
            experiment_types.append("linear")
        if self.config.run_nonlinear_transform:
            experiment_types.append("nonlinear")
        if self.config.run_quantization_transform:
            experiment_types.append("quantization")
        if self.config.run_lowrank_transform:
            experiment_types.append("lowrank")

        for experiment_type in experiment_types:
            print(f"\n" + "="*80)
            print(f"RUNNING {experiment_type.upper()} TRANSFORM IMAGE EXPERIMENTS")
            print("="*80)

            for model_config in model_configs:
                for dataset_size in dataset_sizes:
                    # Skip very large combinations
                    if model_config['num_parameters'] > 100e6 and dataset_size > 100000:
                        print(f"Skipping large combination: {model_config['model_name']} with {dataset_size} samples")
                        continue

                    result = self.run_single_experiment(
                        model_config=model_config,
                        dataset_size=dataset_size,
                        experiment_type=experiment_type,
                    )
                    self.results.append(result)

                    # Save intermediate results
                    self.save_results()

        # Clear dataset cache to free memory
        self.dataset_cache.clear()

        return self.results

    def save_results(self):
        """Save experiment results to file."""
        results_dir = os.path.join(self.config.training_config.output_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        # Save as JSON
        results_file = os.path.join(results_dir, "image_scaling_law_results.json")
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)

        # Save as CSV
        if self.results:
            df = pd.DataFrame(self.results)
            csv_file = os.path.join(results_dir, "image_scaling_law_results.csv")
            df.to_csv(csv_file, index=False)

        print(f"Image scaling law results saved to {results_dir}")

    def analyze_results(self) -> Dict[str, pd.DataFrame]:
        """Analyze and return results as DataFrames."""
        if not self.results:
            raise ValueError("No results to analyze. Run experiments first.")

        df = pd.DataFrame(self.results)

        # Separate results by experiment type
        result_dfs = {"all": df}

        for exp_type in df['experiment_type'].unique():
            result_dfs[exp_type] = df[df['experiment_type'] == exp_type].copy()

        return result_dfs