"""Main scaling law experiment runner."""

import os
import json
from torch.utils.data import DataLoader
import wandb
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any

from config import ExperimentConfig
from data import WikiTextDataset, create_dataloaders
from data.transforms import create_transformed_dataloaders
from models import create_model
from training import Trainer, set_seed
from .compute_tracker import ComputeTracker


class ScalingLawExperiment:
    """Run scaling law experiments."""
    
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
        
    def _get_or_create_datasets(
        self,
        dataset_size: int,
        cache_key: Optional[str] = None
    ) -> Tuple[WikiTextDataset, WikiTextDataset, WikiTextDataset]:
        """Get datasets from cache or create new ones."""
        if cache_key is None:
            cache_key = f"size_{dataset_size}"
            
        if cache_key not in self.dataset_cache:
            print(f"Creating datasets for {dataset_size:,} tokens...")
            train_dataset = WikiTextDataset(
                split="train",
                tokenizer_name=self.config.data_config.tokenizer_name,
                max_length=self.config.data_config.max_length,
                max_tokens=dataset_size,
                seed=self.config.training_config.seed,
            )
            valid_dataset = WikiTextDataset(
                split="validation",
                tokenizer_name=self.config.data_config.tokenizer_name,
                max_length=self.config.data_config.max_length,
                max_tokens=min(dataset_size // 10, int(1e6)),
                seed=self.config.training_config.seed,
            )
            test_dataset = WikiTextDataset(
                split="test",
                tokenizer_name=self.config.data_config.tokenizer_name,
                max_length=self.config.data_config.max_length,
                max_tokens=min(dataset_size // 10, int(1e6)),
                seed=self.config.training_config.seed,
            )
            self.dataset_cache[cache_key] = (train_dataset, valid_dataset, test_dataset)
        else:
            print(f"Using cached datasets for {dataset_size:,} tokens...")
            
        return self.dataset_cache[cache_key]
        
    def run_single_experiment(
        self,
        model_config: Dict[str, Any],
        dataset_size: int,
        experiment_type: str = "standard",
    ) -> Dict[str, Any]:
        """Run a single training experiment."""
        print(f"\n{'='*60}")
        print(f"Running {experiment_type} experiment:")
        print(f"Model: {model_config['model_name']} ({model_config['num_parameters']:,} params)")
        print(f"Dataset size: {dataset_size:,} tokens")
        print(f"{'='*60}\n")
        
        # Create or get cached datasets
        if experiment_type == "standard" and self.config.use_standard_dataloader_optimization:
            # Use optimized dataloader creation for standard experiments
            train_loader, valid_loader, test_loader = create_dataloaders(
                tokenizer_name=self.config.data_config.tokenizer_name,
                max_length=self.config.data_config.max_length,
                train_batch_size=self.config.data_config.train_batch_size,
                eval_batch_size=self.config.data_config.eval_batch_size,
                max_tokens_train=dataset_size,
                max_tokens_valid=min(dataset_size // 10, int(1e6)),  # 10% or 1M tokens
                num_workers=self.config.data_config.num_workers,
                seed=self.config.training_config.seed,
            )
        else:
            # Get base datasets (from cache if available)
            train_dataset, valid_dataset, test_dataset = self._get_or_create_datasets(dataset_size)
            
            if experiment_type == "standard":
                # Create standard dataloaders from cached datasets
                
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.config.data_config.train_batch_size,
                    shuffle=True,
                    num_workers=self.config.data_config.num_workers,
                    pin_memory=True,
                    persistent_workers=True if self.config.data_config.num_workers > 0 else False,
                )
                
                valid_loader = DataLoader(
                    valid_dataset,
                    batch_size=self.config.data_config.eval_batch_size,
                    shuffle=False,
                    num_workers=self.config.data_config.num_workers,
                    pin_memory=True,
                    persistent_workers=True if self.config.data_config.num_workers > 0 else False,
                )
                
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=self.config.data_config.eval_batch_size,
                    shuffle=False,
                    num_workers=self.config.data_config.num_workers,
                    pin_memory=True,
                    persistent_workers=True if self.config.data_config.num_workers > 0 else False,
                )
            elif experiment_type == "transformed":
                # Create transformed dataloaders (both linear and nonlinear)
                train_loader, valid_loader, test_loader, _, _ = create_transformed_dataloaders(
                    base_train_dataset=train_dataset,
                    base_valid_dataset=valid_dataset,
                    base_test_dataset=test_dataset,
                    vocab_size=self.config.model_config.vocab_size,
                    train_batch_size=self.config.data_config.train_batch_size,
                    eval_batch_size=self.config.data_config.eval_batch_size,
                    num_workers=self.config.data_config.num_workers,
                    linear_transform_dim=self.config.transform_config.linear_transform_dim,
                    linear_seed=self.config.transform_config.linear_transform_seed,
                    nonlinear_type=self.config.transform_config.nonlinear_type,
                    nonlinear_scale=self.config.transform_config.nonlinear_scale,
                    nonlinear_seed=self.config.transform_config.nonlinear_seed,
                )
            elif experiment_type == "linear":
                # Create dataloaders with only linear transformation
                train_loader, valid_loader, test_loader, _, _ = create_transformed_dataloaders(
                    base_train_dataset=train_dataset,
                    base_valid_dataset=valid_dataset,
                    base_test_dataset=test_dataset,
                    vocab_size=self.config.model_config.vocab_size,
                    train_batch_size=self.config.data_config.train_batch_size,
                    eval_batch_size=self.config.data_config.eval_batch_size,
                    num_workers=self.config.data_config.num_workers,
                    linear_transform_dim=self.config.transform_config.linear_transform_dim,
                    linear_seed=self.config.transform_config.linear_transform_seed,
                    nonlinear_type=self.config.transform_config.nonlinear_type,
                    nonlinear_scale=self.config.transform_config.nonlinear_scale,
                    nonlinear_seed=self.config.transform_config.nonlinear_seed,
                    apply_linear=True,
                    apply_nonlinear=False,
                )
            elif experiment_type == "nonlinear":
                # Create dataloaders with only nonlinear transformation
                train_loader, valid_loader, test_loader, _, _ = create_transformed_dataloaders(
                    base_train_dataset=train_dataset,
                    base_valid_dataset=valid_dataset,
                    base_test_dataset=test_dataset,
                    vocab_size=self.config.model_config.vocab_size,
                    train_batch_size=self.config.data_config.train_batch_size,
                    eval_batch_size=self.config.data_config.eval_batch_size,
                    num_workers=self.config.data_config.num_workers,
                    linear_transform_dim=self.config.transform_config.linear_transform_dim,
                    linear_seed=self.config.transform_config.linear_transform_seed,
                    nonlinear_type=self.config.transform_config.nonlinear_type,
                    nonlinear_scale=self.config.transform_config.nonlinear_scale,
                    nonlinear_seed=self.config.transform_config.nonlinear_seed,
                    apply_linear=False,
                    apply_nonlinear=True,
                )
            else:
                raise ValueError(f"Unknown experiment type: {experiment_type}")
        
        # Create model
        model = create_model(model_config)
        print(f"Created model with {model.get_num_params():,} parameters")
        
        # Calculate training steps
        max_steps = self.config.training_config.get_max_steps_for_tokens(
            dataset_size,
            self.config.data_config.train_batch_size,
            self.config.data_config.max_length,
        )
        
        # Create compute tracker
        compute_tracker = None
        if self.config.track_compute:
            compute_tracker = ComputeTracker(
                model_params=model.get_num_params(),
                batch_size=self.config.data_config.train_batch_size,
                sequence_length=self.config.data_config.max_length,
                gradient_accumulation_steps=self.config.training_config.gradient_accumulation_steps,
            )
        
        # Create output directory
        output_dir = self.config.get_output_dir(
            experiment_type,
            model_config['model_name'],
            dataset_size
        )
        
        # Create trainer
        wandb_run_name = f"{experiment_type}_{model_config['model_name']}_data{dataset_size}"
        trainer = Trainer(
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
            "final_perplexity": train_results["final_eval_metrics"]["perplexity"],
            "test_loss": train_results["final_eval_metrics"].get("test_loss", None),
            "test_perplexity": train_results["final_eval_metrics"].get("test_perplexity", None),
            "total_steps": train_results["total_steps"],
            "total_time": train_results["total_time"],
            **compute_stats,
        }
        
        # Close wandb run
        if self.config.use_wandb:
            wandb.finish()
        
        return result
    
    def run_all_experiments(self) -> List[Dict[str, Any]]:
        """Run all scaling law experiments."""
        model_configs = self.config.model_config.get_model_size_configs()
        dataset_sizes = self.config.data_config.dataset_sizes
        
        # Pre-cache datasets if running multiple experiment types
        num_experiments = sum([self.config.run_standard, self.config.run_transformed, 
                             self.config.run_linear_transform, self.config.run_nonlinear_transform])
        if num_experiments > 1:
            print("\n" + "="*80)
            print("PRE-CACHING DATASETS FOR EFFICIENCY")
            print("="*80)
            for dataset_size in dataset_sizes:
                self._get_or_create_datasets(dataset_size)
        
        # Run standard experiments
        if self.config.run_standard:
            print("\n" + "="*80)
            print("RUNNING STANDARD EXPERIMENTS")
            print("="*80)
            
            for model_config in model_configs:
                for dataset_size in dataset_sizes:
                    # Skip very large model/dataset combinations to save compute
                    if model_config['num_parameters'] > 1e8 and dataset_size > 1e8:
                        print(f"Skipping large combination: {model_config['model_name']} with {dataset_size} tokens")
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
        if self.config.run_transformed:
            print("\n" + "="*80)
            print("RUNNING TRANSFORMED EXPERIMENTS")
            print("="*80)
            
            for model_config in model_configs:
                for dataset_size in dataset_sizes:
                    # Skip very large model/dataset combinations to save compute
                    if model_config['num_parameters'] > 1e8 and dataset_size > 1e8:
                        print(f"Skipping large combination: {model_config['model_name']} with {dataset_size} tokens")
                        continue
                    
                    result = self.run_single_experiment(
                        model_config=model_config,
                        dataset_size=dataset_size,
                        experiment_type="transformed",
                    )
                    self.results.append(result)
                    
                    # Save intermediate results
                    self.save_results()
        
        # Run linear transform experiments
        if self.config.run_linear_transform:
            print("\n" + "="*80)
            print("RUNNING LINEAR TRANSFORM EXPERIMENTS")
            print("="*80)
            
            for model_config in model_configs:
                for dataset_size in dataset_sizes:
                    # Skip very large model/dataset combinations to save compute
                    if model_config['num_parameters'] > 1e8 and dataset_size > 1e8:
                        print(f"Skipping large combination: {model_config['model_name']} with {dataset_size} tokens")
                        continue
                    
                    result = self.run_single_experiment(
                        model_config=model_config,
                        dataset_size=dataset_size,
                        experiment_type="linear",
                    )
                    self.results.append(result)
                    
                    # Save intermediate results
                    self.save_results()
        
        # Run nonlinear transform experiments
        if self.config.run_nonlinear_transform:
            print("\n" + "="*80)
            print("RUNNING NONLINEAR TRANSFORM EXPERIMENTS")
            print("="*80)
            
            for model_config in model_configs:
                for dataset_size in dataset_sizes:
                    # Skip very large model/dataset combinations to save compute
                    if model_config['num_parameters'] > 1e8 and dataset_size > 1e8:
                        print(f"Skipping large combination: {model_config['model_name']} with {dataset_size} tokens")
                        continue
                    
                    result = self.run_single_experiment(
                        model_config=model_config,
                        dataset_size=dataset_size,
                        experiment_type="nonlinear",
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
        results_file = os.path.join(results_dir, "scaling_law_results.json")
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)
        
        # Save as CSV
        if self.results:
            df = pd.DataFrame(self.results)
            csv_file = os.path.join(results_dir, "scaling_law_results.csv")
            df.to_csv(csv_file, index=False)
            
        print(f"Results saved to {results_dir}")
    
    def analyze_results(self) -> Dict[str, pd.DataFrame]:
        """Analyze and return results as DataFrames."""
        if not self.results:
            raise ValueError("No results to analyze. Run experiments first.")
        
        df = pd.DataFrame(self.results)
        
        # Separate results by experiment type
        standard_df = df[df['experiment_type'] == 'standard'].copy() if 'standard' in df['experiment_type'].values else pd.DataFrame()
        transformed_df = df[df['experiment_type'] == 'transformed'].copy() if 'transformed' in df['experiment_type'].values else pd.DataFrame()
        linear_df = df[df['experiment_type'] == 'linear'].copy() if 'linear' in df['experiment_type'].values else pd.DataFrame()
        nonlinear_df = df[df['experiment_type'] == 'nonlinear'].copy() if 'nonlinear' in df['experiment_type'].values else pd.DataFrame()
        
        return {
            "all": df,
            "standard": standard_df,
            "transformed": transformed_df,
            "linear": linear_df,
            "nonlinear": nonlinear_df,
        } 