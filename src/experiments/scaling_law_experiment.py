"""Main scaling law experiment runner."""

import os
import json
import torch
import wandb
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm

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
        
        # Set seeds
        set_seed(
            self.config.training_config.seed,
            deterministic=self.config.deterministic
        )
        
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
        
        # Create datasets
        if experiment_type == "standard":
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
        else:  # transformed
            # First create base datasets
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
            
            # Create transformed dataloaders
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
            
            # Run on subset of configurations
            selected_model_indices = [0, 2, 4]  # Small, medium, large models
            selected_dataset_indices = [0, 2, 4]  # Small, medium, large datasets
            
            for i in selected_model_indices:
                if i < len(model_configs):
                    model_config = model_configs[i]
                    for j in selected_dataset_indices:
                        if j < len(dataset_sizes):
                            dataset_size = dataset_sizes[j]
                            
                            result = self.run_single_experiment(
                                model_config=model_config,
                                dataset_size=dataset_size,
                                experiment_type="transformed",
                            )
                            self.results.append(result)
                            
                            # Save intermediate results
                            self.save_results()
        
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
        
        # Separate standard and transformed results
        standard_df = df[df['experiment_type'] == 'standard'].copy()
        transformed_df = df[df['experiment_type'] == 'transformed'].copy()
        
        return {
            "all": df,
            "standard": standard_df,
            "transformed": transformed_df,
        } 