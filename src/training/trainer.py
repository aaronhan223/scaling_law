"""Trainer for scaling law experiments."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import wandb
from tqdm import tqdm
import time
import os
from typing import Optional, Dict, Any, Tuple
from .utils import (
    get_optimizer, get_scheduler, save_checkpoint,
    load_checkpoint, compute_gradient_norm
)


class Trainer:
    """Trainer for language models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        test_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.01,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.95,
        adam_epsilon: float = 1e-8,
        max_grad_norm: float = 1.0,
        warmup_steps: int = 1000,
        max_steps: int = 10000,
        eval_steps: int = 500,
        save_steps: int = 1000,
        logging_steps: int = 100,
        gradient_accumulation_steps: int = 1,
        fp16: bool = True,
        output_dir: str = "./outputs",
        experiment_name: str = "scaling_law",
        use_wandb: bool = True,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        device: str = "cuda",
        compute_tracker: Optional[Any] = None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon
        self.max_grad_norm = max_grad_norm
        
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.logging_steps = logging_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        self.fp16 = fp16
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.use_wandb = use_wandb
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.compute_tracker = compute_tracker
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = get_optimizer(
            self.model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_epsilon=adam_epsilon,
        )
        
        self.scheduler = get_scheduler(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
        
        # Mixed precision training
        self.scaler = GradScaler() if fp16 else None
        
        # Initialize wandb
        if use_wandb and wandb_project:
            wandb.init(
                project=wandb_project,
                name=wandb_run_name or experiment_name,
                config={
                    "model_config": self.model.config if hasattr(self.model, "config") else {},
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "max_steps": max_steps,
                    "warmup_steps": warmup_steps,
                    "batch_size": train_dataloader.batch_size,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "fp16": fp16,
                    "num_parameters": self.model.get_num_params(),
                }
            )
            wandb.watch(self.model, log="gradients", log_freq=100)
        
        # Training state
        self.global_step = 0
        self.best_eval_loss = float("inf")
        self.train_loss_accumulator = 0.0
        self.train_steps_accumulator = 0
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def train(self) -> Dict[str, Any]:
        """Run training loop."""
        self.model.train()
        train_iterator = iter(self.train_dataloader)
        
        progress_bar = tqdm(range(self.max_steps), desc="Training")
        start_time = time.time()
        
        if self.compute_tracker:
            self.compute_tracker.start()
        
        for step in range(self.max_steps):
            # Get batch
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(self.train_dataloader)
                batch = next(train_iterator)
            
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            if self.fp16:
                with autocast():
                    outputs = self.model(**batch)
                    loss = outputs["loss"]
            else:
                outputs = self.model(**batch)
                loss = outputs["loss"]
            
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.fp16:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Accumulate loss for logging
            self.train_loss_accumulator += loss.item()
            self.train_steps_accumulator += 1
            
            # Update weights
            if (step + 1) % self.gradient_accumulation_steps == 0:
                if self.fp16:
                    self.scaler.unscale_(self.optimizer)
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                
                if self.fp16:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Update compute tracker
                if self.compute_tracker:
                    self.compute_tracker.step()
                
                # Logging
                if self.global_step % self.logging_steps == 0:
                    avg_train_loss = self.train_loss_accumulator / self.train_steps_accumulator
                    perplexity = torch.exp(torch.tensor(avg_train_loss))
                    
                    elapsed_time = time.time() - start_time
                    samples_per_second = (
                        self.global_step * self.train_dataloader.batch_size *
                        self.gradient_accumulation_steps / elapsed_time
                    )
                    
                    log_dict = {
                        "train/loss": avg_train_loss,
                        "train/perplexity": perplexity.item(),
                        "train/learning_rate": self.scheduler.get_last_lr()[0],
                        "train/grad_norm": grad_norm.item(),
                        "train/samples_per_second": samples_per_second,
                        "train/global_step": self.global_step,
                    }
                    
                    if self.compute_tracker:
                        compute_stats = self.compute_tracker.get_stats()
                        log_dict.update({
                            f"compute/{k}": v for k, v in compute_stats.items()
                        })
                    
                    if self.use_wandb:
                        wandb.log(log_dict, step=self.global_step)
                    
                    self.train_loss_accumulator = 0.0
                    self.train_steps_accumulator = 0
                
                # Evaluation
                if self.global_step % self.eval_steps == 0:
                    eval_metrics = self.evaluate()
                    
                    if self.use_wandb:
                        wandb.log({
                            f"eval/{k}": v for k, v in eval_metrics.items()
                        }, step=self.global_step)
                    
                    # Save best model
                    if eval_metrics["loss"] < self.best_eval_loss:
                        self.best_eval_loss = eval_metrics["loss"]
                        save_checkpoint(
                            self.model,
                            self.optimizer,
                            self.scheduler,
                            self.global_step,
                            self.best_eval_loss,
                            os.path.join(self.output_dir, "best_model"),
                        )
                    
                    self.model.train()
                
                # Save checkpoint
                if self.global_step % self.save_steps == 0:
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        self.global_step,
                        self.best_eval_loss,
                        self.output_dir,
                    )
            
            progress_bar.update(1)
            progress_bar.set_postfix({
                "loss": f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        progress_bar.close()
        
        if self.compute_tracker:
            self.compute_tracker.stop()
        
        # Final evaluation
        print("Running final evaluation...")
        final_eval_metrics = self.evaluate()
        
        if self.test_dataloader is not None:
            print("Running test evaluation...")
            test_metrics = self.evaluate(self.test_dataloader)
            final_eval_metrics.update({f"test_{k}": v for k, v in test_metrics.items()})
        
        if self.use_wandb:
            wandb.log({
                f"final/{k}": v for k, v in final_eval_metrics.items()
            })
        
        # Save final model
        save_checkpoint(
            self.model,
            self.optimizer,
            self.scheduler,
            self.global_step,
            self.best_eval_loss,
            os.path.join(self.output_dir, "final_model"),
        )
        
        return {
            "final_eval_metrics": final_eval_metrics,
            "best_eval_loss": self.best_eval_loss,
            "total_steps": self.global_step,
            "total_time": time.time() - start_time,
        }
    
    @torch.no_grad()
    def evaluate(self, dataloader: Optional[DataLoader] = None) -> Dict[str, float]:
        """Evaluate model on given dataloader."""
        if dataloader is None:
            dataloader = self.eval_dataloader
        
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            if self.fp16:
                with autocast():
                    outputs = self.model(**batch)
            else:
                outputs = self.model(**batch)
            
            loss = outputs["loss"]
            batch_size, seq_length = batch["input_ids"].shape
            num_tokens = batch_size * seq_length
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss))
        
        return {
            "loss": avg_loss,
            "perplexity": perplexity.item(),
        } 