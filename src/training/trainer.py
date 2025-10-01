"""Trainer for scaling law experiments."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from torch.cuda.amp import autocast, GradScaler
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import wandb
from tqdm import tqdm
import time
import os
import gc
from typing import Optional, Dict, Any, Tuple
from .utils import (
    get_optimizer, get_scheduler, save_checkpoint,
    load_checkpoint, compute_gradient_norm
)
import pdb
from .memory_utils import setup_memory_optimization, MemoryMonitor, adaptive_batch_size


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
        use_multi_gpu: bool = True,
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
        self.compute_tracker = compute_tracker
        self.use_multi_gpu = use_multi_gpu
        
        # Device and multi-GPU setup
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()
        
        # Log GPU information
        if torch.cuda.is_available():
            print(f"Using device: {self.device}")
            print(f"Number of GPUs available: {self.n_gpu}")
            if self.n_gpu > 0:
                for i in range(self.n_gpu):
                    gpu_name = torch.cuda.get_device_name(i)
                    print(f"  GPU {i}: {gpu_name}")
        
        # Move model to device(s)
        if self.use_multi_gpu and self.n_gpu > 1:
            print(f"Using DataParallel with {self.n_gpu} GPUs")
            self.model = nn.DataParallel(self.model)
            self.model.to(self.device)
            # Store reference to underlying model for accessing methods
            self._base_model = self.model.module
        else:
            self.model.to(self.device)
            self._base_model = self.model
        
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
            # Set environment variable to disable package tracking (fixes metadata error)
            os.environ["WANDB_DISABLE_CODE"] = "true"

            wandb_config = {
                "model_config": self._base_model.config if hasattr(self._base_model, "config") else {},
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "max_steps": max_steps,
                "warmup_steps": warmup_steps,
                "batch_size": train_dataloader.batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "fp16": fp16,
                "num_parameters": self._base_model.get_num_params(),
                "num_gpus": self.n_gpu if self.use_multi_gpu else 1,
                "multi_gpu": self.use_multi_gpu and self.n_gpu > 1,
            }

            try:
                wandb.init(
                    project=wandb_project,
                    name=wandb_run_name or experiment_name,
                    config=wandb_config
                )
                wandb.watch(self.model, log="gradients", log_freq=100)
            except Exception as e:
                print(f"Warning: Failed to initialize wandb: {e}")
                print("Continuing without wandb logging...")
                self.use_wandb = False
        
        # Training state
        self.global_step = 0
        self.best_eval_loss = float("inf")
        self.train_loss_accumulator = 0.0
        self.train_steps_accumulator = 0
        
        # Memory management settings
        self.memory_cleanup_steps = 100  # Clean memory every 100 steps
        self.enable_memory_cleanup = True
        
        # Set up memory optimization
        setup_memory_optimization()
        
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
            
            # Safety check: ensure token IDs are within valid range
            if "input_ids" in batch:
                batch["input_ids"] = torch.clamp(batch["input_ids"], 0, 50256)
            if "labels" in batch:
                batch["labels"] = torch.clamp(batch["labels"], -100, 50256)
            # Forward pass
            if self.fp16:
                with autocast('cuda'):
                    outputs = self.model(**batch)
                    loss = outputs["loss"]
            else:
                outputs = self.model(**batch)
                loss = outputs["loss"]
            # Ensure loss is scalar (important for multi-GPU training)
            if loss.dim() > 0:
                loss = loss.mean()
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            # Backward pass
            if self.fp16:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            # Accumulate loss for logging
            try:
                loss_value = loss.item()
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Invalid loss detected: {loss_value}")
                    continue
                self.train_loss_accumulator += loss_value
                self.train_steps_accumulator += 1
            except RuntimeError as e:
                print(f"Error computing loss: {e}")
                print(f"Loss tensor: {loss}")
                print(f"Input IDs range: {batch['input_ids'].min().item()} - {batch['input_ids'].max().item()}")
                if "labels" in batch:
                    print(f"Labels range: {batch['labels'].min().item()} - {batch['labels'].max().item()}")
                raise
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
                
                # Memory cleanup to prevent fragmentation
                if (self.enable_memory_cleanup and 
                    self.global_step % self.memory_cleanup_steps == 0 and 
                    torch.cuda.is_available()):
                    self._cleanup_memory()
                
                # Logging
                if self.global_step % self.logging_steps == 0:
                    avg_train_loss = self.train_loss_accumulator / self.train_steps_accumulator
                    perplexity = torch.exp(torch.tensor(avg_train_loss))
                    
                    elapsed_time = time.time() - start_time
                    # Avoid division by zero
                    if elapsed_time > 0:
                        samples_per_second = (
                            self.global_step * self.train_dataloader.batch_size *
                            self.gradient_accumulation_steps / elapsed_time
                        )
                    else:
                        samples_per_second = 0.0
                    
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

        # Debug: Check if dataloader has samples
        print(f"Evaluating on dataloader with {len(dataloader)} batches")
        if len(dataloader) == 0:
            print("Warning: Evaluation dataloader is empty!")
            return {
                "loss": float('inf'),
                "perplexity": float('inf'),
                "accuracy": 0.0,
            }

        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        total_correct = 0
        total_samples = 0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Safety check: ensure token IDs are within valid range (for text models)
            if "input_ids" in batch:
                batch["input_ids"] = torch.clamp(batch["input_ids"], 0, 50256)
            if "labels" in batch:
                batch["labels"] = torch.clamp(batch["labels"], -100, 50256)

            try:
                if self.fp16:
                    with autocast('cuda'):
                        outputs = self.model(**batch)
                else:
                    outputs = self.model(**batch)
            except Exception as e:
                print(f"Error in model forward pass: {e}")
                print(f"Batch keys: {list(batch.keys())}")
                for key, value in batch.items():
                    print(f"  {key}: shape = {value.shape}")
                raise

            loss = outputs["loss"]

            # Ensure loss is scalar (important for multi-GPU training)
            if loss.dim() > 0:
                loss = loss.mean()

            # Calculate tokens/samples based on model type
            if "input_ids" in batch:
                # Text model
                batch_size, seq_length = batch["input_ids"].shape
                num_tokens = batch_size * seq_length
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
            else:
                # Image model - use number of samples
                if "x" in batch:
                    batch_size = batch["x"].shape[0]
                elif "image" in batch:
                    batch_size = batch["image"].shape[0]
                else:
                    batch_size = 1
                    print(f"Warning: Could not determine batch size for batch with keys: {list(batch.keys())}")

                total_loss += loss.item() * batch_size
                total_samples += batch_size

                # Calculate accuracy for classification
                if "accuracy" in outputs:
                    accuracy_tensor = outputs["accuracy"]
                    # Handle both scalar tensors and tensors from DataParallel
                    if accuracy_tensor.dim() == 0:
                        # Scalar tensor
                        accuracy_value = accuracy_tensor.item()
                    else:
                        # Tensor with multiple elements (e.g., from DataParallel)
                        accuracy_value = accuracy_tensor.mean().item()
                    total_correct += accuracy_value * batch_size
                elif "logits" in outputs and "labels" in batch:
                    # Calculate accuracy manually
                    logits = outputs["logits"]
                    labels = batch["labels"]

                    # Handle DataParallel outputs which might be lists or multi-dimensional
                    if isinstance(logits, (list, tuple)):
                        # DataParallel returns list of tensors
                        all_predictions = []
                        all_labels = []
                        for i, logit_tensor in enumerate(logits):
                            all_predictions.append(torch.argmax(logit_tensor, dim=-1))
                            all_labels.append(labels[i] if isinstance(labels, (list, tuple)) else labels)
                        predictions = torch.cat(all_predictions)
                        labels_concat = torch.cat(all_labels) if isinstance(labels, (list, tuple)) else labels
                    else:
                        predictions = torch.argmax(logits, dim=-1)
                        labels_concat = labels

                    correct = (predictions == labels_concat).sum().item()
                    total_correct += correct

        # Calculate final metrics based on model type
        if total_tokens > 0:
            # Text model metrics
            avg_loss = total_loss / total_tokens
            perplexity = torch.exp(torch.tensor(avg_loss))
            print(f"Evaluation complete: {total_tokens} tokens processed, avg_loss = {avg_loss:.4f}")
            return {
                "loss": avg_loss,
                "perplexity": perplexity.item() if isinstance(perplexity, torch.Tensor) else perplexity,
            }
        elif total_samples > 0:
            # Image model metrics
            avg_loss = total_loss / total_samples
            accuracy = total_correct / total_samples if total_samples > 0 else 0.0
            print(f"Evaluation complete: {total_samples} samples processed, avg_loss = {avg_loss:.4f}, accuracy = {accuracy:.4f}")
            return {
                "loss": avg_loss,
                "accuracy": accuracy,
            }
        else:
            print("Warning: No tokens or samples processed during evaluation. Using default loss.")
            return {
                "loss": float('inf'),
                "perplexity": float('inf'),
                "accuracy": 0.0,
            }
    
    def _cleanup_memory(self):
        """Clean up GPU memory to prevent fragmentation and OOM errors."""
        if torch.cuda.is_available():
            # Clear cache
            torch.cuda.empty_cache()
            # Force garbage collection
            gc.collect()
            # Synchronize to ensure cleanup is complete
            torch.cuda.synchronize()
    
    def _get_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory usage information."""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            memory_reserved = torch.cuda.memory_reserved() / (1024**3)   # GB
            memory_max_allocated = torch.cuda.max_memory_allocated() / (1024**3)  # GB
            return {
                "memory_allocated_gb": memory_allocated,
                "memory_reserved_gb": memory_reserved,
                "memory_max_allocated_gb": memory_max_allocated,
                "memory_utilization": memory_allocated / memory_reserved if memory_reserved > 0 else 0.0
            }
        return {}
    
    def enable_memory_efficient_training(self):
        """Enable additional memory optimizations for large models."""
        self.enable_memory_cleanup = True
        self.memory_cleanup_steps = 50  # More frequent cleanup
        
        # Enable gradient checkpointing if available
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        elif hasattr(self._base_model, 'gradient_checkpointing_enable'):
            self._base_model.gradient_checkpointing_enable()
        
        # Reduce precision for memory efficiency
        if not self.fp16:
            print("Warning: Consider enabling fp16 for better memory efficiency")
        
        print("Memory efficient training enabled") 