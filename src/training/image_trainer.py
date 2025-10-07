"""Image trainer with accuracy evaluation for ImageNet scaling law experiments."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import wandb
from tqdm import tqdm
import time
import os
import gc
from typing import Optional, Dict, Any, Tuple, List
from .utils import (
    get_optimizer, get_scheduler, save_checkpoint,
    load_checkpoint, compute_gradient_norm
)
from .memory_utils import setup_memory_optimization


class ImageTrainer:
    """Trainer for image classification models (Vision Transformers) with accuracy evaluation."""

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        test_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.05,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
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
        experiment_name: str = "image_scaling_law",
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
            # Set environment variable to disable package tracking
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
        self.best_eval_accuracy = 0.0
        self.train_loss_accumulator = 0.0
        self.train_accuracy_accumulator = 0.0
        self.train_steps_accumulator = 0

        # Memory management settings
        self.memory_cleanup_steps = 100
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

            # Validate labels are in valid range
            if "labels" in batch and step == 0:
                labels = batch["labels"]
                print(f"First batch labels - min: {labels.min().item()}, max: {labels.max().item()}")
                if hasattr(self._base_model, 'num_classes'):
                    print(f"Model expects {self._base_model.num_classes} classes")
                    if labels.max().item() >= self._base_model.num_classes:
                        print(f"ERROR: Label {labels.max().item()} >= num_classes {self._base_model.num_classes}")
                elif hasattr(self._base_model, 'head'):
                    num_classes = self._base_model.head.out_features
                    print(f"Model head has {num_classes} output classes")
                    if labels.max().item() >= num_classes:
                        print(f"ERROR: Label {labels.max().item()} >= num_classes {num_classes}")

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

            # Accumulate metrics for logging
            try:
                loss_value = loss.item()
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Invalid loss detected: {loss_value}")
                    continue
                self.train_loss_accumulator += loss_value

                # Accumulate accuracy if available
                if "accuracy" in outputs:
                    accuracy = outputs["accuracy"]
                    if accuracy.dim() > 0:
                        accuracy = accuracy.mean()
                    self.train_accuracy_accumulator += accuracy.item()

                self.train_steps_accumulator += 1
            except RuntimeError as e:
                print(f"Error computing loss: {e}")
                print(f"Loss tensor: {loss}")
                if "x" in batch:
                    print(f"Image batch shape: {batch['x'].shape}")
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

                # Memory cleanup
                if (self.enable_memory_cleanup and
                    self.global_step % self.memory_cleanup_steps == 0 and
                    torch.cuda.is_available()):
                    self._cleanup_memory()

                # Logging
                if self.global_step % self.logging_steps == 0:
                    avg_train_loss = self.train_loss_accumulator / self.train_steps_accumulator
                    avg_train_accuracy = self.train_accuracy_accumulator / self.train_steps_accumulator

                    elapsed_time = time.time() - start_time
                    samples_per_second = (
                        self.global_step * self.train_dataloader.batch_size *
                        self.gradient_accumulation_steps / elapsed_time
                    ) if elapsed_time > 0 else 0.0

                    log_dict = {
                        "train/loss": avg_train_loss,
                        "train/accuracy": avg_train_accuracy,
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
                    self.train_accuracy_accumulator = 0.0
                    self.train_steps_accumulator = 0

                # Evaluation
                if self.global_step % self.eval_steps == 0:
                    eval_metrics = self.evaluate()

                    if self.use_wandb:
                        wandb.log({
                            f"eval/{k}": v for k, v in eval_metrics.items()
                        }, step=self.global_step)

                    # Save best model based on accuracy
                    if eval_metrics.get("accuracy", 0.0) > self.best_eval_accuracy:
                        self.best_eval_accuracy = eval_metrics["accuracy"]
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
            "best_eval_accuracy": self.best_eval_accuracy,
            "total_steps": self.global_step,
            "total_time": time.time() - start_time,
        }

    @torch.no_grad()
    def evaluate(self, dataloader: Optional[DataLoader] = None) -> Dict[str, float]:
        """Evaluate model on given dataloader."""
        if dataloader is None:
            dataloader = self.eval_dataloader

        # Check if dataloader has samples
        print(f"Evaluating on dataloader with {len(dataloader)} batches")
        if len(dataloader) == 0:
            print("Warning: Evaluation dataloader is empty!")
            return {
                "loss": float('inf'),
                "accuracy": 0.0,
                "top5_accuracy": 0.0,
            }

        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_top5_correct = 0
        total_samples = 0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            batch = {k: v.to(self.device) for k, v in batch.items()}

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

            # Get batch size
            if "x" in batch:
                batch_size = batch["x"].shape[0]
            elif "images" in batch:
                batch_size = batch["images"].shape[0]
            elif "pixel_values" in batch:
                batch_size = batch["pixel_values"].shape[0]
            else:
                batch_size = batch["labels"].shape[0] if "labels" in batch else 1
                print(f"Warning: Could not determine batch size from keys: {list(batch.keys())}")

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Calculate top-1 accuracy
            if "accuracy" in outputs:
                accuracy_tensor = outputs["accuracy"]
                # Handle both scalar tensors and tensors from DataParallel
                if accuracy_tensor.dim() == 0:
                    accuracy_value = accuracy_tensor.item()
                else:
                    accuracy_value = accuracy_tensor.mean().item()
                total_correct += accuracy_value * batch_size
            elif "logits" in outputs and "labels" in batch:
                logits = outputs["logits"]
                labels = batch["labels"]

                # Calculate top-1 accuracy
                predictions = torch.argmax(logits, dim=-1)
                correct = (predictions == labels).sum().item()
                total_correct += correct

                # Calculate top-5 accuracy for ImageNet
                if logits.shape[-1] >= 5:
                    _, top5_predictions = torch.topk(logits, k=5, dim=-1)
                    # Check if true label is in top 5
                    labels_expanded = labels.unsqueeze(1).expand_as(top5_predictions)
                    top5_correct = (top5_predictions == labels_expanded).any(dim=1).sum().item()
                    total_top5_correct += top5_correct

        # Calculate final metrics
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        top5_accuracy = total_top5_correct / total_samples if total_samples > 0 else 0.0

        print(f"Evaluation complete: {total_samples} samples processed")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Top-1 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        if total_top5_correct > 0:
            print(f"  Top-5 Accuracy: {top5_accuracy:.4f} ({top5_accuracy*100:.2f}%)")

        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
        }

        if total_top5_correct > 0:
            metrics["top5_accuracy"] = top5_accuracy

        return metrics

    def _cleanup_memory(self):
        """Clean up GPU memory to prevent fragmentation and OOM errors."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
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
        self.memory_cleanup_steps = 50

        # Enable gradient checkpointing if available
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        elif hasattr(self._base_model, 'gradient_checkpointing_enable'):
            self._base_model.gradient_checkpointing_enable()

        if not self.fp16:
            print("Warning: Consider enabling fp16 for better memory efficiency")

        print("Memory efficient training enabled")

    def log_model_predictions(self, num_samples: int = 10):
        """Log sample predictions for visualization (useful for debugging)."""
        if not self.use_wandb:
            return

        self.model.eval()
        sample_iterator = iter(self.eval_dataloader)

        try:
            batch = next(sample_iterator)
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = self.model(**batch)

            logits = outputs["logits"][:num_samples]
            labels = batch["labels"][:num_samples]
            predictions = torch.argmax(logits, dim=-1)

            # Log predictions vs ground truth
            table_data = []
            for i in range(min(num_samples, len(labels))):
                table_data.append([
                    i,
                    labels[i].item(),
                    predictions[i].item(),
                    "✓" if labels[i] == predictions[i] else "✗"
                ])

            wandb.log({
                "predictions_table": wandb.Table(
                    data=table_data,
                    columns=["Sample", "True Label", "Predicted Label", "Correct"]
                )
            })
        except Exception as e:
            print(f"Warning: Could not log model predictions: {e}")
        finally:
            self.model.train()
