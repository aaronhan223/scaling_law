"""Speech trainer with WER evaluation for scaling law experiments."""

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
import jiwer


class SpeechTrainer:
    """Trainer for speech recognition models with WER evaluation."""

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
        experiment_name: str = "speech_scaling_law",
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

        # Get processor from dataset for WER computation
        self.processor = self._get_processor_from_dataloader()

        # Initialize wandb
        if use_wandb and wandb_project:
            wandb_config = {
                "model_config": self._get_model_config_dict(),
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

            wandb.init(
                project=wandb_project,
                name=wandb_run_name or experiment_name,
                config=wandb_config
            )
            wandb.watch(self.model, log="gradients", log_freq=100)

        # Training state
        self.global_step = 0
        self.best_eval_loss = float("inf")
        self.best_wer = float("inf")
        self.train_loss_accumulator = 0.0
        self.train_steps_accumulator = 0

        # Memory management
        self.memory_cleanup_steps = 100
        self.enable_memory_cleanup = True

        # Set up memory optimization
        setup_memory_optimization()

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def _get_model_config_dict(self):
        """Extract model configuration as a dictionary."""
        try:
            # Just return the config directly like the regular trainer does
            # WandB can handle different object types
            if hasattr(self._base_model, "config"):
                return self._base_model.config
            else:
                # Return basic model info if no config
                return {
                    "model_type": type(self._base_model).__name__,
                    "num_parameters": getattr(self._base_model, 'get_num_params', lambda: 0)()
                }
        except Exception as e:
            print(f"Warning: Could not extract model config: {e}")
            return {}

    def _get_processor_from_dataloader(self):
        """Extract processor from dataset for WER computation."""
        try:
            # Try to get processor from various possible dataset structures
            dataset = self.train_dataloader.dataset

            # Handle ConcatDataset
            if hasattr(dataset, 'datasets'):
                base_dataset = dataset.datasets[0]
            else:
                base_dataset = dataset

            # Handle transformed datasets
            if hasattr(base_dataset, 'base_dataset'):
                base_dataset = base_dataset.base_dataset
                if hasattr(base_dataset, 'datasets'):
                    base_dataset = base_dataset.datasets[0]

            # Get processor
            if hasattr(base_dataset, 'processor'):
                return base_dataset.processor
            else:
                print("Warning: Could not find processor in dataset. WER evaluation may not work correctly.")
                return None

        except Exception as e:
            print(f"Warning: Error getting processor from dataset: {e}")
            return None

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
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

            # Forward pass
            if self.fp16:
                with autocast('cuda'):
                    outputs = self.model(
                        input_values=batch['input_values'],
                        attention_mask=batch.get('attention_mask'),
                        labels=batch.get('labels')
                    )
                    loss = outputs["loss"] if "loss" in outputs else outputs.get("loss", None)
            else:
                outputs = self.model(
                    input_values=batch['input_values'],
                    attention_mask=batch.get('attention_mask'),
                    labels=batch.get('labels')
                )
                loss = outputs["loss"] if "loss" in outputs else outputs.get("loss", None)

            if loss is None:
                print("Warning: No loss computed, skipping step")
                continue

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

                    elapsed_time = time.time() - start_time
                    if elapsed_time > 0:
                        samples_per_second = (
                            self.global_step * self.train_dataloader.batch_size *
                            self.gradient_accumulation_steps / elapsed_time
                        )
                    else:
                        samples_per_second = 0.0

                    log_dict = {
                        "train/loss": avg_train_loss,
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

                    # Save best model (based on WER if available, otherwise loss)
                    if "wer" in eval_metrics:
                        if eval_metrics["wer"] < self.best_wer:
                            self.best_wer = eval_metrics["wer"]
                            save_checkpoint(
                                self.model,
                                self.optimizer,
                                self.scheduler,
                                self.global_step,
                                eval_metrics["loss"],
                                os.path.join(self.output_dir, "best_model"),
                            )
                    else:
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
            "best_wer": self.best_wer,
            "total_steps": self.global_step,
            "total_time": time.time() - start_time,
        }

    @torch.no_grad()
    def evaluate(self, dataloader: Optional[DataLoader] = None) -> Dict[str, float]:
        """Evaluate model on given dataloader with WER computation."""
        if dataloader is None:
            dataloader = self.eval_dataloader

        print(f"Evaluating on dataloader with {len(dataloader)} batches")
        if len(dataloader) == 0:
            print("Warning: Evaluation dataloader is empty!")
            return {
                "loss": float('inf'),
                "wer": float('inf'),
            }

        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        predictions = []
        references = []

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

            try:
                if self.fp16:
                    with autocast('cuda'):
                        outputs = self.model(
                            input_values=batch['input_values'],
                            attention_mask=batch.get('attention_mask'),
                            labels=batch.get('labels')
                        )
                else:
                    outputs = self.model(
                        input_values=batch['input_values'],
                        attention_mask=batch.get('attention_mask'),
                        labels=batch.get('labels')
                    )
            except Exception as e:
                print(f"Error in model forward pass: {e}")
                print(f"Batch keys: {list(batch.keys())}")
                for key, value in batch.items():
                    if torch.is_tensor(value):
                        print(f"  {key}: shape = {value.shape}")
                raise

            # Accumulate loss
            if "loss" in outputs and outputs["loss"] is not None:
                loss = outputs["loss"]
                if loss.dim() > 0:
                    loss = loss.mean()

                batch_size = batch['input_values'].shape[0]
                total_loss += loss.item() * batch_size
                total_samples += batch_size

            # Collect predictions and references for WER computation
            if "logits" in outputs and self.processor is not None:
                logits = outputs["logits"]

                # Get predicted token IDs
                predicted_ids = torch.argmax(logits, dim=-1)

                # Decode predictions
                try:
                    batch_predictions = self.processor.batch_decode(predicted_ids.cpu().numpy())
                    predictions.extend(batch_predictions)

                    # Get reference transcripts
                    if 'transcripts' in batch:
                        references.extend(batch['transcripts'])
                    elif 'labels' in batch:
                        # Decode labels if transcripts not available
                        labels = batch['labels'].cpu().numpy()
                        # Remove -100 tokens (padding)
                        labels[labels == -100] = self.processor.tokenizer.pad_token_id
                        batch_references = self.processor.batch_decode(labels)
                        references.extend(batch_references)
                except Exception as e:
                    print(f"Warning: Error decoding predictions: {e}")
                    # Skip WER computation for this batch
                    continue

        # Calculate metrics
        metrics = {}

        if total_samples > 0:
            avg_loss = total_loss / total_samples
            metrics["loss"] = avg_loss
            print(f"Evaluation complete: {total_samples} samples processed, avg_loss = {avg_loss:.4f}")
        else:
            metrics["loss"] = float('inf')
            print("Warning: No samples processed during evaluation")

        # Calculate WER if we have predictions and references
        if len(predictions) > 0 and len(references) > 0 and len(predictions) == len(references):
            try:
                # Clean up predictions and references
                clean_predictions = [pred.strip().upper() for pred in predictions]
                clean_references = [ref.strip().upper() for ref in references]

                # Compute WER using jiwer
                wer = jiwer.wer(clean_references, clean_predictions)
                metrics["wer"] = wer
                print(f"WER: {wer:.4f}")

                # Log some examples
                print("Example predictions vs references:")
                for i in range(min(3, len(clean_predictions))):
                    print(f"  Pred: {clean_predictions[i]}")
                    print(f"  Ref:  {clean_references[i]}")
                    print()

            except Exception as e:
                print(f"Warning: Error computing WER: {e}")
                metrics["wer"] = float('inf')
        else:
            print(f"Warning: Cannot compute WER. Predictions: {len(predictions)}, References: {len(references)}")
            metrics["wer"] = float('inf')

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