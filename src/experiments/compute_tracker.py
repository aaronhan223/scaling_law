"""Compute tracking for scaling law experiments."""

import time
from typing import Dict, Any


class ComputeTracker:
    """Track compute usage during training."""
    
    def __init__(
        self,
        model_params: int,
        batch_size: int,
        sequence_length: int,
        gradient_accumulation_steps: int = 1,
    ):
        """
        Initialize compute tracker.
        
        Args:
            model_params: Number of model parameters
            batch_size: Training batch size
            sequence_length: Sequence length
            gradient_accumulation_steps: Gradient accumulation steps
        """
        self.model_params = model_params
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Timing
        self.start_time = None
        self.total_time = 0.0
        self.steps = 0
        
        # FLOPs calculation constants
        # Approximate FLOPs per token for transformer:
        # Forward pass: ~6N (where N is number of parameters)
        # Backward pass: ~12N (approximately 2x forward)
        # Total per token: ~18N
        self.flops_per_token = 18 * model_params
        
    def start(self):
        """Start tracking compute."""
        self.start_time = time.time()
        
    def stop(self):
        """Stop tracking compute."""
        if self.start_time is not None:
            self.total_time += time.time() - self.start_time
            self.start_time = None
            
    def step(self):
        """Increment step counter."""
        self.steps += 1
        
    def get_total_flops(self) -> float:
        """Calculate total FLOPs."""
        tokens_per_step = (
            self.batch_size * self.sequence_length * self.gradient_accumulation_steps
        )
        total_tokens = self.steps * tokens_per_step
        return total_tokens * self.flops_per_token
    
    def get_total_compute_petaflop_days(self) -> float:
        """Calculate total compute in PetaFLOP-days."""
        total_flops = self.get_total_flops()
        # Convert to PetaFLOPs
        petaflops = total_flops / 1e15
        # Convert to PetaFLOP-days (assuming we could sustain 1 PetaFLOP/s)
        petaflop_seconds = petaflops
        petaflop_days = petaflop_seconds / (24 * 3600)
        return petaflop_days
    
    def get_stats(self) -> Dict[str, Any]:
        """Get compute statistics."""
        total_flops = self.get_total_flops()
        petaflop_days = self.get_total_compute_petaflop_days()
        
        # Calculate effective FLOPS/s based on actual time
        if self.total_time > 0:
            flops_per_second = total_flops / self.total_time
            tflops_per_second = flops_per_second / 1e12
        else:
            tflops_per_second = 0.0
        
        return {
            "total_flops": total_flops,
            "petaflop_days": petaflop_days,
            "total_time_seconds": self.total_time,
            "tflops_per_second": tflops_per_second,
            "steps": self.steps,
            "model_params": self.model_params,
        }


def estimate_compute_for_model(
    model_params: int,
    num_tokens: int,
) -> Dict[str, float]:
    """
    Estimate compute requirements for training a model.
    
    Args:
        model_params: Number of model parameters
        num_tokens: Number of training tokens
        
    Returns:
        Dictionary with compute estimates
    """
    # Using approximation from the paper
    # C ≈ 6 * N * D (where C is compute, N is parameters, D is dataset size)
    flops = 6 * model_params * num_tokens
    
    petaflops = flops / 1e15
    petaflop_days = petaflops / (24 * 3600)
    
    return {
        "total_flops": flops,
        "petaflops": petaflops,
        "petaflop_days": petaflop_days,
    } 