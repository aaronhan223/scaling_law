"""Memory utilities for monitoring and optimizing GPU memory usage."""

import torch
import gc
import os
from typing import Dict, Optional
import logging


def get_memory_info() -> Dict[str, float]:
    """Get current GPU memory usage information."""
    if not torch.cuda.is_available():
        return {}
    
    memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
    memory_reserved = torch.cuda.memory_reserved() / (1024**3)   # GB
    memory_max_allocated = torch.cuda.max_memory_allocated() / (1024**3)  # GB
    memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
    
    return {
        "memory_allocated_gb": memory_allocated,
        "memory_reserved_gb": memory_reserved,
        "memory_max_allocated_gb": memory_max_allocated,
        "memory_total_gb": memory_total,
        "memory_utilization": memory_allocated / memory_total,
        "memory_fragmentation": (memory_reserved - memory_allocated) / memory_reserved if memory_reserved > 0 else 0.0
    }


def cleanup_memory():
    """Clean up GPU memory to prevent fragmentation and OOM errors."""
    if torch.cuda.is_available():
        # Clear cache
        torch.cuda.empty_cache()
        # Force garbage collection
        gc.collect()
        # Synchronize to ensure cleanup is complete
        torch.cuda.synchronize()


def setup_memory_optimization():
    """Set up memory optimization environment variables and settings."""
    # Enable expandable segments to reduce fragmentation
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    
    # Set memory fraction to prevent GPU memory exhaustion
    if torch.cuda.is_available() and not hasattr(torch.cuda, '_memory_fraction_set'):
        torch.cuda.set_per_process_memory_fraction(0.85)  # Use 85% of GPU memory
        torch.cuda._memory_fraction_set = True


def check_memory_pressure(threshold_gb: float = 2.0) -> bool:
    """
    Check if we're running low on GPU memory.
    
    Args:
        threshold_gb: Free memory threshold in GB below which memory pressure is considered high
        
    Returns:
        True if memory pressure is high, False otherwise
    """
    if not torch.cuda.is_available():
        return False
    
    memory_info = get_memory_info()
    free_memory = memory_info["memory_total_gb"] - memory_info["memory_allocated_gb"]
    
    return free_memory < threshold_gb


def log_memory_usage(logger: Optional[logging.Logger] = None, prefix: str = ""):
    """Log current memory usage information."""
    memory_info = get_memory_info()
    
    if not memory_info:
        return
    
    message = (
        f"{prefix}GPU Memory: "
        f"{memory_info['memory_allocated_gb']:.2f}GB allocated, "
        f"{memory_info['memory_reserved_gb']:.2f}GB reserved, "
        f"{memory_info['memory_utilization']:.1%} utilization, "
        f"{memory_info['memory_fragmentation']:.1%} fragmentation"
    )
    
    if logger:
        logger.info(message)
    else:
        print(message)


class MemoryMonitor:
    """Context manager for monitoring memory usage during operations."""
    
    def __init__(self, name: str = "Operation", cleanup_on_exit: bool = True, logger: Optional[logging.Logger] = None):
        self.name = name
        self.cleanup_on_exit = cleanup_on_exit
        self.logger = logger
        self.start_memory = None
        
    def __enter__(self):
        if torch.cuda.is_available():
            self.start_memory = get_memory_info()
            log_memory_usage(self.logger, f"{self.name} - Start: ")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            end_memory = get_memory_info()
            log_memory_usage(self.logger, f"{self.name} - End: ")
            
            if self.start_memory:
                memory_diff = end_memory["memory_allocated_gb"] - self.start_memory["memory_allocated_gb"]
                if memory_diff > 0.1:  # Only log significant differences
                    message = f"{self.name} - Memory increase: +{memory_diff:.2f}GB"
                    if self.logger:
                        self.logger.warning(message)
                    else:
                        print(f"WARNING: {message}")
            
            if self.cleanup_on_exit:
                cleanup_memory()


def adaptive_batch_size(base_batch_size: int, memory_pressure_threshold: float = 0.8) -> int:
    """
    Adaptively adjust batch size based on current memory usage.
    
    Args:
        base_batch_size: The desired batch size
        memory_pressure_threshold: Memory utilization threshold above which to reduce batch size
        
    Returns:
        Adjusted batch size
    """
    if not torch.cuda.is_available():
        return base_batch_size
    
    memory_info = get_memory_info()
    utilization = memory_info.get("memory_utilization", 0.0)
    
    if utilization > memory_pressure_threshold:
        # Reduce batch size proportionally to memory pressure
        reduction_factor = max(0.25, 1.0 - (utilization - memory_pressure_threshold) / (1.0 - memory_pressure_threshold))
        adjusted_batch_size = max(1, int(base_batch_size * reduction_factor))
        
        print(f"High memory utilization ({utilization:.1%}), reducing batch size from {base_batch_size} to {adjusted_batch_size}")
        return adjusted_batch_size
    
    return base_batch_size