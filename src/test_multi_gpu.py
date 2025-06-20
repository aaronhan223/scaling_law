#!/usr/bin/env python3
"""Test script to verify multi-GPU training works correctly."""

import torch
import torch.nn as nn
from models.transformer import TransformerLM
from data.dataset import create_dataloaders
from training.trainer import Trainer
from config import ModelConfig, DataConfig, TrainingConfig
import os

def test_multi_gpu():
    """Test multi-GPU training with a small model and dataset."""
    print("Testing multi-GPU training setup...")
    
    # Check available GPUs
    n_gpu = torch.cuda.device_count()
    print(f"Number of GPUs available: {n_gpu}")
    
    if n_gpu == 0:
        print("No GPUs available. Skipping multi-GPU test.")
        return
    
    # Create small model config
    model_config = {
        "vocab_size": 50257,
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "intermediate_size": 512,
        "max_position_embeddings": 1024,
    }
    
    # Create model
    model = TransformerLM(**model_config)
    print(f"Created model with {model.get_num_params():,} parameters")
    
    # Create minimal data config
    data_config = DataConfig(
        dataset_sizes=[10000],  # Very small dataset for testing
        train_batch_size=8,
        eval_batch_size=16,
    )
    
    # Create dataloaders with small dataset
    train_loader, valid_loader, test_loader = create_dataloaders(
        data_config,
        max_train_tokens=10000,
        tokenizer_name="gpt2",
    )
    
    # Create training config
    training_config = TrainingConfig(
        learning_rate=1e-4,
        max_steps=10,  # Just 10 steps for testing
        eval_steps=5,
        logging_steps=1,
        fp16=False,  # Disable FP16 for testing
        output_dir="./test_output",
    )
    
    # Test single GPU training first
    print("\nTesting single-GPU training...")
    trainer_single = Trainer(
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=valid_loader,
        test_dataloader=test_loader,
        learning_rate=training_config.learning_rate,
        max_steps=5,
        eval_steps=training_config.eval_steps,
        logging_steps=training_config.logging_steps,
        fp16=training_config.fp16,
        output_dir=training_config.output_dir + "_single",
        use_wandb=False,
        use_multi_gpu=False,
    )
    
    try:
        trainer_single.train()
        print("✓ Single-GPU training works correctly")
    except Exception as e:
        print(f"✗ Single-GPU training failed: {e}")
        raise
    
    # Test multi-GPU training if multiple GPUs available
    if n_gpu > 1:
        print(f"\nTesting multi-GPU training with {n_gpu} GPUs...")
        
        # Create fresh model for multi-GPU test
        model = TransformerLM(**model_config)
        
        trainer_multi = Trainer(
            model=model,
            train_dataloader=train_loader,
            eval_dataloader=valid_loader,
            test_dataloader=test_loader,
            learning_rate=training_config.learning_rate,
            max_steps=5,
            eval_steps=training_config.eval_steps,
            logging_steps=training_config.logging_steps,
            fp16=training_config.fp16,
            output_dir=training_config.output_dir + "_multi",
            use_wandb=False,
            use_multi_gpu=True,
        )
        
        try:
            trainer_multi.train()
            print(f"✓ Multi-GPU training works correctly with {n_gpu} GPUs")
        except Exception as e:
            print(f"✗ Multi-GPU training failed: {e}")
            raise
    else:
        print("Only 1 GPU available, skipping multi-GPU test")
    
    # Clean up test outputs
    import shutil
    for suffix in ["_single", "_multi"]:
        output_dir = training_config.output_dir + suffix
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
    
    print("\nAll tests passed! ✓")


if __name__ == "__main__":
    test_multi_gpu() 