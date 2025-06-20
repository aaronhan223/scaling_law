"""Dataset loading and processing for WikiText-103."""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm


class WikiTextDataset(Dataset):
    """WikiText-103 dataset for language modeling."""
    
    def __init__(
        self,
        split: str,
        tokenizer_name: str = "gpt2",
        max_length: int = 1024,
        max_tokens: Optional[int] = None,
        seed: int = 42,
    ):
        """
        Initialize WikiText dataset.
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            tokenizer_name: Name of the tokenizer to use
            max_length: Maximum sequence length
            max_tokens: Maximum number of tokens to use (for scaling experiments)
            seed: Random seed for sampling
        """
        self.split = split
        self.max_length = max_length
        self.max_tokens = max_tokens
        self.seed = seed
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load dataset
        print(f"Loading WikiText-103 {split} split...")
        self.dataset = load_dataset("wikitext", "wikitext-103-v1", split=split)
        
        # Tokenize and prepare data
        self._prepare_data()
        
    def _prepare_data(self):
        """Tokenize and prepare the dataset."""
        # Tokenize all texts
        print(f"Tokenizing {self.split} split...")
        all_token_ids = []
        
        for example in tqdm(self.dataset, desc="Tokenizing"):
            text = example["text"]
            if text.strip():  # Skip empty texts
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                all_token_ids.extend(tokens)
        
        # Convert to numpy array for efficient slicing
        all_token_ids = np.array(all_token_ids)
        
        # Limit tokens if specified
        if self.max_tokens is not None and len(all_token_ids) > self.max_tokens:
            np.random.seed(self.seed)
            # For training, take first max_tokens
            # For validation/test, sample randomly but consistently
            if self.split == "train":
                all_token_ids = all_token_ids[:self.max_tokens]
            else:
                # Sample a subset for validation/test
                indices = np.random.choice(len(all_token_ids), self.max_tokens, replace=False)
                indices.sort()  # Keep order
                all_token_ids = all_token_ids[indices]
        
        # Create sequences of max_length
        self.sequences = []
        for i in range(0, len(all_token_ids) - self.max_length, self.max_length):
            self.sequences.append(all_token_ids[i:i + self.max_length + 1])
        
        self.sequences = torch.tensor(self.sequences, dtype=torch.long)
        print(f"Prepared {len(self.sequences)} sequences of length {self.max_length}")
        print(f"Total tokens: {len(all_token_ids):,}")
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """Get a single example."""
        sequence = self.sequences[idx]
        # Input is all tokens except last, target is all tokens except first
        return {
            "input_ids": sequence[:-1],
            "labels": sequence[1:],
        }


def create_dataloaders(
    tokenizer_name: str = "gpt2",
    max_length: int = 1024,
    train_batch_size: int = 32,
    eval_batch_size: int = 64,
    max_tokens_train: Optional[int] = None,
    max_tokens_valid: Optional[int] = None,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for train, validation, and test sets.
    
    Returns:
        Tuple of (train_loader, valid_loader, test_loader)
    """
    # Create datasets
    train_dataset = WikiTextDataset(
        split="train",
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        max_tokens=max_tokens_train,
        seed=seed,
    )
    
    valid_dataset = WikiTextDataset(
        split="validation",
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        max_tokens=max_tokens_valid,
        seed=seed,
    )
    
    test_dataset = WikiTextDataset(
        split="test",
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        max_tokens=max_tokens_valid,  # Use same size as validation
        seed=seed,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, valid_loader, test_loader


def get_tokenizer(tokenizer_name: str = "gpt2"):
    """Get tokenizer with proper configuration."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer 