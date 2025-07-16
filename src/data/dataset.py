"""Dataset loading and processing for WikiText-103."""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm


class CustomDataset(Dataset):
    """WikiText-103 dataset for language modeling."""
    
    def __init__(
        self,
        split: str,
        dataset_name: str,
        dataset_config: Optional[str] = None,
        tokenizer_name: str = "gpt2",
        max_length: int = 1024,
        max_tokens: Optional[int] = None,
        seed: int = 42,
        dataset_obj: Optional[Dataset] = None,
    ):
        """
        Initialize custom dataset.
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            dataset_name: Name of the dataset on Hugging Face Hub
            dataset_config: Optional dataset configuration
            tokenizer_name: Name of the tokenizer to use
            max_length: Maximum sequence length
            max_tokens: Maximum number of tokens to use (for scaling experiments)
            seed: Random seed for sampling
            dataset_obj: Optional pre-loaded dataset object
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
        if dataset_obj is not None:
            self.dataset = dataset_obj
        else:
            print(f"Loading {dataset_name} ({dataset_config or 'default'}) {split} split...")
            if dataset_config:
                self.dataset = load_dataset(dataset_name, dataset_config, split=split)
            else:
                self.dataset = load_dataset(dataset_name, split=split)
        
        # Tokenize and prepare data
        self._prepare_data()
        
    def _prepare_data(self):
        """Tokenize and prepare the dataset."""
        print(f"Tokenizing {self.split} split...")
        
        self.sequences = []
        for example in tqdm(self.dataset, desc="Tokenizing and chunking"):
            text = example["text"]
            if text.strip():
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                
                for i in range(0, len(tokens), self.max_length):
                    sequence_chunk = tokens[i: i + self.max_length + 1]
                    if len(sequence_chunk) == self.max_length + 1:
                        self.sequences.append(sequence_chunk)

        if self.max_tokens is not None:
            num_sequences_to_keep = self.max_tokens // self.max_length
            if len(self.sequences) > num_sequences_to_keep:
                if self.split == 'train':
                    self.sequences = self.sequences[:num_sequences_to_keep]
                else:
                    np.random.seed(self.seed)
                    indices = np.random.choice(len(self.sequences), num_sequences_to_keep, replace=False)
                    indices.sort()
                    self.sequences = [self.sequences[i] for i in indices]

        self.sequences = torch.tensor(self.sequences, dtype=torch.long)
        
        print(f"Prepared {len(self.sequences)} sequences of length {self.max_length}")
        print(f"Total tokens in prepared sequences: {len(self.sequences) * self.max_length:,}")
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """Get a single example."""
        sequence = self.sequences[idx]
        return {
            "input_ids": sequence[:-1],
            "labels": sequence[1:],
        }


def create_dataloaders(
    dataset_name: str,
    dataset_config: Optional[str],
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
    train_dataset = CustomDataset(
        split="train",
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        max_tokens=max_tokens_train,
        seed=seed,
    )
    
    valid_dataset = CustomDataset(
        split="validation",
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        max_tokens=max_tokens_valid,
        seed=seed,
    )
    
    test_dataset = CustomDataset(
        split="test",
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        max_tokens=max_tokens_valid,
        seed=seed,
    )
    
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