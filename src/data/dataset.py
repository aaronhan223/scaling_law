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
        self.dataset_name = dataset_name
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
        
        # Use different strategies based on dataset type
        if self.dataset_name == 'openwebtext':
            # New chunking approach for OpenWebText (longer texts)
            self._prepare_data_chunking()
        else:
            # Original concatenation approach for WikiText and other datasets
            self._prepare_data_concatenation()
    
    def _prepare_data_chunking(self):
        """Tokenize and chunk data (for OpenWebText)."""
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

        # Safety check: ensure we have at least some sequences
        if len(self.sequences) == 0:
            print(f"Warning: No sequences found for {self.split} split. Creating minimal sequences.")
            # Create a few minimal sequences with padding
            pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            minimal_sequence = [pad_token_id] * (self.max_length + 1)
            self.sequences = [minimal_sequence] * max(1, 10)  # At least 10 sequences

        self.sequences = torch.tensor(self.sequences, dtype=torch.long)
        
        print(f"Prepared {len(self.sequences)} sequences of length {self.max_length}")
        print(f"Total tokens in prepared sequences: {len(self.sequences) * self.max_length:,}")
    
    def _prepare_data_concatenation(self):
        """Tokenize and concatenate data (for WikiText and other datasets)."""
        # Tokenize all texts
        all_token_ids = []
        
        for example in tqdm(self.dataset, desc="Tokenizing"):
            text = example["text"]
            if text.strip():  # Skip empty texts
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                all_token_ids.extend(tokens)
        
        # Safety check: ensure we have enough tokens
        if len(all_token_ids) < self.max_length + 1:
            print(f"Warning: Not enough tokens ({len(all_token_ids)}) for sequence length {self.max_length}. Padding with EOS tokens.")
            pad_token_id = self.tokenizer.eos_token_id
            all_token_ids.extend([pad_token_id] * (self.max_length + 1 - len(all_token_ids)))
        
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
                if len(all_token_ids) >= self.max_tokens:
                    indices = np.random.choice(len(all_token_ids), self.max_tokens, replace=False)
                    indices.sort()  # Keep order
                    all_token_ids = all_token_ids[indices]
        
        # Create sequences of max_length
        self.sequences = []
        for i in range(0, len(all_token_ids) - self.max_length, self.max_length):
            self.sequences.append(all_token_ids[i:i + self.max_length + 1])

        # Safety check: ensure we have at least some sequences
        if len(self.sequences) == 0:
            if len(all_token_ids) >= self.max_length + 1:
                # Create at least one sequence from the available tokens
                self.sequences.append(all_token_ids[:self.max_length + 1])
            else:
                # If we don't have enough tokens, create minimal sequences
                print(f"Warning: Only {len(all_token_ids)} tokens available, less than required sequence length {self.max_length + 1}")
                print("Creating minimal sequences with padding...")
                # Create a minimal sequence by padding
                pad_token_id = 50256  # GPT-2 pad token
                if len(all_token_ids) > 0:
                    # Use available tokens and pad the rest
                    minimal_sequence = list(all_token_ids) + [pad_token_id] * (self.max_length + 1 - len(all_token_ids))
                else:
                    # Create completely synthetic sequence
                    minimal_sequence = [pad_token_id] * (self.max_length + 1)
                self.sequences = [minimal_sequence] * max(1, 10)  # At least 10 sequences

        self.sequences = torch.tensor(self.sequences, dtype=torch.long)
        
        print(f"Prepared {len(self.sequences)} sequences of length {self.max_length}")
        print(f"Total tokens: {len(all_token_ids):,}")
        
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