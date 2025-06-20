"""Data transformations for scaling law experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from typing import Optional, Dict, Any


class LinearTransform(nn.Module):
    """Random linear transformation for token embeddings."""
    
    def __init__(
        self,
        vocab_size: int,
        transform_dim: Optional[int] = None,
        seed: int = 42,
    ):
        """
        Initialize linear transformation.
        
        Args:
            vocab_size: Size of vocabulary
            transform_dim: Dimension of transformation (if None, use vocab_size)
            seed: Random seed
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.transform_dim = transform_dim or vocab_size
        
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Initialize random transformation matrix
        # Using Xavier initialization
        self.transform_matrix = nn.Parameter(
            torch.randn(vocab_size, self.transform_dim) * np.sqrt(2.0 / (vocab_size + self.transform_dim))
        )
        
        # If transform_dim != vocab_size, we need a projection back
        if self.transform_dim != vocab_size:
            self.projection_matrix = nn.Parameter(
                torch.randn(self.transform_dim, vocab_size) * np.sqrt(2.0 / (self.transform_dim + vocab_size))
            )
        else:
            self.projection_matrix = None
            
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Apply linear transformation to token IDs.
        
        Args:
            token_ids: Tensor of token IDs
            
        Returns:
            Transformed token IDs
        """
        # Convert token IDs to one-hot
        one_hot = F.one_hot(token_ids, num_classes=self.vocab_size).float()
        
        # Apply transformation
        transformed = torch.matmul(one_hot, self.transform_matrix)
        
        # Project back if needed
        if self.projection_matrix is not None:
            transformed = torch.matmul(transformed, self.projection_matrix)
        
        # Convert back to token IDs by taking argmax
        transformed_ids = torch.argmax(transformed, dim=-1)
        
        return transformed_ids
    
    def transform_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply transformation directly to embeddings.
        
        Args:
            embeddings: Embedding matrix [vocab_size, embedding_dim]
            
        Returns:
            Transformed embeddings
        """
        # Apply transformation to embedding matrix
        transformed_embeddings = torch.matmul(self.transform_matrix.T, embeddings)
        
        if self.projection_matrix is not None:
            transformed_embeddings = torch.matmul(self.projection_matrix.T, transformed_embeddings)
            
        return transformed_embeddings


class NonLinearTransform(nn.Module):
    """Random non-linear transformation for token embeddings."""
    
    def __init__(
        self,
        vocab_size: int,
        nonlinear_type: str = "tanh",
        scale: float = 0.1,
        seed: int = 43,
    ):
        """
        Initialize non-linear transformation.
        
        Args:
            vocab_size: Size of vocabulary
            nonlinear_type: Type of non-linearity ('tanh', 'sigmoid', 'relu')
            scale: Scale factor for random noise
            seed: Random seed
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.nonlinear_type = nonlinear_type
        self.scale = scale
        
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Initialize random parameters for non-linear transformation
        self.shift = nn.Parameter(torch.randn(vocab_size) * scale)
        self.scale_factor = nn.Parameter(1.0 + torch.randn(vocab_size) * scale)
        
        # Select non-linearity
        if nonlinear_type == "tanh":
            self.nonlinearity = torch.tanh
        elif nonlinear_type == "sigmoid":
            self.nonlinearity = torch.sigmoid
        elif nonlinear_type == "relu":
            self.nonlinearity = F.relu
        else:
            raise ValueError(f"Unknown nonlinear_type: {nonlinear_type}")
            
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Apply non-linear transformation to token IDs.
        
        Args:
            token_ids: Tensor of token IDs
            
        Returns:
            Transformed token IDs
        """
        # Convert to one-hot
        one_hot = F.one_hot(token_ids, num_classes=self.vocab_size).float()
        
        # Apply scaling and shifting
        transformed = one_hot * self.scale_factor + self.shift
        
        # Apply non-linearity
        transformed = self.nonlinearity(transformed)
        
        # Normalize to get valid probability distribution
        transformed = F.softmax(transformed, dim=-1)
        
        # Sample from the distribution (or take argmax for deterministic)
        transformed_ids = torch.argmax(transformed, dim=-1)
        
        return transformed_ids
    
    def transform_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply transformation directly to embeddings.
        
        Args:
            embeddings: Embedding matrix [vocab_size, embedding_dim]
            
        Returns:
            Transformed embeddings
        """
        # Apply scaling and shifting to embeddings
        transformed_embeddings = embeddings * self.scale_factor.unsqueeze(1) + self.shift.unsqueeze(1)
        
        # Apply non-linearity
        transformed_embeddings = self.nonlinearity(transformed_embeddings)
        
        return transformed_embeddings


class TransformedDataset(Dataset):
    """Dataset wrapper that applies transformations."""
    
    def __init__(
        self,
        base_dataset: Dataset,
        linear_transform: Optional[LinearTransform] = None,
        nonlinear_transform: Optional[NonLinearTransform] = None,
    ):
        """
        Initialize transformed dataset.
        
        Args:
            base_dataset: Base dataset to transform
            linear_transform: Linear transformation to apply
            nonlinear_transform: Non-linear transformation to apply
        """
        self.base_dataset = base_dataset
        self.linear_transform = linear_transform
        self.nonlinear_transform = nonlinear_transform
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        """Get transformed example."""
        example = self.base_dataset[idx]
        
        # Apply transformations
        input_ids = example["input_ids"]
        labels = example["labels"]
        
        if self.linear_transform is not None:
            input_ids = self.linear_transform(input_ids)
            labels = self.linear_transform(labels)
            
        if self.nonlinear_transform is not None:
            input_ids = self.nonlinear_transform(input_ids)
            labels = self.nonlinear_transform(labels)
            
        return {
            "input_ids": input_ids,
            "labels": labels,
        }


def create_transformed_dataloaders(
    base_train_dataset: Dataset,
    base_valid_dataset: Dataset,
    base_test_dataset: Dataset,
    vocab_size: int,
    train_batch_size: int = 32,
    eval_batch_size: int = 64,
    num_workers: int = 4,
    linear_transform_dim: Optional[int] = None,
    linear_seed: int = 42,
    nonlinear_type: str = "tanh",
    nonlinear_scale: float = 0.1,
    nonlinear_seed: int = 43,
) -> tuple:
    """
    Create transformed dataloaders.
    
    Returns:
        Tuple of (train_loader, valid_loader, test_loader, linear_transform, nonlinear_transform)
    """
    # Create transformations
    linear_transform = LinearTransform(
        vocab_size=vocab_size,
        transform_dim=linear_transform_dim,
        seed=linear_seed,
    )
    
    nonlinear_transform = NonLinearTransform(
        vocab_size=vocab_size,
        nonlinear_type=nonlinear_type,
        scale=nonlinear_scale,
        seed=nonlinear_seed,
    )
    
    # Create transformed datasets
    train_dataset = TransformedDataset(
        base_train_dataset,
        linear_transform=linear_transform,
        nonlinear_transform=nonlinear_transform,
    )
    
    valid_dataset = TransformedDataset(
        base_valid_dataset,
        linear_transform=linear_transform,
        nonlinear_transform=nonlinear_transform,
    )
    
    test_dataset = TransformedDataset(
        base_test_dataset,
        linear_transform=linear_transform,
        nonlinear_transform=nonlinear_transform,
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
    
    return train_loader, valid_loader, test_loader, linear_transform, nonlinear_transform 