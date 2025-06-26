"""Data transformations for scaling law experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Dict, Any, List
from tqdm import tqdm


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
            
        # Precompute combined transformation matrix for efficiency
        self._precompute_transformation()
    
    def _precompute_transformation(self):
        """Precompute the combined transformation matrix."""
        with torch.no_grad():
            if self.projection_matrix is not None:
                # Combine both transformations into a single matrix
                self.combined_transform = torch.matmul(self.transform_matrix, self.projection_matrix)
            else:
                self.combined_transform = self.transform_matrix.clone()
            
            # Normalize rows to create a valid token mapping
            # Apply softmax to get probability distribution over output tokens
            self.token_mapping_probs = F.softmax(self.combined_transform, dim=1)
            
            # Create deterministic mapping by taking argmax
            self.token_mapping = torch.argmax(self.token_mapping_probs, dim=1)
            
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Apply linear transformation to token IDs using efficient lookup.
        
        Args:
            token_ids: Tensor of token IDs
            
        Returns:
            Transformed token IDs
        """
        # Ensure input token IDs are within valid range
        token_ids = torch.clamp(token_ids, 0, self.vocab_size - 1)
        
        # Use direct lookup instead of matrix multiplication
        transformed_ids = self.token_mapping[token_ids]
        
        return transformed_ids
    
    def forward_batch(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Optimized batch transformation.
        
        Args:
            token_ids: Batch tensor of token IDs
            
        Returns:
            Transformed token IDs
        """
        # Flatten for efficient lookup
        original_shape = token_ids.shape
        flat_ids = token_ids.flatten()
        
        # Transform using lookup
        flat_transformed = self.token_mapping[flat_ids]
        
        # Restore original shape
        return flat_transformed.view(original_shape)
    
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
            
        # Precompute token mapping
        self._precompute_mapping()
    
    def _precompute_mapping(self):
        """Precompute the nonlinear token mapping."""
        with torch.no_grad():
            # Create identity matrix for one-hot representation
            eye = torch.eye(self.vocab_size)
            
            # Apply transformations
            transformed = eye * self.scale_factor + self.shift
            transformed = self.nonlinearity(transformed)
            
            # Normalize to get valid probability distribution
            transformed = F.softmax(transformed, dim=1)
            
            # Create deterministic mapping
            self.token_mapping = torch.argmax(transformed, dim=1)
            
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Apply non-linear transformation to token IDs using lookup.
        
        Args:
            token_ids: Tensor of token IDs
            
        Returns:
            Transformed token IDs
        """
        # Ensure input token IDs are within valid range
        token_ids = torch.clamp(token_ids, 0, self.vocab_size - 1)
        
        # Use direct lookup
        transformed_ids = self.token_mapping[token_ids]
        
        return transformed_ids
    
    def forward_batch(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Optimized batch transformation.
        
        Args:
            token_ids: Batch tensor of token IDs
            
        Returns:
            Transformed token IDs
        """
        # Flatten for efficient lookup
        original_shape = token_ids.shape
        flat_ids = token_ids.flatten()
        
        # Transform using lookup
        flat_transformed = self.token_mapping[flat_ids]
        
        # Restore original shape
        return flat_transformed.view(original_shape)
    
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
    """Dataset wrapper that applies transformations with caching."""
    
    def __init__(
        self,
        base_dataset: Dataset,
        linear_transform: Optional[LinearTransform] = None,
        nonlinear_transform: Optional[NonLinearTransform] = None,
        cache_transformed: bool = True,
        batch_size: int = 1000,
    ):
        """
        Initialize transformed dataset.
        
        Args:
            base_dataset: Base dataset to transform
            linear_transform: Linear transformation to apply
            nonlinear_transform: Non-linear transformation to apply
            cache_transformed: Whether to cache transformed data
            batch_size: Batch size for transformation (when caching)
        """
        self.base_dataset = base_dataset
        self.linear_transform = linear_transform
        self.nonlinear_transform = nonlinear_transform
        self.cache_transformed = cache_transformed
        self.transformed_cache = {}
        
        # If caching is enabled, pre-transform all data
        if cache_transformed and (linear_transform is not None or nonlinear_transform is not None):
            self._cache_all_transformations(batch_size)
        
    def _cache_all_transformations(self, batch_size: int):
        """Pre-compute and cache all transformations."""
        print("Pre-computing transformations for faster training...")
        
        # Process in batches for efficiency
        num_samples = len(self.base_dataset)
        
        with torch.no_grad():
            for start_idx in tqdm(range(0, num_samples, batch_size), desc="Caching transformations"):
                end_idx = min(start_idx + batch_size, num_samples)
                
                # Collect batch of examples
                batch_input_ids = []
                batch_labels = []
                
                for idx in range(start_idx, end_idx):
                    example = self.base_dataset[idx]
                    batch_input_ids.append(example["input_ids"])
                    batch_labels.append(example["labels"])
                
                # Stack into tensors
                batch_input_ids = torch.stack(batch_input_ids)
                batch_labels = torch.stack(batch_labels)
                
                # Apply transformations
                if self.linear_transform is not None:
                    batch_input_ids = self.linear_transform.forward_batch(batch_input_ids)
                    batch_labels = self.linear_transform.forward_batch(batch_labels)
                    
                if self.nonlinear_transform is not None:
                    batch_input_ids = self.nonlinear_transform.forward_batch(batch_input_ids)
                    batch_labels = self.nonlinear_transform.forward_batch(batch_labels)
                
                # Store in cache
                for i, idx in enumerate(range(start_idx, end_idx)):
                    self.transformed_cache[idx] = {
                        "input_ids": batch_input_ids[i],
                        "labels": batch_labels[i],
                    }
        
        print(f"Cached {len(self.transformed_cache)} transformed examples")
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        """Get transformed example."""
        # If cached, return from cache
        if idx in self.transformed_cache:
            return self.transformed_cache[idx]
        
        # Otherwise, transform on the fly
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
    apply_linear: bool = True,
    apply_nonlinear: bool = True,
) -> tuple:
    """
    Create transformed dataloaders.
    
    Args:
        base_train_dataset: Base training dataset
        base_valid_dataset: Base validation dataset
        base_test_dataset: Base test dataset
        vocab_size: Size of vocabulary
        train_batch_size: Training batch size
        eval_batch_size: Evaluation batch size
        num_workers: Number of data loading workers
        linear_transform_dim: Dimension of linear transformation
        linear_seed: Seed for linear transformation
        nonlinear_type: Type of nonlinear transformation
        nonlinear_scale: Scale for nonlinear transformation
        nonlinear_seed: Seed for nonlinear transformation
        apply_linear: Whether to apply linear transformation
        apply_nonlinear: Whether to apply nonlinear transformation
    
    Returns:
        Tuple of (train_loader, valid_loader, test_loader, linear_transform, nonlinear_transform)
    """
    # Create transformations
    linear_transform = None
    nonlinear_transform = None
    
    if apply_linear:
        linear_transform = LinearTransform(
            vocab_size=vocab_size,
            transform_dim=linear_transform_dim,
            seed=linear_seed,
        )
        # Move transformation to CPU for data loading
        linear_transform = linear_transform.cpu()
    
    if apply_nonlinear:
        nonlinear_transform = NonLinearTransform(
            vocab_size=vocab_size,
            nonlinear_type=nonlinear_type,
            scale=nonlinear_scale,
            seed=nonlinear_seed,
        )
        # Move transformation to CPU for data loading
        nonlinear_transform = nonlinear_transform.cpu()
    
    # Create transformed datasets with caching enabled
    train_dataset = TransformedDataset(
        base_train_dataset,
        linear_transform=linear_transform,
        nonlinear_transform=nonlinear_transform,
        cache_transformed=True,
        batch_size=1000,
    )
    
    valid_dataset = TransformedDataset(
        base_valid_dataset,
        linear_transform=linear_transform,
        nonlinear_transform=nonlinear_transform,
        cache_transformed=True,
        batch_size=1000,
    )
    
    test_dataset = TransformedDataset(
        base_test_dataset,
        linear_transform=linear_transform,
        nonlinear_transform=nonlinear_transform,
        cache_transformed=True,
        batch_size=1000,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    return train_loader, valid_loader, test_loader, linear_transform, nonlinear_transform 