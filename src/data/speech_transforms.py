"""Speech-specific data transformations for scaling law experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from tqdm import tqdm
import math
from .cache_utils import get_dataset_cache

# Optional import for sklearn
try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    KMeans = None


class SpeechLinearTransform(nn.Module):
    """Linear transformation for speech features and CTC labels."""

    def __init__(
        self,
        vocab_size: int,
        feature_dim: Optional[int] = None,
        transform_feature_dim: Optional[int] = None,
        seed: int = 42,
    ):
        """
        Initialize speech linear transformation.

        Args:
            vocab_size: Size of CTC vocabulary
            feature_dim: Dimension of input audio features (if None, apply only to labels)
            transform_feature_dim: Dimension of feature transformation
            seed: Random seed
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.feature_dim = feature_dim
        self.transform_feature_dim = transform_feature_dim or feature_dim

        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create label transformation matrix (for CTC tokens)
        self.label_transform_matrix = nn.Parameter(
            torch.randn(vocab_size, vocab_size) * np.sqrt(2.0 / (vocab_size + vocab_size))
        )

        # Precompute label mapping
        self._precompute_label_mapping()

        # Create feature transformation if feature_dim is provided
        if feature_dim is not None:
            self.feature_transform_matrix = nn.Parameter(
                torch.randn(feature_dim, self.transform_feature_dim) *
                np.sqrt(2.0 / (feature_dim + self.transform_feature_dim))
            )

            # Projection back to original dimension if needed
            if self.transform_feature_dim != feature_dim:
                self.feature_projection_matrix = nn.Parameter(
                    torch.randn(self.transform_feature_dim, feature_dim) *
                    np.sqrt(2.0 / (self.transform_feature_dim + feature_dim))
                )
            else:
                self.feature_projection_matrix = None
        else:
            self.feature_transform_matrix = None
            self.feature_projection_matrix = None

    def _precompute_label_mapping(self):
        """Precompute the label transformation mapping."""
        with torch.no_grad():
            # Apply softmax to get probability distribution
            label_probs = F.softmax(self.label_transform_matrix, dim=1)

            # Create deterministic mapping by taking argmax
            self.label_mapping = torch.argmax(label_probs, dim=1)

    def transform_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Transform audio features.

        Args:
            features: Audio features [batch_size, sequence_length, feature_dim]

        Returns:
            Transformed features
        """
        if self.feature_transform_matrix is None:
            return features

        # Apply linear transformation
        transformed = torch.matmul(features, self.feature_transform_matrix)

        # Project back to original dimension if needed
        if self.feature_projection_matrix is not None:
            transformed = torch.matmul(transformed, self.feature_projection_matrix)

        return transformed

    def transform_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Transform CTC labels.

        Args:
            labels: CTC label tensor [batch_size, sequence_length]

        Returns:
            Transformed labels
        """
        # Handle padding tokens (-100)
        mask = labels != -100

        # Only transform valid tokens
        valid_labels = labels[mask]
        valid_labels = torch.clamp(valid_labels, 0, self.vocab_size - 1)

        # Transform using lookup
        transformed_valid = self.label_mapping[valid_labels]

        # Reconstruct full tensor
        transformed_labels = labels.clone()
        transformed_labels[mask] = transformed_valid

        return transformed_labels

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply transformation to a batch."""
        result = batch.copy()

        # Transform features if available
        if 'input_values' in batch and self.feature_transform_matrix is not None:
            # For raw audio, we don't transform directly
            pass
        elif 'features' in batch and self.feature_transform_matrix is not None:
            result['features'] = self.transform_features(batch['features'])

        # Transform labels
        if 'labels' in batch:
            result['labels'] = self.transform_labels(batch['labels'])

        return result


class SpeechNonLinearTransform(nn.Module):
    """Non-linear transformation for speech features and CTC labels."""

    def __init__(
        self,
        vocab_size: int,
        feature_dim: Optional[int] = None,
        hidden_dims: Optional[List[int]] = None,
        nonlinear_type: str = "relu",
        dropout_rate: float = 0.1,
        noise_scale: float = 0.05,
        seed: int = 43,
    ):
        """
        Initialize speech non-linear transformation.

        Args:
            vocab_size: Size of CTC vocabulary
            feature_dim: Dimension of input audio features
            hidden_dims: Hidden layer dimensions for feature transformation
            nonlinear_type: Type of non-linearity
            dropout_rate: Dropout rate
            noise_scale: Scale for noise injection
            seed: Random seed
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.feature_dim = feature_dim
        self.nonlinear_type = nonlinear_type
        self.dropout_rate = dropout_rate
        self.noise_scale = noise_scale

        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create label transformation network
        if hidden_dims is None:
            label_hidden_dims = [vocab_size * 2, vocab_size, vocab_size // 2]
        else:
            label_hidden_dims = hidden_dims

        self.label_network = self._build_network(vocab_size, vocab_size, label_hidden_dims)

        # Precompute label mapping
        self._precompute_label_mapping()

        # Create feature transformation network if feature_dim is provided
        if feature_dim is not None:
            if hidden_dims is None:
                feature_hidden_dims = [feature_dim * 2, feature_dim, feature_dim]
            else:
                feature_hidden_dims = [max(dim, feature_dim // 2) for dim in hidden_dims]

            self.feature_network = self._build_network(feature_dim, feature_dim, feature_hidden_dims)
        else:
            self.feature_network = None

    def _build_network(self, input_dim: int, output_dim: int, hidden_dims: List[int]) -> nn.Module:
        """Build a multi-layer network."""
        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))

            if self.nonlinear_type == "relu":
                layers.append(nn.ReLU())
            elif self.nonlinear_type == "gelu":
                layers.append(nn.GELU())
            elif self.nonlinear_type == "tanh":
                layers.append(nn.Tanh())
            elif self.nonlinear_type == "sigmoid":
                layers.append(nn.Sigmoid())
            else:
                raise ValueError(f"Unknown nonlinear_type: {self.nonlinear_type}")

            layers.append(nn.Dropout(self.dropout_rate))
            current_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(current_dim, output_dim))

        return nn.Sequential(*layers)

    def _precompute_label_mapping(self):
        """Precompute the label transformation mapping."""
        with torch.no_grad():
            self.label_network.eval()

            # Create one-hot identity matrix
            eye = torch.eye(self.vocab_size, dtype=torch.float32)

            # Add noise for stochasticity
            if self.noise_scale > 0:
                noise = torch.randn_like(eye) * self.noise_scale
                eye = eye + noise

            # Process in batches to avoid memory issues
            batch_size = min(500, self.vocab_size)
            transformed_outputs = []

            for i in range(0, self.vocab_size, batch_size):
                end_idx = min(i + batch_size, self.vocab_size)
                batch_input = eye[i:end_idx]

                # Forward pass
                batch_output = self.label_network(batch_input)
                transformed_outputs.append(batch_output)

            # Concatenate outputs
            transformed = torch.cat(transformed_outputs, dim=0)

            # Apply softmax and take argmax
            transformed = F.softmax(transformed, dim=1)
            self.label_mapping = torch.argmax(transformed, dim=1)

    def transform_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Transform audio features.

        Args:
            features: Audio features [batch_size, sequence_length, feature_dim]

        Returns:
            Transformed features
        """
        if self.feature_network is None:
            return features

        original_shape = features.shape
        batch_size, seq_len = original_shape[:2]

        # Reshape for processing
        features_flat = features.view(-1, features.size(-1))

        # Apply transformation
        transformed_flat = self.feature_network(features_flat)

        # Add noise during training
        if self.training and self.noise_scale > 0:
            noise = torch.randn_like(transformed_flat) * self.noise_scale * 0.1
            transformed_flat = transformed_flat + noise

        # Reshape back
        transformed = transformed_flat.view(original_shape)

        return transformed

    def transform_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Transform CTC labels.

        Args:
            labels: CTC label tensor [batch_size, sequence_length]

        Returns:
            Transformed labels
        """
        # Handle padding tokens (-100)
        mask = labels != -100

        # Only transform valid tokens
        valid_labels = labels[mask]
        valid_labels = torch.clamp(valid_labels, 0, self.vocab_size - 1)

        # Transform using lookup
        transformed_valid = self.label_mapping[valid_labels]

        # Reconstruct full tensor
        transformed_labels = labels.clone()
        transformed_labels[mask] = transformed_valid

        return transformed_labels

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply transformation to a batch."""
        result = batch.copy()

        # Transform features if available
        if 'features' in batch and self.feature_network is not None:
            result['features'] = self.transform_features(batch['features'])

        # Transform labels
        if 'labels' in batch:
            result['labels'] = self.transform_labels(batch['labels'])

        return result


class SpeechQuantizationTransform(nn.Module):
    """Quantization transformation for speech data."""

    def __init__(
        self,
        vocab_size: int,
        num_levels: int,
        feature_dim: Optional[int] = None,
        quantize_features: bool = False,
        seed: int = 46,
    ):
        """
        Initialize speech quantization transformation.

        Args:
            vocab_size: Size of CTC vocabulary
            num_levels: Number of quantization levels
            feature_dim: Dimension of audio features
            quantize_features: Whether to quantize features as well
            seed: Random seed
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.num_levels = min(num_levels, vocab_size)
        self.feature_dim = feature_dim
        self.quantize_features = quantize_features

        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create quantization levels for labels
        self.register_buffer(
            'label_quantization_levels',
            torch.linspace(0, vocab_size - 1, num_levels).long()
        )

        # Precompute label quantization mapping
        self._precompute_label_quantization()

        # Feature quantization if requested
        if quantize_features and feature_dim is not None:
            # Create quantization levels for features
            self.register_buffer(
                'feature_quantization_levels',
                torch.linspace(-3.0, 3.0, num_levels).unsqueeze(1).expand(-1, feature_dim)
            )
        else:
            self.feature_quantization_levels = None

    def _precompute_label_quantization(self):
        """Precompute label quantization mapping."""
        with torch.no_grad():
            # Map each vocabulary token to nearest quantization level
            vocab_tokens = torch.arange(self.vocab_size, dtype=torch.float32)
            quantization_levels = self.label_quantization_levels.float()

            # Find nearest quantization level for each token
            distances = torch.abs(vocab_tokens.unsqueeze(1) - quantization_levels.unsqueeze(0))
            nearest_levels = torch.argmin(distances, dim=1)

            # Map back to vocabulary tokens using the quantization levels
            self.label_mapping = self.label_quantization_levels[nearest_levels]

    def quantize_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Quantize audio features.

        Args:
            features: Audio features [batch_size, sequence_length, feature_dim]

        Returns:
            Quantized features
        """
        if self.feature_quantization_levels is None:
            return features

        # Find nearest quantization level for each feature element
        distances = torch.cdist(
            features.view(-1, features.size(-1)),
            self.feature_quantization_levels
        )
        nearest_levels = torch.argmin(distances, dim=1)

        # Get quantized values
        quantized_flat = self.feature_quantization_levels[nearest_levels]

        # Reshape back to original shape
        quantized = quantized_flat.view(features.shape)

        return quantized

    def transform_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Quantize CTC labels.

        Args:
            labels: CTC label tensor [batch_size, sequence_length]

        Returns:
            Quantized labels
        """
        # Handle padding tokens (-100)
        mask = labels != -100

        # Only transform valid tokens
        valid_labels = labels[mask]
        valid_labels = torch.clamp(valid_labels, 0, self.vocab_size - 1)

        # Transform using lookup
        transformed_valid = self.label_mapping[valid_labels]

        # Reconstruct full tensor
        transformed_labels = labels.clone()
        transformed_labels[mask] = transformed_valid

        return transformed_labels

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply transformation to a batch."""
        result = batch.copy()

        # Quantize features if available
        if 'features' in batch and self.quantize_features:
            result['features'] = self.quantize_features(batch['features'])

        # Quantize labels
        if 'labels' in batch:
            result['labels'] = self.transform_labels(batch['labels'])

        return result

    def get_compression_ratio(self) -> float:
        """Get the compression ratio."""
        return self.num_levels / self.vocab_size


class SpeechLowRankTransform(nn.Module):
    """Low-rank transformation for speech data."""

    def __init__(
        self,
        vocab_size: int,
        rank: int,
        feature_dim: Optional[int] = None,
        transform_features: bool = False,
        seed: int = 47,
    ):
        """
        Initialize speech low-rank transformation.

        Args:
            vocab_size: Size of CTC vocabulary
            rank: Rank of the low-rank decomposition
            feature_dim: Dimension of audio features
            transform_features: Whether to transform features as well
            seed: Random seed
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.rank = min(rank, vocab_size)
        self.feature_dim = feature_dim
        self.transform_features = transform_features

        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create low-rank matrices for label transformation
        self.label_U = nn.Parameter(
            torch.randn(vocab_size, self.rank) * np.sqrt(2.0 / (vocab_size + self.rank))
        )
        self.label_V = nn.Parameter(
            torch.randn(vocab_size, self.rank) * np.sqrt(2.0 / (vocab_size + self.rank))
        )

        # Precompute label transformation
        self._precompute_label_transformation()

        # Feature transformation if requested
        if transform_features and feature_dim is not None:
            feature_rank = min(rank, feature_dim)
            self.feature_U = nn.Parameter(
                torch.randn(feature_dim, feature_rank) * np.sqrt(2.0 / (feature_dim + feature_rank))
            )
            self.feature_V = nn.Parameter(
                torch.randn(feature_dim, feature_rank) * np.sqrt(2.0 / (feature_dim + feature_rank))
            )
        else:
            self.feature_U = None
            self.feature_V = None

    def _precompute_label_transformation(self):
        """Precompute the low-rank label transformation."""
        with torch.no_grad():
            # Compute M = U @ V^T
            transformation_matrix = torch.matmul(self.label_U, self.label_V.T)

            # Apply softmax and take argmax
            label_probs = F.softmax(transformation_matrix, dim=1)
            self.label_mapping = torch.argmax(label_probs, dim=1)

    def transform_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Transform audio features using low-rank decomposition.

        Args:
            features: Audio features [batch_size, sequence_length, feature_dim]

        Returns:
            Transformed features
        """
        if self.feature_U is None or self.feature_V is None:
            return features

        original_shape = features.shape

        # Reshape for matrix operations
        features_flat = features.view(-1, features.size(-1))

        # Apply low-rank transformation: M @ features where M = U @ V^T
        intermediate = torch.matmul(features_flat, self.feature_V)
        transformed_flat = torch.matmul(intermediate, self.feature_U.T)

        # Reshape back
        transformed = transformed_flat.view(original_shape)

        return transformed

    def transform_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Transform CTC labels using low-rank mapping.

        Args:
            labels: CTC label tensor [batch_size, sequence_length]

        Returns:
            Transformed labels
        """
        # Handle padding tokens (-100)
        mask = labels != -100

        # Only transform valid tokens
        valid_labels = labels[mask]
        valid_labels = torch.clamp(valid_labels, 0, self.vocab_size - 1)

        # Transform using lookup
        transformed_valid = self.label_mapping[valid_labels]

        # Reconstruct full tensor
        transformed_labels = labels.clone()
        transformed_labels[mask] = transformed_valid

        return transformed_labels

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply transformation to a batch."""
        result = batch.copy()

        # Transform features if available
        if 'features' in batch and self.transform_features:
            result['features'] = self.transform_features(batch['features'])

        # Transform labels
        if 'labels' in batch:
            result['labels'] = self.transform_labels(batch['labels'])

        return result

    def get_compression_ratio(self) -> float:
        """Get the compression ratio."""
        low_rank_params = 2 * self.vocab_size * self.rank
        full_rank_params = self.vocab_size * self.vocab_size
        return low_rank_params / full_rank_params


class SpeechTransformedDataset(Dataset):
    """Dataset wrapper for speech data with transformations."""

    def __init__(
        self,
        base_dataset: Dataset,
        linear_transform: Optional[SpeechLinearTransform] = None,
        nonlinear_transform: Optional[SpeechNonLinearTransform] = None,
        quantization_transform: Optional[SpeechQuantizationTransform] = None,
        lowrank_transform: Optional[SpeechLowRankTransform] = None,
        cache_transformed: bool = False,  # Speech data is too large to cache effectively
    ):
        """
        Initialize speech transformed dataset.

        Args:
            base_dataset: Base speech dataset
            linear_transform: Linear transformation to apply
            nonlinear_transform: Non-linear transformation to apply
            quantization_transform: Quantization transformation to apply
            lowrank_transform: Low-rank transformation to apply
            cache_transformed: Whether to cache transformed data (not recommended for speech)
        """
        self.base_dataset = base_dataset
        self.linear_transform = linear_transform
        self.nonlinear_transform = nonlinear_transform
        self.quantization_transform = quantization_transform
        self.lowrank_transform = lowrank_transform
        self.cache_transformed = cache_transformed
        self.transformed_cache = {}

        # Move transforms to evaluation mode for consistent transformations
        if linear_transform is not None:
            linear_transform.eval()
        if nonlinear_transform is not None:
            nonlinear_transform.eval()
        if quantization_transform is not None:
            quantization_transform.eval()
        if lowrank_transform is not None:
            lowrank_transform.eval()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        """Get transformed example."""
        # Check cache first
        if self.cache_transformed and idx in self.transformed_cache:
            return self.transformed_cache[idx]

        # Get base example
        example = self.base_dataset[idx]

        # Apply transformations in order
        if self.linear_transform is not None:
            example = self.linear_transform(example)

        if self.nonlinear_transform is not None:
            example = self.nonlinear_transform(example)

        if self.quantization_transform is not None:
            example = self.quantization_transform(example)

        if self.lowrank_transform is not None:
            example = self.lowrank_transform(example)

        # Cache if enabled (not recommended for large speech datasets)
        if self.cache_transformed:
            self.transformed_cache[idx] = example

        return example


def create_speech_transformed_dataloaders(
    base_train_dataset: Dataset,
    base_valid_dataset: Dataset,
    base_test_dataset: Dataset,
    vocab_size: int,
    feature_dim: Optional[int] = None,
    train_batch_size: int = 8,
    eval_batch_size: int = 16,
    num_workers: int = 4,
    # Linear transform parameters
    apply_linear: bool = False,
    linear_seed: int = 42,
    # Nonlinear transform parameters
    apply_nonlinear: bool = False,
    nonlinear_hidden_dims: Optional[List[int]] = None,
    nonlinear_type: str = "relu",
    nonlinear_dropout_rate: float = 0.1,
    nonlinear_noise_scale: float = 0.05,
    nonlinear_seed: int = 43,
    # Quantization transform parameters
    apply_quantization: bool = False,
    quantization_num_levels: int = 16,
    quantize_features: bool = False,
    quantization_seed: int = 46,
    # Low-rank transform parameters
    apply_lowrank: bool = False,
    lowrank_rank: int = 16,
    lowrank_transform_features: bool = False,
    lowrank_seed: int = 47,
    # Caching parameters
    cache_dir: str = "./dataset_cache",
    use_cache: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Create speech transformed dataloaders.

    Args:
        base_train_dataset: Base training dataset
        base_valid_dataset: Base validation dataset
        base_test_dataset: Base test dataset
        vocab_size: Size of CTC vocabulary
        feature_dim: Dimension of audio features
        train_batch_size: Training batch size
        eval_batch_size: Evaluation batch size
        num_workers: Number of data loading workers
        apply_linear: Whether to apply linear transformation
        linear_seed: Seed for linear transformation
        apply_nonlinear: Whether to apply nonlinear transformation
        nonlinear_hidden_dims: Hidden dimensions for nonlinear transformation
        nonlinear_type: Type of nonlinearity
        nonlinear_dropout_rate: Dropout rate for nonlinear transformation
        nonlinear_noise_scale: Noise scale for nonlinear transformation
        nonlinear_seed: Seed for nonlinear transformation
        apply_quantization: Whether to apply quantization transformation
        quantization_num_levels: Number of quantization levels
        quantize_features: Whether to quantize features
        quantization_seed: Seed for quantization transformation
        apply_lowrank: Whether to apply low-rank transformation
        lowrank_rank: Rank for low-rank transformation
        lowrank_transform_features: Whether to transform features with low-rank
        lowrank_seed: Seed for low-rank transformation

    Returns:
        Tuple of (train_loader, valid_loader, test_loader, transforms_dict)
    """
    # Create transformations
    linear_transform = None
    nonlinear_transform = None
    quantization_transform = None
    lowrank_transform = None

    if apply_linear:
        linear_transform = SpeechLinearTransform(
            vocab_size=vocab_size,
            feature_dim=feature_dim,
            seed=linear_seed,
        )
        linear_transform = linear_transform.cpu()

    if apply_nonlinear:
        nonlinear_transform = SpeechNonLinearTransform(
            vocab_size=vocab_size,
            feature_dim=feature_dim,
            hidden_dims=nonlinear_hidden_dims,
            nonlinear_type=nonlinear_type,
            dropout_rate=nonlinear_dropout_rate,
            noise_scale=nonlinear_noise_scale,
            seed=nonlinear_seed,
        )
        nonlinear_transform = nonlinear_transform.cpu()

    if apply_quantization:
        quantization_transform = SpeechQuantizationTransform(
            vocab_size=vocab_size,
            num_levels=quantization_num_levels,
            feature_dim=feature_dim,
            quantize_features=quantize_features,
            seed=quantization_seed,
        )
        quantization_transform = quantization_transform.cpu()

    if apply_lowrank:
        lowrank_transform = SpeechLowRankTransform(
            vocab_size=vocab_size,
            rank=lowrank_rank,
            feature_dim=feature_dim,
            transform_features=lowrank_transform_features,
            seed=lowrank_seed,
        )
        lowrank_transform = lowrank_transform.cpu()

    # Initialize cache for transforms
    cache = get_dataset_cache(cache_dir) if use_cache else None

    # Helper function to create or load cached transformed dataset
    def create_or_load_transformed_dataset(base_dataset: Dataset, dataset_type: str) -> SpeechTransformedDataset:
        # Create cache parameters for transforms
        transform_params = {
            'vocab_size': vocab_size,
            'feature_dim': feature_dim,
            'dataset_type': dataset_type,
            'dataset_length': len(base_dataset),
            'apply_linear': apply_linear,
            'apply_nonlinear': apply_nonlinear,
            'apply_quantization': apply_quantization,
            'apply_lowrank': apply_lowrank,
            'linear_seed': linear_seed if apply_linear else None,
            'nonlinear_seed': nonlinear_seed if apply_nonlinear else None,
            'nonlinear_hidden_dims': str(nonlinear_hidden_dims) if apply_nonlinear else None,
            'nonlinear_type': nonlinear_type if apply_nonlinear else None,
            'quantization_num_levels': quantization_num_levels if apply_quantization else None,
            'quantization_seed': quantization_seed if apply_quantization else None,
            'lowrank_rank': lowrank_rank if apply_lowrank else None,
            'lowrank_seed': lowrank_seed if apply_lowrank else None,
        }

        # Try to load from cache first
        if cache and cache.cache_exists('transforms', **transform_params):
            print(f"Loading {dataset_type} transformed dataset from cache...")
            dataset = cache.load_dataset('transforms', **transform_params)
            if dataset is not None:
                print(f"Successfully loaded {dataset_type} transformed dataset from cache")
                return dataset
            else:
                print(f"Cache loading failed for {dataset_type} transforms, creating new dataset...")

        # Create new transformed dataset
        print(f"Creating new {dataset_type} transformed dataset...")
        dataset = SpeechTransformedDataset(
            base_dataset,
            linear_transform=linear_transform,
            nonlinear_transform=nonlinear_transform,
            quantization_transform=quantization_transform,
            lowrank_transform=lowrank_transform,
            cache_transformed=False,  # We handle caching at the dataset level
        )

        # Save to cache for future use
        if cache:
            print(f"Saving {dataset_type} transformed dataset to cache...")
            cache.save_dataset(dataset, 'transforms', **transform_params)

        return dataset

    # Create transformed datasets with caching
    train_dataset = create_or_load_transformed_dataset(base_train_dataset, 'train')
    valid_dataset = create_or_load_transformed_dataset(base_valid_dataset, 'valid')
    test_dataset = create_or_load_transformed_dataset(base_test_dataset, 'test')

    # Get collate function from base dataset
    # Handle both single datasets and ConcatDataset
    if hasattr(base_train_dataset, 'collate_fn'):
        collate_fn = base_train_dataset.collate_fn
    elif hasattr(base_train_dataset, 'datasets') and base_train_dataset.datasets:
        # ConcatDataset case - get collate_fn from first underlying dataset
        collate_fn = getattr(base_train_dataset.datasets[0], 'collate_fn', None)
    else:
        collate_fn = None

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    # Create transforms dictionary
    transforms_dict = {
        'linear': linear_transform,
        'nonlinear': nonlinear_transform,
        'quantization': quantization_transform,
        'lowrank': lowrank_transform,
    }

    return train_loader, valid_loader, test_loader, transforms_dict