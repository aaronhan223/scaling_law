"""Image-specific transformations for scaling law experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from tqdm import tqdm
import math

# Optional import for sklearn
try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    KMeans = None


class ImageLinearTransform(nn.Module):
    """Random linear transformation for image patches or pixels."""

    def __init__(
        self,
        input_dim: int,
        transform_dim: Optional[int] = None,
        transform_type: str = "patch",  # "patch", "pixel", "channel"
        seed: int = 42,
    ):
        """
        Initialize image linear transformation.

        Args:
            input_dim: Input dimension (e.g., patch_dim, pixel_dim, or channel_dim)
            transform_dim: Dimension of transformation (if None, use input_dim)
            transform_type: Type of transformation ("patch", "pixel", "channel")
            seed: Random seed
        """
        super().__init__()
        self.input_dim = input_dim
        self.transform_dim = transform_dim or input_dim
        self.transform_type = transform_type

        # Set random seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Initialize random transformation matrix
        # Using Xavier initialization
        self.transform_matrix = nn.Parameter(
            torch.randn(input_dim, self.transform_dim) * np.sqrt(2.0 / (input_dim + self.transform_dim))
        )

        # If transform_dim != input_dim, we need a projection back
        if self.transform_dim != input_dim:
            self.projection_matrix = nn.Parameter(
                torch.randn(self.transform_dim, input_dim) * np.sqrt(2.0 / (self.transform_dim + input_dim))
            )
        else:
            self.projection_matrix = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply linear transformation to image data.

        Args:
            x: Input tensor

        Returns:
            Transformed tensor
        """
        original_shape = x.shape

        if self.transform_type == "patch":
            # x shape: (B, num_patches, embed_dim)
            x_transformed = torch.matmul(x, self.transform_matrix)
            if self.projection_matrix is not None:
                x_transformed = torch.matmul(x_transformed, self.projection_matrix)

        elif self.transform_type == "pixel":
            # x shape: (B, C, H, W) -> transform pixel values
            B, C, H, W = x.shape
            x = x.view(B, C, -1)  # (B, C, H*W)
            x_transformed = torch.matmul(x.transpose(1, 2), self.transform_matrix[:C, :])
            if self.projection_matrix is not None:
                x_transformed = torch.matmul(x_transformed, self.projection_matrix[:, :C])
            x_transformed = x_transformed.transpose(1, 2).view(B, C, H, W)

        elif self.transform_type == "channel":
            # x shape: (B, C, H, W) -> transform channels
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
            # Ensure transform matrix matches input channels
            if C != self.input_dim:
                raise ValueError(f"Expected {self.input_dim} channels, got {C}")
            x_transformed = torch.matmul(x, self.transform_matrix)
            if self.projection_matrix is not None:
                x_transformed = torch.matmul(x_transformed, self.projection_matrix)
            x_transformed = x_transformed.permute(0, 3, 1, 2)  # (B, C, H, W)

        else:
            raise ValueError(f"Unknown transform_type: {self.transform_type}")

        return x_transformed


class ImageNonLinearTransform(nn.Module):
    """Multi-layer non-linear transformation for image data."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        transform_type: str = "patch",  # "patch", "pixel", "channel"
        nonlinear_type: str = "gelu",
        dropout_rate: float = 0.1,
        noise_scale: float = 0.05,
        seed: int = 43,
    ):
        """
        Initialize image non-linear transformation.

        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            transform_type: Type of transformation
            nonlinear_type: Type of non-linearity
            dropout_rate: Dropout rate
            noise_scale: Scale for gaussian noise injection
            seed: Random seed
        """
        super().__init__()
        self.input_dim = input_dim
        self.transform_type = transform_type
        self.nonlinear_type = nonlinear_type
        self.dropout_rate = dropout_rate
        self.noise_scale = noise_scale

        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [
                max(input_dim * 2, 256),  # Expand first
                max(input_dim, 128),      # Maintain
                max(input_dim // 2, 64)   # Compress
            ]
        self.hidden_dims = hidden_dims

        # Build multi-layer network
        layers = []
        current_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(current_dim, hidden_dim))

            # Batch normalization for stability
            layers.append(nn.BatchNorm1d(hidden_dim))

            # Non-linearity
            if nonlinear_type == "gelu":
                layers.append(nn.GELU())
            elif nonlinear_type == "relu":
                layers.append(nn.ReLU())
            elif nonlinear_type == "tanh":
                layers.append(nn.Tanh())
            elif nonlinear_type == "swish":
                layers.append(nn.SiLU())
            else:
                raise ValueError(f"Unknown nonlinear_type: {nonlinear_type}")

            # Dropout for regularization
            layers.append(nn.Dropout(dropout_rate))

            current_dim = hidden_dim

        # Final output layer to map back to input_dim
        layers.append(nn.Linear(current_dim, input_dim))

        # Create the network
        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights for better convergence."""
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply non-linear transformation to image data.

        Args:
            x: Input tensor

        Returns:
            Transformed tensor
        """
        original_shape = x.shape

        if self.transform_type == "patch":
            # x shape: (B, num_patches, embed_dim)
            B, N, D = x.shape
            x = x.view(-1, D)  # (B*N, D)

            # Add noise for stochasticity
            if self.training and self.noise_scale > 0:
                noise = torch.randn_like(x) * self.noise_scale
                x = x + noise

            x_transformed = self.network(x)
            x_transformed = x_transformed.view(B, N, D)

        elif self.transform_type == "pixel":
            # x shape: (B, C, H, W)
            B, C, H, W = x.shape
            x = x.view(B, C, -1)  # (B, C, H*W)
            x = x.permute(0, 2, 1).contiguous()  # (B, H*W, C)
            x = x.view(-1, C)  # (B*H*W, C)

            # Add noise
            if self.training and self.noise_scale > 0:
                noise = torch.randn_like(x) * self.noise_scale
                x = x + noise

            x_transformed = self.network(x)
            x_transformed = x_transformed.view(B, H*W, C)
            x_transformed = x_transformed.permute(0, 2, 1).contiguous()  # (B, C, H*W)
            x_transformed = x_transformed.view(B, C, H, W)

        elif self.transform_type == "channel":
            # x shape: (B, C, H, W)
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
            x = x.contiguous().view(-1, C)  # (B*H*W, C)

            # Add noise
            if self.training and self.noise_scale > 0:
                noise = torch.randn_like(x) * self.noise_scale
                x = x + noise

            x_transformed = self.network(x)
            x_transformed = x_transformed.view(B, H, W, C)
            x_transformed = x_transformed.permute(0, 3, 1, 2)  # (B, C, H, W)

        else:
            raise ValueError(f"Unknown transform_type: {self.transform_type}")

        return x_transformed


class ImageQuantizationTransform(nn.Module):
    """Quantization-based information-losing transformation for image data."""

    def __init__(
        self,
        num_levels: int = 16,
        transform_type: str = "pixel",  # "patch", "pixel", "channel"
        quantization_mode: str = "uniform",  # "uniform", "learned"
        seed: int = 46,
    ):
        """
        Initialize image quantization transformation.

        Args:
            num_levels: Number of quantization levels
            transform_type: Type of transformation
            quantization_mode: Quantization mode
            seed: Random seed
        """
        super().__init__()
        self.num_levels = num_levels
        self.transform_type = transform_type
        self.quantization_mode = quantization_mode

        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        if quantization_mode == "uniform":
            # Uniform quantization levels between -1 and 1
            self.quantization_levels = torch.linspace(-1, 1, num_levels)
        elif quantization_mode == "learned":
            # Learnable quantization levels
            self.quantization_levels = nn.Parameter(
                torch.linspace(-1, 1, num_levels) + torch.randn(num_levels) * 0.1
            )
        else:
            raise ValueError(f"Unknown quantization_mode: {quantization_mode}")

    def quantize_values(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize values to nearest levels."""
        # Normalize input to [-1, 1] range
        x_min, x_max = x.min(), x.max()
        if x_max > x_min:
            x_normalized = 2 * (x - x_min) / (x_max - x_min) - 1
        else:
            x_normalized = x

        # Find nearest quantization level for each value
        distances = torch.abs(x_normalized.unsqueeze(-1) - self.quantization_levels.to(x.device))
        quantized_indices = torch.argmin(distances, dim=-1)

        # Map back to quantized values
        quantized_values = self.quantization_levels[quantized_indices].to(x.device)

        # Denormalize back to original range
        if x_max > x_min:
            quantized_values = (quantized_values + 1) / 2 * (x_max - x_min) + x_min

        return quantized_values

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply quantization transformation to image data.

        Args:
            x: Input tensor

        Returns:
            Quantized tensor
        """
        if self.transform_type == "patch":
            # x shape: (B, num_patches, embed_dim)
            return self.quantize_values(x)

        elif self.transform_type == "pixel":
            # x shape: (B, C, H, W)
            return self.quantize_values(x)

        elif self.transform_type == "channel":
            # x shape: (B, C, H, W) -> quantize per channel
            B, C, H, W = x.shape
            quantized = torch.zeros_like(x)
            for c in range(C):
                quantized[:, c, :, :] = self.quantize_values(x[:, c, :, :])
            return quantized

        else:
            raise ValueError(f"Unknown transform_type: {self.transform_type}")


class ImageLowRankTransform(nn.Module):
    """Low-rank matrix transformation for image data."""

    def __init__(
        self,
        input_dim: int,
        rank: int,
        transform_type: str = "patch",  # "patch", "pixel", "channel"
        seed: int = 47,
    ):
        """
        Initialize image low-rank transformation.

        Args:
            input_dim: Input dimension
            rank: Rank of the low-rank decomposition
            transform_type: Type of transformation
            seed: Random seed
        """
        super().__init__()
        self.input_dim = input_dim

        # Ensure valid rank
        if input_dim is None:
            raise ValueError("input_dim cannot be None for low-rank transforms")
        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError(f"input_dim must be a positive integer, got {input_dim}")
        if not isinstance(rank, int) or rank <= 0:
            raise ValueError(f"rank must be a positive integer, got {rank}")

        self.rank = min(rank, input_dim)  # Ensure valid rank
        self.transform_type = transform_type

        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create low-rank decomposition: M = U @ V^T
        self.U = nn.Parameter(
            torch.randn(input_dim, self.rank) * np.sqrt(2.0 / (input_dim + self.rank))
        )
        self.V = nn.Parameter(
            torch.randn(input_dim, self.rank) * np.sqrt(2.0 / (input_dim + self.rank))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply low-rank transformation to image data.

        Args:
            x: Input tensor

        Returns:
            Transformed tensor
        """
        if self.transform_type == "patch":
            # x shape: (B, num_patches, embed_dim)
            # Apply low-rank transformation: x @ (U @ V^T)^T = x @ V @ U^T
            x_transformed = torch.matmul(x, self.V)  # (B, num_patches, rank)
            x_transformed = torch.matmul(x_transformed, self.U.T)  # (B, num_patches, embed_dim)

        elif self.transform_type == "pixel":
            # x shape: (B, C, H, W)
            B, C, H, W = x.shape
            x = x.view(B, C, -1)  # (B, C, H*W)
            x = x.transpose(1, 2)  # (B, H*W, C)
            x_transformed = torch.matmul(x, self.V[:C, :])  # (B, H*W, rank)
            x_transformed = torch.matmul(x_transformed, self.U[:C, :].T)  # (B, H*W, C)
            x_transformed = x_transformed.transpose(1, 2)  # (B, C, H*W)
            x_transformed = x_transformed.view(B, C, H, W)

        elif self.transform_type == "channel":
            # x shape: (B, C, H, W)
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
            original_shape = x.shape
            x = x.view(-1, C)  # (B*H*W, C)
            x_transformed = torch.matmul(x, self.V[:C, :])  # (B*H*W, rank)
            x_transformed = torch.matmul(x_transformed, self.U[:C, :].T)  # (B*H*W, C)
            x_transformed = x_transformed.view(original_shape)
            x_transformed = x_transformed.permute(0, 3, 1, 2)  # (B, C, H, W)

        else:
            raise ValueError(f"Unknown transform_type: {self.transform_type}")

        return x_transformed


class ImageTransformedDataset(Dataset):
    """Dataset wrapper that applies image transformations."""

    def __init__(
        self,
        base_dataset: Dataset,
        linear_transform: Optional[ImageLinearTransform] = None,
        nonlinear_transform: Optional[ImageNonLinearTransform] = None,
        quantization_transform: Optional[ImageQuantizationTransform] = None,
        lowrank_transform: Optional[ImageLowRankTransform] = None,
        cache_transformed: bool = False,  # Caching disabled by default for images due to memory
    ):
        """
        Initialize transformed image dataset.

        Args:
            base_dataset: Base dataset to transform
            linear_transform: Linear transformation to apply
            nonlinear_transform: Non-linear transformation to apply
            quantization_transform: Quantization transformation to apply
            lowrank_transform: Low-rank transformation to apply
            cache_transformed: Whether to cache transformed data (not recommended for large image datasets)
        """
        self.base_dataset = base_dataset
        self.linear_transform = linear_transform
        self.nonlinear_transform = nonlinear_transform
        self.quantization_transform = quantization_transform
        self.lowrank_transform = lowrank_transform
        self.cache_transformed = cache_transformed
        self.transformed_cache = {}

        # Move transforms to CPU to save GPU memory during data loading
        if self.linear_transform is not None:
            self.linear_transform = self.linear_transform.cpu()
        if self.nonlinear_transform is not None:
            self.nonlinear_transform = self.nonlinear_transform.cpu()
        if self.quantization_transform is not None:
            self.quantization_transform = self.quantization_transform.cpu()
        if self.lowrank_transform is not None:
            self.lowrank_transform = self.lowrank_transform.cpu()

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
        if isinstance(example, tuple):
            image, label = example
        else:
            image = example.get("x", example.get("image"))
            label = example.get("labels", example.get("label"))

        # Apply transforms in sequence
        with torch.no_grad():
            if self.linear_transform is not None:
                image = self.linear_transform(image.unsqueeze(0)).squeeze(0)

            if self.nonlinear_transform is not None:
                image = self.nonlinear_transform(image.unsqueeze(0)).squeeze(0)

            if self.quantization_transform is not None:
                image = self.quantization_transform(image.unsqueeze(0)).squeeze(0)

            if self.lowrank_transform is not None:
                image = self.lowrank_transform(image.unsqueeze(0)).squeeze(0)

        result = {"x": image, "labels": label}

        # Cache if enabled
        if self.cache_transformed:
            self.transformed_cache[idx] = result

        return result


def create_image_transform(
    transform_name: str,
    input_dim: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create image transforms.

    Args:
        transform_name: Name of the transform ("linear", "nonlinear", "quantization", "lowrank")
        input_dim: Input dimension for the transform
        **kwargs: Additional parameters for the transform

    Returns:
        Transform module
    """
    if transform_name == "linear":
        return ImageLinearTransform(
            input_dim=input_dim,
            transform_dim=kwargs.get("transform_dim"),
            transform_type=kwargs.get("transform_type", "patch"),
            seed=kwargs.get("seed", 42),
        )

    elif transform_name == "nonlinear":
        return ImageNonLinearTransform(
            input_dim=input_dim,
            hidden_dims=kwargs.get("hidden_dims"),
            transform_type=kwargs.get("transform_type", "patch"),
            nonlinear_type=kwargs.get("nonlinear_type", "gelu"),
            dropout_rate=kwargs.get("dropout_rate", 0.1),
            noise_scale=kwargs.get("noise_scale", 0.05),
            seed=kwargs.get("seed", 43),
        )

    elif transform_name == "quantization":
        return ImageQuantizationTransform(
            num_levels=kwargs.get("num_levels", 16),
            transform_type=kwargs.get("transform_type", "pixel"),
            quantization_mode=kwargs.get("quantization_mode", "uniform"),
            seed=kwargs.get("seed", 46),
        )

    elif transform_name == "lowrank":
        # Handle rank parameter safely
        if input_dim is None:
            raise ValueError("input_dim cannot be None for lowrank transforms")
        rank = kwargs.get("rank", max(1, input_dim // 4))
        return ImageLowRankTransform(
            input_dim=input_dim,
            rank=rank,
            transform_type=kwargs.get("transform_type", "patch"),
            seed=kwargs.get("seed", 47),
        )

    else:
        raise ValueError(f"Unknown transform name: {transform_name}")