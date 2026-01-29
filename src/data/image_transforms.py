"""Image-specific transformations for scaling law experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from tqdm import tqdm
import math

# Optional imports
try:
    from sklearn.cluster import KMeans, MiniBatchKMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    KMeans = None
    MiniBatchKMeans = None

try:
    import torchvision.models as models
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
    models = None

import warnings


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
    """
    Feature Space Quantization Transform for genuine information loss.

    This transform uses a pretrained feature extractor to map images to a semantic
    feature space, then quantizes those features using k-means clustering. This is
    highly analogous to text token quantization:

    Text analogy:
        sentence → tokens → vocabulary indices (discrete, e.g., 50K → 256)

    Image analogy:
        image → features → cluster indices (discrete, e.g., ∞ → 1000)

    The key insight is that we quantize in a LEARNED SEMANTIC SPACE rather than
    pixel space or patch space. This creates information loss similar to reducing
    vocabulary size in text experiments.

    Process:
    1. Extract semantic features using pretrained model (e.g., ResNet-18)
    2. Cluster features into K discrete clusters (analogous to vocabulary)
    3. Map each image to its nearest cluster
    4. Reconstruct image from cluster prototype + controlled noise

    This creates genuine information loss because:
    - Many images map to the same cluster (many-to-one mapping)
    - Fine-grained visual details are lost
    - Only coarse semantic information is preserved
    """

    def __init__(
        self,
        num_clusters: int = 1000,
        feature_dim: int = 512,
        feature_extractor_name: str = "resnet18",
        pretrained: bool = True,
        noise_scale: float = 0.05,
        use_adaptive_noise: bool = True,
        freeze_extractor: bool = True,
        seed: int = 46,
    ):
        """
        Initialize feature space quantization transformation.

        Args:
            num_clusters: Number of clusters (analogous to vocabulary size)
                         - 1000 clusters: balanced (10-bit quantization)
                         - 500 clusters: more aggressive (9-bit)
                         - 2000 clusters: less aggressive (11-bit)
            feature_dim: Dimension of feature vectors from extractor
                        - ResNet-18/34: 512
                        - ResNet-50/101: 2048
            feature_extractor_name: Name of pretrained model
                                   - "resnet18": Fast, 512-dim features
                                   - "resnet34": Better quality, 512-dim
                                   - "resnet50": Best quality, 2048-dim (slower)
            pretrained: Use pretrained weights (True recommended)
            noise_scale: Scale of noise added during reconstruction
            use_adaptive_noise: Adapt noise to cluster variance
            freeze_extractor: Freeze feature extractor weights (recommended)
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.num_clusters = num_clusters
        self.feature_dim = feature_dim
        self.feature_extractor_name = feature_extractor_name
        self.noise_scale = noise_scale
        self.use_adaptive_noise = use_adaptive_noise
        self.freeze_extractor = freeze_extractor

        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Initialize feature extractor
        self.feature_extractor = self._create_feature_extractor(
            feature_extractor_name, pretrained
        )

        # Freeze feature extractor if requested
        if freeze_extractor:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            self.feature_extractor.eval()

        # Cluster centers (will be initialized during first forward pass or explicit init)
        self.register_buffer('cluster_centers', torch.zeros(num_clusters, feature_dim))
        self.register_buffer('cluster_initialized', torch.tensor(False))

        # Track cluster statistics
        self.register_buffer('cluster_counts', torch.zeros(num_clusters))
        self.register_buffer('cluster_variances', torch.ones(num_clusters))

        # For mapping features back to images (simple linear decoder)
        # This is a simplification - in practice, you might use a more sophisticated decoder
        self.use_reconstruction = True
        if self.use_reconstruction:
            # Simple decoder: feature → image
            # Note: This is a lossy process by design
            self.decoder = nn.Sequential(
                nn.Linear(feature_dim, 2048),
                nn.ReLU(),
                nn.Linear(2048, 4096),
                nn.ReLU(),
                nn.Linear(4096, 3 * 224 * 224),  # Standard ImageNet size
                nn.Tanh()
            )
            # Initialize decoder weights
            for m in self.decoder.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _create_feature_extractor(self, name: str, pretrained: bool) -> nn.Module:
        """Create feature extractor from torchvision models."""
        if not HAS_TORCHVISION:
            raise ImportError("torchvision is required for feature extraction")

        # Load pretrained model
        if name == "resnet18":
            model = models.resnet18(pretrained=pretrained)
            feature_dim_actual = 512
        elif name == "resnet34":
            model = models.resnet34(pretrained=pretrained)
            feature_dim_actual = 512
        elif name == "resnet50":
            model = models.resnet50(pretrained=pretrained)
            feature_dim_actual = 2048
        elif name == "resnet101":
            model = models.resnet101(pretrained=pretrained)
            feature_dim_actual = 2048
        else:
            raise ValueError(f"Unknown feature extractor: {name}")

        # Verify feature dimension matches
        if feature_dim_actual != self.feature_dim:
            warnings.warn(
                f"Feature dim mismatch: expected {self.feature_dim}, "
                f"got {feature_dim_actual}. Adjusting."
            )
            self.feature_dim = feature_dim_actual

        # Remove classification head, keep only feature extractor
        # For ResNet, remove the final FC layer
        feature_extractor = nn.Sequential(*list(model.children())[:-1])

        return feature_extractor

    def initialize_clusters(self, dataset, num_samples: int = 10000):
        """
        Initialize cluster centers using k-means on sampled data.

        Args:
            dataset: Dataset to sample from
            num_samples: Number of samples to use for clustering
        """
        print(f"Initializing {self.num_clusters} clusters from {num_samples} samples...")

        # Extract features from samples
        features = []
        self.feature_extractor.eval()

        with torch.no_grad():
            for i in range(min(num_samples, len(dataset))):
                # Get image from dataset
                item = dataset[i]
                if isinstance(item, tuple):
                    image = item[0]
                else:
                    image = item.get('x', item.get('image'))

                # Add batch dimension if needed
                if image.dim() == 3:
                    image = image.unsqueeze(0)

                # Extract features
                feat = self._extract_features(image)
                features.append(feat.cpu().numpy().flatten())

                if (i + 1) % 1000 == 0:
                    print(f"  Extracted features from {i + 1}/{num_samples} samples")

        features = np.array(features)
        print(f"Feature shape: {features.shape}")

        # Perform k-means clustering
        if HAS_SKLEARN:
            print("Running MiniBatch K-Means clustering...")
            kmeans = MiniBatchKMeans(
                n_clusters=self.num_clusters,
                random_state=42,
                batch_size=256,
                max_iter=100,
                n_init=3,
                verbose=1
            )
            kmeans.fit(features)

            # Store cluster centers
            self.cluster_centers.copy_(torch.from_numpy(kmeans.cluster_centers_).float())

            # Compute initial cluster statistics
            labels = kmeans.labels_
            for k in range(self.num_clusters):
                mask = labels == k
                self.cluster_counts[k] = mask.sum()
                if mask.sum() > 0:
                    cluster_features = features[mask]
                    variance = np.var(cluster_features, axis=0).mean()
                    self.cluster_variances[k] = variance
        else:
            # Fallback: random initialization
            print("sklearn not available, using random initialization")
            indices = np.random.choice(len(features), self.num_clusters, replace=False)
            self.cluster_centers.copy_(torch.from_numpy(features[indices]).float())
            self.cluster_counts.fill_(len(features) / self.num_clusters)
            self.cluster_variances.fill_(1.0)

        self.cluster_initialized.fill_(True)
        print(f"Cluster initialization complete!")
        print(f"  Active clusters: {(self.cluster_counts > 0).sum().item()}/{self.num_clusters}")
        print(f"  Avg cluster size: {self.cluster_counts.mean().item():.1f}")

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from images using the feature extractor."""
        # Ensure input is in correct format: (B, C, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(0)

        # Normalize if needed (ImageNet normalization)
        # Most pretrained models expect ImageNet-normalized inputs
        if x.max() > 2.0:  # Likely in [0, 255] range
            x = x / 255.0

        # Apply ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x = (x - mean) / std

        # Extract features
        with torch.no_grad() if self.freeze_extractor else torch.enable_grad():
            features = self.feature_extractor(x)

        # Flatten spatial dimensions
        features = features.view(features.size(0), -1)

        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply feature space quantization to images.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Quantized tensor with same shape as input
        """
        if not self.cluster_initialized:
            raise RuntimeError(
                "Clusters not initialized! Call initialize_clusters(dataset) first."
            )

        original_shape = x.shape
        device = x.device

        # Ensure cluster centers are on the same device
        if self.cluster_centers.device != device:
            self.cluster_centers = self.cluster_centers.to(device)
            self.cluster_counts = self.cluster_counts.to(device)
            self.cluster_variances = self.cluster_variances.to(device)

        # Extract features
        features = self._extract_features(x)  # (B, feature_dim)

        # Find nearest cluster for each image
        distances = torch.cdist(features, self.cluster_centers)  # (B, num_clusters)
        cluster_indices = torch.argmin(distances, dim=1)  # (B,)

        # Update cluster statistics (only during training)
        if self.training:
            with torch.no_grad():
                for idx in cluster_indices:
                    self.cluster_counts[idx] += 1

        # Get quantized features (cluster centers)
        quantized_features = self.cluster_centers[cluster_indices]  # (B, feature_dim)

        # Add noise based on cluster variance (optional)
        if self.training and self.noise_scale > 0:
            if self.use_adaptive_noise:
                # Noise scaled by cluster variance
                noise_std = torch.sqrt(self.cluster_variances[cluster_indices]).unsqueeze(1)
                noise = torch.randn_like(quantized_features) * noise_std * self.noise_scale
            else:
                # Fixed noise
                noise = torch.randn_like(quantized_features) * self.noise_scale

            quantized_features = quantized_features + noise

        # Reconstruct images from quantized features
        if self.use_reconstruction and hasattr(self, 'decoder'):
            # Use learned decoder
            self.decoder = self.decoder.to(device)
            reconstructed = self.decoder(quantized_features)
            reconstructed = reconstructed.view(original_shape)
        else:
            # Fallback: blend original with noise (preserves more structure)
            # This is a compromise between information loss and maintaining visual quality
            alpha = 0.7  # Weight for original image
            noise_image = torch.randn_like(x) * 0.2
            reconstructed = alpha * x + (1 - alpha) * noise_image

        return reconstructed

    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get statistics about cluster usage."""
        total_assignments = self.cluster_counts.sum()

        if total_assignments == 0:
            return {
                'active_clusters': 0,
                'utilization_rate': 0.0,
                'avg_cluster_size': 0.0,
                'cluster_entropy': 0.0,
                'min_cluster_size': 0,
                'max_cluster_size': 0
            }

        active_clusters = (self.cluster_counts > 0).sum().item()

        # Compute entropy of cluster distribution
        probs = self.cluster_counts / total_assignments
        probs = probs[probs > 0]  # Only non-zero probabilities
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()

        return {
            'active_clusters': active_clusters,
            'utilization_rate': active_clusters / self.num_clusters,
            'avg_cluster_size': self.cluster_counts.mean().item(),
            'cluster_entropy': entropy,
            'min_cluster_size': self.cluster_counts.min().item(),
            'max_cluster_size': self.cluster_counts.max().item(),
            'total_assignments': int(total_assignments.item())
        }

    def get_compression_ratio(self) -> float:
        """
        Get effective compression ratio.

        For an image:
        - Original: 3 × 224 × 224 = 150,528 continuous values (32-bit) = 4,816,896 bits
        - Quantized: 1 cluster index from K clusters = log2(K) bits
        - Compression: log2(K) / (3 × 224 × 224 × 32)

        For K=1000: log2(1000) / 4,816,896 ≈ 0.000002 (extreme compression!)
        """
        bits_per_image_original = 3 * 224 * 224 * 32  # Assuming 224x224 RGB, 32-bit float
        bits_per_image_quantized = np.log2(self.num_clusters)
        return bits_per_image_quantized / bits_per_image_original


class ImagePixelQuantizationTransform(nn.Module):
    """
    DEPRECATED: Legacy pixel-level quantization (not recommended for scaling law experiments).

    This class is kept for backward compatibility but is NOT recommended for studying
    neural scaling laws because pixel-level quantization does not create sufficient
    information loss for vision tasks.

    Use ImageQuantizationTransform (patch-level) instead.
    """

    def __init__(
        self,
        num_levels: int = 16,
        transform_type: str = "pixel",  # "patch", "pixel", "channel"
        quantization_mode: str = "uniform",  # "uniform", "learned"
        seed: int = 46,
    ):
        """Initialize legacy pixel quantization transformation."""
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
        """Apply quantization transformation to image data."""
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
        # NEW: Use feature-space quantization by default (most effective for scaling laws)
        # For backward compatibility, check if legacy pixel-level is explicitly requested
        if kwargs.get("use_legacy_pixel_quantization", False):
            return ImagePixelQuantizationTransform(
                num_levels=kwargs.get("num_levels", 16),
                transform_type=kwargs.get("transform_type", "pixel"),
                quantization_mode=kwargs.get("quantization_mode", "uniform"),
                seed=kwargs.get("seed", 46),
            )
        else:
            # Use new feature-space quantization (recommended)
            return ImageQuantizationTransform(
                num_clusters=kwargs.get("num_clusters", 1000),
                feature_dim=kwargs.get("feature_dim", 512),
                feature_extractor_name=kwargs.get("feature_extractor_name", "resnet18"),
                pretrained=kwargs.get("pretrained", True),
                noise_scale=kwargs.get("noise_scale", 0.05),
                use_adaptive_noise=kwargs.get("use_adaptive_noise", True),
                freeze_extractor=kwargs.get("freeze_extractor", True),
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