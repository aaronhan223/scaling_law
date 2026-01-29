"""
Feature Space Quantization for Image Scaling Law Experiments.

This module implements semantic feature-space quantization that is highly analogous
to text token quantization, creating genuine information loss for studying neural
scaling laws.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any
import warnings

# Optional imports
try:
    from sklearn.cluster import MiniBatchKMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    MiniBatchKMeans = None
    warnings.warn("sklearn not available. Feature space quantization will use simplified clustering.")

try:
    import torchvision.models as models
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
    warnings.warn("torchvision not available. Cannot use pretrained feature extractors.")


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
