"""Data transformations for scaling law experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Dict, Any, List
from tqdm import tqdm
import hashlib

# Optional import for sklearn
try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    KMeans = None


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
    """Multi-layer stochastic non-linear transformation for token embeddings."""
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dims: Optional[List[int]] = None,
        nonlinear_type: str = "tanh",
        dropout_rate: float = 0.1,
        noise_scale: float = 0.1,
        seed: int = 43,
    ):
        """
        Initialize multi-layer non-linear transformation.
        
        Args:
            vocab_size: Size of vocabulary
            hidden_dims: List of hidden layer dimensions (default: [vocab_size*2, vocab_size, vocab_size//2])
            nonlinear_type: Type of non-linearity ('tanh', 'sigmoid', 'relu', 'gelu')
            dropout_rate: Dropout rate for stochastic behavior
            noise_scale: Scale for gaussian noise injection
            seed: Random seed
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.nonlinear_type = nonlinear_type
        self.dropout_rate = dropout_rate
        self.noise_scale = noise_scale
        
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [
                max(vocab_size * 2, 512),  # Expand first
                max(vocab_size, 256),      # Maintain
                max(vocab_size // 2, 128)  # Compress
            ]
        self.hidden_dims = hidden_dims
        
        # Build multi-layer network
        layers = []
        input_dim = vocab_size
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(input_dim, hidden_dim))
            
            # Batch normalization for stability
            layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Non-linearity
            if nonlinear_type == "tanh":
                layers.append(nn.Tanh())
            elif nonlinear_type == "sigmoid":
                layers.append(nn.Sigmoid())
            elif nonlinear_type == "relu":
                layers.append(nn.ReLU())
            elif nonlinear_type == "gelu":
                layers.append(nn.GELU())
            else:
                raise ValueError(f"Unknown nonlinear_type: {nonlinear_type}")
            
            # Dropout for stochasticity
            layers.append(nn.Dropout(dropout_rate))
            
            input_dim = hidden_dim
        
        # Final output layer to map back to vocab_size
        layers.append(nn.Linear(input_dim, vocab_size))
        
        # Create the network
        self.network = nn.Sequential(*layers)
        
        # Initialize weights with Xavier/Glorot initialization
        self._initialize_weights()
        
        # Precompute token mapping
        self._precompute_mapping()
    
    def _initialize_weights(self):
        """Initialize network weights for better convergence."""
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _precompute_mapping(self):
        """Precompute the nonlinear token mapping using the multi-layer network."""
        with torch.no_grad():
            # Set network to evaluation mode for consistent mapping
            self.network.eval()
            
            # Create one-hot identity matrix for all tokens
            eye = torch.eye(self.vocab_size, dtype=torch.float32)
            
            # Add gaussian noise for stochasticity during precomputation
            if self.noise_scale > 0:
                noise = torch.randn_like(eye) * self.noise_scale
                eye = eye + noise
            
            # Pass through the network in batches to avoid memory issues
            batch_size = min(1000, self.vocab_size)
            transformed_outputs = []
            
            for i in range(0, self.vocab_size, batch_size):
                end_idx = min(i + batch_size, self.vocab_size)
                batch_input = eye[i:end_idx]
                
                # Forward pass through network
                batch_output = self.network(batch_input)
                transformed_outputs.append(batch_output)
            
            # Concatenate all outputs
            transformed = torch.cat(transformed_outputs, dim=0)
            
            # Apply softmax to get valid probability distribution
            transformed = F.softmax(transformed, dim=1)
            
            # Create deterministic mapping by taking argmax
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
        Apply multi-layer transformation directly to embeddings.
        
        Args:
            embeddings: Embedding matrix [vocab_size, embedding_dim]
            
        Returns:
            Transformed embeddings
        """
        # Set network to training mode for stochastic behavior
        self.network.train()
        
        with torch.no_grad():
            # Get the transformation matrix from our precomputed mapping
            # This creates a permutation matrix based on token mapping
            vocab_size = embeddings.size(0)
            embedding_dim = embeddings.size(1)
            
            # Create transformation matrix using the token mapping
            transform_matrix = torch.zeros(vocab_size, vocab_size)
            for i in range(vocab_size):
                transform_matrix[i, self.token_mapping[i]] = 1.0
            
            # Apply transformation to embeddings
            transformed_embeddings = torch.matmul(transform_matrix, embeddings)
            
            # Add some gaussian noise for additional stochasticity
            if self.training and self.noise_scale > 0:
                noise = torch.randn_like(transformed_embeddings) * self.noise_scale * 0.1
                transformed_embeddings = transformed_embeddings + noise
        
        return transformed_embeddings



class ClusteringTransform(nn.Module):
    """Information-losing transformation using clustering to map multiple tokens to cluster centers."""
    
    def __init__(
        self,
        vocab_size: int,
        num_clusters: int,
        embedding_dim: int = 128,
        seed: int = 44,
    ):
        """
        Initialize clustering transformation.
        
        Args:
            vocab_size: Size of vocabulary
            num_clusters: Number of clusters (must be < vocab_size for information loss)
            embedding_dim: Dimension for token embeddings used in clustering
            seed: Random seed
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.num_clusters = min(num_clusters, vocab_size)  # Ensure info loss
        self.embedding_dim = embedding_dim
        
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create random embeddings for each token
        self.token_embeddings = torch.randn(vocab_size, embedding_dim)
        
        # Perform clustering
        self._compute_clustering()
        
    def _compute_clustering(self):
        """Compute clustering assignment for tokens."""
        if not HAS_SKLEARN:
            # Fallback to simple random clustering if sklearn is not available
            cluster_assignments = torch.randint(0, self.num_clusters, (self.vocab_size,))
            self.token_mapping = cluster_assignments
            
            # Create simple cluster representatives
            self.cluster_representatives = torch.arange(self.num_clusters) % self.vocab_size
            self.cluster_centers = self.token_embeddings[:self.num_clusters]
        else:
            # Use KMeans to cluster token embeddings
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
            cluster_assignments = kmeans.fit_predict(self.token_embeddings.numpy())
            
            # Create mapping from token to cluster center
            self.token_mapping = torch.tensor(cluster_assignments, dtype=torch.long)
            
            # Store cluster centers as token IDs (map back to vocab space)
            self.cluster_centers = torch.tensor(kmeans.cluster_centers_)
            
            # Find the closest token to each cluster center
            self.cluster_representatives = torch.zeros(self.num_clusters, dtype=torch.long)
            for i in range(self.num_clusters):
                center = self.cluster_centers[i]
                distances = torch.norm(self.token_embeddings - center, dim=1)
                self.cluster_representatives[i] = torch.argmin(distances)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Apply clustering transformation to token IDs.
        
        Args:
            token_ids: Tensor of token IDs
            
        Returns:
            Transformed token IDs (mapped to cluster representatives)
        """
        # Ensure input token IDs are within valid range
        token_ids = torch.clamp(token_ids, 0, self.vocab_size - 1)
        
        # Map tokens to their cluster assignments, then to cluster representatives
        cluster_ids = self.token_mapping[token_ids]
        transformed_ids = self.cluster_representatives[cluster_ids]
        
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
        flat_transformed = self.forward(flat_ids)
        
        # Restore original shape
        return flat_transformed.view(original_shape)
    
    def get_compression_ratio(self) -> float:
        """Get the compression ratio (information loss measure)."""
        return self.num_clusters / self.vocab_size


class ModularTransform(nn.Module):
    """Information-losing transformation using modular arithmetic."""
    
    def __init__(
        self,
        vocab_size: int,
        modulus: int,
        seed: int = 45,
    ):
        """
        Initialize modular transformation.
        
        Args:
            vocab_size: Size of vocabulary
            modulus: Modulus for the transformation (must be < vocab_size for info loss)
            seed: Random seed
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.modulus = min(modulus, vocab_size)  # Ensure info loss
        
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create random offset and scale for more complex modular mapping
        self.offset = torch.randint(0, self.modulus, (1,)).item()
        self.scale = torch.randint(1, max(2, self.modulus // 2), (1,)).item()
        
        # Precompute modular mapping
        self._precompute_mapping()
    
    def _precompute_mapping(self):
        """Precompute the modular token mapping."""
        # Create modular mapping: (scale * token_id + offset) % modulus
        token_ids = torch.arange(self.vocab_size)
        self.token_mapping = ((self.scale * token_ids + self.offset) % self.modulus).long()
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Apply modular transformation to token IDs.
        
        Args:
            token_ids: Tensor of token IDs
            
        Returns:
            Transformed token IDs
        """
        # Ensure input token IDs are within valid range
        token_ids = torch.clamp(token_ids, 0, self.vocab_size - 1)
        
        # Apply modular transformation
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
        flat_transformed = self.forward(flat_ids)
        
        # Restore original shape
        return flat_transformed.view(original_shape)
    
    def get_compression_ratio(self) -> float:
        """Get the compression ratio (information loss measure)."""
        return self.modulus / self.vocab_size


class QuantizationTransform(nn.Module):
    """Information-losing transformation using quantization of token representations."""
    
    def __init__(
        self,
        vocab_size: int,
        num_levels: int,
        embedding_dim: int = 64,
        seed: int = 46,
    ):
        """
        Initialize quantization transformation.
        
        Args:
            vocab_size: Size of vocabulary
            num_levels: Number of quantization levels (must be < vocab_size for info loss)
            embedding_dim: Dimension for token embeddings
            seed: Random seed
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.num_levels = min(num_levels, vocab_size)  # Ensure info loss
        self.embedding_dim = embedding_dim
        
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create random embeddings for each token
        self.token_embeddings = nn.Parameter(
            torch.randn(vocab_size, embedding_dim) * 0.1
        )
        
        # Create quantization levels
        self.quantization_levels = nn.Parameter(
            torch.linspace(-1, 1, num_levels).unsqueeze(1).expand(-1, embedding_dim)
        )
        
        # Precompute quantized mapping
        self._precompute_quantization()
    
    def _precompute_quantization(self):
        """Precompute the quantized token mapping with memory optimization."""
        with torch.no_grad():
            # For each token embedding, find the nearest quantization level
            token_embeddings = self.token_embeddings
            quantization_levels = self.quantization_levels
            
            # Process in chunks to reduce peak memory usage
            chunk_size = min(1000, self.vocab_size)  # Process 1000 tokens at a time
            nearest_levels = torch.zeros(self.vocab_size, dtype=torch.long)
            
            for start_idx in range(0, self.vocab_size, chunk_size):
                end_idx = min(start_idx + chunk_size, self.vocab_size)
                chunk_embeddings = token_embeddings[start_idx:end_idx]
                
                # Compute distances for this chunk
                distances = torch.cdist(chunk_embeddings, quantization_levels)
                nearest_levels[start_idx:end_idx] = torch.argmin(distances, dim=1)
                
                # Clear intermediate tensors to free memory
                del distances, chunk_embeddings
            
            # Map quantization levels back to token IDs
            # Find representative token for each quantization level
            level_representatives = torch.zeros(self.num_levels, dtype=torch.long)
            for level in range(self.num_levels):
                # Find tokens assigned to this level
                tokens_in_level = (nearest_levels == level).nonzero(as_tuple=True)[0]
                if len(tokens_in_level) > 0:
                    # Use the first token in this level as representative
                    level_representatives[level] = tokens_in_level[0]
                else:
                    # If no tokens assigned to this level, use level index % vocab_size
                    level_representatives[level] = level % self.vocab_size
            
            # Create final mapping: token -> quantization level -> representative token
            self.token_mapping = level_representatives[nearest_levels]
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Apply quantization transformation to token IDs.
        
        Args:
            token_ids: Tensor of token IDs
            
        Returns:
            Transformed token IDs
        """
        # Ensure input token IDs are within valid range
        token_ids = torch.clamp(token_ids, 0, self.vocab_size - 1)
        
        # Apply quantized transformation
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
        flat_transformed = self.forward(flat_ids)
        
        # Restore original shape
        return flat_transformed.view(original_shape)
    
    def get_compression_ratio(self) -> float:
        """Get the compression ratio (information loss measure)."""
        return self.num_levels / self.vocab_size 


class LowRankSquareTransform(nn.Module):
    """Low-rank square matrix transformation for token embeddings."""
    
    def __init__(
        self,
        vocab_size: int,
        rank: int,
        seed: int = 47,
    ):
        """
        Initialize low-rank square matrix transformation.
        
        Args:
            vocab_size: Size of vocabulary
            rank: Rank of the low-rank decomposition (must be <= vocab_size)
            seed: Random seed
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.rank = min(rank, vocab_size)  # Ensure valid rank
        
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create low-rank decomposition: M = U @ V^T
        # where U is [vocab_size, rank] and V is [vocab_size, rank]
        # The resulting matrix M is [vocab_size, vocab_size] with rank <= rank
        self.U = nn.Parameter(
            torch.randn(vocab_size, self.rank) * np.sqrt(2.0 / (vocab_size + self.rank))
        )
        self.V = nn.Parameter(
            torch.randn(vocab_size, self.rank) * np.sqrt(2.0 / (vocab_size + self.rank))
        )
        
        # Precompute the transformation matrix and token mapping
        self._precompute_transformation()
    
    def _precompute_transformation(self):
        """Precompute the low-rank transformation matrix and token mapping."""
        with torch.no_grad():
            # Compute the low-rank transformation matrix M = U @ V^T
            self.transformation_matrix = torch.matmul(self.U, self.V.T)
            
            # Apply softmax to get probability distribution over output tokens
            self.token_mapping_probs = F.softmax(self.transformation_matrix, dim=1)
            
            # Create deterministic mapping by taking argmax
            self.token_mapping = torch.argmax(self.token_mapping_probs, dim=1)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Apply low-rank transformation to token IDs using efficient lookup.
        
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
        Apply low-rank transformation directly to embeddings.
        
        Args:
            embeddings: Embedding matrix [vocab_size, embedding_dim]
            
        Returns:
            Transformed embeddings
        """
        # Apply low-rank transformation to embedding matrix
        # M @ embeddings where M = U @ V^T
        # This is equivalent to U @ (V^T @ embeddings)
        intermediate = torch.matmul(self.V.T, embeddings)
        transformed_embeddings = torch.matmul(self.U, intermediate)
        
        return transformed_embeddings
    
    def get_rank(self) -> int:
        """Get the rank of the transformation matrix."""
        return self.rank
    
    def get_compression_ratio(self) -> float:
        """Get the compression ratio (parameter efficiency measure)."""
        # Low-rank matrix uses 2 * vocab_size * rank parameters
        # Full matrix would use vocab_size^2 parameters
        low_rank_params = 2 * self.vocab_size * self.rank
        full_rank_params = self.vocab_size * self.vocab_size
        return low_rank_params / full_rank_params


class TransformedDataset(Dataset):
    """Dataset wrapper that applies transformations with caching."""
    
    def __init__(
        self,
        base_dataset: Dataset,
        linear_transform: Optional[LinearTransform] = None,
        nonlinear_transform: Optional[NonLinearTransform] = None,
        clustering_transform: Optional[ClusteringTransform] = None,
        modular_transform: Optional[ModularTransform] = None,
        quantization_transform: Optional[QuantizationTransform] = None,
        lowrank_transform: Optional[LowRankSquareTransform] = None,
        cache_transformed: bool = True,
        batch_size: int = 1000,
    ):
        """
        Initialize transformed dataset.
        
        Args:
            base_dataset: Base dataset to transform
            linear_transform: Linear transformation to apply
            nonlinear_transform: Non-linear transformation to apply
            clustering_transform: Clustering transformation to apply
            modular_transform: Modular transformation to apply
            quantization_transform: Quantization transformation to apply
            lowrank_transform: Low-rank square matrix transformation to apply
            cache_transformed: Whether to cache transformed data
            batch_size: Batch size for transformation (when caching)
        """
        self.base_dataset = base_dataset
        self.linear_transform = linear_transform
        self.nonlinear_transform = nonlinear_transform
        self.clustering_transform = clustering_transform
        self.modular_transform = modular_transform
        self.quantization_transform = quantization_transform
        self.lowrank_transform = lowrank_transform
        self.cache_transformed = cache_transformed
        self.transformed_cache = {}
        
        # If caching is enabled, pre-transform all data
        transforms_exist = any([
            linear_transform is not None,
            nonlinear_transform is not None,
            clustering_transform is not None,
            modular_transform is not None,
            quantization_transform is not None,
            lowrank_transform is not None
        ])
        if cache_transformed and transforms_exist:
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
                
                if self.clustering_transform is not None:
                    batch_input_ids = self.clustering_transform.forward_batch(batch_input_ids)
                    batch_labels = self.clustering_transform.forward_batch(batch_labels)
                    
                if self.modular_transform is not None:
                    batch_input_ids = self.modular_transform.forward_batch(batch_input_ids)
                    batch_labels = self.modular_transform.forward_batch(batch_labels)
                    
                if self.quantization_transform is not None:
                    batch_input_ids = self.quantization_transform.forward_batch(batch_input_ids)
                    batch_labels = self.quantization_transform.forward_batch(batch_labels)
                    
                if self.lowrank_transform is not None:
                    batch_input_ids = self.lowrank_transform.forward_batch(batch_input_ids)
                    batch_labels = self.lowrank_transform.forward_batch(batch_labels)
                
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
        
        if self.clustering_transform is not None:
            input_ids = self.clustering_transform(input_ids)
            labels = self.clustering_transform(labels)
            
        if self.modular_transform is not None:
            input_ids = self.modular_transform(input_ids)
            labels = self.modular_transform(labels)
            
        if self.quantization_transform is not None:
            input_ids = self.quantization_transform(input_ids)
            labels = self.quantization_transform(labels)
            
        if self.lowrank_transform is not None:
            input_ids = self.lowrank_transform(input_ids)
            labels = self.lowrank_transform(labels)
            
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
    nonlinear_hidden_dims: Optional[List[int]] = None,
    nonlinear_type: str = "tanh",
    nonlinear_dropout_rate: float = 0.1,
    nonlinear_noise_scale: float = 0.1,
    nonlinear_seed: int = 43,
    clustering_num_clusters: Optional[int] = None,
    clustering_embedding_dim: int = 128,
    clustering_seed: int = 44,
    modular_modulus: Optional[int] = None,
    modular_seed: int = 45,
    quantization_num_levels: Optional[int] = None,
    quantization_embedding_dim: int = 64,
    quantization_seed: int = 46,
    lowrank_rank: Optional[int] = None,
    lowrank_seed: int = 47,
    apply_linear: bool = True,
    apply_nonlinear: bool = True,
    apply_clustering: bool = False,
    apply_modular: bool = False,
    apply_quantization: bool = False,
    apply_lowrank: bool = False,
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
        nonlinear_hidden_dims: Hidden layer dimensions for nonlinear transformation
        nonlinear_type: Type of nonlinear transformation
        nonlinear_dropout_rate: Dropout rate for nonlinear transformation
        nonlinear_noise_scale: Noise scale for nonlinear transformation
        nonlinear_seed: Seed for nonlinear transformation
        clustering_num_clusters: Number of clusters for clustering transform
        clustering_embedding_dim: Embedding dimension for clustering
        clustering_seed: Seed for clustering transformation
        modular_modulus: Modulus for modular transformation
        modular_seed: Seed for modular transformation
        quantization_num_levels: Number of quantization levels
        quantization_embedding_dim: Embedding dimension for quantization
        quantization_seed: Seed for quantization transformation
        lowrank_rank: Rank for low-rank square matrix transformation
        lowrank_seed: Seed for low-rank transformation
        apply_linear: Whether to apply linear transformation
        apply_nonlinear: Whether to apply nonlinear transformation
        apply_clustering: Whether to apply clustering transformation
        apply_modular: Whether to apply modular transformation
        apply_quantization: Whether to apply quantization transformation
        apply_lowrank: Whether to apply low-rank transformation
    
    Returns:
        Tuple of (train_loader, valid_loader, test_loader, transforms_dict)
    """
    # Create transformations
    linear_transform = None
    nonlinear_transform = None
    clustering_transform = None
    modular_transform = None
    quantization_transform = None
    lowrank_transform = None
    
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
            hidden_dims=nonlinear_hidden_dims,
            nonlinear_type=nonlinear_type,
            dropout_rate=nonlinear_dropout_rate,
            noise_scale=nonlinear_noise_scale,
            seed=nonlinear_seed,
        )
        # Move transformation to CPU for data loading
        nonlinear_transform = nonlinear_transform.cpu()
    
    if apply_clustering and clustering_num_clusters is not None:
        clustering_transform = ClusteringTransform(
            vocab_size=vocab_size,
            num_clusters=clustering_num_clusters,
            embedding_dim=clustering_embedding_dim,
            seed=clustering_seed,
        )
        # Move transformation to CPU for data loading
        clustering_transform = clustering_transform.cpu()
    
    if apply_modular and modular_modulus is not None:
        modular_transform = ModularTransform(
            vocab_size=vocab_size,
            modulus=modular_modulus,
            seed=modular_seed,
        )
        # Move transformation to CPU for data loading
        modular_transform = modular_transform.cpu()
    
    if apply_quantization and quantization_num_levels is not None:
        quantization_transform = QuantizationTransform(
            vocab_size=vocab_size,
            num_levels=quantization_num_levels,
            embedding_dim=quantization_embedding_dim,
            seed=quantization_seed,
        )
        # Move transformation to CPU for data loading
        quantization_transform = quantization_transform.cpu()
    
    if apply_lowrank and lowrank_rank is not None:
        lowrank_transform = LowRankSquareTransform(
            vocab_size=vocab_size,
            rank=lowrank_rank,
            seed=lowrank_seed,
        )
        # Move transformation to CPU for data loading
        lowrank_transform = lowrank_transform.cpu()
    
    # Create transformed datasets with caching enabled
    train_dataset = TransformedDataset(
        base_train_dataset,
        linear_transform=linear_transform,
        nonlinear_transform=nonlinear_transform,
        clustering_transform=clustering_transform,
        modular_transform=modular_transform,
        quantization_transform=quantization_transform,
        lowrank_transform=lowrank_transform,
        cache_transformed=True,
        batch_size=1000,
    )
    
    valid_dataset = TransformedDataset(
        base_valid_dataset,
        linear_transform=linear_transform,
        nonlinear_transform=nonlinear_transform,
        clustering_transform=clustering_transform,
        modular_transform=modular_transform,
        quantization_transform=quantization_transform,
        lowrank_transform=lowrank_transform,
        cache_transformed=True,
        batch_size=1000,
    )
    
    test_dataset = TransformedDataset(
        base_test_dataset,
        linear_transform=linear_transform,
        nonlinear_transform=nonlinear_transform,
        clustering_transform=clustering_transform,
        modular_transform=modular_transform,
        quantization_transform=quantization_transform,
        lowrank_transform=lowrank_transform,
        cache_transformed=True,
        batch_size=1000,
    )
    
    # Check for empty datasets
    if len(train_dataset) == 0:
        raise ValueError(f"Train dataset is empty. Dataset size: {max_tokens_train}, "
                        f"sequences prepared: {len(train_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    # Check for empty validation dataset
    if len(valid_dataset) == 0:
        print(f"Warning: Validation dataset is empty, creating minimal dataset")
        # Use a small subset of training data for validation
        valid_dataset = train_dataset
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    # Check for empty test dataset
    if len(test_dataset) == 0:
        print(f"Warning: Test dataset is empty, creating minimal dataset")
        # Use a small subset of training data for testing
        test_dataset = train_dataset
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    # Create transforms dictionary
    transforms_dict = {
        'linear': linear_transform,
        'nonlinear': nonlinear_transform,
        'clustering': clustering_transform,
        'modular': modular_transform,
        'quantization': quantization_transform,
        'lowrank': lowrank_transform,
    }
    
    return train_loader, valid_loader, test_loader, transforms_dict 
