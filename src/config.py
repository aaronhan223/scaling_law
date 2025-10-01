"""Configuration for scaling law experiments."""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ModelConfig:
    """Configuration for transformer models."""
    vocab_size: int = 50257  # GPT-2 tokenizer vocab size
    max_position_embeddings: int = 1024
    hidden_sizes: List[int] = None  # Will be set based on experiment
    num_attention_heads: List[int] = None
    num_hidden_layers: List[int] = None
    intermediate_size_factor: int = 4
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12
    
    def __post_init__(self):
        # Define model configurations of different sizes
        # Following similar scaling to the paper
        if self.hidden_sizes is None:
            self.hidden_sizes = [16, 32, 64, 128, 256, 512, 640, 768, 1024]
        if self.num_attention_heads is None:
            self.num_attention_heads = [2, 4, 8, 8, 8, 12, 16, 16, 32]
        if self.num_hidden_layers is None:
            self.num_hidden_layers = [2, 4, 6, 8, 10, 12, 24, 36, 48]
    
    def get_model_size_configs(self):
        """Get configurations for models of different sizes."""
        configs = []
        for i, (hidden, heads, layers) in enumerate(zip(
            self.hidden_sizes, 
            self.num_attention_heads, 
            self.num_hidden_layers
        )):
            config = {
                'hidden_size': hidden,
                'num_attention_heads': heads,
                'num_hidden_layers': layers,
                'intermediate_size': hidden * self.intermediate_size_factor,
                'model_name': f'model_size_{i}'
            }
            # Calculate approximate parameter count
            params = self.calculate_params(hidden, heads, layers)
            config['num_parameters'] = params
            configs.append(config)
        return configs
    
    def calculate_params(self, hidden_size, num_heads, num_layers):
        """Approximate parameter count for a transformer model."""
        # Embedding parameters
        embedding_params = self.vocab_size * hidden_size + self.max_position_embeddings * hidden_size
        
        # Transformer block parameters (per layer)
        # Self-attention: Q, K, V, O projections
        attention_params = 4 * hidden_size * hidden_size
        # MLP: 2 linear layers
        mlp_params = 2 * hidden_size * (hidden_size * self.intermediate_size_factor)
        # Layer norms: 2 per transformer block
        ln_params = 4 * hidden_size  # Approximate
        
        block_params = attention_params + mlp_params + ln_params
        
        # Total
        total_params = embedding_params + num_layers * block_params
        return int(total_params)


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-103-v1"
    tokenizer_name: str = "gpt2"
    max_length: int = 1024
    train_batch_size: int = 32
    eval_batch_size: int = 64
    num_workers: int = 4
    
    # Dataset size configurations for scaling experiments
    dataset_sizes: List[int] = None  # Number of tokens
    
    def __post_init__(self):
        if self.dataset_sizes is None:
            # Dataset sizes in tokens (reduced for memory efficiency)
            # Following exponential scaling
            self.dataset_sizes = [
                int(1e5),    # 100K tokens
                int(3e5),    # 300K tokens
                int(1e6),    # 1M tokens
                int(3e6),    # 3M tokens
                int(1e7),    # 10M tokens
                int(3e7),    # 30M tokens
            ]


@dataclass
class TrainingConfig:
    """Configuration for training."""
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    warmup_steps: int = 1000
    max_steps: Optional[int] = None  # Will be set based on dataset size
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    
    gradient_accumulation_steps: int = 1
    fp16: bool = True
    seed: int = 42
    
    output_dir: str = "./outputs"
    
    def get_max_steps_for_tokens(self, num_tokens: int, batch_size: int, seq_length: int):
        """Calculate max training steps based on number of tokens."""
        # Ensure we see each token ~once (single epoch)
        tokens_per_batch = batch_size * seq_length * self.gradient_accumulation_steps
        return int(max(1000, num_tokens // tokens_per_batch))


@dataclass
class TransformConfig:
    """Configuration for data transformations."""
    # Linear transformation
    linear_transform_dim: Optional[int] = None  # If None, use vocab_size
    linear_transform_seed: int = 42
    
    # Non-linear transformation
    nonlinear_type: str = "tanh"  # Options: "tanh", "sigmoid", "relu"
    nonlinear_hidden_dims: Optional[List[int]] = None
    nonlinear_dropout_rate: float = 0.1
    nonlinear_noise_scale: float = 0.1
    nonlinear_seed: int = 43
    
    # Clustering transformation (information-losing)
    clustering_num_clusters: Optional[int] = None  # Default to vocab_size // 4
    clustering_embedding_dim: int = 128
    clustering_seed: int = 44
    
    # Modular transformation (information-losing)
    modular_modulus: Optional[int] = None  # Default to vocab_size // 8
    modular_seed: int = 45
    
    # Quantization transformation (information-losing)
    quantization_num_levels: Optional[int] = None  # Default to vocab_size // 16
    quantization_embedding_dim: int = 64
    quantization_seed: int = 46
    
    # Low-rank square matrix transformation
    lowrank_rank: Optional[int] = None  # Default to vocab_size // 8
    lowrank_seed: int = 47
    
    def get_default_lossy_params(self, vocab_size: int):
        """Get default parameters for lossy transforms based on vocab size."""
        return {
            'clustering_num_clusters': self.clustering_num_clusters or max(vocab_size // 4, 100),
            'modular_modulus': self.modular_modulus or max(vocab_size // 8, 50),
            'quantization_num_levels': self.quantization_num_levels or max(vocab_size // 16, 25),
            'lowrank_rank': self.lowrank_rank or max(vocab_size // 8, 50),
        }


@dataclass
class ExperimentConfig:
    """Main experiment configuration."""
    model_config: ModelConfig = field(default_factory=ModelConfig)
    data_config: DataConfig = field(default_factory=DataConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    transform_config: TransformConfig = field(default_factory=TransformConfig)
    
    # Experiment settings
    experiment_name: str = "scaling_laws"
    use_wandb: bool = True
    wandb_project: str = "scaling-laws"
    wandb_entity: Optional[str] = None
    
    # Which experiments to run
    run_standard: bool = False
    run_transformed: bool = False
    
    # New flags for separate transform experiments
    run_linear_transform: bool = False
    run_nonlinear_transform: bool = False
    
    # New flags for lossy transform experiments (information-losing)
    run_clustering_transform: bool = True
    run_modular_transform: bool = True
    run_quantization_transform: bool = True
    run_lowrank_transform: bool = True
    
    # Compute tracking
    track_compute: bool = True
    
    # Dataset caching and optimization
    use_dataset_cache: bool = True  # Cache datasets between standard and transformed experiments
    use_standard_dataloader_optimization: bool = False  # Use original create_dataloaders for standard experiments
    
    # Reproducibility
    deterministic: bool = True

    # Private attributes for device settings
    _device: Optional[str] = None
    _multi_gpu: Optional[bool] = None

    @property
    def device(self):
        return self._device if self._device is not None else "cuda"

    @property
    def multi_gpu(self):
        return self._multi_gpu if self._multi_gpu is not None else True
    
    def get_output_dir(self, experiment_type: str, model_name: str, dataset_size: int):
        """Get output directory for a specific experiment."""
        return os.path.join(
            self.training_config.output_dir,
            self.experiment_name,
            experiment_type,
            model_name,
            f"data_{dataset_size}"
        ) 