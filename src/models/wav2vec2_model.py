"""Wav2Vec2 model for speech recognition scaling law experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Wav2Vec2Config,
    Wav2Vec2Model,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor
)
from typing import Dict, List, Optional, Tuple, Any
import math


class ScalableWav2Vec2ForCTC(nn.Module):
    """Scalable Wav2Vec2 model for CTC-based speech recognition."""

    def __init__(
        self,
        config: Dict[str, Any],
        vocab_size: int = 32,  # Default vocab size for CTC
        gradient_checkpointing: bool = False,
    ):
        """
        Initialize scalable Wav2Vec2 model.

        Args:
            config: Model configuration dict with scaling parameters
            vocab_size: Vocabulary size for CTC head
            gradient_checkpointing: Enable gradient checkpointing for memory efficiency
        """
        super().__init__()

        self.config = config
        self.vocab_size = vocab_size

        # Create Wav2Vec2Config based on scaling parameters
        wav2vec2_config = Wav2Vec2Config(
            vocab_size=vocab_size,
            hidden_size=config.get('hidden_size', 768),
            num_hidden_layers=config.get('num_hidden_layers', 12),
            num_attention_heads=config.get('num_attention_heads', 12),
            intermediate_size=config.get('intermediate_size', 3072),
            hidden_dropout=config.get('hidden_dropout', 0.1),
            attention_dropout=config.get('attention_dropout', 0.1),
            feat_proj_dropout=config.get('feat_proj_dropout', 0.0),
            feat_quantizer_dropout=config.get('feat_quantizer_dropout', 0.0),
            final_dropout=config.get('final_dropout', 0.1),
            layerdrop=config.get('layerdrop', 0.1),
            conv_dim=config.get('conv_dim', (512, 512, 512, 512, 512, 512, 512)),
            conv_stride=config.get('conv_stride', (5, 2, 2, 2, 2, 2, 2)),
            conv_kernel=config.get('conv_kernel', (10, 3, 3, 3, 3, 2, 2)),
            conv_bias=config.get('conv_bias', False),
            num_conv_pos_embeddings=config.get('num_conv_pos_embeddings', 128),
            num_conv_pos_embedding_groups=config.get('num_conv_pos_embedding_groups', 16),
            len_codevector_dim=config.get('codevector_dim', 256),
            num_codevector_groups=config.get('num_codevector_groups', 2),
            num_codevectors_per_group=config.get('num_codevectors_per_group', 320),
            contrastive_logits_temperature=config.get('contrastive_logits_temperature', 0.1),
            num_negatives=config.get('num_negatives', 100),
            codevector_dim=config.get('codevector_dim', 256),
            proj_codevector_dim=config.get('proj_codevector_dim', 256),
            diversity_loss_weight=config.get('diversity_loss_weight', 0.1),
            ctc_loss_reduction=config.get('ctc_loss_reduction', "mean"),
            ctc_zero_infinity=config.get('ctc_zero_infinity', False),
            use_weighted_layer_sum=config.get('use_weighted_layer_sum', False),
            classifier_proj_size=config.get('classifier_proj_size', 256),
            pad_token_id=config.get('pad_token_id', 0),
            bos_token_id=config.get('bos_token_id', 1),
            eos_token_id=config.get('eos_token_id', 2),
            gradient_checkpointing=gradient_checkpointing,
        )

        # Initialize the Wav2Vec2 model
        self.wav2vec2 = Wav2Vec2ForCTC(wav2vec2_config)

        # Store config for parameter counting
        self.model_config = wav2vec2_config

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initialize model weights."""
        # The Wav2Vec2 model handles its own weight initialization
        pass

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_values: Raw audio waveform [batch_size, sequence_length]
            attention_mask: Attention mask [batch_size, sequence_length]
            labels: CTC labels [batch_size, target_length]

        Returns:
            Dictionary containing loss, logits, and other outputs
        """
        outputs = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

        result = {
            'logits': outputs.logits,
        }

        if labels is not None:
            result['loss'] = outputs.loss

        # Note: For CTC, accuracy calculation requires special decoding
        # We'll compute accuracy at the evaluation level using decoded predictions
        # Setting accuracy to 0.0 for now as it's computed separately in trainer
        if labels is not None and outputs.logits is not None:
            # Create accuracy tensor with proper batch dimension to avoid DataParallel warnings
            batch_size = outputs.logits.shape[0]
            result['accuracy'] = torch.zeros(batch_size, device=outputs.logits.device)

        return result


    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_feature_extractor(self):
        """Freeze the convolutional feature extractor."""
        self.wav2vec2.wav2vec2.feature_extractor._freeze_parameters()

    def unfreeze_feature_extractor(self):
        """Unfreeze the convolutional feature extractor."""
        for param in self.wav2vec2.wav2vec2.feature_extractor.parameters():
            param.requires_grad = True

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.wav2vec2.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.wav2vec2.gradient_checkpointing_disable()

    def get_feature_extractor_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute the output length of the convolutional layers.

        Args:
            input_lengths: Tensor of input sequence lengths

        Returns:
            Tensor of output sequence lengths
        """
        return self.wav2vec2.wav2vec2._get_feat_extract_output_lengths(input_lengths)


def create_wav2vec2_model_configs() -> List[Dict[str, Any]]:
    """
    Create model configurations for different scales.

    Returns:
        List of model configuration dictionaries
    """
    configs = [
        # Tiny model (~1.5M params)
        {
            'model_name': 'wav2vec2_tiny',
            'hidden_size': 256,
            'num_hidden_layers': 4,
            'num_attention_heads': 4,
            'intermediate_size': 1024,
            'conv_dim': (256, 256, 256, 256),
            'conv_stride': (5, 2, 2, 2),
            'conv_kernel': (10, 3, 3, 3),
            'classifier_proj_size': 128,
            'codevector_dim': 128,
            'proj_codevector_dim': 128,
        },

        # Small model (~5M params)
        {
            'model_name': 'wav2vec2_small',
            'hidden_size': 384,
            'num_hidden_layers': 6,
            'num_attention_heads': 6,
            'intermediate_size': 1536,
            'conv_dim': (384, 384, 384, 384, 384),
            'conv_stride': (5, 2, 2, 2, 2),
            'conv_kernel': (10, 3, 3, 3, 2),
            'classifier_proj_size': 192,
            'codevector_dim': 192,
            'proj_codevector_dim': 192,
        },

        # Medium model (~15M params)
        {
            'model_name': 'wav2vec2_medium',
            'hidden_size': 512,
            'num_hidden_layers': 8,
            'num_attention_heads': 8,
            'intermediate_size': 2048,
            'conv_dim': (512, 512, 512, 512, 512, 512),
            'conv_stride': (5, 2, 2, 2, 2, 2),
            'conv_kernel': (10, 3, 3, 3, 3, 2),
            'classifier_proj_size': 256,
            'codevector_dim': 256,
            'proj_codevector_dim': 256,
        },

        # Large model (~30M params)
        {
            'model_name': 'wav2vec2_large',
            'hidden_size': 640,
            'num_hidden_layers': 10,
            'num_attention_heads': 10,
            'intermediate_size': 2560,
            'conv_dim': (512, 512, 512, 512, 512, 512, 512),
            'conv_stride': (5, 2, 2, 2, 2, 2, 2),
            'conv_kernel': (10, 3, 3, 3, 3, 2, 2),
            'classifier_proj_size': 320,
            'codevector_dim': 320,
            'proj_codevector_dim': 320,
        },

        # Extra Large model (~95M params - similar to base wav2vec2)
        {
            'model_name': 'wav2vec2_xlarge',
            'hidden_size': 768,
            'num_hidden_layers': 12,
            'num_attention_heads': 12,
            'intermediate_size': 3072,
            'conv_dim': (512, 512, 512, 512, 512, 512, 512),
            'conv_stride': (5, 2, 2, 2, 2, 2, 2),
            'conv_kernel': (10, 3, 3, 3, 3, 2, 2),
            'classifier_proj_size': 256,
            'codevector_dim': 256,
            'proj_codevector_dim': 256,
        },
    ]

    # Calculate approximate number of parameters for each config
    for config in configs:
        config['num_parameters'] = estimate_wav2vec2_params(config)

    return configs


def estimate_wav2vec2_params(config: Dict[str, Any]) -> int:
    """
    Estimate the number of parameters for a Wav2Vec2 configuration.

    Args:
        config: Model configuration dictionary

    Returns:
        Estimated number of parameters
    """
    hidden_size = config['hidden_size']
    num_layers = config['num_hidden_layers']
    intermediate_size = config['intermediate_size']
    num_heads = config['num_attention_heads']
    conv_dims = config['conv_dim']
    vocab_size = 32  # Default CTC vocab size

    # Convolutional feature extractor parameters
    conv_params = 0
    prev_dim = 1  # Input is raw audio (1 channel)
    for conv_dim in conv_dims:
        kernel_size = 10 if prev_dim == 1 else 3  # First layer has larger kernel
        conv_params += prev_dim * conv_dim * kernel_size
        conv_params += conv_dim  # Bias terms
        prev_dim = conv_dim

    # Feature projection parameters
    feat_proj_params = conv_dims[-1] * hidden_size + hidden_size

    # Positional embedding parameters
    pos_emb_params = hidden_size * 128  # num_conv_pos_embeddings

    # Transformer encoder parameters
    encoder_params = 0
    for _ in range(num_layers):
        # Self-attention
        qkv_params = 3 * hidden_size * hidden_size
        out_params = hidden_size * hidden_size
        attention_params = qkv_params + out_params

        # Feed-forward
        ff_params = hidden_size * intermediate_size + intermediate_size * hidden_size
        ff_bias = intermediate_size + hidden_size

        # Layer norm (2 per layer)
        ln_params = 2 * hidden_size * 2  # weight + bias

        encoder_params += attention_params + ff_params + ff_bias + ln_params

    # Final layer norm
    final_ln_params = hidden_size * 2

    # CTC head parameters
    ctc_head_params = hidden_size * vocab_size + vocab_size

    # Additional components (quantizer, etc.)
    quantizer_params = hidden_size * config.get('codevector_dim', 256) * 2

    total_params = (
        conv_params +
        feat_proj_params +
        pos_emb_params +
        encoder_params +
        final_ln_params +
        ctc_head_params +
        quantizer_params
    )

    return int(total_params)


def create_wav2vec2_model(config: Dict[str, Any], vocab_size: int = 32) -> ScalableWav2Vec2ForCTC:
    """
    Create a Wav2Vec2 model with the given configuration.

    Args:
        config: Model configuration dictionary
        vocab_size: Vocabulary size for CTC head

    Returns:
        Wav2Vec2 model instance
    """
    model = ScalableWav2Vec2ForCTC(
        config=config,
        vocab_size=vocab_size,
        gradient_checkpointing=config.get('gradient_checkpointing', False)
    )

    print(f"Created {config['model_name']} with {model.get_num_params():,} parameters")

    return model