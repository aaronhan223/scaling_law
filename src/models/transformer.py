"""Transformer language model implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float = 0.1,
    ):
        super().__init__()
        assert hidden_size % num_attention_heads == 0
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(attention_probs_dropout_prob)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.size()
        
        # Compute Q, K, V
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)
        
        # Reshape for multi-head attention
        query_layer = query_layer.view(
            batch_size, seq_length, self.num_attention_heads, self.attention_head_size
        ).transpose(1, 2)
        key_layer = key_layer.view(
            batch_size, seq_length, self.num_attention_heads, self.attention_head_size
        ).transpose(1, 2)
        value_layer = value_layer.view(
            batch_size, seq_length, self.num_attention_heads, self.attention_head_size
        ).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply attention mask (causal mask for language modeling)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Normalize attention scores
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose(1, 2).contiguous()
        context_layer = context_layer.view(batch_size, seq_length, self.hidden_size)
        
        # Output projection
        output = self.output(context_layer)
        
        return output


class TransformerBlock(nn.Module):
    """Single transformer block."""
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        
        self.attention = MultiHeadAttention(
            hidden_size, num_attention_heads, attention_probs_dropout_prob
        )
        self.attention_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU() if hidden_act == "gelu" else nn.ReLU(),
            nn.Linear(intermediate_size, hidden_size),
        )
        self.mlp_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        self.dropout = nn.Dropout(hidden_dropout_prob)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with residual connection
        residual = hidden_states
        hidden_states = self.attention_norm(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        # MLP with residual connection
        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class TransformerLM(nn.Module):
    """Transformer language model."""
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        intermediate_size: int,
        max_position_embeddings: int = 1024,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-12,
        tie_word_embeddings: bool = True,
    ):
        super().__init__()
        
        self.config = {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_attention_heads,
            "intermediate_size": intermediate_size,
            "max_position_embeddings": max_position_embeddings,
            "hidden_act": hidden_act,
            "hidden_dropout_prob": hidden_dropout_prob,
            "attention_probs_dropout_prob": attention_probs_dropout_prob,
            "layer_norm_eps": layer_norm_eps,
            "tie_word_embeddings": tie_word_embeddings,
        }
        
        # Embeddings
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                hidden_act=hidden_act,
                hidden_dropout_prob=hidden_dropout_prob,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                layer_norm_eps=layer_norm_eps,
            )
            for _ in range(num_hidden_layers)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Output projection
        if tie_word_embeddings:
            self.lm_head = None  # Will use word_embeddings.weight
        else:
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def create_causal_mask(self, seq_length: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask."""
        mask = torch.triu(torch.ones(seq_length, seq_length, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask.unsqueeze(0).unsqueeze(0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_length = input_ids.size()
        device = input_ids.device
        
        # Get embeddings
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        word_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        hidden_states = word_embeds + position_embeds
        hidden_states = self.dropout(hidden_states)
        
        # Create causal mask
        if attention_mask is None:
            attention_mask = self.create_causal_mask(seq_length, device)
        
        # Apply transformer blocks
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        hidden_states = self.final_norm(hidden_states)
        
        # Compute logits
        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            # Use tied embeddings
            logits = F.linear(hidden_states, self.word_embeddings.weight)
        
        outputs = {"logits": logits}
        
        # Compute loss if labels provided
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            outputs["loss"] = loss
            
            # Calculate perplexity
            with torch.no_grad():
                outputs["perplexity"] = torch.exp(loss)
        
        return outputs
    
    def get_num_params(self) -> int:
        """Get number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config: Dict[str, Any]) -> TransformerLM:
    """Create a transformer language model from config."""
    return TransformerLM(
        vocab_size=config.get("vocab_size", 50257),
        hidden_size=config["hidden_size"],
        num_hidden_layers=config["num_hidden_layers"],
        num_attention_heads=config["num_attention_heads"],
        intermediate_size=config["intermediate_size"],
        max_position_embeddings=config.get("max_position_embeddings", 1024),
        hidden_act=config.get("hidden_act", "gelu"),
        hidden_dropout_prob=config.get("hidden_dropout_prob", 0.1),
        attention_probs_dropout_prob=config.get("attention_probs_dropout_prob", 0.1),
        layer_norm_eps=config.get("layer_norm_eps", 1e-12),
        tie_word_embeddings=config.get("tie_word_embeddings", True),
    ) 