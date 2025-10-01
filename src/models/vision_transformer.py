"""Vision Transformer implementation for ImageNet scaling law experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any

# Try to import einops, fallback if not available
try:
    from einops import rearrange, repeat
    from einops.layers.torch import Rearrange
    HAS_EINOPS = True
except ImportError:
    HAS_EINOPS = False

    # Simple fallback implementations
    def rearrange(tensor, pattern, **kwargs):
        # Basic implementation for common patterns
        if pattern == 'b c h w -> b (h w) c':
            b, c, h, w = tensor.shape
            return tensor.permute(0, 2, 3, 1).contiguous().view(b, h*w, c)
        elif pattern == 'b (h w) c -> b c h w':
            b, hw, c = tensor.shape
            h = int(hw**0.5)
            w = hw // h
            return tensor.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        else:
            raise NotImplementedError(f"Pattern {pattern} not implemented in fallback")

    def repeat(tensor, pattern, **kwargs):
        # Basic implementation for common patterns
        if pattern == '() n d -> b n d':
            b = kwargs.get('b', 1)
            return tensor.expand(b, -1, -1)
        else:
            raise NotImplementedError(f"Pattern {pattern} not implemented in fallback")

    class Rearrange(nn.Module):
        def __init__(self, pattern, **kwargs):
            super().__init__()
            self.pattern = pattern
            self.kwargs = kwargs

        def forward(self, x):
            return rearrange(x, self.pattern, **self.kwargs)


class PatchEmbedding(nn.Module):
    """Image to patch embedding."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.projection(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self attention for vision transformer."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x


class MLP(nn.Module):
    """MLP block for vision transformer."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block for vision transformer."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self attention with residual
        x = x + self.attn(self.norm1(x))
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer for image classification."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        drop_path: float = 0.0,
    ):
        super().__init__()

        self.config = {
            "img_size": img_size,
            "patch_size": patch_size,
            "in_channels": in_channels,
            "num_classes": num_classes,
            "embed_dim": embed_dim,
            "depth": depth,
            "num_heads": num_heads,
            "mlp_ratio": mlp_ratio,
            "dropout": dropout,
            "attn_dropout": attn_dropout,
            "drop_path": drop_path,
        }

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim
        )
        num_patches = self.patch_embed.n_patches

        # Class token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Transformer blocks
        hidden_dim = int(embed_dim * mlp_ratio)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
            for _ in range(depth)
        ])

        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        # Initialize position embeddings and cls token
        if hasattr(self, 'pos_embed'):
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if hasattr(self, 'cls_token'):
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer blocks."""
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, n_patches + 1, embed_dim)

        # Add position embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x[:, 0]  # Return cls token

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        features = self.forward_features(x)
        logits = self.head(features)

        outputs = {"logits": logits}

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            outputs["loss"] = loss

            # Calculate accuracy
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                accuracy = (pred == labels).float().mean()
                outputs["accuracy"] = accuracy

        return outputs

    def get_num_params(self) -> int:
        """Get number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_vision_model(config: Dict[str, Any]) -> VisionTransformer:
    """Create a vision transformer from config."""
    return VisionTransformer(
        img_size=config.get("img_size", 224),
        patch_size=config.get("patch_size", 16),
        in_channels=config.get("in_channels", 3),
        num_classes=config.get("num_classes", 1000),
        embed_dim=config["embed_dim"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        mlp_ratio=config.get("mlp_ratio", 4.0),
        dropout=config.get("dropout", 0.1),
        attn_dropout=config.get("attn_dropout", 0.1),
        drop_path=config.get("drop_path", 0.0),
    )


# Model size configurations for scaling experiments
def get_vision_model_configs():
    """Get different model size configurations."""
    configs = [
        {
            "model_name": "vit_tiny",
            "embed_dim": 192,
            "depth": 12,
            "num_heads": 3,
            "mlp_ratio": 4.0,
            "num_parameters": 5_500_000,  # ~5.5M params
        },
        {
            "model_name": "vit_small",
            "embed_dim": 384,
            "depth": 12,
            "num_heads": 6,
            "mlp_ratio": 4.0,
            "num_parameters": 22_000_000,  # ~22M params
        },
        {
            "model_name": "vit_base",
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
            "mlp_ratio": 4.0,
            "num_parameters": 86_000_000,  # ~86M params
        },
        {
            "model_name": "vit_large",
            "embed_dim": 1024,
            "depth": 24,
            "num_heads": 16,
            "mlp_ratio": 4.0,
            "num_parameters": 307_000_000,  # ~307M params
        },
    ]

    # Add computed parameters for each config
    for config in configs:
        config.update({
            "img_size": 224,
            "patch_size": 16,
            "in_channels": 3,
            "num_classes": 1000,
            "dropout": 0.1,
            "attn_dropout": 0.1,
            "drop_path": 0.0,
        })

    return configs