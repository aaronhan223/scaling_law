"""Data module for scaling law experiments."""

from .dataset import WikiTextDataset, create_dataloaders
from .transforms import LinearTransform, NonLinearTransform, TransformedDataset

__all__ = [
    "WikiTextDataset",
    "create_dataloaders",
    "LinearTransform",
    "NonLinearTransform",
    "TransformedDataset",
] 