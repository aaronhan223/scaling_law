"""Data module for scaling law experiments."""

from .dataset import CustomDataset, create_dataloaders
from .transforms import LinearTransform, NonLinearTransform, TransformedDataset

__all__ = [
    "CustomDataset",
    "create_dataloaders",
    "LinearTransform",
    "NonLinearTransform",
    "TransformedDataset",
] 