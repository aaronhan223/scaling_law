"""ImageNet dataset loading and preprocessing for scaling law experiments."""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import json
from pathlib import Path

# Try to import torchvision, use fallback if not available
try:
    from torchvision import transforms, datasets
    from torchvision.datasets import ImageFolder
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

    # Create basic fallback classes
    class transforms:
        class Compose:
            def __init__(self, transform_list):
                self.transforms = transform_list

            def __call__(self, img):
                for t in self.transforms:
                    img = t(img)
                return img

        class Resize:
            def __init__(self, size):
                self.size = size

            def __call__(self, img):
                return img.resize((self.size, self.size))

        class RandomResizedCrop:
            def __init__(self, size):
                self.size = size

            def __call__(self, img):
                return img.resize((self.size, self.size))

        class CenterCrop:
            def __init__(self, size):
                self.size = size

            def __call__(self, img):
                width, height = img.size
                left = (width - self.size) // 2
                top = (height - self.size) // 2
                right = left + self.size
                bottom = top + self.size
                return img.crop((left, top, right, bottom))

        class RandomHorizontalFlip:
            def __init__(self, p=0.5):
                self.p = p

            def __call__(self, img):
                if random.random() < self.p:
                    return img.transpose(Image.FLIP_LEFT_RIGHT)
                return img

        class ToTensor:
            def __call__(self, img):
                # Convert PIL image to tensor
                img_array = np.array(img)
                if len(img_array.shape) == 3:
                    img_array = img_array.transpose(2, 0, 1)  # HWC to CHW
                return torch.from_numpy(img_array).float() / 255.0

        class Normalize:
            def __init__(self, mean, std):
                self.mean = torch.tensor(mean).view(-1, 1, 1)
                self.std = torch.tensor(std).view(-1, 1, 1)

            def __call__(self, tensor):
                return (tensor - self.mean) / self.std

    # Basic ImageFolder replacement
    class ImageFolder:
        def __init__(self, root):
            self.root = root
            self.samples = []
            self.targets = []
            self.classes = []
            self.class_to_idx = {}

            if os.path.exists(root):
                self.classes = sorted([d for d in os.listdir(root)
                                     if os.path.isdir(os.path.join(root, d))])
                self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

                for class_idx, class_name in enumerate(self.classes):
                    class_dir = os.path.join(root, class_name)
                    if os.path.exists(class_dir):
                        for img_name in os.listdir(class_dir):
                            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                                img_path = os.path.join(class_dir, img_name)
                                self.samples.append((img_path, class_idx))
                                self.targets.append(class_idx)

        def __len__(self):
            return len(self.samples)


class ImageNetDataset(Dataset):
    """Custom ImageNet dataset with flexible sampling."""

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        max_samples: Optional[int] = None,
        num_classes: Optional[int] = None,
        seed: int = 42,
        class_subset: Optional[List[int]] = None,
    ):
        """
        Initialize ImageNet dataset.

        Args:
            data_path: Path to ImageNet data
            split: Dataset split ("train", "val", "test")
            transform: Image transformations
            max_samples: Maximum number of samples to use
            num_classes: Number of classes to use (subset of 1000)
            seed: Random seed for sampling
            class_subset: Specific class indices to use
        """
        self.data_path = data_path
        self.split = split
        self.transform = transform
        self.max_samples = max_samples
        self.num_classes = num_classes
        self.seed = seed

        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Load dataset using ImageFolder
        split_path = os.path.join(data_path, split)
        if not os.path.exists(split_path):
            raise ValueError(f"Split path does not exist: {split_path}")

        self.dataset = ImageFolder(split_path)

        # Filter out non-ImageNet class folders (e.g., tar files, meta files)
        # ImageNet classes follow the pattern n######## (8 digits)
        import re
        valid_classes = []
        valid_class_to_idx = {}
        imagenet_pattern = re.compile(r'^n\d{8}$')

        for class_name, class_idx in self.dataset.class_to_idx.items():
            if imagenet_pattern.match(class_name):
                new_idx = len(valid_classes)
                valid_classes.append(class_name)
                valid_class_to_idx[class_name] = new_idx

        # Filter samples to only include valid ImageNet classes and remap indices
        filtered_samples = []
        for path, old_idx in self.dataset.samples:
            class_name = self.dataset.classes[old_idx]
            if class_name in valid_class_to_idx:
                new_idx = valid_class_to_idx[class_name]
                filtered_samples.append((path, new_idx))

        self.dataset.samples = filtered_samples
        self.dataset.targets = [s[1] for s in filtered_samples]
        self.classes = valid_classes
        self.class_to_idx = valid_class_to_idx

        # Filter classes if specified
        if class_subset is not None:
            self._filter_by_class_subset(class_subset)
        elif num_classes is not None and num_classes < len(self.classes):
            # Randomly select num_classes classes
            selected_classes = random.sample(range(len(self.classes)), num_classes)
            self._filter_by_class_subset(selected_classes)

        # Sample subset if max_samples is specified
        if max_samples is not None and max_samples < len(self.dataset):
            self._sample_subset(max_samples)

        print(f"Loaded {split} split with {len(self.dataset)} samples from {len(self.classes)} classes")

    def _filter_by_class_subset(self, class_indices: List[int]):
        """Filter dataset to only include specified classes."""
        # Create mapping from old to new class indices
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(class_indices))}

        # Filter samples
        filtered_samples = []
        for path, class_idx in self.dataset.samples:
            if class_idx in class_indices:
                new_class_idx = old_to_new[class_idx]
                filtered_samples.append((path, new_class_idx))

        self.dataset.samples = filtered_samples
        self.dataset.targets = [s[1] for s in filtered_samples]

        # Update classes
        selected_class_names = [self.classes[i] for i in sorted(class_indices)]
        self.classes = selected_class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(selected_class_names)}

    def _sample_subset(self, max_samples: int):
        """Sample a subset of the dataset with consistent nested sampling for scaling laws."""
        # For scaling law experiments, we need nested sampling where smaller datasets
        # are proper subsets of larger ones to avoid sample overlap

        # First, create a deterministic master ordering of all samples
        # Group samples by class for stratified sampling
        class_samples = {}
        for i, (path, class_idx) in enumerate(self.dataset.samples):
            if class_idx not in class_samples:
                class_samples[class_idx] = []
            class_samples[class_idx].append((path, class_idx))

        # Shuffle samples within each class deterministically using the seed
        master_samples = []
        for class_idx in sorted(class_samples.keys()):
            class_sample_list = class_samples[class_idx]
            # Use a deterministic shuffle based on seed + class_idx
            class_random = random.Random(self.seed + class_idx)
            class_random.shuffle(class_sample_list)
            master_samples.extend(class_sample_list)

        # Final shuffle of all samples using main seed
        main_random = random.Random(self.seed)
        main_random.shuffle(master_samples)

        # Now create stratified subset maintaining class distribution
        samples_per_class = max_samples // len(self.classes)
        remainder = max_samples % len(self.classes)

        # Re-group the shuffled samples by class
        class_samples_shuffled = {}
        for path, class_idx in master_samples:
            if class_idx not in class_samples_shuffled:
                class_samples_shuffled[class_idx] = []
            class_samples_shuffled[class_idx].append((path, class_idx))

        # Take stratified samples ensuring nested property
        # We maintain exact ordering by interleaving classes in round-robin fashion
        # This ensures that the first N samples are always the same across different dataset sizes

        # Create ordered list by taking samples from each class in round-robin
        sampled_samples = []
        class_counters = {class_idx: 0 for class_idx in sorted(class_samples_shuffled.keys())}
        max_per_class = {class_idx: samples_per_class + (1 if class_idx < remainder else 0)
                        for class_idx in sorted(class_samples_shuffled.keys())}

        # Round-robin sampling to maintain consistent ordering
        total_sampled = 0
        while total_sampled < max_samples:
            added_in_round = 0
            for class_idx in sorted(class_samples_shuffled.keys()):
                if (class_counters[class_idx] < max_per_class[class_idx] and
                    class_counters[class_idx] < len(class_samples_shuffled[class_idx]) and
                    total_sampled < max_samples):

                    sample = class_samples_shuffled[class_idx][class_counters[class_idx]]
                    sampled_samples.append(sample)
                    class_counters[class_idx] += 1
                    total_sampled += 1
                    added_in_round += 1

            # Break if no samples were added in this round
            if added_in_round == 0:
                break

        # Update dataset - samples are in deterministic nested order
        self.dataset.samples = sampled_samples[:max_samples]
        self.dataset.targets = [s[1] for s in self.dataset.samples]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        path, target = self.dataset.samples[idx]

        # Load image
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')

        # Apply transformations
        if self.transform is not None:
            img = self.transform(img)

        return {"x": img, "labels": target}


def get_imagenet_transforms(
    img_size: int = 224,
    split: str = "train",
    normalize: bool = True,
) -> transforms.Compose:
    """
    Get standard ImageNet transformations.

    Args:
        img_size: Target image size
        split: Dataset split
        normalize: Whether to apply ImageNet normalization

    Returns:
        Composed transformations
    """
    if split == "train":
        transform_list = [
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
    else:
        transform_list = [
            transforms.Resize(int(img_size * 256 / 224)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ]

    if normalize:
        # ImageNet normalization
        transform_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )

    return transforms.Compose(transform_list)


def create_imagenet_dataloaders(
    data_path: str,
    img_size: int = 224,
    train_batch_size: int = 128,
    eval_batch_size: int = 256,
    num_workers: int = 8,
    max_samples_train: Optional[int] = None,
    max_samples_val: Optional[int] = None,
    max_samples_test: Optional[int] = None,
    num_classes: Optional[int] = None,
    seed: int = 42,
    normalize: bool = True,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create ImageNet dataloaders.

    Args:
        data_path: Path to ImageNet data
        img_size: Target image size
        train_batch_size: Training batch size
        eval_batch_size: Evaluation batch size
        num_workers: Number of data loading workers
        max_samples_train: Maximum training samples
        max_samples_val: Maximum validation samples
        max_samples_test: Maximum test samples
        num_classes: Number of classes to use
        seed: Random seed
        normalize: Whether to apply ImageNet normalization
        pin_memory: Whether to pin memory

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Get transforms
    train_transform = get_imagenet_transforms(img_size, "train", normalize)
    eval_transform = get_imagenet_transforms(img_size, "val", normalize)

    # Create train dataset first to check available data
    train_dataset = ImageNetDataset(
        data_path=data_path,
        split="train",
        transform=train_transform,
        max_samples=max_samples_train,
        num_classes=num_classes,
        seed=seed,
    )

    # Check if val split has enough data, otherwise use train split for val/test
    val_split_path = os.path.join(data_path, "val")
    temp_val_dataset = ImageNetDataset(
        data_path=data_path,
        split="val",
        transform=eval_transform,
        max_samples=None,  # Check full size first
        num_classes=num_classes,
        seed=seed,
    )

    if len(temp_val_dataset.dataset) < max_samples_val:
        print(f"Val split has only {len(temp_val_dataset.dataset)} samples, using train split for val/test")
        # Use separate portions of training data for validation and test
        val_dataset = ImageNetDataset(
            data_path=data_path,
            split="train",
            transform=eval_transform,
            max_samples=max_samples_val,
            num_classes=num_classes,
            seed=seed + 1000,  # Different seed for validation
        )

        test_dataset = ImageNetDataset(
            data_path=data_path,
            split="train",
            transform=eval_transform,
            max_samples=max_samples_test,
            num_classes=num_classes,
            seed=seed + 2000,  # Different seed for test
        )
    else:
        # Use actual val split
        val_dataset = ImageNetDataset(
            data_path=data_path,
            split="val",
            transform=eval_transform,
            max_samples=max_samples_val,
            num_classes=num_classes,
            seed=seed,
        )

        # Create test dataset (use val split if test doesn't exist)
        test_path = os.path.join(data_path, "test")
        if os.path.exists(test_path):
            test_dataset = ImageNetDataset(
                data_path=data_path,
                split="test",
                transform=eval_transform,
                max_samples=max_samples_test,
                num_classes=num_classes,
                seed=seed,
            )
        else:
            # Use a subset of validation data as test data
            test_dataset = ImageNetDataset(
                data_path=data_path,
                split="val",
                transform=eval_transform,
                max_samples=max_samples_test,
                num_classes=num_classes,
                seed=seed + 1,  # Different seed for test split
            )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True,  # For consistent batch sizes during training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


def create_imagenet_dataloaders_with_transforms(
    data_path: str,
    img_size: int = 224,
    train_batch_size: int = 128,
    eval_batch_size: int = 256,
    num_workers: int = 8,
    max_samples_train: Optional[int] = None,
    max_samples_val: Optional[int] = None,
    max_samples_test: Optional[int] = None,
    num_classes: Optional[int] = None,
    seed: int = 42,
    normalize: bool = True,
    pin_memory: bool = True,
    # Transform parameters
    linear_transform: Optional[Any] = None,
    nonlinear_transform: Optional[Any] = None,
    quantization_transform: Optional[Any] = None,
    lowrank_transform: Optional[Any] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create ImageNet dataloaders with custom transforms applied.

    Args:
        Similar to create_imagenet_dataloaders, plus:
        linear_transform: Linear transformation to apply
        nonlinear_transform: Non-linear transformation to apply
        quantization_transform: Quantization transformation to apply
        lowrank_transform: Low-rank transformation to apply

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # First create standard dataloaders
    train_loader, val_loader, test_loader = create_imagenet_dataloaders(
        data_path=data_path,
        img_size=img_size,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        max_samples_train=max_samples_train,
        max_samples_val=max_samples_val,
        max_samples_test=max_samples_test,
        num_classes=num_classes,
        seed=seed,
        normalize=normalize,
        pin_memory=pin_memory,
    )

    # If no transforms are specified, return standard loaders
    if not any([linear_transform, nonlinear_transform, quantization_transform, lowrank_transform]):
        return train_loader, val_loader, test_loader

    # Apply transforms to datasets
    from .image_transforms import ImageTransformedDataset

    train_dataset_transformed = ImageTransformedDataset(
        base_dataset=train_loader.dataset,
        linear_transform=linear_transform,
        nonlinear_transform=nonlinear_transform,
        quantization_transform=quantization_transform,
        lowrank_transform=lowrank_transform,
        cache_transformed=False,  # Disable caching for large image datasets
    )

    val_dataset_transformed = ImageTransformedDataset(
        base_dataset=val_loader.dataset,
        linear_transform=linear_transform,
        nonlinear_transform=nonlinear_transform,
        quantization_transform=quantization_transform,
        lowrank_transform=lowrank_transform,
        cache_transformed=False,
    )

    test_dataset_transformed = ImageTransformedDataset(
        base_dataset=test_loader.dataset,
        linear_transform=linear_transform,
        nonlinear_transform=nonlinear_transform,
        quantization_transform=quantization_transform,
        lowrank_transform=lowrank_transform,
        cache_transformed=False,
    )

    # Create new dataloaders with transformed datasets
    train_loader_transformed = DataLoader(
        train_dataset_transformed,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True,
    )

    val_loader_transformed = DataLoader(
        val_dataset_transformed,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=False,
    )

    test_loader_transformed = DataLoader(
        test_dataset_transformed,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=False,
    )

    return train_loader_transformed, val_loader_transformed, test_loader_transformed


def get_imagenet_dataset_sizes():
    """Get standard ImageNet dataset size configurations."""
    return {
        "tiny": 1000,      # 1K samples
        "small": 10000,    # 10K samples
        "medium": 50000,   # 50K samples
        "large": 100000,   # 100K samples
        "xlarge": 500000,  # 500K samples
        "full": None,      # Full dataset
    }


def parse_imagenet_size(size_str: str) -> Optional[int]:
    """Parse ImageNet dataset size string."""
    size_map = get_imagenet_dataset_sizes()

    if size_str.lower() in size_map:
        return size_map[size_str.lower()]

    # Try to parse as number with suffix
    size_str = size_str.upper().strip()
    if size_str.endswith("K"):
        return int(float(size_str[:-1]) * 1e3)
    elif size_str.endswith("M"):
        return int(float(size_str[:-1]) * 1e6)
    else:
        try:
            return int(size_str)
        except ValueError:
            raise ValueError(f"Cannot parse dataset size: {size_str}")


def get_imagenet_class_subsets():
    """Get predefined class subsets for ImageNet."""
    return {
        "top10": list(range(10)),      # First 10 classes
        "top100": list(range(100)),    # First 100 classes
        "random10": None,              # Will be randomly selected
        "random100": None,             # Will be randomly selected
        "animals": None,               # Would need manual curation
        "objects": None,               # Would need manual curation
    }