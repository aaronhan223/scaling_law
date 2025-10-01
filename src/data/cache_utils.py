"""Utilities for dataset caching to disk."""

import os
import pickle
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import torch


class DatasetCache:
    """Handles disk-based caching of datasets."""

    def __init__(self, cache_dir: str = "./dataset_cache"):
        """
        Initialize dataset cache.

        Args:
            cache_dir: Directory to store cached datasets
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.cache_dir / "librispeech").mkdir(exist_ok=True)
        (self.cache_dir / "transforms").mkdir(exist_ok=True)
        (self.cache_dir / "metadata").mkdir(exist_ok=True)

    def _generate_cache_key(self, **kwargs) -> str:
        """Generate a unique cache key based on parameters."""
        # Create a deterministic hash from the parameters
        key_data = {k: v for k, v in kwargs.items() if v is not None}
        key_str = json.dumps(key_data, sort_keys=True)
        cache_key = hashlib.md5(key_str.encode()).hexdigest()
        return cache_key

    def _get_cache_path(self, cache_type: str, cache_key: str) -> Path:
        """Get the full path for a cache file."""
        return self.cache_dir / cache_type / f"{cache_key}.pkl"

    def _get_metadata_path(self, cache_type: str, cache_key: str) -> Path:
        """Get the path for cache metadata."""
        return self.cache_dir / "metadata" / f"{cache_type}_{cache_key}.json"

    def save_dataset(self, dataset: Any, cache_type: str, **kwargs) -> str:
        """
        Save a dataset to disk cache.

        Args:
            dataset: Dataset object to cache
            cache_type: Type of cache (e.g., 'librispeech', 'transforms')
            **kwargs: Parameters used to create the dataset (for cache key)

        Returns:
            Cache key for the saved dataset
        """
        cache_key = self._generate_cache_key(**kwargs)
        cache_path = self._get_cache_path(cache_type, cache_key)
        metadata_path = self._get_metadata_path(cache_type, cache_key)

        print(f"Saving dataset to cache: {cache_path}")

        # Save dataset
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"Error saving dataset to cache: {e}")
            return None

        # Save metadata
        metadata = {
            'cache_key': cache_key,
            'cache_type': cache_type,
            'parameters': kwargs,
            'dataset_length': len(dataset) if hasattr(dataset, '__len__') else 0,
            'creation_time': str(torch.utils.data.get_worker_info() or 'main_process')
        }

        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Could not save metadata: {e}")

        print(f"Dataset cached successfully with key: {cache_key}")
        return cache_key

    def load_dataset(self, cache_type: str, **kwargs) -> Optional[Any]:
        """
        Load a dataset from disk cache.

        Args:
            cache_type: Type of cache (e.g., 'librispeech', 'transforms')
            **kwargs: Parameters used to create the dataset (for cache key)

        Returns:
            Cached dataset or None if not found
        """
        cache_key = self._generate_cache_key(**kwargs)
        cache_path = self._get_cache_path(cache_type, cache_key)

        if not cache_path.exists():
            return None

        print(f"Loading dataset from cache: {cache_path}")

        try:
            with open(cache_path, 'rb') as f:
                dataset = pickle.load(f)
            print(f"Dataset loaded successfully from cache: {cache_key}")
            return dataset
        except Exception as e:
            print(f"Error loading dataset from cache: {e}")
            # Remove corrupted cache file
            try:
                cache_path.unlink()
                metadata_path = self._get_metadata_path(cache_type, cache_key)
                if metadata_path.exists():
                    metadata_path.unlink()
                print("Removed corrupted cache file")
            except:
                pass
            return None

    def cache_exists(self, cache_type: str, **kwargs) -> bool:
        """
        Check if a cache file exists for the given parameters.

        Args:
            cache_type: Type of cache
            **kwargs: Parameters used to create the dataset

        Returns:
            True if cache exists, False otherwise
        """
        cache_key = self._generate_cache_key(**kwargs)
        cache_path = self._get_cache_path(cache_type, cache_key)
        return cache_path.exists()

    def clear_cache(self, cache_type: Optional[str] = None):
        """
        Clear cache files.

        Args:
            cache_type: Specific cache type to clear, or None to clear all
        """
        if cache_type:
            cache_dir = self.cache_dir / cache_type
            if cache_dir.exists():
                for cache_file in cache_dir.glob("*.pkl"):
                    cache_file.unlink()
                print(f"Cleared {cache_type} cache")
        else:
            for cache_file in self.cache_dir.rglob("*.pkl"):
                cache_file.unlink()
            for metadata_file in self.cache_dir.rglob("*.json"):
                metadata_file.unlink()
            print("Cleared all cache files")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached datasets."""
        info = {
            'cache_dir': str(self.cache_dir),
            'total_files': len(list(self.cache_dir.rglob("*.pkl"))),
            'cache_types': {}
        }

        for cache_type_dir in self.cache_dir.iterdir():
            if cache_type_dir.is_dir() and cache_type_dir.name != 'metadata':
                cache_files = list(cache_type_dir.glob("*.pkl"))
                total_size = sum(f.stat().st_size for f in cache_files)
                info['cache_types'][cache_type_dir.name] = {
                    'files': len(cache_files),
                    'total_size_mb': total_size / (1024 * 1024)
                }

        return info

    def list_cached_datasets(self, cache_type: Optional[str] = None) -> Dict[str, Dict]:
        """List all cached datasets with their metadata."""
        cached_datasets = {}

        metadata_dir = self.cache_dir / "metadata"
        if not metadata_dir.exists():
            return cached_datasets

        for metadata_file in metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                if cache_type is None or metadata.get('cache_type') == cache_type:
                    cached_datasets[metadata['cache_key']] = metadata
            except Exception as e:
                print(f"Warning: Could not read metadata from {metadata_file}: {e}")

        return cached_datasets


# Global cache instance
_global_cache = None

def get_dataset_cache(cache_dir: str = "./dataset_cache") -> DatasetCache:
    """Get or create global dataset cache instance."""
    global _global_cache
    if _global_cache is None or _global_cache.cache_dir != Path(cache_dir):
        _global_cache = DatasetCache(cache_dir)
    return _global_cache