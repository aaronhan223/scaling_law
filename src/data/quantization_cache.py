"""
Enhanced caching system for LibriSpeech quantization analysis experiments.
"""

import os
import pickle
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch


class QuantizationExperimentCache:
    """Handles disk-based caching of processed datasets for quantization experiments."""

    def __init__(self, cache_dir: str = "./dataset_cache/quantization"):
        """
        Initialize quantization experiment cache.

        Args:
            cache_dir: Directory to store cached datasets
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.cache_dir / "datasets").mkdir(exist_ok=True)
        (self.cache_dir / "experiments").mkdir(exist_ok=True)
        (self.cache_dir / "metadata").mkdir(exist_ok=True)

        print(f"Quantization cache initialized at: {self.cache_dir}")

    def _generate_dataset_key(
        self,
        librispeech_root: str,
        dataset_hours: List[int],
        model_indices: List[int],
        quantization_levels: int,
        batch_size: int,
        max_audio_length: float,
        **kwargs
    ) -> str:
        """Generate unique cache key for dataset configuration."""

        # Create deterministic hash of parameters
        params = {
            'librispeech_root': str(librispeech_root),
            'dataset_hours': sorted(dataset_hours),
            'model_indices': sorted(model_indices),
            'quantization_levels': quantization_levels,
            'batch_size': batch_size,
            'max_audio_length': max_audio_length,
        }

        # Add any additional relevant parameters
        for key, value in kwargs.items():
            if key in ['processor_name', 'target_sample_rate', 'seed']:
                params[key] = value

        # Generate hash
        param_str = json.dumps(params, sort_keys=True)
        cache_key = hashlib.md5(param_str.encode()).hexdigest()

        return cache_key

    def _get_dataset_cache_path(self, cache_key: str) -> Tuple[Path, Path]:
        """Get paths for dataset cache file and metadata."""
        cache_file = self.cache_dir / "datasets" / f"{cache_key}.pkl"
        metadata_file = self.cache_dir / "metadata" / f"{cache_key}.json"
        return cache_file, metadata_file

    def dataset_exists(
        self,
        librispeech_root: str,
        dataset_hours: List[int],
        model_indices: List[int],
        quantization_levels: int,
        batch_size: int,
        max_audio_length: float,
        **kwargs
    ) -> bool:
        """Check if processed dataset exists in cache."""

        cache_key = self._generate_dataset_key(
            librispeech_root, dataset_hours, model_indices,
            quantization_levels, batch_size, max_audio_length, **kwargs
        )

        cache_file, metadata_file = self._get_dataset_cache_path(cache_key)

        return cache_file.exists() and metadata_file.exists()

    def save_processed_dataset(
        self,
        processed_data: Dict[str, Any],
        librispeech_root: str,
        dataset_hours: List[int],
        model_indices: List[int],
        quantization_levels: int,
        batch_size: int,
        max_audio_length: float,
        **kwargs
    ) -> str:
        """
        Save processed dataset to cache.

        Args:
            processed_data: Dictionary containing processed dataloaders and metadata
            Other args: Configuration parameters for cache key generation

        Returns:
            Cache key for the saved dataset
        """

        cache_key = self._generate_dataset_key(
            librispeech_root, dataset_hours, model_indices,
            quantization_levels, batch_size, max_audio_length, **kwargs
        )

        cache_file, metadata_file = self._get_dataset_cache_path(cache_key)

        # Save metadata
        metadata = {
            'cache_key': cache_key,
            'librispeech_root': str(librispeech_root),
            'dataset_hours': dataset_hours,
            'model_indices': model_indices,
            'quantization_levels': quantization_levels,
            'batch_size': batch_size,
            'max_audio_length': max_audio_length,
            'timestamp': time.time(),
            'data_info': {
                'num_dataloaders': len(processed_data.get('dataloaders', {})),
                'dataset_sizes': processed_data.get('dataset_sizes', []),
            },
            **kwargs
        }

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save processed data (without actual dataloaders to save space, just metadata)
        cache_data = {
            'dataset_sizes': processed_data.get('dataset_sizes', []),
            'cache_info': processed_data.get('cache_info', {}),
            'config_hash': cache_key,
        }

        # If we want to cache actual dataloaders, we need to handle them carefully
        # For now, we'll cache the dataset configuration and rely on fast dataloader recreation

        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)

        print(f"Saved processed dataset to cache: {cache_key}")
        print(f"  Cache file: {cache_file}")
        print(f"  Metadata: {metadata_file}")

        return cache_key

    def load_processed_dataset(
        self,
        librispeech_root: str,
        dataset_hours: List[int],
        model_indices: List[int],
        quantization_levels: int,
        batch_size: int,
        max_audio_length: float,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Load processed dataset from cache.

        Returns:
            Cached data if available, None otherwise
        """

        cache_key = self._generate_dataset_key(
            librispeech_root, dataset_hours, model_indices,
            quantization_levels, batch_size, max_audio_length, **kwargs
        )

        cache_file, metadata_file = self._get_dataset_cache_path(cache_key)

        if not (cache_file.exists() and metadata_file.exists()):
            return None

        try:
            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # Load cached data
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)

            print(f"Loaded processed dataset from cache: {cache_key}")
            print(f"  Dataset hours: {metadata['dataset_hours']}")
            print(f"  Quantization levels: {metadata['quantization_levels']}")
            print(f"  Cached on: {time.ctime(metadata['timestamp'])}")

            return {
                'cache_data': cache_data,
                'metadata': metadata,
                'cache_key': cache_key,
            }

        except Exception as e:
            print(f"Error loading cached dataset {cache_key}: {e}")
            # Clean up corrupted cache files
            try:
                cache_file.unlink(missing_ok=True)
                metadata_file.unlink(missing_ok=True)
            except:
                pass
            return None

    def cache_experiment_results(
        self,
        results: Dict[str, Any],
        experiment_config: Dict[str, Any],
        cache_key: str
    ) -> None:
        """Cache experiment results."""

        results_file = self.cache_dir / "experiments" / f"{cache_key}_results.pkl"
        results_metadata = self.cache_dir / "metadata" / f"{cache_key}_results.json"

        # Save results
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)

        # Save metadata
        metadata = {
            'cache_key': cache_key,
            'experiment_config': experiment_config,
            'timestamp': time.time(),
            'results_summary': {
                'num_results': len(results) if isinstance(results, (list, dict)) else 1,
            }
        }

        with open(results_metadata, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Cached experiment results: {cache_key}")

    def load_experiment_results(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load cached experiment results."""

        results_file = self.cache_dir / "experiments" / f"{cache_key}_results.pkl"
        results_metadata = self.cache_dir / "metadata" / f"{cache_key}_results.json"

        if not (results_file.exists() and results_metadata.exists()):
            return None

        try:
            with open(results_file, 'rb') as f:
                results = pickle.load(f)

            with open(results_metadata, 'r') as f:
                metadata = json.load(f)

            print(f"Loaded cached experiment results: {cache_key}")
            return {
                'results': results,
                'metadata': metadata,
            }

        except Exception as e:
            print(f"Error loading cached results {cache_key}: {e}")
            return None

    def clear_cache(self, older_than_days: Optional[int] = None) -> None:
        """Clear cache files, optionally only those older than specified days."""

        if older_than_days:
            cutoff_time = time.time() - (older_than_days * 24 * 3600)

            for cache_type in ["datasets", "experiments", "metadata"]:
                cache_subdir = self.cache_dir / cache_type
                for cache_file in cache_subdir.glob("*"):
                    if cache_file.stat().st_mtime < cutoff_time:
                        cache_file.unlink()
                        print(f"Removed old cache file: {cache_file}")
        else:
            # Clear all cache
            for cache_type in ["datasets", "experiments", "metadata"]:
                cache_subdir = self.cache_dir / cache_type
                for cache_file in cache_subdir.glob("*"):
                    cache_file.unlink()
                    print(f"Removed cache file: {cache_file}")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about current cache."""

        info = {
            'cache_dir': str(self.cache_dir),
            'total_files': 0,
            'total_size_mb': 0,
            'dataset_caches': 0,
            'experiment_caches': 0,
            'metadata_files': 0,
        }

        for cache_type in ["datasets", "experiments", "metadata"]:
            cache_subdir = self.cache_dir / cache_type
            files = list(cache_subdir.glob("*"))

            for file_path in files:
                if file_path.is_file():
                    info['total_files'] += 1
                    info['total_size_mb'] += file_path.stat().st_size / (1024 * 1024)

                    if cache_type == "datasets":
                        info['dataset_caches'] += 1
                    elif cache_type == "experiments":
                        info['experiment_caches'] += 1
                    elif cache_type == "metadata":
                        info['metadata_files'] += 1

        return info

    def print_cache_status(self) -> None:
        """Print current cache status."""

        info = self.get_cache_info()

        print("\n" + "="*50)
        print("QUANTIZATION CACHE STATUS")
        print("="*50)
        print(f"Cache directory: {info['cache_dir']}")
        print(f"Total files: {info['total_files']}")
        print(f"Total size: {info['total_size_mb']:.1f} MB")
        print(f"Dataset caches: {info['dataset_caches']}")
        print(f"Experiment caches: {info['experiment_caches']}")
        print(f"Metadata files: {info['metadata_files']}")
        print("="*50 + "\n")


def get_quantization_cache(cache_dir: str = "./dataset_cache/quantization") -> QuantizationExperimentCache:
    """Get or create quantization experiment cache."""
    return QuantizationExperimentCache(cache_dir)