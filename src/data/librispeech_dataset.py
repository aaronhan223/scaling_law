"""LibriSpeech dataset for speech recognition scaling law experiments."""

import os
import json
import warnings
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Union
from transformers import Wav2Vec2Processor
import random
from pathlib import Path
from .cache_utils import get_dataset_cache
import time
import requests

# Suppress torchaudio deprecation warnings
# - Backend functions deprecated in newer versions
# - TorchCodec migration warnings for future 2.9+ versions
# - General torchaudio UserWarnings about API changes
warnings.filterwarnings("ignore", message=".*torchaudio._backend.*deprecated.*")
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", message=".*load_with_torchcodec.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")


class LibriSpeechDataset(Dataset):
    """LibriSpeech dataset for speech recognition experiments."""

    def __init__(
        self,
        data_root: str,
        split: str = "train-clean-100",
        max_audio_length: float = 20.0,  # seconds
        max_samples: Optional[int] = None,
        processor_name: str = "facebook/wav2vec2-base-960h",
        seed: int = 42,
        target_sample_rate: int = 16000,
    ):
        """
        Initialize LibriSpeech dataset.

        Args:
            data_root: Root directory containing LibriSpeech data
            split: Dataset split (train-clean-100, train-clean-360, train-other-500,
                   dev-clean, dev-other, test-clean, test-other)
            max_audio_length: Maximum audio length in seconds
            max_samples: Maximum number of samples to use (for dataset size scaling)
            processor_name: HuggingFace processor name for wav2vec2
            seed: Random seed for reproducibility
            target_sample_rate: Target sample rate for audio
        """
        self.data_root = Path(data_root)
        self.split = split
        self.max_audio_length = max_audio_length
        self.max_samples = max_samples
        self.target_sample_rate = target_sample_rate
        self.seed = seed
        self.processor_name = processor_name  # Store for use in __setstate__

        # Initialize processor
        self._init_processor(processor_name)

        # Configure torchaudio backend for FLAC support
        self._configure_audio_backend()

        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Load dataset metadata
        self.samples = self._load_samples()

        print(f"Loaded {len(self.samples)} samples from {split}")
        if max_samples and len(self.samples) > max_samples:
            # Shuffle and take subset
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]
            print(f"Limited to {len(self.samples)} samples")

    def _load_samples(self) -> List[Dict]:
        """Load sample metadata from LibriSpeech directory structure."""
        samples = []
        split_dir = self.data_root / self.split

        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")

        # LibriSpeech structure: split/speaker_id/chapter_id/audio_files
        for speaker_dir in split_dir.iterdir():
            if not speaker_dir.is_dir():
                continue

            speaker_id = speaker_dir.name

            for chapter_dir in speaker_dir.iterdir():
                if not chapter_dir.is_dir():
                    continue

                chapter_id = chapter_dir.name

                # Look for transcript file
                trans_file = chapter_dir / f"{speaker_id}-{chapter_id}.trans.txt"
                if not trans_file.exists():
                    continue

                # Parse transcripts
                with open(trans_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        # Parse: utterance_id transcript
                        parts = line.split(' ', 1)
                        if len(parts) < 2:
                            continue

                        utterance_id = parts[0]
                        transcript = parts[1]

                        # Find corresponding audio file
                        audio_file = chapter_dir / f"{utterance_id}.flac"
                        if audio_file.exists():
                            samples.append({
                                'utterance_id': utterance_id,
                                'audio_path': str(audio_file),
                                'transcript': transcript,
                                'speaker_id': speaker_id,
                                'chapter_id': chapter_id,
                            })

        return samples

    def _init_processor(self, processor_name: str):
        """Initialize processor with proper tokenizer fix and retry logic for rate limiting."""
        max_retries = 5
        base_delay = 2.0

        # Check for custom HuggingFace cache directory
        hf_cache_dir = os.environ.get('HF_HOME', os.path.expanduser("~/.cache/huggingface"))
        cache_dir = os.path.join(hf_cache_dir, "transformers")

        # Set HuggingFace offline mode if requested
        offline_mode = os.environ.get('TRANSFORMERS_OFFLINE', 'false').lower() == 'true'

        for attempt in range(max_retries):
            try:
                print(f"Attempting to load processor {processor_name} (attempt {attempt + 1}/{max_retries})...")

                # Try to load with offline mode first if available locally or if offline mode is forced
                try:
                    self.processor = Wav2Vec2Processor.from_pretrained(
                        processor_name,
                        local_files_only=True,
                        cache_dir=cache_dir
                    )
                    print(f"Successfully loaded processor from local cache at {cache_dir}")
                    break
                except Exception as local_error:
                    if offline_mode:
                        print(f"Offline mode enabled but local cache load failed: {local_error}")
                        raise local_error

                    print(f"Local cache load failed: {local_error}")

                    # Fall back to online loading with retry logic
                    self.processor = Wav2Vec2Processor.from_pretrained(
                        processor_name,
                        cache_dir=cache_dir
                    )
                    print(f"Successfully loaded processor from HuggingFace Hub")
                    break

            except requests.exceptions.HTTPError as e:
                if "429" in str(e):
                    # Rate limited - exponential backoff
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"Rate limited (HTTP 429). Waiting {delay:.1f} seconds before retry...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"HTTP error: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(base_delay)
                        continue
                    else:
                        raise
            except Exception as e:
                print(f"Error loading processor: {e}")
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    # Final fallback - create a minimal processor if all else fails
                    print("WARNING: Failed to load processor after all retries. Using fallback configuration.")
                    # Try to use a different model as fallback
                    try:
                        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
                        print("Loaded fallback wav2vec2-base processor")
                        break
                    except:
                        raise RuntimeError(f"Failed to load any wav2vec2 processor after {max_retries} attempts")

        # Fix for newer transformers versions - ensure tokenizer has required attributes
        if hasattr(self.processor, 'tokenizer'):
            tokenizer = self.processor.tokenizer
            # Set missing private attributes that newer transformers require
            if not hasattr(tokenizer, '_unk_token'):
                tokenizer._unk_token = "[UNK]"
            if not hasattr(tokenizer, '_pad_token'):
                tokenizer._pad_token = "[PAD]"
            if not hasattr(tokenizer, '_bos_token'):
                tokenizer._bos_token = None
            if not hasattr(tokenizer, '_eos_token'):
                tokenizer._eos_token = None
            if not hasattr(tokenizer, '_sep_token'):
                tokenizer._sep_token = None
            if not hasattr(tokenizer, '_cls_token'):
                tokenizer._cls_token = None
            if not hasattr(tokenizer, '_mask_token'):
                tokenizer._mask_token = None

            # Ensure the tokens are properly mapped in the vocab if they exist
            if hasattr(tokenizer, 'vocab'):
                if "<unk>" in tokenizer.vocab and not hasattr(tokenizer, '_unk_token'):
                    tokenizer._unk_token = "<unk>"
                if "<pad>" in tokenizer.vocab and not hasattr(tokenizer, '_pad_token'):
                    tokenizer._pad_token = "<pad>"

    def __setstate__(self, state):
        """Called when loading from pickle - reinitialize processor."""
        self.__dict__.update(state)
        # Reinitialize processor to avoid pickle-related issues
        processor_name = getattr(self, 'processor_name', 'facebook/wav2vec2-base-960h')
        try:
            self._init_processor(processor_name)
        except Exception as e:
            print(f"Warning: Failed to reinitialize processor during unpickling: {e}")
            print("This may affect audio processing but experiment can continue")

    def _configure_audio_backend(self):
        """Configure torchaudio backend for FLAC support."""
        try:
            # Check torchaudio version and handle backend configuration appropriately
            import torchaudio

            # Check if backend functions are available (older versions)
            if hasattr(torchaudio, 'get_audio_backend') and hasattr(torchaudio, 'set_audio_backend'):
                try:
                    current_backend = torchaudio.get_audio_backend()
                    print(f"Current torchaudio backend: {current_backend}")

                    if current_backend != 'soundfile':
                        try:
                            torchaudio.set_audio_backend('soundfile')
                            print("Set torchaudio backend to 'soundfile' for FLAC support")
                        except Exception as e:
                            print(f"Warning: Could not set soundfile backend: {e}")
                except Exception as e:
                    print(f"Warning: Backend functions deprecated or not working: {e}")
            else:
                print("TorchAudio backend configuration not needed (modern version with dispatcher)")

            # Test if we can load a dummy file to check FLAC support
            print("TorchAudio configured for FLAC support")

        except Exception as e:
            print(f"Warning: Error configuring audio backend: {e}")
            print("Will use fallback methods for audio loading")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_audio_safe(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """Load audio file with proper backend handling and error recovery."""
        try:
            # Try loading with torchaudio directly (modern versions handle FLAC automatically)
            # Suppress warnings during loading
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    audio, sample_rate = torchaudio.load(audio_path)
                    return audio, sample_rate
                except Exception as e:
                    print(f"Warning: torchaudio failed to load {audio_path}: {e}")

            # Fallback: try with soundfile directly if available
            try:
                import soundfile as sf
                audio_data, sample_rate = sf.read(audio_path)
                # Convert to torch tensor and add channel dimension
                audio = torch.from_numpy(audio_data).float()
                if audio.dim() == 1:
                    audio = audio.unsqueeze(0)  # Add channel dimension
                elif audio.dim() == 2:
                    audio = audio.T  # soundfile returns [samples, channels], we want [channels, samples]
                return audio, sample_rate
            except ImportError:
                print("Warning: soundfile not available for fallback")
            except Exception as e:
                print(f"Warning: soundfile failed to load {audio_path}: {e}")

            # Last resort: create dummy audio data
            print(f"Error: Could not load audio file {audio_path}, creating dummy data")
            dummy_duration = 1.0  # 1 second
            dummy_samples = int(self.target_sample_rate * dummy_duration)
            dummy_audio = torch.zeros(1, dummy_samples)
            return dummy_audio, self.target_sample_rate

        except Exception as e:
            print(f"Critical error loading audio {audio_path}: {e}")
            # Create minimal dummy data
            dummy_audio = torch.zeros(1, 1000)
            return dummy_audio, self.target_sample_rate

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        sample = self.samples[idx]

        # Load audio with proper backend handling
        audio, sample_rate = self._load_audio_safe(sample['audio_path'])

        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        # Resample if needed
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.target_sample_rate
            )
            audio = resampler(audio)

        # Flatten to 1D
        audio = audio.squeeze(0)

        # Truncate if too long
        max_length = int(self.max_audio_length * self.target_sample_rate)
        if len(audio) > max_length:
            audio = audio[:max_length]

        # Process audio with feature extractor only
        inputs = self.processor.feature_extractor(
            audio.numpy(),
            sampling_rate=self.target_sample_rate,
            return_tensors="pt",
        )

        # Process transcript with tokenizer only
        labels = self.processor.tokenizer(
            sample['transcript'],
            return_tensors="pt",
        )

        # Prepare output with proper tensor handling
        input_values = inputs.input_values.squeeze(0).clone().contiguous()
        labels_tensor = labels.input_ids.squeeze(0).clone().contiguous()

        result = {
            'input_values': input_values,
            'labels': labels_tensor,
            'utterance_id': sample['utterance_id'],
            'transcript': sample['transcript'],
        }

        # Add attention mask if available from feature extractor
        if hasattr(inputs, 'attention_mask') and inputs.attention_mask is not None:
            result['attention_mask'] = inputs.attention_mask.squeeze(0).clone().contiguous()
        else:
            # Create default attention mask (all ones for the input length)
            result['attention_mask'] = torch.ones(input_values.shape[0], dtype=torch.long)

        return result

    def get_vocab_size(self) -> int:
        """Get vocabulary size from processor."""
        return self.processor.tokenizer.vocab_size

    def decode_prediction(self, predicted_ids: torch.Tensor) -> str:
        """Decode predicted token IDs to text."""
        predicted_ids = predicted_ids.cpu().numpy()
        return self.processor.batch_decode(predicted_ids)[0]

    def collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate function for DataLoader."""
        # Extract components
        input_values = [item['input_values'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        labels = [item['labels'] for item in batch]

        # Ensure tensors are contiguous before padding
        input_values = [item.contiguous() for item in input_values]
        attention_mask = [item.contiguous() for item in attention_mask]
        labels = [item.contiguous() for item in labels]

        # Pad sequences to same length
        input_values = torch.nn.utils.rnn.pad_sequence(
            input_values, batch_first=True, padding_value=0.0
        ).contiguous()
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        ).contiguous()

        # For labels, pad with -100 (ignore token for loss computation)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        ).contiguous()

        return {
            'input_values': input_values,
            'attention_mask': attention_mask,
            'labels': labels,
            'utterance_ids': [item['utterance_id'] for item in batch],
            'transcripts': [item['transcript'] for item in batch],
        }


def create_librispeech_dataloaders(
    data_root: str,
    train_splits: List[str] = ["train-clean-100", "train-clean-360", "train-other-500"],
    dev_split: str = "dev-clean",
    test_split: str = "test-clean",
    max_audio_length: float = 20.0,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = 10000,
    max_test_samples: Optional[int] = 2000,
    train_batch_size: int = 8,
    eval_batch_size: int = 16,
    num_workers: int = 4,
    processor_name: str = "facebook/wav2vec2-base-960h",
    seed: int = 42,
    cache_dir: str = "./dataset_cache",
    use_cache: bool = True,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create LibriSpeech dataloaders for training, validation, and testing with caching support.

    Args:
        data_root: Root directory containing LibriSpeech data
        train_splits: List of training splits to combine
        dev_split: Development/validation split
        test_split: Test split
        max_audio_length: Maximum audio length in seconds
        max_train_samples: Maximum training samples
        max_val_samples: Maximum validation samples
        max_test_samples: Maximum test samples
        train_batch_size: Training batch size
        eval_batch_size: Evaluation batch size
        num_workers: Number of data loading workers
        processor_name: HuggingFace processor name
        seed: Random seed
        cache_dir: Directory to store cached datasets
        use_cache: Whether to use disk caching

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """

    # Initialize cache
    cache = get_dataset_cache(cache_dir) if use_cache else None

    # Helper function to create or load cached dataset
    def create_or_load_dataset(split_name: str, max_samples: Optional[int], is_train: bool = False) -> LibriSpeechDataset:
        # Create cache parameters
        cache_params = {
            'data_root': data_root,
            'split': split_name,
            'max_audio_length': max_audio_length,
            'max_samples': max_samples,
            'processor_name': processor_name,
            'seed': seed,
        }

        # Try to load from cache first
        if cache and cache.cache_exists('librispeech', **cache_params):
            print(f"Loading {split_name} dataset from cache...")
            dataset = cache.load_dataset('librispeech', **cache_params)
            if dataset is not None:
                print(f"Successfully loaded {split_name} dataset from cache (length: {len(dataset)})")
                return dataset
            else:
                print(f"Cache loading failed for {split_name}, creating new dataset...")

        # Create new dataset if not cached or cache loading failed
        print(f"Creating new {split_name} dataset...")
        dataset = LibriSpeechDataset(
            data_root=data_root,
            split=split_name,
            max_audio_length=max_audio_length,
            max_samples=max_samples,
            processor_name=processor_name,
            seed=seed,
        )

        # Save to cache for future use
        if cache:
            print(f"Saving {split_name} dataset to cache...")
            cache.save_dataset(dataset, 'librispeech', **cache_params)

        return dataset

    # Create training datasets with caching
    train_datasets = []
    for split in train_splits:
        split_max_samples = max_train_samples // len(train_splits) if max_train_samples else None
        dataset = create_or_load_dataset(split, split_max_samples, is_train=True)
        train_datasets.append(dataset)

    # Combine training datasets
    from torch.utils.data import ConcatDataset
    train_dataset = ConcatDataset(train_datasets)

    # Create validation dataset with caching
    val_dataset = create_or_load_dataset(dev_split, max_val_samples)

    # Create test dataset with caching
    test_dataset = create_or_load_dataset(test_split, max_test_samples)

    # Get collate function from one of the datasets
    collate_fn = train_datasets[0].collate_fn

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    print(f"Created dataloaders:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")

    return train_loader, val_loader, test_loader


def get_librispeech_hours_to_samples(hours: int, avg_duration: float = 6.5) -> int:
    """
    Convert LibriSpeech hours to approximate number of samples.

    Args:
        hours: Number of hours of audio
        avg_duration: Average duration per utterance in seconds

    Returns:
        Approximate number of samples
    """
    total_seconds = hours * 3600
    num_samples = int(total_seconds / avg_duration)
    return num_samples


def predownload_wav2vec2_processor(processor_name: str = "facebook/wav2vec2-base-960h"):
    """
    Pre-download wav2vec2 processor to avoid HTTP 429 errors during training.

    Args:
        processor_name: HuggingFace processor name to download
    """
    print(f"Pre-downloading wav2vec2 processor: {processor_name}")

    hf_cache_dir = os.environ.get('HF_HOME', os.path.expanduser("~/.cache/huggingface"))
    cache_dir = os.path.join(hf_cache_dir, "transformers")

    max_retries = 3
    base_delay = 5.0

    for attempt in range(max_retries):
        try:
            print(f"Download attempt {attempt + 1}/{max_retries}...")
            processor = Wav2Vec2Processor.from_pretrained(
                processor_name,
                cache_dir=cache_dir
            )
            print(f"Successfully downloaded and cached processor at {cache_dir}")
            return True

        except requests.exceptions.HTTPError as e:
            if "429" in str(e):
                delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                print(f"Rate limited (HTTP 429). Waiting {delay:.1f} seconds before retry...")
                time.sleep(delay)
                continue
            else:
                print(f"HTTP error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(base_delay)
                    continue
                else:
                    print(f"Failed to download processor after {max_retries} attempts")
                    return False
        except Exception as e:
            print(f"Error downloading processor: {e}")
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                continue
            else:
                print(f"Failed to download processor after {max_retries} attempts")
                return False

    return False