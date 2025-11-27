import os
from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchaudio
from sklearn.model_selection import train_test_split
import numpy as np

# Public genres list for reference
GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]


class GTZANDataset(Dataset):
    """PyTorch Dataset for GTZAN music genre classification.

    Expects folder structure:
    root_dir/
      blues/*.wav
      classical/*.wav
      ...
    """

    def __init__(
        self,
        root_dir: str,
        sample_rate: int = 22050,
        duration: int = 30,
        transform: Optional[torch.nn.Module] = None,
        cache_to_memory: bool = False,
    ) -> None:
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.transform = transform
        self.cache_to_memory = cache_to_memory
        self.genres = GENRES
        self.genre_to_idx = {genre: idx for idx, genre in enumerate(self.genres)}

        self.files = []
        self.labels = []
        self.cache = {}

        for genre in self.genres:
            genre_path = os.path.join(root_dir, genre)
            if not os.path.isdir(genre_path):
                continue
            for filename in os.listdir(genre_path):
                if filename.lower().endswith(".wav"):
                    self.files.append(os.path.join(genre_path, filename))
                    self.labels.append(self.genre_to_idx[genre])

        if len(self.files) == 0:
            raise RuntimeError(
                f"No .wav files found under '{root_dir}'. Expected subfolders: {', '.join(self.genres)}"
            )
            
        if self.cache_to_memory:
            print(f"Caching {len(self.files)} audio files to memory...")
            for idx in range(len(self.files)):
                self._load_item(idx)
            print("Caching complete.")

    def _load_item(self, idx: int) -> Tuple[torch.Tensor, int]:
        if self.cache_to_memory and idx in self.cache:
            return self.cache[idx]

        audio_path = self.files[idx]
        label = self.labels[idx]

        # Load audio (C x T)
        waveform, sr = torchaudio.load(audio_path)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # To mono if multi-channel
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Pad/trim to fixed duration
        target_length = self.sample_rate * self.duration
        if waveform.shape[1] > target_length:
            waveform = waveform[:, :target_length]
        elif waveform.shape[1] < target_length:
            padding = target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
            
        if self.cache_to_memory:
            self.cache[idx] = (waveform, label)

        return waveform, label

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        waveform, label = self._load_item(idx)

        if self.transform is not None:
            waveform = self.transform(waveform)

        return waveform, label


class TransformDataset(Dataset):
    """Wrapper to apply transforms to a Subset."""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


class AudioAugmentation:
    """Simple audio augmentation pipeline."""
    def __init__(self, noise_level=0.005, shift_max=0.2):
        self.noise_level = noise_level
        self.shift_max = shift_max

    def __call__(self, waveform):
        # Add Gaussian noise
        if self.noise_level > 0:
            noise = torch.randn_like(waveform) * self.noise_level
            waveform = waveform + noise
        
        # Time shift (roll)
        if self.shift_max > 0:
            shift_amt = int(torch.rand(1).item() * self.shift_max * waveform.shape[1])
            if torch.rand(1).item() > 0.5:
                shift_amt = -shift_amt
            waveform = torch.roll(waveform, shifts=shift_amt, dims=1)
        
        return waveform


def create_dataloaders(
    dataset: Dataset,
    batch_size: int = 16,
    train_split: float = 0.8,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    persistent_workers: Optional[bool] = None,
    train_transform: Optional[object] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create train/val DataLoaders with Windows-safe defaults.

    Args:
        dataset: The base dataset
        batch_size: Batch size
        train_split: Fraction of data to use for training
        num_workers: Number of worker threads
        pin_memory: Whether to pin memory for GPU
        persistent_workers: Keep workers alive
        train_transform: Optional transform to apply ONLY to training set
    
    Notes:
    - On Windows (spawn), using workers > 0 with classes defined in notebooks can crash.
      This helper defaults to num_workers=0 on Windows to be robust.
    """
    if num_workers is None:
        # Default to 0 on Windows to avoid pickling issues in notebooks
        num_workers = 0 if os.name == "nt" else 2

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    if persistent_workers is None:
        persistent_workers = bool(num_workers > 0)

    # Use stratified split to ensure balanced validation set
    try:
        # Extract labels for stratification
        labels = dataset.labels
        indices = np.arange(len(dataset))
        
        train_indices, val_indices = train_test_split(
            indices, 
            train_size=train_split, 
            stratify=labels, 
            random_state=42
        )
        
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        print(f"Created stratified split: {len(train_subset)} train, {len(val_subset)} val")
        
    except Exception as e:
        print(f"Stratified split failed (dataset might not have .labels), falling back to random split: {e}")
        train_size = int(train_split * len(dataset))
        val_size = len(dataset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Apply augmentation if provided
    if train_transform:
        train_dataset = TransformDataset(train_subset, train_transform)
    else:
        train_dataset = train_subset
    
    val_dataset = val_subset

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    return train_loader, val_loader
