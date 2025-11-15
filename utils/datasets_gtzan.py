import os
from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio

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
    ) -> None:
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.transform = transform
        self.genres = GENRES
        self.genre_to_idx = {genre: idx for idx, genre in enumerate(self.genres)}

        self.files = []
        self.labels = []

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

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
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

        if self.transform is not None:
            waveform = self.transform(waveform)

        return waveform, label


def create_dataloaders(
    dataset: Dataset,
    batch_size: int = 16,
    train_split: float = 0.8,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    persistent_workers: Optional[bool] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create train/val DataLoaders with Windows-safe defaults.

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

    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

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
