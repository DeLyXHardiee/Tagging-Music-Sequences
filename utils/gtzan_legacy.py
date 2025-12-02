
"""Legacy GTZAN utilities for older notebooks.

Provides generate_stratified_split and create_dataloaders_from_split so
notebooks depending on the old interface keep working.
"""
import json
import os
import random
import warnings
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio

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
    def __init__(
        self,
        root_dir: str,
        sample_rate: int = 22050,
        duration: int = 30,
        transform: Optional[torch.nn.Module] = None,
        file_list: Optional[List[str]] = None,
    ) -> None:
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.transform = transform
        self.genre_to_idx = {g: i for i, g in enumerate(GENRES)}

        self.files: List[str] = []
        self.labels: List[int] = []

        if file_list is not None:
            for rel_path in file_list:
                abs_path = rel_path if os.path.isabs(rel_path) else os.path.join(root_dir, rel_path)
                if not os.path.isfile(abs_path):
                    continue
                genre = os.path.relpath(abs_path, root_dir).split(os.sep)[0]
                if genre not in self.genre_to_idx:
                    continue
                if self._is_valid_audio(abs_path):
                    self.files.append(abs_path)
                    self.labels.append(self.genre_to_idx[genre])
        else:
            for genre in GENRES:
                genre_path = os.path.join(root_dir, genre)
                if not os.path.isdir(genre_path):
                    continue
                for fname in os.listdir(genre_path):
                    if fname.lower().endswith('.wav'):
                        fpath = os.path.join(genre_path, fname)
                        if self._is_valid_audio(fpath):
                            self.files.append(fpath)
                            self.labels.append(self.genre_to_idx[genre])

        if len(self.files) == 0:
            raise RuntimeError(f"No .wav files found under {root_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = self.files[idx]
        label = self.labels[idx]
        waveform, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        target_len = self.sample_rate * self.duration
        if waveform.shape[1] > target_len:
            waveform = waveform[:, :target_len]
        elif waveform.shape[1] < target_len:
            waveform = torch.nn.functional.pad(waveform, (0, target_len - waveform.shape[1]))
        if self.transform is not None:
            waveform = self.transform(waveform)
        return waveform, label

    @staticmethod
    def _is_valid_audio(file_path: str) -> bool:
        try:
            torchaudio.info(file_path)
            return True
        except (RuntimeError, OSError) as exc:
            warnings.warn(f"Skipping unreadable audio file '{file_path}': {exc}")
            return False


def generate_stratified_split(
    root_dir: str,
    output_path: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Dict[str, List[str]]:
    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    rng = random.Random(seed)
    files_by_genre: Dict[str, List[str]] = {g: [] for g in GENRES}
    for genre in GENRES:
        gpath = os.path.join(root_dir, genre)
        if not os.path.isdir(gpath):
            continue
        for fname in os.listdir(gpath):
            if fname.lower().endswith('.wav'):
                abs_path = os.path.join(gpath, fname)
                rel_path = os.path.relpath(abs_path, root_dir)
                files_by_genre[genre].append(rel_path)

    splits = {"train": [], "val": [], "test": []}
    for genre, paths in files_by_genre.items():
        if not paths:
            continue
        paths = sorted(paths)
        rng.shuffle(paths)
        n = len(paths)
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)
        splits["train"].extend(paths[:n_train])
        splits["val"].extend(paths[n_train:n_train + n_val])
        splits["test"].extend(paths[n_train + n_val:])

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(splits, f, indent=2)
    return splits


def create_dataloaders_from_split(
    root_dir: str,
    split_path: str,
    batch_size: int = 16,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    persistent_workers: Optional[bool] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    with open(split_path, 'r', encoding='utf-8') as f:
        splits = json.load(f)

    if num_workers is None:
        num_workers = 0 if os.name == 'nt' else 2
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    if persistent_workers is None:
        persistent_workers = bool(num_workers > 0)

    def _ds(paths: List[str]) -> GTZANDataset:
        return GTZANDataset(root_dir, file_list=paths)

    train_ds = _ds(splits.get('train', []))
    val_ds = _ds(splits.get('val', []))
    test_ds = _ds(splits.get('test', []))

    def _loader(ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

    return _loader(train_ds, True), _loader(val_ds, False), _loader(test_ds, False)

__all__ = [
    "GENRES",
    "GTZANDataset",
    "generate_stratified_split",
    "create_dataloaders_from_split",
]
