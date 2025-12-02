# utils/advanced_cnn.py

import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


def random_crop(waveform: torch.Tensor, sample_rate: int, crop_seconds: float = 10.0) -> torch.Tensor:
    """
    Randomly crop a single waveform (1, T) to crop_seconds if longer; otherwise return as-is.
    """
    target = int(sample_rate * crop_seconds)
    if waveform.shape[1] <= target:
        return waveform
    start = random.randint(0, waveform.shape[1] - target)
    return waveform[:, start:start + target]


def augment_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    training: bool = True,
    max_shift_seconds: float = 0.8,
    noise_std: float = 0.0003,
    crop_seconds: float = 10.0,
) -> torch.Tensor:
    """
    Time-shift, crop, and inject small Gaussian noise in waveform space for a SINGLE example (1, T).
    """
    if training:
        waveform = random_crop(waveform, sample_rate, crop_seconds)

        if max_shift_seconds > 0:
            max_shift = int(max_shift_seconds * sample_rate)
            if max_shift > 0:
                shift = torch.randint(-max_shift, max_shift + 1, (1,), device=waveform.device).item()
                waveform = torch.roll(waveform, shifts=shift, dims=-1)

        if noise_std > 0:
            waveform = waveform + noise_std * torch.randn_like(waveform)

        waveform = waveform.clamp(-1.0, 1.0)
    return waveform


def augment_waveform_batch(
    waveforms: torch.Tensor,
    sample_rate: int,
    training: bool = True,
    max_shift_seconds: float = 0.8,
    noise_std: float = 0.0003,
    crop_seconds: float = 10.0,
) -> torch.Tensor:
    """
    Apply augment_waveform independently to each sample in a batch.

    waveforms: (B, 1, T)
    Returns a batch (B, 1, T') where all samples are padded to the same length.
    """
    if not training:
        return waveforms

    B, C, T = waveforms.shape
    assert C == 1, "Expected mono audio (B, 1, T)"

    processed = []
    for b in range(B):
        w = waveforms[b]  # (1, T)
        w = augment_waveform(
            w,
            sample_rate,
            training=True,
            max_shift_seconds=max_shift_seconds,
            noise_std=noise_std,
            crop_seconds=crop_seconds,
        )
        processed.append(w)

    # Pad to the max length in the batch (just in case cropping produced different lengths)
    max_len = max(w.shape[1] for w in processed)
    out = []
    for w in processed:
        if w.shape[1] < max_len:
            pad = max_len - w.shape[1]
            w = F.pad(w, (0, pad))
        out.append(w)

    return torch.stack(out, dim=0)


class SpecAugment(nn.Module):
    """Apply SpecAugment-style freq/time masking with some probability."""

    def __init__(self, freq_mask_param: int = 18, time_mask_param: int = 24,
                 num_masks: int = 1, p: float = 0.35):
        super().__init__()
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)
        self.num_masks = num_masks
        self.p = p

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return spec
        out = spec
        for _ in range(self.num_masks):
            out = self.freq_mask(out)
            out = self.time_mask(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.08):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout2d(dropout)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                      stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

        squeeze_channels = max(out_channels // 4, 8)
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, squeeze_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeeze_channels, out_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))

        out = out * self.scale(out)
        out = F.relu(out + residual)
        return out


class AdvancedCNN(nn.Module):
    def __init__(self, n_classes: int = 10, sample_rate: int = 22050, n_mels: int = 128,
                 spec_augment: Optional[SpecAugment] = None):
        super().__init__()
        self.sample_rate = sample_rate
        self.spec_augment = spec_augment

        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=n_mels,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        self.stem = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),

            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.ModuleList([
            ResidualBlock(64, 128, stride=2, dropout=0.08),
            ResidualBlock(128, 192, stride=2, dropout=0.12),
            ResidualBlock(192, 256, stride=2, dropout=0.15),
        ])

        self.conv_final = nn.Conv2d(256, 320, kernel_size=3, padding=1)
        self.bn_final = nn.BatchNorm2d(320)

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Linear(320, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T)
        x = self.mel_spec(x)         # (B, 1, n_mels, time)
        x = x.clamp(min=1e-10)       # Prevent log(0) -> -inf
        x = self.amplitude_to_db(x)

        if self.training and self.spec_augment is not None:
            x = self.spec_augment(x)

        x = self.stem(x)

        for block in self.blocks:
            x = block(x)

        x = F.relu(self.bn_final(self.conv_final(x)))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        return self.head(x)


__all__ = [
    "AdvancedCNN",
    "SpecAugment",
    "augment_waveform_batch",
    "augment_waveform",
    "random_crop",
]
