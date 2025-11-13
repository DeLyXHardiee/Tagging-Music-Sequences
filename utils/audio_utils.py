"""Audio processing utilities."""

import torch
import torchaudio
import numpy as np


def load_audio(audio_path, sample_rate=22050, duration=None, mono=True):
    """
    Load and preprocess audio file.
    
    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate
        duration: Duration in seconds (None for full audio)
        mono: Convert to mono if True
        
    Returns:
        waveform: Audio tensor (1, samples)
        sr: Sample rate
    """
    waveform, sr = torchaudio.load(audio_path)
    
    # Resample if necessary
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
        sr = sample_rate
    
    # Convert to mono
    if mono and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Truncate or pad to duration
    if duration is not None:
        target_length = sample_rate * duration
        if waveform.shape[1] > target_length:
            waveform = waveform[:, :target_length]
        elif waveform.shape[1] < target_length:
            padding = target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
    
    return waveform, sr


def compute_mel_spectrogram(waveform, sample_rate=22050, n_mels=128, 
                           n_fft=2048, hop_length=512):
    """
    Compute mel-spectrogram from waveform.
    
    Args:
        waveform: Audio tensor
        sample_rate: Sample rate
        n_mels: Number of mel bands
        n_fft: FFT size
        hop_length: Hop length
        
    Returns:
        mel_spec_db: Mel-spectrogram in dB
    """
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
    
    mel_spec = mel_transform(waveform)
    mel_spec_db = amplitude_to_db(mel_spec)
    
    return mel_spec_db
