"""Visualization utilities for audio and predictions."""

import matplotlib.pyplot as plt
import numpy as np


def plot_waveform(waveform, sample_rate=22050, title="Waveform", save_path=None):
    """Plot audio waveform."""
    plt.figure(figsize=(12, 4))
    plt.plot(waveform[0].numpy())
    plt.title(title)
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_spectrogram(spectrogram, title="Spectrogram", save_path=None):
    """Plot spectrogram."""
    plt.figure(figsize=(12, 6))
    plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.colorbar(label='Amplitude (dB)')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_predictions(probabilities, class_names, top_k=None, save_path=None):
    """
    Plot prediction probabilities as horizontal bar chart.
    
    Args:
        probabilities: Array of probabilities
        class_names: List of class names
        top_k: Show only top k predictions (None for all)
        save_path: Path to save the plot
    """
    if top_k is not None:
        indices = np.argsort(probabilities)[-top_k:][::-1]
        probs = probabilities[indices]
        names = [class_names[i] for i in indices]
    else:
        probs = probabilities
        names = class_names
    
    plt.figure(figsize=(10, max(6, len(names) * 0.3)))
    bars = plt.barh(names, probs, color='steelblue')
    
    # Highlight top prediction
    bars[0].set_color('coral')
    
    plt.xlabel('Probability')
    plt.title('Prediction Probabilities')
    plt.xlim([0, 1])
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (name, prob) in enumerate(zip(names, probs)):
        plt.text(prob + 0.01, i, f'{prob:.3f}', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_training_history(history, metrics=['loss', 'accuracy'], save_path=None):
    """
    Plot training history.
    
    Args:
        history: Dictionary with training history
        metrics: List of metrics to plot
        save_path: Path to save the plot
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 4))
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        train_key = f'train_{metric}'
        val_key = f'val_{metric}'
        
        if train_key in history:
            ax.plot(history[train_key], label=f'Train {metric.capitalize()}')
        if val_key in history:
            ax.plot(history[val_key], label=f'Val {metric.capitalize()}')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
