# Tagging-Music-Sequences

Deep Learning for Music Genre Classification and Tagging using PyTorch

## Overview

This project implements deep learning models (CNN and RNN/LSTM) for automatic music genre classification and tagging. The models can process arbitrary-length audio sequences and classify them into genres or assign multiple tags.

### Supported Datasets

1. **GTZAN**: 1,000 audio tracks (30s each) across 10 genres
2. **MagnaTagATune (MTAT)**: ~25,000 audio clips (29s each) with 188 tags
3. **Free Music Archive (FMA)**: Large-scale dataset with multiple subsets

## Features

- 🎵 Multiple model architectures: CNN, LSTM, GRU, and hybrid CNN-LSTM
- 📊 Support for both single-label (genre) and multi-label (tags) classification
- 🔧 Easy-to-use PyTorch implementations in Jupyter notebooks
- 📈 Training with early stopping and learning rate scheduling
- 🎯 Inference utilities for new audio files
- 📉 Comprehensive evaluation metrics and visualizations

## Project Structure

```
Tagging-Music-Sequences/
├── notebooks/
│   ├── 01_data_loading_gtzan.ipynb    # GTZAN dataset loader
│   ├── 02_data_loading_mtat.ipynb     # MagnaTagATune dataset loader
│   ├── 03_data_loading_fma.ipynb      # FMA dataset loader
│   ├── 04_model_cnn.ipynb             # CNN architectures
│   ├── 05_model_rnn.ipynb             # RNN/LSTM architectures
│   ├── 06_training.ipynb              # Training pipeline
│   └── 07_inference.ipynb             # Inference and prediction
├── data/                               # Dataset directory (user-provided)
├── models/                             # Saved model checkpoints
├── utils/                              # Utility functions
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/DeLyXHardiee/Tagging-Music-Sequences.git
cd Tagging-Music-Sequences
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download datasets (see Dataset Setup section)

## Dataset Setup

### GTZAN

1. Download from: [GTZAN Dataset](http://marsyas.info/downloads/datasets.html)
2. Extract to `data/gtzan/` with structure:
```
data/gtzan/
├── blues/
├── classical/
├── country/
...
```

### MagnaTagATune (MTAT)

1. Download from: [MTAT Dataset](https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset)
2. Extract to:
```
data/mtat/
├── audio/
└── annotations_final.csv
```

### FMA

1. Download from: [FMA Dataset](https://github.com/mdeff/fma)
2. Extract to `data/fma_small/` (or fma_medium, fma_large)
3. Download metadata to `data/fma_metadata/`

## Quick Start

### 1. Load and Explore Data

Open and run `notebooks/01_data_loading_gtzan.ipynb` to explore the GTZAN dataset:

```python
from notebooks.data_loading_gtzan import GTZANDataset, create_dataloaders

dataset = GTZANDataset('data/gtzan', sample_rate=22050, duration=30)
train_loader, val_loader = create_dataloaders(dataset, batch_size=16)
```

### 2. Create a Model

Use `notebooks/04_model_cnn.ipynb` or `notebooks/05_model_rnn.ipynb`:

```python
from notebooks.model_cnn import SimpleCNN

model = SimpleCNN(n_classes=10)
```

### 3. Train the Model

Open `notebooks/06_training.ipynb`:

```python
history = train_model(
    model, train_loader, val_loader,
    num_epochs=50,
    learning_rate=0.001,
    device='cuda',
    save_path='models/best_model.pth'
)
```

### 4. Make Predictions

Use `notebooks/07_inference.ipynb`:

```python
predicted_genre, confidence = predict_genre(
    model, 'path/to/audio.wav', genre_names, device='cuda'
)
```

## Model Architectures

### CNN Models

- **SimpleCNN**: Basic CNN with 4 conv layers + global pooling
- **DeepCNN**: Deeper architecture with residual-like blocks

### RNN Models

- **MusicLSTM**: Bidirectional LSTM for sequential modeling
- **MusicGRU**: GRU-based alternative to LSTM
- **CNNLSTM**: Hybrid model combining CNN feature extraction with LSTM temporal modeling

All models use mel-spectrogram representations computed on-the-fly.

## Training Tips

1. **Start small**: Use a subset of data to verify your pipeline works
2. **Monitor overfitting**: Watch validation loss and use early stopping
3. **Adjust learning rate**: Use learning rate scheduling (ReduceLROnPlateau)
4. **Data augmentation**: Consider time-stretching, pitch-shifting for better generalization
5. **GPU memory**: Reduce batch size if you encounter OOM errors

## Evaluation Metrics

- **Single-label**: Accuracy, Precision, Recall, F1-Score
- **Multi-label**: Precision@k, Recall@k, AUC-ROC, mAP

## Results

Model performance will vary based on:
- Dataset quality and size
- Model architecture and hyperparameters
- Training duration and regularization
- Audio preprocessing choices

Typical GTZAN accuracy: 70-90% depending on model complexity.

## Advanced Usage

### Custom Datasets

Create your own dataset class following the pattern in the data loading notebooks:

```python
class CustomDataset(Dataset):
    def __init__(self, audio_dir, sample_rate=22050):
        # Your implementation
        pass
    
    def __getitem__(self, idx):
        # Return (waveform, label)
        pass
```

### Model Customization

Modify model architectures in the model notebooks to experiment with:
- Different numbers of layers
- Alternative activation functions
- Various pooling strategies
- Attention mechanisms

## Citation

If you use this code in your research, please cite the relevant datasets:

- GTZAN: Tzanetakis, G., & Cook, P. (2002). Musical genre classification of audio signals.
- MTAT: Law, E., et al. (2009). Evaluation of algorithms using games: The case of music tagging.
- FMA: Defferrard, M., et al. (2017). FMA: A dataset for music analysis.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- Librosa and torchaudio for audio processing utilities
- Dataset creators for making their data publicly available

## Contact

For questions or issues, please open an issue on GitHub.