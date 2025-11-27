# Run 20251127_162412

## Configuration
- Batch Size: 32
- Learning Rate: 0.0001
- Epochs: 50
- Device: cuda
- Data Strategy: Chunking (3s chunks, 50% overlap)
- Augmentation: Noise=0.01, Shift=0.3
- Optimization: In-memory caching + Mixed Precision (AMP)
- Stability: Seed=42, Weight Decay=1e-4 (Standard), Gradient Clipping=1.0
- Data Split: Stratified (Balanced Validation Set)

## Changes
- Implemented Data Chunking: Splitting 30s audio into 3s chunks with 50% overlap. This increases dataset size by ~19x.
- Relaxed Regularization: Reduced Weight Decay back to 1e-4 and Dropout to 0.5 as we have more data.
- Adjusted SpecAugment: Reduced TimeMasking to 40 to account for shorter 3s chunks.
- Maintained Reduced Model Capacity: Keeping 32-filter base to prevent overfitting.

## Results
- Final Train Loss: 0.5350
- Final Val Loss: 1.4683
- Final Train Acc: 99.37%
- Final Val Acc: 64.63%
