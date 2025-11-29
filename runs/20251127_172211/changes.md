# Run 20251127_172211

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
- Added more dropout layers to the ImprovedCNN model to reduce overfitting and removed layer 4 in the model.
## Results
- Final Train Loss: 0.7059
- Final Val Loss: 1.4643
- Final Train Acc: 94.64%
- Final Val Acc: 67.53%
