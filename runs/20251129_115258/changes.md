# Run 20251129_115258

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
- Final Train Loss: 1.2174
- Final Val Loss: 1.1539
- Final Train Acc: 75.31%
- Final Val Acc: 76.03%
- Test Accuracy (Chunk): 76.03%
- Test Precision: 0.7805
- Test Recall: 0.7603
- Test F1-Score: 0.7583
- Song-Level Accuracy: 80.50%
