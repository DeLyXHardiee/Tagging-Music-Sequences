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
- Added mixup augmentation which will mix inputs and labelsduring training forcing the model to learn more robust features and reducing overfitting.
- Added song-level voting which aggregates the predictions of all chunks belonging to a single song (using soft voting/averaging logits) to produce a final song-level prediction (expected to have a higher accuracy).

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

The mixup helped a lot with overfitting. The training and validation follows each other much closer. The validation accuracy is at an all time high and the new song-level accuracy is even higher.