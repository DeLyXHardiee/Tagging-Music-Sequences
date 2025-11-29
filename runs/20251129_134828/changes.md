# Run 20251129_134828

## Configuration
- Batch Size: 32
- Learning Rate: 0.001
- Epochs: 50
- Device: cuda
- Data Strategy: Chunking (3s chunks, 50% overlap)
- Augmentation: Noise=0.01, Shift=0.3
- Optimization: In-memory caching + Mixed Precision (AMP)
- Stability: Seed=42, Weight Decay=1e-4 (Standard), Gradient Clipping=1.0
- Data Split: Stratified (Balanced Validation Set)

## Changes
- Increased model capacity by restoring the 4th residual layer and double the channel depth (up to 512 channels)
- Updated scheduler to use OneCycleLR (SOTA for CNN models). It starts with a low learning rate, ramps up to a high one, and then anneals down to near zero.
- Switched from Adam optimizer to AdamW with higher weight decay (0.01). AdamW decouples weight decay from the gradient update, which usually leads to better generalization.

## Results
- Final Train Loss: 0.9455
- Final Val Loss: 1.0872
- Final Train Acc: 83.75%
- Final Val Acc: 80.21%
- Test Accuracy (Chunk): 80.21%
- Test Precision: 0.8102
- Test Recall: 0.8021
- Test F1-Score: 0.7961
- Song-Level Accuracy: 85.50%
