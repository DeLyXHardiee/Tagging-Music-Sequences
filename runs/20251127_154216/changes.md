# Run 20251127_154216

## Configuration
- Batch Size: 32
- Learning Rate: 0.0001
- Epochs: 50
- Device: cuda
- Augmentation: Noise=0.005, Shift=0.2
- Optimization: In-memory caching + Mixed Precision (AMP)
- Stability: Seed=42, Weight Decay=1e-4, Gradient Clipping=1.0
- Data Split: Stratified (Balanced Validation Set)

## Changes
- Used new model with deeper complexity.

## Results
- Final Train Loss: 0.1129
- Final Val Loss: 1.4449
- Final Train Acc: 97.75%
- Final Val Acc: 70.00%
