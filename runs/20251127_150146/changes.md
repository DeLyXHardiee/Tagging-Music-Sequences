# Run 20251127_150146

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
- Implemented Stratified Split to ensure validation set has balanced genre distribution.
- Lowered Learning Rate to 1e-4 to prevent oscillation.
- Added Weight Decay (1e-4) and Gradient Clipping (1.0) for regularization and stability.
- Set fixed seed for reproducibility.

## Results
- Final Train Loss: 0.8695
- Final Val Loss: 1.2050
- Final Train Acc: 73.97%
- Final Val Acc: 61.00%
