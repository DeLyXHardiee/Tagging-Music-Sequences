# Run 20251127_143203

## Configuration
- Batch Size: 32
- Learning Rate: 0.001
- Epochs: 20
- Device: cuda
- Augmentation: Noise=0.005, Shift=0.2
- Optimization: In-memory caching + Mixed Precision (AMP)

## Changes
- Added data augmentation (noise + time shift) and increased batch size to 32 to stabilize training.

## Results
- Final Train Loss: 0.7694
- Final Val Loss: 1.6303
- Final Train Acc: 75.09%
- Final Val Acc: 50.00%

Training still isn't stable.