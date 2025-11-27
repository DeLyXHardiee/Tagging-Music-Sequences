# Run 20251127_160422

## Configuration
- Batch Size: 32
- Learning Rate: 0.0001
- Epochs: 50
- Device: cuda
- Augmentation: Noise=0.01, Shift=0.3 (Increased)
- Optimization: In-memory caching + Mixed Precision (AMP)
- Stability: Seed=42, Weight Decay=1e-3 (Increased), Gradient Clipping=1.0
- Data Split: Stratified (Balanced Validation Set)

## Changes
- Reduced Model Capacity: Reduced ResNet channel sizes (32, 64, 128, 256) to prevent overfitting on small dataset.
- Increased Regularization: Increased Dropout to 0.6 and Weight Decay to 1e-3.
- Enhanced Augmentation: Increased SpecAugment masking parameters and Audio Augmentation (Noise/Shift).
- Implemented Stratified Split to ensure validation set has balanced genre distribution.
- Lowered Learning Rate to 1e-4 to prevent oscillation.
- Set fixed seed for reproducibility.

## Results
- Final Train Loss: 0.9922
- Final Val Loss: 3.5007
- Final Train Acc: 81.23%
- Final Val Acc: 49.50%

Still overfitting, early stopping after 10 epochs because validation wasn't improving.