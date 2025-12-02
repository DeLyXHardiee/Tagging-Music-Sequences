# Run 20251202_114502

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
- Increased Dropout in Residual Blocks from 0.2 to 0.3 to combat slight overfitting.

## Results
- Final Train Loss: 0.9397
- Final Val Loss: 1.0682
- Final Train Acc: 84.04%
- Final Val Acc: 78.80%

--- Validation Set ---
- Validation Accuracy (Chunk): 80.23%
- Validation Precision: 0.8043
- Validation Recall: 0.8023
- Validation F1-Score: 0.7947
- Validation Song-Level Accuracy: 85.56%

--- Test Set ---
- Test Accuracy (Chunk): 77.16%
- Test Precision: 0.7823
- Test Recall: 0.7716
- Test F1-Score: 0.7657
- Test Song-Level Accuracy: 83.00%

## Inference Results (Test Set)
- Test Accuracy: 77.16%
- Test Precision: 0.7823
- Test Recall: 0.7716
- Test F1-Score: 0.7657
- Confusion Matrix: [confusion_matrix_test.png](./confusion_matrix_test.png)
- Batch Predictions: [prediction_batch_test.png](./prediction_batch_test.png)
- Test Song-Level Accuracy: 83.00%
