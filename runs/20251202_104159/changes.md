# Run 20251202_104159

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
- Added a missing test split: Train 80%, Validation 10%, Test 10%

## Results
- Final Train Loss: 0.9345
- Final Val Loss: 1.0572
- Final Train Acc: 84.32%
- Final Val Acc: 78.83%

--- Validation Set ---
- Validation Accuracy (Chunk): 78.83%
- Validation Precision: 0.7897
- Validation Recall: 0.7883
- Validation F1-Score: 0.7806
- Validation Song-Level Accuracy: 83.33%

## Inference Results (Test Set)
- Test Accuracy: 78.16%
- Test Precision: 0.7932
- Test Recall: 0.7816
- Test F1-Score: 0.7786
- Confusion Matrix: [confusion_matrix_test.png](./confusion_matrix_test.png)
- Batch Predictions: [prediction_batch_test.png](./prediction_batch_test.png)
- Test Song-Level Accuracy: 82.00%
