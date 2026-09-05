# MediScan — Day 13 Experiment Tracking

## Tracking Platform

- Platform: Weights & Biases (W&B)
- Project: MediScan-Medical-Image-Classification
- Run: day13-experiment-tracking
- Run ID: xzkmsk5u
- Run URL: https://wandb.ai/sagarsharma131006-panipat-institute-of-engineering-and-t/MediScan-Medical-Image-Classification/runs/xzkmsk5u

## Model

- Architecture: EfficientNet-B0
- Classes: 4
- Input Size: 224×224
- Pretrained: True

## Hyperparameters

| Parameter | Value |
|---|---|
| Learning Rate | 0.002 |
| Batch Size | 32 |
| Dropout | 0.1 |
| Epochs | 10 |
| Optimizer | Adam |
| Loss | CrossEntropyLoss |
| Scheduler | StepLR |
| Augmentation | Weak |

## Dataset

- Dataset: Brain Tumor MRI Dataset
- Training: 5,040
- Validation: 1,080
- Testing: 1,080

## Performance

- Validation Accuracy: 90.09%
- Test Accuracy: 91.20%
- Test Precision: 91.32%
- Test Recall: 91.20%
- Test F1 Score: 91.16%

## Tracked Metrics

- Training loss
- Training accuracy
- Validation loss
- Validation accuracy
- Test accuracy
- Test precision
- Test recall
- Test F1
- Per-class Precision / Recall / F1
- Per-class ROC-AUC

## W&B Artifact

- Name: mediscan-efficientnet-b0-results
- Type: model
- Contains final model, reports and evaluation figures.

## Status

**Day 13 — COMPLETED ✅**
