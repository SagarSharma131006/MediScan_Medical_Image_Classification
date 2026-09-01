# Day 6 - Training Loop Setup

## Objective

The objective of Day 6 was to configure and verify the training components required for the MediScan brain MRI classification model.

The training pipeline was connected with the EfficientNet-B0 model, loss function, optimizer, learning-rate scheduler, and PyTorch DataLoaders.

## Work Completed

- Restored the pretrained EfficientNet-B0 model.
- Restored the existing training, validation, and testing datasets.
- Restored PyTorch DataLoaders.
- Verified the input batch shape.
- Verified the model forward pass.
- Configured CrossEntropyLoss.
- Configured Adam optimizer.
- Configured StepLR learning-rate scheduler.
- Verified compatibility between model outputs and target labels.
- Successfully calculated the training loss.

## Training Configuration

| Component | Configuration |
|---|---|
| Model | EfficientNet-B0 |
| Pretrained | Yes |
| Number of Classes | 4 |
| Device | CUDA |
| Batch Size | 32 |
| Loss Function | CrossEntropyLoss |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Scheduler | StepLR |
| Scheduler Step Size | 5 |
| Scheduler Gamma | 0.1 |

## Dataset Configuration

| Dataset | Images |
|---|---:|
| Training | 5,040 |
| Validation | 1,080 |
| Testing | 1,080 |

## Verification Results

```text
========================================
DAY 6 TRAINING LOOP VERIFICATION
========================================

Model              : EfficientNet-B0
Device             : cuda
Loss Function      : CrossEntropyLoss
Optimizer          : Adam
Learning Rate      : 0.001
Scheduler          : StepLR
Scheduler Step Size: 5
Scheduler Gamma    : 0.1

Batch verification:
Input shape        : torch.Size([32, 3, 224, 224])
Output shape       : torch.Size([32, 4])
Labels shape       : torch.Size([32])
Labels dtype       : torch.int64
Loss               : 1.47794771194458

========================================
DAY 6 VERIFICATION PASSED
========================================
