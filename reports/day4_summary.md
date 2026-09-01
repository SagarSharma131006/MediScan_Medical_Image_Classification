# Day 4 - Custom Dataset and DataLoader

## Objective

The objective of Day 4 was to create a custom PyTorch Dataset and configure DataLoaders for efficient batch-wise loading of the brain MRI dataset.

The DataLoader pipeline was configured with batching, multiple workers, pinned memory, and prefetching.

## Dataset Split

| Dataset | Images |
|---|---:|
| Training | 5,040 |
| Validation | 1,080 |
| Testing | 1,080 |
| Total | 7,200 |

## Custom Dataset

A custom `BrainMRIDataset` class was implemented using PyTorch's `Dataset` interface.

The dataset performs the following operations:

- Loads MRI images from the dataset paths
- Converts images to RGB
- Applies the appropriate transformation pipeline
- Converts class names into integer class indices
- Returns image tensors and corresponding labels

## Transform Pipeline

### Training

The training pipeline includes:

- Resize to 224 × 224
- Random Horizontal Flip
- Random Rotation
- Color Jitter
- Random Affine Transformation
- Conversion to Tensor
- ImageNet normalization

### Validation and Testing

The validation and testing pipeline includes:

- Resize to 224 × 224
- Conversion to Tensor
- ImageNet normalization

## DataLoader Configuration

| Parameter | Value |
|---|---:|
| Batch Size | 32 |
| Number of Workers | 2 |
| Prefetch Factor | 2 |
| Pin Memory | True |
| Training Shuffle | True |
| Validation Shuffle | False |
| Testing Shuffle | False |

## Batch Verification

The training DataLoader successfully produced batches with the following dimensions:

```text
Images shape : torch.Size([32, 3, 224, 224])
Labels shape : torch.Size([32])
Images dtype : torch.float32
Labels dtype : torch.int64
