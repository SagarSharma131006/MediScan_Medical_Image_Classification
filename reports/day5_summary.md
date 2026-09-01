# Day 5 - Base Model Setup

## Objective

The objective of Day 5 was to set up a pre-trained CNN model using transfer learning and modify its classifier head for the four-class brain MRI classification task.

EfficientNet-B0 was selected as the primary architecture.

## Model Architecture

- Model: EfficientNet-B0
- Pre-trained weights: Yes
- Framework: PyTorch / torchvision
- Input image size: 224 × 224
- Number of output classes: 4

## Class Mapping

| Class | Index |
|---|---:|
| glioma | 0 |
| meningioma | 1 |
| notumor | 2 |
| pituitary | 3 |

## Classifier Modification

The original ImageNet classifier was replaced with a custom classifier for the four MRI classes.

Original final classifier:

```text
Linear(in_features=1280, out_features=1000)
