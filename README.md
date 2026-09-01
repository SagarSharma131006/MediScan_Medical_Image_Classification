# MediScan - Medical Image Classification

MediScan is a deep learning project focused on classifying brain MRI images into four medical categories: glioma, meningioma, pituitary tumor, and no tumor. The project follows a structured 15-day workflow covering dataset exploration, preprocessing, data augmentation, model training, evaluation, Grad-CAM explainability, and web app deployment.

## Day 1 - Project Setup and Dataset Exploration

### Objective

The goal of Day 1 was to set up the project environment, connect Google Colab with the GitHub repository, load the brain MRI dataset, and perform an initial dataset exploration.

### Work Completed

- Google Colab environment was successfully configured.
- GitHub repository was cloned inside Colab.
- Brain MRI dataset was loaded from Google Drive.
- Dataset was extracted successfully.
- Dataset folder structure was verified.
- Dataset manifest CSV was created.
- Class distribution was analyzed.
- Sample MRI images were visualized and saved.

### Dataset Overview

| Detail | Value |
|---|---|
| Dataset Type | Brain MRI Images |
| Total Images | 7,200 |
| Number of Classes | 4 |
| Training Images | 5,600 |
| Testing Images | 1,600 |

### Classes

| Class Name | Meaning |
|---|---|
| glioma | MRI images showing glioma tumor |
| meningioma | MRI images showing meningioma tumor |
| pituitary | MRI images showing pituitary tumor |
| notumor | MRI images with no tumor |

### Key Insight

The dataset is well-balanced across all four classes. Each class contains an equal number of training and testing images, which is helpful for building a fair classification model. This reduces the risk of the model becoming biased toward one class during training.

### Day 1 Outputs

| Output File | Purpose |
|---|---|
| `data/processed/day1_dataset_manifest.csv` | Stores image paths, labels, and original split information |
| `figures/day1_sample_images.png` | Shows sample MRI images from each class |
| `reports/day1_summary.md` | Contains Day 1 summary and dataset details |

### Day 1 Status

| Task | Status |
|---|---|
| Colab setup completed | Done |
| GitHub repo cloned | Done |
| Dataset extracted | Done |
| Total 7,200 images verified | Done |
| Four classes verified | Done |
| Sample images visualized | Done |
| Day 1 summary saved | Done |

## Important Note

The dataset file `archive.zip` and extracted `data/raw/` folder are not uploaded to GitHub because the dataset is large. The dataset is stored in Google Drive and used directly in Google Colab.

---

# Day 2 - Data Preprocessing and Dataset Split

### Objective

The goal of Day 2 was to analyze the image data, prepare the dataset for model training, and create a stratified 70/15/15 train-validation-test split.

### Work Completed

- Image dimensions and color modes were analyzed.
- Images were prepared for resizing to 224 × 224 pixels.
- Images were converted to RGB format during preprocessing.
- A stratified 70/15/15 train-validation-test split was created.
- Class balance was preserved across all splits.
- Class-to-index mapping was created and saved.

### Dataset Split

| Split | Images |
|---|---:|
| Training | 5,040 |
| Validation | 1,080 |
| Testing | 1,080 |
| Total | 7,200 |

### Class Mapping

| Class | Index |
|---|---:|
| glioma | 0 |
| meningioma | 1 |
| notumor | 2 |
| pituitary | 3 |

### Day 2 Outputs

- `data/processed/day2_train_val_test_split.csv`
- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`
- `data/processed/class_to_idx.json`
- `figures/day2_split_distribution.png`
- `reports/day2_summary.md`

---

# Day 3 - Data Augmentation

### Objective

The goal of Day 3 was to implement data augmentation techniques for the MRI training images and visually verify their effects.

### Work Completed

The following augmentation techniques were implemented:

- Random Horizontal Flip
- Random Rotation
- Color Jitter
- Random Affine Transformation

The augmentation pipeline was created using PyTorch/Torchvision transforms.

### Augmentation Pipeline

```text
Resize
   ↓
Random Horizontal Flip
   ↓
Random Rotation
   ↓
Color Jitter
   ↓
Random Affine Transformation
   ↓
ToTensor

## Day 4 - Custom Dataset and DataLoader

### Objective

The goal of Day 4 was to implement a custom PyTorch Dataset and DataLoaders for efficient batch-wise loading of MRI images.

### Work Completed

- Implemented a custom PyTorch Dataset.
- Loaded training, validation, and testing images.
- Converted images to RGB.
- Applied training and evaluation transformations.
- Converted class names into integer class indices.
- Created PyTorch DataLoaders.
- Configured batch size of 32.
- Configured multiple workers for parallel data loading.
- Enabled pinned memory.
- Enabled DataLoader prefetching.
- Verified complete training DataLoader iteration.

### DataLoader Configuration

| Parameter | Value |
|---|---:|
| Batch Size | 32 |
| Workers | 2 |
| Prefetch Factor | 2 |
| Pin Memory | True |

### Batch Verification

```text
Images shape : torch.Size([32, 3, 224, 224])
Labels shape : torch.Size([32])
Images dtype : torch.float32
Labels dtype : torch.int64

## Day 5 - Base Model Setup

### Objective

The goal of Day 5 was to set up a pre-trained CNN using transfer learning and modify its classifier head for four-class brain MRI classification.

### Work Completed

- Loaded a pre-trained EfficientNet-B0 model.
- Used pre-trained ImageNet weights.
- Inspected the original classifier.
- Replaced the original classifier with a four-class classifier.
- Frozen the EfficientNet-B0 feature extraction layers.
- Moved the model to the CUDA device.
- Verified trainable and frozen parameters.

### Model Configuration

| Parameter | Value |
|---|---|
| Architecture | EfficientNet-B0 |
| Pre-trained | Yes |
| Input Size | 224 × 224 |
| Number of Classes | 4 |
| Device | CUDA |
| Dropout | 0.2 |
| Classifier | Linear(1280 → 4) |

### Class Mapping

| Class | Index |
|---|---:|
| glioma | 0 |
| meningioma | 1 |
| notumor | 2 |
| pituitary | 3 |

### Parameter Verification

| Parameter | Count |
|---|---:|
| Total Parameters | 4,012,672 |
| Trainable Parameters | 5,124 |
| Frozen Parameters | 4,007,548 |

### Day 5 Status

| Task | Status |
|---|---|
| Pre-trained EfficientNet-B0 loaded | Done |
| Classifier head modified | Done |
| Base model frozen | Done |
| CUDA setup | Done |
| Parameter verification | Done |
| Final model verification | Done |
