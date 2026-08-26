# MediScan — Medical Image Classification

## 🧠 Brain Tumor MRI Classification using Deep Learning

MediScan is a deep learning-based medical image classification project designed to classify brain MRI images into four categories:

- Glioma
- Meningioma
- No Tumor
- Pituitary Tumor

The project focuses on building a complete machine learning pipeline including data preprocessing, exploratory data analysis, model training, evaluation, explainability using Grad-CAM, experiment tracking, and deployment.

---

## 🎯 Project Objective

The objective of MediScan is to develop an image classification system capable of identifying the category of a brain MRI scan.

The project will use transfer learning models and deep learning techniques to achieve reliable classification performance.

> **Important:** This project is intended for educational and research purposes only. It is not a medical diagnostic system and should not be used for clinical decision-making.

---

## 🗂️ Dataset

The project uses the Brain Tumor MRI Dataset containing four classes:

| Class | Description |
|---|---|
| Glioma | Brain tumor category |
| Meningioma | Brain tumor category |
| No Tumor | MRI without detected tumor |
| Pituitary | Pituitary tumor category |

### Dataset Statistics

| Split | Images |
|---|---:|
| Training | 5,600 |
| Testing | 1,600 |
| Total | 7,200 |

Each class initially contains:

- 1,400 training images
- 400 testing images

Therefore, the dataset is initially balanced across the four classes.

---

## 🔍 Day 1 — Dataset Verification

Day 1 focused on setting up the development environment and verifying the dataset.

### Completed Tasks

- [x] Created GitHub repository
- [x] Created project directory structure
- [x] Set up Google Colab
- [x] Enabled GPU acceleration
- [x] Installed PyTorch and required libraries
- [x] Connected Google Drive
- [x] Configured MediScan dataset path
- [x] Verified four dataset classes
- [x] Counted dataset images
- [x] Loaded sample MRI images
- [x] Visualized samples from all four classes
- [x] Checked image dimensions
- [x] Checked for corrupted images
- [x] Recorded dataset statistics

---

## 📊 Day 1 Findings

### Classes

```text
glioma
meningioma
notumor
pituitary

---

# 🔬 Day 2 — EDA & Data Preparation

Day 2 focused on performing exploratory data analysis (EDA), creating the required dataset split, preparing the MRI images for deep learning, and building the PyTorch data pipeline.

## Completed Tasks

* [x] Analyzed class distribution
* [x] Visualized class distribution
* [x] Analyzed image dimensions
* [x] Analyzed pixel intensity
* [x] Visualized representative MRI samples
* [x] Created a stratified 70/15/15 dataset split
* [x] Verified no data leakage
* [x] Created custom PyTorch Dataset
* [x] Resized images to 224 × 224
* [x] Converted grayscale images to RGB
* [x] Added training data augmentation
* [x] Created validation and testing preprocessing pipeline
* [x] Created PyTorch DataLoaders
* [x] Verified final training batch

---

## 📊 Day 2 — Dataset Split

The original dataset contained:

| Original Split |    Images |
| -------------- | --------: |
| Training       |     5,600 |
| Testing        |     1,600 |
| **Total**      | **7,200** |

The assignment requires a **70/15/15 split**, so the original split was reorganized.

### Final Split

| Split      |    Images | Percentage |
| ---------- | --------: | ---------: |
| Training   |     5,040 |        70% |
| Validation |     1,080 |        15% |
| Testing    |     1,080 |        15% |
| **Total**  | **7,200** |   **100%** |

A **stratified split** was used to maintain class representation across all three subsets.

---

## 🔒 Data Leakage Verification

The image paths between the three datasets were compared.

```text
Train ∩ Validation = 0
Train ∩ Test       = 0
Validation ∩ Test  = 0
```

### Result

```text
No data leakage detected ✅
```

This ensures that an image used during training does not appear in validation or testing.

---

## 🧪 Exploratory Data Analysis

### Class Distribution

The dataset contains four classes:

```text
0 → glioma
1 → meningioma
2 → notumor
3 → pituitary
```

Class distribution was analyzed and visualized to verify the representation of each category.

### Image Dimensions

The original MRI images were inspected to understand their dimensions.

Since deep learning models require a fixed input size, all images were resized to:

```text
224 × 224
```

### Pixel Intensity

Pixel intensity values were examined during EDA.

After converting images using PyTorch's `ToTensor()` transformation, pixel values were represented in the range:

```text
0.0 → 1.0
```

---

## 🖼️ Image Preprocessing

The original MRI images are grayscale.

For compatibility with the pretrained transfer-learning models planned for this project, the images were converted from grayscale to RGB.

```text
Grayscale
    ↓
RGB
    ↓
3 Channels
    ↓
224 × 224
    ↓
PyTorch Tensor
```

The resulting tensor format is:

```text
[3, 224, 224]
```

---

## 🔄 Data Augmentation

Data augmentation was applied **only to the training dataset**.

### Training Transformations

```text
Resize → 224 × 224
Random Horizontal Flip
Random Rotation ±10°
Convert to Tensor
```

### Validation/Test Transformations

```text
Resize → 224 × 224
Convert to Tensor
```

Validation and testing images were not randomly augmented to ensure consistent model evaluation.

---

## 🧠 PyTorch Dataset

A custom `BrainTumorDataset` class was created using:

```python
torch.utils.data.Dataset
```

The Dataset handles:

* Image loading
* Grayscale-to-RGB conversion
* Image resizing
* Data augmentation
* Tensor conversion
* Class-to-index mapping

### Class Mapping

```text
glioma    → 0
meningioma → 1
notumor   → 2
pituitary → 3
```

---

## 📦 DataLoaders

PyTorch DataLoaders were created for all three datasets.

Configuration:

```text
Batch Size = 32
```

### Training

```text
shuffle = True
```

### Validation

```text
shuffle = False
```

### Testing

```text
shuffle = False
```

---

## ✅ Final Batch Verification

The final training DataLoader was successfully verified.

```text
Images shape : torch.Size([32, 3, 224, 224])
Labels shape : torch.Size([32])
Images dtype : torch.float32
Labels dtype : torch.int64

Pixel range:
Min: 0.0
Max: 1.0
```

### Result

```text
PyTorch data pipeline verified successfully ✅
```

---

## 📁 Day 2 Notebook

The complete Day 2 implementation is available in:

```text
notebooks/
└── 02_Day2_EDA_and_Data_Preparation.ipynb
```

---

## 📈 Project Progress

```text
Day 1 → Dataset Setup & Verification       ✅
Day 2 → EDA & Data Preparation              ✅
Day 3 → Transfer Learning                   ⏳
Day 4 → Model Training                      ⏳
Day 5 → Evaluation & Comparison             ⏳
Day 6 → Grad-CAM Explainability             ⏳
Day 7 → Experiment Tracking                 ⏳
Day 8 → Deployment                          ⏳
```

---

## 🚀 Next Step

The next stage of MediScan will focus on **transfer learning**.

The planned models are:

* EfficientNet-B0
* ResNet50

The next phase will include:

1. GPU verification
2. Loading pretrained models
3. Modifying classification heads for four classes
4. Defining loss functions
5. Configuring optimizers
6. Training models
7. Monitoring training and validation performance
8. Comparing model performance

---

