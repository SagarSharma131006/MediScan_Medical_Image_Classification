# 🧠 MediScan — Medical Image Classification

> **Deep Learning-based Brain MRI Classification using PyTorch and EfficientNet-B0**

MediScan is a deep learning project focused on classifying brain MRI images into four categories:

- 🧠 **Glioma**
- 🧠 **Meningioma**
- 🧠 **Pituitary Tumor**
- ✅ **No Tumor**

The project follows a structured **15-day deep learning workflow**, covering:

**Dataset Exploration → Preprocessing → Data Augmentation → Data Loading → Transfer Learning → Model Training → Evaluation → Explainability → Deployment**

The implementation is being developed using **Python, PyTorch, Torchvision, Google Colab, CUDA, and EfficientNet-B0**.

---

# 📌 Project Overview

Medical image classification can assist in the preliminary analysis of MRI scans by automatically identifying patterns associated with different tumor categories.

MediScan aims to build a complete end-to-end image classification pipeline that can:

1. Load and analyze brain MRI images.
2. Prepare and preprocess the dataset.
3. Apply controlled image augmentation.
4. Efficiently load images using PyTorch DataLoaders.
5. Train a transfer-learning based CNN.
6. Evaluate classification performance.
7. Generate explainable predictions using Grad-CAM.
8. Deploy the trained model through a web application.

---

# 🎯 Project Goals

The main goals of MediScan are:

- Build a complete medical image classification pipeline.
- Understand and prepare a real-world MRI image dataset.
- Implement proper train-validation-test splitting.
- Reduce overfitting using data augmentation.
- Use transfer learning with a pretrained CNN.
- Train an efficient four-class classifier.
- Evaluate the model using multiple performance metrics.
- Understand model decisions using Grad-CAM.
- Deploy the final model as an interactive application.

---

# 🗂️ Dataset

The project uses a Brain MRI image dataset containing four classes:

| Class | Description |
|---|---|
| `glioma` | MRI images showing glioma tumor |
| `meningioma` | MRI images showing meningioma tumor |
| `pituitary` | MRI images showing pituitary tumor |
| `notumor` | MRI images showing no tumor |

The original dataset contains:

| Detail | Value |
|---|---:|
| Total Images | 7,200 |
| Number of Classes | 4 |
| Original Training Images | 5,600 |
| Original Testing Images | 1,600 |

> **Dataset Note:** The dataset used in this project was selected independently from the workflow specification. The original dataset structure was preserved initially and then reorganized into the required train-validation-test pipeline.

---

# 🏗️ Project Architecture

The overall MediScan pipeline is:

**MRI Images**

↓  

**Dataset Exploration**

↓

**Preprocessing**

↓

**70/15/15 Dataset Split**

↓

**Data Augmentation**

↓

**Custom PyTorch Dataset**

↓

**PyTorch DataLoader**

↓

**EfficientNet-B0**

↓

**Four-Class Classifier**

↓

**Model Training**

↓

**Model Evaluation**

↓

**Grad-CAM Explainability**

↓

**Streamlit Deployment**

---

# 📅 15-Day Development Roadmap

| Day | Task | Status |
|---|---|---|
| Day 1 | Project Setup & Dataset Exploration | ✅ Completed |
| Day 2 | Data Preprocessing & Dataset Split | ✅ Completed |
| Day 3 | Data Augmentation | ✅ Completed |
| Day 4 | Custom Dataset & DataLoader | ✅ Completed |
| Day 5 | Transfer Learning & Base Model | ✅ Completed |
| Day 6 | Training Loop Setup | ✅ Completed |
| Day 7 | Model Training | 🔄 Upcoming |
| Day 8 | Validation & Model Monitoring | 🔄 Upcoming |
| Day 9 | Model Evaluation | 🔄 Upcoming |
| Day 10 | Confusion Matrix & Classification Report | 🔄 Upcoming |
| Day 11 | Model Improvement & Fine-Tuning | 🔄 Upcoming |
| Day 12 | Grad-CAM Explainability | 🔄 Upcoming |
| Day 13 | Model Saving & Inference | 🔄 Upcoming |
| Day 14 | Streamlit Web Application | 🔄 Upcoming |
| Day 15 | Final Integration & Documentation | 🔄 Upcoming |

---

# 📅 Day 1 — Project Setup and Dataset Exploration

## 🎯 Objective

The goal of Day 1 was to set up the project environment, connect Google Colab with the GitHub repository, load the brain MRI dataset, and perform initial dataset exploration.

## ✅ Work Completed

- Configured the Google Colab environment.
- Connected Google Drive with Colab.
- Cloned the GitHub repository.
- Loaded the Brain MRI dataset.
- Extracted and verified the dataset.
- Verified the dataset folder structure.
- Created a dataset manifest.
- Analyzed class distribution.
- Visualized sample MRI images.
- Saved Day 1 outputs and documentation.

## 📊 Original Dataset Overview

| Detail | Value |
|---|---:|
| Dataset Type | Brain MRI Images |
| Total Images | 7,200 |
| Number of Classes | 4 |
| Training Images | 5,600 |
| Testing Images | 1,600 |

## 🧠 Classes

| Class Name | Meaning |
|---|---|
| `glioma` | MRI images showing glioma tumor |
| `meningioma` | MRI images showing meningioma tumor |
| `pituitary` | MRI images showing pituitary tumor |
| `notumor` | MRI images with no tumor |

## 🔍 Key Insight

The original dataset is balanced across the four classes.

Each class contains an equal number of images in the original training and testing sets. This provides a balanced starting point for classification and helps reduce class imbalance during model development.

## 📁 Day 1 Outputs

| Output | Purpose |
|---|---|
| `data/processed/day1_dataset_manifest.csv` | Stores image paths, labels, and original split information |
| `figures/day1_sample_images.png` | Sample MRI images from the dataset |
| `reports/day1_summary.md` | Day 1 documentation |

## ✅ Day 1 Status

| Task | Status |
|---|---|
| Colab setup | ✅ Done |
| GitHub repository setup | ✅ Done |
| Dataset extraction | ✅ Done |
| 7,200 images verified | ✅ Done |
| Four classes verified | ✅ Done |
| Sample images visualized | ✅ Done |
| Dataset manifest created | ✅ Done |
| Day 1 summary saved | ✅ Done |

---

# 📅 Day 2 — Data Preprocessing and Dataset Split

## 🎯 Objective

The goal of Day 2 was to analyze the image data, prepare the dataset for deep learning, and create a stratified **70/15/15 train-validation-test split**.

## ✅ Work Completed

- Analyzed image dimensions.
- Analyzed image color modes.
- Prepared images for resizing.
- Converted images to RGB during preprocessing.
- Created a stratified 70/15/15 split.
- Preserved class balance across all splits.
- Created a class-to-index mapping.
- Saved train, validation, and test CSV files.
- Verified the dataset split.

## 📊 Dataset Split

| Split | Images |
|---|---:|
| Training | 5,040 |
| Validation | 1,080 |
| Testing | 1,080 |
| **Total** | **7,200** |

## 🔢 Class Mapping

| Class | Index |
|---|---:|
| `glioma` | 0 |
| `meningioma` | 1 |
| `notumor` | 2 |
| `pituitary` | 3 |

## 📁 Day 2 Outputs

- `data/processed/day2_train_val_test_split.csv`
- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`
- `data/processed/class_to_idx.json`
- `figures/day2_split_distribution.png`
- `reports/day2_summary.md`

## ⚠️ Preprocessing Note

The current pipeline uses the image normalization required for the pretrained EfficientNet-B0 model during the later training stages.

The initial Day 2 preprocessing stage did not apply normalization as a permanent dataset transformation. Normalization is applied in the PyTorch transformation pipeline used by the model.

---

# 📅 Day 3 — Data Augmentation

## 🎯 Objective

The goal of Day 3 was to implement controlled data augmentation techniques to improve the model's ability to generalize to variations in MRI images.

## ✅ Augmentation Techniques

The training pipeline includes:

- Random Horizontal Flip
- Random Rotation
- Color Jitter
- Random Affine Transformation

These transformations are applied only to the training data.

Validation and testing images use deterministic preprocessing without random augmentation.

## 🔄 Augmentation Pipeline

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
       ↓
    Normalize

## 🧪 Training Transform

The training transformation uses:

- Resize to `224 × 224`
- Random Horizontal Flip
- Random Rotation up to ±15°
- Color Jitter
- Random Affine transformation
- Conversion to tensor
- ImageNet normalization

## 🧪 Evaluation Transform

Validation and testing use:

- Resize to `224 × 224`
- Conversion to tensor
- ImageNet normalization

No random augmentation is applied during validation or testing.

## 📁 Day 3 Outputs

| Output | Purpose |
|---|---|
| `figures/day3_augmentation_comparison.png` | Original vs augmented image comparison |
| `figures/day3_combined_augmentation.png` | Visualization of augmentation effects |
| `reports/day3_summary.md` | Day 3 documentation |

## ✅ Day 3 Status

| Task | Status |
|---|---|
| Training augmentation implemented | ✅ Done |
| Evaluation transform implemented | ✅ Done |
| Augmented samples visualized | ✅ Done |
| Augmentation pipeline verified | ✅ Done |
| Day 3 summary saved | ✅ Done |

---

# 📅 Day 4 — Custom Dataset and DataLoader

## 🎯 Objective

The goal of Day 4 was to implement a custom PyTorch Dataset and DataLoaders for efficient batch-wise loading of MRI images.

## ✅ Work Completed

- Implemented a custom PyTorch Dataset.
- Loaded training images.
- Loaded validation images.
- Loaded testing images.
- Converted images to RGB.
- Applied training transformations.
- Applied evaluation transformations.
- Converted class names into integer class indices.
- Created PyTorch DataLoaders.
- Configured batch size.
- Configured multiple workers.
- Enabled pinned memory.
- Enabled DataLoader prefetching.
- Verified DataLoader iteration.

## ⚙️ DataLoader Configuration

| Parameter | Value |
|---|---:|
| Batch Size | 32 |
| Number of Workers | 2 |
| Prefetch Factor | 2 |
| Pin Memory | True |
| Training Shuffle | True |

## 📊 Dataset Sizes

| Dataset | Images |
|---|---:|
| Training | 5,040 |
| Validation | 1,080 |
| Testing | 1,080 |

## 🧪 Batch Verification

    Images shape : torch.Size([32, 3, 224, 224])
    Labels shape : torch.Size([32])
    Images dtype : torch.float32
    Labels dtype : torch.int64

## 🔍 Input Configuration

| Parameter | Value |
|---|---:|
| Batch Size | 32 |
| Channels | 3 |
| Image Height | 224 |
| Image Width | 224 |

## ✅ Day 4 Verification

The complete training DataLoader was successfully iterated through all **158 batches**.

**Day 4 DataLoader Verification: PASSED**

## 📁 Day 4 Output

- `notebooks/Day_04_Custom_Dataset_and_DataLoader.ipynb`
- `reports/day4_summary.md`

---

# 📅 Day 5 — Base Model Setup

## 🎯 Objective

The goal of Day 5 was to configure a pretrained CNN using transfer learning and modify its classifier for four-class brain MRI classification.

## 🧠 Model

The project uses **EfficientNet-B0** with pretrained ImageNet weights.

The pretrained convolutional feature extraction layers are frozen, while a new classifier head is trained for the four MRI classes.

## ✅ Work Completed

- Loaded pretrained EfficientNet-B0.
- Loaded ImageNet pretrained weights.
- Inspected the original classifier.
- Replaced the original classifier.
- Configured the classifier for four classes.
- Frozen the feature extraction layers.
- Moved the model to CUDA.
- Verified trainable parameters.
- Verified frozen parameters.

## ⚙️ Model Configuration

| Parameter | Value |
|---|---|
| Architecture | EfficientNet-B0 |
| Pretrained | Yes |
| Pretrained Weights | ImageNet |
| Input Size | 224 × 224 |
| Number of Classes | 4 |
| Device | CUDA |
| Dropout | 0.2 |
| Classifier | Linear(1280 → 4) |

## 🏷️ Class Mapping

| Class | Index |
|---|---:|
| `glioma` | 0 |
| `meningioma` | 1 |
| `notumor` | 2 |
| `pituitary` | 3 |

## 📊 Parameter Verification

| Parameter | Count |
|---|---:|
| Total Parameters | 4,012,672 |
| Trainable Parameters | 5,124 |
| Frozen Parameters | 4,007,548 |

## 🧩 Classifier Architecture

    EfficientNet-B0
          ↓
    Feature Extractor
          ↓
    Global Pooling
          ↓
    Dropout (0.2)
          ↓
    Linear (1280 → 4)
          ↓
    Class Predictions

## ✅ Day 5 Status

| Task | Status |
|---|---|
| EfficientNet-B0 loaded | ✅ Done |
| ImageNet weights loaded | ✅ Done |
| Classifier replaced | ✅ Done |
| Feature layers frozen | ✅ Done |
| CUDA configured | ✅ Done |
| Parameter verification | ✅ Done |
| Base model verification | ✅ Done |

---

# 📅 Day 6 — Training Loop Setup

## 🎯 Objective

The goal of Day 6 was to configure and verify the components required for training the EfficientNet-B0 model.

## ✅ Work Completed

- Restored the pretrained EfficientNet-B0 model.
- Restored the dataset pipeline.
- Restored training, validation, and testing datasets.
- Restored PyTorch DataLoaders.
- Configured CrossEntropyLoss.
- Configured Adam optimizer.
- Configured StepLR scheduler.
- Verified the model forward pass.
- Verified output and label compatibility.
- Successfully calculated the training loss.

## ⚙️ Training Configuration

| Detail | Value |
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
| Step Size | 5 |
| Gamma | 0.1 |

## 📊 Dataset Configuration

| Dataset | Images |
|---|---:|
| Training | 5,040 |
| Validation | 1,080 |
| Testing | 1,080 |

## 🔬 Forward Pass Verification

    Input shape  : torch.Size([32, 3, 224, 224])
    Output shape : torch.Size([32, 4])
    Labels shape : torch.Size([32])
    Labels dtype : torch.int64
    Loss         : 1.47794771194458

The model successfully accepts a batch of 32 RGB MRI images and produces four output logits corresponding to the four target classes.

## 🧩 Training Pipeline

    MRI Batch
        ↓
    EfficientNet-B0
        ↓
    4-Class Logits
        ↓
    CrossEntropyLoss
        ↓
    Adam Optimizer
        ↓
    StepLR Scheduler

## ✅ Day 6 Verification

    ========================================
    DAY 6 TRAINING LOOP VERIFICATION
    ========================================

    Model              : EfficientNet-B0
    Device             : cuda
    Loss Function      : CrossEntropyLoss
    Optimizer          : Adam
    Learning Rate      : 0.001
    Scheduler           : StepLR
    Scheduler Step Size: 5
    Scheduler Gamma     : 0.1

    Batch verification:
    Input shape        : torch.Size([32, 3, 224, 224])
    Output shape       : torch.Size([32, 4])
    Labels shape       : torch.Size([32])
    Labels dtype       : torch.int64
    Loss               : 1.47794771194458

    ========================================
    DAY 6 VERIFICATION PASSED
    ========================================

## 📁 Day 6 Output

- `notebooks/Day_06_Training_Loop.ipynb`
- `reports/day6_summary.md`

---

# 📊 Current Project Status

| Component | Status |
|---|---|
| Dataset setup | ✅ Completed |
| Dataset exploration | ✅ Completed |
| Train/Validation/Test split | ✅ Completed |
| Data augmentation | ✅ Completed |
| Custom Dataset | ✅ Completed |
| DataLoader | ✅ Completed |
| EfficientNet-B0 | ✅ Completed |
| Transfer learning setup | ✅ Completed |
| Training components | ✅ Completed |
| Actual model training | 🔄 Next |
| Model evaluation | 🔄 Upcoming |
| Grad-CAM | 🔄 Upcoming |
| Deployment | 🔄 Upcoming |

---

# 🗃️ Repository Structure

    MediScan_Medical_Image_Classification/
    │
    ├── app/
    │
    ├── data/
    │   └── processed/
    │       ├── class_to_idx.json
    │       ├── day1_dataset_manifest.csv
    │       ├── day2_image_info.csv
    │       ├── day2_train_val_test_split.csv
    │       ├── train.csv
    │       ├── val.csv
    │       └── test.csv
    │
    ├── figures/
    │   ├── day1_sample_images.png
    │   ├── day2_split_distribution.png
    │   ├── day3_augmentation_comparison.png
    │   └── day3_combined_augmentation.png
    │
    ├── models/
    │
    ├── notebooks/
    │   ├── Day_01_Project_Setup_and_Exploration.ipynb
    │   ├── Day_02_Preprocessing_and_Split.ipynb
    │   ├── Day_03_Data_Augmentation.ipynb
    │   ├── Day_04_Custom_Dataset_and_DataLoader.ipynb
    │   ├── Day_05_Base_Model_Setup.ipynb
    │   └── Day_06_Training_Loop.ipynb
    │
    ├── reports/
    │   ├── day1_summary.md
    │   ├── day2_summary.md
    │   ├── day3_summary.md
    │   ├── day4_summary.md
    │   ├── day5_summary.md
    │   └── day6_summary.md
    │
    ├── src/
    │
    ├── .gitignore
    ├── README.md
    └── requirements.txt

---

# 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| PyTorch | Deep Learning Framework |
| Torchvision | Computer Vision & Model Utilities |
| EfficientNet-B0 | Transfer Learning Model |
| CUDA | GPU Acceleration |
| Google Colab | Development Environment |
| NumPy | Numerical Computing |
| Pandas | Data Processing |
| Matplotlib | Visualization |
| Seaborn | Data Visualization |
| Scikit-learn | Evaluation & Data Splitting |
| Pillow | Image Processing |
| Grad-CAM | Model Explainability |
| Streamlit | Web Application |
| Git & GitHub | Version Control |

---

# 🚫 Dataset and Large Files

The original dataset is not uploaded to GitHub because of its large size.

The following files/directories are intentionally excluded:

- `archive.zip`
- `data/raw/`
- Trained model files such as `.pt` and `.pth`

The dataset is stored separately in Google Drive and accessed through Google Colab during development.

The repository contains the processed metadata, notebooks, figures, reports, source code, and project documentation required to reproduce the workflow.

---

# 🔐 Reproducibility

The project is being developed using a day-by-day workflow.

Each completed day contains:

- A dedicated notebook.
- Verification outputs.
- A project report.
- Relevant generated figures or metadata.

This structure makes it easier to track the development process and verify every stage before moving to the next stage.

---

# 🚀 Upcoming Work

The next stages of MediScan will focus on:

### Day 7
Train the EfficientNet-B0 model.

### Day 8
Monitor training and validation performance.

### Day 9
Evaluate the trained model.

### Day 10
Generate confusion matrix and classification report.

### Day 11
Fine-tune the model and improve performance.

### Day 12
Implement Grad-CAM for model explainability.

### Day 13
Save the final model and build inference functionality.

### Day 14
Develop the Streamlit web application.

### Day 15
Integrate the complete pipeline and finalize documentation.

---

# 🎯 Final Project Goal

The final goal of MediScan is to demonstrate a complete end-to-end deep learning workflow for brain MRI image classification:

    Data
      ↓
    Exploration
      ↓
    Preprocessing
      ↓
    Dataset Split
      ↓
    Augmentation
      ↓
    DataLoader
      ↓
    Transfer Learning
      ↓
    Model Training
      ↓
    Evaluation
      ↓
    Explainability
      ↓
    Deployment
      ↓
    ⭐ MediScan

---

# 🧠 From MRI Images to Explainable Deep Learning

MediScan is designed as a structured deep learning project demonstrating the complete journey from raw medical images to an explainable image classification system.

The project combines:

**Medical Imaging + Computer Vision + Transfer Learning + Model Evaluation + Explainable AI + Deployment**

---

# ⚠️ Disclaimer

MediScan is an educational and research-oriented deep learning project.

It is **not a medical diagnostic system** and should not be used as a substitute for professional medical diagnosis, clinical evaluation, or medical advice.

---

# 👨‍💻 Project Development

**Project:** MediScan — Medical Image Classification

**Focus:** Brain MRI Image Classification

**Framework:** PyTorch

**Model:** EfficientNet-B0

**Classes:** 4

**Workflow:** 15-Day Deep Learning Project

**Development Environment:** Google Colab + CUDA

---

## 📅 Day 7 — Model Training

### 🎯 Objective

Train the pretrained EfficientNet-B0 model and monitor training and validation performance over multiple epochs.

### 🔧 Training Configuration

- **Architecture:** EfficientNet-B0
- **Pretrained Weights:** ImageNet
- **Number of Classes:** 4
- **Trainable Parameters:** 5,124
- **Frozen Parameters:** 4,007,548
- **Training Samples:** 5,040
- **Validation Samples:** 1,080
- **Batch Size:** 32
- **Epochs:** 10
- **Loss Function:** CrossEntropyLoss
- **Optimizer:** Adam
- **Initial Learning Rate:** 0.001
- **Scheduler:** StepLR
- **Step Size:** 5
- **Gamma:** 0.1
- **Device:** CUDA / NVIDIA Tesla T4

### 📊 Training Results

| Metric | Result |
|---|---:|
| Best Validation Accuracy | **87.22%** |
| Best Epoch | **6** |
| Final Training Accuracy | **85.73%** |
| Final Validation Accuracy | **86.67%** |
| Best Validation Loss | **0.3943** |

### 📈 Learning Rate Schedule

- Epochs 1–5: `0.001`
- Epochs 6–10: `0.0001`

The learning rate was reduced by the StepLR scheduler after epoch 5.

### 🔍 Observations

- Training loss decreased from **0.6854** to **0.3948**.
- Validation loss decreased overall from **0.5241** to **0.4072**.
- Validation accuracy improved from **82.96%** to a maximum of **87.22%**.
- The project target of **>85% accuracy** was achieved.
- Training and validation curves were generated to monitor model performance.
- The test dataset was **not used during Day 7 training**.

### 📁 Day 7 Artifacts

```text
figures/
├── day7_loss_curve.png
└── day7_accuracy_curve.png

models/
└── efficientnet_b0_best_day7.pth

reports/
├── day7_summary.md
├── day7_training_history.csv
└── day7_training_history.json
```
