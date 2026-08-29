# 🧠 MediScan — Medical Image Classification

## 🔬 Brain Tumor MRI Classification using Deep Learning

MediScan is a deep learning-based medical image classification project designed to classify brain MRI images into four categories:

- Glioma
- Meningioma
- No Tumor
- Pituitary Tumor

The project focuses on building a complete deep learning pipeline including:

- Dataset preparation
- Exploratory Data Analysis
- Data preprocessing
- Transfer learning
- Model training
- Model evaluation
- Explainability using Grad-CAM
- Experiment tracking
- Deployment

> ⚠️ **Important:** This project is intended for educational and research purposes only. It is not a medical diagnostic system and should not be used for clinical decision-making.

---

# 🎯 Project Objective

The objective of MediScan is to develop an image classification system capable of classifying brain MRI images into four categories using deep learning and transfer learning.

The project uses two pretrained CNN architectures:

- EfficientNet-B0
- ResNet50

The models are trained using a stratified dataset split and evaluated on unseen test data.

---

# 🗂️ Dataset

The project uses the **Brain Tumor MRI Dataset** containing four classes:

| Class | Description |
|---|---|
| Glioma | Brain tumor category |
| Meningioma | Brain tumor category |
| No Tumor | MRI without detected tumor |
| Pituitary | Pituitary tumor category |

## Original Dataset

| Original Split | Images |
|---|---:|
| Training | 5,600 |
| Testing | 1,600 |
| **Total** | **7,200** |

Each class initially contains:

- 1,400 training images
- 400 testing images

Therefore, the original dataset is balanced across the four classes.

---

# 📊 Final Dataset Split

The assignment requires a **70/15/15 split**, so the original dataset was reorganized.

| Split | Images | Percentage |
|---|---:|---:|
| Training | 5,040 | 70% |
| Validation | 1,080 | 15% |
| Testing | 1,080 | 15% |
| **Total** | **7,200** | **100%** |

A **stratified split** was used to maintain class representation across all three subsets.

---

# 🏷️ Class Mapping

```text
0 → glioma
1 → meningioma
2 → notumor
3 → pituitary
```

---

# 📅 Project Progress

```text
Day 1 → Dataset Setup & Verification        ✅
Day 2 → EDA & Data Preparation              ✅
Day 3 → Transfer Learning                   ✅
Day 4 → Model Training                      ✅
Day 5 → Evaluation & Comparison             ✅
Day 6 → Grad-CAM Explainability             ⏳
Day 7 → Experiment Tracking                 ⏳
Day 8 → Deployment                          ⏳
```

---

# 🔰 Day 1 — Dataset Setup & Verification

Day 1 focused on setting up the MediScan development environment, configuring Google Colab, preparing the dataset, and verifying the initial dataset structure.

## Completed Tasks

- [x] Created GitHub repository
- [x] Created project directory structure
- [x] Set up Google Colab
- [x] Enabled GPU acceleration
- [x] Installed required libraries
- [x] Connected Google Drive
- [x] Configured MediScan dataset path
- [x] Verified dataset classes
- [x] Counted dataset images
- [x] Loaded sample MRI images
- [x] Visualized representative MRI samples
- [x] Checked image dimensions
- [x] Checked dataset structure
- [x] Recorded dataset statistics

## Dataset Classes

```text
glioma
meningioma
notumor
pituitary
```

## Initial Dataset Statistics

```text
Training Images : 5600
Testing Images  : 1600
Total Images    : 7200
Classes         : 4
```

---

## 📁 Day 1 Notebook

```text
notebooks/
└── 01_Day1_Setup_and_EDA.ipynb
```

---

# 🔬 Day 2 — EDA & Data Preparation

Day 2 focused on performing exploratory data analysis, creating the required dataset split, preparing the MRI images for deep learning, and building the PyTorch data pipeline.

## Completed Tasks

- [x] Analyzed class distribution
- [x] Visualized class distribution
- [x] Analyzed image dimensions
- [x] Analyzed pixel intensity
- [x] Visualized representative MRI samples
- [x] Created a stratified 70/15/15 dataset split
- [x] Verified no data leakage
- [x] Created custom PyTorch Dataset
- [x] Resized images to 224 × 224
- [x] Converted grayscale images to RGB
- [x] Added training data augmentation
- [x] Created validation and testing preprocessing pipeline
- [x] Created PyTorch DataLoaders
- [x] Verified final training batch

---

## 📊 Day 2 — Dataset Split

The original dataset contained:

| Original Split | Images |
|---|---:|
| Training | 5,600 |
| Testing | 1,600 |
| **Total** | **7,200** |

The assignment requires a **70/15/15 split**, so the original split was reorganized.

### Final Split

| Split | Images | Percentage |
|---|---:|---:|
| Training | 5,040 | 70% |
| Validation | 1,080 | 15% |
| Testing | 1,080 | 15% |
| **Total** | **7,200** | **100%** |

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

For compatibility with the pretrained transfer-learning models, the images were converted from grayscale to RGB.

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

- Image loading
- Grayscale-to-RGB conversion
- Image resizing
- Data augmentation
- Tensor conversion
- Class-to-index mapping

### Class Mapping

```text
glioma     → 0
meningioma → 1
notumor    → 2
pituitary  → 3
```

---

## 📦 DataLoaders

PyTorch DataLoaders were created for all three datasets.

### Configuration

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

```text
notebooks/
└── 02_Day2_EDA_and_Data_Preparation.ipynb
```

---

# 🧠 Day 3 — Transfer Learning

Day 3 focused on preparing pretrained deep learning models for the four-class brain MRI classification task.

The two models selected were:

- EfficientNet-B0
- ResNet50

Both models use pretrained ImageNet weights.

---

## Completed Tasks

- [x] Verified dataset sizes
- [x] Verified four-class classification setup
- [x] Loaded pretrained EfficientNet-B0
- [x] Loaded pretrained ResNet50
- [x] Modified classification heads for 4 classes
- [x] Frozen pretrained backbone layers
- [x] Configured trainable classification layers
- [x] Defined CrossEntropyLoss
- [x] Configured Adam optimizers
- [x] Verified model outputs
- [x] Verified input/output tensor shapes

---

## 📊 Dataset Used

```text
Training samples   : 5040
Validation samples : 1080
Testing samples    : 1080
Number of classes  : 4
```

---

## 🏗️ Model Configuration

### EfficientNet-B0

```text
Pretrained ImageNet Model
        ↓
Frozen Feature Extractor
        ↓
Dropout
        ↓
Linear Layer
        ↓
4 Classes
```

### ResNet50

```text
Pretrained ImageNet Model
        ↓
Frozen Feature Extractor
        ↓
Fully Connected Layer
        ↓
4 Classes
```

---

## 🔢 Trainable Parameters

```text
EfficientNet-B0 trainable parameters : 5124
ResNet50 trainable parameters       : 8196
```

Only the classification layers were trained while the pretrained feature extraction layers remained frozen.

---

## ⚙️ Loss Function

Both models used:

```text
CrossEntropyLoss
```

---

## 🚀 Optimizers

Both models used the Adam optimizer.

| Model | Optimizer | Learning Rate |
|---|---|---:|
| EfficientNet-B0 | Adam | 0.001 |
| ResNet50 | Adam | 0.001 |

---

## ✅ Model Output Verification

Both models successfully produced four-class outputs.

```text
EfficientNet-B0
Input shape : torch.Size([32, 3, 224, 224])
Output shape: torch.Size([32, 4])
Output dtype: torch.float32
```

```text
ResNet50
Input shape : torch.Size([32, 3, 224, 224])
Output shape: torch.Size([32, 4])
Output dtype: torch.float32
```

### Result

```text
DAY 3 COMPLETED SUCCESSFULLY! ✅
```

---

## 📁 Day 3 Notebook

```text
notebooks/
└── 03_Day3_Transfer_Learning.ipynb
```

---

# 🚀 Day 4 — Model Training

Day 4 focused on training the transfer learning models prepared during Day 3.

The two models trained were:

- EfficientNet-B0
- ResNet50

Both models were trained using the prepared **70/15/15 dataset split** and evaluated using the validation dataset during training.

---

## Completed Tasks

- [x] Recreated pretrained EfficientNet-B0
- [x] Recreated pretrained ResNet50
- [x] Configured both models for 4-class classification
- [x] Loaded pretrained ImageNet weights
- [x] Frozen pretrained backbone layers
- [x] Kept classification layers trainable
- [x] Defined CrossEntropyLoss
- [x] Configured Adam optimizers
- [x] Verified training pipeline with a 1-epoch test
- [x] Trained EfficientNet-B0 for 10 epochs
- [x] Trained ResNet50 for 10 epochs
- [x] Recorded training and validation loss
- [x] Recorded training and validation accuracy
- [x] Generated loss curves
- [x] Generated accuracy curves
- [x] Saved best model weights
- [x] Saved training history

---

## ⚙️ Training Configuration

The same basic training configuration was used for both models.

| Parameter | Value |
|---|---|
| Dataset Split | 70/15/15 |
| Training Images | 5,040 |
| Validation Images | 1,080 |
| Testing Images | 1,080 |
| Number of Classes | 4 |
| Input Size | 224 × 224 |
| Batch Size | 32 |
| Number of Epochs | 10 |
| Loss Function | CrossEntropyLoss |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Pretrained Weights | ImageNet |
| Training Device | Google Colab GPU |

---

# 🧠 Transfer Learning Setup

Both models use pretrained ImageNet weights.

The pretrained feature extraction layers were frozen, while the final classification layers were trained for the four MediScan classes.

```text
Pretrained Model
       ↓
Feature Extraction
       ↓
Frozen Layers
       ↓
Trainable Classification Layer
       ↓
4-Class Output
```

---

# 🧪 Training Verification

Before full training, a **1-epoch training test** was performed to verify the complete training pipeline.

### EfficientNet-B0

```text
Epoch [1/1]
Train Loss : 0.6605
Train Acc  : 78.95%
Val Loss   : 0.4742
Val Acc    : 84.63%

Best Validation Accuracy: 84.63%
```

The successful test confirmed that the training pipeline was working correctly.

---

# 📈 EfficientNet-B0 Training Results

EfficientNet-B0 was trained for 10 epochs.

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---:|---:|---:|---:|---:|
| 1 | 0.6631 | 78.21% | 0.4712 | 85.46% |
| 2 | 0.4387 | 84.80% | 0.4124 | 86.11% |
| 3 | 0.3930 | 86.49% | 0.3884 | 86.76% |
| 4 | 0.3771 | 87.04% | 0.3749 | 87.69% |
| 5 | 0.3699 | 87.30% | 0.3517 | 87.78% |
| 6 | 0.3537 | 87.02% | 0.3500 | 88.70% |
| 7 | 0.3370 | 88.15% | 0.3310 | 88.61% |
| 8 | 0.3308 | 88.35% | 0.3256 | 88.61% |
| 9 | 0.3293 | 88.21% | 0.3361 | 88.52% |
| 10 | 0.3299 | 88.12% | 0.3249 | **88.80%** |

### Best Validation Performance

```text
Best Validation Accuracy: 88.80%
```

---

# 📈 ResNet50 Training Results

ResNet50 was also trained for 10 epochs.

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---:|---:|---:|---:|---:|
| 1 | 0.7555 | 76.33% | 0.6367 | 81.57% |
| 2 | 0.4781 | 84.35% | 0.5031 | 84.26% |
| 3 | 0.4245 | 85.36% | 0.4351 | 86.11% |
| 4 | 0.3922 | 86.77% | 0.4220 | 86.11% |
| 5 | 0.3573 | 87.40% | 0.4111 | 85.65% |
| 6 | 0.3489 | 88.13% | 0.3935 | 86.48% |
| 7 | 0.3263 | 88.89% | 0.3797 | 86.76% |
| 8 | 0.3261 | 88.29% | 0.3579 | 88.15% |
| 9 | 0.3012 | 89.60% | 0.3632 | 87.59% |
| 10 | 0.3026 | 89.23% | 0.3391 | **88.33%** |

### Best Validation Performance

```text
Best Validation Accuracy: 88.33%
```

---

# 📊 Training Comparison

| Model | Best Validation Accuracy |
|---|---:|
| **EfficientNet-B0** | **88.80%** |
| ResNet50 | 88.33% |

EfficientNet-B0 achieved a slightly higher validation accuracy during training.

---

# 💾 Saved Model Checkpoints

The best model weights and training histories were saved in Google Drive.

```text
/content/drive/MyDrive/MediScan/models/
│
├── efficientnet_b0_best.pth
├── efficientnet_b0_history.json
├── resnet50_best.pth
└── resnet50_history.json
```

The model checkpoint files were kept outside GitHub because they can be large.

---

## 📁 Day 4 Notebook

```text
notebooks/
└── 04_Day4_Model_Training.ipynb
```

---

# 📊 Day 5 — Model Evaluation

Day 5 focused on evaluating the trained transfer learning models on the **unseen testing dataset**.

The models trained during Day 4 were evaluated without retraining:

- EfficientNet-B0
- ResNet50

Evaluation was performed using the **1,080 test images** created through the stratified 70/15/15 dataset split.

---

## Completed Tasks

- [x] Loaded the trained EfficientNet-B0 checkpoint
- [x] Loaded the trained ResNet50 checkpoint
- [x] Recreated the model architectures
- [x] Verified model checkpoints
- [x] Created the test DataLoader
- [x] Verified test batch shape and data types
- [x] Generated predictions for the complete test dataset
- [x] Calculated test accuracy
- [x] Generated classification reports
- [x] Calculated precision, recall, and F1-score
- [x] Generated confusion matrices
- [x] Compared EfficientNet-B0 and ResNet50
- [x] Saved EfficientNet-B0 evaluation results
- [x] Saved ResNet50 evaluation results
- [x] Identified the better-performing model

---

# 🧪 Test DataLoader Verification

The Day 5 test DataLoader was successfully verified.

```text
Day 5 Test DataLoader Verification
==================================================
Images shape : torch.Size([32, 3, 224, 224])
Labels shape : torch.Size([32])
Images dtype : torch.float32
Labels dtype : torch.int64

Pixel range:
Min: 0.0
Max: 1.0
```

---

# 🧠 EfficientNet-B0 Evaluation

The trained EfficientNet-B0 checkpoint was successfully loaded.

```text
EfficientNet-B0 Checkpoint Verification
==================================================
Input shape : torch.Size([32, 3, 224, 224])
Output shape: torch.Size([32, 4])
Output dtype: torch.float32
```

The model generated predictions for all:

```text
1080 / 1080 test samples
```

No retraining was performed during evaluation.

---

# 📈 EfficientNet-B0 Test Performance

```text
EfficientNet-B0 Test Evaluation
============================================================
Test Accuracy: 90.09%
```

## Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| Glioma | 0.9467 | 0.8556 | 0.8988 | 270 |
| Meningioma | 0.8169 | 0.8593 | 0.8375 | 270 |
| No Tumor | 0.9072 | 0.9778 | 0.9412 | 270 |
| Pituitary | 0.9425 | 0.9111 | 0.9266 | 270 |
| **Accuracy** | | | **0.9009** | **1080** |
| **Macro Avg** | **0.9033** | **0.9009** | **0.9010** | **1080** |
| **Weighted Avg** | **0.9033** | **0.9009** | **0.9010** | **1080** |

### Overall Result

```text
Test Accuracy : 90.09%
Macro F1      : 90.10%
```

---

# 🔲 EfficientNet-B0 Confusion Matrix

```text
[[231  31   7   1]
 [ 12 232  15  11]
 [  1   2 264   3]
 [  0  19   5 246]]
```

Rows represent actual classes and columns represent predicted classes.

```text
                 Predicted
              G   M   N   P

Actual G     231  31   7   1
Actual M      12 232  15  11
Actual N       1   2 264   3
Actual P       0  19   5 246
```

Where:

```text
G = Glioma
M = Meningioma
N = No Tumor
P = Pituitary
```

Correct predictions:

```text
Glioma     → 231
Meningioma → 232
No Tumor   → 264
Pituitary  → 246
```

The highest recall was achieved for the **No Tumor** class:

```text
Recall = 97.78%
```

---

# 💾 EfficientNet-B0 Evaluation Results

The evaluation results were saved as:

```text
results/
└── efficientnet_b0_evaluation.json
```

Google Drive location:

```text
/content/drive/MyDrive/MediScan/results/efficientnet_b0_evaluation.json
```

---

# 🧠 ResNet50 Evaluation

The trained ResNet50 checkpoint was loaded and evaluated on the same unseen testing dataset.

No retraining was performed.

---

# 📈 ResNet50 Test Performance

```text
ResNet50 Test Evaluation
============================================================
Test Accuracy: 89.26%
```

## Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| Glioma | 0.9137 | 0.8630 | 0.8876 | 270 |
| Meningioma | 0.8681 | 0.7556 | 0.8079 | 270 |
| No Tumor | 0.9288 | 0.9667 | 0.9474 | 270 |
| Pituitary | 0.8608 | 0.9852 | 0.9188 | 270 |
| **Accuracy** | | | **0.8926** | **1080** |
| **Macro Avg** | **0.8929** | **0.8926** | **0.8904** | **1080** |
| **Weighted Avg** | **0.8929** | **0.8926** | **0.8904** | **1080** |

### Overall Result

```text
Test Accuracy : 89.26%
Macro F1      : 89.04%
```

---

# 🔲 ResNet50 Confusion Matrix

```text
[[233  25   6   6]
 [ 20 204  13  33]
 [  2   3 261   4]
 [  0   3   1 266]]
```

Rows represent actual classes and columns represent predicted classes.

```text
                 Predicted
              G   M   N   P

Actual G     233  25   6   6
Actual M      20 204  13  33
Actual N       2   3 261   4
Actual P       0   3   1 266
```

Where:

```text
G = Glioma
M = Meningioma
N = No Tumor
P = Pituitary
```

Correct predictions:

```text
Glioma     → 233
Meningioma → 204
No Tumor   → 261
Pituitary  → 266
```

The highest recall was achieved for the **Pituitary** class:

```text
Recall = 98.52%
```

---

# 💾 ResNet50 Evaluation Results

The evaluation results were saved as:

```text
results/
└── resnet50_evaluation.json
```

Google Drive location:

```text
/content/drive/MyDrive/MediScan/results/resnet50_evaluation.json
```

---

# 🏆 Model Comparison

Both models were evaluated using the same unseen test dataset.

```text
MediScan — Model Comparison
============================================================

Model                    Accuracy     Macro F1
------------------------------------------------------------
EfficientNet-B0            90.09%       90.10%
ResNet50                   89.26%       89.04%
```

## Comparison Table

| Model | Test Accuracy | Macro F1 |
|---|---:|---:|
| **EfficientNet-B0** | **90.09%** | **90.10%** |
| ResNet50 | 89.26% | 89.04% |

---

# 🥇 Best Performing Model

Based on the final test evaluation:

```text
Best Model: EfficientNet-B0
```

### Performance Difference

```text
Accuracy Difference:
90.09% - 89.26% = 0.83 percentage points

Macro F1 Difference:
90.10% - 89.04% = 1.06 percentage points
```

Therefore, EfficientNet-B0 performed slightly better than ResNet50 on the MediScan test dataset.

---

# 📊 Final Day 5 Results

```text
============================================================
              MEDISCAN MODEL EVALUATION
============================================================

EfficientNet-B0
------------------------------------------------------------
Test Accuracy : 90.09%
Macro F1      : 90.10%

ResNet50
------------------------------------------------------------
Test Accuracy : 89.26%
Macro F1      : 89.04%

------------------------------------------------------------
Best Model: EfficientNet-B0
------------------------------------------------------------
```

---

# 📁 Day 5 Files

### Notebook

```text
notebooks/
└── 05_Day5_Model_Evaluation.ipynb
```

### Evaluation Results

```text
results/
├── efficientnet_b0_evaluation.json
└── resnet50_evaluation.json
```

---

# 🔄 Complete Evaluation Pipeline

```text
Trained Model Checkpoint
          ↓
Recreate Model Architecture
          ↓
Load Best Model Weights
          ↓
Set Model to Evaluation Mode
          ↓
Load Test Dataset
          ↓
Generate Predictions
          ↓
Compare Predictions with True Labels
          ↓
Calculate Accuracy
          ↓
Generate Classification Report
          ↓
Generate Confusion Matrix
          ↓
Save Evaluation Results
          ↓
Compare Models
```

---

# 📂 Project Structure

The current GitHub project structure is:

```text
MediScan/
│
├── app/
│   └── .gitkeep
│
├── data/
│   └── .gitkeep
│
├── models/
│   └── .gitkeep
│
├── notebooks/
│   ├── 01_Day1_Setup_and_EDA.ipynb
│   ├── 02_Day2_EDA_and_Data_Preparation.ipynb
│   ├── 03_Day3_Transfer_Learning.ipynb
│   ├── 04_Day4_Model_Training.ipynb
│   └── 05_Day5_Model_Evaluation.ipynb
│
├── results/
│   ├── .gitkeep
│   ├── efficientnet_b0_evaluation.json
│   └── resnet50_evaluation.json
│
├── src/
│   └── .gitkeep
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

# 💾 Model Files

The trained model checkpoints are stored in Google Drive and are intentionally not included in the GitHub repository.

```text
models/
├── efficientnet_b0_best.pth
└── resnet50_best.pth
```

The `.gitignore` file is configured to prevent model checkpoint files from being accidentally committed.

```text
# Model checkpoints
*.pth
*.pt

# Dataset
data/

# Python
__pycache__/
*.pyc

# Jupyter
.ipynb_checkpoints/

# Environment
.env
```

---

# 🛠️ Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Pillow
- OpenCV
- tqdm
- Grad-CAM
- Weights & Biases
- Google Colab
- Google Drive
- GitHub

---

# 🧰 Development Environment

The project is developed primarily using:

```text
Google Colab
       ↓
GPU Acceleration
       ↓
PyTorch
       ↓
Google Drive
       ↓
GitHub
```

---

# 🚀 Future Work

The upcoming stages of MediScan will include:

## Day 6 — Grad-CAM Explainability

- Load the best-performing model
- Select representative test images
- Generate Grad-CAM heatmaps
- Visualize model attention
- Overlay heatmaps on MRI images
- Analyze model predictions

## Day 7 — Experiment Tracking

- Configure Weights & Biases
- Track training metrics
- Log experiments
- Compare model runs

## Day 8 — Deployment

- Build inference pipeline
- Create Streamlit application
- Load trained model
- Upload MRI image
- Display predicted class
- Display prediction confidence
- Integrate explainability

---

# 📈 Project Status

```text
Day 1 → Dataset Setup & Verification        ✅
Day 2 → EDA & Data Preparation              ✅
Day 3 → Transfer Learning                   ✅
Day 4 → Model Training                      ✅
Day 5 → Evaluation & Comparison             ✅
Day 6 → Grad-CAM Explainability             ⏳
Day 7 → Experiment Tracking                 ⏳
Day 8 → Deployment                          ⏳
```

---

# 🏁 Current Achievement

After five completed stages, MediScan has successfully:

```text
Dataset
   ↓
EDA
   ↓
70/15/15 Stratified Split
   ↓
PyTorch Data Pipeline
   ↓
Transfer Learning
   ↓
EfficientNet-B0 + ResNet50
   ↓
10-Epoch Training
   ↓
Test Evaluation
   ↓
Model Comparison
```

### Current Best Model

```text
EfficientNet-B0
```

### Current Best Test Performance

```text
Test Accuracy : 90.09%
Macro F1      : 90.10%
```

---

# ⚠️ Disclaimer

MediScan is an educational deep learning project.

The predictions generated by this project should **not** be considered medical diagnoses or used for clinical decision-making.

A qualified medical professional should always be consulted for actual medical diagnosis and treatment.

---

# 👨‍💻 Author

**Sagar Sharma**

B.Tech CSE — Artificial Intelligence & Machine Learning

---

# ⭐ Project Progress

```text
MediScan — Medical Image Classification

Day 1  ✅ Dataset Setup & Verification
Day 2  ✅ EDA & Data Preparation
Day 3  ✅ Transfer Learning
Day 4  ✅ Model Training
Day 5  ✅ Evaluation & Comparison
Day 6  ⏳ Grad-CAM Explainability
Day 7  ⏳ Experiment Tracking
Day 8  ⏳ Deployment
```

**More updates coming as the project progresses. 🚀**
