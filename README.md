🧠 MediScan — Medical Image Classification

Deep Learning-based Brain MRI Classification using PyTorch and EfficientNet-B0

MediScan is an end-to-end deep learning project for classifying brain MRI images into four categories:

🧠 Glioma

🧠 Meningioma

🧠 Pituitary Tumor

✅ No Tumor

The project follows a structured 15-day workflow covering data preparation, augmentation, transfer learning, training, evaluation, explainability, and deployment.

🎯 Project Overview

Goal: Build and evaluate a four-class brain MRI image classification system using transfer learning.

Core pipeline:

MRI Images → Preprocessing → 70/15/15 Split → Augmentation → PyTorch DataLoader → EfficientNet-B0 → Training → Evaluation → Grad-CAM → Deployment

Current Model

Architecture: EfficientNet-B0

Pretrained weights: ImageNet

Input: 224 × 224 RGB

Classes: 4

Framework: PyTorch

GPU: CUDA / NVIDIA Tesla T4

Dataset: 7,200 brain MRI images

🗂️ Dataset

Class

Images

Glioma

1,800

Meningioma

1,800

No Tumor

1,800

Pituitary

1,800

Total

7,200

The dataset was reorganized into a stratified 70/15/15 split:

Split

Images

Training

5,040

Validation

1,080

Testing

1,080

Class Mapping

glioma      → 0
meningioma  → 1
notumor     → 2
pituitary   → 3

The test set is kept separate and was not used during hyperparameter tuning.

📅 15-Day Development Roadmap

Day

Task

Status

1

Project Setup & Dataset Exploration

✅ Completed

2

Data Preprocessing & Dataset Split

✅ Completed

3

Data Augmentation

✅ Completed

4

Custom Dataset & DataLoader

✅ Completed

5

Transfer Learning & Base Model

✅ Completed

6

Training Loop Setup

✅ Completed

7

Model Training & Performance Curves

✅ Completed

8

Hyperparameter Tuning

✅ Completed

9

Model Evaluation

✅ Completed

10

Grad-CAM Explainability

✅ Completed

11

Second Architecture Comparison

🔄 Upcoming

12

Streamlit/Flask Web Application

🔄 Upcoming

13

Experiment Tracking

🔄 Upcoming

14

Error & Edge-Case Analysis

🔄 Upcoming

15

Deployment, Model Card & Presentation

🔄 Upcoming

📊 Completed Work

Day 1–6 — Foundation

Completed the complete data and training pipeline foundation:

Dataset exploration and class verification

Stratified 70/15/15 train-validation-test split

Image preprocessing and controlled augmentation

Custom PyTorch Dataset

Train/validation/test DataLoaders

EfficientNet-B0 transfer-learning setup

ImageNet pretrained weights

Training loss, optimizer, and scheduler configuration

CUDA/GPU verification

DataLoader

Batch Size       : 32
Workers          : 2
Prefetch Factor  : 2
Pin Memory       : True
Input Shape      : [32, 3, 224, 224]

Day 7 — Model Training

The baseline EfficientNet-B0 model was trained for 10 epochs.

Metric

Result

Best Validation Accuracy

87.22%

Best Epoch

6

Best Validation Loss

0.3943

Training setup: Adam + CrossEntropyLoss + StepLR.

Artifacts include training history, loss/accuracy curves, and the best Day 7 model checkpoint.

Day 8 — Hyperparameter Tuning

Four hyperparameters were evaluated:

Learning Rate

Batch Size

Dropout

Augmentation Strength

Best Configuration

Parameter

Value

Learning Rate

0.002

Batch Size

32

Dropout

0.1

Augmentation

Weak

Epochs

10

Optimizer

Adam

Result

Best Validation Accuracy: 90.09%

Compared with Day 7 (87.22%), validation accuracy improved by 2.87 percentage points.

The test set remained untouched during tuning.

Day 9 — Model Evaluation

The optimized Day 8 model was evaluated on the held-out test set of 1,080 images.

Overall Performance

Metric

Score

Accuracy

91.20%

Weighted Precision

91.32%

Weighted Recall

91.20%

Weighted F1-score

91.16%

Per-Class Performance

Class

Precision

Recall

F1

Glioma

94.58%

84.07%

89.02%

Meningioma

84.23%

87.04%

85.61%

No Tumor

93.01%

98.52%

95.68%

Pituitary

93.45%

95.19%

94.31%

ROC-AUC

Class

ROC-AUC

Glioma

0.9776

Meningioma

0.9669

No Tumor

0.9970

Pituitary

0.9921

Key observation: Meningioma was the most challenging class by F1-score, while No Tumor achieved the strongest overall class performance.

Day 10 — Grad-CAM Explainability

Grad-CAM was implemented using pytorch-grad-cam with the final EfficientNet-B0 feature block as the target layer.

Explainability Results

12 Grad-CAM samples

3 samples per class

11 correct predictions

1 incorrect prediction

Sample accuracy: 91.67%

Average confidence: 89.52%

The selected incorrect example was:

True Class       : Pituitary
Predicted Class  : No Tumor
Confidence       : 62.26%

The incorrect example was retained for qualitative inspection of the model's attention.

Generated Artifacts

figures/
├── day10_gradcam_sample_01...12.png
└── day10_gradcam_contact_sheet.png

reports/
├── day10_gradcam_samples.csv
├── day10_gradcam_summary.md
└── day10_gradcam_summary.json

Grad-CAM is used to inspect model behavior and attention regions. The highlighted regions should not be interpreted as clinical reasoning or proof of a diagnosis.

🛠️ Technology Stack

Technology

Purpose

Python

Programming

PyTorch

Deep Learning

Torchvision

Computer Vision & Models

EfficientNet-B0

Transfer Learning

CUDA

GPU Acceleration

Google Colab

Development

NumPy / Pandas

Data Processing

Matplotlib / Seaborn

Visualization

Scikit-learn

Evaluation & Splitting

Pillow

Image Processing

Grad-CAM

Explainable AI

Streamlit

Planned Deployment

Git & GitHub

Version Control

📁 Repository Structure

MediScan-Medical-Image-Classification/
│
├── app/                 # Web application
├── data/
│   └── processed/       # Dataset metadata and split CSVs
├── figures/             # Generated visualizations
├── models/              # Local model checkpoints
├── notebooks/           # Day-wise Colab notebooks
├── reports/             # Day-wise reports and metrics
├── src/                 # Source code
│
├── .gitignore
├── README.md
└── requirements.txt

Large/private development files are excluded from GitHub:

data/raw/

archive.zip

.pt / .pth model weights

🔬 Current Project Status

Completed through Day 10.

Best Model So Far

Model              : EfficientNet-B0
Validation Accuracy: 90.09%
Test Accuracy      : 91.20%
Test F1-score      : 91.16%
Grad-CAM Samples   : 12

Next

Day 11 — Second Architecture Comparison

The next stage will compare a second CNN architecture against the current EfficientNet-B0 model.

⚠️ Disclaimer

MediScan is an educational and research-oriented project.

It is not a medical diagnostic system and should not be used as a substitute for professional medical diagnosis, clinical evaluation, or medical advice.

👨‍💻 Project

MediScan — Medical Image Classification

Focus: Brain MRI Image Classification
Framework: PyTorch
Current Model: EfficientNet-B0
Classes: 4
Progress: Day 10 / 15 Completed
