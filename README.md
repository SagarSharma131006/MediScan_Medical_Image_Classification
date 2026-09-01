# 🧠 MediScan — Medical Image Classification

> **Deep Learning-based Brain MRI Classification using PyTorch and EfficientNet-B0**

MediScan is a deep learning project focused on classifying brain MRI images into four categories:

- 🧠 Glioma
- 🧠 Meningioma
- 🧠 Pituitary Tumor
- ✅ No Tumor

The project follows a structured **15-day deep learning workflow** covering:

**Dataset Exploration → Preprocessing → Data Augmentation → Data Loading → Transfer Learning → Model Training → Evaluation → Explainability → Deployment**

The implementation is being developed using **Python, PyTorch, Torchvision, Google Colab, CUDA, and EfficientNet-B0**.

---

## 📌 Project Overview

Medical image classification can assist in the preliminary analysis of MRI scans by automatically identifying patterns associated with different tumor categories.

MediScan aims to build a complete image classification pipeline that can:

1. Load and analyze brain MRI images.
2. Prepare and preprocess the dataset.
3. Apply controlled image augmentation.
4. Efficiently load images using PyTorch DataLoaders.
5. Train a transfer-learning based CNN.
6. Evaluate classification performance.
7. Generate explainable predictions using Grad-CAM.
8. Deploy the trained model through a web interface.

> ⚠️ **Disclaimer:** MediScan is an educational/research project and is not intended to provide medical diagnosis or replace professional medical evaluation.

---

# 🎯 Project Objectives

- Build a complete medical image classification pipeline.
- Classify brain MRI images into four categories.
- Use transfer learning to reduce training requirements.
- Maintain a reproducible data preparation workflow.
- Evaluate the trained model using multiple performance metrics.
- Add visual explainability using Grad-CAM.
- Develop a simple web-based interface for inference.

---

# 🛠️ Technology Stack

| Category | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Computer Vision | Torchvision |
| Model | EfficientNet-B0 |
| Pretrained Weights | ImageNet |
| Data Processing | Pandas, NumPy |
| Image Processing | PIL |
| Visualization | Matplotlib |
| Development Environment | Google Colab |
| GPU Acceleration | CUDA |
| Explainability | Grad-CAM |
| Deployment | Streamlit |
| Version Control | Git & GitHub |

---

# 📂 Dataset

The project uses a brain MRI image dataset containing four classes:

| Class | Description |
|---|---|
| `glioma` | MRI images containing glioma tumor |
| `meningioma` | MRI images containing meningioma tumor |
| `pituitary` | MRI images containing pituitary tumor |
| `notumor` | MRI images without tumor |

## Dataset Overview

| Detail | Value |
|---|---:|
| Total Images | 7,200 |
| Number of Classes | 4 |
| Original Training Images | 5,600 |
| Original Testing Images | 1,600 |

The dataset is balanced across the four classes.

## Dataset Storage

The raw dataset is **not uploaded to GitHub** because of its size.

It is stored in Google Drive and accessed through Google Colab.

```text
Google Drive
└── MediScan
    └── data
        └── raw
            ├── Training
            │   ├── glioma
            │   ├── meningioma
            │   ├── notumor
            │   └── pituitary
            │
            └── Testing
                ├── glioma
                ├── meningioma
                ├── notumor
                └── pituitary
📅 15-Day Development Roadmap
Day	Focus	Status
1	Project Setup & Dataset Exploration	✅ Complete
2	Data Preprocessing & Dataset Split	✅ Complete
3	Data Augmentation	✅ Complete
4	Custom Dataset & DataLoaders	✅ Complete
5	Base Model / Transfer Learning Setup	✅ Complete
6	Training Loop Setup	✅ Complete
7	Model Training	⏳ Upcoming
8	Training Analysis & Improvement	⏳ Upcoming
9	Model Evaluation	⏳ Upcoming
10	Confusion Matrix & Classification Report	⏳ Upcoming
11	Model Improvement / Comparison	⏳ Upcoming
12	Grad-CAM Explainability	⏳ Upcoming
13	Inference Pipeline	⏳ Upcoming
14	Streamlit Web Application	⏳ Upcoming
15	Final Testing & Documentation	⏳ Upcoming
📅 Day 1 — Project Setup & Dataset Exploration
🎯 Objective

The goal of Day 1 was to set up the project environment, connect Google Colab with GitHub, load the brain MRI dataset, and perform initial dataset exploration.

✅ Work Completed
Configured the Google Colab environment.
Connected Google Drive with Colab.
Cloned the GitHub repository.
Loaded the brain MRI dataset.
Extracted the dataset successfully.
Verified the dataset folder structure.
Created a dataset manifest.
Verified the four classes.
Analyzed class distribution.
Visualized sample MRI images.
📊 Dataset Statistics
Detail	Value
Total Images	7,200
Number of Classes	4
Original Training Set	5,600
Original Testing Set	1,600
🧠 Classes
Class	Meaning
glioma	Glioma tumor
meningioma	Meningioma tumor
pituitary	Pituitary tumor
notumor	No tumor
💡 Key Insight

The dataset is balanced across the four classes. Each class contains an equal number of images within the original training and testing sets.

This provides a balanced starting point for the classification task and reduces the risk of class imbalance affecting the model.

📁 Day 1 Outputs
data/processed/day1_dataset_manifest.csv
figures/day1_sample_images.png
reports/day1_summary.md
✅ Day 1 Status
Task	Status
Colab setup	✅ Done
GitHub repository setup	✅ Done
Dataset extraction	✅ Done
7,200 images verified	✅ Done
Four classes verified	✅ Done
Class distribution analyzed	✅ Done
Sample visualization	✅ Done
Day 1 summary saved	✅ Done
📅 Day 2 — Data Preprocessing & Dataset Split
🎯 Objective

The goal of Day 2 was to analyze the image data, prepare the dataset for model development, and create a stratified train-validation-test split.

✅ Work Completed
Analyzed image dimensions.
Analyzed image color modes.
Prepared images for 224 × 224 input size.
Converted images to RGB during dataset loading.
Created a stratified 70/15/15 train-validation-test split.
Preserved class balance across all splits.
Created class-to-index mapping.
Saved processed CSV files.
📊 Dataset Split
Split	Images
Training	5,040
Validation	1,080
Testing	1,080
Total	7,200
🔢 Class Mapping
Class	Index
glioma	0
meningioma	1
notumor	2
pituitary	3
📁 Day 2 Outputs
data/processed/day2_train_val_test_split.csv
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
data/processed/class_to_idx.json

figures/day2_split_distribution.png
reports/day2_summary.md
⚠️ Preprocessing Note

The normalization step was not performed as a standalone Day 2 operation.

Normalization was later incorporated into the model input pipeline using ImageNet normalization, which is appropriate for the pretrained EfficientNet-B0 model.

The normalization used later in the pipeline is:

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
📅 Day 3 — Data Augmentation
🎯 Objective

The goal of Day 3 was to implement controlled image augmentation for the training MRI images and visually verify the effects of the transformations.

✅ Work Completed

The following augmentation techniques were implemented:

Random Horizontal Flip
Random Rotation
Color Jitter
Random Affine Transformation

The augmentation pipeline was implemented using PyTorch/Torchvision transforms.

🔄 Augmentation Pipeline
Resize (224 × 224)
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
🧪 Training Transform

The training pipeline uses:

Resize(224, 224)
RandomHorizontalFlip(p=0.5)
RandomRotation(±15°)
ColorJitter
RandomAffine
ToTensor
🔍 Evaluation Transform

Validation and test images use deterministic preprocessing without random augmentation:

Resize(224, 224)
        ↓
ToTensor
        ↓
Normalization
📁 Day 3 Outputs
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
data/processed/class_to_idx.json

figures/day3_augmentation_comparison.png
figures/day3_combined_augmentation_samples.png

reports/day3_summary.md
✅ Day 3 Verification
========================================
DAY 3 AUGMENTATION VERIFICATION PASSED
========================================
📅 Day 4 — Custom Dataset & DataLoader
🎯 Objective

The goal of Day 4 was to implement a custom PyTorch Dataset and efficient DataLoaders for batch-wise MRI image loading.

✅ Work Completed
Implemented a custom BrainMRIDataset.
Loaded training, validation, and testing images.
Converted images to RGB.
Applied training augmentation.
Applied evaluation preprocessing.
Converted class names into integer class indices.
Created PyTorch DataLoaders.
Configured batch size of 32.
Configured multiple workers.
Enabled pinned memory.
Enabled DataLoader prefetching.
Verified DataLoader iteration.
⚙️ DataLoader Configuration
Parameter	Value
Batch Size	32
Workers	2
Prefetch Factor	2
Pin Memory	True
Training Shuffle	True
Validation Shuffle	False
Testing Shuffle	False
🧪 Batch Verification
Images shape : torch.Size([32, 3, 224, 224])
Labels shape : torch.Size([32])

Images dtype : torch.float32
Labels dtype : torch.int64
🔄 DataLoader Iteration

The complete training DataLoader was successfully iterated.

DataLoader iteration successful!
Batches loaded: 158
✅ Day 4 Verification
========================================
DAY 4 DATALOADER VERIFICATION
========================================

Images shape : torch.Size([32, 3, 224, 224])
Labels shape : torch.Size([32])
Images dtype : torch.float32
Labels dtype : torch.int64

DataLoader configuration:
Batch size   : 32
Num workers  : 2
Prefetch     : 2
Pin memory   : True

========================================
DAY 4 VERIFICATION PASSED
========================================
📅 Day 5 — Base Model & Transfer Learning
🎯 Objective

The goal of Day 5 was to set up a pretrained CNN using transfer learning and modify its classifier head for four-class brain MRI classification.

🧠 Model
EfficientNet-B0

The project uses EfficientNet-B0 with pretrained ImageNet weights.

The original classifier was replaced with a custom four-class classification head.

🏗️ Model Architecture
Input MRI Image
       ↓
EfficientNet-B0
       ↓
Feature Extraction
       ↓
Dropout (0.2)
       ↓
Linear Layer
1280 → 4
       ↓
Four Class Logits
⚙️ Model Configuration
Parameter	Value
Architecture	EfficientNet-B0
Pretrained	Yes
Pretrained Weights	ImageNet
Input Size	224 × 224
Number of Classes	4
Device	CUDA
Dropout	0.2
Classifier	Linear(1280 → 4)
🔢 Class Mapping
Class	Index
glioma	0
meningioma	1
notumor	2
pituitary	3
📊 Parameter Verification
Parameter	Count
Total Parameters	4,012,672
Trainable Parameters	5,124
Frozen Parameters	4,007,548

The EfficientNet-B0 feature extraction layers were frozen while the custom classifier was configured for the four target classes.

✅ Day 5 Verification
========================================
DAY 5 MODEL VERIFICATION
========================================

Architecture : EfficientNet-B0
Pre-trained  : Yes
Num classes  : 4
Device       : cuda

Classifier:
Sequential(
  (0): Dropout(p=0.2, inplace=True)
  (1): Linear(in_features=1280, out_features=4, bias=True)
)

Total parameters     : 4012672
Trainable parameters : 5124
Frozen parameters    : 4007548

========================================
DAY 5 BASE MODEL SETUP PASSED
========================================
📅 Day 6 — Training Loop Setup
🎯 Objective

The goal of Day 6 was to configure and verify the components required for EfficientNet-B0 training.

✅ Work Completed
Restored the pretrained EfficientNet-B0 model.
Restored the training dataset.
Restored the validation dataset.
Restored the testing dataset.
Restored PyTorch DataLoaders.
Configured CrossEntropyLoss.
Configured Adam optimizer.
Configured StepLR learning-rate scheduler.
Verified the model forward pass.
Verified model output dimensions.
Verified target label compatibility.
Successfully calculated the training loss.
⚙️ Training Configuration
Component	Configuration
Model	EfficientNet-B0
Pretrained	Yes
Number of Classes	4
Device	CUDA
Batch Size	32
Loss Function	CrossEntropyLoss
Optimizer	Adam
Learning Rate	0.001
Scheduler	StepLR
Scheduler Step Size	5
Scheduler Gamma	0.1
📊 Dataset Configuration
Dataset	Images
Training	5,040
Validation	1,080
Testing	1,080
🧪 Forward Pass Verification
Input shape  : torch.Size([32, 3, 224, 224])
Output shape : torch.Size([32, 4])
Labels shape : torch.Size([32])
Labels dtype : torch.int64
Loss         : 1.47794771194458
🔁 Training Components
MRI Batch
   ↓
EfficientNet-B0
   ↓
4 Class Logits
   ↓
CrossEntropyLoss
   ↓
Adam Optimizer
   ↓
StepLR Scheduler
✅ Day 6 Verification
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
🔄 Current Training Pipeline

The current MediScan pipeline is structured as:

                    Brain MRI Dataset
                           │
                           ▼
                 Dataset Exploration
                           │
                           ▼
                 Stratified Data Split
                     70 / 15 / 15
                           │
                           ▼
                 Image Preprocessing
                     224 × 224 RGB
                           │
                           ▼
                  Image Augmentation
                    Training Only
                           │
                           ▼
                  Custom PyTorch Dataset
                           │
                           ▼
                       DataLoader
                       Batch Size 32
                           │
                           ▼
                    EfficientNet-B0
                  ImageNet Pretrained
                           │
                           ▼
                   Custom Classifier
                      1280 → 4
                           │
                           ▼
                    Model Logits
                           │
                           ▼
                  CrossEntropyLoss
                           │
                           ▼
                   Adam Optimizer
                           │
                           ▼
                     StepLR Scheduler
                           │
                           ▼
                        Training
                           │
                           ▼
                       Evaluation
                           │
                           ▼
                    Grad-CAM Analysis
                           │
                           ▼
                  Streamlit Deployment
🧪 Current Project Status
✅ Completed
 Project environment setup
 Google Colab setup
 Google Drive integration
 GitHub repository setup
 Dataset extraction
 Dataset exploration
 Class verification
 Dataset manifest creation
 Stratified 70/15/15 split
 Class-to-index mapping
 Data augmentation
 Augmentation verification
 Custom PyTorch Dataset
 PyTorch DataLoaders
 DataLoader verification
 EfficientNet-B0 setup
 Transfer learning configuration
 Classifier head modification
 Parameter verification
 CrossEntropyLoss configuration
 Adam optimizer configuration
 StepLR scheduler configuration
 Forward-pass verification
 Training-loop verification
⏳ Upcoming
 Multi-epoch model training
 Training loss monitoring
 Validation loss monitoring
 Training accuracy analysis
 Validation accuracy analysis
 Model checkpointing
 Test-set evaluation
 Confusion matrix
 Classification report
 Precision, Recall and F1-score analysis
 Model improvement
 Grad-CAM explainability
 Inference pipeline
 Streamlit application
 Final testing
 Final documentation
📁 Repository Structure
MediScan/
│
├── data/
│   └── processed/
│       ├── day1_dataset_manifest.csv
│       ├── train.csv
│       ├── val.csv
│       ├── test.csv
│       └── class_to_idx.json
│
├── figures/
│   ├── day1_sample_images.png
│   ├── day2_split_distribution.png
│   ├── day3_augmentation_comparison.png
│   └── day3_combined_augmentation_samples.png
│
├── notebooks/
│   ├── MediScan_Day1_*.ipynb
│   ├── MediScan_Day2_*.ipynb
│   ├── MediScan_Day3_*.ipynb
│   ├── MediScan_Day4_*.ipynb
│   ├── MediScan_Day5_*.ipynb
│   └── MediScan_Day6_*.ipynb
│
├── reports/
│   ├── day1_summary.md
│   ├── day2_summary.md
│   ├── day3_summary.md
│   ├── day4_summary.md
│   ├── day5_summary.md
│   └── day6_summary.md
│
├── .gitignore
└── README.md
🚫 Dataset & Large Files

The following files and directories are intentionally excluded from GitHub:

archive.zip
data/raw/
models/*.pt
models/*.pth
__pycache__/
.ipynb_checkpoints/

The raw MRI dataset is stored in Google Drive and accessed through Google Colab.

Model weights will be added later after the training and model checkpoint stages are completed.

🔬 Model Development
EfficientNet-B0

EfficientNet-B0 is currently being used as the primary transfer-learning model.

The pretrained feature extraction layers provide learned visual representations, while the final classifier is adapted for the four MRI classes.

EfficientNet-B0
       │
       ├── Pretrained Feature Extractor
       │
       └── Custom Classifier
              │
              ├── Dropout(0.2)
              │
              └── Linear(1280 → 4)
📊 Evaluation Plan

After model training, MediScan will be evaluated using:

Accuracy
Precision
Recall
F1-score
Confusion Matrix
Training Loss
Validation Loss
Test Loss
Per-class performance

These metrics will be used to determine how effectively the model distinguishes between the four MRI categories.

🔍 Explainable AI — Grad-CAM

A major objective of MediScan is to make model predictions more interpretable.

The project will use Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize image regions that contribute to the model's prediction.

Intended Workflow
MRI Image
    ↓
Trained EfficientNet-B0
    ↓
Predicted Class
    ↓
Grad-CAM
    ↓
Activation Heatmap
    ↓
Heatmap Overlay
    ↓
Explainable Prediction

This will provide a visual indication of the regions the model focuses on when making a classification.

🌐 Deployment Plan

The final project is planned to include a Streamlit web application.

Expected Workflow
User Uploads MRI
        ↓
Image Preprocessing
        ↓
Trained MediScan Model
        ↓
Prediction
        ↓
Predicted Class
        ↓
Confidence Score
        ↓
Optional Grad-CAM Visualization

The application will provide an easy-to-use interface for testing the trained model.

📚 Learning Outcomes

This project provides hands-on experience with:

Computer Vision
Medical Image Classification
PyTorch
Torchvision
Transfer Learning
CNN architectures
EfficientNet
Data Augmentation
Stratified Dataset Splitting
Custom PyTorch Dataset
DataLoader optimization
GPU acceleration
Model training
Model evaluation
Explainable AI
Grad-CAM
Model deployment
Streamlit
Git & GitHub
Reproducible Machine Learning workflows
⚠️ Medical Disclaimer

MediScan is developed for educational and research purposes only.

The predictions generated by this project should not be considered medical diagnoses.

MRI interpretation and tumor diagnosis must be performed by qualified medical professionals using appropriate clinical information and diagnostic procedures.

👨‍💻 Development

MediScan is being developed as a structured 15-day deep learning project.

Each development stage is documented through:

Jupyter/Google Colab notebooks
Markdown reports
Dataset artifacts
Visualization figures
Verification outputs
GitHub commits

This approach helps maintain a reproducible and organized machine-learning development workflow.

📈 Project Progress
Day 01  ████████████████████  Complete
Day 02  ████████████████████  Complete
Day 03  ████████████████████  Complete
Day 04  ████████████████████  Complete
Day 05  ████████████████████  Complete
Day 06  ████████████████████  Complete

Day 07  ░░░░░░░░░░░░░░░░░░░░  Upcoming
Day 08  ░░░░░░░░░░░░░░░░░░░░  Upcoming
Day 09  ░░░░░░░░░░░░░░░░░░░░  Upcoming
Day 10  ░░░░░░░░░░░░░░░░░░░░  Upcoming
Day 11  ░░░░░░░░░░░░░░░░░░░░  Upcoming
Day 12  ░░░░░░░░░░░░░░░░░░░░  Upcoming
Day 13  ░░░░░░░░░░░░░░░░░░░░  Upcoming
Day 14  ░░░░░░░░░░░░░░░░░░░░  Upcoming
Day 15  ░░░░░░░░░░░░░░░░░░░░  Upcoming
🚀 Final Project Goal

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
⭐ MediScan

From MRI Images to Explainable Deep Learning

A structured deep learning project demonstrating the complete journey from raw medical images to an explainable classification system
