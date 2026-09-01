# 🧠 MediScan — Medical Image Classification

> **Deep Learning-based Brain MRI Classification using PyTorch and EfficientNet-B0**

MediScan is a deep learning project focused on classifying brain MRI images into four categories:

- 🧠 **Glioma**
- 🧠 **Meningioma**
- 🧠 **Pituitary Tumor**
- ✅ **No Tumor**

The project follows a structured **15-day deep learning workflow** covering dataset exploration, preprocessing, data augmentation, data loading, transfer learning, model training, evaluation, explainability, and deployment.

---

## 🚀 Project Overview

Medical image classification can assist in the preliminary analysis of MRI scans by automatically identifying visual patterns associated with different tumor categories.

The goal of MediScan is to build an end-to-end image classification pipeline that can:

1. Load and analyze brain MRI images.
2. Prepare and preprocess the dataset.
3. Apply controlled image augmentation.
4. Efficiently load images using PyTorch DataLoaders.
5. Train a transfer-learning based CNN.
6. Evaluate classification performance.
7. Generate explainable predictions using Grad-CAM.
8. Deploy the trained model through a web application.

---

## 🎯 Project Workflow

```text
Brain MRI Dataset
        ↓
Dataset Exploration
        ↓
Preprocessing
        ↓
Stratified Dataset Split
        ↓
Data Augmentation
        ↓
Custom Dataset
        ↓
PyTorch DataLoader
        ↓
Transfer Learning
        ↓
EfficientNet-B0
        ↓
Model Training
        ↓
Evaluation
        ↓
Grad-CAM Explainability
        ↓
Streamlit Deployment
        ↓
        🧠 MediScan
📊 Dataset

The project uses a Brain MRI image dataset containing four classes:

Class	Description
glioma	MRI images showing glioma tumor
meningioma	MRI images showing meningioma tumor
pituitary	MRI images showing pituitary tumor
notumor	MRI images with no tumor
Dataset Statistics
Detail	Value
Total Images	7,200
Number of Classes	4
Original Training Images	5,600
Original Testing Images	1,600
Final Training Images	5,040
Final Validation Images	1,080
Final Testing Images	1,080
Dataset Note

The dataset is stored locally in Google Drive and is not uploaded to GitHub because of its large size.

The raw dataset is excluded from version control using .gitignore.

📅 Day 1 — Project Setup & Dataset Exploration
Objective

The goal of Day 1 was to set up the project environment, connect Google Colab with GitHub, load the brain MRI dataset, and perform initial dataset exploration.

Work Completed
Configured the Google Colab environment.
Connected Google Drive with Google Colab.
Cloned the GitHub repository.
Loaded the Brain MRI dataset.
Extracted and verified the dataset.
Checked the dataset folder structure.
Created a dataset manifest CSV.
Verified the four classes.
Analyzed class distribution.
Visualized sample MRI images.
Dataset Overview
Detail	Value
Dataset Type	Brain MRI Images
Total Images	7,200
Number of Classes	4
Original Training Images	5,600
Original Testing Images	1,600
Class Distribution

Each of the four classes contains an equal number of images in the original dataset split.

This provides a balanced starting point for the classification task and reduces the risk of strong class imbalance.

Day 1 Outputs
File	Purpose
data/processed/day1_dataset_manifest.csv	Dataset image paths, labels and original split information
figures/day1_sample_images.png	Sample MRI images from different classes
reports/day1_summary.md	Day 1 project summary
Day 1 Status
Task	Status
Colab setup	✅ Done
GitHub repository setup	✅ Done
Dataset extraction	✅ Done
Dataset verification	✅ Done
Four classes verified	✅ Done
Class distribution analyzed	✅ Done
Sample images visualized	✅ Done
Day 1 summary created	✅ Done
📅 Day 2 — Data Preprocessing & Dataset Split
Objective

The goal of Day 2 was to analyze the MRI images, prepare the dataset for deep learning, and create a stratified 70/15/15 train-validation-test split.

Work Completed
Analyzed image dimensions.
Analyzed image color modes.
Prepared images for 224 × 224 input resolution.
Converted images to RGB during preprocessing.
Created a stratified 70/15/15 dataset split.
Preserved class balance across all splits.
Created a class-to-index mapping.
Saved processed dataset CSV files.
Final Dataset Split
Split	Images
Training	5,040
Validation	1,080
Testing	1,080
Total	7,200
Class Mapping
Class	Index
glioma	0
meningioma	1
notumor	2
pituitary	3
Day 2 Outputs
data/processed/day2_train_val_test_split.csv
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
data/processed/class_to_idx.json
figures/day2_split_distribution.png
reports/day2_summary.md
Day 2 Status
Task	Status
Image analysis	✅ Done
RGB preprocessing	✅ Done
70/15/15 split	✅ Done
Stratification	✅ Done
Class mapping	✅ Done
Dataset CSVs	✅ Done
Split visualization	✅ Done

Note: The project uses the selected dataset and preprocessing pipeline established during Days 1–2. Subsequent days build on this pipeline.

📅 Day 3 — Data Augmentation
Objective

The goal of Day 3 was to implement controlled data augmentation for training MRI images and visually verify the effect of the transformations.

Augmentation Techniques

The training pipeline uses the following transformations:

🔄 Random Horizontal Flip
🔄 Random Rotation
💡 Color Jitter
📐 Random Affine Transformation

These transformations are applied to the training data to improve model generalization.

Augmentation Pipeline
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
Training Augmentation Configuration
Transformation	Configuration
Resize	224 × 224
Horizontal Flip	p = 0.5
Rotation	±15°
Brightness	0.85 – 1.15
Contrast	0.85 – 1.15
Saturation	0.90 – 1.10
Hue	-0.02 – 0.02
Translation	5%
Scale	0.95 – 1.05
Shear	±5°
Evaluation Transform

Validation and test images use a deterministic preprocessing pipeline without random augmentation.

Resize
   ↓
ToTensor
   ↓
Normalize
Day 3 Outputs
figures/day3_augmentation_comparison.png
figures/day3_combined_augmentation_samples.png
reports/day3_summary.md
Day 3 Status
Task	Status
Augmentation pipeline	✅ Done
Horizontal flip	✅ Done
Rotation	✅ Done
Color jitter	✅ Done
Affine transformation	✅ Done
Augmentation visualization	✅ Done
Day 3 summary	✅ Done
📅 Day 4 — Custom Dataset & DataLoader
Objective

The goal of Day 4 was to implement a custom PyTorch Dataset and DataLoaders for efficient batch-wise loading of MRI images.

Work Completed
Implemented a custom PyTorch Dataset.
Loaded training, validation, and testing images.
Converted images to RGB.
Applied training and evaluation transformations.
Converted class names into integer class indices.
Created PyTorch DataLoaders.
Configured batch-wise image loading.
Enabled multiple workers.
Enabled pinned memory.
Configured DataLoader prefetching.
Verified successful DataLoader iteration.
DataLoader Configuration
Parameter	Value
Batch Size	32
Number of Workers	2
Prefetch Factor	2
Pin Memory	True
Training Shuffle	True
Batch Verification
Images shape : torch.Size([32, 3, 224, 224])
Labels shape : torch.Size([32])
Images dtype : torch.float32
Labels dtype : torch.int64
Data Representation
Batch
 ↓
32 MRI Images
 ↓
3 RGB Channels
 ↓
224 × 224 Resolution
 ↓
4-Class Integer Labels
Day 4 Verification
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
Day 4 Status
Task	Status
Custom Dataset	✅ Done
Training DataLoader	✅ Done
Validation DataLoader	✅ Done
Test DataLoader	✅ Done
Batch verification	✅ Done
DataLoader iteration	✅ Done
Configuration verification	✅ Done
📅 Day 5 — Base Model Setup
Objective

The goal of Day 5 was to configure a pretrained CNN using transfer learning and modify its classifier for four-class brain MRI classification.

Model

The project uses EfficientNet-B0 with pretrained ImageNet weights.

Why EfficientNet-B0?

EfficientNet-B0 provides a strong balance between:

Model performance
Computational efficiency
Parameter count
Training speed
Work Completed
Loaded pretrained EfficientNet-B0.
Loaded ImageNet pretrained weights.
Inspected the original classifier.
Replaced the classifier with a four-class output layer.
Frozen the feature extraction layers.
Kept the classifier trainable.
Moved the model to CUDA.
Verified model parameters.
Model Configuration
Parameter	Value
Architecture	EfficientNet-B0
Pretrained	Yes
Pretrained Weights	ImageNet
Input Size	224 × 224
Number of Classes	4
Device	CUDA
Dropout	0.2
Classifier	Linear(1280 → 4)
Classifier
EfficientNet-B0
       ↓
Feature Extractor
       ↓
1280 Features
       ↓
Dropout(0.2)
       ↓
Linear(1280 → 4)
       ↓
4 Class Logits
Class Mapping
Class	Index
glioma	0
meningioma	1
notumor	2
pituitary	3
Parameter Verification
Parameter	Count
Total Parameters	4,012,672
Trainable Parameters	5,124
Frozen Parameters	4,007,548

Only the newly added classifier parameters are trainable during the initial transfer-learning setup.

Day 5 Status
Task	Status
EfficientNet-B0 loaded	✅ Done
ImageNet weights loaded	✅ Done
Classifier modified	✅ Done
Feature extractor frozen	✅ Done
CUDA setup	✅ Done
Parameter verification	✅ Done
Model verification	✅ Done
📅 Day 6 — Training Loop Setup
Objective

The goal of Day 6 was to configure and verify the complete training pipeline required to train EfficientNet-B0.

Work Completed
Restored the EfficientNet-B0 model.
Restored the processed dataset.
Restored the custom Dataset.
Restored the DataLoaders.
Configured CrossEntropyLoss.
Configured Adam optimizer.
Configured StepLR learning-rate scheduler.
Verified the model forward pass.
Verified model output dimensions.
Verified label dimensions and data type.
Successfully calculated the training loss.
Training Configuration
Parameter	Value
Model	EfficientNet-B0
Pretrained	Yes
Number of Classes	4
Device	CUDA
Batch Size	32
Loss Function	CrossEntropyLoss
Optimizer	Adam
Learning Rate	0.001
Scheduler	StepLR
Step Size	5
Gamma	0.1
Training Pipeline
MRI Batch
   ↓
EfficientNet-B0
   ↓
Feature Extraction
   ↓
Classifier
   ↓
4 Class Logits
   ↓
CrossEntropyLoss
   ↓
Adam Optimizer
   ↓
Parameter Update
   ↓
StepLR Scheduler
Forward Pass Verification
Input shape  : torch.Size([32, 3, 224, 224])
Output shape : torch.Size([32, 4])
Labels shape : torch.Size([32])
Labels dtype : torch.int64
Loss         : 1.47794771194458
Day 6 Verification
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
Day 6 Status
Task	Status
Model restored	✅ Done
Dataset restored	✅ Done
DataLoaders restored	✅ Done
Loss function configured	✅ Done
Adam optimizer configured	✅ Done
StepLR scheduler configured	✅ Done
Forward pass verified	✅ Done
Loss calculation verified	✅ Done
Training loop components verified	✅ Done
📈 Project Progress
Day	Module	Status
Day 1	Project Setup & Dataset Exploration	✅ Completed
Day 2	Preprocessing & Dataset Split	✅ Completed
Day 3	Data Augmentation	✅ Completed
Day 4	Custom Dataset & DataLoader	✅ Completed
Day 5	EfficientNet-B0 Base Model	✅ Completed
Day 6	Training Loop Setup	✅ Completed
Day 7	Model Training	🔜 Upcoming
Day 8	Training Analysis	🔜 Upcoming
Day 9	Model Evaluation	🔜 Upcoming
Day 10	Performance Analysis	🔜 Upcoming
Day 11	Grad-CAM	🔜 Upcoming
Day 12	Explainability Analysis	🔜 Upcoming
Day 13	Model Optimization	🔜 Upcoming
Day 14	Streamlit Deployment	🔜 Upcoming
Day 15	Final Project Integration	🔜 Upcoming
🗂️ Repository Structure
MediScan_Medical_Image_Classification/
│
├── app/
│   └── .gitkeep
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
│   └── day3_combined_augmentation_samples.png
│
├── models/
│   └── .gitkeep
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
🛠️ Technology Stack
Programming Language
Python
Deep Learning
PyTorch
Torchvision
EfficientNet-B0
Data Processing
NumPy
Pandas
Pillow
Visualization
Matplotlib
Seaborn
Development Environment
Google Colab
CUDA
Git
GitHub
Future Deployment
Streamlit
Explainability
Grad-CAM
🔬 Model Architecture

The current model uses transfer learning with EfficientNet-B0.

Input MRI
   │
   ▼
224 × 224 RGB Image
   │
   ▼
EfficientNet-B0
   │
   ├── Pretrained Feature Extractor
   │
   ▼
1280 Feature Vector
   │
   ▼
Dropout (0.2)
   │
   ▼
Linear Layer
1280 → 4
   │
   ▼
Class Logits
   │
   ├── Glioma
   ├── Meningioma
   ├── No Tumor
   └── Pituitary
📋 Current Dataset Pipeline
7,200 MRI Images
        ↓
70 / 15 / 15 Stratified Split
        ↓
┌───────────────┬───────────────┬───────────────┐
│    Training   │  Validation   │     Testing   │
│     5,040     │     1,080     │     1,080     │
└───────────────┴───────────────┴───────────────┘
        ↓
Image Preprocessing
        ↓
224 × 224 RGB
        ↓
Training Augmentation
        ↓
PyTorch Dataset
        ↓
DataLoader
        ↓
EfficientNet-B0
        ↓
4-Class Classification
📌 Important Dataset & Project Notes
The dataset is not stored inside this GitHub repository.
The original dataset is maintained in Google Drive.
Large raw files are excluded using .gitignore.
The project uses a stratified 70/15/15 split.
Training images use augmentation.
Validation and test images use deterministic preprocessing.
EfficientNet-B0 uses pretrained ImageNet weights.
The initial transfer-learning setup freezes the feature extractor.
Only the classifier head is trainable in the base model setup.
The model is configured to run on CUDA when available.
📊 Current Verification Summary
Component	Result
Dataset	✅ Verified
Dataset Split	✅ Verified
Class Mapping	✅ Verified
Augmentation	✅ Verified
Custom Dataset	✅ Verified
DataLoader	✅ Verified
EfficientNet-B0	✅ Verified
Transfer Learning	✅ Verified
Forward Pass	✅ Verified
Loss Calculation	✅ Verified
Training Components	✅ Verified
🚀 Upcoming Work

The next stages of MediScan will focus on:

Training EfficientNet-B0.
Monitoring training and validation performance.
Saving model checkpoints.
Evaluating the trained model.
Generating confusion matrices and classification reports.
Performing error analysis.
Implementing Grad-CAM explainability.
Visualizing model attention on MRI images.
Optimizing the final model.
Building a Streamlit web application.
Integrating the complete MediScan pipeline.
🎯 Final Project Goal

The final goal of MediScan is to demonstrate a complete end-to-end deep learning workflow for brain MRI classification:

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
🧠 MediScan
From MRI Images to Explainable Deep Learning

MediScan demonstrates the complete journey from raw medical images to an explainable deep learning classification system.

⭐ Project Status

Current Progress: Day 6 / 15 Completed

🚧 Project is actively under development.

👨‍💻 Author

Sagar Sharma

B.Tech CSE — AI & ML

⭐ If you find this project interesting

Feel free to explore the notebooks, reports, figures, and implementation as the project progresses toward the final MediScan deployment.

MediScan — From MRI Images to Explainable Deep Learning. 🧠🚀
