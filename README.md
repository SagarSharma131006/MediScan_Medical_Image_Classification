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
