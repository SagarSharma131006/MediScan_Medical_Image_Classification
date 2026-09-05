# 🧠 MediScan — Medical Image Classification

MediScan is a deep learning project for classifying brain MRI images into four categories:

- Glioma
- Meningioma
- No Tumor
- Pituitary

The project focuses on model training, evaluation, explainability using Grad-CAM, model comparison, and a Streamlit-based web interface.

---

## 🚀 Project Pipeline

Dataset → EDA & Preprocessing → Model Training → Hyperparameter Tuning → Evaluation → Grad-CAM → Model Comparison → Streamlit App → Experiment Tracking → Error Analysis → Deployment

---

## 📊 Dataset

**Brain Tumor MRI Dataset**

- Total Images: 7,200
- Classes: 4
- Input Size: 224 × 224
- Split: 70% Train / 15% Validation / 15% Test
- Train: 5,040
- Validation: 1,080
- Test: 1,080

### Classes

| Class | Label |
|---|---:|
| Glioma | 0 |
| Meningioma | 1 |
| No Tumor | 2 |
| Pituitary | 3 |

---

## 🛠️ Tech Stack

- Python
- PyTorch
- Torchvision
- EfficientNet-B0
- ResNet50
- Scikit-learn
- OpenCV
- Pillow
- Matplotlib
- Grad-CAM
- Streamlit
- Google Colab
- Weights & Biases / MLflow

---

# 📅 15-Day Development Roadmap

| Day | Task | Status |
|---|---|---|
| Day 1 | Project Setup & Dataset | ✅ |
| Day 2 | EDA & Data Preparation | ✅ |
| Day 3 | DataLoader & Pipeline Verification | ✅ |
| Day 4 | Model Architecture Setup | ✅ |
| Day 5 | Training Pipeline | ✅ |
| Day 6 | Initial Model Verification | ✅ |
| Day 7 | Baseline Model Training | ✅ |
| Day 8 | Hyperparameter Tuning | ✅ |
| Day 9 | Model Evaluation | ✅ |
| Day 10 | Grad-CAM Explainability | ✅ |
| Day 11 | Model Comparison | ✅ |
| Day 12 | Streamlit Web Interface | ✅ |
| Day 13 | Experiment Tracking | ✅ |
| Day 14 | Error & Edge Case Analysis | ✅ |
| Day 15 | Deployment, Model Card & Presentation | ✅ |

---

# 🤖 Models

## EfficientNet-B0

The primary model used for the project.

Final tuned configuration:

- Learning Rate: `0.002`
- Batch Size: `32`
- Dropout: `0.1`
- Augmentation: Weak
- Optimizer: Adam
- Loss: CrossEntropyLoss
- Scheduler: StepLR
- Input Size: `224 × 224`

### Performance

- Best Validation Accuracy: **90.09%**
- Test Accuracy: **91.20%**
- Weighted F1 Score: **91.16%**

EfficientNet-B0 was selected as the final architecture after comparison with ResNet50.

---

## ResNet50

Second architecture trained for comparison.

- Learning Rate: `0.002`
- Batch Size: `32`
- Dropout: `0.1`
- Input Size: `224 × 224`

### Performance

- Best Validation Accuracy: **90.00%**
- Test Accuracy: **90.93%**
- Weighted F1 Score: **90.83%**

EfficientNet-B0 performed slightly better on the final evaluation setup.

---

# 🔍 Model Evaluation

Evaluation includes:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC Curves
- Per-class ROC-AUC

### EfficientNet-B0 Test Accuracy

**91.20%**

### ROC-AUC

| Class | ROC-AUC |
|---|---:|
| Glioma | 0.9776 |
| Meningioma | 0.9669 |
| No Tumor | 0.9970 |
| Pituitary | 0.9921 |

---

# 🔥 Grad-CAM Explainability

Grad-CAM was implemented to visualize the regions influencing model predictions.

Day 10 generated:

- 12 Grad-CAM samples
- 3 samples per class
- Original MRI + heatmap overlays
- Correct and incorrect prediction examples
- Grad-CAM contact sheet

Sample accuracy: **91.67%**

---

# 🌐 Streamlit Web Application

Day 12 introduced a Streamlit web interface.

### Features

- MRI image upload
- Brain tumor classification
- Prediction confidence
- Class-wise probabilities
- Grad-CAM visualization
- Medical-use disclaimer

Supported image formats:

- JPG
- JPEG
- PNG

The application uses the final EfficientNet-B0 model.

> ⚠️ This project is intended for educational and research purposes only. It is not a medical diagnostic system and should not be used for clinical decision-making.

---

---

# 📈 Experiment Tracking

Day 13 integrated **Weights & Biases (W&B)** for experiment tracking and reproducibility.

### Tracked Information

- Model architecture and configuration
- Learning rate
- Batch size
- Dropout
- Epochs
- Optimizer and loss function
- Training loss and accuracy
- Validation loss and accuracy
- Test Accuracy, Precision, Recall and F1
- Per-class Precision, Recall and F1
- Per-class ROC-AUC
- Model weights
- Evaluation reports
- Training and evaluation figures

### W&B Experiment

- Project: `MediScan-Medical-Image-Classification`
- Run: `day13-experiment-tracking`
- Run ID: `xzkmsk5u`

[View W&B Experiment](https://wandb.ai/sagarsharma131006-panipat-institute-of-engineering-and-t/MediScan-Medical-Image-Classification/runs/xzkmsk5u)

### W&B Artifact

`mediscan-efficientnet-b0-results`

The artifact contains the final EfficientNet-B0 model weights, evaluation reports, and important project figures.

### Day 14 — Error & Edge Case Analysis ✅
- Identified 95 misclassified test samples
- Analyzed class-wise error rates and failure modes
- Identified Glioma → Meningioma as the dominant failure mode
- Performed class-balance and prediction-distribution bias analysis
- Visualized representative misclassified MRI samples
- Documented model limitations and recommended improvements
- Generated detailed error-analysis reports

### Day 14 Error Analysis

- Test accuracy: 91.20%
- Misclassified samples: 95 / 1080
- Highest error class: Glioma — 15.93%
- Dominant failure mode: Glioma → Meningioma — 33 cases
- Test set: Balanced — 270 samples per class
- Most under-predicted class: Glioma — −2.78 percentage points
- Most over-predicted class: No Tumor — +1.48 percentage points

# 📁 Project Structure

```text
MediScan/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── train.csv
│       ├── val.csv
│       ├── test.csv
│       └── class_to_idx.json
│
├── figures/
│   ├── day7_*
│   ├── day8_*
│   ├── day9_*
│   ├── day10_*
│   ├── day11_*
│   └── day15_cloud_inference_gradcam_test.png
│
├── models/
│   └── *.pth
│
├── reports/
│   ├── day7_*
│   ├── day8_*
│   ├── day9_*
│   ├── day10_*
│   ├── day11_*
│   ├── day12_*
│   ├── day13_*
│   ├── day14_*
│   ├── MediScan_Model_Card.md
│   └── MediScan_Final_Presentation_Content.md
│
├── notebooks/
│   ├── MediScan_Day1_Setup.ipynb
│   ├── MediScan_Day2_EDA_Data_Preparation.ipynb
│   ├── ...
│   ├── MediScan_Day14_Error_Analysis.ipynb
│   └── MediScan_Day15_Deployment_Final_Demo.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 🚀 Day 15 — Deployment & Final Demo

MediScan is deployed as a public Streamlit web application.

**Live Application:** https://mediscanmedicalimageclassification-ywdozzsxzf5nngqvvb48ve.streamlit.app/

### Web App Features

- Brain MRI image upload
- Four-class classification: Glioma, Meningioma, No Tumor, Pituitary
- Prediction confidence
- Class probability distribution
- Grad-CAM explainability visualization
- Medical-use disclaimer

### Final Model

- Architecture: EfficientNet-B0
- Test Accuracy: **91.20%**
- Weighted F1 Score: **91.16%**
- Best Validation Accuracy: **90.09%**

### Final Documentation

- Model Card
- Final Presentation Content
- Day 15 Deployment & Final Demo Notebook
- Cloud inference + Grad-CAM verification

> ⚠️ This application is an educational/research demonstration and is not intended for clinical diagnosis or medical decision-making.
