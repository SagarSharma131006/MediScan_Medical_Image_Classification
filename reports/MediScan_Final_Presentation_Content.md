
# MediScan — Brain MRI Classification
## Final 7-Minute Project Presentation

---

## Slide 1 — Title

### MediScan
Brain MRI Classification with Explainable AI

- EfficientNet-B0
- PyTorch
- Grad-CAM
- Streamlit Web Application

### Presenter
Sagar Sharma

---

## Slide 2 — Problem Statement

### Problem

Brain tumor classification from MRI images is an important medical imaging task.

### Objective

Build an end-to-end machine learning system that can:

- Classify brain MRI images
- Predict one of four classes
- Provide confidence scores
- Explain predictions using Grad-CAM
- Provide an interactive web interface

### Classes

- Glioma
- Meningioma
- No Tumor
- Pituitary

---

## Slide 3 — Dataset

### Brain Tumor MRI Dataset

Total images: 7,200

| Class | Images |
|---|---:|
| Glioma | 1,800 |
| Meningioma | 1,800 |
| No Tumor | 1,800 |
| Pituitary | 1,800 |

### Stratified Split

- Training: 5,040
- Validation: 1,080
- Testing: 1,080

Split ratio:

70 / 15 / 15

---

## Slide 4 — ML Pipeline

### Complete Pipeline

MRI Image
↓
Preprocessing
↓
224 × 224 RGB
↓
Data Augmentation
↓
EfficientNet-B0
↓
4-Class Prediction
↓
Confidence + Probabilities
↓
Grad-CAM Explanation
↓
Streamlit Web Application

### Technologies

- Python
- PyTorch
- Torchvision
- Scikit-learn
- Grad-CAM
- Streamlit
- Weights & Biases

---

## Slide 5 — Model Training & Hyperparameter Tuning

### Initial Model

EfficientNet-B0 pretrained on ImageNet.

Backbone was frozen.

### Hyperparameters

- Learning Rate: 0.002
- Batch Size: 32
- Dropout: 0.1
- Optimizer: Adam
- Loss: CrossEntropyLoss
- Scheduler: StepLR
- Epochs: 10
- Input: 224 × 224

### Augmentation

Weak augmentation performed best during tuning.

---

## Slide 6 — Model Comparison

Two architectures were evaluated.

| Metric | EfficientNet-B0 | ResNet50 |
|---|---:|---:|
| Validation Accuracy | 90.09% | 90.00% |
| Test Accuracy | 91.20% | 90.93% |
| Weighted F1 | 91.16% | 90.83% |

### Final Selection

EfficientNet-B0 was selected because it achieved slightly better overall performance in this experimental setup.

---

## Slide 7 — Final Model Performance

### Test Performance

Accuracy: 91.20%

Precision: 91.32%

Recall: 91.20%

Weighted F1: 91.16%

### ROC-AUC

- Glioma: 0.9776
- Meningioma: 0.9669
- No Tumor: 0.9970
- Pituitary: 0.9921

---

## Slide 8 — Explainable AI

### Grad-CAM

Grad-CAM was implemented to visualize image regions contributing to model predictions.

### Explainability Samples

12 samples generated:

- 3 Glioma
- 3 Meningioma
- 3 No Tumor
- 3 Pituitary

Sample accuracy:

91.67%

### Important

Grad-CAM is an explainability aid and does not establish clinical causality.

---

## Slide 9 — Error Analysis

### Test Results

- Correct predictions: 985
- Misclassified: 95
- Error rate: 8.80%

### Highest Error Class

Glioma:

15.93% error rate

### Most Common Failure Mode

Glioma → Meningioma

33 cases

### Observation

Glioma was slightly under-predicted while No Tumor was slightly over-predicted.

---

## Slide 10 — Web Application

### MediScan Streamlit App

Features:

- MRI image upload
- Predicted class
- Confidence score
- Class probabilities
- Grad-CAM visualization
- Medical disclaimer

Supported formats:

- JPG
- JPEG
- PNG

### Live Demo

Upload:
Te-me_178.jpg

Expected demonstration:

Prediction: Meningioma

Confidence: 65.65%

---

## Slide 11 — Experiment Tracking & Engineering

### Weights & Biases

Tracked:

- Hyperparameters
- Training metrics
- Validation metrics
- Test metrics
- Per-class metrics
- ROC-AUC
- Model artifact
- Training configuration

### Experiment

Project:
MediScan-Medical-Image-Classification

Run:
day13-experiment-tracking

---

## Slide 12 — Limitations & Ethical Considerations

### Limitations

- Single public dataset
- No external clinical validation
- Dataset-specific characteristics
- Glioma has higher error rate
- No sufficient demographic metadata
- Confidence is not clinical certainty

### Ethical Considerations

- Not a diagnostic system
- Should not replace medical professionals
- Clinical deployment requires extensive validation
- Patient privacy must be protected
- Human clinical oversight is required

---

## Slide 13 — Conclusion

### MediScan Successfully Demonstrates

- End-to-end MRI classification
- 91.20% test accuracy
- 91.16% weighted F1
- Explainable AI with Grad-CAM
- Model comparison
- Error analysis
- Experiment tracking
- Interactive Streamlit application

### Final Model

EfficientNet-B0

---

## Slide 14 — Live Demo

### Demo Flow

1. Open MediScan web application
2. Upload brain MRI
3. Show original image
4. Show predicted class
5. Show confidence
6. Show class probabilities
7. Show Grad-CAM
8. Explain that Grad-CAM provides model interpretability
9. Mention medical disclaimer

### Demo Target

Keep the live demo within approximately 1 minute.

---

## Slide 15 — Thank You

# Thank You

Questions?


## Day 15 — Live Deployment

**Live Streamlit Application:** https://mediscanmedicalimageclassification-ywdozzsxzf5nngqvvb48ve.streamlit.app/

### Web Application Features
- Brain MRI image upload
- Four-class prediction
- Confidence score
- Class probability distribution
- Grad-CAM visualization
- Medical-use disclaimer
