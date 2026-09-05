
# MediScan — Model Card

## 1. Model Overview

**Model Name:** MediScan EfficientNet-B0  
**Task:** Brain MRI Image Classification  
**Architecture:** EfficientNet-B0  
**Framework:** PyTorch  
**Input Size:** 224 × 224 RGB image  
**Number of Classes:** 4

The model classifies brain MRI images into:

- Glioma
- Meningioma
- No Tumor
- Pituitary

---

## 2. Intended Use

MediScan is intended for:

- Educational purposes
- Machine learning experimentation
- Research and demonstration of medical image classification
- Explainable AI experimentation using Grad-CAM

This model is **NOT intended to provide a medical diagnosis** and must not be used as a standalone clinical decision-making system.

---

## 3. Dataset

**Dataset:** Brain Tumor MRI Dataset

Total images:

**7,200**

Class distribution:

| Class | Images |
|---|---:|
| Glioma | 1,800 |
| Meningioma | 1,800 |
| No Tumor | 1,800 |
| Pituitary | 1,800 |

The dataset was reorganized using a stratified:

- 70% Training
- 15% Validation
- 15% Testing

Final split:

| Split | Images |
|---|---:|
| Training | 5,040 |
| Validation | 1,080 |
| Testing | 1,080 |

---

## 4. Preprocessing

Images were:

- Resized to 224 × 224
- Converted to RGB where required
- Converted to tensors
- Normalized using ImageNet normalization

Training used weak augmentation:

- Random horizontal flip
- Random rotation up to 10 degrees
- Small brightness variation
- Small contrast variation

---

## 5. Model Architecture

EfficientNet-B0 was initialized with ImageNet-pretrained weights.

The backbone was frozen during training.

The final classifier was replaced with:

- Dropout: 0.1
- Linear layer: 1280 → 4 classes

Total parameters:

**4,012,672**

Trainable parameters:

**5,124**

---

## 6. Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.002 |
| Batch Size | 32 |
| Loss Function | CrossEntropyLoss |
| Scheduler | StepLR |
| Step Size | 5 |
| Gamma | 0.1 |
| Epochs | 10 |
| Dropout | 0.1 |
| Input Size | 224 × 224 |

---

## 7. Validation Performance

Best validation accuracy:

**90.09%**

Best validation loss:

**0.2833**

Best epoch:

**Epoch 9**

---

## 8. Test Performance

Test accuracy:

**91.20%**

Weighted Precision:

**91.32%**

Weighted Recall:

**91.20%**

Weighted F1-score:

**91.16%**

### Per-Class Performance

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Glioma | 94.58% | 84.07% | 89.02% |
| Meningioma | 84.23% | 87.04% | 85.61% |
| No Tumor | 93.01% | 98.52% | 95.68% |
| Pituitary | 93.45% | 95.19% | 94.31% |

---

## 9. ROC-AUC

| Class | ROC-AUC |
|---|---:|
| Glioma | 0.9776 |
| Meningioma | 0.9669 |
| No Tumor | 0.9970 |
| Pituitary | 0.9921 |

---

## 10. Explainability

Grad-CAM was implemented to visualize regions contributing to model predictions.

A total of **12 Grad-CAM samples** were generated:

- 3 Glioma samples
- 3 Meningioma samples
- 3 No Tumor samples
- 3 Pituitary samples

Sample accuracy:

**91.67%**

Grad-CAM is provided as an explainability aid and should not be interpreted as clinical evidence.

---

## 11. Model Comparison

A second architecture, ResNet50, was trained and evaluated.

| Metric | EfficientNet-B0 | ResNet50 |
|---|---:|---:|
| Validation Accuracy | 90.09% | 90.00% |
| Test Accuracy | 91.20% | 90.93% |
| Weighted F1 | 91.16% | 90.83% |

EfficientNet-B0 was selected as the final model because it achieved slightly better overall validation and test performance in this experimental setup.

---

## 12. Error Analysis

The final EfficientNet-B0 model produced:

- 985 correct predictions
- 95 incorrect predictions
- Test error rate: 8.80%

Highest class-wise error rate:

**Glioma — 15.93%**

Most frequent failure mode:

**Glioma → Meningioma: 33 cases**

The test set was balanced across the four classes. However, prediction distributions showed that Glioma was slightly under-predicted while No Tumor was slightly over-predicted.

No demographic metadata was available, so demographic fairness could not be directly evaluated.

---

## 13. Limitations

Important limitations include:

1. The model was evaluated on a single public dataset.
2. External clinical validation was not performed.
3. Dataset-specific characteristics may affect generalization.
4. Glioma showed the highest classification error.
5. The dataset does not provide sufficient demographic metadata for demographic bias analysis.
6. Model confidence should not be interpreted as clinical certainty.
7. Grad-CAM highlights model-relevant regions but does not establish medical causality.
8. MRI acquisition conditions, scanners, protocols, and patient populations may differ in real clinical environments.

---

## 14. Ethical Considerations

MediScan should be used responsibly.

- It should not replace qualified medical professionals.
- Predictions should not be used as a standalone diagnostic decision.
- Clinical deployment would require extensive external validation.
- Patient privacy must be protected when handling medical images.
- Potential dataset and model biases must be evaluated before clinical use.
- Human clinical oversight is required for any real-world medical application.

---

## 15. Deployment

MediScan includes a Streamlit web interface supporting:

- Brain MRI image upload
- Predicted class
- Prediction confidence
- Class probabilities
- Grad-CAM visualization
- Medical disclaimer

Supported image formats in the current application:

- JPG
- JPEG
- PNG

The current Colab URL is a temporary development/testing endpoint. A permanent public deployment URL should be used for the final production deployment.

---

## 16. Final Model

**Selected Model:** EfficientNet-B0

**Model Weights:**

`efficientnet_b0_best_day8.pth`

**Final Test Accuracy:** 91.20%

**Final Weighted F1:** 91.16%

**Purpose:** Educational and research demonstration of brain MRI classification and explainable AI.

---

## 17. Disclaimer

MediScan is an educational/research project and is **not a medical device**.

The predictions generated by this model must not be used for diagnosis, treatment, or other clinical decisions without appropriate professional medical evaluation and validated clinical systems.


## Live Deployment

**Streamlit Application:** https://mediscanmedicalimageclassification-ywdozzsxzf5nngqvvb48ve.streamlit.app/

The deployed application provides brain MRI classification,
confidence scores, class probabilities, and Grad-CAM explainability.

> ⚠️ This application is intended for educational/research purposes only
> and must not be used for clinical diagnosis or medical decision-making.
