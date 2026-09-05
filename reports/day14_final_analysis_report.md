# MediScan — Day 14 Error & Edge Case Analysis

## 1. Overview

Day 14 focused on analyzing model errors, failure modes,
edge cases and prediction-distribution bias using the
EfficientNet-B0 test-set predictions.

## 2. Overall Performance

- Test Samples: **1080**
- Correct Predictions: **985**
- Misclassified Samples: **95**
- Accuracy: **91.20%**
- Error Rate: **8.80%**

## 3. Class-wise Error Analysis

| Class | Accuracy | Error Rate |
|---|---:|---:|
| glioma | 84.07% | 15.93% |
| meningioma | 87.04% | 12.96% |
| notumor | 98.52% | 1.48% |
| pituitary | 95.19% | 4.81% |

### Highest Error Class

**glioma** had the highest error rate at
**15.93%**.

## 4. Major Failure Modes

| True Class | Predicted Class | Errors | % of Errors |
|---|---|---:|---:|
| glioma | meningioma | 33 | 34.74% |
| meningioma | pituitary | 15 | 15.79% |
| meningioma | glioma | 13 | 13.68% |
| glioma | notumor | 8 | 8.42% |
| pituitary | meningioma | 8 | 8.42% |
| meningioma | notumor | 7 | 7.37% |
| pituitary | notumor | 5 | 5.26% |
| notumor | meningioma | 3 | 3.16% |
| glioma | pituitary | 2 | 2.11% |
| notumor | pituitary | 1 | 1.05% |

### Dominant Failure Mode

The largest error pattern was:

**glioma → meningioma**

- Errors: **33**
- Share of all errors: **34.74%**

## 5. Bias Analysis

The test set contains **270 samples per class**, making it
class-balanced.

Prediction distribution:

- Glioma: **-2.78 percentage points**
- Meningioma: **+0.83 percentage points**
- No Tumor: **+1.48 percentage points**
- Pituitary: **+0.46 percentage points**

Glioma was the most under-predicted class, while No Tumor
was the most over-predicted class.

The observed prediction shift suggests some model-level
prediction bias, but the test dataset itself is balanced.

## 6. Visual Inspection

A representative set of **10 misclassified MRI samples**
was generated for visual inspection.

Figure:

`figures/day14_misclassified_samples_visualization.png`

## 7. Identified Failure Patterns

- Glioma has the highest error rate.
- Glioma and meningioma are the most frequently confused classes.
- Meningioma also shows confusion with pituitary.
- Glioma is slightly under-predicted.
- No Tumor is highly accurate compared with the other classes.
- The test set is balanced, so class imbalance does not explain
  the observed error pattern.

## 8. Limitations

- Visually similar tumor classes can be difficult to distinguish.
- Model performance is not uniform across classes.
- Dataset-specific patterns may limit generalization.
- Additional independent validation is required before considering
  real-world use.
- This project is for educational and research purposes only.

## 9. Recommended Improvements

1. Increase training-data diversity.
2. Investigate difficult glioma and meningioma samples.
3. Explore medically appropriate augmentation.
4. Fine-tune deeper backbone layers.
5. Evaluate the model on an independent external dataset.

## 10. Status

**Day 14 — COMPLETED ✅**
