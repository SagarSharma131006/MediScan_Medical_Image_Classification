# MediScan — Day 9 Evaluation Summary

## Objective

Evaluate the trained EfficientNet-B0 model on the held-out test set using
classification metrics, confusion matrix, and multi-class ROC-AUC analysis.

## Test Dataset

- Total test images: 1080
- Classes: 4
- Images per class: 270
- Test set was not used during hyperparameter tuning.

## Overall Performance

| Metric | Score |
|---|---:|
| Accuracy | 91.20% |
| Weighted Precision | 91.32% |
| Weighted Recall | 91.20% |
| Weighted F1-score | 91.16% |

## Per-Class Performance

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Glioma | 94.58% | 84.07% | 89.02% |
| Meningioma | 84.23% | 87.04% | 85.61% |
| No Tumor | 93.01% | 98.52% | 95.68% |
| Pituitary | 93.45% | 95.19% | 94.31% |

## Confusion Matrix

The confusion matrix was generated and saved as:

`figures/day9_confusion_matrix.png`

Confusion matrix:

| Actual \ Predicted | Glioma | Meningioma | No Tumor | Pituitary |
|---|---:|---:|---:|---:|
| Glioma | 227 | 33 | 8 | 2 |
| Meningioma | 13 | 235 | 7 | 15 |
| No Tumor | 0 | 3 | 266 | 1 |
| Pituitary | 0 | 8 | 5 | 257 |

## ROC-AUC Analysis

| Class | ROC-AUC |
|---|---:|
| Glioma | 0.9776 |
| Meningioma | 0.9669 |
| No Tumor | 0.9970 |
| Pituitary | 0.9921 |

ROC curves were generated and saved as:

`figures/day9_roc_curves.png`

## Observations

- The model achieved 91.20% accuracy on the unseen test set.
- No Tumor achieved the strongest classification performance with an F1-score
  of 95.68% and ROC-AUC of 0.9970.
- Pituitary also showed strong performance with an F1-score of 94.31%.
- Glioma had high precision (94.58%) but comparatively lower recall (84.07%).
- Meningioma was the most challenging class based on F1-score (85.61%).
- All four classes achieved ROC-AUC values above 0.96, indicating strong
  class-separation capability.

## Day 9 Conclusion

The EfficientNet-B0 model demonstrated strong generalization on the held-out
test set, achieving 91.20% accuracy and a weighted F1-score of 91.16%.

The evaluation confirms that the optimized Day 8 configuration performs well
on unseen MRI images. The test set remained untouched during hyperparameter
tuning, providing an independent evaluation of the final model.

## Saved Artifacts

- `reports/day9_classification_report.csv`
- `reports/day9_roc_auc.csv`
- `reports/day9_evaluation_metrics.json`
- `reports/day9_summary.md`
- `figures/day9_confusion_matrix.png`
- `figures/day9_roc_curves.png`