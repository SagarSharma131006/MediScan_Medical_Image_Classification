# MediScan — Day 10 Grad-CAM Explainability

## Objective

Grad-CAM was implemented to visualize the image regions that contributed to the EfficientNet-B0 model's predicted class.

## Grad-CAM Configuration

- Architecture: EfficientNet-B0
- Dataset: Brain MRI classification
- Number of classes: 4
- Input resolution: 224 × 224
- Target layer: Final feature block of EfficientNet-B0
- Target class: Model's predicted class
- Visualization: Grad-CAM heatmap overlaid on MRI image

## Explainability Samples

A total of **12 Grad-CAM samples** were generated, with three samples selected from each class.

- **glioma:** 3 samples, 3 correct, 0 incorrect
- **meningioma:** 3 samples, 3 correct, 0 incorrect
- **notumor:** 3 samples, 3 correct, 0 incorrect
- **pituitary:** 3 samples, 2 correct, 1 incorrect

## Sample Prediction Summary

- Total Grad-CAM samples: **12**
- Correct predictions: **11**
- Incorrect predictions: **1**
- Sample accuracy: **91.67%**
- Average confidence: **89.52%**

## Incorrect Prediction

- True class: **pituitary**
- Predicted class: **notumor**
- Prediction confidence: **62.26%**
- Grad-CAM visualization: `day10_gradcam_sample_11_pituitary_pred_notumor.png`

## Interpretation

The Grad-CAM heatmaps provide a visual indication of the image regions contributing to the model's prediction. They can help inspect whether the model is focusing on meaningful image regions rather than relying on obviously irrelevant areas.

The generated examples include both correct and incorrect predictions. The incorrect pituitary example is particularly useful for qualitative inspection because the model predicted `notumor` instead of the true `pituitary` class.

## Important Limitation

Grad-CAM is an explainability aid and should not be interpreted as proof that the highlighted region represents the actual clinical cause of a diagnosis. The visualization describes model behavior, not medical reasoning.

This model is intended for research and educational purposes and is not a substitute for professional medical diagnosis.

## Generated Artifacts

- `day10_gradcam_samples.csv` — Grad-CAM sample metadata
- `day10_gradcam_sample_01...12.png` — 12 explainability samples
- `day10_gradcam_contact_sheet.png` — combined visualization

## Day 10 Status

Grad-CAM implementation and explainability sample generation completed successfully.