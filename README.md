# MediScan - Medical Image Classification

MediScan is a deep learning project focused on classifying brain MRI images into four medical categories: glioma, meningioma, pituitary tumor, and no tumor. The project follows a structured 15-day workflow covering dataset exploration, preprocessing, model training, evaluation, Grad-CAM explainability, and web app deployment.

## Day 1 - Project Setup and Dataset Exploration

### Objective

The goal of Day 1 was to set up the project environment, connect Google Colab with the GitHub repository, load the brain MRI dataset, and perform an initial dataset exploration.

### Work Completed

- Google Colab environment was successfully configured.
- GitHub repository was cloned inside Colab.
- Brain MRI dataset was loaded from Google Drive.
- Dataset was extracted successfully.
- Dataset folder structure was verified.
- Dataset manifest CSV was created.
- Class distribution was analyzed.
- Sample MRI images were visualized and saved.

### Dataset Overview

| Detail | Value |
|---|---|
| Dataset Type | Brain MRI Images |
| Total Images | 7,200 |
| Number of Classes | 4 |
| Training Images | 5,600 |
| Testing Images | 1,600 |

### Classes

| Class Name | Meaning |
|---|---|
| glioma | MRI images showing glioma tumor |
| meningioma | MRI images showing meningioma tumor |
| pituitary | MRI images showing pituitary tumor |
| notumor | MRI images with no tumor |

### Key Insight

The dataset is well-balanced across all four classes. Each class contains an equal number of training and testing images, which is helpful for building a fair classification model. This reduces the risk of the model becoming biased toward one class during training.

### Day 1 Outputs

| Output File | Purpose |
|---|---|
| `data/processed/day1_dataset_manifest.csv` | Stores image paths, labels, and original split information |
| `figures/day1_sample_images.png` | Shows sample MRI images from each class |
| `reports/day1_summary.md` | Contains Day 1 summary and dataset details |

### Day 1 Status

| Task | Status |
|---|---|
| Colab setup completed | Done |
| GitHub repo cloned | Done |
| Dataset extracted | Done |
| Total 7,200 images verified | Done |
| Four classes verified | Done |
| Sample images visualized | Done |
| Day 1 summary saved | Done |

## Important Note

The dataset file `archive.zip` and extracted `data/raw/` folder are not uploaded to GitHub because the dataset is large. The dataset is stored in Google Drive and used directly in Google Colab.

## Day 2 - Data Preprocessing and Split

- Image dimensions and color modes were analyzed.
- Images will be resized and converted to RGB during training.
- A stratified 70/15/15 train-validation-test split was created.
- Class balance was preserved across all splits.
- Class-to-index mapping was saved for model training.

### Day 2 Outputs

- `data/processed/day2_train_val_test_split.csv`
- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`
- `data/processed/class_to_idx.json`
- `figures/day2_split_distribution.png`
- `reports/day2_summary.md`
