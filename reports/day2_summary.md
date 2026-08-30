
# Day 2 Summary - Data Preprocessing and Split

## Work Completed

- Day 1 dataset manifest loaded successfully.
- Image dimensions and image modes were inspected.
- All future model inputs will be converted to RGB format.
- Dataset was split into train, validation, and test sets.
- Stratified splitting was used to keep all classes balanced.
- Class-to-index mapping was created.
- Class distribution visualization was saved.

## Split Ratio

| Split | Images |
|---|---:|
| Train | 5040 |
| Validation | 1080 |
| Test | 1080 |
| Total | 7200 |

## Classes

|            |   0 |
|:-----------|----:|
| glioma     |   0 |
| meningioma |   1 |
| notumor    |   2 |
| pituitary  |   3 |

## Key Insight

The dataset remains balanced after the 70/15/15 split. This is important because a balanced split helps the model learn all classes fairly and gives a more reliable validation and testing result.

## Output Files

- `data/processed/day2_image_info.csv`
- `data/processed/day2_train_val_test_split.csv`
- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`
- `data/processed/class_to_idx.json`
- `figures/day2_split_distribution.png`
