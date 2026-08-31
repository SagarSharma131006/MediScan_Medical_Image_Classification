# MediScan — Day 3: Data Augmentation

## Objective

Implement and verify the data augmentation techniques required for the MediScan project.

## Augmentation Techniques

- Random Horizontal Flip
- Random Rotation
- Color Jitter
- Random Affine Transformation

## Dataset Split

| Split | Images |
|---|---:|
| Training | 5040 |
| Validation | 1080 |
| Testing | 1080 |
| Total | 7200 |

## Augmentation Pipeline

Images are resized to 224 × 224 and then processed using the training augmentation pipeline:

Resize
→ Random Horizontal Flip
→ Random Rotation
→ Color Jitter
→ Random Affine
→ ToTensor

## Verification

Augmented image tensor:

- Shape: 3 × 224 × 224
- Dtype: float32
- Pixel values: within [0, 1]

Final verification:

DAY 3 AUGMENTATION VERIFICATION PASSED

## Visualizations

- day3_augmentation_comparison.png
- day3_combined_augmentation_samples.png

## Next Step

Day 4 — Custom Dataset and DataLoader setup.