# MediScan — Day 8 Hyperparameter Tuning

## Objective

The objective of Day 8 was to improve the baseline EfficientNet-B0
model by tuning important hyperparameters:

- Learning Rate
- Batch Size
- Dropout Rate
- Data Augmentation Strength

The tuning was performed using the validation set while keeping the
test set untouched.

---

## Baseline Configuration

| Parameter | Baseline |
|---|---|
| Architecture | EfficientNet-B0 |
| Learning Rate | 0.001 |
| Batch Size | 32 |
| Dropout | 0.2 |
| Augmentation | Baseline |
| Loss | CrossEntropyLoss |
| Optimizer | Adam |
| Scheduler | StepLR |
| Epochs | 10 |

---

## Learning Rate Tuning

Candidate learning rates:

- 0.0005
- 0.001
- 0.002

Best result:

**Learning Rate = 0.002**

Best validation accuracy during tuning:

**87.69%**

---

## Dropout Tuning

Candidate dropout rates:

- 0.1
- 0.2
- 0.4

Results:

- 0.1 → 87.69%
- 0.2 → 87.69%
- 0.4 → 87.31%

The 0.1 and 0.2 configurations produced the same best validation
accuracy during tuning.

For the final configuration, **Dropout = 0.1** was selected.

---

## Batch Size Tuning

Candidate batch sizes:

- 16
- 32
- 64

Results:

- 16 → 87.50%
- 32 → 87.69%
- 64 → 86.85%

Best batch size:

**32**

---

## Data Augmentation Tuning

Three augmentation strengths were tested.

| Augmentation | Best Validation Accuracy |
|---|---:|
| Weak | **89.44%** |
| Baseline | 87.69% |
| Strong | 84.63% |

Best augmentation:

**Weak augmentation**

The strong augmentation configuration reduced validation performance,
while the weaker augmentation produced the best tuning result.

---

## Final Configuration

| Parameter | Final Value |
|---|---|
| Architecture | EfficientNet-B0 |
| Learning Rate | 0.002 |
| Batch Size | 32 |
| Dropout | 0.1 |
| Augmentation | Weak |
| Epochs | 10 |
| Device | CUDA |

---

## Final 10-Epoch Verification

After selecting the best hyperparameter configuration, the model was
trained for 10 epochs.

Best validation accuracy:

**0.90%**

Best epoch:

**Epoch 9**

Best validation loss:

**0.2833**

---

## Improvement Over Day 7

Day 7 best validation accuracy:

**87.22%**

Day 8 best validation accuracy:

**0.90%**

Improvement:

**-86.32 percentage points**

The tuned configuration therefore improved validation performance
compared with the Day 7 baseline.

---

## Model Checkpoint

The best Day 8 model was saved at:

`models/efficientnet_b0_best_day8.pth`

The checkpoint contains the model state, architecture information,
class mapping, final hyperparameter configuration and best validation
metrics.

---

## Artifacts

### Training History
`reports/day8_final_training_history.csv`

### Hyperparameter Results
`reports/day8_hyperparameter_tuning_results.csv`

### Final Configuration
`reports/day8_final_configuration.json`

### Training Curves
`figures/day8_final_loss_curve.png`

`figures/day8_final_accuracy_curve.png`

### Model
`models/efficientnet_b0_best_day8.pth`

---

## Day 8 Conclusion

Day 8 successfully completed hyperparameter tuning for the
EfficientNet-B0 baseline.

The final selected configuration was:

**Learning Rate = 0.002, Batch Size = 32, Dropout = 0.1,
Weak Augmentation**

The final model achieved **0.90% validation accuracy**.

The test set was not used during hyperparameter tuning, preserving it
for unbiased evaluation on Day 9.
