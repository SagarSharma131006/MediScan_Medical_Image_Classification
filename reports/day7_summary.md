# Day 7 — Model Training

## Objective

Train the pretrained EfficientNet-B0 model and monitor training and validation performance.

## Training Configuration

- Architecture: EfficientNet-B0
- Pretrained: ImageNet
- Number of classes: 4
- Trainable parameters: 5,124
- Frozen parameters: 4,007,548
- Training samples: 5,040
- Validation samples: 1,080
- Batch size: 32
- Epochs: 10
- Loss function: CrossEntropyLoss
- Optimizer: Adam
- Initial learning rate: 0.001
- Scheduler: StepLR
- Scheduler step size: 5
- Scheduler gamma: 0.1
- Device: CUDA

## Results

- Best validation accuracy: 87.22%
- Best validation epoch: 6
- Final training accuracy: 85.73%
- Final validation accuracy: 86.67%
- Best validation loss: 0.3943

## Observations

- Training loss decreased substantially during training.
- Validation loss generally decreased throughout training.
- Validation accuracy reached 87.22%.
- The project target of greater than 85% validation accuracy was achieved.
- Learning rate was reduced from 0.001 to 0.0001 after epoch 5.
- No test-set evaluation was performed during Day 7.

## Saved Artifacts

- Best model: `models/efficientnet_b0_best_day7.pth`
- Loss curve: `figures/day7_loss_curve.png`
- Accuracy curve: `figures/day7_accuracy_curve.png`
- Training history: `reports/day7_training_history.csv`
- Training history JSON: `reports/day7_training_history.json`

## Day 7 Status

**COMPLETED**
