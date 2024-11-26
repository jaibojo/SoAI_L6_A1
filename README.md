# MNIST Classifier with CI/CD Pipeline

[![ML Pipeline](https://github.com/jaibojo/SoAI_L5_A1/actions/workflows/ml-pipeline.yml/badge.svg?branch=main)](https://github.com/jaibojo/SoAI_L5_A1/actions/workflows/ml-pipeline.yml)

A deep learning project implementing a lightweight CNN classifier for the MNIST dataset with automated testing and CI/CD pipeline. The model achieves >93% accuracy while maintaining a parameter count under 25,000.

## Project Overview

This project demonstrates:
- Efficient CNN architecture design
- Curriculum learning implementation
- Automated testing and CI/CD
- Parameter budget optimization
- Progressive training strategy

## Results and Achievements

### Model Performance
- **Training Accuracy**: 93.98% (Best achieved)
- **Test Accuracy**: >95% (Meeting test requirements)
- **Parameter Count**: 24,494 (Under 25K budget)
- **Training Time**: ~5-10 minutes on CPU

### Key Milestones
1. **Architecture Optimization**:
   - less parameters <25000
   - Efficient parameter utilization across layers

2. **Training Strategy Success**:
   - Implemented 6-stage curriculum learning
   - Progressive difficulty increase (8% → 12% → 15% → 20% → 20% → 25%)
   - Dynamic learning rate adaptation (0.01 → 0.002)
## Project Structure
```
project/
├── model.py              # CNN model architecture (~24K parameters)
├── train.py             # Training script with curriculum learning
├── test_model.py        # Automated model tests
├── requirements.txt     # Project dependencies
├── .gitignore          # Git ignore rules
├── augmentation_samples/# Augmented image samples
│   └── augmented_samples.png
├── models/             # Saved model checkpoints
└── .github/workflows   # CI/CD configuration
    └── ml-pipeline.yml # GitHub Actions workflow
```

## Model Architecture
- **Convolutional Layers**:
  - Conv1: 1→16 channels, 3×3 kernel (160 params)
  - Conv2: 16→32 channels, 3×3 kernel (4,640 params)
  - Conv3: 32→32 channels, 3×3 kernel (9,248 params)
  - Conv4: 32→32 channels, 3×3 kernel (9,248 params)
- **Fully Connected Layers**:
  - FC1: 32→20 neurons (660 params)
  - FC2: 20→10 neurons (210 params)
- **Additional Features**:
  - Batch Normalization after each layer
  - GELU activation functions
  - Progressive dropout (0.1→0.2→0.3)
  - Global Average Pooling

## Training Strategy
### Curriculum Learning
1. Stage 1: Easiest 8% samples (LR: 0.01)
2. Stage 2: Next 12% samples (LR: 0.008)
3. Stage 3: Next 15% samples (LR: 0.006)
4. Stage 4: Next 20% samples (LR: 0.004)
5. Stage 5: Next 20% samples (LR: 0.003)
6. Stage 6: Hardest 25% samples (LR: 0.002)

### Data Augmentation
- Random rotation (±15°)
- Random translation (±10%)
- Random scaling (90-110%)
- Random shearing (±10°)
- Random perspective transformation
- Random erasing

## Testing
Automated tests verify:
1. Parameter count < 25,000
2. Input shape compatibility (28×28)
3. Output shape correctness (10 classes)
4. Model accuracy > 95%
5. Inference time
6. Augmentation consistency
7. Shape preservation
8. Random erasing functionality

## CI/CD Pipeline
GitHub Actions workflow:
1. Triggers on push
2. Sets up Python 3.11
3. Installs dependencies
4. Trains model
5. Runs test suite
6. Saves model artifacts
7. Saves augmentation samples

## Requirements
- Python 3.11
- PyTorch 2.5.1
- TorchVision 0.20.1
- pytest ≥ 6.0.0
- matplotlib ≥ 3.7.1