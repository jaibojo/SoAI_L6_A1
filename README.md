# MNIST Classifier with Progressive Learning

A PyTorch implementation of a CNN classifier for MNIST digits using staged training and progressive learning techniques.

## Model Architecture

The model (`SimpleCNN`) is a convolutional neural network with:

- 3 convolutional layers with feature maps (16→8→16)
- Batch normalization after each conv layer
- GELU activation functions
- Progressive dropout (2%→5%→10%)
- Global average pooling
- 2 fully connected layers (16→16→10)

Total parameters: ~2,962

## Training Approach

The training uses a staged learning approach with curriculum learning:

### Stages
1. Stage 1: 10% easiest samples (lr=0.001)
2. Stage 2: 15% next samples (lr=0.0008)
3. Stage 3: 20% medium samples (lr=0.0006)
4. Stage 4: 25% harder samples (lr=0.0004)
5. Stage 5: 30% hardest samples (lr=0.0002)

### Training Details
- Epochs: 20
- Early stopping patience: 5
- Batch size: 32
- Dataset split: 50,000 training, 10,000 testing
- Optimizer: Adam with stage-specific learning rates
- Loss: CrossEntropy with label smoothing (0.1)

## Requirements

- Python 3.11
- PyTorch 2.5.1
- TorchVision 0.20.1
- pytest ≥ 6.0.0
- matplotlib ≥ 3.7.1