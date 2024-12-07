# MNIST Classifier with CNN Architecture

A PyTorch implementation of a CNN classifier for MNIST digits using modern architecture and training techniques.

## Model Architecture

The model (`SimpleCNN`) is a convolutional neural network with:
- 4 convolutional layers with feature maps (8→16→32→32)
- Batch normalization after each conv layer
- ReLU activation functions
- Mix of average and max pooling
- Global average pooling
- Fully connected layer (32→10)
- Progressive dropout (2%→5%→10%)

Total parameters: < 20,000

## Training Configuration

The training uses a straightforward approach with the following settings:

### Training Details
- Epochs: 20
- Early stopping patience: 5
- Batch size: 32
- Dataset split: 50,000 training, 10,000 testing
- Optimizer: AdamW with weight decay 0.0001
- Loss: Focal Loss (gamma=2)
- Learning Rate Schedule: OneCycleLR
  - Initial LR: 0.015
  - Max LR: 0.02
  - Pct_start: 0.1
  - Div_factor: 10.0
  - Final_div_factor: 10.0
  - Anneal_strategy: 'cos'

### Data Augmentation
- Random rotation (±10 degrees)
- Random affine translation (±0.1)
- Normalization (mean=0.1307, std=0.3081)

## Requirements

### Model Requirements
1. Parameters < 20,000
2. Use Batch Normalization
3. Include Fully Connected layer
4. Train for ≤ 20 epochs
5. Achieve ≥ 99.4% test accuracy

### Dependencies
- Python 3.11
- PyTorch 2.0+
- torchvision
- numpy
- pytest

## Usage

1. Install dependencies:
bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python train.py
```

3. Run tests:
```bash
python test_requirements.py
```

## Model Performance

The model achieves:
- Test Accuracy: ≥ 99.4%
- Training Time: ~20 epochs
- Memory Efficient: < 20k parameters
```