import pytest
import torch
from model import SimpleCNN
from train import load_mnist_data
import torch.nn as nn

def test_parameter_count():
    """Test if model has less than 20,000 parameters"""
    model = SimpleCNN()
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 20000, f"Model has {total_params:,} parameters, should be < 20,000"

def test_has_batch_norm():
    """Test if model uses batch normalization"""
    model = SimpleCNN()
    has_bn = any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) for m in model.modules())
    assert has_bn, "Model should use batch normalization"

def test_has_dropout():
    """Test if model uses dropout with correct rates"""
    model = SimpleCNN()
    dropout_rates = []
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            dropout_rates.append(m.p)
    
    assert len(dropout_rates) >= 3, "Model should have at least 3 dropout layers"
    assert 0.02 in dropout_rates, "Model should have 2% dropout"
    assert 0.05 in dropout_rates, "Model should have 5% dropout"
    assert 0.10 in dropout_rates, "Model should have 10% dropout"

def test_epoch_count():
    """Test if training epochs are less than 20"""
    from train import train
    assert train.__defaults__[0] <= 20, "Number of epochs should be <= 20"

def test_training_dataset_size():
    """Test if training dataset has exactly 50,000 samples"""
    train_loader, _ = load_mnist_data(batch_size=32)
    
    # Calculate total samples in training loader
    train_samples = len(train_loader.dataset)  # Direct dataset length
    
    assert train_samples == 50000, f"Training set should have 50,000 samples, got {train_samples}"

def test_test_dataset_size():
    """Test if test dataset has exactly 10,000 samples"""
    _, test_loader = load_mnist_data(batch_size=32)
    
    # Calculate total samples in test loader
    test_samples = len(test_loader.dataset)  # Direct dataset length
    
    assert test_samples == 10000, f"Test set should have 10,000 samples, got {test_samples}"

def test_model_accuracy():
    """Test if model achieves 99.4% test accuracy"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    
    # Load a pre-trained model if available
    try:
        model.load_state_dict(torch.load('models/best_model.pth'))
    except:
        pytest.skip("No pre-trained model found to test accuracy")
    
    train_loader, test_loader = load_mnist_data(batch_size=32)
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    assert accuracy >= 99.4, f"Model accuracy {accuracy:.2f}% is below required 99.4%"

def test_model_architecture():
    """Test various architectural requirements"""
    model = SimpleCNN()
    
    # Test conv layer progression
    conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    assert len(conv_layers) == 4, "Model should have exactly 4 conv layers"
    
    # Test feature map progression
    feature_maps = [conv.out_channels for conv in conv_layers]
    assert feature_maps == [16, 16, 32, 32], f"Incorrect feature map progression: {feature_maps}"
    
    # Test FC layer sizes
    fc_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert len(fc_layers) == 2, "Model should have exactly 2 FC layers"
    assert fc_layers[-1].out_features == 10, "Final layer should output 10 classes"

if __name__ == "__main__":
    pytest.main([__file__]) 