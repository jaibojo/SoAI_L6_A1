import pytest
import torch
from model import SimpleCNN
from train import load_mnist_data
import torch.nn as nn
import torch.nn.functional as F

def test_parameter_count():
    """Test if model has less than 20,000 parameters"""
    model = SimpleCNN()
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 20000, f"Model has {total_params:,} parameters, should be < 20,000"

def test_has_batch_norm():
    """Test if model uses batch normalization"""
    model = SimpleCNN()
    bn_layers = [m for m in model.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))]
    assert len(bn_layers) >= 4, "Model should have at least 4 batch normalization layers"
    assert all(isinstance(bn, nn.BatchNorm2d) for bn in bn_layers), "All batch norm layers should be BatchNorm2d"

def test_has_pooling():
    """Test if model uses pooling layers"""
    model = SimpleCNN()
    
    # Get the source code of the forward method to check for pooling operations
    forward_code = model.forward.__code__.co_code
    
    # Check if the model's forward pass contains pooling operations
    has_avg_pool = 'avg_pool2d' in str(forward_code)
    has_max_pool = 'max_pool2d' in str(forward_code)
    
    assert has_avg_pool or has_max_pool, "Model should use pooling layers"
    assert has_avg_pool, "Model should use average pooling"
    assert has_max_pool, "Model should use max pooling"

def test_has_fc_layer():
    """Test if model has fully connected layer with correct output size"""
    model = SimpleCNN()
    fc_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    
    assert len(fc_layers) >= 1, "Model should have at least one fully connected layer"
    assert fc_layers[-1].out_features == 10, "Final FC layer should have 10 output features (for MNIST classes)"

def test_epoch_count():
    """Test if training epochs are less than 20"""
    from train import train
    assert train.__defaults__[0] <= 20, "Number of epochs should be <= 20"
    assert train.__defaults__[0] == 20, "Number of epochs should be exactly 20"

def test_model_accuracy():
    """Test if model achieves 99.4% test accuracy"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    
    # Load a pre-trained model if available
    try:
        model.load_state_dict(torch.load('models/best_model.pth'))
    except:
        pytest.skip("No pre-trained model found to test accuracy")
    
    _, test_loader = load_mnist_data(batch_size=32)
    
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

if __name__ == "__main__":
    pytest.main([__file__]) 