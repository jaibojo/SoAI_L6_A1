import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import SimpleCNN
from train import load_mnist_data, train

def test_requirement_1_params():
    """Requirement 1: Model should have less than 20k parameters"""
    model = SimpleCNN()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    assert total_params < 20000, f"Model has {total_params:,} parameters, should be < 20,000"

def test_requirement_2_batchnorm():
    """Requirement 2: Model should use Batch Normalization"""
    model = SimpleCNN()
    bn_layers = [m for m in model.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))]
    print(f"\nNumber of BatchNorm layers: {len(bn_layers)}")
    assert len(bn_layers) >= 4, "Model should have at least 4 batch normalization layers"
    assert all(isinstance(bn, nn.BatchNorm2d) for bn in bn_layers), "All batch norm layers should be BatchNorm2d"

def test_requirement_3_fc():
    """Requirement 3: Model should have FC layer"""
    model = SimpleCNN()
    fc_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    
    print(f"\nNumber of FC layers: {len(fc_layers)}")
    if len(fc_layers) > 0:
        print(f"Output features of final FC layer: {fc_layers[-1].out_features}")
    
    assert len(fc_layers) >= 1, "Model should have at least one fully connected layer"
    assert fc_layers[-1].out_features == 10, "Final FC layer should have 10 output features (for MNIST classes)"

def test_requirement_4_epochs():
    """Requirement 4: Training should use less than 20 epochs"""
    num_epochs = train.__defaults__[0]  # Get default value of num_epochs parameter
    print(f"\nNumber of epochs: {num_epochs}")
    assert num_epochs <= 20, "Number of epochs should be <= 20"

def test_requirement_5_dropout():
    """Requirement 5: Model should use progressive dropout"""
    model = SimpleCNN()
    dropout_layers = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    dropout_rates = [layer.p for layer in dropout_layers]
    
    print(f"\nNumber of dropout layers: {len(dropout_layers)}")
    print(f"Dropout rates: {[f'{rate:.1%}' for rate in dropout_rates]}")
    
    assert len(dropout_layers) >= 3, "Model should have at least 3 dropout layers"
    assert 0.02 in dropout_rates, "Model should have 2% dropout"
    assert 0.05 in dropout_rates, "Model should have 5% dropout"
    assert 0.10 in dropout_rates, "Model should have 10% dropout"
    assert sorted(dropout_rates) == [0.02, 0.05, 0.10], "Dropout rates should be progressive (2%, 5%, 10%)"

def test_requirement_6_accuracy():
    """Requirement 6: Model should achieve > 99.4% accuracy"""
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
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    assert accuracy >= 99.4, f"Model accuracy {accuracy:.2f}% is below required 99.4%"

if __name__ == "__main__":
    print("\nTesting Model Requirements")
    print("="*50)
    pytest.main([__file__, "-v"]) 