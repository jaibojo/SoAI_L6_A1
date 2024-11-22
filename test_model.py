import torch
import pytest
from torchvision import datasets, transforms
from model import SimpleCNN

@pytest.fixture
def model():
    return SimpleCNN()

def test_parameter_count(model):
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 100000, f"Model has {total_params} parameters, should be less than 100000"

def test_input_shape(model):
    test_input = torch.randn(1, 1, 28, 28)
    try:
        output = model(test_input)
    except:
        pytest.fail("Model failed to process 28x28 input")

def test_output_shape(model):
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape[1] == 10, f"Output should have 10 classes, got {output.shape[1]}"

def test_model_accuracy():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    
    # Load test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    # Load latest model
    import glob
    import os
    model_files = glob.glob('models/model_*.pth')
    latest_model = max(model_files, key=os.path.getctime)
    model.load_state_dict(torch.load(latest_model))
    
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
    assert accuracy > 80, f"Accuracy is {accuracy}%, should be > 80%" 