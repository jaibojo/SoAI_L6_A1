import torch
import pytest
from torchvision import datasets, transforms
from model import SimpleCNN
import time
import numpy as np

@pytest.fixture
def model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return SimpleCNN().to(device)

def print_test_result(test_name, passed, expected=None, actual=None, message=None):
    """Helper function to print test results in a formatted way"""
    status = "\033[92mPASSED\033[0m" if passed else "\033[91mFAILED\033[0m"
    print(f"\n{test_name}: {status}")
    if expected is not None:
        print(f"Expected: {expected}")
    if actual is not None:
        print(f"Actual: {actual}")
    if message and not passed:
        print(f"Message: {message}")
    print("-" * 50)

def test_parameter_count(model):
    total_params = sum(p.numel() for p in model.parameters())
    passed = total_params < 25000
    print_test_result(
        "Parameter Count Test",
        passed,
        expected="< 25,000",
        actual=f"{total_params:,}",
        message=f"Model has {total_params:,} parameters, should be less than 25,000"
    )
    assert passed

def test_input_shape(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_input = torch.randn(1, 1, 28, 28).to(device)
    try:
        output = model(test_input)
        passed = True
        message = None
    except Exception as e:
        passed = False
        message = str(e)
    
    print_test_result(
        "Input Shape Test",
        passed,
        expected="Model processes 28x28 input",
        actual="Success" if passed else "Failed to process input",
        message=message
    )
    assert passed

def test_output_shape(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_input = torch.randn(1, 1, 28, 28).to(device)
    output = model(test_input)
    passed = output.shape[1] == 10
    print_test_result(
        "Output Shape Test",
        passed,
        expected="10 classes",
        actual=f"{output.shape[1]} classes",
        message=f"Output should have 10 classes, got {output.shape[1]}"
    )
    assert passed

def test_model_accuracy():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
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
    passed = accuracy > 95
    print_test_result(
        "Model Accuracy Test",
        passed,
        expected="> 95%",
        actual=f"{accuracy:.2f}%",
        message=f"Accuracy is {accuracy:.2f}%, should be > 95%"
    )
    assert passed

def test_inference_time():
    """Test model inference time"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    model.eval()
    
    # Warm up
    for _ in range(10):
        test_input = torch.randn(1, 1, 28, 28).to(device)
        with torch.no_grad():
            _ = model(test_input)
    
    # Measure inference time
    times = []
    for _ in range(100):
        test_input = torch.randn(1, 1, 28, 28).to(device)
        start_time = time.time()
        with torch.no_grad():
            _ = model(test_input)
        times.append(time.time() - start_time)
    
    avg_time = np.mean(times)
    passed = avg_time < 0.01
    print_test_result(
        "Inference Time Test",
        passed,
        expected="< 10ms",
        actual=f"{avg_time*1000:.2f}ms",
        message=f"Inference too slow: {avg_time*1000:.2f}ms per image"
    )
    assert passed

def test_augmentation_consistency():
    """Test if augmentation produces different results for same image"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomAffine(
            degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = datasets.MNIST('./data', train=True, download=True)
    image = dataset.data[0].numpy()
    
    augmented_images = []
    for _ in range(5):
        aug_img = transform(image)
        augmented_images.append(aug_img.numpy())
    
    # Check if images are different
    differences = []
    for i in range(len(augmented_images)-1):
        diff = not np.allclose(augmented_images[i], augmented_images[i+1])
        differences.append(diff)
    
    passed = all(differences)
    print_test_result(
        "Augmentation Consistency Test",
        passed,
        expected="Different augmentations for same image",
        actual=f"{sum(differences)}/4 image pairs are different",
        message="Augmentation not producing different results"
    )
    assert passed

def test_augmentation_bounds():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomAffine(
            degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10
        ),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = datasets.MNIST('./data', train=True, download=True)
    image = dataset.data[0].numpy()
    
    values_in_range = []
    for _ in range(10):
        aug_img = transform(image)
        in_range = aug_img.min() >= -5 and aug_img.max() <= 5
        values_in_range.append(in_range)
    
    passed = all(values_in_range)
    print_test_result(
        "Augmentation Bounds Test",
        passed,
        expected="Values between -5 and 5",
        actual=f"Range: [{aug_img.min():.2f}, {aug_img.max():.2f}]",
        message="Augmented image values out of expected range"
    )
    assert passed

def test_augmentation_shape_preservation():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomAffine(
            degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = datasets.MNIST('./data', train=True, download=True)
    image = dataset.data[0].numpy()
    original_shape = image.shape
    
    shapes_preserved = []
    for _ in range(10):
        aug_img = transform(image)
        shapes_preserved.append(aug_img.shape[1:] == original_shape)
    
    passed = all(shapes_preserved)
    print_test_result(
        "Shape Preservation Test",
        passed,
        expected=f"Shape {original_shape}",
        actual=f"Shape {aug_img.shape[1:]}",
        message="Augmentation changed image dimensions"
    )
    assert passed

def test_random_erasing():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomErasing(p=1.0, scale=(0.02, 0.1))
    ])
    
    dataset = datasets.MNIST('./data', train=True, download=True)
    image = dataset.data[0].numpy()
    
    aug_img = transform(image)
    has_erased = torch.any(aug_img == 0)
    
    print_test_result(
        "Random Erasing Test",
        has_erased,
        expected="Contains erased regions",
        actual="Erased regions found" if has_erased else "No erased regions",
        message="Random erasing not applied"
    )
    assert has_erased