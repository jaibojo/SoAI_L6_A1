import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import SimpleCNN
from datetime import datetime
import os
from torch.utils.data import random_split, DataLoader

def load_mnist_data(batch_size=32, train_size=50000, test_size=10000):
    """
    Load and split MNIST dataset into train and test sets
    Args:
        batch_size: size of batches for dataloaders
        train_size: number of samples for training (default 50000)
        test_size: number of samples for testing (default 10000)
    Returns:
        train_loader, test_loader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    full_dataset = datasets.MNIST(
        root='./data', 
        train=True,
        download=True,
        transform=transform
    )

    train_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, test_loader

def train(num_epochs=5, patience=5, min_delta=0.001):
    """
    Train the model with staged learning and early stopping
    Each epoch runs through all stages before moving to next epoch
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader, test_loader = load_mnist_data(batch_size=32)
    
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Early stopping variables
    best_accuracy = 0.0
    epochs_without_improvement = 0
    
    # Convert train_loader to list for easier splitting
    all_data = []
    all_targets = []
    print("\nPreparing data for staged training...")
    
    for data, target in train_loader:
        all_data.append(data)
        all_targets.append(target)
    
    all_data = torch.cat(all_data)
    all_targets = torch.cat(all_targets)
    num_samples = len(all_targets)
    
    # Define stages with percentages and learning rates
    stages = [
        {"size": 0.10, "lr": 0.001},   # 10% easiest samples, highest lr
        {"size": 0.15, "lr": 0.0008},  # 15% next samples
        {"size": 0.20, "lr": 0.0006},  # 20% medium samples
        {"size": 0.25, "lr": 0.0004},  # 25% harder samples
        {"size": 0.30, "lr": 0.0002},  # 30% hardest samples, lowest lr
    ]
    
    # Prepare data loaders for all stages
    stage_loaders = []
    start_idx = 0
    for stage_num, stage in enumerate(stages, 1):
        size = int(stage["size"] * num_samples)
        end_idx = start_idx + size
        
        stage_data = all_data[start_idx:end_idx]
        stage_targets = all_targets[start_idx:end_idx]
        stage_dataset = torch.utils.data.TensorDataset(stage_data, stage_targets)
        stage_loader = torch.utils.data.DataLoader(
            stage_dataset, 
            batch_size=32, 
            shuffle=True
        )
        stage_loaders.append({
            "loader": stage_loader,
            "lr": stage["size"],
            "size": size
        })
        start_idx = end_idx
    
    # Training loop - epoch first, then stages
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        # Run through each stage for this epoch
        for stage_num, (stage, stage_data) in enumerate(zip(stages, stage_loaders), 1):
            print(f"\nStage {stage_num}:")
            print(f"Samples: {stage_data['size']:,} ({stage['size']*100:.1f}%)")
            print(f"Learning Rate: {stage['lr']}")
            
            # Optimizer for this stage
            optimizer = optim.Adam(model.parameters(), lr=stage["lr"])
            
            # Training loop for this stage
            model.train()
            stage_loss = 0.0
            stage_correct = 0
            stage_total = 0
            
            for batch_idx, (data, target) in enumerate(stage_data["loader"]):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                _, predicted = torch.max(output.data, 1)
                stage_total += target.size(0)
                stage_correct += (predicted == target).sum().item()
                stage_loss += loss.item()
                
                if batch_idx % 50 == 0:
                    accuracy = 100 * stage_correct / stage_total
                    avg_loss = stage_loss / (batch_idx + 1)
                    print(f'Epoch {epoch + 1}, Stage {stage_num}, Batch {batch_idx}/{len(stage_data["loader"])}, '
                          f'Loss: {avg_loss:.4f}, '
                          f'Accuracy: {accuracy:.2f}%')
            
            # Accumulate epoch statistics
            epoch_loss += stage_loss
            epoch_correct += stage_correct
            epoch_total += stage_total
        
        # Evaluation on test set after completing all stages
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                test_total += target.size(0)
                test_correct += (predicted == target).sum().item()
        
        test_accuracy = 100 * test_correct / test_total
        print(f'\nEpoch {epoch + 1} Summary:')
        print(f'Training Accuracy: {100 * epoch_correct / epoch_total:.2f}%')
        print(f'Test Accuracy: {test_accuracy:.2f}%')
        
        # Early stopping check
        if test_accuracy > best_accuracy + min_delta:
            best_accuracy = test_accuracy
            epochs_without_improvement = 0
            if not os.path.exists('models'):
                os.makedirs('models')
            torch.save(model.state_dict(), 'models/best_model.pth')
            print(f'New best test accuracy: {test_accuracy:.2f}%')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f'\nEarly stopping triggered after {epoch + 1} epochs')
                break
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    torch.save(model.state_dict(), f'models/model_{timestamp}_acc{test_accuracy:.1f}.pth')
    print(f"\nTraining completed:")
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")
    print(f"Best Test Accuracy: {best_accuracy:.2f}%")

if __name__ == "__main__":
    train(num_epochs=20, patience=5, min_delta=0.001) 