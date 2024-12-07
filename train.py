import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import SimpleCNN
from datetime import datetime
import os
from torch.utils.data import random_split, DataLoader
import math
import numpy as np
import torch.nn.functional as F

def load_mnist_data(batch_size=32, train_size=50000, test_size=10000):
    """
    Load and split MNIST dataset with moderate augmentation
    """
    transform_train = transforms.Compose([
        transforms.RandomRotation(10),  # Back to 10 degrees
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Back to 0.1
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load full training dataset
    full_dataset = datasets.MNIST(
        root='./data', 
        train=True,
        download=True,
        transform=transform_train
    )
    
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform_test
    )

    # Split into train and test
    train_dataset, _ = random_split(
        full_dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
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

def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def get_lr(epoch, stage_num, current_loss, prev_loss=None, total_epochs=5, total_stages=5):
    # Base learning rate from epoch and stage
    epoch_factor = 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))
    stage_factor = 1.0 - (stage_num - 1) / total_stages
    base_lr = initial_lr * epoch_factor * stage_factor
    
    # Loss-based adjustment
    if prev_loss is not None:
        loss_change = (current_loss - prev_loss) / prev_loss
        
        if loss_change > 0.1:  # Loss increased significantly
            loss_factor = 0.5
        elif loss_change < -0.1:  # Loss decreased significantly
            loss_factor = 1.2
        else:
            loss_factor = 1.0
            
        base_lr *= loss_factor
    
    return max(base_lr, 1e-5)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

criterion = FocalLoss(gamma=2)  # Higher gamma gives more weight to hard examples

def train(num_epochs=20, patience=5, min_delta=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model and optimizer
    model = SimpleCNN().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print("\nModel Configuration:")
    print("="*70)
    print(f"Total Parameters: {num_params:,}")
    print(f"Architecture: 8→16→32→32 channels")
    print(f"Batch Size: 32")
    print(f"Initial LR: 0.015")
    print(f"Max LR: 0.02")
    print(f"Weight Decay: 0.0001")
    print(f"Scheduler: OneCycleLR (pct_start=0.1, div_factor=10.0)")
    print(f"Dropout: Early 2% → Mid 5% → Late 10%")
    print("="*70 + "\n")
    
    optimizer = optim.AdamW(model.parameters(), lr=0.015, weight_decay=0.0001)
    criterion = FocalLoss(gamma=2)
    
    # Store results for summary table
    results = []
    best_accuracy = 0
    epochs_without_improvement = 0
    
    # Load data with DataLoader for random batching
    train_loader, test_loader = load_mnist_data(batch_size=32)
    
    # Calculate steps per epoch for scheduler
    steps_per_epoch = len(train_loader)
    
    # Single scheduler for all epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.02,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,  # Quick warmup
        div_factor=10.0,
        final_div_factor=10.0,
        anneal_strategy='cos'
    )
    
    print(f"Training Strategy:")
    print("="*70)
    print(f"Training for {num_epochs} epochs with batch_size=32")
    print(f"Total steps per epoch: {steps_per_epoch}")
    print("="*70 + "\n")
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        print(f"\nEpoch {epoch + 1}")
        print("----------------------------------------")
        print(f"Batch Size: 32")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        print(f"Training mode: Random batches")
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            _, predicted = torch.max(output.data, 1)
            epoch_total += target.size(0)
            epoch_correct += (predicted == target).sum().item()
            epoch_loss += loss.item()
            current_lr = scheduler.get_last_lr()[0]
            
            if batch_idx % 50 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Accuracy: {100 * epoch_correct / epoch_total:.2f}%')
                print(f'Learning rate: {current_lr:.6f}')
        
        # Evaluation
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
        train_accuracy = 100 * epoch_correct / epoch_total
        
        print(f'\nEpoch {epoch + 1} Summary:')
        print(f'Training Accuracy: {train_accuracy:.2f}%')
        print(f'Test Accuracy: {test_accuracy:.2f}%')
        print(f'Average Loss: {epoch_loss / epoch_total:.4f}')
        
        # Store results for summary table
        results.append({
            'epoch': epoch + 1,
            'train_acc': train_accuracy,
            'test_acc': test_accuracy,
            'loss': epoch_loss / epoch_total,
            'lr': current_lr
        })
        
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
    
    # After training loop, print summary table
    print("\n" + "="*100)
    print("Training Summary:")
    print("="*100)
    print(f"Model Parameters: {num_params:,}")
    print(f"Batch Size: 32")
    print(f"Initial LR: 0.015, Max LR: 0.02")
    print(f"Architecture: 8→16→32→32 channels")
    print("-"*100)
    print(f"{'Epoch':<10} {'Train Accuracy':<20} {'Test Accuracy':<20} {'Loss':<10} {'LR':<10}")
    print("-"*100)
    
    for epoch_results in results:
        print(f"{epoch_results['epoch']:<10} "
              f"{epoch_results['train_acc']:>18.2f}% "
              f"{epoch_results['test_acc']:>18.2f}% "
              f"{epoch_results['loss']:>10.4f} "
              f"{epoch_results['lr']:>10.6f}")
    
    print("="*100)
    print(f"Best Test Accuracy: {best_accuracy:.2f}%")
    print("="*100 + "\n")

if __name__ == "__main__":
    train(num_epochs=20, patience=5, min_delta=0.001) 