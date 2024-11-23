import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import SimpleCNN
from datetime import datetime
import os
import math
import random
import torch.nn.functional as F

def get_sample_difficulty(outputs, targets):
    # Calculate sample difficulty based on prediction confidence
    probs = torch.softmax(outputs, dim=1)
    correct_probs = probs[range(len(targets)), targets]
    return 1 - correct_probs  # Higher value = more difficult

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # Initialize model and optimizer
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Print parameter count
    total_params = model.count_parameters()
    print("\nModel Parameter Summary:")
    print("="*50)
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Parameter budget remaining: {25000 - total_params:,}")
    print("="*50)
    
    # First, evaluate difficulty of all samples
    print("\nEvaluating sample difficulties...")
    difficulties = []
    all_data = []
    all_targets = []
    
    # Create a temporary dataloader
    temp_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1000, shuffle=False)
    
    model.eval()
    with torch.no_grad():
        for data, target in temp_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = F.softmax(output, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            difficulties.extend(entropy.cpu().tolist())
            all_data.append(data.cpu())
            all_targets.append(target.cpu())
    
    # Convert to tensors
    difficulties = torch.tensor(difficulties)
    all_data = torch.cat(all_data)
    all_targets = torch.cat(all_targets)
    
    # Sort by difficulty
    sorted_indices = torch.argsort(difficulties)
    num_samples = len(sorted_indices)
    
    # Create 6 stages with increasing difficulty
    stage_indices = [
        sorted_indices[:int(0.08 * num_samples)],                     # Easiest 8%
        sorted_indices[int(0.08 * num_samples):int(0.20 * num_samples)],  # Next 12%
        sorted_indices[int(0.20 * num_samples):int(0.35 * num_samples)],  # Next 15%
        sorted_indices[int(0.35 * num_samples):int(0.55 * num_samples)],  # Next 20%
        sorted_indices[int(0.55 * num_samples):int(0.75 * num_samples)],  # Next 20%
        sorted_indices[int(0.75 * num_samples):]                      # Hardest 25%
    ]
    
    # Learning rates for each stage - even more gradual decrease
    learning_rates = [0.01, 0.008, 0.006, 0.004, 0.003, 0.002]
    
    print("\nStarting staged training...")
    model.train()
    best_accuracy = 0.0
    
    # Train on each stage
    for stage, (indices, lr) in enumerate(zip(stage_indices, learning_rates)):
        print(f"\nStage {stage + 1}: Training with {len(indices)} samples ({len(indices)/num_samples*100:.1f}%), LR: {lr}")
        
        # Create dataset for this stage
        stage_data = all_data[indices]
        stage_targets = all_targets[indices]
        stage_dataset = torch.utils.data.TensorDataset(stage_data, stage_targets)
        stage_loader = torch.utils.data.DataLoader(stage_dataset, batch_size=32, shuffle=True)
        
        # Optimizer for this stage
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
        
        # Training loop for this stage
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(stage_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            running_loss += loss.item()
            
            if batch_idx % 50 == 0:
                accuracy = 100 * correct / total
                avg_loss = running_loss / (batch_idx + 1)
                print(f'Stage {stage + 1}, Batch {batch_idx}/{len(stage_loader)}, '
                      f'Loss: {avg_loss:.4f}, '
                      f'Accuracy: {accuracy:.2f}%, '
                      f'LR: {lr:.6f}')
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    if not os.path.exists('models'):
                        os.makedirs('models')
                    torch.save(model.state_dict(), 'models/best_model.pth')
                    print(f'New best accuracy: {accuracy:.2f}%')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_accuracy = 100 * correct / total
    torch.save(model.state_dict(), f'models/model_{timestamp}_acc{final_accuracy:.1f}.pth')
    print(f"\nTraining completed:")
    print(f"Final Training Accuracy: {final_accuracy:.2f}%")
    print(f"Best Training Accuracy: {best_accuracy:.2f}%")

if __name__ == "__main__":
    train() 