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
import matplotlib.pyplot as plt
import numpy as np

def show_augmented_images_ascii(dataset, num_images=3):
    """Display original and augmented images in terminal using ASCII characters"""
    # Characters to represent different pixel intensities (from dark to light)
    ascii_chars = ' .:-=+*#%@'
    
    # Get random samples
    indices = random.sample(range(len(dataset)), num_images)
    
    for idx in indices:
        print("\n" + "="*80)
        print(f"Image {idx}")
        
        # Original image
        orig_img = dataset.data[idx].numpy()
        print("\nOriginal:")
        
        # Convert to ASCII
        for i in range(0, 28, 2):  # Skip every other row for better aspect ratio
            line = ""
            for j in range(0, 28, 2):  # Skip every other column
                pixel = orig_img[i, j]
                char_idx = int(pixel / 256 * len(ascii_chars))
                line += ascii_chars[char_idx] * 2  # Double the character for better visibility
            print(line)
        
        # Augmented image
        aug_img, _ = dataset[idx]
        aug_img = aug_img.squeeze().numpy()
        print("\nAugmented:")
        
        # Normalize augmented image to 0-255 range for ASCII conversion
        aug_img = (aug_img - aug_img.min()) * 255 / (aug_img.max() - aug_img.min())
        
        for i in range(0, 28, 2):
            line = ""
            for j in range(0, 28, 2):
                pixel = aug_img[i, j]
                char_idx = int(pixel / 256 * len(ascii_chars))
                char_idx = max(0, min(char_idx, len(ascii_chars)-1))  # Ensure index is in bounds
                line += ascii_chars[char_idx] * 2
            print(line)
        
        print("="*80)

def get_sample_difficulty(outputs, targets):
    # Calculate sample difficulty based on prediction confidence
    probs = torch.softmax(outputs, dim=1)
    correct_probs = probs[range(len(targets)), targets]
    return 1 - correct_probs  # Higher value = more difficult

def save_augmented_samples(dataset, num_images=10, save_path='augmentation_samples'):
    """Save original and augmented images as a grid"""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    # Create a figure with subplots
    fig, axes = plt.subplots(2, num_images, figsize=(20, 4))
    
    # Get random samples
    indices = random.sample(range(len(dataset)), num_images)
    
    for i, idx in enumerate(indices):
        # Original image
        orig_img = dataset.data[idx].numpy()
        
        # Augmented image
        aug_img, _ = dataset[idx]
        aug_img = aug_img.squeeze().numpy()
        
        # Plot original
        axes[0, i].imshow(orig_img, cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original')
        
        # Plot augmented
        axes[1, i].imshow(aug_img, cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Augmented')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'augmented_samples.png'))
    plt.close()

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Enhanced augmentation pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomAffine(
            degrees=15,  # Rotation
            translate=(0.1, 0.1),  # Translation
            scale=(0.9, 1.1),  # Scaling
            shear=10  # Shearing
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # Show augmented samples
    print("\nSaving augmented samples visualization...")
    save_augmented_samples(train_dataset, num_images=10)
    print("Augmented samples saved to 'augmentation_samples/augmented_samples.png'")
    print("\nStarting training...")
    
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