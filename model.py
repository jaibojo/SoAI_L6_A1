import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First conv block - keep same
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)    # Params: (3*3*1)*16 + 16 = 160
        self.bn1 = nn.BatchNorm2d(16)                              # Params: 16*2 = 32
        
        # Second conv block - changed from 32 to 16 kernels
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)   # Params: (3*3*16)*16 + 16 = 2,320
        self.bn2 = nn.BatchNorm2d(16)                              # Params: 16*2 = 32
        
        # Third conv block - adjust input channels to match previous layer
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)   # Params: (3*3*16)*32 + 32 = 4,640
        self.bn3 = nn.BatchNorm2d(32)                              # Params: 32*2 = 64
        
        # Fourth conv block - reduced size
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)   # Params: (3*3*32)*32 + 32 = 9,248
        self.bn4 = nn.BatchNorm2d(32)                              # Params: 32*2 = 64
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)                 # Params: 0
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(32)                         # Params: 32*2 = 64
        
        # Fully connected layers - adjusted for new size
        self.fc1 = nn.Linear(32, 20)                               # Params: 32*20 + 20 = 660
        self.bn5 = nn.BatchNorm1d(20)                              # Params: 20*2 = 40
        self.fc2 = nn.Linear(20, 10)                               # Params: 20*10 + 10 = 210
        
        # Total params: ~24,494
        
        self.gelu = nn.GELU()
        # Progressive dropout
        self.dropout1 = nn.Dropout(0.02)   # 2% dropout early
        self.dropout2 = nn.Dropout(0.05)   # 5% dropout medium
        self.dropout3 = nn.Dropout(0.10)   # 10% dropout late
        
        self._initialize_weights()
        
        # Print parameter counts
        print("\nParameter counts by layer:")
        print(f"Conv1: {sum(p.numel() for p in self.conv1.parameters()):,}")
        print(f"Conv2: {sum(p.numel() for p in self.conv2.parameters()):,}")
        print(f"Conv3: {sum(p.numel() for p in self.conv3.parameters()):,}")
        print(f"Conv4: {sum(p.numel() for p in self.conv4.parameters()):,}")
        print(f"FC1: {sum(p.numel() for p in self.fc1.parameters()):,}")
        print(f"FC2: {sum(p.numel() for p in self.fc2.parameters()):,}")
        print(f"Total parameters: {self.count_parameters():,}\n")

    def _initialize_weights(self):
        # Keep the same initialization as before
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    high_freq_idx = m.out_channels // 4
                    m.weight.data[:high_freq_idx] *= 2.0
                    m.bias.data.zero_()
                
            elif isinstance(m, nn.Linear):
                bound = 1 / math.sqrt(m.weight.size(1))
                init.uniform_(m.weight, -bound * 1.5, bound * 1.5)
                if m.bias is not None:
                    m.bias.data.zero_()
                
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.weight.data.fill_(1.1)
                m.bias.data.zero_()
                
            elif isinstance(m, nn.LayerNorm):
                init.constant_(m.weight, 1.0)
                init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)
        
        # First block
        x = self.conv1(x)                    # [B, 16, 28, 28]
        if batch_size > 1:
            x = self.bn1(x)
        x = self.gelu(x)
        x = F.max_pool2d(x, 2, 2)           # [B, 16, 14, 14]
        x = self.dropout1(x)
        
        # Second block
        x = self.conv2(x)                    # [B, 16, 14, 14]
        if batch_size > 1:
            x = self.bn2(x)
        x = self.gelu(x)
        x = F.max_pool2d(x, 2, 2)           # [B, 16, 7, 7]
        x = self.dropout2(x)
        
        # Third block
        x = self.conv3(x)                    # [B, 32, 7, 7]
        if batch_size > 1:
            x = self.bn3(x)
        x = self.gelu(x)
        x = self.dropout2(x)
        
        # Fourth block
        x = self.conv4(x)                    # [B, 32, 7, 7]
        if batch_size > 1:
            x = self.bn4(x)
        x = self.gelu(x)
        x = F.max_pool2d(x, 2, 2)           # [B, 32, 3, 3]
        x = self.dropout3(x)
        
        # Global pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))  # Use functional pooling for all cases
        x = x.view(batch_size, -1)           # Flatten using batch size
        
        # FC layers
        x = self.fc1(x)                      # [B, 20]
        if batch_size > 1:
            x = self.bn5(x)
        x = self.gelu(x)
        x = self.dropout3(x)
        
        x = self.fc2(x)                      # [B, 10]
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())