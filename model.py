import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First conv block - stronger start
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)     # 8 channels
        self.bn1 = nn.BatchNorm2d(8)
        
        # Second conv block - expand
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)    # 16 channels
        self.bn2 = nn.BatchNorm2d(16)
        self.res1 = nn.Conv2d(8, 16, kernel_size=1)
        
        # Third conv block - expand further
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)   # 32 channels
        self.bn3 = nn.BatchNorm2d(32)
        
        # Fourth conv block - maintain width
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)   # 32 channels
        self.bn4 = nn.BatchNorm2d(32)
        self.res2 = nn.Conv2d(32, 32, kernel_size=1)
        
        # Global Average Pooling and FC
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)
        
        # Activation and regularization
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.02)   # 2% dropout early
        self.dropout2 = nn.Dropout(0.05)   # 5% dropout middle
        self.dropout3 = nn.Dropout(0.10)   # 10% dropout late
        
        # Initialize weights for better initial learning
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Modified initialization for better initial learning
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                # Slightly larger initial weights for FC
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        # Input normalization with MNIST stats
        x = (x - 0.1307) / 0.3081
        
        # First block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.avg_pool2d(x, 2, 2)  # Use avg pooling early
        x = self.dropout1(x)
        identity1 = x
        
        # Second block with residual
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = x + self.res1(identity1)
        x = F.avg_pool2d(x, 2, 2)  # Use avg pooling early
        x = self.dropout2(x)
        identity2 = x
        
        # Third block
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = F.max_pool2d(x, 2, 2)  # Switch to max pooling
        x = self.dropout2(x)
        identity3 = x
        
        # Fourth block with residual
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = x + self.res2(identity3)
        x = self.dropout3(x)
        
        # Global pooling and FC
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())