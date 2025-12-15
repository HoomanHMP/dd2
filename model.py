import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn. Module):
    """Efficient residual block with 1x1 and 3x3 convolutions"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # Bottleneck design to reduce parameters
        mid_channels = out_channels // 4
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self. shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self. shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        out += identity
        out = F. relu(out)
        
        return out


class SudokuCNN(nn.Module):
    """
    Multi-task CNN for SudokuMNIST with three heads:
    1. Missing digit classification (10 classes)
    2. Row/column sortedness classification (6 positions × 3 classes)
    3. Row/column sum regression (6 values)
    
    Target:  < 20,000 parameters
    """
    def __init__(self):
        super(SudokuCNN, self).__init__()
        
        # Initial convolution - input:  1×84×84
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Residual blocks with downsampling
        self.layer1 = ResidualBlock(16, 32, stride=2)  # 42×42
        self.layer2 = ResidualBlock(32, 64, stride=2)  # 21×21
        self.layer3 = ResidualBlock(64, 64, stride=2)  # 10×10 (actually 11×11)
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Spatial pooling for sum/sortedness tasks (preserve some spatial info)
        self.spatial_pool = nn.AdaptiveAvgPool2d((3, 3))
        
        # Head 1: Missing digit classification
        self.fc_missing = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 10)
        )
        
        # Head 2: Sortedness classification (6 positions, 3 classes each)
        # Use spatial features - flatten 3×3 grid
        self.fc_sorted = nn.Sequential(
            nn.Linear(64 * 9, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 18)  # 6 positions × 3 classes = 18, will reshape later
        )
        
        # Head 3: Sum regression (6 values)
        self.fc_sum = nn.Sequential(
            nn.Linear(64 * 9, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 6)
        )
    
    def forward(self, x):
        # Shared backbone
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Global pooling for missing digit task
        global_features = self.global_pool(x)  # (batch, 64, 1, 1)
        global_features = global_features.view(global_features.size(0), -1)  # (batch, 64)
        
        # Spatial pooling for sorted and sum tasks
        spatial_features = self.spatial_pool(x)  # (batch, 64, 3, 3)
        spatial_features = spatial_features. view(spatial_features.size(0), -1)  # (batch, 64*9)
        
        # Head 1: Missing digit
        missing_digit = self.fc_missing(global_features)  # (batch, 10)
        
        # Head 2: Sortedness (reshape to (batch, 6, 3))
        sorted_logits = self.fc_sorted(spatial_features)  # (batch, 18)
        sorted_labels = sorted_logits.view(-1, 6, 3)  # (batch, 6, 3)
        
        # Head 3: Sum regression
        sum_labels = self.fc_sum(spatial_features)  # (batch, 6)
        sum_labels = torch.sigmoid(sum_labels)  # Normalize to [0, 1]
        
        return {
            'missing_digit':  missing_digit,
            'sorted_labels': sorted_labels,
            'sum_labels': sum_labels
        }
    
    def count_parameters(self):
        """Count total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__": 
    # Test the model
    model = SudokuCNN()
    print(f"Total parameters: {model.count_parameters()}")
    
    # Test forward pass
    dummy_input = torch.randn(4, 1, 84, 84)
    output = model(dummy_input)
    
    print(f"Missing digit output shape: {output['missing_digit'].shape}")  # Should be (4, 10)
    print(f"Sorted labels output shape: {output['sorted_labels'].shape}")  # Should be (4, 6, 3)
    print(f"Sum labels output shape: {output['sum_labels'].shape}")  # Should be (4, 6)
