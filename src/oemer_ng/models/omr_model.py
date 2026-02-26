"""
OMR Model architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class ConvBlock(nn.Module):
    """Convolutional block with batch norm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class AttentionModule(nn.Module):
    """Simple spatial attention module."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.sigmoid(self.conv(x))
        return x * attention


class OMRModel(nn.Module):
    """
    OMR Model for optical music recognition.

    Supports both classification and segmentation tasks.
    """

    def __init__(
        self,
        num_classes: int = 128,
        n_channels: int = 3,
        base_channels: int = 64,
        use_attention: bool = True
    ):
        """
        Initialize OMR model.

        Args:
            num_classes: Number of output classes
            n_channels: Number of input channels (1 for grayscale, 3 for RGB)
            base_channels: Base number of channels for the network
            use_attention: Whether to use attention modules
        """
        super().__init__()
        self.num_classes = num_classes
        self.n_channels = n_channels
        self.use_attention = use_attention

        # Encoder
        self.conv1 = ConvBlock(n_channels, base_channels)
        self.conv2 = ConvBlock(base_channels, base_channels * 2, stride=2)
        self.res1 = ResidualBlock(base_channels * 2)

        self.conv3 = ConvBlock(base_channels * 2, base_channels * 4, stride=2)
        self.res2 = ResidualBlock(base_channels * 4)

        self.conv4 = ConvBlock(base_channels * 4, base_channels * 8, stride=2)
        self.res3 = ResidualBlock(base_channels * 8)

        # Attention
        if use_attention:
            self.attn1 = AttentionModule(base_channels * 2)
            self.attn2 = AttentionModule(base_channels * 4)
            self.attn3 = AttentionModule(base_channels * 8)

        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(base_channels * 8, num_classes)

    def get_feature_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Get intermediate feature maps.

        Args:
            x: Input tensor

        Returns:
            List of feature map tensors
        """
        features = []

        # Stage 1
        x = self.conv1(x)
        features.append(x)

        # Stage 2
        x = self.conv2(x)
        x = self.res1(x)
        if self.use_attention:
            x = self.attn1(x)
        features.append(x)

        # Stage 3
        x = self.conv3(x)
        x = self.res2(x)
        if self.use_attention:
            x = self.attn2(x)
        features.append(x)

        # Stage 4
        x = self.conv4(x)
        x = self.res3(x)
        if self.use_attention:
            x = self.attn3(x)
        features.append(x)

        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, num_classes)
        """
        # Get features
        features = self.get_feature_maps(x)

        # Global pooling
        x = features[-1]
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Classifier
        x = self.classifier(x)

        return x
