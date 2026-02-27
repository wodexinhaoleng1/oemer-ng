"""
Unit tests for OMR model.
"""

import torch
import pytest
from oemer_ng.models.omr_model import OMRModel, ConvBlock, ResidualBlock, AttentionModule


def test_conv_block():
    """Test ConvBlock forward pass."""
    block = ConvBlock(3, 64)
    x = torch.randn(2, 3, 64, 64)
    out = block(x)
    assert out.shape == (2, 64, 64, 64)


def test_residual_block():
    """Test ResidualBlock forward pass."""
    block = ResidualBlock(64)
    x = torch.randn(2, 64, 32, 32)
    out = block(x)
    assert out.shape == x.shape


def test_attention_module():
    """Test AttentionModule forward pass."""
    attn = AttentionModule(64)
    x = torch.randn(2, 64, 16, 16)
    out = attn(x)
    assert out.shape == x.shape


def test_omr_model_forward():
    """Test OMRModel forward pass."""
    model = OMRModel(num_classes=128)
    x = torch.randn(2, 3, 512, 512)
    out = model(x)
    assert out.shape == (2, 128)


def test_omr_model_feature_maps():
    """Test feature map extraction."""
    model = OMRModel(num_classes=128)
    x = torch.randn(1, 3, 512, 512)
    features = model.get_feature_maps(x)
    assert len(features) == 4
    assert all(isinstance(f, torch.Tensor) for f in features)


def test_omr_model_no_attention():
    """Test model without attention."""
    model = OMRModel(num_classes=128, use_attention=False)
    x = torch.randn(2, 3, 512, 512)
    out = model(x)
    assert out.shape == (2, 128)


def test_omr_model_custom_channels():
    """Test model with custom base channels."""
    model = OMRModel(num_classes=64, base_channels=32)
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    assert out.shape == (1, 64)


if __name__ == "__main__":
    pytest.main([__file__])
