"""
Unit tests for OMR loss functions.
"""

import torch
import pytest
import torch.nn as nn
from oemer_ng.training.loss import FocalTverskyLoss


def test_focal_tversky_loss_initialization():
    """Test initialization with default and custom parameters."""
    # Default parameters
    loss_fn = FocalTverskyLoss()
    assert loss_fn.fw == 0.7
    assert loss_fn.alpha == 0.7
    assert loss_fn.smooth == 1.0
    assert loss_fn.gamma == 0.75
    assert loss_fn.tp_weight == 0.4

    # Custom parameters
    loss_fn = FocalTverskyLoss(fw=0.5, alpha=0.5, smooth=2.0, gamma=1.0, tp_weight=0.8)
    assert loss_fn.fw == 0.5
    assert loss_fn.alpha == 0.5
    assert loss_fn.smooth == 2.0
    assert loss_fn.gamma == 1.0
    assert loss_fn.tp_weight == 0.8


def test_focal_tversky_loss_forward_shape():
    """Test the output shape of the loss function."""
    loss_fn = FocalTverskyLoss()
    # Batch size 2, 3 channels, 32x32 image
    inputs = torch.randn(2, 3, 32, 32)
    targets = torch.randint(0, 2, (2, 3, 32, 32)).float()

    loss = loss_fn(inputs, targets)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Should be a scalar


def test_focal_tversky_loss_values():
    """Test loss values for known inputs."""
    loss_fn = FocalTverskyLoss()

    # Case 1: Perfect match
    # For target 1, use high positive logit. For target 0, use high negative logit.
    targets = torch.tensor([[[[1.0, 0.0]]]])
    inputs = torch.tensor([[[[10.0, -10.0]]]])
    loss_perfect = loss_fn(inputs, targets)

    # Case 2: Complete mismatch
    inputs_wrong = torch.tensor([[[[-10.0, 10.0]]]])
    loss_wrong = loss_fn(inputs_wrong, targets)

    # Loss should be lower for correct predictions
    assert loss_perfect < loss_wrong
    # Loss should be non-negative
    assert loss_perfect >= 0


def test_focal_tversky_loss_backward():
    """Test if gradients can be computed."""
    loss_fn = FocalTverskyLoss()
    inputs = torch.randn(2, 2, 8, 8, requires_grad=True)
    targets = torch.randint(0, 2, (2, 2, 8, 8)).float()

    loss = loss_fn(inputs, targets)
    loss.backward()

    assert inputs.grad is not None
    assert not torch.isnan(inputs.grad).any()


def test_focal_tversky_loss_consistency():
    """Test that combined loss correctly weights Focal and Tversky components."""
    inputs = torch.randn(1, 1, 4, 4)
    targets = torch.randint(0, 2, (1, 1, 4, 4)).float()

    # Test with different weights
    fw = 0.4
    loss_focal = FocalTverskyLoss(fw=1.0)(inputs, targets)
    loss_tversky = FocalTverskyLoss(fw=0.0)(inputs, targets)
    loss_combined = FocalTverskyLoss(fw=fw)(inputs, targets)

    expected = fw * loss_focal + (1 - fw) * loss_tversky
    assert torch.allclose(loss_combined, expected)


def test_focal_tversky_loss_edge_cases():
    """Test edge cases like all zeros or all ones."""
    loss_fn = FocalTverskyLoss()

    # All zeros inputs
    inputs = torch.zeros(1, 1, 4, 4)
    targets = torch.zeros(1, 1, 4, 4)
    loss_zeros = loss_fn(inputs, targets)
    assert not torch.isnan(loss_zeros)

    # All ones targets
    targets_ones = torch.ones(1, 1, 4, 4)
    loss_ones = loss_fn(inputs, targets_ones)
    assert not torch.isnan(loss_ones)


if __name__ == "__main__":
    pytest.main([__file__])
