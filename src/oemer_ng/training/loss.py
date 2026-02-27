"""
Custom loss functions for OMR training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss for segmentation.
    Combination of Focal Loss and Tversky Loss.
    """

    def __init__(self, fw=0.7, alpha=0.7, smooth=1.0, gamma=0.75, tp_weight=0.4):
        super(FocalTverskyLoss, self).__init__()
        self.fw = fw
        self.alpha = alpha
        self.smooth = smooth
        self.gamma = gamma
        self.tp_weight = tp_weight

    def forward(self, inputs, targets):
        # inputs: (B, C, H, W) logits
        # targets: (B, C, H, W) 0 or 1

        # Calculate Probabilities
        probs = torch.sigmoid(inputs)

        # Flatten
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        probs_flat = probs.view(-1)

        # Tversky Loss
        tp = (probs_flat * targets_flat).sum() * self.tp_weight
        fn = (targets_flat * (1 - probs_flat)).sum()
        fp = ((1 - targets_flat) * probs_flat).sum()

        tversky_index = (tp + self.smooth) / (
            tp + self.alpha * fn + (1 - self.alpha) * fp + self.smooth
        )
        tversky_loss = 1 - tversky_index
        t_loss = torch.pow(tversky_loss, self.gamma)

        # Focal Loss
        # sigmoid_focal_loss takes logits
        f_loss = sigmoid_focal_loss(
            inputs_flat, targets_flat, alpha=0.25, gamma=2.0, reduction="mean"
        )

        return self.fw * f_loss + (1 - self.fw) * t_loss
