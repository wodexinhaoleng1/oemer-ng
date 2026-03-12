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

    def __init__(self, fw=0.7, alpha=0.7, smooth=1.0, gamma=0.75, tp_weight=1.0):
        super(FocalTverskyLoss, self).__init__()
        self.fw = fw
        self.alpha = alpha
        self.smooth = smooth
        self.gamma = gamma
        self.tp_weight = tp_weight

    def forward(self, inputs, targets):
        # inputs: (B, C, H, W) logits from model
        # targets: (B, C, H, W) 0 or 1 masks

        # Ensure inputs and targets have same spatial dimensions
        assert (
            inputs.shape == targets.shape
        ), f"Shape mismatch: inputs {inputs.shape} vs targets {targets.shape}"

        # Calculate Probabilities
        probs = torch.sigmoid(inputs)

        # Flatten spatial dimensions: (B, C, H, W) -> (B, C, H*W) -> (B, C*H*W)
        # Cast to float32 to prevent overflow under AMP (float16 has limited range)
        inputs_flat = inputs.float().view(inputs.size(0), -1)
        targets_flat = targets.float().view(targets.size(0), -1)
        probs_flat = probs.float().view(probs.size(0), -1)

        # Tversky Loss per batch
        tversky_losses = []
        for b in range(inputs.size(0)):
            inputs_b = inputs_flat[b]
            targets_b = targets_flat[b]
            probs_b = probs_flat[b]

            tp = (probs_b * targets_b).sum() * self.tp_weight
            fn = (targets_b * (1 - probs_b)).sum()
            fp = ((1 - targets_b) * probs_b).sum()

            # Numerical stability: smooth term prevents division by zero
            tversky_index = (tp + self.smooth) / (
                tp + self.alpha * fn + (1 - self.alpha) * fp + self.smooth
            )
            tversky_loss = 1 - tversky_index
            t_loss = torch.pow(tversky_loss, self.gamma)
            tversky_losses.append(t_loss)

        tversky_loss = torch.stack(tversky_losses).mean()

        # Focal Loss
        # sigmoid_focal_loss expects logits of shape (N, C), where each row is a sample.
        # Treat each pixel as a separate sample with C channels/classes.
        b, c, h, w = inputs.shape
        inputs_focal = inputs.permute(0, 2, 3, 1).reshape(-1, c)
        targets_focal = targets.permute(0, 2, 3, 1).reshape(-1, c)
        f_loss = sigmoid_focal_loss(
            inputs_focal, targets_focal, alpha=0.25, gamma=2.0, reduction="mean"
        )

        return self.fw * f_loss + (1 - self.fw) * tversky_loss
