"""
OMR Model architecture.

Implements a two-stage OMR pipeline:
  - OMRModel (Model 1 / StaffSymbolModel): Full U-Net segmenting staff lines and
    symbol regions (3 classes) on CVC-MUSCIMA data.
  - NoteHeadModel (Model 2): Full U-Net subdividing symbol-region pixels into 8
    fine-grained categories on DeepScores V2 data.

Both models are built on shared UNetEncoder / UNetDecoder backbones.  A
classification head is retained on OMRModel for backward compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Class-label constants for Model 2
# ---------------------------------------------------------------------------

NOTE_HEAD_CLASSES: Dict[int, str] = {
    0: "background",
    1: "note_head",
    2: "stem",
    3: "beam",
    4: "rest",
    5: "flag",
    6: "dot",
    7: "accidental",
}


# ---------------------------------------------------------------------------
# Building-block modules
# ---------------------------------------------------------------------------


class ConvBlock(nn.Module):
    """Single convolutional block: Conv2d (bias=False) → BatchNorm2d → ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class DoubleConvBlock(nn.Module):
    """Two consecutive ConvBlocks — the standard U-Net building block."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """Residual block with skip connection placed after each encoder level."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual, inplace=True)
        return out


class AttentionModule(nn.Module):
    """Spatial attention gate: 1×1 conv → sigmoid, used to weight feature maps."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.sigmoid(self.conv(x))
        return x * attention


# ---------------------------------------------------------------------------
# U-Net encoder / decoder
# ---------------------------------------------------------------------------


class UNetEncoder(nn.Module):
    """4-level U-Net encoder.

    Each level applies a DoubleConvBlock followed by a ResidualBlock.  The
    output of each level is saved as a skip connection.  A final bottleneck
    DoubleConvBlock is applied after the deepest MaxPool.

    Args:
        in_channels: Number of input image channels.
        base_channels: Channel width at the first encoder level.  Subsequent
            levels double the channel count (×2, ×4, ×8).  The bottleneck
            uses base_channels × 16.
    """

    def __init__(self, in_channels: int, base_channels: int = 64):
        super().__init__()
        bc = base_channels

        # Level 1  (no downsampling — first conv)
        self.enc1 = DoubleConvBlock(in_channels, bc)
        self.res1 = ResidualBlock(bc)

        # Level 2
        self.down2 = nn.MaxPool2d(2)
        self.enc2 = DoubleConvBlock(bc, bc * 2)
        self.res2 = ResidualBlock(bc * 2)

        # Level 3
        self.down3 = nn.MaxPool2d(2)
        self.enc3 = DoubleConvBlock(bc * 2, bc * 4)
        self.res3 = ResidualBlock(bc * 4)

        # Level 4
        self.down4 = nn.MaxPool2d(2)
        self.enc4 = DoubleConvBlock(bc * 4, bc * 8)
        self.res4 = ResidualBlock(bc * 8)

        # Bottleneck
        self.down_bn = nn.MaxPool2d(2)
        self.bottleneck = DoubleConvBlock(bc * 8, bc * 16)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Return (skip_connections, bottleneck_features).

        skip_connections is a list [s1, s2, s3, s4] ordered from shallowest
        to deepest.
        """
        s1 = self.res1(self.enc1(x))
        s2 = self.res2(self.enc2(self.down2(s1)))
        s3 = self.res3(self.enc3(self.down3(s2)))
        s4 = self.res4(self.enc4(self.down4(s3)))
        bn = self.bottleneck(self.down_bn(s4))
        return [s1, s2, s3, s4], bn


class UNetDecoder(nn.Module):
    """4-level U-Net decoder.

    Each level upsamples via ConvTranspose2d, concatenates the corresponding
    skip connection, and refines with a DoubleConvBlock.  Spatial size
    mismatches caused by odd input dimensions are corrected with
    ``F.interpolate`` before concatenation.

    Optionally applies an AttentionModule to each skip connection before
    concatenation.

    Args:
        base_channels: Must match the value used in the paired UNetEncoder.
        num_classes: Number of output segmentation classes.
        use_attention: Whether to apply spatial attention to skip connections.
    """

    def __init__(
        self,
        base_channels: int = 64,
        num_classes: int = 3,
        use_attention: bool = True,
    ):
        super().__init__()
        bc = base_channels

        # Level 4: bottleneck (bc*16) → up → concat with skip4 (bc*8) → bc*8
        self.up4 = nn.ConvTranspose2d(bc * 16, bc * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConvBlock(bc * 16, bc * 8)

        # Level 3
        self.up3 = nn.ConvTranspose2d(bc * 8, bc * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConvBlock(bc * 8, bc * 4)

        # Level 2
        self.up2 = nn.ConvTranspose2d(bc * 4, bc * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConvBlock(bc * 4, bc * 2)

        # Level 1
        self.up1 = nn.ConvTranspose2d(bc * 2, bc, kernel_size=2, stride=2)
        self.dec1 = DoubleConvBlock(bc * 2, bc)

        # Final 1×1 projection to class scores
        self.out_conv = nn.Conv2d(bc, num_classes, kernel_size=1)

        # Optional attention gates on skip connections
        self.use_attention = use_attention
        if use_attention:
            self.attn4 = AttentionModule(bc * 8)
            self.attn3 = AttentionModule(bc * 4)
            self.attn2 = AttentionModule(bc * 2)
            self.attn1 = AttentionModule(bc)

    def _upsample_to(
        self, x: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Align *x* to the spatial size of *target* when they differ."""
        if x.shape[2:] != target.shape[2:]:
            x = F.interpolate(
                x, size=target.shape[2:], mode="bilinear", align_corners=False
            )
        return x

    def forward(
        self,
        skips: List[torch.Tensor],
        bottleneck: torch.Tensor,
    ) -> torch.Tensor:
        s1, s2, s3, s4 = skips

        x = self._upsample_to(self.up4(bottleneck), s4)
        if self.use_attention:
            s4 = self.attn4(s4)
        x = self.dec4(torch.cat([x, s4], dim=1))

        x = self._upsample_to(self.up3(x), s3)
        if self.use_attention:
            s3 = self.attn3(s3)
        x = self.dec3(torch.cat([x, s3], dim=1))

        x = self._upsample_to(self.up2(x), s2)
        if self.use_attention:
            s2 = self.attn2(s2)
        x = self.dec2(torch.cat([x, s2], dim=1))

        x = self._upsample_to(self.up1(x), s1)
        if self.use_attention:
            s1 = self.attn1(s1)
        x = self.dec1(torch.cat([x, s1], dim=1))

        return self.out_conv(x)


# ---------------------------------------------------------------------------
# Model 1 — StaffSymbolModel / OMRModel
# ---------------------------------------------------------------------------


class OMRModel(nn.Module):
    """U-Net based OMR model for staff-line and symbol segmentation (Model 1).

    Trained on the CVC-MUSCIMA dataset.  Outputs 3 classes by default:
        0 = Background, 1 = Staff lines, 2 = Symbol regions.

    A ``mode="classification"`` option is retained for backward compatibility;
    in that mode a global-average-pool + Linear head is used instead of the
    decoder.

    Args:
        num_classes: Number of output classes (default 128 for backward compat;
            use 3 for the standard staff/symbol segmentation task).
        n_channels: Input image channels — 1 for grayscale, 3 for RGB.
        base_channels: Channel width at the first encoder level.
        use_attention: Whether to apply spatial attention in the decoder.
        mode: ``"segmentation"`` (default) or ``"classification"``.
    """

    def __init__(
        self,
        num_classes: int = 128,
        n_channels: int = 3,
        base_channels: int = 64,
        use_attention: bool = True,
        mode: str = "classification",
    ):
        if mode not in ("classification", "segmentation"):
            raise ValueError(
                f"Invalid mode '{mode}'. Expected 'classification' or 'segmentation'."
            )
        super().__init__()

        self.num_classes = num_classes
        self.n_channels = n_channels
        self.use_attention = use_attention
        self.mode = mode

        self.encoder = UNetEncoder(n_channels, base_channels)
        self.decoder = UNetDecoder(base_channels, num_classes, use_attention)

        # Classification head (used when mode="classification")
        bottleneck_channels = base_channels * 16
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(bottleneck_channels, num_classes)

    def get_feature_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return the four encoder skip-connection feature maps.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.

        Returns:
            List of four tensors ``[s1, s2, s3, s4]`` from shallowest to
            deepest encoder level.
        """
        skips, _ = self.encoder(x)
        return skips

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.

        Returns:
            ``(B, num_classes)`` for ``mode="classification"``;
            ``(B, num_classes, H, W)`` for ``mode="segmentation"``.
        """
        skips, bottleneck = self.encoder(x)

        if self.mode == "classification":
            pooled = self.global_pool(bottleneck)
            flat = pooled.view(pooled.size(0), -1)
            return self.classifier(flat)
        else:
            return self.decoder(skips, bottleneck)


# ---------------------------------------------------------------------------
# Model 2 — NoteHeadModel
# ---------------------------------------------------------------------------


class NoteHeadModel(nn.Module):
    """U-Net based model for fine-grained note-head classification (Model 2).

    Trained on the DeepScores V2 dataset.  Outputs 8 classes (see
    :data:`NOTE_HEAD_CLASSES`).

    When ``use_stage1_mask=True`` the model accepts an additional input channel
    containing the *symbol probability map* produced by Model 1.  The extra
    channel is concatenated with the image inside :meth:`forward`.

    Args:
        num_classes: Number of output classes (default 8).
        n_channels: Number of image channels (1 for grayscale, 3 for RGB).
        base_channels: Channel width at the first encoder level.
        use_stage1_mask: If ``True``, the model expects ``n_channels + 1``
            actual input channels (the extra channel is the stage-1 symbol
            probability map).
    """

    def __init__(
        self,
        num_classes: int = 8,
        n_channels: int = 1,
        base_channels: int = 64,
        use_stage1_mask: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.n_channels = n_channels
        self.use_stage1_mask = use_stage1_mask

        actual_in_channels = n_channels + 1 if use_stage1_mask else n_channels
        self.encoder = UNetEncoder(actual_in_channels, base_channels)
        self.decoder = UNetDecoder(base_channels, num_classes, use_attention=True)

    def forward(
        self,
        image: torch.Tensor,
        stage1_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            image: Input image tensor ``(B, n_channels, H, W)``.
            stage1_mask: Symbol probability map from Model 1,
                ``(B, 1, H, W)`` in ``[0, 1]``.  When
                ``use_stage1_mask=True`` and this argument is ``None``, an
                all-zero placeholder mask is used automatically.

        Returns:
            Segmentation logits ``(B, num_classes, H, W)``.
        """
        if self.use_stage1_mask:
            if stage1_mask is None:
                stage1_mask = torch.zeros(
                    image.size(0), 1, image.size(2), image.size(3),
                    device=image.device,
                    dtype=image.dtype,
                )
            x = torch.cat([image, stage1_mask], dim=1)
        else:
            x = image

        skips, bottleneck = self.encoder(x)
        return self.decoder(skips, bottleneck)
