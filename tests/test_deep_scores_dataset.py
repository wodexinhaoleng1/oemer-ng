"""Tests for the enhanced DeepScoresDataset (Model 1 / seg_net data loading)."""

import random
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from oemer_ng.training.dataset import DeepScoresDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset_dir(
    tmp_path: Path,
    n_images: int = 5,
    img_h: int = 400,
    img_w: int = 600,
    use_seg_suffix: bool = True,
) -> Path:
    """Create a minimal DeepScores-style dataset directory under *tmp_path*.

    Each image is a white RGB PNG.  Each segmentation file is a grayscale
    PNG where pixel values encode class indices (0–3).  Staff-line pixels
    (class 1) are placed in the upper two-thirds of the image so that the
    blank-bottom-skip logic is exercised.
    """
    images_dir = tmp_path / "images"
    seg_dir = tmp_path / "segmentation"
    images_dir.mkdir()
    seg_dir.mkdir()

    for i in range(n_images):
        stem = f"score_{i:03d}"

        # Image: solid white
        img = np.full((img_h, img_w, 3), 200, dtype=np.uint8)
        cv2.imwrite(str(images_dir / f"{stem}.png"), img)

        # Segmentation: background (0) everywhere; staff (1) in top 60 %
        seg = np.zeros((img_h, img_w), dtype=np.uint8)
        staff_bottom = int(img_h * 0.6)
        seg[:staff_bottom, :] = 1   # staff lines
        seg[staff_bottom:, :] = 2   # symbols (to add variety)

        suffix = "_seg.png" if use_seg_suffix else ".png"
        cv2.imwrite(str(seg_dir / f"{stem}{suffix}"), seg)

    return tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_find_pairs_with_seg_suffix(tmp_path):
    """_find_pairs should locate *_seg.png files correctly."""
    root = _make_dataset_dir(tmp_path, n_images=3, use_seg_suffix=True)
    ds = DeepScoresDataset(str(root), win_size=64, stride=64, augment=False)
    # All 3 images should be paired (80 % are train, 20 % are val; with 3
    # images n_val=1 so train gets 2, but all pairs were found).
    all_pairs = ds._find_pairs()
    assert len(all_pairs) == 3


def test_find_pairs_without_seg_suffix(tmp_path):
    """_find_pairs falls back to the plain *.png name when *_seg.png absent."""
    root = _make_dataset_dir(tmp_path, n_images=2, use_seg_suffix=False)
    ds = DeepScoresDataset(str(root), win_size=64, stride=64, augment=False)
    all_pairs = ds._find_pairs()
    assert len(all_pairs) == 2


def test_train_val_split_is_disjoint(tmp_path):
    """Train and val splits must contain distinct image files."""
    root = _make_dataset_dir(tmp_path, n_images=10)
    train_ds = DeepScoresDataset(
        str(root), win_size=64, stride=64, augment=False, split="train", seed=0
    )
    val_ds = DeepScoresDataset(
        str(root), win_size=64, stride=64, augment=False, split="val", seed=0
    )
    train_stems = {p.stem for p, _ in train_ds.pairs}
    val_stems = {p.stem for p, _ in val_ds.pairs}
    assert len(train_stems) > 0
    assert len(val_stems) > 0
    assert train_stems.isdisjoint(val_stems), "Train and val share images!"


def test_train_val_split_covers_all_files(tmp_path):
    """Together, train and val splits should account for all paired images."""
    root = _make_dataset_dir(tmp_path, n_images=10)
    train_ds = DeepScoresDataset(
        str(root), win_size=64, stride=64, augment=False, split="train", seed=0
    )
    val_ds = DeepScoresDataset(
        str(root), win_size=64, stride=64, augment=False, split="val", seed=0
    )
    combined = {p.stem for p, _ in train_ds.pairs} | {p.stem for p, _ in val_ds.pairs}
    all_stems = {p.stem for p in (tmp_path / "images").glob("*.png")}
    assert combined == all_stems


def test_patches_count_positive(tmp_path):
    """At least some patches should be generated."""
    root = _make_dataset_dir(tmp_path, n_images=2, img_h=300, img_w=400)
    ds = DeepScoresDataset(str(root), win_size=64, stride=64, augment=False)
    assert len(ds) > 0


def test_getitem_output_shapes(tmp_path):
    """__getitem__ must return tensors with the expected shapes."""
    win = 64
    root = _make_dataset_dir(tmp_path, n_images=3, img_h=300, img_w=400)
    ds = DeepScoresDataset(str(root), win_size=win, stride=win, augment=False)
    img, label = ds[0]
    assert img.shape == (3, win, win), f"Unexpected image shape: {img.shape}"
    assert label.shape == (DeepScoresDataset.CHANNEL_NUM, win, win), (
        f"Unexpected label shape: {label.shape}"
    )


def test_getitem_image_range(tmp_path):
    """Image tensor values should be normalised to [0, 1]."""
    root = _make_dataset_dir(tmp_path, n_images=3, img_h=300, img_w=400)
    ds = DeepScoresDataset(str(root), win_size=64, stride=64, augment=False)
    img, _ = ds[0]
    assert img.min() >= 0.0, "Image values below 0"
    assert img.max() <= 1.0, "Image values above 1"


def test_label_is_one_hot(tmp_path):
    """Each spatial location must belong to exactly one class."""
    root = _make_dataset_dir(tmp_path, n_images=3, img_h=300, img_w=400)
    ds = DeepScoresDataset(str(root), win_size=64, stride=64, augment=False)
    _, label = ds[0]
    channel_sum = label.sum(dim=0)  # (H, W)
    assert torch.allclose(
        channel_sum, torch.ones_like(channel_sum)
    ), "Label is not one-hot (channel sums ≠ 1)"


def test_blank_bottom_skip(tmp_path):
    """Patches entirely below the last staff-line row should be excluded."""
    img_h, img_w = 500, 300
    win = 64
    stride = 64
    images_dir = tmp_path / "images"
    seg_dir = tmp_path / "segmentation"
    images_dir.mkdir()
    seg_dir.mkdir()

    img = np.full((img_h, img_w, 3), 200, dtype=np.uint8)
    cv2.imwrite(str(images_dir / "score_000.png"), img)

    # Staff lines only in the top 100 rows; everything below is blank.
    seg = np.zeros((img_h, img_w), dtype=np.uint8)
    seg[:100, :] = 1  # staff in top 100 px
    cv2.imwrite(str(seg_dir / "score_000_seg.png"), seg)

    ds = DeepScoresDataset(
        str(tmp_path), win_size=win, stride=stride, augment=False, val_ratio=0.0
    )

    # With val_ratio=0 there is only 1 file in pairs, so n_val=0 → all in train.
    max_expected_top = 100 + win  # = 164, but capped at img_h=500
    for _, top, _ in ds.patches:
        assert top < min(max_expected_top, img_h), (
            f"Patch at top={top} is beyond blank-skip boundary"
        )


def test_augment_output_shape(tmp_path):
    """Augmented output must still have the correct spatial size."""
    win = 64
    root = _make_dataset_dir(tmp_path, n_images=2, img_h=300, img_w=400)
    ds = DeepScoresDataset(str(root), win_size=win, stride=win, augment=True)
    img, label = ds[0]
    assert img.shape == (3, win, win)
    assert label.shape == (DeepScoresDataset.CHANNEL_NUM, win, win)


def test_transform_applied(tmp_path):
    """The optional normalisation transform should be applied to the image."""
    from torchvision import transforms

    root = _make_dataset_dir(tmp_path, n_images=2, img_h=300, img_w=400)
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ds = DeepScoresDataset(
        str(root), win_size=64, stride=64, augment=False, transform=norm
    )
    img, _ = ds[0]
    # After normalisation some values will be outside [0, 1].
    assert img.min() < 0.0 or img.max() > 1.0, (
        "Transform was not applied (image still in [0, 1])"
    )


def test_channel_num_constant():
    """CHANNEL_NUM class constant must equal 4."""
    assert DeepScoresDataset.CHANNEL_NUM == 4


def test_default_stride_is_half_win_size(tmp_path):
    """When stride is None the default stride equals win_size // 2."""
    root = _make_dataset_dir(tmp_path, n_images=2, img_h=300, img_w=400)
    win = 64
    ds = DeepScoresDataset(str(root), win_size=win, stride=None, augment=False)
    assert ds.stride == win // 2


def test_val_ratio_zero_all_train(tmp_path):
    """val_ratio=0.0 should put all files in the train split."""
    root = _make_dataset_dir(tmp_path, n_images=5)
    ds = DeepScoresDataset(
        str(root), win_size=64, stride=64, augment=False, val_ratio=0.0, split="train"
    )
    all_pairs = ds._find_pairs()
    assert len(ds.pairs) == len(all_pairs)
