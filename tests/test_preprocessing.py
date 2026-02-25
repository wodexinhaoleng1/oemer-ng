"""
Unit tests for preprocessing utilities.
"""

import numpy as np
import torch
import pytest
from PIL import Image
from oemer_ng.utils.preprocessing import ImagePreprocessor, enhance_sheet_music


def test_preprocessor_initialization():
    """Test preprocessor initialization."""
    prep = ImagePreprocessor()
    assert prep.target_size == (512, 512)
    assert prep.normalize is True


def test_resize_image():
    """Test image resizing."""
    prep = ImagePreprocessor(target_size=(256, 256))
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    resized = prep.resize_image(img, keep_aspect=True)
    assert resized.shape[:2] == (256, 256)


def test_normalize_image():
    """Test image normalization."""
    prep = ImagePreprocessor()
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    normalized = prep.normalize_image(img)
    assert normalized.dtype == np.float32
    # Check if normalization was applied (values should be around 0)
    assert normalized.mean() < 10  # Should be roughly centered


def test_preprocess_numpy():
    """Test preprocessing numpy array."""
    prep = ImagePreprocessor(target_size=(256, 256))
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = prep.preprocess(img, return_tensor=True)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3, 256, 256)


def test_preprocess_pil():
    """Test preprocessing PIL Image."""
    prep = ImagePreprocessor(target_size=(256, 256))
    img = Image.new('RGB', (100, 100), color='red')
    result = prep.preprocess(img, return_tensor=True)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3, 256, 256)


def test_batch_preprocess():
    """Test batch preprocessing."""
    prep = ImagePreprocessor(target_size=(128, 128))
    images = [
        np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        for _ in range(4)
    ]
    batch = prep.batch_preprocess(images, return_tensor=True)
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (4, 3, 128, 128)


def test_enhance_sheet_music():
    """Test sheet music enhancement."""
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    enhanced = enhance_sheet_music(img)
    assert enhanced.shape == img.shape
    assert enhanced.dtype == img.dtype


def test_enhance_grayscale():
    """Test enhancement on grayscale image."""
    img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    enhanced = enhance_sheet_music(img)
    assert len(enhanced.shape) == 2  # Should remain grayscale


if __name__ == '__main__':
    pytest.main([__file__])
