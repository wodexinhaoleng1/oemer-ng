"""
Unit tests for inference pipeline.
"""

import torch
import pytest
import tempfile
import os
from oemer_ng.inference.pipeline import OMRPipeline
from oemer_ng.models.omr_model import OMRModel


def test_pipeline_initialization():
    """Test pipeline initialization."""
    pipeline = OMRPipeline(num_classes=128)
    assert pipeline.model is not None
    assert pipeline.device is not None


def test_pipeline_save_load():
    """Test model save and load."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test_model.pth")

        # Create and save
        pipeline = OMRPipeline(num_classes=64)
        pipeline.save_model(model_path)

        # Load
        pipeline2 = OMRPipeline(model_path=model_path, num_classes=64)
        assert pipeline2.model is not None


def test_predict_tensor():
    """Test prediction with tensor input."""
    pipeline = OMRPipeline(num_classes=128)
    img = torch.randn(1, 3, 512, 512)
    result = pipeline.predict(img, enhance=False, return_probabilities=True)
    assert "prediction" in result
    assert "confidence" in result
    assert "probabilities" in result
    assert 0 <= result["prediction"] < 128


def test_predict_simple():
    """Test simple prediction."""
    pipeline = OMRPipeline(num_classes=128)
    img = torch.randn(1, 3, 512, 512)
    result = pipeline.predict(img, enhance=False, return_probabilities=False)
    assert isinstance(result, int)
    assert 0 <= result < 128


def test_quantization():
    """Test quantization."""
    pipeline = OMRPipeline(num_classes=128, use_quantized=True)
    assert pipeline.model is not None
    # Model should work after quantization
    img = torch.randn(1, 3, 512, 512)
    result = pipeline.predict(img, enhance=False)
    assert isinstance(result, int)


if __name__ == "__main__":
    pytest.main([__file__])
