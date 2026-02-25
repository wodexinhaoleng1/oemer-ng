"""
oemer-ng: A modern, fast Optical Music Recognition (OMR) package.

This package provides state-of-the-art optical music recognition with support for:
- Modern deep learning architectures
- Model quantization for efficiency
- GGML inference for embedded devices
"""

__version__ = "0.1.0"

from .inference.pipeline import OMRPipeline
from .models.omr_model import OMRModel

__all__ = ["OMRPipeline", "OMRModel", "__version__"]
