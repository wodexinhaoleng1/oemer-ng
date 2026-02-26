"""
Model quantization utilities for efficient inference.
"""

import torch
import torch.nn as nn
from typing import Optional, Set, Type
import copy


class ModelQuantizer:
    """
    Utility class for quantizing PyTorch models.

    Supports various quantization methods including dynamic, static, and QAT.
    """

    @staticmethod
    def dynamic_quantize(
        model: nn.Module,
        qconfig_spec: Optional[Set[Type[nn.Module]]] = None,
        dtype: torch.dtype = torch.qint8,
        inplace: bool = False,
    ) -> nn.Module:
        """
        Apply dynamic quantization to model.

        Dynamic quantization quantizes weights ahead of time but quantizes
        activations dynamically at runtime.

        Args:
            model: Model to quantize
            qconfig_spec: Set of layer types to quantize
            dtype: Quantization data type
            inplace: Whether to modify model inplace

        Returns:
            Quantized model
        """
        if not inplace:
            model = copy.deepcopy(model)

        if qconfig_spec is None:
            qconfig_spec = {nn.Linear, nn.Conv2d}

        quantized_model = torch.quantization.quantize_dynamic(model, qconfig_spec, dtype=dtype)

        return quantized_model

    @staticmethod
    def prepare_static_quantization(
        model: nn.Module, qconfig: str = "fbgemm", inplace: bool = False
    ) -> nn.Module:
        """
        Prepare model for static quantization.

        Static quantization requires calibration with representative data.

        Args:
            model: Model to prepare
            qconfig: Quantization configuration ('fbgemm' for server, 'qnnpack' for mobile)
            inplace: Whether to modify model inplace

        Returns:
            Prepared model ready for calibration
        """
        if not inplace:
            model = copy.deepcopy(model)

        model.eval()

        # Set quantization config
        if qconfig == "fbgemm":
            model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
        elif qconfig == "qnnpack":
            model.qconfig = torch.quantization.get_default_qconfig("qnnpack")
        else:
            raise ValueError(f"Unknown qconfig: {qconfig}")

        # Prepare model
        prepared_model = torch.quantization.prepare(model, inplace=True)

        return prepared_model

    @staticmethod
    def convert_static_quantization(model: nn.Module, inplace: bool = True) -> nn.Module:
        """
        Convert prepared model to quantized version.

        Should be called after calibration with representative data.

        Args:
            model: Prepared and calibrated model
            inplace: Whether to modify model inplace

        Returns:
            Quantized model
        """
        quantized_model = torch.quantization.convert(model, inplace=inplace)
        return quantized_model

    @staticmethod
    def calibrate_model(
        model: nn.Module,
        calibration_data: torch.utils.data.DataLoader,
        num_batches: Optional[int] = None,
    ):
        """
        Calibrate model for static quantization.

        Args:
            model: Prepared model
            calibration_data: DataLoader with calibration data
            num_batches: Number of batches to use (None for all)
        """
        model.eval()

        with torch.no_grad():
            for i, (images, _) in enumerate(calibration_data):
                if num_batches is not None and i >= num_batches:
                    break
                model(images)

        print(f"Calibration complete with {i + 1} batches")

    @staticmethod
    def compare_model_sizes(original_model: nn.Module, quantized_model: nn.Module) -> dict:
        """
        Compare sizes of original and quantized models.

        Args:
            original_model: Original model
            quantized_model: Quantized model

        Returns:
            Dictionary with size comparison
        """
        import tempfile
        import os

        # Save models to temporary files
        with tempfile.NamedTemporaryFile(delete=False) as f:
            torch.save(original_model.state_dict(), f.name)
            original_size = os.path.getsize(f.name)
            os.unlink(f.name)

        with tempfile.NamedTemporaryFile(delete=False) as f:
            torch.save(quantized_model.state_dict(), f.name)
            quantized_size = os.path.getsize(f.name)
            os.unlink(f.name)

        compression_ratio = original_size / quantized_size
        size_reduction = (1 - quantized_size / original_size) * 100

        return {
            "original_size_mb": original_size / (1024 * 1024),
            "quantized_size_mb": quantized_size / (1024 * 1024),
            "compression_ratio": compression_ratio,
            "size_reduction_percent": size_reduction,
        }


def quantize_model_for_inference(
    model: nn.Module,
    quantization_type: str = "dynamic",
    calibration_data: Optional[torch.utils.data.DataLoader] = None,
    qconfig: str = "fbgemm",
) -> nn.Module:
    """
    Convenience function to quantize a model.

    Args:
        model: Model to quantize
        quantization_type: Type of quantization ('dynamic' or 'static')
        calibration_data: Calibration data for static quantization
        qconfig: Quantization configuration

    Returns:
        Quantized model
    """
    quantizer = ModelQuantizer()

    if quantization_type == "dynamic":
        return quantizer.dynamic_quantize(model)

    elif quantization_type == "static":
        if calibration_data is None:
            raise ValueError("Calibration data required for static quantization")

        # Prepare
        prepared_model = quantizer.prepare_static_quantization(model, qconfig=qconfig)

        # Calibrate
        quantizer.calibrate_model(prepared_model, calibration_data)

        # Convert
        quantized_model = quantizer.convert_static_quantization(prepared_model)

        return quantized_model

    else:
        raise ValueError(f"Unknown quantization type: {quantization_type}")
