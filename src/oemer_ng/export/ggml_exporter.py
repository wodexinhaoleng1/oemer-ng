"""
GGML export utilities.

This module provides utilities to export PyTorch models to GGML format
for efficient inference on embedded devices.
"""

import torch
import torch.nn as nn
import struct
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional


class GGMLExporter:
    """
    Export PyTorch models to GGML format.
    
    GGML is a tensor library for machine learning that enables
    efficient inference on CPUs, including embedded devices.
    """
    
    # GGML data types
    GGML_TYPE_F32 = 0
    GGML_TYPE_F16 = 1
    GGML_TYPE_Q4_0 = 2
    GGML_TYPE_Q4_1 = 3
    GGML_TYPE_Q8_0 = 8
    
    def __init__(self, model: nn.Module):
        """
        Initialize exporter.
        
        Args:
            model: PyTorch model to export
        """
        self.model = model
        self.model.eval()
    
    @staticmethod
    def write_header(f, magic: int = 0x67676d6c, version: int = 1):
        """
        Write GGML file header.
        
        Args:
            f: File handle
            magic: Magic number for file format
            version: Format version
        """
        f.write(struct.pack('I', magic))  # Magic number
        f.write(struct.pack('I', version))  # Version
    
    @staticmethod
    def write_tensor(
        f,
        name: str,
        tensor: torch.Tensor,
        data_type: int = GGML_TYPE_F32
    ):
        """
        Write a tensor to GGML file.
        
        Args:
            f: File handle
            name: Tensor name
            tensor: Tensor data
            data_type: GGML data type
        """
        # Handle quantized tensors
        if hasattr(tensor, 'dequantize'):
            # Dequantize quantized tensors
            tensor = tensor.dequantize()
        
        # Convert to numpy
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()
        elif not isinstance(tensor, np.ndarray):
            # Skip non-tensor types (like dtypes)
            print(f"Warning: Skipping non-tensor parameter: {name}")
            return
        
        # Get dimensions
        n_dims = len(tensor.shape)
        
        # Write tensor name
        name_bytes = name.encode('utf-8')
        f.write(struct.pack('I', len(name_bytes)))
        f.write(name_bytes)
        
        # Write number of dimensions
        f.write(struct.pack('I', n_dims))
        
        # Write dimension sizes
        for dim in tensor.shape:
            f.write(struct.pack('I', dim))
        
        # Write data type
        f.write(struct.pack('I', data_type))
        
        # Write tensor name
        name_bytes = name.encode('utf-8')
        f.write(struct.pack('I', len(name_bytes)))
        f.write(name_bytes)
        
        # Write number of dimensions
        f.write(struct.pack('I', n_dims))
        
        # Write dimension sizes
        for dim in tensor.shape:
            f.write(struct.pack('I', dim))
        
        # Write data type
        f.write(struct.pack('I', data_type))
        
        # Write tensor data
        if data_type == GGMLExporter.GGML_TYPE_F32:
            data = tensor.astype(np.float32).tobytes()
        elif data_type == GGMLExporter.GGML_TYPE_F16:
            data = tensor.astype(np.float16).tobytes()
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
        
        f.write(data)
    
    def export(
        self,
        output_path: str,
        use_fp16: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Export model to GGML format.
        
        Args:
            output_path: Path to output file
            use_fp16: Whether to use FP16 instead of FP32
            metadata: Optional metadata to include
        """
        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Determine data type
        data_type = self.GGML_TYPE_F16 if use_fp16 else self.GGML_TYPE_F32
        
        with open(output_path, 'wb') as f:
            # Write header
            self.write_header(f)
            
            # Write metadata if provided
            if metadata is not None:
                metadata_str = str(metadata)
                metadata_bytes = metadata_str.encode('utf-8')
                f.write(struct.pack('I', len(metadata_bytes)))
                f.write(metadata_bytes)
            else:
                f.write(struct.pack('I', 0))
            
            # Write model parameters
            state_dict = self.model.state_dict()
            
            # Write number of tensors
            f.write(struct.pack('I', len(state_dict)))
            
            # Write each tensor
            for name, tensor in state_dict.items():
                try:
                    self.write_tensor(f, name, tensor, data_type)
                except Exception as e:
                    print(f"Warning: Failed to write tensor {name}: {e}")
                    continue
        
        print(f"Model exported to GGML format: {output_path}")
        print(f"Data type: {'FP16' if use_fp16 else 'FP32'}")
        print(f"Number of tensors: {len(state_dict)}")
    
    @staticmethod
    def export_quantized(
        model: nn.Module,
        output_path: str,
        quantization_bits: int = 8
    ):
        """
        Export model with quantization.
        
        Args:
            model: Model to export
            output_path: Output file path
            quantization_bits: Number of bits for quantization (4 or 8)
        """
        if quantization_bits not in [4, 8]:
            raise ValueError("Only 4-bit and 8-bit quantization supported")
        
        # For now, use FP16 as a placeholder
        # Full quantization would require more sophisticated quantization schemes
        exporter = GGMLExporter(model)
        exporter.export(output_path, use_fp16=True)
        
        print(f"Note: Using FP16 export. Full {quantization_bits}-bit quantization "
              "requires additional quantization implementation.")


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: tuple = (1, 3, 512, 512),
    opset_version: int = 11
):
    """
    Export model to ONNX format.
    
    ONNX can be used as an intermediate format before GGML conversion
    or for deployment on various platforms.
    
    Args:
        model: PyTorch model
        output_path: Output file path
        input_shape: Input tensor shape
        opset_version: ONNX opset version
    """
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to ONNX: {output_path}")


def convert_model_for_deployment(
    model: nn.Module,
    output_dir: str,
    formats: list = ['ggml', 'onnx'],
    use_fp16: bool = True
):
    """
    Convert model to multiple deployment formats.
    
    Args:
        model: PyTorch model
        output_dir: Output directory
        formats: List of formats to export ('ggml', 'onnx')
        use_fp16: Whether to use FP16 precision
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    if 'ggml' in formats:
        ggml_path = output_path / 'model.ggml'
        exporter = GGMLExporter(model)
        exporter.export(str(ggml_path), use_fp16=use_fp16)
        results['ggml'] = str(ggml_path)
    
    if 'onnx' in formats:
        onnx_path = output_path / 'model.onnx'
        export_to_onnx(model, str(onnx_path))
        results['onnx'] = str(onnx_path)
    
    return results
