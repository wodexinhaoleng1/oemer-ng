#!/usr/bin/env python3
"""
Example: Export model to GGML format for embedded deployment.
"""

import torch
from oemer_ng.models.omr_model import OMRModel
from oemer_ng.export.ggml_exporter import GGMLExporter, export_to_onnx, convert_model_for_deployment
from oemer_ng.quantization.quantizer import ModelQuantizer


def main():
    print("Creating model...")
    model = OMRModel(num_classes=128)

    # Example 1: Export to GGML format
    print("\n1. Exporting to GGML (FP32)...")
    exporter = GGMLExporter(model)
    exporter.export("model_fp32.ggml", use_fp16=False)

    # Example 2: Export to GGML with FP16
    print("\n2. Exporting to GGML (FP16)...")
    exporter.export("model_fp16.ggml", use_fp16=True)

    # Example 3: Export to ONNX
    print("\n3. Exporting to ONNX...")
    try:
        export_to_onnx(model, "model.onnx", input_shape=(1, 3, 512, 512))
    except ImportError as e:
        print(f"ONNX export requires additional dependencies: {e}")
        print("Install with: pip install onnx onnxscript")

    # Example 4: Quantize model first, then export
    print("\n4. Quantizing model...")
    quantizer = ModelQuantizer()
    quantized_model = quantizer.dynamic_quantize(model)

    # Compare sizes
    size_comparison = quantizer.compare_model_sizes(model, quantized_model)
    print("\nModel size comparison:")
    print(f"  Original: {size_comparison['original_size_mb']:.2f} MB")
    print(f"  Quantized: {size_comparison['quantized_size_mb']:.2f} MB")
    print(f"  Compression ratio: {size_comparison['compression_ratio']:.2f}x")
    print(f"  Size reduction: {size_comparison['size_reduction_percent']:.1f}%")

    # Example 5: Export quantized model to GGML
    print("\n5. Exporting quantized model to GGML...")
    quantized_exporter = GGMLExporter(quantized_model)
    quantized_exporter.export("model_quantized.ggml", use_fp16=True)

    # Example 6: Convert to multiple formats at once
    print("\n6. Converting to multiple formats...")
    try:
        results = convert_model_for_deployment(
            model, output_dir="./deployment", formats=["ggml", "onnx"], use_fp16=True
        )
        print("Exported files:")
        for format_name, path in results.items():
            print(f"  {format_name}: {path}")
    except ImportError:
        print("Skipping ONNX export (optional dependency not installed)")
        results = convert_model_for_deployment(
            model, output_dir="./deployment", formats=["ggml"], use_fp16=True
        )
        print("Exported files:")
        for format_name, path in results.items():
            print(f"  {format_name}: {path}")

    print("\nExport completed successfully!")
    print("\nThese exported models can be used for:")
    print("  - GGML: Efficient inference on CPUs and embedded devices")
    print("  - ONNX: Cross-platform deployment (ONNX Runtime, TensorRT, etc.)")


if __name__ == "__main__":
    main()
