#!/usr/bin/env python3
"""
Example: Basic OMR inference with a pretrained model.
"""

import torch
from oemer_ng import OMRPipeline, OMRModel


def main():
    # Example 1: Create a pipeline with default settings
    print("Creating OMR pipeline...")
    pipeline = OMRPipeline(num_classes=128)
    
    # Example 2: Create and save a model
    print("\nCreating and saving model...")
    model = OMRModel(num_classes=128)
    pipeline.model = model
    pipeline.save_model('demo_model.pth')
    print("Model saved to demo_model.pth")
    
    # Example 3: Load a model
    print("\nLoading model...")
    pipeline_with_model = OMRPipeline(
        model_path='demo_model.pth',
        num_classes=128
    )
    print("Model loaded successfully")
    
    # Example 4: Use quantization for faster inference
    print("\nCreating quantized pipeline...")
    quantized_pipeline = OMRPipeline(
        model_path='demo_model.pth',
        use_quantized=True,
        num_classes=128
    )
    print("Quantized pipeline created")
    
    # Example 5: Mock inference (since we don't have a real image)
    print("\nRunning mock inference...")
    dummy_image = torch.randn(1, 3, 512, 512)
    result = pipeline.predict(dummy_image, enhance=False, return_probabilities=True)
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
    
    print("\nExample completed successfully!")


if __name__ == '__main__':
    main()
