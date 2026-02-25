#!/usr/bin/env python3
"""
Quick start script for oemer-ng.

This script demonstrates the basic workflow:
1. Create a model
2. Train on synthetic data (demo)
3. Export to GGML format
4. Run inference
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from oemer_ng.models.omr_model import OMRModel
from oemer_ng.inference.pipeline import OMRPipeline
from oemer_ng.training.trainer import Trainer
from oemer_ng.training.dataset import create_demo_dataloaders
from oemer_ng.export.ggml_exporter import GGMLExporter


def main():
    print("=" * 70)
    print("oemer-ng Quick Start Demo")
    print("=" * 70)
    
    # Configuration
    num_classes = 128
    num_epochs = 1  # Very small for quick demo
    batch_size = 32
    
    # Step 1: Create model
    print("\n[1/5] Creating OMR model...")
    model = OMRModel(num_classes=num_classes, base_channels=16)  # Very small for demo
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {num_params:,} parameters")
    
    # Step 2: Create synthetic data
    print("\n[2/5] Creating synthetic training data...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_loader, val_loader = create_demo_dataloaders(
        num_train=64,  # Very small dataset for quick demo
        num_val=16,
        batch_size=batch_size,
        num_classes=num_classes,
        transform=transform
    )
    print(f"✓ Created dataloaders: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Step 3: Train model
    print(f"\n[3/5] Training model for {num_epochs} epochs...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optim.Adam(model.parameters(), lr=1e-3),
        criterion=nn.CrossEntropyLoss(),
        checkpoint_dir='./quickstart_checkpoints',
        log_interval=2
    )
    
    trainer.train(num_epochs=num_epochs)
    print("✓ Training complete!")
    
    # Step 4: Export model
    print("\n[4/5] Exporting model to GGML format...")
    exporter = GGMLExporter(model)
    exporter.export('quickstart_model.ggml', use_fp16=True)
    print("✓ Model exported to quickstart_model.ggml")
    
    # Step 5: Run inference
    print("\n[5/5] Testing inference...")
    pipeline = OMRPipeline(num_classes=num_classes)
    pipeline.model = model
    
    # Test with random image
    test_image = torch.randn(1, 3, 512, 512)
    result = pipeline.predict(test_image, enhance=False, return_probabilities=True)
    
    print(f"✓ Inference successful!")
    print(f"  Predicted class: {result['prediction']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Quick Start Complete! 🎉")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Replace synthetic data with your own music sheet dataset")
    print("  2. Adjust hyperparameters in config.yaml")
    print("  3. Train for more epochs to improve accuracy")
    print("  4. Export and deploy your model to embedded devices")
    print("\nFor more examples, see the examples/ directory:")
    print("  - examples/basic_inference.py")
    print("  - examples/train_model.py")
    print("  - examples/export_model.py")
    print("\nDocumentation: README.md")
    print("=" * 70)


if __name__ == '__main__':
    main()
