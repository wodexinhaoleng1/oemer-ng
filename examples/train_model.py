#!/usr/bin/env python3
"""
Train an OMR model.

Usage:
    python examples/train_model.py --dataset_path data/sample_cvc --dataset_type cvc
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from oemer_ng.models.omr_model import OMRModel
from oemer_ng.training.trainer import Trainer
from oemer_ng.training.dataset import create_dataloaders
from oemer_ng.training.loss import FocalTverskyLoss


def main():
    parser = argparse.ArgumentParser(description="Train OMR Model")
    parser.add_argument('--dataset_path', type=str, default='data/sample_cvc', help='Path to dataset')
    parser.add_argument('--dataset_type', type=str, default='cvc', choices=['cvc', 'ds2', 'simple'], help='Dataset type')
    parser.add_argument('--val_path', type=str, default=None, help='Path to validation dataset')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')

    args = parser.parse_args()
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Model Configuration based on dataset
    if args.dataset_type == 'cvc':
        num_classes = 3  # bg, staff, symbol
        n_channels = 1   # Grayscale input
    elif args.dataset_type == 'ds2':
        num_classes = 4  # bg, staff, symbol, other (assumed)
        n_channels = 3   # RGB input
    else: # simple
        num_classes = 3
        n_channels = 1

    print(f"Initializing model with n_channels={n_channels}, num_classes={num_classes}")
    model = OMRModel(n_channels=n_channels, num_classes=num_classes)

    # Data Transforms
    # Basic transforms. More complex augmentations are handled in Dataset or can be added here.
    # Note: Dataset classes currently expect transforms to be applied to image only,
    # or handle them internally if geometric.
    # We use basic Normalization here.
    if args.dataset_type == 'ds2':
         transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Grayscale normalization
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    print(f"Loading data from {args.dataset_path} ({args.dataset_type})")
    
    # Create dataloaders
    # If val_path is not provided, we can split train set or just not validate (or use same set for demo)
    # create_dataloaders expects val_dir to be Optional
    
    train_loader, val_loader = create_dataloaders(
        train_dir=args.dataset_path,
        val_dir=args.val_path,
        dataset_type=args.dataset_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transform=transform,
        win_size=256 # Default window size
    )
    
    print(f"Train batches: {len(train_loader)}")
    if val_loader:
        print(f"Val batches: {len(val_loader)}")
    
    # Optimization
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Loss function
    # FocalTverskyLoss is good for segmentation with class imbalance
    criterion = FocalTverskyLoss()

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=str(device),
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Start training
    trainer.train(
        num_epochs=args.epochs,
        scheduler=scheduler,
        early_stopping_patience=5
    )
    
    # Save final model
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    final_path = os.path.join(args.checkpoint_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to {final_path}")


if __name__ == '__main__':
    main()
