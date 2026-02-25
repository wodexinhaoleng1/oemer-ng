#!/usr/bin/env python3
"""
Example: Training an OMR model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from oemer_ng.models.omr_model import OMRModel
from oemer_ng.training.trainer import Trainer
from oemer_ng.training.dataset import create_demo_dataloaders


def main():
    # Configuration
    num_epochs = 5
    batch_size = 16
    learning_rate = 1e-3
    num_classes = 128
    
    print("Setting up training...")
    
    # Create model
    model = OMRModel(num_classes=num_classes)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataloaders (using synthetic data for demo)
    print("Creating demo dataloaders...")
    train_loader, val_loader = create_demo_dataloaders(
        num_train=500,
        num_val=100,
        batch_size=batch_size,
        num_classes=num_classes,
        transform=transform
    )
    
    # Create optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        checkpoint_dir='./checkpoints',
        log_interval=5
    )
    
    # Train
    print("\nStarting training...")
    trainer.train(
        num_epochs=num_epochs,
        scheduler=scheduler,
        early_stopping_patience=3
    )
    
    print("\nTraining completed!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    
    # Save final model
    trainer.save_checkpoint('final_model.pth')
    print("Final model saved to checkpoints/final_model.pth")


if __name__ == '__main__':
    main()
