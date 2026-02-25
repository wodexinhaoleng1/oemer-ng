"""
Dataset utilities for OMR training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Callable, List, Tuple
import json
from PIL import Image
import numpy as np


class OMRDataset(Dataset):
    """
    Dataset class for Optical Music Recognition.
    
    Assumes a directory structure:
        data_dir/
            images/
                img1.jpg
                img2.jpg
                ...
            annotations.json  # Format: {filename: label_id}
    """
    
    def __init__(
        self,
        data_dir: str,
        annotations_file: str = 'annotations.json',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Root directory of dataset
            annotations_file: Name of annotations file
            transform: Transform to apply to images
            target_transform: Transform to apply to labels
        """
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / 'images'
        self.transform = transform
        self.target_transform = target_transform
        
        # Load annotations
        annotations_path = self.data_dir / annotations_file
        with open(annotations_path, 'r') as f:
            self.annotations = json.load(f)
        
        # Create list of samples
        self.samples = [
            (self.images_dir / filename, label)
            for filename, label in self.annotations.items()
            if (self.images_dir / filename).exists()
        ]
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in {data_dir}")
        
        print(f"Loaded {len(self.samples)} samples from {data_dir}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get item by index.
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (image, label)
        """
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label


class SimpleOMRDataset(Dataset):
    """
    Simple dataset for testing/demo purposes.
    
    Generates synthetic data for quick prototyping.
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        num_classes: int = 128,
        image_size: Tuple[int, int] = (512, 512),
        transform: Optional[Callable] = None,
    ):
        """
        Initialize synthetic dataset.
        
        Args:
            num_samples: Number of samples to generate
            num_classes: Number of classes
            image_size: Image size (height, width)
            transform: Transform to apply
        """
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        self.transform = transform
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Generate a random sample."""
        # Generate random image
        image = np.random.randint(0, 256, (*self.image_size, 3), dtype=np.uint8)
        image = Image.fromarray(image)
        
        # Generate random label
        label = np.random.randint(0, self.num_classes)
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        return image, label


def create_dataloaders(
    train_dir: str,
    val_dir: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    transform: Optional[Callable] = None,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create training and validation dataloaders.
    
    Args:
        train_dir: Training data directory
        val_dir: Validation data directory
        batch_size: Batch size
        num_workers: Number of worker processes
        transform: Transform to apply
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = OMRDataset(train_dir, transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = None
    if val_dir is not None:
        val_dataset = OMRDataset(val_dir, transform=transform)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_loader, val_loader


def create_demo_dataloaders(
    num_train: int = 1000,
    num_val: int = 200,
    batch_size: int = 32,
    num_classes: int = 128,
    transform: Optional[Callable] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create demo dataloaders with synthetic data.
    
    Args:
        num_train: Number of training samples
        num_val: Number of validation samples
        batch_size: Batch size
        num_classes: Number of classes
        transform: Transform to apply
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = SimpleOMRDataset(num_train, num_classes, transform=transform)
    val_dataset = SimpleOMRDataset(num_val, num_classes, transform=transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Don't use multiprocessing for synthetic data
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader
