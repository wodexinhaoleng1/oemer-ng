"""
Dataset utilities for OMR training.
"""

import os
import random
from pathlib import Path
from typing import Optional, Callable, List, Tuple, Dict, Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms as T


class OMRDataset(Dataset):
    """
    Base Dataset class for Optical Music Recognition.
    """

    def __init__(self, transform: Optional[Callable] = None):
        self.transform = transform

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class CvcMuscimaDataset(OMRDataset):
    """
    Dataset for CvcMuscima-Distortions.
    Structure:
        root_dir/
            {distortion_type}/
                {folder}/
                    image/
                    gt/
                    symbol/
    """

    def __init__(
        self,
        root_dir: str,
        win_size: int = 256,
        transform: Optional[Callable] = None,
        samples_per_epoch: Optional[int] = None,
    ):
        super().__init__(transform=transform)
        self.root_dir = Path(root_dir)
        self.win_size = win_size
        self.data_paths = self._get_data_paths()
        self.samples_per_epoch = samples_per_epoch if samples_per_epoch else len(self.data_paths)

    def _get_data_paths(self) -> List[Dict[str, Path]]:
        data = []
        # Walk through directories
        for root, dirs, files in os.walk(self.root_dir):
            if "image" in dirs and "gt" in dirs and "symbol" in dirs:
                img_dir = Path(root) / "image"
                gt_dir = Path(root) / "gt"
                sym_dir = Path(root) / "symbol"

                for img_file in os.listdir(img_dir):
                    if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                        base_name = img_file
                        data.append(
                            {
                                "image": img_dir / base_name,
                                "staff": gt_dir / base_name,
                                "symbol": sym_dir / base_name,
                            }
                        )
        return data

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # If samples_per_epoch is larger than data, we wrap around or pick random
        # Here we just pick random if idx >= len(data)
        if idx >= len(self.data_paths):
            idx = random.randint(0, len(self.data_paths) - 1)

        paths = self.data_paths[idx]

        # Load images
        # Use grayscale for simplicity and consistency with OMR
        image = cv2.imread(str(paths["image"]), cv2.IMREAD_GRAYSCALE)
        staff = cv2.imread(str(paths["staff"]), cv2.IMREAD_GRAYSCALE)
        symbol = cv2.imread(str(paths["symbol"]), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Failed to load image: {paths['image']}")

        # Ensure sizes match
        h, w = image.shape
        if staff.shape != (h, w) or symbol.shape != (h, w):
            staff = cv2.resize(staff, (w, h), interpolation=cv2.INTER_NEAREST)
            symbol = cv2.resize(symbol, (w, h), interpolation=cv2.INTER_NEAREST)

        # Random Crop
        if h > self.win_size and w > self.win_size:
            top = random.randint(0, h - self.win_size)
            left = random.randint(0, w - self.win_size)

            image = image[top : top + self.win_size, left : left + self.win_size]
            staff = staff[top : top + self.win_size, left : left + self.win_size]
            symbol = symbol[top : top + self.win_size, left : left + self.win_size]
        else:
            # Resize if smaller
            image = cv2.resize(image, (self.win_size, self.win_size))
            staff = cv2.resize(
                staff, (self.win_size, self.win_size), interpolation=cv2.INTER_NEAREST
            )
            symbol = cv2.resize(
                symbol, (self.win_size, self.win_size), interpolation=cv2.INTER_NEAREST
            )

        # Binarize masks
        staff = (staff > 127).astype(np.float32)
        symbol = (symbol > 127).astype(np.float32)

        # Create background channel
        # 0: background, 1: staff, 2: symbol
        # We want multi-channel output for segmentation
        # Channel 0: Background
        # Channel 1: Staff
        # Channel 2: Symbol
        # Note: They are mutually exclusive in the loss usually, but here they might overlap in raw data?
        # Typically staff and symbol are separate.
        bg = 1.0 - np.maximum(staff, symbol)

        # Stack: (3, H, W)
        mask = np.stack([bg, staff, symbol], axis=0)

        # Image to tensor (1, H, W) or (3, H, W)
        # Normalize to 0-1
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).unsqueeze(0)  # 1CH

        # Mask to tensor
        mask = torch.from_numpy(mask)

        # Apply transform
        if self.transform:
            # Transform expects PIL or Tensor.
            # If tensor, we might need to concat image and mask to transform them together
            # or use torchvision.transforms.v2
            # For simplicity, we assume transform handles image only or we do manual augmentations above.
            # Here we just apply to image
            image = self.transform(image)

        return image, mask


class DeepScoresDataset(OMRDataset):
    """
    Dataset for DeepScores.
    Structure:
        root_dir/
            images/
            segmentation/
    """

    def __init__(
        self,
        root_dir: str,
        win_size: int = 256,  # DeepScores usually uses larger win_size
        transform: Optional[Callable] = None,
    ):
        super().__init__(transform=transform)
        self.root_dir = Path(root_dir)
        self.win_size = win_size
        self.images_dir = self.root_dir / "images"
        self.seg_dir = self.root_dir / "segmentation"

        self.image_files = sorted(list(self.images_dir.glob("*.png")))

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_files[idx]
        # Assumes segmentation file has same name but _seg.png suffix or similar
        # oemer logic: img.png -> img_seg.png
        seg_path = self.seg_dir / (img_path.stem + "_seg.png")
        if not seg_path.exists():
            seg_path = self.seg_dir / (img_path.stem + ".png")

        image = cv2.imread(str(img_path))  # RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        seg = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE)  # Class indices

        if image is None or seg is None:
            raise ValueError(f"Error loading {img_path} or {seg_path}")

        # Resize/Crop
        h, w, _ = image.shape
        if h > self.win_size and w > self.win_size:
            top = random.randint(0, h - self.win_size)
            left = random.randint(0, w - self.win_size)

            image = image[top : top + self.win_size, left : left + self.win_size]
            seg = seg[top : top + self.win_size, left : left + self.win_size]
        else:
            image = cv2.resize(image, (self.win_size, self.win_size))
            seg = cv2.resize(seg, (self.win_size, self.win_size), interpolation=cv2.INTER_NEAREST)

        # Convert seg to one-hot
        # Assume max class index is small. oemer uses ~4 channels?
        # We will assume 4 classes for now: 0: bg, 1: staff, 2: symbol, 3: other
        num_classes = 4
        mask = np.zeros((num_classes, self.win_size, self.win_size), dtype=np.float32)
        for i in range(num_classes):
            mask[i] = (seg == i).astype(np.float32)

        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # 3CH
        mask = torch.from_numpy(mask)

        if self.transform:
            image = self.transform(image)

        return image, mask


class SimpleOMRDataset(OMRDataset):
    """
    Simple dataset for testing/demo purposes.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        num_classes: int = 3,
        image_size: Tuple[int, int] = (256, 256),
        transform: Optional[Callable] = None,
        win_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(transform=transform)
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        if win_size:
            self.image_size = (win_size, win_size)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Generate random image (1 channel)
        image = np.random.randint(0, 256, (*self.image_size, 1), dtype=np.uint8)
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # 1CH

        # Generate random mask (num_classes)
        # Random class per pixel
        label = np.random.randint(0, self.num_classes, self.image_size)
        mask = np.zeros((self.num_classes, *self.image_size), dtype=np.float32)
        for i in range(self.num_classes):
            mask[i] = (label == i).astype(np.float32)

        mask = torch.from_numpy(mask)

        if self.transform:
            image = self.transform(image)

        return image, mask


def create_dataloaders(
    train_dir: str,
    dataset_type: str = "cvc",  # 'cvc', 'ds2', 'simple'
    val_dir: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    transform: Optional[Callable] = None,
    **kwargs,
) -> Tuple[DataLoader, Optional[DataLoader]]:

    if dataset_type == "cvc":
        DatasetClass = CvcMuscimaDataset
    elif dataset_type == "ds2":
        DatasetClass = DeepScoresDataset
    elif dataset_type == "simple":
        DatasetClass = lambda *args, **kw: SimpleOMRDataset(num_samples=1000, **kw)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    train_dataset = DatasetClass(train_dir, transform=transform, **kwargs)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )

    val_loader = None
    if val_dir is not None:
        val_dataset = DatasetClass(val_dir, transform=transform, **kwargs)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader
