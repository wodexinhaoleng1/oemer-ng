"""
Image preprocessing utilities for OMR.
"""

import cv2
import numpy as np
import torch
from PIL import Image
from typing import Union, Tuple, Optional


class ImagePreprocessor:
    """
    Image preprocessor for optical music recognition.
    
    Handles image loading, resizing, normalization, and augmentation.
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (512, 512),
        normalize: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        """
        Initialize preprocessor.
        
        Args:
            target_size: Target image size (height, width)
            normalize: Whether to normalize images
            mean: Mean values for normalization (ImageNet defaults)
            std: Standard deviation values for normalization
        """
        self.target_size = target_size
        self.normalize = normalize
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load an image from file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image as numpy array (H, W, C) in RGB format
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def resize_image(
        self, 
        image: np.ndarray, 
        keep_aspect: bool = True
    ) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image
            keep_aspect: Whether to keep aspect ratio
            
        Returns:
            Resized image
        """
        if keep_aspect:
            h, w = image.shape[:2]
            target_h, target_w = self.target_size
            
            # Calculate scaling factor
            scale = min(target_h / h, target_w / w)
            new_h, new_w = int(h * scale), int(w * scale)
            
            # Resize
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Pad to target size
            pad_h = target_h - new_h
            pad_w = target_w - new_w
            padded = cv2.copyMakeBorder(
                resized,
                0, pad_h,
                0, pad_w,
                cv2.BORDER_CONSTANT,
                value=(255, 255, 255)
            )
            return padded
        else:
            return cv2.resize(image, (self.target_size[1], self.target_size[0]))
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image using mean and std.
        
        Args:
            image: Input image (H, W, C) in range [0, 255]
            
        Returns:
            Normalized image
        """
        # Convert to float and scale to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Normalize
        if self.normalize:
            image = (image - self.mean) / self.std
            
        return image
    
    def preprocess(
        self, 
        image: Union[str, np.ndarray, Image.Image],
        return_tensor: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Complete preprocessing pipeline.
        
        Args:
            image: Input image (path, numpy array, or PIL Image)
            return_tensor: Whether to return PyTorch tensor
            
        Returns:
            Preprocessed image
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = self.load_image(image)
        elif isinstance(image, Image.Image):
            image = np.array(image)
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Resize
        image = self.resize_image(image)
        
        # Normalize
        image = self.normalize_image(image)
        
        # Convert to tensor if requested
        if return_tensor:
            # Change from (H, W, C) to (C, H, W)
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image).float()
            
        return image
    
    def batch_preprocess(
        self,
        images: list,
        return_tensor: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Preprocess a batch of images.
        
        Args:
            images: List of images (paths, arrays, or PIL Images)
            return_tensor: Whether to return PyTorch tensor
            
        Returns:
            Batch of preprocessed images
        """
        processed = [self.preprocess(img, return_tensor=return_tensor) for img in images]
        
        if return_tensor:
            return torch.stack(processed)
        else:
            return np.stack(processed)


def enhance_sheet_music(image: np.ndarray) -> np.ndarray:
    """
    Enhance sheet music image for better recognition.
    
    Applies techniques like binarization, noise removal, and contrast enhancement.
    
    Args:
        image: Input image (RGB or grayscale)
        
    Returns:
        Enhanced image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Noise removal
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    # Convert back to RGB
    if len(image.shape) == 3:
        cleaned = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)
    
    return cleaned
