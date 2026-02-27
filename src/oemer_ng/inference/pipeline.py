"""
OMR Inference Pipeline.
"""

import torch
import numpy as np
import os
import concurrent.futures
from pathlib import Path
from typing import Union, List, Dict, Optional
from PIL import Image

try:
    from ..models.omr_model import OMRModel
except ImportError:
    OMRModel = None
from ..utils.preprocessing import ImagePreprocessor, enhance_sheet_music


class OMRPipeline:
    """
    Complete pipeline for optical music recognition.

    Handles preprocessing, model inference, and postprocessing.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        use_quantized: bool = False,
        num_classes: int = 128,
    ):
        """
        Initialize OMR pipeline.

        Args:
            model_path: Path to pretrained model weights
            device: Device to run inference on ('cpu', 'cuda', or None for auto)
            use_quantized: Whether to use quantized model
            num_classes: Number of output classes
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize preprocessor
        self.preprocessor = ImagePreprocessor()

        # Initialize model
        if OMRModel is None:
            raise ImportError(
                "OMRModel is not available. "
                "Please install the required model module or create the models directory."
            )
        self.model = OMRModel(num_classes=num_classes)

        # Load weights if provided
        if model_path is not None:
            self.load_model(model_path)

        # Move to device
        self.model = self.model.to(self.device)
        self.model.eval()

        # Apply quantization if requested
        if use_quantized:
            self.quantize_model()

    def load_model(self, model_path: str):
        """
        Load model weights from file.

        Args:
            model_path: Path to model checkpoint
        """
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            elif "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            self.model.load_state_dict(checkpoint)

        print(f"Model loaded from {model_path}")

    def quantize_model(self):
        """Apply dynamic quantization to the model for efficiency."""
        self.model = torch.quantization.quantize_dynamic(
            self.model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
        )
        print("Model quantized successfully")

    def preprocess_image(
        self, image: Union[str, np.ndarray, Image.Image], enhance: bool = True
    ) -> torch.Tensor:
        """
        Preprocess image for inference.

        Args:
            image: Input image
            enhance: Whether to apply enhancement

        Returns:
            Preprocessed tensor ready for model
        """
        # Load and convert to numpy if needed
        if isinstance(image, str):
            img = Image.open(image)
            img = np.array(img)
        elif isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image

        # Apply enhancement if requested
        if enhance:
            img = enhance_sheet_music(img)

        # Preprocess
        tensor = self.preprocessor.preprocess(img, return_tensor=True)

        # Add batch dimension
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)

        return tensor

    @torch.no_grad()
    def predict(
        self,
        image: Union[str, np.ndarray, Image.Image, torch.Tensor],
        enhance: bool = True,
        return_probabilities: bool = False,
    ) -> Union[int, Dict]:
        """
        Run inference on a single image.

        Args:
            image: Input image
            enhance: Whether to apply preprocessing enhancement
            return_probabilities: Whether to return class probabilities

        Returns:
            Predicted class index or dictionary with prediction and probabilities
        """
        # Preprocess if not already a tensor
        if not isinstance(image, torch.Tensor):
            image = self.preprocess_image(image, enhance=enhance)

        # Move to device
        image = image.to(self.device)

        # Inference
        output = self.model(image)

        # Get prediction
        probabilities = torch.softmax(output, dim=1)

        if output.dim() == 4:
            # Segmentation: (B, C, H, W) -> (B, H, W)
            prediction = torch.argmax(probabilities, dim=1).cpu().numpy()[0]
            confidence = torch.max(probabilities, dim=1).values.cpu().numpy()[0]
        else:
            # Classification: (B, C) -> scalar
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, prediction].item()

        if return_probabilities:
            return {
                "prediction": prediction,
                "probabilities": probabilities.cpu().numpy()[0],
                "confidence": confidence,
            }
        else:
            return prediction

    @torch.no_grad()
    def predict_batch(
        self,
        images: List[Union[str, np.ndarray, Image.Image]],
        enhance: bool = True,
        batch_size: int = 8,
        num_workers: Optional[int] = None,
    ) -> List[int]:
        """
        Run inference on a batch of images.

        Args:
            images: List of input images
            enhance: Whether to apply preprocessing enhancement
            batch_size: Batch size for inference
            num_workers: Number of workers for parallel preprocessing.
                If None, defaults to min(32, number_of_cpus + 4).

        Returns:
            List of predicted class indices
        """
        if not images:
            return []
        predictions = []

        if num_workers is None:
            # Default to number of available CPUs, capped at 32
            num_workers = min(32, os.cpu_count() or 1)

        # Process in batches
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Parallelize preprocessing across all images.
            # executor.map submits all tasks and returns an iterator that yields results in order.
            # This naturally overlaps preprocessing with inference and parallelizes within batches
            # without nested executor calls that could lead to deadlocks.
            # NOTE: self.preprocess_image (and the underlying ImagePreprocessor) is called from
            # multiple threads. This relies on ImagePreprocessor being effectively stateless /
            # read-only and therefore thread-safe. If mutable state or caching is added to
            # ImagePreprocessor in the future, it must remain thread-safe or this parallel
            # preprocessing implementation must be revisited.
            preprocess_fn = lambda img: self.preprocess_image(img, enhance=enhance)
            tensors_iter = executor.map(preprocess_fn, images, chunksize=batch_size)

            for i in range(0, len(images), batch_size):
                # Collect preprocessed tensors for the current batch
                batch_tensors = []
                for _ in range(min(batch_size, len(images) - i)):
                    batch_tensors.append(next(tensors_iter))

                if not batch_tensors:
                    break

                batch_tensor = torch.cat(batch_tensors, dim=0).to(self.device)

                # Inference
                output = self.model(batch_tensor)
                if output.dim() == 4:
                    batch_predictions = torch.argmax(output, dim=1).cpu().numpy()
                    predictions.extend([p for p in batch_predictions])
                else:
                    batch_predictions = torch.argmax(output, dim=1).cpu().tolist()
                    predictions.extend(batch_predictions)

        return predictions

    def save_model(self, save_path: str):
        """
        Save model weights.

        Args:
            save_path: Path to save model
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "num_classes": self.model.num_classes,
            },
            save_path,
        )
        print(f"Model saved to {save_path}")
