"""
OMR Inference Pipeline.
"""

import torch
import numpy as np
import os
import concurrent.futures
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple
from PIL import Image

try:
    from ..models.omr_model import OMRModel, NoteHeadModel
except ImportError:
    OMRModel = None
    NoteHeadModel = None
from ..utils.preprocessing import ImagePreprocessor, enhance_sheet_music


class OMRPipeline:
    """
    Complete pipeline for optical music recognition (Model 1).

    Handles preprocessing, model inference, and postprocessing.  Primarily
    intended for staff-line / symbol segmentation (3 classes), but retains a
    ``mode`` parameter for backward compatibility with classification use-cases.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        use_quantized: bool = False,
        num_classes: int = 128,
        mode: str = "classification",
        n_channels: int = 3,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ):
        """
        Initialize OMR pipeline.

        Args:
            model_path: Path to pretrained model weights
            device: Device to run inference on ('cpu', 'cuda', or None for auto)
            use_quantized: Whether to use quantized model
            num_classes: Number of output classes
            mode: "classification" or "segmentation"
            n_channels: Number of input channels (1 for grayscale, 3 for RGB)
            mean: Normalization mean, e.g. [0.5] for grayscale or [0.485,0.456,0.406] for RGB
            std: Normalization std, e.g. [0.5] for grayscale or [0.229,0.224,0.225] for RGB
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Store channel config
        self.n_channels = n_channels

        # Set normalization params (must match training)
        if mean is None:
            mean = [0.5] if n_channels == 1 else [0.485, 0.456, 0.406]
        if std is None:
            std = [0.5] if n_channels == 1 else [0.229, 0.224, 0.225]
        self._norm_mean = np.array(mean, dtype=np.float32)
        self._norm_std = np.array(std, dtype=np.float32)

        # Initialize preprocessor (normalization overridden below)
        self.preprocessor = ImagePreprocessor(normalize=False)

        # Initialize model
        if OMRModel is None:
            raise ImportError(
                "OMRModel is not available. "
                "Please install the required model module or create the models directory."
            )
        self.model = OMRModel(n_channels=n_channels, num_classes=num_classes, mode=mode)

        # Load weights if provided
        if model_path is not None:
            self.load_model(model_path)

        # Move to device
        self.model = self.model.to(self.device)
        self.model.eval()

        # Store mode for prediction handling
        self.mode = mode

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

        # Preprocess (resize only, normalization handled below)
        tensor = self.preprocessor.preprocess(img, return_tensor=True)

        # Convert to grayscale if model expects 1 channel: (3,H,W) -> (1,H,W)
        if self.n_channels == 1 and tensor.shape[0] == 3:
            tensor = 0.299 * tensor[0:1] + 0.587 * tensor[1:2] + 0.114 * tensor[2:3]

        # Apply correct normalization (matching training)
        mean = torch.tensor(self._norm_mean, dtype=torch.float32).view(-1, 1, 1)
        std = torch.tensor(self._norm_std, dtype=torch.float32).view(-1, 1, 1)
        tensor = (tensor - mean) / std

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
    ) -> Union[int, np.ndarray, Dict]:
        """
        Run inference on a single image.

        Args:
            image: Input image
            enhance: Whether to apply preprocessing enhancement
            return_probabilities: Whether to return class probabilities

        Returns:
            For ``mode="classification"``: predicted class index (int) or dict.
            For ``mode="segmentation"``: segmentation map ``(H, W)`` ndarray or dict.
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
        return prediction

    @torch.no_grad()
    def get_symbol_mask(
        self, image: Union[str, np.ndarray, Image.Image, torch.Tensor], enhance: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run Model 1 in segmentation mode and return the symbol probability map.

        This method is only meaningful when the pipeline was created with
        ``mode="segmentation"``.

        Args:
            image: Input image (path, numpy array, PIL Image, or pre-processed
                tensor of shape ``(1, C, H, W)``).
            enhance: Whether to apply sheet-music enhancement before inference.

        Returns:
            A tuple ``(seg_map, symbol_prob_map)`` where:

            * ``seg_map`` is a ``(H, W)`` int64 ndarray with per-pixel class
              indices (0 = background, 1 = staff, 2 = symbol).
            * ``symbol_prob_map`` is a ``(1, 1, H, W)`` float32 ndarray
              containing the softmax probability of the *symbol* class
              (class index 2), suitable for direct concatenation in Model 2.
        """
        if not isinstance(image, torch.Tensor):
            image = self.preprocess_image(image, enhance=enhance)

        image = image.to(self.device)
        output = self.model(image)  # (1, num_classes, H, W)
        probabilities = torch.softmax(output, dim=1)  # (1, num_classes, H, W)

        seg_map = torch.argmax(probabilities, dim=1).cpu().numpy()[0]  # (H, W)

        # Symbol class is assumed to be index 2 (as in CVC-MUSCIMA: 0=bg,1=staff,2=symbol)
        symbol_idx = min(2, probabilities.shape[1] - 1)
        symbol_prob_map = (
            probabilities[:, symbol_idx : symbol_idx + 1, :, :].cpu().numpy()
        )  # (1, 1, H, W)

        return seg_map, symbol_prob_map

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


# ---------------------------------------------------------------------------
# NoteHeadPipeline — Model 2 inference
# ---------------------------------------------------------------------------


class NoteHeadPipeline:
    """Inference pipeline for NoteHeadModel (Model 2).

    Accepts a raw sheet-music image and an optional stage-1 symbol probability
    map, and returns fine-grained per-pixel class predictions across 8 classes
    (see :data:`~oemer_ng.models.omr_model.NOTE_HEAD_CLASSES`).

    Args:
        model_path: Optional path to pre-trained NoteHeadModel weights.
        device: Compute device (``"cpu"``, ``"cuda"``, or ``None`` for auto).
        num_classes: Number of output classes (default 8).
        n_channels: Number of image channels (1 = grayscale, 3 = RGB).
        base_channels: Channel width at the first encoder level.
        use_stage1_mask: Whether Model 2 expects a stage-1 probability map as
            an extra input channel.
        mean: Per-channel normalisation mean (defaults to ``[0.5]`` for
            grayscale or ``[0.485, 0.456, 0.406]`` for RGB).
        std: Per-channel normalisation std (defaults to ``[0.5]`` for
            grayscale or ``[0.229, 0.224, 0.225]`` for RGB).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        num_classes: int = 8,
        n_channels: int = 1,
        base_channels: int = 64,
        use_stage1_mask: bool = True,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.n_channels = n_channels
        self.use_stage1_mask = use_stage1_mask

        if mean is None:
            mean = [0.5] if n_channels == 1 else [0.485, 0.456, 0.406]
        if std is None:
            std = [0.5] if n_channels == 1 else [0.229, 0.224, 0.225]
        self._norm_mean = np.array(mean, dtype=np.float32)
        self._norm_std = np.array(std, dtype=np.float32)

        self.preprocessor = ImagePreprocessor(normalize=False)

        if NoteHeadModel is None:
            raise ImportError("NoteHeadModel is not available.")

        self.model = NoteHeadModel(
            num_classes=num_classes,
            n_channels=n_channels,
            base_channels=base_channels,
            use_stage1_mask=use_stage1_mask,
        )

        if model_path is not None:
            self.load_model(model_path)

        self.model = self.model.to(self.device)
        self.model.eval()

    def load_model(self, model_path: str):
        """Load NoteHeadModel weights from a checkpoint file.

        Args:
            model_path: Path to the checkpoint (``*.pth``).
        """
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            elif "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            self.model.load_state_dict(checkpoint)
        print(f"NoteHeadModel loaded from {model_path}")

    def save_model(self, save_path: str):
        """Save NoteHeadModel weights to disk.

        Args:
            save_path: Destination file path (``*.pth``).
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "num_classes": self.model.num_classes,
            },
            save_path,
        )
        print(f"NoteHeadModel saved to {save_path}")

    def quantize_model(self):
        """Apply dynamic quantization to the model for CPU efficiency."""
        self.model = torch.quantization.quantize_dynamic(
            self.model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
        )
        print("NoteHeadModel quantized successfully")

    def preprocess_image(
        self, image: Union[str, np.ndarray, Image.Image], enhance: bool = True
    ) -> torch.Tensor:
        """Preprocess an image into a normalised tensor.

        Args:
            image: Input image (path, numpy array, or PIL Image).
            enhance: Whether to apply sheet-music enhancement.

        Returns:
            Float tensor of shape ``(1, n_channels, H, W)``.
        """
        if isinstance(image, str):
            img = np.array(Image.open(image))
        elif isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image

        if enhance:
            img = enhance_sheet_music(img)

        tensor = self.preprocessor.preprocess(img, return_tensor=True)

        if self.n_channels == 1 and tensor.shape[0] == 3:
            tensor = 0.299 * tensor[0:1] + 0.587 * tensor[1:2] + 0.114 * tensor[2:3]

        mean = torch.tensor(self._norm_mean, dtype=torch.float32).view(-1, 1, 1)
        std = torch.tensor(self._norm_std, dtype=torch.float32).view(-1, 1, 1)
        tensor = (tensor - mean) / std

        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)

        return tensor

    @torch.no_grad()
    def predict(
        self,
        image: Union[str, np.ndarray, Image.Image, torch.Tensor],
        stage1_mask: Optional[np.ndarray] = None,
        enhance: bool = True,
        return_probabilities: bool = False,
    ) -> Union[np.ndarray, Dict]:
        """Run Model 2 inference on a single image.

        Args:
            image: Input image.
            stage1_mask: Symbol probability map from Model 1.  Expected shape
                ``(1, 1, H, W)`` or ``(H, W)`` ndarray in ``[0, 1]``.  When
                ``use_stage1_mask=True`` and this is ``None``, an all-zero
                placeholder is used automatically.
            enhance: Whether to apply sheet-music enhancement.
            return_probabilities: Whether to include class probabilities in the
                return value.

        Returns:
            ``(H, W)`` integer ndarray with per-pixel class indices, or a
            dictionary ``{"prediction": ..., "probabilities": ...,
            "confidence": ...}`` when ``return_probabilities=True``.
        """
        if not isinstance(image, torch.Tensor):
            image = self.preprocess_image(image, enhance=enhance)

        image = image.to(self.device)

        # Prepare stage-1 mask tensor
        s1_tensor: Optional[torch.Tensor] = None
        if self.use_stage1_mask:
            if stage1_mask is not None:
                arr = np.asarray(stage1_mask, dtype=np.float32)
                if arr.ndim == 2:
                    arr = arr[np.newaxis, np.newaxis]  # (1, 1, H, W)
                elif arr.ndim == 3:
                    arr = arr[np.newaxis]  # (1, 1, H, W)
                s1_tensor = torch.from_numpy(arr).to(self.device)
            # If None, NoteHeadModel.forward() fills in zeros automatically.

        output = self.model(image, stage1_mask=s1_tensor)
        probabilities = torch.softmax(output, dim=1)

        prediction = torch.argmax(probabilities, dim=1).cpu().numpy()[0]
        confidence = torch.max(probabilities, dim=1).values.cpu().numpy()[0]

        if return_probabilities:
            return {
                "prediction": prediction,
                "probabilities": probabilities.cpu().numpy()[0],
                "confidence": confidence,
            }
        return prediction


# ---------------------------------------------------------------------------
# TwoStagePipeline — end-to-end two-stage OMR
# ---------------------------------------------------------------------------


class TwoStagePipeline:
    """End-to-end two-stage OMR pipeline.

    Chains :class:`OMRPipeline` (Model 1: staff/symbol segmentation) and
    :class:`NoteHeadPipeline` (Model 2: fine-grained note-head classification).

    Args:
        stage1_model_path: Path to the OMRModel checkpoint (Model 1).
        stage2_model_path: Path to the NoteHeadModel checkpoint (Model 2).
        device: Compute device (``"cpu"``, ``"cuda"``, or ``None`` for auto).
        stage1_num_classes: Number of classes for Model 1 (default 3).
        stage2_num_classes: Number of classes for Model 2 (default 8).
        stage1_base_channels: Base channels for Model 1 (default 64).
        stage2_base_channels: Base channels for Model 2 (default 64).
        n_channels: Number of input image channels shared by both models.
        use_stage1_mask: Whether Model 2 accepts the stage-1 probability map.
    """

    def __init__(
        self,
        stage1_model_path: Optional[str] = None,
        stage2_model_path: Optional[str] = None,
        device: Optional[str] = None,
        stage1_num_classes: int = 3,
        stage2_num_classes: int = 8,
        stage1_base_channels: int = 64,
        stage2_base_channels: int = 64,
        n_channels: int = 1,
        use_stage1_mask: bool = True,
    ):
        self.stage1 = OMRPipeline(
            model_path=stage1_model_path,
            device=device,
            num_classes=stage1_num_classes,
            mode="segmentation",
            n_channels=n_channels,
        )
        # Infer device string from stage1 for consistency
        _device_str = str(self.stage1.device)

        self.stage2 = NoteHeadPipeline(
            model_path=stage2_model_path,
            device=_device_str,
            num_classes=stage2_num_classes,
            n_channels=n_channels,
            base_channels=stage2_base_channels,
            use_stage1_mask=use_stage1_mask,
        )

    @torch.no_grad()
    def predict(
        self,
        image: Union[str, np.ndarray, Image.Image],
        enhance: bool = True,
        return_probabilities: bool = False,
    ) -> Dict:
        """Run the full two-stage pipeline on a single image.

        Args:
            image: Input image (path, numpy array, or PIL Image).
            enhance: Whether to apply sheet-music enhancement before inference.
            return_probabilities: When ``True``, include ``"stage1_probs"`` and
                ``"stage2_probs"`` in the returned dictionary.

        Returns:
            Dictionary with keys:

            * ``"stage1"``: ``(H, W)`` int ndarray — Model 1 segmentation map.
            * ``"stage2"``: ``(H, W)`` int ndarray — Model 2 classification map.
            * ``"stage1_probs"`` *(optional)*: ``(num_classes1, H, W)`` float
              ndarray of Model 1 class probabilities.
            * ``"stage2_probs"`` *(optional)*: ``(num_classes2, H, W)`` float
              ndarray of Model 2 class probabilities.
        """
        # Stage 1
        s1_result = self.stage1.predict(
            image, enhance=enhance, return_probabilities=True
        )
        seg_map_1 = s1_result["prediction"]  # (H, W) ndarray
        probs_1 = s1_result["probabilities"]  # (num_classes1, H, W) ndarray

        # Extract symbol (class 2) probability map as stage-1 mask for stage 2
        symbol_idx = min(2, probs_1.shape[0] - 1)
        symbol_prob = probs_1[symbol_idx]  # (H, W)

        # Stage 2
        s2_result = self.stage2.predict(
            image,
            stage1_mask=symbol_prob,
            enhance=enhance,
            return_probabilities=True,
        )
        seg_map_2 = s2_result["prediction"]  # (H, W) ndarray
        probs_2 = s2_result["probabilities"]  # (num_classes2, H, W) ndarray

        result: Dict = {"stage1": seg_map_1, "stage2": seg_map_2}
        if return_probabilities:
            result["stage1_probs"] = probs_1
            result["stage2_probs"] = probs_2

        return result
