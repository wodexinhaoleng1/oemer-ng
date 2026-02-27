"""
Image preprocessing utilities for OMR.
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image
from typing import Union, Tuple, Optional
import onnxruntime as ort


def _calculate_iou(box1, box2):
    """计算两个框的IoU"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def _non_max_suppression(boxes, iou_threshold):
    """非极大值抑制"""
    if len(boxes) == 0:
        return []

    # 按置信度排序
    boxes = sorted(boxes, key=lambda x: x["conf"], reverse=True)

    keep = []
    while len(boxes) > 0:
        current = boxes.pop(0)
        keep.append(current)

        boxes = [
            box
            for box in boxes
            if box["cls"] != current["cls"]
            or _calculate_iou(current["xyxy"], box["xyxy"]) < iou_threshold
        ]

    return keep


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
        onnx_model_path: Optional[str] = None,
        onnx_image_size: int = 640,
    ):
        """
        Initialize preprocessor.

        Args:
            target_size: Target image size (height, width)
            normalize: Whether to normalize images
            mean: Mean values for normalization (ImageNet defaults)
            std: Standard deviation values for normalization
            onnx_model_path: Optional path to an ONNX model for initial cleanup.
            onnx_image_size: The input size for the ONNX model.
        """
        self.target_size = target_size
        self.normalize = normalize
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

        self.onnx_session = None
        self.onnx_input_name = None
        self.onnx_output_names = None
        self.onnx_image_size = onnx_image_size
        if onnx_model_path:
            self._load_onnx_model(onnx_model_path)

    def _load_onnx_model(self, model_path: str):
        """加载ONNX模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model file not found: {model_path}")

        providers = ["CPUExecutionProvider"]
        if ort.get_device() == "GPU":
            providers.insert(0, "CUDAExecutionProvider")

        self.onnx_session = ort.InferenceSession(model_path, providers=providers)
        self.onnx_input_name = self.onnx_session.get_inputs()[0].name
        self.onnx_output_names = [output.name for output in self.onnx_session.get_outputs()]

    def _preprocess_for_onnx(self, image: np.ndarray):
        """Preprocesses an image for ONNX inference."""
        original_shape = image.shape[:2]

        h, w = image.shape[:2]
        scale = min(self.onnx_image_size / h, self.onnx_image_size / w)
        new_h, new_w = int(h * scale), int(w * scale)

        img_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        img_padded = np.full((self.onnx_image_size, self.onnx_image_size, 3), 114, dtype=np.uint8)
        top = (self.onnx_image_size - new_h) // 2
        left = (self.onnx_image_size - new_w) // 2
        img_padded[top : top + new_h, left : left + new_w] = img_resized

        img_norm = img_padded.astype(np.float32) / 255.0

        img_transposed = np.transpose(img_norm, (2, 0, 1))
        img_batch = np.expand_dims(img_transposed, axis=0)

        return img_batch, original_shape, (scale, top, left)

    def _postprocess_onnx_outputs(
        self, outputs, original_shape, preprocess_params, conf_threshold=0.25, iou_threshold=0.45
    ):
        """Post-processes the output of the ONNX model."""
        scale, pad_top, pad_left = preprocess_params
        orig_h, orig_w = original_shape

        output = outputs[0]

        if len(output.shape) == 3 and output.shape[1] < output.shape[2]:
            output = np.transpose(output, (0, 2, 1))

        output = output[0]

        boxes = []
        for detection in output:
            if len(detection) < 5:
                continue

            x_center, y_center, width, height = detection[:4]
            class_scores = detection[4:]

            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]

            if confidence < conf_threshold:
                continue

            x1 = (x_center - width / 2 - pad_left) / scale
            y1 = (y_center - height / 2 - pad_top) / scale
            x2 = (x_center + width / 2 - pad_left) / scale
            y2 = (y_center + height / 2 - pad_top) / scale

            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))

            boxes.append({"xyxy": np.array([x1, y1, x2, y2]), "cls": class_id, "conf": confidence})

        if len(boxes) > 0:
            boxes = _non_max_suppression(boxes, iou_threshold)

        return boxes

    def _merge_overlapping_boxes(self, boxes, iou_threshold=0.3):
        """Merges overlapping bounding boxes."""
        if len(boxes) == 0:
            return []

        box_list = sorted(
            [list(b["xyxy"]) + [b["cls"], b["conf"]] for b in boxes],
            key=lambda x: x[5],
            reverse=True,
        )

        merged = []
        used = [False] * len(box_list)

        for i in range(len(box_list)):
            if used[i]:
                continue

            current = box_list[i]
            overlapping = [current]
            used[i] = True

            for j in range(i + 1, len(box_list)):
                if used[j]:
                    continue

                other = box_list[j]
                if current[4] != other[4]:  # Different class
                    continue

                iou = _calculate_iou(current[:4], other[:4])
                if iou > iou_threshold:
                    overlapping.append(other)
                    used[j] = True

            if len(overlapping) > 1:
                all_coords = np.array([b[:4] for b in overlapping])
                min_coords = np.min(all_coords, axis=0)
                max_coords = np.max(all_coords, axis=0)
                max_conf = max([b[5] for b in overlapping])
                merged_box = [
                    min_coords[0],
                    min_coords[1],
                    max_coords[2],
                    max_coords[3],
                    current[4],
                    max_conf,
                ]
                merged.append(merged_box)
            else:
                merged.append(current)

        return merged

    def _extract_and_paste_regions(self, image: np.ndarray, merged_boxes):
        """Extracts detected regions and pastes them onto a white canvas."""
        h, w = image.shape[:2]
        white_canvas = np.ones((h, w, 3), dtype=np.uint8) * 255

        for box in merged_boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

            region = image[y1:y2, x1:x2]
            white_canvas[y1:y2, x1:x2] = region

        return white_canvas

    def _cleanup_with_onnx(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        merge_iou_threshold: float = 0.3,
    ) -> np.ndarray:
        """
        Cleans up the image using the loaded ONNX model.
        This involves detecting regions, merging them, and creating a new image
        with only those regions on a white background.
        """
        # ONNX model expects BGR, but we work with RGB.
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        input_data, original_shape, preprocess_params = self._preprocess_for_onnx(image_bgr)

        outputs = self.onnx_session.run(self.onnx_output_names, {self.onnx_input_name: input_data})

        boxes = self._postprocess_onnx_outputs(
            outputs, original_shape, preprocess_params, conf_threshold, iou_threshold
        )

        merged_boxes = self._merge_overlapping_boxes(boxes, merge_iou_threshold)

        cleaned_bgr = self._extract_and_paste_regions(image_bgr, merged_boxes)

        # Convert back to RGB for consistency within the class
        cleaned_rgb = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)

        return cleaned_rgb

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

    def resize_image(self, image: np.ndarray, keep_aspect: bool = True) -> np.ndarray:
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
                resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(255, 255, 255)
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
        self, image: Union[str, np.ndarray, Image.Image], return_tensor: bool = True
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
            image = np.array(image.convert("RGB"))

        # In-place conversion for grayscale to RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Step 1: Clean image using ONNX model if available
        if self.onnx_session:
            image = self._cleanup_with_onnx(image)

        # Step 2: Resize
        image = self.resize_image(image)

        # Step 3: Normalize
        image = self.normalize_image(image)

        # Step 4: Convert to tensor if requested
        if return_tensor:
            # Change from (H, W, C) to (C, H, W)
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image.copy()).float()

        return image

    def batch_preprocess(
        self, images: list, return_tensor: bool = True
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
