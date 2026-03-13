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
    """Dataset for DeepScores V2 (Model 1 / seg_net) segmentation training.

    Directory structure::

        root_dir/
            images/         # Input sheet music images (.png)
            segmentation/   # Multi-channel segmentation labels (*_seg.png or *.png)

    Loading logic:

    1. All ``.png`` files in ``images/`` are paired with their corresponding
       segmentation file (``*_seg.png`` first, then ``*.png``) in
       ``segmentation/``.
    2. Paired files are split into train / validation subsets using a fixed
       random seed so the split is reproducible.
    3. Sliding-window crop coordinates (``win_size × win_size``) are
       pre-computed with step size ``stride`` for every image.  For each image
       the maximum *y*-coordinate that contains staff-line content (label
       channel index 1) is determined so that blank bottom regions of the
       sheet are excluded from the patch list.  The resulting coordinate list
       per image is randomly shuffled.
    4. During training (``augment=True``) the following augmentations are
       applied **synchronously** to the image *and* every channel of the
       multi-channel label:

       * Random scaling (factor 0.8–1.2)
       * Random perspective warp (up to 5 % margin distortion)

       The following are applied to the **image only**:

       * Gaussian blur (kernel 3 or 5, probability 0.3)
       * Additive noise / colour dithering (σ = 10, probability 0.5)
       * Brightness / contrast pixel perturbation (probability 0.5)

    The returned ``label`` tensor has shape ``(CHANNEL_NUM, H, W)`` as a
    one-hot float32 array:

    * channel 0 – background
    * channel 1 – staff lines
    * channel 2 – symbol regions
    * channel 3 – other

    Args:
        root_dir: Root directory (see layout above).
        win_size: Sliding-window crop size in pixels.  Defaults to 288.
        stride: Sliding-window step size in pixels.  Defaults to
            ``win_size // 2`` when ``None``.
        transform: Optional normalisation transform applied to the *image*
            tensor only, after augmentation.
        augment: Whether to apply training augmentations.  Defaults to
            ``True``.
        val_ratio: Fraction of images reserved for validation.  Defaults to
            0.2.
        split: ``"train"`` or ``"val"``.  Defaults to ``"train"``.
        seed: Random seed used for the reproducible train/val file split.
            Defaults to 42.
    """

    #: Number of output label channels (background + 3 foreground classes).
    CHANNEL_NUM: int = 4

    def __init__(
        self,
        root_dir: str,
        win_size: int = 288,
        stride: Optional[int] = None,
        transform: Optional[Callable] = None,
        augment: bool = True,
        val_ratio: float = 0.2,
        split: str = "train",
        seed: int = 42,
        samples_per_epoch: Optional[int] = None,
    ):
        super().__init__(transform=transform)
        self.root_dir = Path(root_dir)
        self.win_size = win_size
        self.stride = stride if stride is not None else win_size // 2
        self.augment = augment
        self.samples_per_epoch = samples_per_epoch

        self.images_dir = self.root_dir / "images"
        self.seg_dir = self.root_dir / "segmentation"

        # Collect all (image_path, seg_path) pairs and split train / val.
        all_pairs = self._find_pairs()
        rng_split = random.Random(seed)
        shuffled = list(all_pairs)
        rng_split.shuffle(shuffled)
        if val_ratio > 0.0 and len(shuffled) > 1:
            n_val = max(1, int(len(shuffled) * val_ratio))
        else:
            n_val = 0
        if split == "val":
            self.pairs: List[Tuple[Path, Path]] = shuffled[:n_val]
        else:
            self.pairs = shuffled[n_val:]

        # Pre-compute all (pair_idx, top, left) sliding-window patches.
        self.patches: List[Tuple[int, int, int]] = self._compute_patches()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _find_pairs(self) -> List[Tuple[Path, Path]]:
        """Return sorted list of (image_path, seg_path) pairs."""
        pairs: List[Tuple[Path, Path]] = []
        for img_path in sorted(self.images_dir.glob("*.png")):
            seg_path = self.seg_dir / (img_path.stem + "_seg.png")
            if not seg_path.exists():
                seg_path = self.seg_dir / (img_path.stem + ".png")
            if seg_path.exists():
                pairs.append((img_path, seg_path))
        return pairs

    def _compute_patches(self) -> List[Tuple[int, int, int]]:
        """Pre-compute ``(pair_idx, top, left)`` for all valid patches.

        For each image the segmentation label is loaded to determine the
        maximum *y*-coordinate with staff-line content (channel 1).  Patches
        that start entirely below this row are skipped, which avoids
        training on the blank bottom margins common in printed music sheets.
        """
        patches: List[Tuple[int, int, int]] = []
        rng = random.Random(0)  # Fixed seed: reproducible per-image shuffle.
        for pair_idx, (img_path, seg_path) in enumerate(self.pairs):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            # Find the lowest row that contains a staff-line pixel so we can
            # skip blank bottom regions of the sheet.
            max_y = h
            seg = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE)
            if seg is not None:
                if seg.shape != (h, w):
                    seg = cv2.resize(seg, (w, h), interpolation=cv2.INTER_NEAREST)
                staff_rows = np.where(seg == 1)[0]
                if len(staff_rows) > 0:
                    # Allow one extra window past the last staff row so that
                    # patches which *contain* the last staff line (i.e. start
                    # above it but extend below it) are not accidentally
                    # excluded.
                    max_y = min(int(staff_rows.max()) + self.win_size, h)

            # Enumerate all top-left corners of the sliding window grid.
            coords: List[Tuple[int, int]] = []
            top = 0
            while top < max_y:
                left = 0
                while left < w:
                    coords.append((top, left))
                    left += self.stride
                top += self.stride

            rng.shuffle(coords)
            for top_c, left_c in coords:
                patches.append((pair_idx, top_c, left_c))

        return patches

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        if self.samples_per_epoch is not None:
            return min(self.samples_per_epoch, len(self.patches))
        return len(self.patches)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pair_idx, top, left = self.patches[idx % len(self.patches)]
        img_path, seg_path = self.pairs[pair_idx]

        # Load full images.
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        seg = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE)
        if seg is None:
            raise ValueError(f"Failed to load segmentation: {seg_path}")

        h, w, _ = image.shape
        if seg.shape != (h, w):
            seg = cv2.resize(seg, (w, h), interpolation=cv2.INTER_NEAREST)

        # Extract the sliding-window crop, padding if the window extends
        # beyond the image boundary.
        pad_bottom = max(0, top + self.win_size - h)
        pad_right = max(0, left + self.win_size - w)
        if pad_bottom > 0 or pad_right > 0:
            image = cv2.copyMakeBorder(
                image, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT
            )
            seg = cv2.copyMakeBorder(
                seg, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=0
            )

        image = image[top : top + self.win_size, left : left + self.win_size]
        seg = seg[top : top + self.win_size, left : left + self.win_size]

        # Convert class-index segmentation map to one-hot multi-channel label.
        label = np.zeros(
            (self.CHANNEL_NUM, self.win_size, self.win_size), dtype=np.float32
        )
        for i in range(self.CHANNEL_NUM):
            label[i] = (seg == i).astype(np.float32)

        # Apply synchronised augmentations (image + label together).
        if self.augment:
            image, label = self._augment(image, label)

        # Convert to tensors.
        img_tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(
            2, 0, 1
        )  # (3, H, W)
        label_tensor = torch.from_numpy(label)  # (CHANNEL_NUM, H, W)

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, label_tensor

    # ------------------------------------------------------------------
    # Augmentation helpers
    # ------------------------------------------------------------------

    def _augment(
        self, image: np.ndarray, label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply the full augmentation pipeline to *image* and *label*.

        Geometric transforms (scaling, perspective) are applied to both the
        image and every channel of the multi-channel label so that spatial
        correspondence is preserved.  Photometric transforms (blur, noise,
        brightness / contrast) are applied to the image only.
        """
        # --- Geometric (synchronized) ---
        if random.random() < 0.5:
            image, label = self._random_scale(image, label)

        if random.random() < 0.3:
            image, label = self._random_perspective(image, label)

        # --- Photometric (image only) ---
        if random.random() < 0.3:
            ksize = random.choice([3, 5])
            image = cv2.GaussianBlur(image, (ksize, ksize), 0)

        if random.random() < 0.5:
            noise = np.random.normal(0, 10, image.shape).astype(np.float32)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        if random.random() < 0.5:
            alpha = random.uniform(0.8, 1.2)
            beta = float(random.randint(-20, 20))
            image = np.clip(
                alpha * image.astype(np.float32) + beta, 0, 255
            ).astype(np.uint8)

        return image, label

    def _random_scale(
        self, image: np.ndarray, label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Scale image and label by a random factor in [0.8, 1.2]."""
        scale = random.uniform(0.8, 1.2)
        h, w = image.shape[:2]
        new_h = max(1, int(h * scale))
        new_w = max(1, int(w * scale))
        image = cv2.resize(image, (new_w, new_h))
        scaled: List[np.ndarray] = [
            cv2.resize(label[i], (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            for i in range(self.CHANNEL_NUM)
        ]
        label = np.stack(scaled, axis=0)
        return self._crop_or_pad(image, label)

    def _crop_or_pad(
        self, image: np.ndarray, label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Pad (reflect) or centre-crop image/label to ``win_size``."""
        ws = self.win_size
        h, w = image.shape[:2]
        pad_h = max(0, ws - h)
        pad_w = max(0, ws - w)
        if pad_h > 0 or pad_w > 0:
            tp, bp = pad_h // 2, pad_h - pad_h // 2
            lp, rp = pad_w // 2, pad_w - pad_w // 2
            image = cv2.copyMakeBorder(image, tp, bp, lp, rp, cv2.BORDER_REFLECT)
            padded: List[np.ndarray] = [
                cv2.copyMakeBorder(
                    label[i], tp, bp, lp, rp, cv2.BORDER_CONSTANT, value=0
                )
                for i in range(self.CHANNEL_NUM)
            ]
            label = np.stack(padded, axis=0)
            h, w = image.shape[:2]
        top = (h - ws) // 2
        left = (w - ws) // 2
        image = image[top : top + ws, left : left + ws]
        label = label[:, top : top + ws, left : left + ws]
        return image, label

    def _random_perspective(
        self, image: np.ndarray, label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply a random perspective warp to image and every label channel."""
        h, w = image.shape[:2]
        margin = max(1, int(min(h, w) * 0.05))
        src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst_pts = np.float32(
            [
                [random.randint(0, margin), random.randint(0, margin)],
                [w - random.randint(0, margin), random.randint(0, margin)],
                [w - random.randint(0, margin), h - random.randint(0, margin)],
                [random.randint(0, margin), h - random.randint(0, margin)],
            ]
        )
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        image = cv2.warpPerspective(
            image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )
        warped: List[np.ndarray] = [
            cv2.warpPerspective(
                label[i],
                M,
                (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            for i in range(self.CHANNEL_NUM)
        ]
        label = np.stack(warped, axis=0)
        return image, label


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


class NoteHeadDataset(OMRDataset):
    """Dataset for training NoteHeadModel (Model 2) on DeepScores V2 data.

    Expected directory layout::

        root_dir/
            images/             # Raw grayscale score images (.png)
            stage1_masks/       # Symbol probability maps from Model 1 (optional)
            note_head/          # Note-head binary masks
            stem/               # Note-stem binary masks
            beam/               # Beam binary masks
            rest/               # Rest binary masks
            flag/               # Flag binary masks
            dot/                # Dot binary masks
            accidental/         # Accidental binary masks

    If a foreground sub-directory does not exist its contribution to the
    target mask is treated as all-zero (incomplete annotations are allowed).

    If ``stage1_masks/`` exists *or* ``stage1_masks_dir`` is supplied the
    symbol probability map is appended as a second input channel, yielding an
    input tensor of shape ``(2, H, W)``.  Otherwise the input shape is
    ``(1, H, W)``.

    The target mask has shape ``(8, H, W)`` in one-hot format; channel 0 is
    the background (``1 − union_of_foreground_classes``).

    Args:
        root_dir: Root directory of the dataset (see layout above).
        win_size: Size of the random square crop.
        transform: Optional transform applied to the image tensor only.
        samples_per_epoch: Virtual epoch size (random sampling with
            replacement when larger than the number of available files).
        stage1_masks_dir: Explicit path to the stage-1 probability maps
            directory.  Overrides the default ``root_dir/stage1_masks/``.
    """

    # Ordered list of foreground class sub-directories (classes 1–7)
    FOREGROUND_DIRS: List[str] = [
        "note_head",
        "stem",
        "beam",
        "rest",
        "flag",
        "dot",
        "accidental",
    ]

    def __init__(
        self,
        root_dir: str,
        win_size: int = 256,
        transform: Optional[Callable] = None,
        samples_per_epoch: Optional[int] = None,
        stage1_masks_dir: Optional[str] = None,
    ):
        super().__init__(transform=transform)
        self.root_dir = Path(root_dir)
        self.win_size = win_size

        self.images_dir = self.root_dir / "images"
        self.image_files = sorted(self.images_dir.glob("*.png"))
        if not self.image_files:
            raise ValueError(f"No PNG images found in {self.images_dir}")

        self.samples_per_epoch = (
            samples_per_epoch if samples_per_epoch is not None else len(self.image_files)
        )

        # Foreground mask directories (may not all exist)
        self.fg_dirs: List[Optional[Path]] = []
        for sub in self.FOREGROUND_DIRS:
            p = self.root_dir / sub
            self.fg_dirs.append(p if p.is_dir() else None)

        # Stage-1 probability map directory (optional)
        if stage1_masks_dir is not None:
            s1_dir = Path(stage1_masks_dir)
        else:
            s1_dir = self.root_dir / "stage1_masks"
        self.stage1_masks_dir: Optional[Path] = s1_dir if s1_dir.is_dir() else None

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = idx % len(self.image_files)

        img_path = self.image_files[idx]
        stem = img_path.stem

        # Load grayscale image
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        h, w = image.shape

        # Load foreground masks (resize to match image if needed)
        fg_masks: List[np.ndarray] = []
        for fg_dir in self.fg_dirs:
            if fg_dir is not None:
                mask_path = fg_dir / f"{stem}.png"
                if mask_path.exists():
                    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if m is not None:
                        if m.shape != (h, w):
                            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                        fg_masks.append((m > 127).astype(np.float32))
                        continue
            fg_masks.append(np.zeros((h, w), dtype=np.float32))

        # Load stage-1 mask (optional)
        stage1_arr: Optional[np.ndarray] = None
        if self.stage1_masks_dir is not None:
            s1_path = self.stage1_masks_dir / f"{stem}.png"
            if s1_path.exists():
                s1 = cv2.imread(str(s1_path), cv2.IMREAD_GRAYSCALE)
                if s1 is not None:
                    if s1.shape != (h, w):
                        s1 = cv2.resize(s1, (w, h), interpolation=cv2.INTER_NEAREST)
                    stage1_arr = s1.astype(np.float32) / 255.0

        # Random crop / resize
        if h > self.win_size and w > self.win_size:
            top = random.randint(0, h - self.win_size)
            left = random.randint(0, w - self.win_size)
            image = image[top : top + self.win_size, left : left + self.win_size]
            fg_masks = [m[top : top + self.win_size, left : left + self.win_size] for m in fg_masks]
            if stage1_arr is not None:
                stage1_arr = stage1_arr[top : top + self.win_size, left : left + self.win_size]
        else:
            image = cv2.resize(image, (self.win_size, self.win_size))
            fg_masks = [
                cv2.resize(m, (self.win_size, self.win_size), interpolation=cv2.INTER_NEAREST)
                for m in fg_masks
            ]
            if stage1_arr is not None:
                stage1_arr = cv2.resize(
                    stage1_arr, (self.win_size, self.win_size), interpolation=cv2.INTER_LINEAR
                )

        # Build one-hot target (8, H, W): channel 0 = background
        fg_union = np.zeros((self.win_size, self.win_size), dtype=np.float32)
        for m in fg_masks:
            fg_union = np.maximum(fg_union, m)
        background = 1.0 - fg_union
        mask = np.stack([background] + fg_masks, axis=0)  # (8, H, W)

        # Image tensor (normalised to [0, 1])
        image_f = image.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(image_f).unsqueeze(0)  # (1, H, W)

        if stage1_arr is not None:
            s1_tensor = torch.from_numpy(stage1_arr).unsqueeze(0)  # (1, H, W)
            img_tensor = torch.cat([img_tensor, s1_tensor], dim=0)  # (2, H, W)

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, torch.from_numpy(mask)


def create_dataloaders(
    train_dir: str,
    dataset_type: str = "cvc",  # 'cvc', 'ds2', 'simple', 'note_head'
    val_dir: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    transform: Optional[Callable] = None,
    **kwargs,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create training (and optionally validation) DataLoaders.

    Args:
        train_dir: Path to the training dataset root directory.
        dataset_type: One of ``"cvc"``, ``"ds2"``, ``"simple"``, or
            ``"note_head"``.
        val_dir: Optional path to the validation dataset root directory.
        batch_size: Batch size for both loaders.
        num_workers: Number of DataLoader worker processes.
        transform: Optional transform applied to images.
        **kwargs: Extra keyword arguments forwarded to the dataset constructor.

    Returns:
        ``(train_loader, val_loader)`` — ``val_loader`` is ``None`` when
        ``val_dir`` is not provided.
    """
    if dataset_type == "cvc":
        DatasetClass = CvcMuscimaDataset
    elif dataset_type == "ds2":
        DatasetClass = DeepScoresDataset
    elif dataset_type == "simple":
        DatasetClass = lambda *args, **kw: SimpleOMRDataset(num_samples=1000, **kw)
    elif dataset_type == "note_head":
        DatasetClass = NoteHeadDataset
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    train_dataset = DatasetClass(train_dir, transform=transform, **kwargs)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
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
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
        )

    return train_loader, val_loader


def create_notehead_dataloaders(
    train_dir: str,
    val_dir: Optional[str] = None,
    stage1_masks_dir: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    transform: Optional[Callable] = None,
    **kwargs,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Convenience wrapper for creating NoteHeadDataset DataLoaders (Model 2).

    Args:
        train_dir: Path to the training dataset root directory.
        val_dir: Optional path to the validation dataset root directory.
        stage1_masks_dir: Path to the directory containing Model 1 symbol
            probability maps.  When ``None`` the dataset looks for a
            ``stage1_masks/`` sub-directory inside ``train_dir`` / ``val_dir``.
        batch_size: Batch size for both loaders.
        num_workers: Number of DataLoader worker processes.
        transform: Optional transform applied to images.
        **kwargs: Extra keyword arguments forwarded to NoteHeadDataset.

    Returns:
        ``(train_loader, val_loader)`` — ``val_loader`` is ``None`` when
        ``val_dir`` is not provided.
    """
    return create_dataloaders(
        train_dir=train_dir,
        dataset_type="note_head",
        val_dir=val_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        transform=transform,
        stage1_masks_dir=stage1_masks_dir,
        **kwargs,
    )
