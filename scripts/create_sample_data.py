#!/usr/bin/env python3
"""
Script to generate sample data for OMR training.
Mimics CvcMuscima and DeepScores directory structures.
"""

import os
import random
import numpy as np
import cv2
from pathlib import Path

def create_cvc_sample(root_dir):
    """
    Create sample CvcMuscima dataset.
    Structure:
        root_dir/
            curvature/
                sample_1/
                    image/
                    gt/
                    symbol/
    """
    root_dir = Path(root_dir)
    distortion = "curvature"
    sample_name = "sample_1"

    base_dir = root_dir / distortion / sample_name
    img_dir = base_dir / "image"
    gt_dir = base_dir / "gt"
    sym_dir = base_dir / "symbol"

    for d in [img_dir, gt_dir, sym_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Generating CvcMuscima samples in {base_dir}")

    for i in range(5):
        filename = f"{i:03d}.png"
        h, w = 1000, 1000

        # Staff (gt): Horizontal lines
        staff = np.zeros((h, w), dtype=np.uint8)
        for y in range(100, h-100, 20):
            cv2.line(staff, (0, y), (w, y), 255, 2)

        # Symbol: Random circles/rectangles
        symbol = np.zeros((h, w), dtype=np.uint8)
        for _ in range(10):
            x = random.randint(50, w-50)
            y = random.randint(50, h-50)
            if random.random() > 0.5:
                cv2.circle(symbol, (x, y), 20, 255, -1)
            else:
                cv2.rectangle(symbol, (x-20, y-20), (x+20, y+20), 255, -1)

        # Image: Combined + Noise
        # Inverted logic: white background, black ink usually.
        # But masks are white on black background.
        # Image is typically grayscale.
        # Let's make image white background (255) and black ink (0).
        image = np.ones((h, w), dtype=np.uint8) * 255

        # Apply staff and symbol (where mask is 255, image becomes 0)
        image[staff > 127] = 0
        image[symbol > 127] = 0

        # Add noise
        noise = np.random.normal(0, 10, (h, w)).astype(np.int16)
        image = np.clip(image + noise, 0, 255).astype(np.uint8)

        cv2.imwrite(str(img_dir / filename), image)
        cv2.imwrite(str(gt_dir / filename), staff)
        cv2.imwrite(str(sym_dir / filename), symbol)


def create_ds2_sample(root_dir):
    """
    Create sample DeepScores dataset.
    Structure:
        root_dir/
            images/
            segmentation/
    """
    root_dir = Path(root_dir)
    img_dir = root_dir / "images"
    seg_dir = root_dir / "segmentation"

    for d in [img_dir, seg_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Generating DeepScores samples in {root_dir}")

    for i in range(5):
        filename = f"ds2_{i:03d}.png"
        seg_filename = f"ds2_{i:03d}_seg.png"
        h, w = 1000, 1000

        # Segmentation mask: 0=bg, 1=staff, 2=symbol, 3=other
        seg = np.zeros((h, w), dtype=np.uint8)

        # Staff (1)
        for y in range(100, h-100, 20):
            cv2.line(seg, (0, y), (w, y), 1, 2)

        # Symbol (2)
        for _ in range(10):
            x = random.randint(50, w-50)
            y = random.randint(50, h-50)
            cv2.circle(seg, (x, y), 20, 2, -1)

        # Other (3)
        for _ in range(5):
            x = random.randint(50, w-50)
            y = random.randint(50, h-50)
            cv2.rectangle(seg, (x-10, y-10), (x+10, y+10), 3, -1)

        # Image based on mask
        image = np.ones((h, w), dtype=np.uint8) * 255
        image[seg > 0] = 0 # Ink is black

        # Save
        # Image is RGB usually
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(str(img_dir / filename), image_rgb)
        cv2.imwrite(str(seg_dir / seg_filename), seg)


def main():
    data_dir = Path("data")
    if not data_dir.exists():
        data_dir.mkdir()

    create_cvc_sample(data_dir / "sample_cvc")
    create_ds2_sample(data_dir / "sample_ds2")

    print("\nSample data generated successfully!")
    print(f"CvcMuscima sample: {data_dir / 'sample_cvc'}")
    print(f"DeepScores sample: {data_dir / 'sample_ds2'}")

if __name__ == "__main__":
    main()
