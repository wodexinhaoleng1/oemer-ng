"""
Test: Evaluate model recognition of musical notes and staff positions,
then render an annotated image with staff regions framed in yellow and
note regions framed in green.
"""

import os
import tempfile

import cv2
import numpy as np
import pytest
import torch

from oemer_ng.models.omr_model import OMRModel


# ---------------------------------------------------------------------------
# Segmentation-class constants (matching the CVC-MUSCIMA / SimpleOMR layout)
# ---------------------------------------------------------------------------
CLASS_BACKGROUND = 0
CLASS_STAFF = 1
CLASS_NOTES = 2

# Visualization colors (BGR for OpenCV)
COLOR_STAFF_BGR = (0, 255, 255)   # yellow
COLOR_NOTES_BGR = (0, 255, 0)     # green


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_sheet(height: int = 256, width: int = 512) -> np.ndarray:
    """Create a minimal synthetic sheet-music image (uint8, 3-channel RGB).

    The image contains:
    - A white background
    - Five horizontal staff lines (thin, dark)
    - Several small filled circles that represent note heads
    """
    img = np.full((height, width, 3), 255, dtype=np.uint8)

    # Staff lines – five horizontal lines evenly spaced in the upper half
    line_spacing = height // 12
    staff_y_start = height // 4
    for i in range(5):
        y = staff_y_start + i * line_spacing
        cv2.line(img, (20, y), (width - 20, y), (0, 0, 0), 2)

    # Note heads – small filled circles positioned on / between staff lines
    note_positions = [
        (80,  staff_y_start),
        (160, staff_y_start + line_spacing),
        (240, staff_y_start + 2 * line_spacing),
        (320, staff_y_start + 3 * line_spacing),
        (400, staff_y_start + 4 * line_spacing),
    ]
    for cx, cy in note_positions:
        cv2.circle(img, (cx, cy), 8, (30, 30, 30), -1)

    return img


def _make_synthetic_segmentation(height: int = 256, width: int = 512) -> np.ndarray:
    """Return a (height, width) int64 segmentation map that matches the
    synthetic sheet produced by *_make_synthetic_sheet*.

    This simulates what a trained model should predict, so the
    visualization test is deterministic and independent of model weights.
    """
    seg = np.full((height, width), CLASS_BACKGROUND, dtype=np.int64)

    # Staff-line pixels (class 1)
    line_spacing = height // 12
    staff_y_start = height // 4
    for i in range(5):
        y = staff_y_start + i * line_spacing
        seg[max(0, y - 2): y + 3, 20: width - 20] = CLASS_STAFF

    # Note pixels (class 2) – small rectangular regions around each note head
    note_positions = [
        (80,  staff_y_start),
        (160, staff_y_start + line_spacing),
        (240, staff_y_start + 2 * line_spacing),
        (320, staff_y_start + 3 * line_spacing),
        (400, staff_y_start + 4 * line_spacing),
    ]
    for cx, cy in note_positions:
        r = 10
        seg[max(0, cy - r): cy + r, max(0, cx - r): cx + r] = CLASS_NOTES

    return seg


def _bounding_boxes_from_mask(
    binary_mask: np.ndarray,
    min_area: int = 10,
) -> list:
    """Return a list of (x, y, w, h) bounding boxes from a binary mask.

    Uses connected-component analysis so that each isolated region gets
    its own box.
    """
    mask_u8 = binary_mask.astype(np.uint8) * 255
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    boxes = []
    for label in range(1, num_labels):          # skip background label 0
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        boxes.append((x, y, w, h))
    return boxes


def visualize_staff_and_notes(
    image: np.ndarray,
    segmentation_map: np.ndarray,
    output_path: str,
) -> dict:
    """Draw yellow rectangles around detected staff regions and green
    rectangles around detected note regions, then save the result.

    Args:
        image:            Original sheet-music image (H, W, 3) uint8 RGB.
        segmentation_map: Per-pixel class index map (H, W) int.
        output_path:      File path where the annotated image is saved.

    Returns:
        Dictionary with ``staff_boxes`` and ``note_boxes`` keys, each
        containing the list of bounding boxes ``(x, y, w, h)``.
    """
    # Work on a BGR copy (OpenCV convention)
    annotated = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)

    # --- Staff regions (class 1) → yellow frame ---
    staff_mask = segmentation_map == CLASS_STAFF
    staff_boxes = _bounding_boxes_from_mask(staff_mask)
    for x, y, w, h in staff_boxes:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), COLOR_STAFF_BGR, 2)

    # --- Note regions (class 2) → green frame ---
    note_mask = segmentation_map == CLASS_NOTES
    note_boxes = _bounding_boxes_from_mask(note_mask)
    for x, y, w, h in note_boxes:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), COLOR_NOTES_BGR, 2)

    cv2.imwrite(output_path, annotated)
    return {"staff_boxes": staff_boxes, "note_boxes": note_boxes}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNoteStaffVisualization:
    """Tests for model recognition of musical notes and staff positions."""

    def test_bounding_boxes_from_staff_mask(self):
        """Staff mask produces at least one bounding box per staff line."""
        seg = _make_synthetic_segmentation()
        staff_mask = seg == CLASS_STAFF
        boxes = _bounding_boxes_from_mask(staff_mask)
        # Five staff lines → expect at least five boxes
        assert len(boxes) >= 5, f"Expected ≥5 staff boxes, got {len(boxes)}"

    def test_bounding_boxes_from_notes_mask(self):
        """Note mask produces exactly one bounding box per note head."""
        seg = _make_synthetic_segmentation()
        note_mask = seg == CLASS_NOTES
        boxes = _bounding_boxes_from_mask(note_mask)
        assert len(boxes) == 5, f"Expected 5 note boxes, got {len(boxes)}"

    def test_visualize_saves_image(self):
        """Annotated image is written to disk."""
        img = _make_synthetic_sheet()
        seg = _make_synthetic_segmentation()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "annotated.png")
            result = visualize_staff_and_notes(img, seg, out_path)
            assert os.path.exists(out_path), "Output image was not written."
            assert os.path.getsize(out_path) > 0, "Output image is empty."
            assert "staff_boxes" in result
            assert "note_boxes" in result

    def test_visualize_box_colors(self):
        """Staff pixels in saved image contain yellow; note pixels contain green."""
        height, width = 256, 512
        img = _make_synthetic_sheet(height, width)
        seg = _make_synthetic_segmentation(height, width)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "annotated.png")
            result = visualize_staff_and_notes(img, seg, out_path)

            annotated_bgr = cv2.imread(out_path)
            assert annotated_bgr is not None, "Could not read back the saved image."

            # Check that yellow (BGR: 0,255,255) pixels exist (staff frames)
            yellow_mask = (
                (annotated_bgr[:, :, 0] == 0)
                & (annotated_bgr[:, :, 1] == 255)
                & (annotated_bgr[:, :, 2] == 255)
            )
            assert yellow_mask.any(), "No yellow pixels found; staff frames missing."

            # Check that green (BGR: 0,255,0) pixels exist (note frames)
            green_mask = (
                (annotated_bgr[:, :, 0] == 0)
                & (annotated_bgr[:, :, 1] == 255)
                & (annotated_bgr[:, :, 2] == 0)
            )
            assert green_mask.any(), "No green pixels found; note frames missing."

    def test_segmentation_model_output_shape(self):
        """The 3-class segmentation model returns the expected (B, 3, H, W) tensor."""
        model = OMRModel(num_classes=3, n_channels=3, mode="segmentation")
        model.eval()
        dummy = torch.zeros(1, 3, 256, 512)
        with torch.no_grad():
            output = model(dummy)
        assert output.shape == (1, 3, 256, 512), (
            f"Unexpected output shape: {output.shape}"
        )

    def test_pipeline_with_model_inference(self):
        """End-to-end: model inference → visualization → image saved."""
        height, width = 256, 512

        # Create model
        model = OMRModel(num_classes=3, n_channels=3, mode="segmentation")
        model.eval()

        # Create a synthetic sheet-music tensor (normalized to [0,1])
        img_np = _make_synthetic_sheet(height, width)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # (1, 3, H, W)

        # Run inference
        with torch.no_grad():
            logits = model(img_tensor)          # (1, 3, H, W)
        seg_map = torch.argmax(logits, dim=1).squeeze(0).numpy()  # (H, W)

        assert seg_map.shape == (height, width)

        # Visualize and save
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "model_output.png")
            result = visualize_staff_and_notes(img_np, seg_map, out_path)
            assert os.path.exists(out_path)
            assert "staff_boxes" in result
            assert "note_boxes" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
