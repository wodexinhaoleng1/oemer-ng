#!/usr/bin/env python3
"""Train NoteHeadModel (Model 2) for fine-grained note-head classification.

This script follows the same structure as ``train_model.py`` but is tailored
for the two-stage OMR pipeline.  Model 2 is trained on a dataset that
follows the :class:`~oemer_ng.training.dataset.NoteHeadDataset` layout.

Optionally, if a stage-1 model checkpoint is provided and ``stage1_masks/``
does not yet exist inside the dataset directory, the script will run Model 1
inference on every image in ``images/`` and save the symbol probability maps
to ``{dataset_path}/stage1_masks/`` before training begins.

Usage::

    python examples/train_stage2.py \\
        --dataset_path data/note_head_train \\
        --stage1_model_path checkpoints/stage1_final.pth \\
        --epochs 20 \\
        --batch_size 4

"""

import os
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms

from oemer_ng.models.omr_model import NoteHeadModel
from oemer_ng.training.trainer import Trainer
from oemer_ng.training.dataset import create_notehead_dataloaders
from oemer_ng.training.loss import FocalTverskyLoss


def _generate_stage1_masks(
    images_dir: Path,
    stage1_masks_dir: Path,
    stage1_model_path: str,
    device: torch.device,
) -> None:
    """Run OMRPipeline on every image in *images_dir* and save probability maps.

    Maps are saved as 8-bit grayscale PNGs where pixel value 255 corresponds
    to symbol probability 1.0.

    Args:
        images_dir: Directory containing the raw sheet-music images.
        stage1_masks_dir: Destination directory for the probability maps.
        stage1_model_path: Path to the OMRModel checkpoint.
        device: Torch device used for inference.
    """
    from oemer_ng.inference.pipeline import OMRPipeline

    stage1_masks_dir.mkdir(parents=True, exist_ok=True)

    pipeline = OMRPipeline(
        model_path=stage1_model_path,
        device=str(device),
        num_classes=3,
        mode="segmentation",
        n_channels=1,
    )

    image_paths = sorted(images_dir.glob("*.png"))
    print(f"Generating stage-1 masks for {len(image_paths)} images …")

    for img_path in image_paths:
        out_path = stage1_masks_dir / img_path.name
        if out_path.exists():
            continue

        _, symbol_prob_map = pipeline.get_symbol_mask(str(img_path))
        # symbol_prob_map shape: (1, 1, H, W)
        prob_hw = symbol_prob_map[0, 0]
        prob_uint8 = (prob_hw * 255.0).clip(0, 255).astype(np.uint8)
        cv2.imwrite(str(out_path), prob_uint8)

    print(f"Stage-1 masks saved to {stage1_masks_dir}")


def main() -> None:
    """Entry point for the stage-2 training script."""
    parser = argparse.ArgumentParser(description="Train NoteHeadModel (Stage 2)")

    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Root directory of NoteHeadDataset",
    )
    parser.add_argument(
        "--stage1_masks_dir",
        type=str,
        default=None,
        help="Directory containing Model 1 symbol probability maps (optional)",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default=None,
        help="Root directory of the validation NoteHeadDataset (optional)",
    )
    parser.add_argument(
        "--stage1_model_path",
        type=str,
        default=None,
        help=(
            "Path to a trained OMRModel checkpoint.  When --stage1_masks_dir is "
            "not provided, stage-1 masks are generated automatically from this model."
        ),
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader worker processes")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument(
        "--no_stage1_mask",
        action="store_true",
        help="Disable stage-1 mask input (train Model 2 without stage-1 context)",
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Device
    # -------------------------------------------------------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    print(f"Using device: {device}")

    use_stage1_mask = not args.no_stage1_mask

    # -------------------------------------------------------------------------
    # Auto-generate stage-1 masks when requested
    # -------------------------------------------------------------------------
    dataset_path = Path(args.dataset_path)
    stage1_masks_dir: str | None = args.stage1_masks_dir

    if use_stage1_mask and stage1_masks_dir is None:
        default_s1_dir = dataset_path / "stage1_masks"
        if not default_s1_dir.is_dir() and args.stage1_model_path is not None:
            print("stage1_masks/ not found — generating from stage-1 model …")
            _generate_stage1_masks(
                images_dir=dataset_path / "images",
                stage1_masks_dir=default_s1_dir,
                stage1_model_path=args.stage1_model_path,
                device=device,
            )
        # Let NoteHeadDataset pick up the default directory automatically.
        stage1_masks_dir = None  # dataset will look for root_dir/stage1_masks/

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    print(
        f"Initializing NoteHeadModel (use_stage1_mask={use_stage1_mask}, "
        f"num_classes=8, n_channels=1)"
    )
    model = NoteHeadModel(
        num_classes=8,
        n_channels=1,
        base_channels=64,
        use_stage1_mask=use_stage1_mask,
    )

    # -------------------------------------------------------------------------
    # Data transforms & loaders
    # -------------------------------------------------------------------------
    transform = transforms.Compose([transforms.Normalize(mean=[0.5], std=[0.5])])

    print(f"Loading training data from {args.dataset_path}")
    train_loader, val_loader = create_notehead_dataloaders(
        train_dir=args.dataset_path,
        val_dir=args.val_path,
        stage1_masks_dir=stage1_masks_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transform=transform,
        win_size=256,
    )

    print(f"Train batches: {len(train_loader)}")
    if val_loader:
        print(f"Val batches: {len(val_loader)}")

    # -------------------------------------------------------------------------
    # Optimisation
    # -------------------------------------------------------------------------
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = FocalTverskyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # -------------------------------------------------------------------------
    # Trainer
    # -------------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=str(device),
        checkpoint_dir=args.checkpoint_dir,
    )

    trainer.train(num_epochs=args.epochs, scheduler=scheduler, early_stopping_patience=5)

    # -------------------------------------------------------------------------
    # Save final model
    # -------------------------------------------------------------------------
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    final_path = os.path.join(args.checkpoint_dir, "stage2_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Final stage-2 model saved to {final_path}")


if __name__ == "__main__":
    main()
