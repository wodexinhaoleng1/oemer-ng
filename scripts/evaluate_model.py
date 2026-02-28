#!/usr/bin/env python3
"""
Evaluate OMR model on validation/test dataset.
Calculates IoU, Dice Coefficient, and Pixel Accuracy.
"""

import os
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from oemer_ng.models.omr_model import OMRModel
from oemer_ng.training.dataset import create_dataloaders
from torchvision import transforms


def calculate_iou(pred, target, num_classes):
    """Calculate IoU for each class"""
    ious = []
    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls

        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()

        if union == 0:
            # Class not present in both prediction and ground truth: IoU is undefined.
            # Use NaN so downstream aggregation can explicitly handle or skip this class.
            ious.append(float("nan"))
        else:
            ious.append(intersection / union)

    return ious


def calculate_dice(pred, target, num_classes):
    """Calculate Dice coefficient for each class"""
    dices = []
    smooth = 1e-6
    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()

        intersection = (pred_cls * target_cls).sum().item()
        union = pred_cls.sum().item() + target_cls.sum().item()

        dice = (2 * intersection + smooth) / (union + smooth)
        dices.append(dice)

    return dices


def evaluate_model(model_path, dataset_path, dataset_type, batch_size=4, num_workers=4):
    """Evaluate model on dataset"""

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    if dataset_type == "cvc":
        num_classes = 3
        n_channels = 1
    elif dataset_type == "ds2":
        num_classes = 4
        n_channels = 3
    else:
        num_classes = 3
        n_channels = 1

    model = OMRModel(n_channels=n_channels, num_classes=num_classes, mode="segmentation")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    # Data transforms
    if dataset_type == "ds2":
        transform = transforms.Compose(
            [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )
    else:
        transform = transforms.Compose([transforms.Normalize(mean=[0.5], std=[0.5])])

    # Create dataloader
    _, val_loader = create_dataloaders(
        train_dir=dataset_path,
        val_dir=None,  # Use same as train for now
        dataset_type=dataset_type,
        batch_size=batch_size,
        num_workers=num_workers,
        transform=transform,
        win_size=128,
    )

    if val_loader is None:
        print("No validation loader available")
        return

    # Evaluation
    total_iou = {cls: 0.0 for cls in range(num_classes)}
    total_dice = {cls: 0.0 for cls in range(num_classes)}
    total_pixels = 0
    correct_pixels = 0

    print(f"\nEvaluating on {len(val_loader)} batches...")

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(val_loader)):
            data, target = data.to(device), target.to(device)

            # Forward
            output = model(data)

            # Get predictions
            pred = torch.argmax(output, dim=1)
            target_indices = torch.argmax(target, dim=1)

            # Pixel accuracy
            total_pixels += target_indices.numel()
            correct_pixels += (pred == target_indices).sum().item()

            # IoU and Dice for each batch
            for b in range(data.size(0)):
                pred_b = pred[b].cpu()
                target_b = target_indices[b].cpu()

                ious = calculate_iou(pred_b, target_b, num_classes)
                dices = calculate_dice(pred_b, target_b, num_classes)

                for cls in range(num_classes):
                    total_iou[cls] += ious[cls]
                    total_dice[cls] += dices[cls]

    # Calculate averages
    num_samples = len(val_loader) * batch_size
    num_batches = len(val_loader)

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    # Pixel accuracy
    pixel_acc = correct_pixels / total_pixels
    print(f"\nPixel Accuracy: {pixel_acc:.4%}")

    # IoU per class
    print(f"\nIoU per class:")
    avg_ious = []
    for cls in range(num_classes):
        avg_iou = total_iou[cls] / num_batches
        avg_ious.append(avg_iou)
        print(f"  Class {cls}: {avg_iou:.4f}")

    mean_iou = sum(avg_ious) / num_classes
    print(f"\nMean IoU: {mean_iou:.4f}")

    # Dice per class
    print(f"\nDice Coefficient per class:")
    avg_dices = []
    for cls in range(num_classes):
        avg_dice = total_dice[cls] / num_batches
        avg_dices.append(avg_dice)
        print(f"  Class {cls}: {avg_dice:.4f}")

    mean_dice = sum(avg_dices) / num_classes
    print(f"\nMean Dice: {mean_dice:.4f}")

    print("=" * 60)

    return {
        "pixel_accuracy": pixel_acc,
        "mean_iou": mean_iou,
        "mean_dice": mean_dice,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate OMR Model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset")
    parser.add_argument(
        "--dataset_type", type=str, default="cvc", choices=["cvc", "ds2"], help="Dataset type"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} does not exist")
        return

    evaluate_model(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        dataset_type=args.dataset_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
