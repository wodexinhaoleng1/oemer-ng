#!/usr/bin/env python3
"""
Evaluate OMR model on validation/test dataset.
Calculates IoU, Dice Coefficient, Pixel Accuracy, and mAP.
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
import numpy as np


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


def calculate_precision_recall_iou(pred, target, num_classes):
    """Calculate precision, recall, and IoU for each class"""
    precisions = []
    recalls = []
    ious = []

    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls

        tp = (pred_cls & target_cls).sum().item()  # True positives
        fp = (pred_cls & ~target_cls).sum().item()  # False positives
        fn = (~pred_cls & target_cls).sum().item()  # False negatives

        # Precision = TP / (TP + FP)
        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0.0

        # Recall = TP / (TP + FN)
        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0.0

        # IoU = TP / (TP + FP + FN)
        if tp + fp + fn > 0:
            iou = tp / (tp + fp + fn)
        else:
            iou = 0.0

        precisions.append(precision)
        recalls.append(recall)
        ious.append(iou)

    return precisions, recalls, ious


def calculate_ap_per_class(pred, target, num_classes, num_thresholds=100):
    """Calculate Average Precision for each class using precision-recall curves"""
    aps = []

    for cls in range(num_classes):
        # Get confidence scores from probabilities
        # For segmentation, we use the probability of the predicted class
        # Here we simplify by using binary masks
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()

        # Calculate precision-recall at different thresholds
        precisions = []
        recalls = []

        for threshold in np.linspace(0, 1, num_thresholds):
            pred_binary = (pred_cls >= threshold).float()

            tp = (pred_binary * target_cls).sum().item()
            fp = ((1 - pred_binary) * pred_cls).sum().item()
            fn = (pred_binary * (1 - target_cls)).sum().item()

            if tp + fp > 0:
                precision = tp / (tp + fp)
            else:
                precision = 0.0

            if tp + fn > 0:
                recall = tp / (tp + fn)
            else:
                recall = 0.0

            if recall > 0 or precision > 0:
                precisions.append(precision)
                recalls.append(recall)

        if len(precisions) == 0:
            aps.append(0.0)
            continue

        # Sort by recall
        sorted_indices = np.argsort(recalls)
        recalls = [recalls[i] for i in sorted_indices]
        precisions = [precisions[i] for i in sorted_indices]

        # Compute precision envelope
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])

        # Integrate precision-recall curve
        recalls = [0.0] + recalls + [1.0]
        precisions = [0.0] + precisions + [0.0]

        ap = 0.0
        for i in range(len(recalls) - 1):
            ap += (recalls[i + 1] - recalls[i]) * precisions[i + 1]

        aps.append(ap)

    return aps


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
    total_precision = {cls: 0.0 for cls in range(num_classes)}
    total_recall = {cls: 0.0 for cls in range(num_classes)}
    total_pixels = 0
    correct_pixels = 0

    # For mAP calculation
    all_preds = []
    all_targets = []

    print(f"\nEvaluating on {len(val_loader)} batches...")

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(val_loader)):
            data, target = data.to(device), target.to(device)

            # Forward
            output = model(data)

            # Get predictions
            pred = torch.argmax(output, dim=1)
            target_indices = torch.argmax(target, dim=1)

            # Store for mAP calculation
            all_preds.append(pred.cpu())
            all_targets.append(target_indices.cpu())

            # Pixel accuracy
            total_pixels += target_indices.numel()
            correct_pixels += (pred == target_indices).sum().item()

            # IoU, Dice, Precision, Recall for each batch
            for b in range(data.size(0)):
                pred_b = pred[b].cpu()
                target_b = target_indices[b].cpu()

                ious = calculate_iou(pred_b, target_b, num_classes)
                dices = calculate_dice(pred_b, target_b, num_classes)
                precisions, recalls, ious_pr = calculate_precision_recall_iou(
                    pred_b, target_b, num_classes
                )

                for cls in range(num_classes):
                    total_iou[cls] += ious[cls]
                    total_dice[cls] += dices[cls]
                    total_precision[cls] += precisions[cls]
                    total_recall[cls] += recalls[cls]

    # Calculate averages
    num_samples = len(val_loader) * batch_size
    num_batches = len(val_loader)

    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)

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

    # Precision per class
    print(f"\nPrecision per class:")
    avg_precisions = []
    for cls in range(num_classes):
        avg_precision = total_precision[cls] / num_batches
        avg_precisions.append(avg_precision)
        print(f"  Class {cls}: {avg_precision:.4f}")

    mean_precision = sum(avg_precisions) / num_classes
    print(f"\nMean Precision: {mean_precision:.4f}")

    # Recall per class
    print(f"\nRecall per class:")
    avg_recalls = []
    for cls in range(num_classes):
        avg_recall = total_recall[cls] / num_batches
        avg_recalls.append(avg_recall)
        print(f"  Class {cls}: {avg_recall:.4f}")

    mean_recall = sum(avg_recalls) / num_classes
    print(f"\nMean Recall: {mean_recall:.4f}")

    # Calculate mAP
    print(f"\nCalculating mAP...")
    all_preds_concat = torch.cat(all_preds, dim=0)
    all_targets_concat = torch.cat(all_targets, dim=0)

    # Flatten for mAP calculation
    all_preds_flat = all_preds_concat.view(-1).numpy()
    all_targets_flat = all_targets_concat.view(-1).numpy()

    aps = calculate_ap_per_class(all_preds_flat, all_targets_flat, num_classes)
    mean_ap = sum(aps) / num_classes

    print(f"\nAverage Precision per class:")
    for cls in range(num_classes):
        print(f"  Class {cls}: {aps[cls]:.4f}")

    print(f"\nMean Average Precision (mAP): {mean_ap:.4f}")

    print("=" * 70)

    return {
        "pixel_accuracy": pixel_acc,
        "mean_iou": mean_iou,
        "mean_dice": mean_dice,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "mean_ap": mean_ap,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate OMR Model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset")
    parser.add_argument(
        "--dataset_type", type=str, default="ds2", choices=["cvc", "ds2"], help="Dataset type"
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
