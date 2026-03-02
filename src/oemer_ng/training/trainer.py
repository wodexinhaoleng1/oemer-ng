"""
Training utilities for OMR models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Callable
from pathlib import Path
from tqdm import tqdm
import time


class Trainer:
    """
    Trainer class for OMR models.

    Handles training loop, validation, checkpointing, AMP, gradient accumulation, and logging.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        device: Optional[str] = None,
        checkpoint_dir: str = "./checkpoints",
        log_interval: int = 10,
        use_amp: bool = True,
        accumulation_steps: int = 1,
        metric_fn: Optional[Callable[[torch.Tensor, torch.Tensor], Dict[str, float]]] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer (defaults to Adam)
            criterion: Loss function (defaults to CrossEntropyLoss)
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            log_interval: Logging interval in batches
            use_amp: Whether to use Automatic Mixed Precision (AMP)
            accumulation_steps: Number of steps for gradient accumulation
            metric_fn: Custom metric function for OMR (e.g., Symbol Error Rate).
                       Should take (output, target) and return a dict of metrics.
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Set optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        else:
            self.optimizer = optimizer

        # Set criterion
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval

        # Mixed Precision (AMP) & Gradient Accumulation
        self.use_amp = use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(device=self.device.type, enabled=self.use_amp)
        self.accumulation_steps = max(1, accumulation_steps)

        # Custom Metrics (OMR specific like SER, Edit Distance)
        self.metric_fn = metric_fn

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.history = {"train_loss": [], "val_loss": []}

    def _compute_metrics(self, output: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Helper to compute metrics based on metric_fn or fallback to simple accuracy."""
        if self.metric_fn is not None:
            return self.metric_fn(output, target)

        # Fallback for simple classification/sequence (fixes the dim=1 bug)
        pred = output.argmax(dim=-1)
        if target.dim() > pred.dim():
            target_indices = target.argmax(dim=-1)
        else:
            target_indices = target

        correct = pred.eq(target_indices).sum().item()
        total = target_indices.numel()
        return {"accuracy": 100.0 * correct / max(total, 1), "correct": correct, "total": total}

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch with AMP and Gradient Accumulation.

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0
        epoch_metrics = {}

        # Variables for default accuracy fallback
        total_correct = 0
        total_samples = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        self.optimizer.zero_grad()

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            # Forward pass with AMP
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                output = self.model(data)
                loss = self.criterion(output, target)
                # Normalize loss for accumulation
                loss = loss / self.accumulation_steps

            # Check for NaN loss
            if torch.isnan(loss):
                print(
                    f"\nWarning: NaN loss detected at epoch {self.current_epoch + 1}, batch {batch_idx}"
                )
                print("Skipping this batch...")
                self.optimizer.zero_grad()
                continue

            # Backward pass with scaler
            self.scaler.scale(loss).backward()

            # Gradient accumulation step
            if ((batch_idx + 1) % self.accumulation_steps == 0) or (
                (batch_idx + 1) == len(self.train_loader)
            ):
                # Unscale before gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Optimizer step and scaler update
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            # Statistics (multiply back by accumulation_steps for accurate logging)
            actual_loss = loss.item() * self.accumulation_steps
            total_loss += actual_loss
            self.global_step += 1

            # Compute metrics
            metrics = self._compute_metrics(output, target)
            for k, v in metrics.items():
                if k not in ["correct", "total"]:
                    epoch_metrics[k] = epoch_metrics.get(k, 0) + v
            if "correct" in metrics:
                total_correct += metrics["correct"]
                total_samples += metrics["total"]

            # Logging
            if batch_idx % self.log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                postfix_dict = {"loss": f"{avg_loss:.4f}"}

                if self.metric_fn is not None:
                    for k in epoch_metrics:
                        postfix_dict[k] = f"{epoch_metrics[k] / (batch_idx + 1):.4f}"
                else:
                    acc = 100.0 * total_correct / max(total_samples, 1)
                    postfix_dict["acc"] = f"{acc:.2f}%"

                pbar.set_postfix(postfix_dict)

        # Finalize epoch metrics
        result = {"loss": total_loss / len(self.train_loader)}
        if self.metric_fn is not None:
            for k in epoch_metrics:
                result[k] = epoch_metrics[k] / len(self.train_loader)
        else:
            result["accuracy"] = 100.0 * total_correct / max(total_samples, 1)

        return result

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate the model with AMP.

        Returns:
            Dictionary with validation metrics
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0
        epoch_metrics = {}
        total_correct = 0
        total_samples = 0

        for data, target in tqdm(self.val_loader, desc="Validation"):
            data, target = data.to(self.device), target.to(self.device)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                output = self.model(data)
                loss = self.criterion(output, target)

            total_loss += loss.item()

            # Compute metrics
            metrics = self._compute_metrics(output, target)
            for k, v in metrics.items():
                if k not in ["correct", "total"]:
                    epoch_metrics[k] = epoch_metrics.get(k, 0) + v
            if "correct" in metrics:
                total_correct += metrics["correct"]
                total_samples += metrics["total"]

        result = {"loss": total_loss / len(self.val_loader)}
        if self.metric_fn is not None:
            for k in epoch_metrics:
                result[k] = epoch_metrics[k] / len(self.val_loader)
        else:
            result["accuracy"] = 100.0 * total_correct / max(total_samples, 1)

        return result

    def save_checkpoint(self, filename: str = "checkpoint.pth", is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch + 1,  # Save NEXT epoch to resume from
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }

        # Only save scaler state if AMP is enabled
        if self.use_amp:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scaler state if available and AMP is enabled
        if "scaler_state_dict" in checkpoint:
            if self.use_amp:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
                print(f"  Loaded scaler state from checkpoint")
            else:
                print(f"  Warning: Checkpoint has scaler state but AMP is disabled")
        elif self.use_amp:
            print(f"  Warning: AMP enabled but no scaler state in checkpoint (starting fresh)")

        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.history = checkpoint["history"]

        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resuming training from epoch {self.current_epoch}")

    def train(
        self,
        num_epochs: int,
        scheduler: Optional[Any] = None,
        early_stopping_patience: Optional[int] = None,
    ):
        """
        Train the model for multiple epochs.

        Args:
            num_epochs: Additional number of epochs to train (from current_epoch)
            scheduler: Learning rate scheduler
            early_stopping_patience: Patience for early stopping
        """
        start_time = time.time()
        patience_counter = 0

        # FIXED: target_epochs calculates exactly when to stop, enabling smooth resuming
        target_epochs = self.current_epoch + num_epochs

        for epoch in range(self.current_epoch, target_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()
            self.history["train_loss"].append(train_metrics["loss"])

            # Dynamically save metrics to history
            for k, v in train_metrics.items():
                if k != "loss":
                    hist_key = f"train_{k}"
                    if hist_key not in self.history:
                        self.history[hist_key] = []
                    self.history[hist_key].append(v)

            print(f"\nEpoch {epoch + 1}/{target_epochs}")
            train_log = f"Train Loss: {train_metrics['loss']:.4f}"
            for k, v in train_metrics.items():
                if k != "loss":
                    train_log += f", Train {k.capitalize()}: {v:.4f}"
            print(train_log)

            # Validate
            if self.val_loader is not None:
                val_metrics = self.validate()
                self.history["val_loss"].append(val_metrics["loss"])

                val_log = f"Val Loss: {val_metrics['loss']:.4f}"
                for k, v in val_metrics.items():
                    if k != "loss":
                        hist_key = f"val_{k}"
                        if hist_key not in self.history:
                            self.history[hist_key] = []
                        self.history[hist_key].append(v)
                        val_log += f", Val {k.capitalize()}: {v:.4f}"
                print(val_log)

                # Check if best model
                is_best = val_metrics["loss"] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics["loss"]
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Save checkpoint
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth", is_best)

                # Early stopping
                if early_stopping_patience and patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
            else:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth")

            # Learning rate scheduling
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(
                        val_metrics["loss"] if self.val_loader else train_metrics["loss"]
                    )
                else:
                    scheduler.step()

            print(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time / 60:.2f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
