#!/usr/bin/env python3
"""
Fruits-262 CNN Classifier — Three Model Comparison
Based on: Minuț & Iftene, "Creating a Dataset and Models Based on CNNs
to Improve Fruit Classification" (SYNASC 2021)

Models:
  1. alexnet       — Paper replication (Table VII FTP), 52×64, from scratch
  2. alexnet_bn    — Improved: + BatchNorm + LR scheduling + AdamW, 52x64 safe rerun default
  3. resnet50      — Transfer learning from ImageNet, 224×224

Usage:
  python 01_train_cnn.py --model alexnet       [--epochs 200]
  python 01_train_cnn.py --model alexnet_bn    [--epochs 150]
  python 01_train_cnn.py --model resnet50      [--epochs 50]
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration defaults
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_DATA_DIR = "/sciclone/scr10/gzdata440/fruitsdata2/Fruit-262"
DEFAULT_OUTPUT_DIR = "/sciclone/scr10/gzdata440/fruitsdata2/output"

# Per-model defaults: (height, width, epochs, batch_size, learning_rate)
MODEL_DEFAULTS = {
    "alexnet":    {"h": 64,  "w": 52,  "epochs": 200, "bs": 256, "lr": 2.5e-4},
    "alexnet_bn": {"h": 64,  "w": 52,  "epochs": 150, "bs": 256, "lr": 1e-3},
    "resnet50":   {"h": 224, "w": 224, "epochs": 50,  "bs": 64,  "lr": 1e-4},
}

NUM_WORKERS = 8


# ═══════════════════════════════════════════════════════════════════════════════
# Model 1: Paper Replication — AlexNet FTP (Table VII, SYNASC 2021)
# ═══════════════════════════════════════════════════════════════════════════════

class FruitAlexNet(nn.Module):
    """
    Exact replication of the paper's Final Training Pipeline (Table VII).
    Designed for 52x64x3 RGB input.

    Architecture:
      64 Conv2d 11x11 pad -> Pool -> 128 Conv2d 5x5 pad -> Pool ->
      256 Conv2d 3x3 -> 256 Conv2d 3x3 -> 256 Conv2d 3x3 -> Pool ->
      Flatten -> Dropout(0.5) -> Dense(1024) -> Dropout(0.5) ->
      Dense(1024) -> Dense(num_classes)
    """

    def __init__(self, num_classes=262):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: paper specifies 64 filters, 11x11, padding
            nn.Conv2d(3, 64, kernel_size=11, stride=1, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 128 filters, 5x5, padding
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3-5: three 256-filter 3x3 convs, no padding (per paper)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Dynamically compute flatten size for the target input resolution
        self._flat_size = self._get_flat_size(3, 64, 52)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self._flat_size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def _get_flat_size(self, c, h, w):
        dummy = torch.zeros(1, c, h, w)
        out = self.features(dummy)
        return out.view(1, -1).size(1)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# Model 2: Improved AlexNet — BatchNorm + LR Scheduling + AdamW
# ═══════════════════════════════════════════════════════════════════════════════

class FruitAlexNetBN(nn.Module):
    """
    Same AlexNet conv structure from the paper but with modern improvements:
      - BatchNorm2d after every conv layer (before ReLU)
      - BatchNorm1d after dense hidden layers
      - AdaptiveAvgPool2d before flatten — resolution-agnostic
      - Designed for higher resolution (104x128) since GPU makes it feasible
      - padding=1 on 3x3 convs to preserve spatial dims better

    Paired with:
      - AdamW optimizer (decoupled weight decay)
      - ReduceLROnPlateau scheduler
    """

    def __init__(self, num_classes=262):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Block 4
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Block 5
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            # AdaptiveAvgPool makes this resolution-agnostic
            # Output always: 256 x 4 x 3 regardless of input size
            nn.AdaptiveAvgPool2d((4, 3)),
        )

        # 256 * 4 * 3 = 3072 always
        self._flat_size = 256 * 4 * 3

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self._flat_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# Model 3: Transfer Learning — ResNet50 pretrained on ImageNet
# ═══════════════════════════════════════════════════════════════════════════════

class FruitResNet50(nn.Module):
    """
    ResNet50 pretrained on ImageNet, fine-tuned for Fruits-262.
      - Replaces final FC layer: 2048 -> num_classes
      - All layers trainable (full fine-tuning)
      - Standard 224x224 input with ImageNet normalization
      - Differential LR: backbone 10x lower than head
    """

    def __init__(self, num_classes=262):
        super().__init__()

        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Replace final classifier head
        in_features = self.backbone.fc.in_features  # 2048
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_data_loaders(data_dir, img_h, img_w, batch_size, num_workers,
                     use_imagenet_norm=False):
    """
    Load Fruits-262 with 70/20/10 train/val/test split (paper Section IV.A).
    Training set gets augmentation; val/test get only resize + tensor conversion.
    """

    # --- Build transforms ---------------------------------------------------
    if use_imagenet_norm:
        normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        train_transform = transforms.Compose([
            transforms.Resize((img_h, img_w)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
        ])
        eval_transform = transforms.Compose([
            transforms.Resize((img_h, img_w)),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        # From-scratch models: paper scales to [0,1] via ToTensor()
        train_transform = transforms.Compose([
            transforms.Resize((img_h, img_w)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])
        eval_transform = transforms.Compose([
            transforms.Resize((img_h, img_w)),
            transforms.ToTensor(),
        ])

    # --- Load full dataset to compute split indices -------------------------
    full_dataset = datasets.ImageFolder(data_dir)
    num_images = len(full_dataset)

    print(f"  Total images found:  {num_images}")
    print(f"  Number of classes:   {len(full_dataset.classes)}")

    # Reproducible 70/20/10 split (same seed = same split across all models)
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(num_images, generator=generator).tolist()

    n_train = int(0.7 * num_images)
    n_val = int(0.2 * num_images)

    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    print(f"  Split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")

    # Three dataset copies with different transforms, then subset
    train_ds = Subset(datasets.ImageFolder(data_dir, transform=train_transform), train_idx)
    val_ds   = Subset(datasets.ImageFolder(data_dir, transform=eval_transform), val_idx)
    test_ds  = Subset(datasets.ImageFolder(data_dir, transform=eval_transform), test_idx)

    pin = torch.cuda.is_available()

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin,
    }
    if num_workers > 0:
        loader_kwargs.update({
            "persistent_workers": True,
            "prefetch_factor": 4,
        })

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        drop_last=True, **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs,
    )

    return train_loader, val_loader, test_loader, full_dataset.classes


# ═══════════════════════════════════════════════════════════════════════════════
# Training & Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """Train for one epoch. Supports mixed precision via GradScaler."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 100 == 0:
            print(f"    Batch {batch_idx+1:4d}/{len(loader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Running Acc: {100.*correct/total:.2f}%")

    return running_loss / total, 100. * correct / total


def evaluate(model, loader, criterion, device, topk=(1, 5, 10)):
    """Evaluate with top-k accuracy (paper reports top-1, 5, 10 in Table VIII)."""
    model.eval()
    running_loss = 0.0
    topk_correct = {k: 0 for k in topk}
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            total += labels.size(0)

            maxk = max(topk)
            _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
            correct_matrix = pred.eq(labels.unsqueeze(1).expand_as(pred))

            for k in topk:
                topk_correct[k] += correct_matrix[:, :k].any(dim=1).sum().item()

    topk_acc = {k: 100. * topk_correct[k] / total for k in topk}
    return running_loss / total, topk_acc


# ═══════════════════════════════════════════════════════════════════════════════
# Model Builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_model(model_name, num_classes, device):
    """Instantiate model, optimizer, and scheduler."""

    if model_name == "alexnet":
        # ── Paper replication: Adam, no scheduler ────────────────────────
        model = FruitAlexNet(num_classes=num_classes).to(device)
        optimizer = optim.Adam(
            model.parameters(),
            lr=MODEL_DEFAULTS["alexnet"]["lr"],
        )
        scheduler = None

    elif model_name == "alexnet_bn":
        # ── Improved: AdamW + weight decay + ReduceLROnPlateau ───────────
        model = FruitAlexNetBN(num_classes=num_classes).to(device)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=MODEL_DEFAULTS["alexnet_bn"]["lr"],
            weight_decay=1e-4,
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10,
            min_lr=1e-6,
        )

    elif model_name == "resnet50":
        # ── Transfer learning: differential LR + cosine schedule ─────────
        model = FruitResNet50(num_classes=num_classes).to(device)

        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            if "backbone.fc" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        base_lr = MODEL_DEFAULTS["resnet50"]["lr"]
        optimizer = optim.AdamW([
            {"params": backbone_params, "lr": base_lr * 0.1},  # backbone: 1e-5
            {"params": head_params,     "lr": base_lr},         # head:     1e-4
        ], weight_decay=1e-4)

        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model, optimizer, scheduler


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Fruits-262 CNN — Paper Replication + Improvements + Transfer Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Models:
  alexnet      Paper replication (Table VII), 52x64, Adam, no scheduling
  alexnet_bn   + BatchNorm + LR scheduling + AdamW, 52x64 safe rerun default
  resnet50     Transfer learning (ImageNet -> Fruits-262), 224x224
        """,
    )
    parser.add_argument("--model", type=str, required=True,
                        choices=["alexnet", "alexnet_bn", "resnet50"])
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--img-h", type=int, default=None,
                        help="Override input image height")
    parser.add_argument("--img-w", type=int, default=None,
                        help="Override input image width")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override default epochs for chosen model")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--allow-cpu", action="store_true",
                        help="Allow CPU training instead of failing when CUDA is unavailable")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training")
    args = parser.parse_args()

    # ── Resolve per-model defaults ───────────────────────────────────────
    defaults = MODEL_DEFAULTS[args.model]
    img_h      = args.img_h or defaults["h"]
    img_w      = args.img_w or defaults["w"]
    epochs     = args.epochs     or defaults["epochs"]
    batch_size = args.batch_size or defaults["bs"]
    lr         = args.lr         or defaults["lr"]

    # Output to model-specific subdirectory
    output_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(output_dir, exist_ok=True)

    # ── Device ───────────────────────────────────────────────────────────
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        print("=" * 60)
        print("WARNING: No GPU detected - training on CPU will be SLOW.")
        print("  Submit to a GPU partition instead:")
        print("    #SBATCH --partition=astral   (8x A30, 24GB)")
        print("    #SBATCH --gres=gpu:1")
        print("  Check available: sinfo -o '%P %G %N %l'")
        print("=" * 60)
        if not args.allow_cpu:
            raise SystemExit(
                "CUDA is unavailable, so aborting before a slow CPU run. "
                "Pass --allow-cpu only if this is intentional."
            )

    use_amp = has_cuda

    # ── Print config ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Model:       {args.model}")
    print(f"  Resolution:  {img_w} x {img_h}")
    print(f"  Epochs:      {epochs}")
    print(f"  Batch size:  {batch_size}")
    print(f"  LR:          {lr}")
    print(f"  Workers:     {args.workers}")
    print(f"  Device:      {device}")
    print(f"  Mixed prec:  {use_amp}")
    print(f"  Output:      {output_dir}")
    print(f"  Data:        {args.data_dir}")
    print(f"{'='*60}\n")

    # ── Data ─────────────────────────────────────────────────────────────
    print("Loading dataset...")
    t0 = time.time()

    use_imagenet_norm = (args.model == "resnet50")
    train_loader, val_loader, test_loader, class_names = get_data_loaders(
        args.data_dir, img_h, img_w, batch_size, args.workers,
        use_imagenet_norm=use_imagenet_norm,
    )
    num_classes = len(class_names)
    print(f"  Loaded in {time.time()-t0:.1f}s\n")

    # Save class names for inference
    with open(os.path.join(output_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f)

    # ── Model, optimizer, scheduler ──────────────────────────────────────
    model, optimizer, scheduler = build_model(args.model, num_classes, device)

    if args.lr:
        for pg in optimizer.param_groups:
            pg["lr"] = args.lr

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    scheduler_name = type(scheduler).__name__ if scheduler else "None"

    print(f"Model: {args.model}")
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {train_params:,}")
    print(f"  Optimizer:        {type(optimizer).__name__}")
    print(f"  Scheduler:        {scheduler_name}\n")

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    criterion = nn.CrossEntropyLoss()

    # ── Resume from checkpoint ───────────────────────────────────────────
    start_epoch = 0
    best_val_acc = 0.0

    if args.resume:
        print(f"Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        if scheduler and ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler and ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        print(f"  Resumed at epoch {start_epoch}, best val: {best_val_acc:.2f}%\n")

    # ── CSV log ──────────────────────────────────────────────────────────
    log_path = os.path.join(output_dir, "training_log.csv")
    if start_epoch == 0:
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,train_acc,val_loss,val_top1,"
                    "val_top5,val_top10,lr,time_s\n")

    # ═════════════════════════════════════════════════════════════════════
    # Training Loop
    # ═════════════════════════════════════════════════════════════════════
    print(f"{'='*70}")
    print(f"Training {args.model}: epochs {start_epoch} -> {epochs-1}")
    print(f"{'='*70}\n")

    total_train_start = time.time()

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )

        val_loss, val_topk = evaluate(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start

        # Step scheduler
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        print(f"Epoch {epoch:3d}/{epochs-1} | "
              f"Train {train_acc:5.2f}% (loss {train_loss:.4f}) | "
              f"Val Top1 {val_topk[1]:5.2f}% Top5 {val_topk[5]:5.2f}% "
              f"Top10 {val_topk[10]:5.2f}% | "
              f"LR {current_lr:.2e} | {epoch_time:.0f}s")

        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{train_acc:.4f},"
                    f"{val_loss:.6f},{val_topk[1]:.4f},{val_topk[5]:.4f},"
                    f"{val_topk[10]:.4f},{current_lr:.8f},{epoch_time:.1f}\n")

        # Save best
        if val_topk[1] > best_val_acc:
            best_val_acc = val_topk[1]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "scaler_state_dict": scaler.state_dict() if scaler else None,
                "best_val_acc": best_val_acc,
                "num_classes": num_classes,
                "model_name": args.model,
            }, os.path.join(output_dir, "best_model.pth"))
            print(f"  -> New best model (val top-1: {best_val_acc:.2f}%)")

        # Periodic checkpoint
        if (epoch + 1) % 25 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "scaler_state_dict": scaler.state_dict() if scaler else None,
                "best_val_acc": best_val_acc,
                "num_classes": num_classes,
                "model_name": args.model,
            }, os.path.join(output_dir, f"checkpoint_ep{epoch:03d}.pth"))

    total_time = time.time() - total_train_start
    print(f"\nTraining complete: {total_time/3600:.2f} hours")

    # ═════════════════════════════════════════════════════════════════════
    # Final Test Evaluation
    # ═════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"Test evaluation: {args.model} (best checkpoint)")
    print(f"{'='*70}\n")

    best_path = os.path.join(output_dir, "best_model.pth")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        best_epoch = ckpt["epoch"]
    else:
        print("  WARNING: No best_model.pth found, using last epoch")
        best_epoch = epochs - 1

    test_loss, test_topk = evaluate(model, test_loader, criterion, device)

    print(f"  Model:           {args.model}")
    print(f"  Best epoch:      {best_epoch}")
    print(f"  Test Loss:       {test_loss:.4f}")
    print(f"  Top-1 Accuracy:  {test_topk[1]:.2f}%")
    print(f"  Top-5 Accuracy:  {test_topk[5]:.2f}%")
    print(f"  Top-10 Accuracy: {test_topk[10]:.2f}%")
    print(f"  Training time:   {total_time/3600:.2f} hours")

    # Paper comparison for alexnet replication
    if args.model == "alexnet":
        print(f"\n  --- Paper benchmarks (52x64 RGB, Table VIII) ---")
        print(f"    Paper Top-1:  59.15%    Ours: {test_topk[1]:.2f}%")
        print(f"    Paper Top-5:  80.40%    Ours: {test_topk[5]:.2f}%")
        print(f"    Paper Top-10: 86.66%    Ours: {test_topk[10]:.2f}%")

    # Save results JSON
    results = {
        "model": args.model,
        "resolution": f"{img_w}x{img_h}",
        "best_epoch": best_epoch,
        "test_loss": test_loss,
        "test_top1": test_topk[1],
        "test_top5": test_topk[5],
        "test_top10": test_topk[10],
        "best_val_acc": best_val_acc,
        "total_params": total_params,
        "trainable_params": train_params,
        "training_hours": total_time / 3600,
        "epochs_trained": epochs,
        "batch_size": batch_size,
        "initial_lr": lr,
        "optimizer": type(optimizer).__name__,
        "scheduler": scheduler_name,
    }

    with open(os.path.join(output_dir, "test_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to: {output_dir}")
    print("Done.\n")


if __name__ == "__main__":
    main()
