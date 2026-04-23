# SUPERCOMPUTING_PROJECT 

This project trains and outputs benchmarks on 3 convolutional neural network approaches for the [Fruits-262](https://www.kaggle.com/datasets/aelchimminut/fruits262) dataset. The pipeline begins by downloading the dataset, sequentially trains three different CNN architectures ( AlexNet replication from the dataset authors, a modified AlexNet with larger image resolutions, and a fine-tuned ResNet50). Lastly, it generates a comparison of all 3 models.


## Pipeline 

```
pipeline.sh
  │
  ├─ 00_download_data.sh      Download & extract Fruits-262 from Kaggle
  │
  ├─ 01_train_cnn.slurm       Train 3 models sequentially:
  │     ├─ alexnet             CNN Derived from kaggle paper  (52×64,  200 epochs)
  │     ├─ alexnet_bn          Improved paper CNN with larger image resolution   (104×128, 150 epochs)
  │     └─ resnet50            Transfer learning derived from documentation (tbd)  (224×224,  50 epochs)
  │
  └─ 02_compare_models.py     Aggregated results & comparison table
```

Each model run returns `test_results.json` and `training_log.csv` and the comparison script uses the outputs from model runs to produce a summary table and `model_comparison.csv`.

## Structure

```
SUPERCOMPUTING_PROJECT/
├── pipeline.sh                 # SLURM Pipeline
├── output/
│   ├── model_comparison.csv
│   ├── alexnet/
│   │   ├── test_results.json
│   │   └── training_log.csv
│   ├── alexnet_bn/
│   │   ├── test_results.json
│   │   └── training_log.csv
│   └── resnet50/
│       ├── test_results.json
│       └── training_log.csv
├── scripts/
│   ├── 00_download_data.sh     # Downloads Fruits-262 from Kaggle API
│   ├── 01_train_cnn.py         # Model training
│   ├── 01_train_cnn.slurm      # Slurm Script for training
│   └── 02_compare_models.py    # comparison
├── .gitignore
└── README.md
```

```
EXTERNAL_DIR (we used /sciclone/scr10/gzdata440/)
└── fruitsdata/
    ├── Fruit-262/
    └── output/
        ├── model_comparison.csv
        ├── alexnet/
        │   ├── best_model.pth
        │   ├── training_log.csv
        │   ├── test_results.json
        │   └── class_names.json
        ├── alexnet_bn/
        │   ├── best_model.pth
        │   ├── training_log.csv
        │   ├── test_results.json
        │   └── class_names.json
        └── resnet50/
            ├── best_model.pth
            ├── training_log.csv
            ├── test_results.json
            └── class_names.json
```

## Setup 

### Kaggle setup  
We use the kaggle python package in `00_download_data.sh` to download the dataset. 
The download commandused `kaggle datasets download aelchimminut/fruits262` might require Kaggle authentication on different machines, but it worked without authentication on the WM cluster. 
Follow instructions here for setting up kaggle on different machines: 
https://github.com/Kaggle/kaggle-cli 
https://www.kaggle.com/docs/api 

### Running Pipeline 

If you are running this pipeline on a different machine or HPC cluster, you must update the hardcoded paths before execution:

1. Change `SHARED_DIR` in `scripts/00_download_data.sh` and `scripts/01_train_cnn.slurm` to point to a folder on a drive with at least 15GB of free space
2. Change `DEFAULT_DATA_DIR` and `DEFAULT_OUTPUT_DIR` in `scripts/01_train_cnn.py` and `scripts/02_compare_models.py` to match the new `SHARED_DIR` paths.
3. Change `SCRIPT_DIR` and `REPORT_DIR` in `scripts/01_train_cnn.slurm` to point to cloned repository.
4. In `pipeline.sh`, update `#SBATCH --mail-user=` to your email address 
5. Run `mkdir logs` in root directory before submitting the SLURM job, as SLURM requires the output directory to exist.
6. Run `sbatch pipeline.sh`

## Scripts / Code 

## Script 1 - scripts/00_download_data.sh

Creates a conda environment (`kaggleenv`) with PyTorch and dependencies, then downloads and extracts the Fruits-262 dataset from Kaggle. The dataset is saved to `EXTERNAL_DIR/fruitsdata/Fruit-262/`.

```bash

#!/bin/bash

##Load WM HPC specific modules for working with conda environments
module load miniforge3
source /sciclone/apps/miniforge3-24.9.2-0/etc/profile.d/conda.sh

##Create Conda environment and install necessary packages
mamba create -n kaggleenv -y -c pytorch -c nvidia -c conda-forge \
    python=3.11 \
    pytorch torchvision torchaudio pytorch-cuda=12.1 \
    numpy pillow kaggle

##Activate the environment
conda activate kaggleenv

##Define necessary paths
SHARED_DIR="/sciclone/scr10/gzdata440"

##Remove any leftover fruit data from previous incomplete runs
rm -rf "${SHARED_DIR}/fruitsdata"
echo "removed fruitsdata folder"

##Recreate the folder structure for storing training data
mkdir -p "${SHARED_DIR}/fruitsdata"
echo "created fruitsdata folder"
cd "${SHARED_DIR}/fruitsdata"

##Download the data from github user 'aelchimminut', unzip it and remove leftover junk files
kaggle datasets download aelchimminut/fruits262

unzip fruits262.zip -d "${SHARED_DIR}/fruitsdata"

rm fruits262.zip

```

## Script 2 - scripts/01_train_cnn.py

Defines three CNN architectures (alexnet, alexnet_bn, resnet50) and handles training, validation, and test evaluation. Takes the Fruit-262 image directory as input and outputs `best_model.pth`, `training_log.csv`, `test_results.json`, and `class_names.json` to `EXTERNAL_DIR/fruitsdata/output/<model_name>/`.

```python

#!/usr/bin/env python3
"""
Fruits-262 CNN Classifier — Three Model Comparison
Based on: Minuț & Iftene, "Creating a Dataset and Models Based on CNNs
to Improve Fruit Classification" (SYNASC 2021)

Models:
  1. alexnet       — Paper replication (Table VII FTP), 52×64, from scratch
  2. alexnet_bn    — Improved: + BatchNorm + LR scheduling + AdamW, 104×128
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

DEFAULT_DATA_DIR = "/sciclone/scr10/gzdata440/fruitsdata/Fruit-262"
DEFAULT_OUTPUT_DIR = "/sciclone/scr10/gzdata440/fruitsdata/output"

# Per-model defaults: (height, width, epochs, batch_size, learning_rate)
MODEL_DEFAULTS = {
    "alexnet":    {"h": 64,  "w": 52,  "epochs": 200, "bs": 256, "lr": 2.5e-4},
    "alexnet_bn": {"h": 128, "w": 104, "epochs": 150, "bs": 128, "lr": 1e-3},
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

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
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
  alexnet_bn   + BatchNorm + LR scheduling + AdamW, 104x128
  resnet50     Transfer learning (ImageNet -> Fruits-262), 224x224
        """,
    )
    parser.add_argument("--model", type=str, required=True,
                        choices=["alexnet", "alexnet_bn", "resnet50"])
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override default epochs for chosen model")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training")
    args = parser.parse_args()

    # ── Resolve per-model defaults ───────────────────────────────────────
    defaults = MODEL_DEFAULTS[args.model]
    img_h      = defaults["h"]
    img_w      = defaults["w"]
    epochs     = args.epochs     or defaults["epochs"]
    batch_size = args.batch_size or defaults["bs"]
    lr         = args.lr         or defaults["lr"]

    # Output to model-specific subdirectory
    output_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(output_dir, exist_ok=True)

    # ── Device ───────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("=" * 60)
        print("WARNING: No GPU detected — training on CPU will be SLOW.")
        print("  Submit to a GPU partition instead:")
        print("    #SBATCH --partition=astral   (8x A30, 24GB)")
        print("    #SBATCH --gres=gpu:1")
        print("  Check available: sinfo -o '%P %G %N %l'")
        print("=" * 60)

    use_amp = torch.cuda.is_available()

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

```

## Script 3 - scripts/01_train_cnn.slurm

Orchestrates sequential training of all three models by calling `01_train_cnn.py` three times, then runs `02_compare_models.py` and copies deliverable outputs (`test_results.json`, `training_log.csv`, `model_comparison.csv`) to the project's `output/` folder.

```bash

#!/bin/bash
# ─── Environment Setup ───────────────────────────────────────────────────────
module load miniforge3
source /sciclone/apps/miniforge3-24.9.2-0/etc/profile.d/conda.sh

##Activate environmnt from download data script
conda activate kaggleenv

# ─── Paths ───────────────────────────────────────────────────────────────────
SHARED_DIR="/sciclone/scr10/gzdata440"
DATA_DIR="${SHARED_DIR}/fruitsdata/Fruit-262"
OUTPUT_DIR="${SHARED_DIR}/fruitsdata/output"
SCRIPT_DIR="$HOME/SUPERCOMPUTING_PROJECT/scripts"

# Quick GPU check
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'CPU only')"

# ─── Model 1: Paper Replication (AlexNet, 52×64) ────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  MODEL 1/3: alexnet (paper replication)"
echo "════════════════════════════════════════════════════════════"
echo ""

python "${SCRIPT_DIR}/01_train_cnn.py" \
    --model alexnet \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --epochs 200 \
    --workers 8

# ─── Model 2: Improved AlexNet (+ BatchNorm + LR scheduling, 104×128) ───────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  MODEL 2/3: alexnet_bn (improved)"
echo "════════════════════════════════════════════════════════════"
echo ""

python "${SCRIPT_DIR}/01_train_cnn.py" \
    --model alexnet_bn \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --epochs 150 \
    --workers 8

# ─── Model 3: Transfer Learning (ResNet50, 224×224) ─────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  MODEL 3/3: resnet50 (transfer learning)"
echo "════════════════════════════════════════════════════════════"
echo ""

python "${SCRIPT_DIR}/01_train_cnn.py" \
    --model resnet50 \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --epochs 50 \
    --workers 8

# ─── Comparison Summary ─────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  COMPARISON SUMMARY"
echo "════════════════════════════════════════════════════════════"
echo ""

python "${SCRIPT_DIR}/02_compare_models.py" \
    --output-dir "$OUTPUT_DIR"

#TEST OUTPUT CHANGE
REPORT_DIR="$HOME/SUPERCOMPUTING_PROJECT/output"
mkdir -p "$REPORT_DIR"
for model in alexnet alexnet_bn resnet50; do
    mkdir -p "$REPORT_DIR/$model"
    cp "$OUTPUT_DIR/$model/test_results.json"  "$REPORT_DIR/$model/" 2>/dev/null
    cp "$OUTPUT_DIR/$model/training_log.csv"   "$REPORT_DIR/$model/" 2>/dev/null
done
cp "$OUTPUT_DIR/model_comparison.csv" "$REPORT_DIR/" 2>/dev/null

echo ""
echo "All done: $(date)"


```

## Script 4 - scripts/02_compare_models.py

Reads `test_results.json` from each model's output directory, prints a formatted comparison table with top-1/5/10 accuracy alongside the paper's benchmarks, and saves `model_comparison.csv` to the output directory.

```python
#!/usr/bin/env python3
"""
Compare test results from all three Fruits-262 models.
Reads test_results.json from each model subdirectory and prints a
formatted comparison table including paper benchmarks.
"""

import os
import json
import argparse
#TEST OUTPUT CHANGE
import shutil

DEFAULT_OUTPUT_DIR = "/sciclone/scr10/gzdata440/fruitsdata/output" 
#mark for later

# Paper benchmarks from Table VIII (52x64 RGB model)
PAPER_BENCHMARKS = {
    "model": "paper (52x64 RGB)",
    "resolution": "52x64",
    "test_top1": 59.15,
    "test_top5": 80.40,
    "test_top10": 86.66,
    "total_params": "~5M",
    "training_hours": "N/A",
}


def main():
    parser = argparse.ArgumentParser(description="Compare Fruits-262 model results")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    #TEST OUTPUT CHANGE
    parser.add_argument("--report-dir", type=str, default=None)
    args = parser.parse_args()

    models = ["alexnet", "alexnet_bn", "resnet50"]
    results = []

    for model_name in models:
        path = os.path.join(args.output_dir, model_name, "test_results.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            results.append(data)
        else:
            print(f"  WARNING: {path} not found, skipping {model_name}")

    if not results:
        print("No results found. Have the models finished training?")
        return

    # ── Print comparison table ───────────────────────────────────────────
    print()
    print("=" * 90)
    print("  FRUITS-262 MODEL COMPARISON")
    print("=" * 90)
    print()

    header = (f"{'Model':<18} {'Resolution':<12} {'Top-1':>7} {'Top-5':>7} "
              f"{'Top-10':>7} {'Params':>12} {'Hours':>7} {'Optimizer':<10}")
    print(header)
    print("-" * 90)

    # Paper benchmark row
    p = PAPER_BENCHMARKS
    print(f"{'paper (Table VIII)':<18} {'52x64':<12} {p['test_top1']:>6.2f}% "
          f"{p['test_top5']:>6.2f}% {p['test_top10']:>6.2f}% "
          f"{'~5M':>12} {'N/A':>7} {'Adam':<10}")
    print("-" * 90)

    # Our models
    for r in results:
        params_str = f"{r['total_params']:,}" if isinstance(r['total_params'], int) else str(r['total_params'])
        hours_str = f"{r['training_hours']:.2f}" if isinstance(r['training_hours'], (int, float)) else str(r['training_hours'])
        opt = r.get('optimizer', 'N/A')

        print(f"{r['model']:<18} {r['resolution']:<12} {r['test_top1']:>6.2f}% "
              f"{r['test_top5']:>6.2f}% {r['test_top10']:>6.2f}% "
              f"{params_str:>12} {hours_str:>7} {opt:<10}")

    print("-" * 90)
    print()

    # ── Highlights ───────────────────────────────────────────────────────
    if len(results) >= 2:
        best = max(results, key=lambda r: r["test_top1"])
        worst = min(results, key=lambda r: r["test_top1"])

        print(f"  Best model:   {best['model']} ({best['test_top1']:.2f}% top-1)")
        print(f"  Worst model:  {worst['model']} ({worst['test_top1']:.2f}% top-1)")

        # Compare to paper
        paper_top1 = PAPER_BENCHMARKS["test_top1"]
        for r in results:
            delta = r["test_top1"] - paper_top1
            direction = "above" if delta > 0 else "below"
            print(f"  {r['model']}: {abs(delta):.2f}% {direction} paper benchmark")

    print()

    # ── Save comparison CSV ──────────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, "model_comparison.csv")
    with open(csv_path, "w") as f:
        f.write("model,resolution,test_top1,test_top5,test_top10,"
                "params,hours,optimizer,scheduler,epochs,batch_size\n")
        for r in results:
            f.write(f"{r['model']},{r['resolution']},{r['test_top1']:.4f},"
                    f"{r['test_top5']:.4f},{r['test_top10']:.4f},"
                    f"{r['total_params']},{r['training_hours']:.4f},"
                    f"{r.get('optimizer','')},{r.get('scheduler','')},{r['epochs_trained']},"
                    f"{r['batch_size']}\n")

    print(f"  Comparison CSV saved to: {csv_path}")

    #TEST OUTPUT CHANGE
    if args.report_dir:
        os.makedirs(args.report_dir, exist_ok=True)
        shutil.copy2(csv_path, os.path.join(args.report_dir, "model_comparison.csv"))
        print(f"  Also copied to: {args.report_dir}")

    print()


if __name__ == "__main__":
    main()
```

## Script 5 - pipeline.sh

SLURM entry point that runs the full pipeline end-to-end. Submits with `sbatch pipeline.sh` and sequentially calls `00_download_data.sh` then `01_train_cnn.slurm`. SLURM logs are saved to `./logs/`.

```bash

#!/bin/bash
#SBATCH --job-name=fruit_cnn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00          # 24 hours (3 models sequentially)
#SBATCH --mem=64G
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=mrmellors@wm.edu
#SBATCH -o ./logs/fruit_cnn_%j.out
#SBATCH -e ./logs/fruit_cnn_%j.err
#SBATCH --gpus=2

##Download the data from kaggle user aelchimminut and create a conda environment with these packages: pytorch torchvision torchaudio numpy pillow kaggle, see .yml file for more specific information
./scripts/00_download_data.sh

##Train 3 separate CNN models on the downloaded fruits data, alexnet, alexnet_bn, resnet50. Also calculates each models accuracy in terms of classifying groups and compares the models with different metrics
##See 01_train_cnn.py and readme for additional explanation
./scripts/01_train_cnn.slurm

```

### Results
in our outputs we found 


## Works Cited

Minuț, M.-D. (2021). Fruits-262 [Data set]. Kaggle. https://www.kaggle.com/datasets/aelchimminut/fruits262

Minuț, M.-D., & Iftene, A. (2021). Creating a dataset and models based on convolutional neural networks to improve fruit classification. In 2021 23rd International Symposium on Symbolic and Numeric Algorithms for Scientific Computing (SYNASC) (pp. 155–162). IEEE. https://doi.org/10.1109/SYNASC54541.2021.00035 

