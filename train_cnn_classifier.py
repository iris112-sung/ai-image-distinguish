#!/usr/bin/env python3
"""
Train a CNN classifier for AiArtData vs RealArt.

Example:
  python3 train_cnn_classifier.py --data-dir "/Users/yunsung5387/Downloads/ai images" --epochs 8
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".tif"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_samples(
    data_dir: Path,
    keep_duplicates: bool,
    max_per_class: int = 0,
    class_names: Sequence[str] | None = None,
    verify_images: bool = True,
) -> Tuple[List[Tuple[Path, int]], List[str], Dict[str, int]]:
    if class_names is None:
        label_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
        class_names = [p.name for p in label_dirs]
    else:
        label_dirs = [data_dir / name for name in class_names]
    class_names = list(class_names)
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    samples: List[Tuple[Path, int]] = []
    seen_hashes: Dict[str, Path] = {}
    skipped_duplicates = 0
    skipped_unreadable = 0

    for label_dir in label_dirs:
        if not label_dir.exists():
            raise ValueError(f"Class folder not found: {label_dir}")
        label = class_to_idx[label_dir.name]
        class_paths: List[Path] = []
        for path in sorted(label_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                continue
            class_paths.append(path)
        if max_per_class and max_per_class > 0:
            random.shuffle(class_paths)
            class_paths = class_paths[:max_per_class]
        for path in class_paths:
            if verify_images:
                try:
                    with Image.open(path) as img:
                        img.verify()
                except Exception:
                    skipped_unreadable += 1
                    continue
            if not keep_duplicates:
                digest = sha256_file(path)
                if digest in seen_hashes:
                    skipped_duplicates += 1
                    continue
                seen_hashes[digest] = path
            samples.append((path, label))

    stats = {
        "found_classes": len(class_names),
        "kept_images": len(samples),
        "skipped_exact_duplicates": skipped_duplicates,
        "skipped_unreadable": skipped_unreadable,
    }
    return samples, class_names, stats


class ImagePathDataset(Dataset):
    def __init__(self, samples: Sequence[Tuple[Path, int]], transform=None) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_transforms(image_size: int):
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.12),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, eval_tf


def split_samples(
    samples: Sequence[Tuple[Path, int]], val_size: float, test_size: float, seed: int
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    labels = [label for _, label in samples]
    train_val, test = train_test_split(
        list(samples),
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    train_val_labels = [label for _, label in train_val]
    adjusted_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=adjusted_val,
        random_state=seed,
        stratify=train_val_labels,
    )
    return train, val, test


def make_loader(samples, transform, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        ImagePathDataset(samples, transform=transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(model, loader, criterion, optimizer, device: torch.device, train: bool):
    model.train(train)
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "preds": all_preds,
        "labels": all_labels,
    }


def write_split_csv(path: Path, splits: Dict[str, Sequence[Tuple[Path, int]]], class_names: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "label", "class_name", "path"])
        for split_name, samples in splits.items():
            for image_path, label in samples:
                writer.writerow([split_name, label, class_names[label], image_path])


def save_plots(history: List[Dict], output_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / "mplconfig"))
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(epochs, [h["train_accuracy"] for h in history], label="train")
    axes[1].plot(epochs, [h["val_accuracy"] for h in history], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CNN AI-vs-real image classifier.")
    parser.add_argument("--data-dir", default="/Users/yunsung5387/Downloads/ai images")
    parser.add_argument("--output-dir", default="models/cnn_ai_real")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--train-dir", default="", help="Optional explicit train folder with class subfolders")
    parser.add_argument("--test-dir", default="", help="Optional explicit test folder with class subfolders")
    parser.add_argument("--max-per-class", type=int, default=0, help="Limit images per class in each scanned folder")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--keep-duplicates", action="store_true")
    parser.add_argument("--skip-verify", action="store_true", help="Skip PIL image verification during scanning")
    args = parser.parse_args()

    seed_everything(args.seed)
    data_dir = Path(args.data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scan_root = Path(args.train_dir).expanduser().resolve() if args.train_dir else data_dir
    samples, class_names, scan_stats = scan_samples(
        scan_root,
        keep_duplicates=args.keep_duplicates,
        max_per_class=args.max_per_class,
        verify_images=not args.skip_verify,
    )
    if len(class_names) != 2:
        raise ValueError(f"Expected exactly 2 class folders, found {class_names}")
    if len(samples) < 20:
        raise ValueError(f"Not enough images to train: {len(samples)}")

    if args.test_dir:
        train_samples, val_samples, _unused = split_samples(
            samples, val_size=args.val_size, test_size=0.001, seed=args.seed
        )
        test_samples, _, test_scan_stats = scan_samples(
            Path(args.test_dir).expanduser().resolve(),
            keep_duplicates=args.keep_duplicates,
            max_per_class=args.max_per_class,
            class_names=class_names,
            verify_images=not args.skip_verify,
        )
        scan_stats["test_kept_images"] = test_scan_stats["kept_images"]
        scan_stats["test_skipped_exact_duplicates"] = test_scan_stats["skipped_exact_duplicates"]
        scan_stats["test_skipped_unreadable"] = test_scan_stats["skipped_unreadable"]
    else:
        train_samples, val_samples, test_samples = split_samples(
            samples, val_size=args.val_size, test_size=args.test_size, seed=args.seed
        )
    write_split_csv(
        output_dir / "splits.csv",
        {"train": train_samples, "val": val_samples, "test": test_samples},
        class_names,
    )

    train_tf, eval_tf = build_transforms(args.image_size)
    train_loader = make_loader(train_samples, train_tf, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_samples, eval_tf, args.batch_size, False, args.num_workers)
    test_loader = make_loader(test_samples, eval_tf, args.batch_size, False, args.num_workers)

    device = pick_device(args.device)
    model = SmallCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

    print(f"Data: {data_dir}")
    if args.train_dir:
        print(f"Train dir: {Path(args.train_dir).expanduser().resolve()}")
    if args.test_dir:
        print(f"Test dir: {Path(args.test_dir).expanduser().resolve()}")
    print(f"Classes: {class_names}")
    print(f"Images kept: {scan_stats['kept_images']} | duplicate skips: {scan_stats['skipped_exact_duplicates']}")
    print(f"Split: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")
    print(f"Device: {device}")

    best_val_acc = -1.0
    best_epoch = 0
    history = []
    best_path = output_dir / "best_cnn_ai_real.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_metrics["loss"])

        row = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 6),
            "train_accuracy": round(train_metrics["accuracy"], 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_accuracy": round(val_metrics["accuracy"], 6),
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_loss={row['train_loss']:.4f} train_acc={row['train_accuracy']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_acc={row['val_accuracy']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "image_size": args.image_size,
                    "architecture": "SmallCNN",
                    "best_epoch": best_epoch,
                    "best_val_accuracy": best_val_acc,
                    "args": vars(args),
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = run_epoch(model, test_loader, criterion, optimizer=None, device=device, train=False)
    report = classification_report(
        test_metrics["labels"],
        test_metrics["preds"],
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(test_metrics["labels"], test_metrics["preds"]).tolist()

    metrics = {
        "class_names": class_names,
        "scan_stats": scan_stats,
        "split_counts": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        },
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "classification_report": report,
        "confusion_matrix": matrix,
        "history": history,
        "model_path": str(best_path),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    save_plots(history, output_dir)

    print(f"Best epoch: {best_epoch} | best val accuracy: {best_val_acc:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f} | test loss: {test_metrics['loss']:.4f}")
    print(f"Saved model: {best_path}")
    print(f"Saved metrics: {output_dir / 'metrics.json'}")
    print(f"Saved curves: {output_dir / 'training_curves.png'}")


if __name__ == "__main__":
    main()
