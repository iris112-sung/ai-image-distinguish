#!/usr/bin/env python3
"""
Predict whether images are AiArtData or RealArt using a trained CNN checkpoint.

Example:
  python3 predict_ai_real.py --image "/path/to/image.jpg"
  python3 predict_ai_real.py --image "/path/to/folder" --recursive
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".tif"}


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


def pick_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def find_images(path: Path, recursive: bool):
    if path.is_file():
        if path.suffix.lower() in IMAGE_EXTS:
            yield path
        return
    pattern = "**/*" if recursive else "*"
    for p in sorted(path.glob(pattern)):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def build_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CNN AI-vs-real image predictions.")
    parser.add_argument("--image", required=True, help="Image file or image folder")
    parser.add_argument("--model", default="models/cnn_ai_real/best_cnn_ai_real.pt")
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    args = parser.parse_args()

    device = pick_device(args.device)
    checkpoint = torch.load(Path(args.model).expanduser().resolve(), map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]
    image_size = checkpoint["image_size"]

    model = SmallCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    tf = build_transform(image_size)

    rows = []
    for path in find_images(Path(args.image).expanduser().resolve(), recursive=args.recursive):
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            x = tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1).squeeze(0).detach().cpu().tolist()
        pred_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
        row = {
            "path": str(path),
            "prediction": class_names[pred_idx],
            "confidence": round(probs[pred_idx], 6),
        }
        for i, name in enumerate(class_names):
            row[f"prob_{name}"] = round(probs[i], 6)
        rows.append(row)
        print(f"{path.name}: {row['prediction']} ({row['confidence']:.4f})")

    if args.output_csv and rows:
        output = Path(args.output_csv).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved predictions: {output}")


if __name__ == "__main__":
    main()
