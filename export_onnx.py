#!/usr/bin/env python3
"""
Export the trained PyTorch CNN checkpoint to ONNX for browser inference.

Example:
  python3 export_onnx.py \
    --checkpoint models/cnn_archive_fast/best_cnn_ai_real.pt \
    --output web/model.onnx \
    --metadata web/model-metadata.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from train_cnn_classifier import SmallCNN


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SmallCNN checkpoint to ONNX.")
    parser.add_argument("--checkpoint", default="models/cnn_archive_fast/best_cnn_ai_real.pt")
    parser.add_argument("--output", default="web/model.onnx")
    parser.add_argument("--metadata", default="web/model-metadata.json")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    metadata_path = Path(args.metadata).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    class_names = checkpoint["class_names"]
    image_size = int(checkpoint["image_size"])

    model = SmallCNN(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        external_data=False,
    )

    metadata = {
        "class_names": class_names,
        "image_size": image_size,
        "input_name": "image",
        "output_name": "logits",
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "ai_class_names": ["FAKE", "AiArtData"],
        "source_checkpoint": str(checkpoint_path),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Exported ONNX model: {output_path}")
    print(f"Exported metadata: {metadata_path}")


if __name__ == "__main__":
    main()
