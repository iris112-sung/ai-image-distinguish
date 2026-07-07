#!/usr/bin/env python3
"""
Visualize local AI image files in a folder.

Usage:
  python visualize_ai_images.py --dir "/path/to/ai images"

Optional:
  --recursive   search subfolders
  --cols        number of columns in each page/grid (default: 4)
  --thumb       thumbnail edge size for display (default: 256)
  --max         maximum images to load (default: 0 = no limit)
  --save        save each page as PNG and quit (default: show in popup window)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Tuple

from PIL import Image
import matplotlib.pyplot as plt


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".tif"}


def find_images(root: Path, recursive: bool) -> Iterable[Path]:
    pattern = "**/*" if recursive else "*"
    for p in root.glob(pattern):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def chunked(items: List[Path], size: int) -> Iterable[List[Path]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def make_grid(paths: List[Path], cols: int, thumb: int, page_idx: int, save_dir: Path | None) -> None:
    rows = math.ceil(len(paths) / cols)
    fig_w = max(6, cols * 3.5)
    fig_h = max(4, rows * 3.5)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes = axes.ravel()

    for ax in axes:
        ax.axis("off")

    for ax, img_path in zip(axes, paths):
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img.thumbnail((thumb, thumb))
        ax.imshow(img)
        ax.set_title(img_path.name, fontsize=8)

    unused_axes = axes[len(paths) :]
    for ax in unused_axes:
        ax.axis("off")

    title = f"AI image gallery - page {page_idx}"
    if len(paths) < cols * rows:
        title += f" ({len(paths)} images)"
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if save_dir is not None:
        out_file = save_dir / f"ai_image_page_{page_idx:02d}.png"
        fig.savefig(out_file, dpi=200)
        print(f"[saved] {out_file}")
        plt.close(fig)
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize local AI image files")
    parser.add_argument(
        "--dir",
        type=str,
        default=".",
        help="Directory that contains AI image files",
    )
    parser.add_argument("--recursive", action="store_true", help="Search subdirectories")
    parser.add_argument("--cols", type=int, default=4, help="Number of columns")
    parser.add_argument("--thumb", type=int, default=256, help="Thumbnail max size")
    parser.add_argument("--rows", type=int, default=4, help="Images per row group (rows per page)")
    parser.add_argument("--max", type=int, default=0, help="Max number of images (0=all)")
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Save pages into this folder instead of opening interactive window",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Directory not found: {root}")

    paths = sorted(find_images(root, args.recursive))
    if args.max and args.max > 0:
        paths = paths[: args.max]

    if not paths:
        print(f"No image file found under: {root}")
        return

    save_dir = Path(args.save).expanduser().resolve() if args.save else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    page_size = max(1, args.cols * args.rows)
    for i, page in enumerate(chunked(paths, page_size), start=1):
        make_grid(page, cols=args.cols, thumb=args.thumb, page_idx=i, save_dir=save_dir)


if __name__ == "__main__":
    main()
