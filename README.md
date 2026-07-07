# AI Image Classifier

CNN-based image classifier for distinguishing `AiArtData` images from `RealArt` images.

## Dataset Layout

Place the image dataset in this structure:

```text
ai images/
  AiArtData/
    ...
  RealArt/
    ...
```

The current local dataset was stored outside this repository at:

```text
/Users/yunsung5387/Downloads/ai images
```

Raw images and generated gallery outputs are intentionally ignored by git.

## Install

```bash
pip install -r requirements.txt
```

## Visualize Images

```bash
python3 visualize_ai_images.py --dir "/Users/yunsung5387/Downloads/ai images" --recursive --cols 5 --rows 3 --save ./gallery_output
```

## Train CNN

```bash
python3 train_cnn_classifier.py --data-dir "/Users/yunsung5387/Downloads/ai images" --epochs 8 --batch-size 32 --output-dir models/cnn_ai_real
```

The baseline training run produced:

```text
Images used: 973
Train / Val / Test: 681 / 146 / 146
Best validation accuracy: 66.44%
Test accuracy: 66.44%
```

## Train From Archive Dataset

For an archive dataset with this structure:

```text
archive/
  train/
    FAKE/
    REAL/
  test/
    FAKE/
    REAL/
```

use:

```bash
python3 train_cnn_classifier.py \
  --data-dir data/archive \
  --train-dir data/archive/train \
  --test-dir data/archive/test \
  --output-dir models/cnn_archive_fast \
  --epochs 6 \
  --batch-size 64 \
  --image-size 128 \
  --max-per-class 2000 \
  --keep-duplicates \
  --skip-verify \
  --device cpu
```

The archive-based run produced:

```text
Classes: FAKE, REAL
Train / Val / Test: 3396 / 600 / 4000
Best validation accuracy: 87.33%
Test accuracy: 85.30%
FAKE f1: 86.35%
REAL f1: 84.07%
```

## Predict

Single image:

```bash
python3 predict_ai_real.py --image "/path/to/image.jpg" --model models/cnn_ai_real/best_cnn_ai_real.pt
```

Folder:

```bash
python3 predict_ai_real.py --image "/path/to/folder" --recursive --output-csv predictions.csv
```

## Files

```text
visualize_ai_images.py       Build image gallery pages.
train_cnn_classifier.py      Train the CNN model.
predict_ai_real.py           Run inference with the trained model.
models/cnn_ai_real/          Saved baseline model and metrics.
models/cnn_archive_fast/     Saved archive-based model and metrics.
```
