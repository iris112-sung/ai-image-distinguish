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
```
