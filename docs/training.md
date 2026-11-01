# Training and inference guide

This guide describes the maintained Python pipeline. It is not a claim that the original hackathon notebook completed a full run.

## Environment

The package targets Python 3.10 or 3.11 and TensorFlow 2.16.2. Use Linux, WSL2, or Google Colab; native Windows is not supported by this TensorFlow version. Training can use a GPU, but no GPU is required by the code. The first EfficientNetB3 build may download ImageNet weights.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

## Train and evaluate

```bash
python -m pulmoscan train \
  --data-dir TB_Dataset \
  --output-dir artifacts/run-001
```

Override paths when the folder names differ:

```bash
python -m pulmoscan train \
  --data-dir /path/to/competition \
  --train-dir /path/to/competition/Training_Dataset_TB \
  --labels /path/to/competition/TB_train.csv \
  --output-dir artifacts/experiment-a
```

Before TensorFlow starts training, the loader checks required columns, nonempty labels, unique IDs, binary targets, both classes and the existence of every labeled image. It then creates a deterministic stratified 80/20 split with seed 42. The split is at image level because patient identifiers are unavailable.

## Two training phases

### 1. Train the classification head

EfficientNetB3 starts with ImageNet weights and a frozen backbone. The new layers are trained with Adam at `1e-4` using binary cross-entropy.

### 2. Fine-tune the backbone

By default, the last 100 backbone layers are made trainable while batch-normalization layers stay frozen. Training continues with Adam at `1e-5`. Set `--epochs-fine 0` to stop after the head phase.

Each phase monitors validation loss, saves a best checkpoint and uses early stopping with patience 3. When both phases run, their best validation losses are compared and the lower-loss checkpoint is saved as `best_model.keras`.

The label distribution observed in the local CSV is imbalanced (2,450 normal vs. 490 TB-positive). The current implementation does not apply class weights, oversampling or probability calibration. Compare class-sensitive metrics and confusion matrices when a complete run becomes available; do not treat accuracy alone as sufficient evidence.

## Main options

| CLI flag | Default | Meaning |
|---|---:|---|
| `--data-dir` | `TB_Dataset` | Directory containing labels and image folders |
| `--train-dir` | `<data-dir>/Training_Dataset_TB` | Training PNG directory |
| `--labels` | `<data-dir>/TB_train.csv` | Label CSV |
| `--image-size` | `224` | Square image dimension after resize |
| `--batch-size` | `32` | Images per training batch |
| `--seed` | `42` | Split/training random seed |
| `--validation-fraction` | `0.2` | Holdout share of labeled data |
| `--epochs-head` | `10` | Maximum head-training epochs |
| `--epochs-fine` | `10` | Maximum fine-tuning epochs |
| `--fine-tune-layers` | `100` | Final backbone layers to unfreeze |
| `--learning-rate-head` | `1e-4` | Head-stage learning rate |
| `--learning-rate-fine` | `1e-5` | Fine-tuning learning rate |
| `--dropout` | `0.5` | Dropout rate in classification head |
| `--threshold` | `0.5` | Probability threshold for predicted class 1 |

Run `python -m pulmoscan train --help` for the complete flag list.

## Metrics and artifacts

Validation predictions are generated in deterministic order. Accuracy, precision, recall and F1 are computed at the fixed threshold. Training writes:

```text
artifacts/run-001/
├── best_model.keras
├── validation_metrics.json
└── validation_predictions.csv
```

The JSON includes the threshold, validation image count, metrics, class counts for both splits, configuration and TensorFlow version. Validation probabilities and labels are available per image in the CSV.

## Generate test predictions

```bash
python -m pulmoscan predict \
  --model artifacts/run-001/best_model.keras \
  --data-dir TB_Dataset \
  --output artifacts/run-001/submission.csv
```

Or supply the test path directly with `--test-dir`. Test images are read in sorted filename order. The submission file contains only `ID,Target`; probabilities are not added to the competition-format file.

## Reproducibility notes

- Seed initialization uses `tf.keras.utils.set_random_seed` and the same seed controls the stratified split and training shuffle.
- GPU kernels and framework/platform differences can still produce small numerical differences.
- Record any non-default CLI flags with the resulting metrics file when sharing a run.
- Do not compare scores from different splits, thresholds or label versions as if they were the same experiment.
