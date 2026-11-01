# PulmoScan

**A chest X-ray tuberculosis classification project built during the CDIS, IIT Kanpur AI/ML hackathon.**

PulmoScan explores transfer learning with an ImageNet-pretrained EfficientNetB3 model to classify X-ray images as normal (`0`) or TB-positive (`1`). The repository contains the original Colab notebooks and a cleaned, reproducible Python pipeline for preparing data, training, evaluating and generating a Kaggle-format submission.

> **Project status:** The dataset copy is present locally but was incomplete at the latest check; its folder is ignored by Git. The original notebook has training code but no successful saved model, test prediction or submission; one empty cell retains a stale `KeyError: 'class'` output. The cleaned pipeline is provided to make those steps reproducible once the dataset copy is complete. No competition submission is recorded for the connected Kaggle account, and no PulmoScan performance score is claimed.

## What’s here

- EfficientNetB3 transfer learning with a frozen-feature phase and optional fine-tuning.
- A stratified train/validation split and validation metrics (accuracy, precision, recall and F1).
- Test prediction output in `ID,Target` format.
- Data validation, setup instructions, a model card and preserved original notebooks.

## Quick start

Use Python 3.10 or 3.11 in Linux, WSL2, or Google Colab. TensorFlow 2.16 does not support native Windows training; a Linux or Colab GPU runtime is recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

Place the competition files under `TB_Dataset/` as described in [the data guide](docs/data.md), then run:

```bash
python -m pulmoscan train --data-dir TB_Dataset --output-dir artifacts/run-001
python -m pulmoscan predict --model artifacts/run-001/best_model.keras --data-dir TB_Dataset --output artifacts/run-001/submission.csv
```

If using WSL2, keep the dataset in the repository folder mounted under `/mnt/<drive>/...`, or pass its absolute path to `--data-dir`.

## Dataset layout

```text
TB_Dataset/
├── TB_train.csv
├── Training_Dataset_TB/
│   ├── TB_train_0001.png
│   └── ...
└── Test_Dataset_TB/
    ├── TB_test_0001.png
    └── ...
```

The dataset is not included in Git. See [data handling and validation](docs/data.md) before running the pipeline.

## Method at a glance

1. Read `ID,Target` labels and validate each training image path and binary label.
2. Make a reproducible, stratified 80/20 holdout split.
3. Train a binary sigmoid head on frozen EfficientNetB3 features.
4. Optionally fine-tune the final 100 backbone layers at a lower learning rate.
5. Evaluate the holdout predictions and save the best model.
6. Predict test labels and write `ID,Target` CSV.

The pipeline resizes images to 224 × 224 and uses TensorFlow EfficientNet preprocessing as defined by the installed framework. The parameters are configurable from the command line. Training and predictions should be run on the same compatible TensorFlow/Keras environment.

## Repository map

```text
.
├── docs/                 # Data guide, competition notes and model card
├── notebooks/original/   # Original Colab notebooks, preserved
├── src/pulmoscan/        # Reusable data, model, training and inference code
├── artifacts/            # Local outputs (ignored by Git)
└── TB_Dataset/           # Local dataset (ignored by Git)
```

## Competition context

The supplied notes identify the event as **Tuberculosis Detection from Chest X-Rays**, hosted by CDIS, IIT Kanpur. The Kaggle page is [tuberculosis-detection-from-chest-x-rays](https://www.kaggle.com/competitions/tuberculosis-detection-from-chest-x-rays). Kaggle CLI access was checked while preparing this repository: the competition leaderboard is accessible, but the account has no recorded submissions. The currently listed competition download file is only `nothing.txt`, so dataset counts and submission rules in the original notes have not been independently confirmed here. See [competition notes and provenance](docs/competition.md).

## Limitations

This is a hackathon/portfolio project, not a medical device. Predictions are not a diagnosis and must not guide clinical decisions. Image-level random splits may overestimate generalization if multiple images belong to the same patient; patient identifiers were not available in the project materials. Model performance is unknown until the dataset is added and the pipeline is run.

## Acknowledgements

Competition attribution and initial modeling approach are based on the materials supplied with this project. EfficientNetB3 pretrained weights are provided by TensorFlow/Keras Applications and are subject to their upstream terms.
