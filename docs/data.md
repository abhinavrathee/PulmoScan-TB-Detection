# Dataset setup

## Expected files

The pipeline expects this local layout. The default commands match the `TB_Dataset/` folder used in this workspace:

```text
TB_Dataset/
├── TB_train.csv
├── Training_Dataset_TB/
│   └── TB_train_*.png
└── Test_Dataset_TB/
    └── TB_test_*.png
```

The label CSV must have columns `ID` and `Target`; `ID` is a filename relative to `Training_Dataset_TB/`, and `Target` must be `0` (normal) or `1` (TB-positive). Test images are discovered from the test directory and their filenames are retained in the generated submission.

Folder capitalization in the original notes and notebooks is inconsistent. Use the names above or pass `--train-dir`, `--test-dir` and `--labels` explicitly to the CLI.

### Workspace copy status

During repository setup on 2026-11-01, the workspace contained the full 2,940-row label CSV and 1,260 test PNGs. The training image folder changed from 207 to 1,309 PNGs between checks, leaving 1,631 CSV-referenced images missing at the latest check. This is a point-in-time observation while the dataset was being copied, not the expected final state. The training command checks that every labeled image exists and will stop with a missing-files error until the copy is complete.

## Getting the data

Use the files provided to your hackathon team or an authorized competition download. The Kaggle CLI can list/download competition files after you accept the competition rules:

```powershell
kaggle competitions files tuberculosis-detection-from-chest-x-rays
kaggle competitions download tuberculosis-detection-from-chest-x-rays -p TB_Dataset
```

The dataset can be large and may have redistribution restrictions. Check the competition terms and dataset license before adding it to a public repository. The locally copied dataset may be unzipped into the expected folders.

At repository setup, the Kaggle competition file listing returned only `nothing.txt`, not the dataset archive. The commands above are the standard Kaggle CLI workflow, but they may not retrieve the data for this event; keep using the authorized local copy if the competition page does not expose the files.

## Validation behavior

Before training, PulmoScan checks required columns, nonempty labels, binary target values, duplicate IDs and missing image files. The holdout split is stratified by label and seeded for repeatability. It is image-level because patient identifiers are not present in the supplied materials; if patient IDs become available, replace this with a patient-level split to reduce leakage risk.

## Output files

Training writes `best_model.keras`, `validation_metrics.json` and `validation_predictions.csv` into the selected output directory. Prediction writes `ID,Target` to the requested output path. Do not interpret these outputs as clinical findings.
