# Results and evidence

## Current result status

There is no completed PulmoScan result to report. The original notebooks contain partial outputs and calls to `model.fit`, but do not include a reliably saved final checkpoint, complete holdout evaluation, test prediction or submission file. The maintained pipeline has not been run end-to-end because the local copy was incomplete at the latest data check.

| Evidence item | Status |
|---|---|
| Label CSV | 2,940 rows observed locally; class distribution 2,450 class 0 and 490 class 1 |
| Test images | 1,260 PNGs observed locally |
| Training images | 1,309 PNGs observed at the latest setup check; 1,631 CSV-referenced files missing then |
| Notebook model setup | Present: EfficientNetB3, 224×224 input, binary classification head |
| Notebook training code | Present: frozen-backbone and fine-tuning `fit()` calls |
| Successful saved model | Not found in supplied project files |
| Validation score from PulmoScan | Not available |
| Kaggle submission for connected account | None found during CLI check on 2026-11-01 |
| Kaggle competition score for PulmoScan | Not claimed |

The local label distribution observed at setup was 2,450 normal and 490 TB-positive samples. The current training implementation does not compensate for this imbalance with class weights or sampling changes.

Counts are a workspace snapshot, not a guarantee about the current copy. Confirm file completeness before a run; the loader checks all labeled paths and will stop if any are absent.

## Why there is no score

The notebooks do not preserve a completed prediction/evaluation result, and the connected Kaggle account had no submission record when checked. Public leaderboard scores belong to other entries and are not evidence of this model's performance. No values have been inferred from the leaderboard or generated from incomplete data.

## How to add future results

After a full run, record the following alongside the saved `validation_metrics.json`:

1. Dataset version/source and a file inventory or checksum.
2. Exact command-line configuration, seed and software/hardware environment.
3. Validation split method and validation metrics at the stated threshold.
4. If submitted, the Kaggle submission reference and returned score.

Keep validation results separate from competition leaderboard results. Do not present either as clinical performance.
