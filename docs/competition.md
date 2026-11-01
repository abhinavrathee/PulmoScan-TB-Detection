# Competition notes and provenance

## Source materials in this repository

- `competition-link.txt` (renamed from the supplied link file) supplied the Kaggle URL.
- `PulmoScan-Project-Overview.md` supplied the event name, host attribution, dates, dataset counts, metric claims, data layout and model hyperparameters.
- Three original notebook files contain a Colab mount, paths, preprocessing setup, an EfficientNetB3 model, two training phases and `fit()` calls.

The archived overview is a sanitized copy; a hard-coded dataset-access password from the supplied notes was redacted before it was placed in the repository.

These files document what was provided by the project author; they are not equivalent to an official competition rulebook or a completed experiment log.

## Kaggle account/API check (2026-11-01)

Using the Kaggle CLI configured in the local environment:

- The competition leaderboard endpoint returned public scores and entries.
- The connected account's competition submissions endpoint returned **No submissions found**; the account's lifetime submission count for this event was zero at the time checked.
- The competition file endpoint returned a single `nothing.txt` placeholder. It did not expose the training/test dataset described in the supplied notes.

Therefore this repository does not state a personal rank, score, or successful competition submission. The supplied overview says the primary metric is F1 with recall as a tie-breaker, and reports 2,940 training plus 1,260 test images; those details have not been verified against accessible competition files/rules and should be confirmed against the dataset/rule page when available.

## Notebook implementation status

The notebooks show a 224×224 EfficientNetB3 transfer-learning experiment with batch size 32, 10 head-training epochs, a fine-tuning phase, learning rates `1e-4` and `1e-5`, and a nominal 80/20 validation split. They contain partial cell outputs but no saved checkpoint/export step, test prediction, submission writer, or reliable evaluation result. An empty cell in two notebook copies retains a stale `KeyError: 'class'` traceback. The second notebook is a duplicate/incomplete variant. The cleaned pipeline in `src/pulmoscan/` makes the missing evaluation and prediction steps explicit; it is a maintained implementation, not evidence that the original hackathon run completed successfully.

## Verify before publishing

When the dataset is available, confirm official event dates, allowed data use, target labels, evaluation metric, sample counts and required submission columns. Add actual run configuration and validation results only after executing the pipeline. Do not claim a Kaggle score unless it is supported by an account submission record.
