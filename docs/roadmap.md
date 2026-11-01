# Project roadmap

This roadmap is a set of possible portfolio improvements. Items are proposals, not completed work.

## Data readiness

- [ ] Finish and inventory the authorized dataset copy.
- [ ] Verify that every CSV ID maps to exactly one training image.
- [ ] Confirm the competition's official metric, submission schema and allowed-data terms.
- [ ] Check whether patient identifiers are available so validation can split by patient rather than image.

## Experiment quality

- [ ] Run the maintained baseline and preserve its configuration and validation outputs.
- [ ] Compare a small number of justified baselines (for example, a compact CNN and transfer learning) on the same split.
- [ ] Inspect confusion matrix and precision/recall tradeoffs; keep threshold selection on validation data only.
- [ ] Add an experiment tracking table with dataset version, seed, model, threshold and metrics.

## Repository polish

- [ ] Add lightweight automated checks for data validation and submission schema.
- [ ] Add a sample, non-sensitive output artifact or rendered notebook only after a complete run.
- [ ] Update README results and changelog from verifiable run artifacts.
