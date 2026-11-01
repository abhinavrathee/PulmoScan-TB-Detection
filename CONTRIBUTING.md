# Contributing to PulmoScan

Thanks for helping improve the educational project. Contributions should keep the implementation reproducible and the claims tied to evidence.

## Before changing the pipeline

- Keep dataset access and machine-specific paths outside source code.
- Preserve `ID,Target` as the competition submission schema unless official rules confirm a different format.
- Keep validation metrics and competition scores clearly distinguished.
- Do not describe model outputs as diagnosis or clinical advice.
- Do not add dataset images, private credentials, Kaggle tokens or account files to a change.

## Development setup

Use Python 3.10 or 3.11 in Linux, WSL2, or Google Colab:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

## Change checklist

1. Keep preprocessing and prediction behavior documented.
2. Update the relevant guide when defaults or output formats change.
3. If reporting a metric, include the data version, split, threshold and run configuration.
4. Review the final diff for dataset files, notebook outputs, credentials and unsupported performance claims.

Pull requests should explain the motivation, the files changed and how the result was checked.
