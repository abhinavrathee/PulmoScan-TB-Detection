# Model card: PulmoScan EfficientNetB3

## Intended use

Educational and portfolio demonstration of binary chest X-ray classification. It is not intended for diagnosis, triage, treatment decisions, or use with patients.

## Model

An ImageNet-pretrained EfficientNetB3 backbone with global average pooling, a 256-unit ReLU dense layer, dropout (0.5), and a sigmoid output. Training has a frozen-backbone phase and an optional fine-tuning phase over the final 100 backbone layers. Default input size is 224×224.

## Data

The supplied materials describe competition-provided chest X-rays and labels. A partial dataset copy was present in the local workspace and was still missing many labeled training images at the latest check. Dataset origin, patient overlap, acquisition settings, demographic coverage and label quality have not been independently assessed.

## Evaluation

The pipeline reports accuracy, precision, recall and F1 on a stratified image-level holdout split. No performance values are reported because no complete run is recorded.

## Known limitations and risks

- Chest X-ray distribution shifts, imaging devices, populations and clinical settings may materially change performance.
- Image-level splitting may leak patient-specific information if the dataset includes multiple images per person.
- A single threshold does not capture clinical costs of false negatives and false positives.
- False predictions could cause harm if mistaken for medical advice; outputs must remain educational only.
- The model has not been externally validated, calibrated, clinically reviewed or cleared for clinical use.
