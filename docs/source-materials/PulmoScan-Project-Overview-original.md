# 🫁 PulmoScan — Tuberculosis Detection from Chest X-Rays

> A deep learning-based binary classification system to detect Tuberculosis (TB) from chest X-ray images, built as part of a competitive ML challenge hosted on the CDIS IIT Kanpur platform.

---

## 📌 Table of Contents

1. [Project Overview](#project-overview)
2. [Competition Details](#competition-details)
3. [Dataset Description](#dataset-description)
4. [Tech Stack](#tech-stack)
5. [Project Architecture & Workflow](#project-architecture--workflow)
6. [Model Architecture (EfficientNetB3)](#model-architecture-efficientnetb3)
7. [Training Strategy](#training-strategy)
8. [Hyperparameters & Configuration](#hyperparameters--configuration)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Submission Format](#submission-format)
11. [File & Directory Structure](#file--directory-structure)
12. [How to Run](#how-to-run)
13. [Key Design Decisions](#key-design-decisions)
14. [Author & Context](#author--context)
15. [Skills Used (Resume Section)](#skills-used-resume-section)

---

## 🧠 Project Overview

**PulmoScan** is a machine learning project focused on automated **Tuberculosis (TB) diagnosis** from chest X-ray images. Tuberculosis remains one of the world's deadliest infectious diseases, and accurate automated detection can significantly support clinical diagnosis — especially in resource-constrained settings.

This project uses a **Convolutional Neural Network (CNN)** with a **Transfer Learning** approach (EfficientNetB3 pre-trained on ImageNet) to classify chest X-rays into two categories:

| Class | Label | Meaning |
|-------|-------|---------|
| Normal | `0` | No Tuberculosis detected |
| TB Positive | `1` | Tuberculosis detected |

The model is designed to generalize well across varied chest X-ray cases and is evaluated primarily on the **F1-Score** metric, with **Recall** as a tiebreaker.

---

## 🏆 Competition Details

| Field | Details |
|-------|---------|
| **Competition Name** | Tuberculosis Detection from Chest X-Rays |
| **Hosted By** | CDIS, IIT Kanpur |
| **Competition URL** | https://www.kaggle.com/competitions/tuberculosis-detection-from-chest-x-rays |
| **Problem Type** | Binary Image Classification |
| **Start Date** | March 28, 2025 |
| **Close Date** | March 30, 2025 |
| **Submission Type** | Late Submission |
| **Primary Metric** | F1-Score |
| **Tiebreaker Metric** | Recall |
| **Training Samples** | 2,940 chest X-ray images |
| **Test Samples** | 1,260 chest X-ray images |

### 🎯 Goal

> Develop a robust classification model that can predict whether or not a patient has Tuberculosis disease based on Chest X-Rays with minimal error.

### ✅ Allowed Methods
- Any machine learning technique
- Deep learning models (CNN, Transfer Learning, etc.)
- Automated Machine Learning (AMLT) tools
- Final submission must demonstrate a strong, novel approach that improves upon the benchmark model

---

## 📂 Dataset Description

### Source
Dataset is provided by CDIS, IIT Kanpur.

**Download Options:**
```
# Browser download
http://cdis.iitk.ac.in/get_data/TB_Dataset

# Windows PowerShell (wget)
wget "https://cdis.iitk.ac.in/get_data/TB_Dataset" -OutFile "TB_Dataset.zip"

# SCP command
scp -P 5222 -r data@cdis.iitk.ac.in:/home/data/TB_Dataset "path_to_your_folder"
# Password: [redacted]
```

### Dataset Structure

```
TB_Dataset/
├── Training_dataset_TB/        # 2,940 chest X-ray images for training
│   ├── TB_train_0001.png
│   ├── TB_train_0002.png
│   └── ...
├── TB_train.csv                # Ground truth labels for training images
│   ├── ID     → image filename
│   └── Target → 0 (Normal) or 1 (TB Positive)
└── Test_dataset_TB/            # 1,260 chest X-ray images for testing (labels hidden)
    ├── TB_test_0001.png
    ├── TB_test_0002.png
    └── ...
```

### Data Split
The training data is internally split as:
- **Training Set**: 80% of training images
- **Validation Set**: 20% of training images (via `validation_split=0.2`)

---

## 🛠️ Tech Stack

### Programming Language
| Technology | Version / Details |
|------------|-------------------|
| **Python** | 3.x (Google Colab environment) |

### Cloud & Development Platform
| Technology | Purpose |
|------------|---------|
| **Google Colab** | Cloud-based Jupyter notebook IDE for training (GPU-accelerated) |
| **Google Drive** | Cloud storage for dataset and model checkpoints |

### Core Libraries & Frameworks

#### Deep Learning
| Library | Purpose |
|---------|---------|
| **TensorFlow** | Core deep learning framework |
| **Keras** (via TensorFlow) | High-level API for model building, training, and inference |
| `tensorflow.keras.applications` | Pre-trained model hub (EfficientNetB3 from ImageNet) |
| `tensorflow.keras.preprocessing.image` | Image data loading and augmentation (ImageDataGenerator) |
| `tensorflow.keras.layers` | Neural network layer components (Dense, Dropout, GlobalAveragePooling2D, Input) |
| `tensorflow.keras.models` | Model definition and saving (Model) |
| `tensorflow.keras.optimizers` | Optimizers (Adam) |
| `tensorflow.keras.callbacks` | Training callbacks (EarlyStopping, ModelCheckpoint, etc.) |

#### Data Science & Numerical Computing
| Library | Purpose |
|---------|---------|
| **NumPy** | Numerical operations and array manipulation |
| **Pandas** | Data loading, CSV parsing, and DataFrame operations |

#### Machine Learning Utilities
| Library | Purpose |
|---------|---------|
| **scikit-learn** (`sklearn.metrics`) | F1-Score computation for model evaluation |

#### System Utilities
| Library | Purpose |
|---------|---------|
| **os** | File path management and directory navigation |

### Pre-trained Model
| Model | Source | Use |
|-------|--------|-----|
| **EfficientNetB3** | ImageNet weights (via Keras Applications) | Feature extractor backbone for Transfer Learning |

---

## 🔁 Project Architecture & Workflow

```
Raw Chest X-Rays (PNG)
        |
        v
+--------------------------------------+
|      Data Loading & Preprocessing    |
|  - Read TB_train.csv via Pandas      |
|  - Rename: ID→image, Target→class    |
|  - Convert labels to string type     |
+-------------------+------------------+
                    |
                    v
+--------------------------------------+
|    Image Data Generator Setup        |
|  - Rescale pixel values (/255)       |
|  - Validation split: 20%             |
|  - Train + Validation generators     |
|  - Image size: 224x224x3             |
|  - Batch size: 32                    |
|  - Binary class mode                 |
+-------------------+------------------+
                    |
                    v
+--------------------------------------+
|       Model Construction             |
|  - Base: EfficientNetB3 (ImageNet)   |
|  - Freeze all base layers            |
|  - Add: GlobalAveragePooling2D       |
|  - Add: Dense(256, relu)             |
|  - Add: Dropout(0.5)                 |
|  - Add: Dense(1, sigmoid) -> output  |
+-------------------+------------------+
                    |
                    v
+--------------------------------------+
|    Phase 1: Top-Layer Training       |
|  - Optimizer: Adam (lr=1e-4)         |
|  - Loss: Binary Crossentropy         |
|  - Epochs: 10 (EPOCHS_TOP)           |
|  - Train only custom head layers     |
+-------------------+------------------+
                    |
                    v
+--------------------------------------+
|    Phase 2: Fine-Tuning              |
|  - Unfreeze base_model               |
|  - Freeze all but last 100 layers    |
|  - Optimizer: Adam (lr=1e-5)         |
|  - Epochs: 20 (TOP + FINE = 10+10)   |
|  - Gradual domain adaptation         |
+-------------------+------------------+
                    |
                    v
+--------------------------------------+
|    Evaluation & Submission           |
|  - F1-Score on validation set        |
|  - Generate predictions on test set  |
|  - Create submission CSV file        |
+--------------------------------------+
```

---

## 🏗️ Model Architecture (EfficientNetB3)

### Why EfficientNetB3?
EfficientNetB3 is part of the EfficientNet family — CNNs that uniformly scale all dimensions (width, depth, resolution) using a **compound scaling method**. It offers an excellent accuracy-vs-parameters trade-off, making it ideal for medical image classification.

### Architecture Details

```
Input Layer  →  (224, 224, 3)
       |
       v
EfficientNetB3 Base (ImageNet weights)
  +----------------------------------+
  | ~10M parameters                  |
  | Pre-trained feature extractor    |
  | Initially FROZEN                 |
  | Phase 2: Partial unfreeze        |
  |  (only last 100 layers trained)  |
  +----------------------------------+
       |
       v
GlobalAveragePooling2D     <- Reduces spatial dims to a vector
       |
       v
Dense(256, activation='relu')  <- Custom classification head
       |
       v
Dropout(0.5)               <- Regularization to prevent overfitting
       |
       v
Dense(1, activation='sigmoid') <- Binary output [0,1] probability
       |
       v
Output: TB Probability (>0.5 = TB Positive, <=0.5 = Normal)
```

### Loss Function
- **Binary Crossentropy** — Standard loss for binary classification

### Optimizer
- **Adam** (Adaptive Moment Estimation)
  - Phase 1 LR: `1e-4` (higher — training only new head)
  - Phase 2 LR: `1e-5` (lower — fine-tuning pre-trained weights carefully)

---

## 🎯 Training Strategy

This project uses a **Two-Phase Transfer Learning** approach:

### Phase 1 — Feature Extraction (Top-Layer Training)
- EfficientNetB3 base model weights are **frozen** (`base_model.trainable = False`)
- Only the custom classification head layers are trained
- High learning rate (`1e-4`) since only new layers need rapid adjustment
- Runs for `EPOCHS_TOP = 10` epochs
- Objective: Quickly adapt the new head to the TB classification task

### Phase 2 — Fine-Tuning
- EfficientNetB3 base is **unfrozen** (`base_model.trainable = True`)
- All layers **except the last 100** are re-frozen (`base_model.layers[:-100]`)
- Only the last 100 layers of EfficientNetB3 + the head are trainable
- Lower learning rate (`1e-5`) to avoid destroying pre-trained weights
- Continues training for additional `EPOCHS_FINE = 10` epochs (total 20)
- Objective: Gradually adapt higher-level feature extractors to X-ray domain

---

## ⚙️ Hyperparameters & Configuration

```python
IMG_SIZE         = (224, 224)  # Input image dimensions (width x height)
BATCH_SIZE       = 32          # Number of images per training step
EPOCHS_TOP       = 10          # Epochs for Phase 1 (head-only training)
EPOCHS_FINE      = 10          # Epochs for Phase 2 (fine-tuning)
SEED             = 42          # Random seed for reproducibility

# Optimizer learning rates
LR_PHASE1        = 1e-4        # Phase 1: Adam learning rate
LR_PHASE2        = 1e-5        # Phase 2: Fine-tuning learning rate

# Data splitting
VALIDATION_SPLIT = 0.2         # 20% of training data used for validation

# Regularization
DROPOUT_RATE     = 0.5         # Dropout probability in classification head

# Fine-tuning layer threshold
FINE_TUNE_FROM   = -100        # Unfreeze last 100 layers of EfficientNetB3
```

---

## 📈 Evaluation Metrics

The model performance is measured using the following metrics:

| Metric | Formula | Role |
|--------|---------|------|
| **F1-Score** | 2 x (Precision x Recall) / (Precision + Recall) | **Primary** competition metric |
| **Recall** | TP / (TP + FN) | Tiebreaker metric |
| **Precision** | TP / (TP + FP) | Supporting metric |
| **Accuracy** | (TP + TN) / Total | Monitored during training |

> **Note:** In medical diagnostics, Recall (sensitivity) is especially critical — a missed TB case (False Negative) is more dangerous than a false alarm (False Positive).

```python
from sklearn.metrics import f1_score
# Used to compute F1-score on validation predictions
```

---

## 📄 Submission Format

The final submission CSV file must have the following format:

```csv
ID,Target
TB_test_0001.png,0
TB_test_0002.png,1
TB_test_0003.png,0
```

- `ID` — The filename of the test image
- `Target` — Predicted label: `0` (Normal) or `1` (TB Positive)

---

## 📁 File & Directory Structure

```
D:\PulmoScan\
|
├── README.md                    <- This documentation file
|
├── PulmoScan1.ipynb             <- Main Google Colab Notebook (training pipeline)
|
└── (Dataset — downloaded separately)
    ├── Training_Dataset_TB/     <- 2,940 training X-ray images (.png)
    ├── TB_train.csv             <- Ground truth labels (ID, Target)
    └── Test_Dataset_TB/         <- 1,260 test X-ray images (.png)
```

**Google Drive Path (Colab):**
```
/content/drive/MyDrive/X-ray chest/Dataset/
├── Training_Dataset_TB/
├── TB_train.csv
└── Test_Dataset_TB/
```

---

## ▶️ How to Run

### Prerequisites
1. A Google account with Google Colab access
2. Dataset downloaded and uploaded to Google Drive at:
   `/MyDrive/X-ray chest/Dataset/`

### Steps

**Step 1 — Mount Google Drive**
```python
from google.colab import drive
drive.mount('/content/drive')
```

**Step 2 — Set Correct Paths**
```python
train_dir = '/content/drive/MyDrive/X-ray chest/Dataset/Training_Dataset_TB'
test_dir  = '/content/drive/MyDrive/X-ray chest/Dataset/Test_Dataset_TB'
csv_path  = '/content/drive/MyDrive/X-ray chest/Dataset/TB_train.csv'
```

**Step 3 — Run All Cells**
- Open `PulmoScan1.ipynb` in Google Colab
- Click `Runtime → Run all`
- Training will proceed automatically through Phase 1 and Phase 2

**Step 4 — Generate Submission**
- After training completes, run the prediction cell on test images
- Export the results as a CSV in the required submission format

---

## 🔑 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **EfficientNetB3 over ResNet/VGG** | Better accuracy-to-parameter ratio; well-suited for medical imaging |
| **Two-phase training** | Prevents destroying pre-trained weights; allows stable convergence |
| **Sigmoid output (not Softmax)** | Correct for binary classification; outputs a probability score |
| **flow_from_dataframe over flow_from_directory** | Dataset labels are in a CSV, not organized into class folders |
| **Labels converted to string** | Keras flow_from_dataframe with class_mode='binary' requires string labels |
| **Dropout(0.5)** | Reduces overfitting on the small training set (~2,940 images) |
| **Seed = 42** | Ensures reproducibility across runs |
| **LR reduction in Phase 2** | Protects pre-trained weights from large gradient updates during fine-tuning |
| **Image size 224x224** | EfficientNetB3's recommended input resolution |
| **Validation split = 0.2** | Standard 80/20 split for internal performance monitoring |

---

## 👤 Author & Context

| Field | Details |
|-------|---------|
| **Project Name** | PulmoScan |
| **Notebook** | PulmoScan1 (Google Colab) |
| **Competition** | CDIS IIT Kanpur — TB Detection Challenge |
| **Domain** | Healthcare AI / Medical Image Analysis |
| **Problem Type** | Binary Classification |
| **Approach** | Transfer Learning with EfficientNetB3 |

---

*This project demonstrates the application of state-of-the-art deep learning techniques to a real-world medical diagnosis problem, combining computer vision, transfer learning, and clinical domain knowledge to build an automated TB screening tool.*

---

## 💼 Skills Used (Resume Section)

> This section is specifically curated for your resume. Copy-paste or adapt any of these skills into your CV.

---

### 🔬 Machine Learning & AI
- **Deep Learning** — Built and fine-tuned deep neural networks for image classification using TensorFlow/Keras
- **Transfer Learning** — Applied ImageNet pre-trained EfficientNetB3 as a feature extractor and fine-tuned for domain-specific tasks
- **Binary Image Classification** — End-to-end pipeline for classifying medical images into two categories (Normal vs TB)
- **Convolutional Neural Networks (CNNs)** — Leveraged CNN architectures for automated feature extraction from chest X-ray images
- **Model Fine-Tuning** — Progressive unfreezing and layer-specific learning rate tuning for domain adaptation
- **Hyperparameter Tuning** — Configured batch size, epochs, learning rate, dropout rate, and image size for optimal performance
- **Regularization Techniques** — Applied Dropout layers to reduce overfitting in deep learning models

### 📊 Data Science & Analytics
- **Data Preprocessing** — Cleaned and prepared CSV-based labels, renamed columns, and type-converted target labels for Keras compatibility
- **Image Data Augmentation** — Used ImageDataGenerator with rescaling and validation splitting for robust generalization
- **Data Pipeline Design** — Built efficient data pipelines using flow_from_dataframe for directory-based image loading
- **Pandas** — CSV ingestion, DataFrame manipulation, column renaming, and type conversion
- **NumPy** — Numerical array operations for data handling

### 🧪 Model Evaluation
- **F1-Score Optimization** — Primary evaluation metric for imbalanced binary classification
- **Precision & Recall Analysis** — Assessed model performance using medical diagnostic trade-offs
- **Validation Strategy** — Implemented 80/20 train-validation split with reproducible seeding

### 🏥 Domain Knowledge
- **Medical Image Analysis** — Analyzed chest X-ray images for pathological pattern detection
- **Tuberculosis (TB) Diagnostics** — Understood clinical context of binary TB classification from radiological images
- **Healthcare AI** — Applied AI in a real-world clinical diagnosis scenario

### ☁️ Cloud & Tools
- **Google Colab** — GPU-accelerated cloud training environment for deep learning models
- **Google Drive Integration** — Mounted and managed large datasets from Google Drive in Colab notebooks
- **Jupyter Notebooks** — Interactive notebook-based development and experimentation
- **Python** — Core language for end-to-end ML development

### 🏆 Competitive ML
- **Kaggle ML Competitions** — Participated in a Kaggle data science competition with submission file generation
- **Benchmark Improvement** — Designed solutions aimed to outperform provided baseline/benchmark models
