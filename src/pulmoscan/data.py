from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


def load_labels(labels_path: Path, train_dir: Path) -> pd.DataFrame:
    """Load and validate the competition label CSV and corresponding files."""
    frame = pd.read_csv(labels_path)
    required = {"ID", "Target"}
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise ValueError(f"{labels_path} is missing required columns: {sorted(missing_columns)}")
    frame = frame[["ID", "Target"]].copy()
    if frame.empty:
        raise ValueError("The label CSV contains no rows.")
    if frame["ID"].isna().any() or frame["ID"].duplicated().any():
        raise ValueError("ID values must be present and unique.")
    frame["ID"] = frame["ID"].astype(str)
    frame["Target"] = pd.to_numeric(frame["Target"], errors="coerce")
    if frame["Target"].isna().any() or not set(frame["Target"].unique()).issubset({0, 1}):
        raise ValueError("Target values must be binary integers 0 or 1.")
    frame["Target"] = frame["Target"].astype("int32")
    missing_files = [name for name in frame["ID"] if not (train_dir / name).is_file()]
    if missing_files:
        sample = ", ".join(missing_files[:5])
        raise FileNotFoundError(f"{len(missing_files)} training images are missing from {train_dir}; examples: {sample}")
    if frame["Target"].nunique() != 2:
        raise ValueError("Both classes (0 and 1) are required for a stratified split.")
    return frame


def split_labels(frame: pd.DataFrame, validation_fraction: float, seed: int):
    train_frame, validation_frame = train_test_split(
        frame,
        test_size=validation_fraction,
        random_state=seed,
        stratify=frame["Target"],
    )
    return train_frame.reset_index(drop=True), validation_frame.reset_index(drop=True)


def _decode_image(path: tf.Tensor, label: tf.Tensor, image_size: int):
    image = tf.io.read_file(path)
    image = tf.io.decode_png(image, channels=3)
    image = tf.image.resize(image, [image_size, image_size])
    return image, tf.cast(label, tf.float32)


def make_dataset(paths, labels, image_size: int, batch_size: int, training: bool, seed: int):
    dataset = tf.data.Dataset.from_tensor_slices((np.asarray(paths), np.asarray(labels, dtype="int32")))
    if training:
        dataset = dataset.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.map(
        lambda path, label: _decode_image(path, label, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
