import json
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from pulmoscan.data import load_labels, make_dataset, split_labels
from pulmoscan.model import build_model


def _fit_phase(model, train_ds, validation_ds, epochs, learning_rate, checkpoint, initial_epoch=0):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint), monitor="val_loss", mode="min", save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=3, restore_best_weights=True
        ),
    ]
    return model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
    )


def train(args):
    tf.keras.utils.set_random_seed(args.seed)
    data_dir = Path(args.data_dir)
    train_dir = Path(args.train_dir) if args.train_dir else data_dir / "Training_Dataset_TB"
    labels_path = Path(args.labels) if args.labels else data_dir / "TB_train.csv"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_labels(labels_path, train_dir)
    train_frame, validation_frame = split_labels(frame, args.validation_fraction, args.seed)
    train_paths = [str(train_dir / name) for name in train_frame["ID"]]
    validation_paths = [str(train_dir / name) for name in validation_frame["ID"]]
    train_ds = make_dataset(train_paths, train_frame["Target"], args.image_size, args.batch_size, True, args.seed)
    validation_ds = make_dataset(
        validation_paths, validation_frame["Target"], args.image_size, args.batch_size, False, args.seed
    )

    model, backbone = build_model(args.image_size, args.dropout)
    checkpoint = output_dir / "best_model.keras"
    _fit_phase(model, train_ds, validation_ds, args.epochs_head, args.learning_rate_head, checkpoint)

    if args.epochs_fine > 0:
        backbone.trainable = True
        for layer in backbone.layers[:-args.fine_tune_layers]:
            layer.trainable = False
        # Keep batch normalization in inference mode while fine-tuning a small dataset.
        for layer in backbone.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        fine_checkpoint = output_dir / "fine_tuned_model.keras"
        _fit_phase(
            model,
            train_ds,
            validation_ds,
            args.epochs_head + args.epochs_fine,
            args.learning_rate_fine,
            fine_checkpoint,
            initial_epoch=args.epochs_head,
        )
        head_loss = float(tf.keras.models.load_model(checkpoint).evaluate(validation_ds, verbose=0)[0])
        fine_loss = float(tf.keras.models.load_model(fine_checkpoint).evaluate(validation_ds, verbose=0)[0])
        if fine_loss < head_loss:
            shutil.copy2(fine_checkpoint, checkpoint)
        fine_checkpoint.unlink(missing_ok=True)

    best_model = tf.keras.models.load_model(checkpoint)
    probabilities = best_model.predict(validation_ds, verbose=0).reshape(-1)
    predictions = (probabilities >= args.threshold).astype(int)
    truth = validation_frame["Target"].to_numpy()
    metrics = {
        "threshold": args.threshold,
        "validation_images": int(len(truth)),
        "accuracy": float(accuracy_score(truth, predictions)),
        "precision": float(precision_score(truth, predictions, zero_division=0)),
        "recall": float(recall_score(truth, predictions, zero_division=0)),
        "f1": float(f1_score(truth, predictions, zero_division=0)),
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "train_class_counts": {str(k): int(v) for k, v in train_frame["Target"].value_counts().sort_index().items()},
        "validation_class_counts": {
            str(k): int(v) for k, v in validation_frame["Target"].value_counts().sort_index().items()
        },
        "configuration": {
            "backbone": "EfficientNetB3",
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "epochs_head": args.epochs_head,
            "epochs_fine": args.epochs_fine,
            "fine_tune_layers": args.fine_tune_layers,
            "learning_rate_head": args.learning_rate_head,
            "learning_rate_fine": args.learning_rate_fine,
            "dropout": args.dropout,
        },
        "tensorflow_version": tf.__version__,
    }
    (output_dir / "validation_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    validation_output = validation_frame[["ID", "Target"]].copy()
    validation_output["Probability"] = probabilities
    validation_output["Predicted"] = predictions
    validation_output.to_csv(output_dir / "validation_predictions.csv", index=False)
    print(json.dumps(metrics, indent=2))
