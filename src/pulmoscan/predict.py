from pathlib import Path

import pandas as pd
import tensorflow as tf

from pulmoscan.data import make_dataset


def predict(args):
    data_dir = Path(args.data_dir)
    test_dir = Path(args.test_dir) if args.test_dir else data_dir / "Test_Dataset_TB"
    output = Path(args.output)
    image_paths = sorted(path for path in test_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png")
    if not image_paths:
        raise FileNotFoundError(f"No PNG test images found in {test_dir}.")
    dataset = make_dataset(
        [str(path) for path in image_paths],
        [0] * len(image_paths),
        args.image_size,
        args.batch_size,
        False,
        args.seed,
    )
    model = tf.keras.models.load_model(args.model)
    probabilities = model.predict(dataset, verbose=0).reshape(-1)
    result = pd.DataFrame(
        {"ID": [path.name for path in image_paths], "Target": (probabilities >= args.threshold).astype(int)}
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Wrote {len(result)} predictions to {output}")
