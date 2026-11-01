import argparse

from pulmoscan.predict import predict
from pulmoscan.train import train


def _common(parser):
    parser.add_argument("--data-dir", default="TB_Dataset", help="Root directory containing competition data")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)


def main():
    parser = argparse.ArgumentParser(prog="pulmoscan", description="Train and run the PulmoScan TB image classifier")
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="Train and evaluate an EfficientNetB3 model")
    _common(train_parser)
    train_parser.add_argument("--train-dir")
    train_parser.add_argument("--labels")
    train_parser.add_argument("--output-dir", default="artifacts/run-001")
    train_parser.add_argument("--validation-fraction", type=float, default=0.2)
    train_parser.add_argument("--epochs-head", type=int, default=10)
    train_parser.add_argument("--epochs-fine", type=int, default=10)
    train_parser.add_argument("--fine-tune-layers", type=int, default=100)
    train_parser.add_argument("--learning-rate-head", type=float, default=1e-4)
    train_parser.add_argument("--learning-rate-fine", type=float, default=1e-5)
    train_parser.add_argument("--dropout", type=float, default=0.5)
    train_parser.set_defaults(func=train)

    predict_parser = commands.add_parser("predict", help="Write Kaggle-format predictions for the test images")
    _common(predict_parser)
    predict_parser.add_argument("--test-dir")
    predict_parser.add_argument("--model", required=True)
    predict_parser.add_argument("--output", default="artifacts/submission.csv")
    predict_parser.set_defaults(func=predict)

    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        parser.error("--threshold must be between 0 and 1")
    if args.command == "train" and not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be between 0 and 1")
    args.func(args)
