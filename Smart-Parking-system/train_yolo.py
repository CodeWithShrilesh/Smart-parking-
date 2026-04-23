import argparse
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(BASE_DIR, ".ultralytics"))

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a parking-lot car detector for this project."
    )
    parser.add_argument(
        "--data",
        default=os.path.join(BASE_DIR, "training", "parking_cars.yaml"),
        help="Path to YOLO dataset YAML file.",
    )
    parser.add_argument(
        "--model",
        default="yolov8s.pt",
        help="Base YOLO model checkpoint to fine-tune.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=960, help="Training image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument(
        "--project",
        default=os.path.join(BASE_DIR, "training", "runs"),
        help="Directory where Ultralytics stores training runs.",
    )
    parser.add_argument(
        "--name",
        default="parking_cars",
        help="Run name inside the training project folder.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
