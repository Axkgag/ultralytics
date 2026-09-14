# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Train the four-level elevator multi-task model."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path("/mnt/yihao/codes/buttonDet/datasets_qwen_lit_yolo/elevator-button.yaml")
DEFAULT_WEIGHTS = Path("/mnt/yihao/codes/buttonDet/ultralytics/weights/yolov8l.pt")


def parse_args() -> argparse.Namespace:
    """Parse training options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "ultralytics/cfg/models/v8/yolov8-elevator-p2.yaml")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Detection checkpoint for initialization.")
    parser.add_argument("--from-scratch", action="store_true", help="Do not initialize from --weights.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1, help="Batch size; -1 selects it automatically.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=Path, default=ROOT / "runs/elevator")
    parser.add_argument("--name", default="train")
    parser.add_argument("--cache", action="store_true", help="Cache images in RAM or disk as chosen by Ultralytics.")
    parser.add_argument("--exist-ok", action="store_true", help="Reuse --project/--name instead of incrementing it.")
    return parser.parse_args()


def main() -> None:
    """Initialize the custom model and run Ultralytics training."""
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"Model YAML does not exist: {args.model}")
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset YAML does not exist: {args.data}")
    if not args.from_scratch and not args.weights.is_file():
        raise FileNotFoundError(f"Initialization checkpoint does not exist: {args.weights}")

    model = YOLO(args.model)
    if not args.from_scratch:
        model.load(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        project=args.project,
        name=args.name,
        cache=args.cache,
        exist_ok=args.exist_ok,
        pretrained=not args.from_scratch,
    )


if __name__ == "__main__":
    main()
