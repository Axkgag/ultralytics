# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Train the four-level elevator multi-task model."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, (str(ROOT), os.environ.get("PYTHONPATH"))))
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402


DEFAULT_DATA = Path("/mnt/yihao/codes/buttonDet/ultralytics-uda/scripts/elevator-button.yaml")
DEFAULT_WEIGHTS = Path("/mnt/yihao/codes/buttonDet/ultralytics/weights/yolov8l.pt")


def parse_args() -> argparse.Namespace:
    """Parse training options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "ultralytics/cfg/models/v8/yolov8-elevator-p2.yaml")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--weights", type=Path, default=DEFAULT_WEIGHTS, help="Detection checkpoint for initialization."
    )
    parser.add_argument("--from-scratch", action="store_true", help="Do not initialize from --weights.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1, help="Batch size; -1 selects it automatically.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--save-period", type=int, default=-1, help="Save a checkpoint every N epochs; -1 disables it.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=Path, default=ROOT / "runs/elevator")
    parser.add_argument("--name", default="train")
    parser.add_argument("--cache", action="store_true", help="Cache images in RAM or disk as chosen by Ultralytics.")
    parser.add_argument("--exist-ok", action="store_true", help="Reuse --project/--name instead of incrementing it.")
    parser.add_argument("--unsup", action="store_true", help="Enable ElevatorDetect EMA teacher-student training.")
    parser.add_argument("--target-data", type=Path, help="Target ElevatorDetect dataset YAML; omit for epoch splits.")
    parser.add_argument("--unsup-ratio", type=float, default=0.25)
    parser.add_argument("--unsup-weight", type=float, default=0.25)
    parser.add_argument("--unsup-warmup-epochs", type=int, default=5)
    parser.add_argument("--uda-teacher-decay", type=float, default=0.999)
    parser.add_argument("--pseudo-conf", type=float, default=0.9)
    parser.add_argument("--pseudo-iou", type=float, default=0.7)
    parser.add_argument("--pseudo-floor-conf", type=float, default=0.9)
    parser.add_argument("--pseudo-light-conf", type=float, default=0.9)
    parser.add_argument("--pseudo-max-boxes", type=int, default=35)
    parser.add_argument("--pseudo-max-floor-ratio", type=float, default=0.98)
    parser.add_argument("--optimizer", help="Optimizer override; pretrained UDA defaults to AdamW.")
    parser.add_argument("--lr0", type=float, help="Initial learning rate override; pretrained UDA defaults to 1e-4.")
    parser.add_argument(
        "--warmup-bias-lr",
        type=float,
        help="Warmup bias learning rate override; pretrained UDA defaults to the selected --lr0.",
    )
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
    if args.target_data and not args.target_data.is_file():
        raise FileNotFoundError(f"Target dataset YAML does not exist: {args.target_data}")

    pretrained_uda = args.unsup and not args.from_scratch
    optimizer = args.optimizer or ("AdamW" if pretrained_uda else "auto")
    lr0 = args.lr0 if args.lr0 is not None else (1e-4 if pretrained_uda else 0.01)
    warmup_bias_lr = (
        args.warmup_bias_lr if args.warmup_bias_lr is not None else (lr0 if pretrained_uda else 0.1)
    )

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        save_period=args.save_period,
        seed=args.seed,
        project=args.project,
        name=args.name,
        cache=args.cache,
        exist_ok=args.exist_ok,
        optimizer=optimizer,
        lr0=lr0,
        warmup_bias_lr=warmup_bias_lr,
        pretrained=str(args.weights) if not args.from_scratch else False,
        unsup=args.unsup,
        target_data=str(args.target_data) if args.target_data else None,
        unsup_ratio=args.unsup_ratio,
        unsup_weight=args.unsup_weight,
        unsup_warmup_epochs=args.unsup_warmup_epochs,
        uda_teacher_decay=args.uda_teacher_decay,
        pseudo_conf=args.pseudo_conf,
        pseudo_iou=args.pseudo_iou,
        pseudo_floor_conf=args.pseudo_floor_conf,
        pseudo_light_conf=args.pseudo_light_conf,
        pseudo_max_boxes=args.pseudo_max_boxes,
        pseudo_max_floor_ratio=args.pseudo_max_floor_ratio,
    )


if __name__ == "__main__":
    main()
