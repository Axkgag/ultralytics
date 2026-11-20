# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Train YOLOE to find every target button matching an independently cropped visual prompt."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    """Parse paired visual-prompt training arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "scripts/yoloe-button.yaml")
    parser.add_argument("--model", default="yoloe-26s.yaml", help="Detection model YAML or checkpoint.")
    parser.add_argument("--weights", default="yoloe-26s-seg.pt", help="Pretrained YOLOE checkpoint, or 'none'.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--project", type=Path, default=ROOT / "runs/yoloe-button")
    parser.add_argument("--name", default="stage1-savpe")
    parser.add_argument("--cache", choices=("false", "ram", "disk"), default="false")
    parser.add_argument("--full-loss", action="store_true", help="Unfreeze the detector for stage-2 fine-tuning.")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Load pretrained weights and launch paired YOLOE training."""
    args = parse_args()
    device = args.device
    requested = device.split(",")
    if "CUDA_VISIBLE_DEVICES" not in os.environ and requested and all(x.isdigit() for x in requested):
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        device = ",".join(str(index) for index in range(len(requested)))

    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEPairedVPTrainer

    model = YOLOE(args.model)
    if args.weights.lower() != "none":
        model.load(args.weights)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        patience=args.patience,
        project=str(args.project),
        name=args.name,
        cache=False if args.cache == "false" else args.cache,
        exist_ok=args.exist_ok,
        visual_prompt_full_loss=args.full_loss,
        trainer=YOLOEPairedVPTrainer,
    )


if __name__ == "__main__":
    main()
