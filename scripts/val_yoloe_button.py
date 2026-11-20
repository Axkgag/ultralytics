# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Validate a paired visual-prompt YOLOE checkpoint on query-conditioned elevator-button episodes."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_BANK = ROOT / "runs/yoloe-button/prompt-bank/prompt-bank.yaml"
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    """Parse paired validation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "scripts/yoloe-button.yaml")
    parser.add_argument("--prompt-bank", type=Path, default=DEFAULT_PROMPT_BANK)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence floor used to build COCO-style curves.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/yoloe-button")
    parser.add_argument("--name", default="val")
    parser.add_argument("--no-plots", action="store_false", dest="plots", help="Disable validation plots.")
    parser.add_argument("--fail-cases", action="store_true", help="Save categorized fail-case visualizations.")
    parser.add_argument("--fail-conf", type=float, default=0.25)
    parser.add_argument("--fail-iou", type=float, default=0.5)
    parser.add_argument("--fail-max-per-category", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    """Run deterministic paired validation."""
    args = parse_args()
    device = args.device
    requested = device.split(",")
    if "CUDA_VISIBLE_DEVICES" not in os.environ and requested and all(x.isdigit() for x in requested):
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        device = ",".join(str(index) for index in range(len(requested)))

    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEPairedValidator

    YOLOE(args.weights).val(
        data=str(args.data),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        conf=args.conf,
        iou=args.iou,
        project=str(args.project),
        name=args.name,
        plots=args.plots,
        visual_prompt_bank=str(args.prompt_bank),
        visual_prompt_fail_cases=args.fail_cases,
        visual_prompt_fail_conf=args.fail_conf,
        visual_prompt_fail_iou=args.fail_iou,
        visual_prompt_fail_max=args.fail_max_per_category,
        validator=YOLOEPairedValidator,
    )


if __name__ == "__main__":
    main()
