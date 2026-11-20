# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Detect all elevator buttons matching one cropped prompt image."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    """Parse visual-prompt inference arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True, help="Cropped target button image.")
    parser.add_argument("--source", required=True, help="Target image, directory, video, or stream.")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--prompt-sizes", default="64,96,160", help="Comma-separated prompt copy sizes in pixels.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--project", type=Path, default=ROOT / "runs/yoloe-button")
    parser.add_argument("--name", default="predict")
    parser.add_argument("--save-txt", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Use the complete prompt crop as a one-class visual prompt and predict target boxes."""
    args = parse_args()
    device = args.device
    requested = device.split(",")
    if "CUDA_VISIBLE_DEVICES" not in os.environ and requested and all(x.isdigit() for x in requested):
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        device = ",".join(str(index) for index in range(len(requested)))

    from ultralytics import YOLOE
    from ultralytics.data.augment import VisualPromptCanvas
    from ultralytics.models.yolo.yoloe import YOLOEVPDetectPredictor

    prompt = cv2.imread(str(args.prompt))
    if prompt is None:
        raise FileNotFoundError(f"Unable to read prompt image: {args.prompt}")
    sizes = tuple(int(x) for x in args.prompt_sizes.split(","))
    prompt, boxes = VisualPromptCanvas((args.imgsz, args.imgsz), sizes=sizes)(prompt)
    prompts = {
        "bboxes": boxes,
        "cls": np.zeros(len(boxes), dtype=np.int64),
    }
    YOLOE(args.weights).predict(
        source=args.source,
        refer_image=prompt,
        visual_prompts=prompts,
        predictor=YOLOEVPDetectPredictor,
        imgsz=args.imgsz,
        device=device,
        conf=args.conf,
        iou=args.iou,
        project=str(args.project),
        name=args.name,
        save=True,
        save_txt=args.save_txt,
    )


if __name__ == "__main__":
    main()
