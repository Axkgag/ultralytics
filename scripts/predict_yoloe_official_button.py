# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Run official pretrained YOLOE with a cropped button placed on the shared gray prompt canvas."""

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
    """Parse official YOLOE visual-prompt inference arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", type=Path, required=True, help="Tightly cropped elevator-button image.")
    parser.add_argument("--source", required=True, help="Target image or directory to search.")
    parser.add_argument("--weights", default=str(ROOT / "yoloe-26s-seg.pt"), help="Official YOLOE *-seg.pt weights.")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--prompt-sizes", default="64,96,160", help="Comma-separated prompt sizes in pixels.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--label", default="prompt_button", help="Display name for object0 predictions.")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/yoloe-button")
    parser.add_argument("--name", default="predict-official")
    parser.add_argument("--save-txt", action="store_true", help="Save normalized boxes and confidence values.")
    return parser.parse_args()


def main() -> None:
    """Use the complete crop as an official YOLOE reference box and save bbox-only predictions."""
    args = parse_args()
    prompt = cv2.imread(str(args.prompt))
    if prompt is None:
        raise FileNotFoundError(f"Unable to read prompt image: {args.prompt}")

    device = args.device
    requested = device.split(",")
    if "CUDA_VISIBLE_DEVICES" not in os.environ and requested and all(index.isdigit() for index in requested):
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        device = ",".join(str(index) for index in range(len(requested)))

    from ultralytics import YOLOE
    from ultralytics.data.augment import VisualPromptCanvas
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
    from ultralytics.utils.files import increment_path

    sizes = tuple(int(value) for value in args.prompt_sizes.split(","))
    refer_image, boxes = VisualPromptCanvas((args.imgsz, args.imgsz), sizes=sizes)(prompt)
    visual_prompts = {
        "bboxes": boxes,
        "cls": np.zeros(len(boxes), dtype=np.int64),
    }
    save_dir = increment_path(args.project / args.name, mkdir=True)
    cv2.imwrite(str(save_dir / "visual_prompt.jpg"), refer_image)
    results = YOLOE(args.weights).predict(
        source=args.source,
        refer_image=refer_image,
        visual_prompts=visual_prompts,
        predictor=YOLOEVPSegPredictor,
        imgsz=args.imgsz,
        device=device,
        conf=args.conf,
        iou=args.iou,
        save=False,
    )

    for result in results:
        result.names[0] = args.label
        output = save_dir / Path(result.path).name
        result.save(str(output), masks=False)
        if args.save_txt:
            result.save_txt(save_dir / "labels" / f"{Path(result.path).stem}.txt", save_conf=True)
        print(f"{result.path}: {len(result.boxes)} detections")
        for box in result.boxes:
            xyxy = ", ".join(f"{value:.1f}" for value in box.xyxy[0].tolist())
            print(f"  {args.label} conf={float(box.conf[0]):.4f} xyxy=[{xyxy}]")
    print(f"Results saved to {save_dir}")


if __name__ == "__main__":
    main()
