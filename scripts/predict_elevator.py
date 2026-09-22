# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Run elevator multi-task inference and save visualizations and structured results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from scripts.val_elevator import IMAGE_SUFFIXES, draw_prediction
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    """Parse inference options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="Trained elevator model checkpoint.")
    parser.add_argument("--source", type=Path, required=True, help="Input image or directory containing images.")
    parser.add_argument("--output", type=Path, required=True, help="Root directory for vis/ and output/.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--light-threshold", type=float, default=0.5)
    return parser.parse_args()


def resolve_images(source: Path) -> list[Path]:
    """Resolve one image or all supported images directly inside a directory."""
    if source.is_file():
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image file: {source}")
        return [source]
    if source.is_dir():
        images = sorted(path for path in source.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        if images:
            return images
        raise FileNotFoundError(f"No supported images found in: {source}")
    raise FileNotFoundError(f"Source does not exist: {source}")


def qwen_label(result, light_threshold: float) -> dict:
    """Serialize one image result as input for the Qwen review script."""
    if result.elevator is None:
        raise ValueError("The checkpoint did not return elevator attributes; use an elevator multi-task checkpoint")

    annotations = []
    class_names = {"floor": "floor_button", "other": "other_button"}
    for index, item in enumerate(result.summary(normalize=False, decimals=6)):
        if item["name"] not in class_names:
            raise ValueError(f"Unsupported button class for Qwen review: {item['name']}")
        box = item["box"]
        bbox = [
            box["x1"],
            box["y1"],
            round(box["x2"] - box["x1"], 6),
            round(box["y2"] - box["y1"], 6),
        ]
        light_probability = item["light_probability"]
        lit = int(light_probability >= light_threshold)
        annotations.append(
            {
                "class": class_names[item["name"]],
                "bbox": bbox,
                "floor_label": item["floor"] if item["name"] == "floor" else "unknown",
                "object_id": f"det_{index:03d}",
                "lit": lit,
                "light_valid": True,
                "light_state": "selected" if lit else "not_selected",
                "light_confidence": round(light_probability if lit else 1.0 - light_probability, 6),
                "light_source": "elevator_multitask_model",
                "yolo_class_id": item["class"],
                "yolo_button_confidence": item["confidence"],
                "yolo_floor_confidence": item["floor_confidence"],
                "yolo_floor_score": item["floor_score"],
                "yolo_light_probability": light_probability,
            }
        )

    height, width = result.orig_shape
    return {
        "image": Path(result.path).name,
        "width": width,
        "height": height,
        "light_threshold": light_threshold,
        "annotations": annotations,
    }


def main() -> None:
    """Run inference and save visualizations and per-image JSON results."""
    args = parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.weights}")
    if not 0 <= args.light_threshold <= 1:
        raise ValueError("--light-threshold must be between 0 and 1")

    images = resolve_images(args.source)
    if len({path.stem for path in images}) != len(images):
        raise ValueError("--source contains image filenames with duplicate stems")
    vis_dir = args.output / "vis"
    result_dir = args.output / "output"
    vis_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    results = model.predict(
        source=[str(path) for path in images],
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        stream=True,
        verbose=False,
    )
    saved = 0
    for result in results:
        image_path = Path(result.path)
        vis_path = vis_dir / image_path.name
        if vis_path.resolve() == image_path.resolve():
            raise ValueError("--output must not overwrite the source image")
        if not cv2.imwrite(str(vis_path), draw_prediction(result, args.light_threshold)):
            raise OSError(f"Failed to save visualization: {vis_path}")
        json_path = result_dir / f"{image_path.stem}_anno.json"
        json_path.write_text(
            json.dumps(qwen_label(result, args.light_threshold), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        saved += 1
    print(f"Saved {saved} visualizations to {vis_dir}")
    print(f"Saved {saved} prediction files to {result_dir}")


if __name__ == "__main__":
    main()
