# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Validate an elevator multi-task checkpoint and optionally save prediction visualizations."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from ultralytics import YOLO
from ultralytics.data.utils import img2label_paths
from ultralytics.engine.results import ElevatorAttributes
from ultralytics.utils import YAML
from ultralytics.utils.plotting import Annotator, colors

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path("/mnt/yihao/codes/buttonDet/datasets_qwen_lit_yolo/elevator-button.yaml")
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
FLOOR_BY_SLOTS = {(slot1, slot2): floor for floor, slot1, slot2 in ElevatorAttributes.floor_encodings}


def parse_args() -> argparse.Namespace:
    """Parse validation and visualization options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="Trained elevator model checkpoint.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", type=Path, default=ROOT / "runs/elevator")
    parser.add_argument("--name", default="val")
    parser.add_argument("--plots", action="store_true", help="Save standard validation plots and batch previews.")
    parser.add_argument("--visualize", action="store_true", help="Save annotated predictions for selected split images.")
    parser.add_argument("--visualize-count", type=int, default=200)
    parser.add_argument("--visual-conf", type=float, default=0.25, help="Confidence threshold used only for visualizations.")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold used only for visualizations.")
    parser.add_argument(
        "--visual-light-threshold",
        type=float,
        default=0.5,
        help="Prediction threshold for displaying light as 0 or 1.",
    )
    return parser.parse_args()


def resolve_split_images(data_path: Path, split: str) -> list[Path]:
    """Resolve a directory-based Ultralytics dataset split into image paths."""
    data = YAML.load(data_path)
    if split not in data:
        raise KeyError(f"Dataset YAML has no '{split}' split: {data_path}")
    root = Path(data.get("path") or data_path.parent)
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()
    value = data[split]
    if not isinstance(value, str):
        raise TypeError(f"Visualization requires a directory split, got {type(value).__name__} for '{split}'")
    image_dir = Path(value)
    if not image_dir.is_absolute():
        image_dir = root / image_dir
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Visualization split is not an image directory: {image_dir}")
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def draw_ground_truth(result) -> object:
    """Draw nine-column ground-truth boxes with '<floor>:<light>' labels."""
    label_path = Path(img2label_paths([str(result.path)])[0])
    if not label_path.is_file():
        raise FileNotFoundError(f"Ground-truth label does not exist: {label_path}")
    image = result.orig_img.copy()
    height, width = image.shape[:2]
    annotator = Annotator(image, example=result.names)
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = [float(value) for value in line.split()]
        if len(row) != 9:
            raise ValueError(f"{label_path}:{line_number} requires 9 columns, got {len(row)}")
        class_id, cx, cy, box_width, box_height, slot1, slot2, light, light_valid = row
        class_id, slot1, slot2, light, light_valid = map(int, (class_id, slot1, slot2, light, light_valid))
        x1 = (cx - box_width / 2) * width
        y1 = (cy - box_height / 2) * height
        x2 = (cx + box_width / 2) * width
        y2 = (cy + box_height / 2) * height
        floor = FLOOR_BY_SLOTS[(slot1, slot2)] if class_id == 0 else result.names[class_id]
        light_label = str(light) if light_valid else "?"
        annotator.box_label((x1, y1, x2, y2), f"{floor}:{light_label}", color=colors(class_id, True))
    return annotator.result()


def draw_prediction(result, light_threshold: float) -> object:
    """Draw predicted boxes with '<floor>:<light>' labels."""
    if result.elevator is None:
        raise ValueError("Visualization requires a checkpoint with ElevatorDetect outputs")
    boxes = result.boxes.cpu()
    elevator = result.elevator.cpu()
    floors = elevator.floor
    annotator = Annotator(result.orig_img.copy(), example=result.names)
    for index, box in enumerate(boxes):
        class_id = int(box.cls.item())
        floor = floors[index] if class_id == 0 else result.names[class_id]
        light = int(float(elevator.light_probability[index]) >= light_threshold)
        annotator.box_label(box.xyxy.squeeze(), f"{floor}:{light}", color=colors(class_id, True))
    return annotator.result()


def add_header(image, title: str) -> object:
    """Add a visible title above one comparison panel."""
    panel = cv2.copyMakeBorder(image, 44, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(panel, title, (14, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (25, 25, 25), 2, cv2.LINE_AA)
    return panel


def save_visualizations(model: YOLO, args: argparse.Namespace) -> None:
    """Save side-by-side ground-truth and prediction visualizations."""
    if args.visualize_count <= 0:
        return
    if not 0 <= args.visual_light_threshold <= 1:
        raise ValueError("--visual-light-threshold must be between 0 and 1")
    images = resolve_split_images(args.data, args.split)[: args.visualize_count]
    if not images:
        raise FileNotFoundError(f"No images found for visualization in split '{args.split}'")
    directory = args.project / args.name / "visualizations"
    directory.mkdir(parents=True, exist_ok=True)
    results = model.predict(
        source=[str(path) for path in images],
        imgsz=args.imgsz,
        device=args.device,
        conf=args.visual_conf,
        iou=args.iou,
        verbose=False,
    )
    for result in results:
        ground_truth = add_header(draw_ground_truth(result), "GT")
        prediction = add_header(draw_prediction(result, args.visual_light_threshold), "PREDICT")
        comparison = cv2.hconcat((ground_truth, prediction))
        output = directory / Path(result.path).name
        if not cv2.imwrite(str(output), comparison):
            raise OSError(f"Failed to save visualization: {output}")
    print(f"Saved {len(results)} visualizations to {directory}")


def main() -> None:
    """Run validation, then optionally generate annotated prediction images."""
    args = parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.weights}")
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset YAML does not exist: {args.data}")

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True,
        plots=args.plots or args.visualize,
    )
    print({key: float(value) for key, value in metrics.results_dict.items()})
    if args.visualize:
        save_visualizations(model, args)


if __name__ == "__main__":
    main()
