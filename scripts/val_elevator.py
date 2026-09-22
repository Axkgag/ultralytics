# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Validate an elevator multi-task checkpoint and optionally save prediction visualizations."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import torch

from ultralytics import YOLO
from ultralytics.data.utils import img2label_paths
from ultralytics.engine.results import ElevatorAttributes
from ultralytics.utils import YAML
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.plotting import Annotator, colors

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path("/mnt/yihao/codes/buttonDet/ultralytics/scripts/elevator-button.yaml")
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
    parser.add_argument(
        "--iou", type=float, default=0.5, help="NMS and fail-case matching IoU threshold for visualizations."
    )
    parser.add_argument(
        "--light-threshold",
        "--visual-light-threshold",
        dest="light_threshold",
        type=float,
        default=0.5,
        help="Prediction threshold used by light metrics and visualizations.",
    )
    return parser.parse_args()


def resolve_split_images(data_path: Path, split: str) -> list[Path]:
    """Resolve one or more directory-based Ultralytics dataset splits into image paths."""
    data = YAML.load(data_path)
    if split not in data:
        raise KeyError(f"Dataset YAML has no '{split}' split: {data_path}")
    root = Path(data.get("path") or data_path.parent)
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()
    value = data[split]
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
        raise TypeError(f"Visualization requires a directory split, got {type(value).__name__} for '{split}'")
    images = []
    for item in values:
        image_dir = Path(item)
        if not image_dir.is_absolute():
            image_dir = root / image_dir
        if not image_dir.is_dir():
            raise NotADirectoryError(f"Visualization split is not an image directory: {image_dir}")
        images.extend(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    return sorted(images)


def load_ground_truth(result) -> list[dict]:
    """Load nine-column ground-truth boxes in pixel coordinates."""
    label_path = Path(img2label_paths([str(result.path)])[0])
    if not label_path.is_file():
        raise FileNotFoundError(f"Ground-truth label does not exist: {label_path}")
    height, width = result.orig_img.shape[:2]
    targets = []
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
        targets.append(
            {
                "box": (x1, y1, x2, y2),
                "class_id": class_id,
                "slot1": slot1,
                "slot2": slot2,
                "light": light,
                "light_valid": light_valid,
            }
        )
    return targets


def draw_ground_truth(result, targets: list[dict]) -> object:
    """Draw ground-truth boxes with '<floor>:<light>' labels."""
    annotator = Annotator(result.orig_img.copy(), example=result.names)
    for target in targets:
        class_id = target["class_id"]
        floor = FLOOR_BY_SLOTS[(target["slot1"], target["slot2"])] if class_id == 0 else result.names[class_id]
        light_label = str(target["light"]) if target["light_valid"] else "?"
        annotator.box_label(target["box"], f"{floor}:{light_label}", color=colors(class_id, True))
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


def match_failures(result, targets: list[dict], args: argparse.Namespace) -> tuple[dict[str, set[int]], Counter]:
    """Match predictions to ground truth and return prediction indices for each failure category."""
    failures = {category: set() for category in ("detection", "floor", "light")}
    counts = Counter()
    boxes = result.boxes.cpu()
    if targets and len(boxes):
        target_boxes = torch.tensor([target["box"] for target in targets], dtype=boxes.xyxy.dtype)
        ious = box_iou(target_boxes, boxes.xyxy)
        candidates = torch.nonzero(ious >= args.iou, as_tuple=False)
        candidates = sorted(
            ((float(ious[gt_idx, pred_idx]), int(gt_idx), int(pred_idx)) for gt_idx, pred_idx in candidates),
            reverse=True,
        )
    else:
        candidates = []

    matches = []
    used_targets, used_predictions = set(), set()
    for overlap, target_index, prediction_index in candidates:
        if target_index not in used_targets and prediction_index not in used_predictions:
            matches.append((target_index, prediction_index, overlap))
            used_targets.add(target_index)
            used_predictions.add(prediction_index)

    elevator = result.elevator.cpu()
    predicted_classes = boxes.cls.int().tolist()
    predicted_floors = elevator.floor
    predicted_lights = (elevator.light_probability >= args.light_threshold).int().tolist()
    for target_index, prediction_index, _ in matches:
        target = targets[target_index]
        if target["class_id"] != predicted_classes[prediction_index]:
            failures["detection"].add(prediction_index)
            counts["class_errors"] += 1
            continue
        if target["class_id"] == 0:
            target_floor = FLOOR_BY_SLOTS[(target["slot1"], target["slot2"])]
            if predicted_floors[prediction_index] != target_floor:
                failures["floor"].add(prediction_index)
                counts["floor_errors"] += 1
        if target["light_valid"] and predicted_lights[prediction_index] != target["light"]:
            failures["light"].add(prediction_index)
            counts["light_errors"] += 1

    counts["missed_gt"] = len(targets) - len(used_targets)
    counts["false_positives"] = len(boxes) - len(used_predictions)
    failures["detection"].update(set(range(len(boxes))) - used_predictions)
    return failures, counts


def draw_failure_predictions(
    result, prediction_indices: set[int], light_threshold: float, failure_category: str
) -> object:
    """Draw errors in red, using p(light) for light failures and box confidence otherwise."""
    annotator = Annotator(result.orig_img.copy(), example=result.names)
    boxes = result.boxes.cpu()
    elevator = result.elevator.cpu()
    light_probabilities = elevator.light_probability.tolist()
    predicted_lights = [int(probability >= light_threshold) for probability in light_probabilities]
    for index in sorted(prediction_indices):
        class_id = int(boxes.cls[index])
        floor = elevator.floor[index] if class_id == 0 else result.names[class_id]
        confidence = float(boxes.conf[index])
        if failure_category == "light":
            confidence = light_probabilities[index]
        annotator.box_label(
            boxes.xyxy[index],
            f"{floor}:{predicted_lights[index]} {confidence:.2f}",
            color=(0, 0, 255),
        )
    return annotator.result()


def add_header(image, title: str) -> object:
    """Add a visible title above one comparison panel."""
    panel = cv2.copyMakeBorder(image, 44, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(panel, title, (14, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (25, 25, 25), 2, cv2.LINE_AA)
    return panel


def save_visualizations(model: YOLO, args: argparse.Namespace) -> None:
    """Save full comparisons and categorized fail-case visualizations."""
    if args.visualize_count <= 0:
        return
    images = resolve_split_images(args.data, args.split)[: args.visualize_count]
    if not images:
        raise FileNotFoundError(f"No images found for visualization in split '{args.split}'")
    directory = args.project / args.name / "visualizations"
    directory.mkdir(parents=True, exist_ok=True)
    failure_directory = args.project / args.name / "fail_cases"
    failure_summary = Counter(images=len(images))
    results = model.predict(
        source=[str(path) for path in images],
        imgsz=args.imgsz,
        device=args.device,
        conf=args.visual_conf,
        iou=args.iou,
        verbose=False,
    )
    for result in results:
        targets = load_ground_truth(result)
        ground_truth = add_header(draw_ground_truth(result, targets), "GT")
        prediction = add_header(draw_prediction(result, args.light_threshold), "PREDICT")
        comparison = cv2.hconcat((ground_truth, prediction))
        output = directory / Path(result.path).name
        if not cv2.imwrite(str(output), comparison):
            raise OSError(f"Failed to save visualization: {output}")
        failures, counts = match_failures(result, targets, args)
        failure_summary.update(counts)
        failed = False
        for category, prediction_indices in failures.items():
            category_count = counts[f"{category}_errors"]
            if category == "detection":
                category_count = counts["missed_gt"] + counts["false_positives"] + counts["class_errors"]
            if not category_count:
                continue
            failed = True
            failure_summary[f"{category}_fail_images"] += 1
            category_directory = failure_directory / category
            category_directory.mkdir(parents=True, exist_ok=True)
            failure_prediction = add_header(
                draw_failure_predictions(result, prediction_indices, args.light_threshold, category), "PREDICT"
            )
            failure_output = category_directory / Path(result.path).name
            if not cv2.imwrite(str(failure_output), cv2.hconcat((ground_truth, failure_prediction))):
                raise OSError(f"Failed to save fail-case visualization: {failure_output}")
        failure_summary["failed_images" if failed else "clean_images"] += 1
    failure_summary.update(
        confidence_threshold=args.visual_conf,
        iou_threshold=args.iou,
        light_threshold=args.light_threshold,
    )
    failure_directory.mkdir(parents=True, exist_ok=True)
    summary_path = failure_directory / "summary.json"
    summary_path.write_text(json.dumps(dict(failure_summary), indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(results)} visualizations to {directory}")
    print(f"Saved fail cases and summary to {failure_directory}")


def main() -> None:
    """Run validation, then optionally generate annotated prediction images."""
    args = parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.weights}")
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset YAML does not exist: {args.data}")
    if not 0 <= args.light_threshold <= 1:
        raise ValueError("--light-threshold must be between 0 and 1")

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
        light_threshold=args.light_threshold,
    )
    results = {"light_threshold": args.light_threshold}
    results.update({key: float(value) for key, value in metrics.results_dict.items()})
    metrics_path = args.project / args.name / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(results)
    print(f"Saved metrics to {metrics_path}")
    if args.visualize:
        save_visualizations(model, args)


if __name__ == "__main__":
    main()
