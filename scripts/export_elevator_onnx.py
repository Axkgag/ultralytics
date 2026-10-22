# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Export an elevator multi-task PyTorch checkpoint to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


DEFAULT_WEIGHTS = Path(
    "/mnt/yihao/codes/buttonDet/ultralytics/runs/elevator/1010/train_qwen_adv_150/weights/best.pt"
)


def parse_args() -> argparse.Namespace:
    """Parse ONNX export options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Elevator model checkpoint.")
    parser.add_argument("--imgsz", type=int, nargs="+", default=[640], help="Input size as one value or height width.")
    parser.add_argument("--batch", type=int, default=1, help="Export batch size.")
    parser.add_argument("--device", default="cpu", help="Export device, for example cpu or 0.")
    parser.add_argument("--opset", type=int, default=None, help="ONNX opset; use the framework default when omitted.")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic batch, height, and width axes.")
    parser.add_argument("--simplify", action="store_true", help="Simplify the exported ONNX graph.")
    return parser.parse_args()


def main() -> None:
    """Export the checkpoint and validate the generated ONNX graph."""
    args = parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.weights}")
    if len(args.imgsz) not in {1, 2}:
        raise ValueError("--imgsz accepts one value or two values: height width")
    if args.batch < 1:
        raise ValueError("--batch must be at least 1")

    imgsz = args.imgsz[0] if len(args.imgsz) == 1 else args.imgsz
    model = YOLO(args.weights)
    output = Path(
        model.export(
            format="onnx",
            imgsz=imgsz,
            batch=args.batch,
            device=args.device,
            opset=args.opset,
            dynamic=args.dynamic,
            simplify=args.simplify,
        )
    )

    import onnx

    onnx.checker.check_model(onnx.load(output))
    print(f"Exported and validated ONNX model: {output}")


if __name__ == "__main__":
    main()
