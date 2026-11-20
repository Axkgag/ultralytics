# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from ultralytics.data import YOLOConcatDataset, build_dataloader, build_yolo_dataset
from ultralytics.data.augment import LoadVisualPrompt, VisualPromptCanvas
from ultralytics.data.utils import check_det_dataset, get_split_fraction, img2label_paths
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.nn.modules.head import YOLOEDetect
from ultralytics.nn.tasks import YOLOEModel
from ultralytics.utils import LOGGER, TQDM, YAML, ops
from ultralytics.utils.metrics import ConfusionMatrix, box_iou
from ultralytics.utils.torch_utils import select_device, smart_inference_mode


class YOLOEDetectValidator(DetectionValidator):
    """A validator class for YOLOE detection models that handles both text and visual prompt embeddings.

    This class extends DetectionValidator to provide specialized validation functionality for YOLOE models. It supports
    validation using either text prompts or visual prompt embeddings extracted from training samples, enabling flexible
    evaluation strategies for prompt-based object detection.

    Attributes:
        device (torch.device): The device on which validation is performed.
        args (namespace): Configuration arguments for validation.
        dataloader (DataLoader): DataLoader for validation data.

    Methods:
        get_visual_pe: Extract visual prompt embeddings from training samples.
        preprocess: Preprocess batch data ensuring visuals are on the same device as images.
        get_vpe_dataloader: Create a dataloader for LVIS training visual prompt samples.
        __call__: Run validation using either text or visual prompt embeddings.

    Examples:
        Validate with text prompts
        >>> validator = YOLOEDetectValidator()
        >>> stats = validator(model=model, load_vp=False)

        Validate with visual prompts
        >>> stats = validator(model=model, refer_data="path/to/data.yaml", load_vp=True)
    """

    @smart_inference_mode()
    def get_visual_pe(self, dataloader: torch.utils.data.DataLoader, model: YOLOEModel) -> torch.Tensor:
        """Extract visual prompt embeddings from training samples.

        This method processes a dataloader to compute visual prompt embeddings for each class using a YOLOE model. It
        normalizes the embeddings and handles cases where no samples exist for a class by setting their embeddings to
        zero.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader providing training samples.
            model (YOLOEModel): The YOLOE model from which to extract visual prompt embeddings.

        Returns:
            (torch.Tensor): Visual prompt embeddings with shape (1, num_classes, embed_dim).
        """
        assert isinstance(model, YOLOEModel)
        names = [name.split("/", 1)[0] for name in list(dataloader.dataset.data["names"].values())]
        visual_pe = torch.zeros(len(names), model.model[-1].embed, device=self.device)
        cls_visual_num = torch.zeros(len(names))

        desc = "Get visual prompt embeddings from samples"

        # Count samples per class
        for batch in dataloader:
            cls = batch["cls"].squeeze(-1).to(torch.int).unique()
            count = torch.bincount(cls, minlength=len(names))
            cls_visual_num += count

        cls_visual_num = cls_visual_num.to(self.device)

        # Extract visual prompt embeddings
        pbar = TQDM(dataloader, total=len(dataloader), desc=desc)
        for batch in pbar:
            batch = self.preprocess(batch)
            preds = model.get_visual_pe(batch["img"], visual=batch["visuals"])  # (B, max_n, embed_dim)

            batch_idx = batch["batch_idx"]
            for i in range(preds.shape[0]):
                cls = batch["cls"][batch_idx == i].squeeze(-1).to(torch.int).unique(sorted=True)
                pad_cls = torch.ones(preds.shape[1], device=self.device) * -1
                pad_cls[: cls.shape[0]] = cls
                for c in cls:
                    visual_pe[c] += preds[i][pad_cls == c].sum(0) / cls_visual_num[c]

        # Normalize embeddings for classes with samples, set others to zero
        visual_pe[cls_visual_num != 0] = F.normalize(visual_pe[cls_visual_num != 0], dim=-1, p=2)
        visual_pe[cls_visual_num == 0] = 0
        return visual_pe.unsqueeze(0)

    def get_vpe_dataloader(self, data: dict[str, Any]) -> torch.utils.data.DataLoader:
        """Create a dataloader for LVIS training visual prompt samples.

        This method prepares a dataloader for visual prompt embeddings (VPE) using the specified dataset. It applies
        necessary transformations including LoadVisualPrompt and configurations to the dataset for validation purposes.

        Args:
            data (dict): Dataset configuration dictionary containing paths and settings.

        Returns:
            (torch.utils.data.DataLoader): The dataloader for visual prompt samples.
        """
        dataset = build_yolo_dataset(
            self.args,
            data.get(self.args.split, data.get("val")),
            self.args.batch,
            data,
            mode="val",
            rect=False,
            fraction=get_split_fraction(self.args.fraction, self.args.split or "val"),
        )
        if isinstance(dataset, YOLOConcatDataset):
            for d in dataset.datasets:
                d.transforms.append(LoadVisualPrompt())
        else:
            dataset.transforms.append(LoadVisualPrompt())
        return build_dataloader(
            dataset,
            self.args.batch,
            self.args.workers,
            shuffle=False,
            rank=-1,
            device=self.device,
        )

    def __call__(
        self,
        trainer: Any | None = None,
        model: YOLOEModel | str | None = None,
        refer_data: str | None = None,
        load_vp: bool = False,
    ) -> dict[str, Any]:
        """Run validation on the model using either text or visual prompt embeddings.

        This method validates the model using either text prompts or visual prompts, depending on the load_vp flag. It
        supports validation during training (using a trainer object) or standalone validation with a provided model. For
        visual prompts, reference data can be specified to extract embeddings from a different dataset.

        Args:
            trainer (object, optional): Trainer object containing the model and device.
            model (YOLOEModel | str, optional): Model to validate. Required if trainer is not provided.
            refer_data (str, optional): Path to reference data for visual prompts.
            load_vp (bool): Whether to load visual prompts. If False, text prompts are used.

        Returns:
            (dict): Validation statistics containing metrics computed during validation.
        """
        if trainer is not None:
            self.device = trainer.device
            model = trainer.ema.ema
            names = [name.split("/", 1)[0] for name in list(self.dataloader.dataset.data["names"].values())]

            if load_vp:
                LOGGER.info("Validate using the visual prompt.")
                self.args.quantize = None
                # Directly use the same dataloader for visual embeddings extracted during training
                vpe = self.get_visual_pe(self.dataloader, model)
                model.set_classes(names, vpe)
            else:
                LOGGER.info("Validate using the text prompt.")
                tpe = model.get_text_pe(names)
                model.set_classes(names, tpe)
            stats = super().__call__(trainer, model)
        else:
            if refer_data is not None:
                assert load_vp, "Refer data is only used for visual prompt validation."
            self.device = select_device(self.args.device, verbose=False)

            if isinstance(model, (str, Path)):
                from ultralytics.nn.tasks import load_checkpoint

                model, _ = load_checkpoint(model, device=self.device)  # model, ckpt
            model.eval().to(self.device)
            data = check_det_dataset(refer_data or self.args.data)
            names = [name.split("/", 1)[0] for name in list(data["names"].values())]

            if refer_data is not None:
                eval_data = check_det_dataset(self.args.data)
                eval_names = [name.split("/", 1)[0] for name in list(eval_data["names"].values())]
                if names != eval_names:
                    LOGGER.warning(
                        f"Class names from refer data {names} do not match evaluation dataset {eval_names}. "
                        f"This may lead to incorrect validation results."
                    )

            if load_vp:
                LOGGER.info("Validate using the visual prompt.")
                self.args.quantize = None
                dataloader = self.get_vpe_dataloader(data)
                vpe = self.get_visual_pe(dataloader, model)
                model.set_classes(names, vpe)
                stats = super().__call__(model=deepcopy(model))
            elif isinstance(model.model[-1], YOLOEDetect) and hasattr(model.model[-1], "lrpc"):  # prompt-free
                return super().__call__(trainer, model)
            else:
                LOGGER.info("Validate using the text prompt.")
                tpe = model.get_text_pe(names)
                model.set_classes(names, tpe)
                stats = super().__call__(model=deepcopy(model))
        return stats


class YOLOEPairedValidator(DetectionValidator):
    """Validate sampled prompt pairs or every target image against a fixed prompt bank."""

    def __call__(self, trainer=None, model=None, **kwargs):
        """Run paired validation during training and restore the EMA model's prompt state afterwards."""
        self.all_prompts = bool(self.args.visual_prompt_bank)
        if self.all_prompts:
            self._load_prompt_specs()
        if trainer is not None:
            self.prompt_model = trainer.ema.ema or trainer.model
        else:
            if isinstance(model, (str, Path)):
                from ultralytics.nn.tasks import load_checkpoint

                model, _ = load_checkpoint(model, device=select_device(self.args.device, verbose=False))
            self.prompt_model = model
        head = self.prompt_model.model[-1]
        old_names, old_nc = self.prompt_model.names, head.nc
        had_pe, old_pe = hasattr(self.prompt_model, "pe"), getattr(self.prompt_model, "pe", None)
        placeholder = next(self.prompt_model.parameters()).new_zeros(1, 1, head.embed)
        self.prompt_model.set_classes(["object0"], placeholder)
        try:
            return super().__call__(trainer, self.prompt_model)
        finally:
            self.prompt_model.names, head.nc = old_names, old_nc
            if had_pe:
                self.prompt_model.pe = old_pe
            else:
                del self.prompt_model.pe

    def _load_prompt_specs(self) -> None:
        """Load and validate the fixed 24-floor prompt manifest."""
        path = Path(self.args.visual_prompt_bank).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Visual prompt bank does not exist: {path}")
        bank = YAML.load(path)
        prompts = bank.get("prompts", [])
        if len(prompts) != 24:
            raise ValueError(f"All-query validation requires 24 prompts, found {len(prompts)} in {path}")
        self.prompt_specs = []
        for prompt in prompts:
            image = path.parent / prompt["image"]
            slots = tuple(map(int, prompt["slots"]))
            if not image.is_file():
                raise FileNotFoundError(f"Prompt image does not exist: {image}")
            self.prompt_specs.append({"floor": str(prompt["floor"]), "slots": slots, "image": image})
        self.query_ids = [prompt["slots"] for prompt in self.prompt_specs]
        if len(set(self.query_ids)) != len(self.query_ids):
            raise ValueError(f"Prompt bank contains duplicate slot identities: {path}")
        self.query_names = {index: prompt["floor"] for index, prompt in enumerate(self.prompt_specs)}
        self.slot_to_query = {slots: index for index, slots in enumerate(self.query_ids)}
        self.prompt_bank_path = path

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None):
        """Build the same deterministic paired dataset used by the trainer's validation loader."""
        fraction = get_split_fraction(self.args.fraction, self.args.split or "val")
        return build_yolo_dataset(
            self.args,
            img_path,
            batch,
            self.data,
            mode=mode,
            rect=False,
            stride=self.stride,
            paired_visual_prompt=True,
            all_visual_prompts=self.all_prompts,
            visual_prompt_queries=self.query_ids if self.all_prompts else None,
            fraction=fraction,
        )

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize standard COCO-style metrics and cache fixed prompt embeddings."""
        super().init_metrics(model)
        if not self.all_prompts:
            return
        self.names = self.query_names
        self.metrics.names = self.query_names
        self.confusion_matrix = ConfusionMatrix(names=self.query_names, save_matches=False)
        self.confusion_matrix_conf = self.args.visual_prompt_fail_conf
        self.episode_records = []
        self.target_cache = {}
        self._encode_prompt_bank()

    def _encode_prompt_bank(self) -> None:
        """Encode all fixed prompts once and cache one embedding per floor."""
        data_cfg = YAML.load(self.args.data)
        cfg = data_cfg.get("visual_prompt", {})
        canvas_builder = VisualPromptCanvas(
            (self.args.imgsz, self.args.imgsz),
            sizes=tuple(cfg.get("sizes", (64, 96, 160))),
            size_jitter=0.0,
            degrees=0.0,
            perspective=0.0,
            brightness=0.0,
            contrast=0.0,
            blur=0.0,
        )
        images, visuals = [], []
        for prompt in self.prompt_specs:
            crop = cv2.imread(str(prompt["image"]))
            if crop is None:
                raise FileNotFoundError(f"Unable to read prompt image: {prompt['image']}")
            canvas, boxes = canvas_builder(crop)
            image = torch.from_numpy(np.ascontiguousarray(canvas[..., ::-1].transpose(2, 0, 1)))
            visual = LoadVisualPrompt().get_visuals(
                np.zeros(len(boxes), dtype=np.int64), canvas.shape[:2], bboxes=boxes
            )
            images.append(image)
            visuals.append(visual)
        dtype = next(self.prompt_model.parameters()).dtype
        images = torch.stack(images).to(self.device, dtype=dtype) / 255
        visuals = torch.stack(visuals).to(self.device, dtype=dtype)
        embeddings = self.prompt_model.get_visual_pe(images, visuals)
        self.prompt_embeddings = embeddings[:, 0]
        LOGGER.info(f"Cached {len(self.prompt_specs)} fixed visual-prompt embeddings from {self.prompt_bank_path}")

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Normalize both image branches and install the current batch's visual prompt embeddings."""
        batch = super().preprocess(batch)
        dtype = next(self.prompt_model.parameters()).dtype
        if self.all_prompts:
            vpe = self.prompt_embeddings[batch["query_idx"].long()].unsqueeze(1).to(dtype)
            self.prompt_model.set_classes(["object0"], vpe)
            return batch
        batch["refer_img"] = batch["refer_img"].to(dtype) / 255
        batch["visuals"] = batch["visuals"].to(dtype)
        vpe = self.prompt_model.get_visual_pe(batch["refer_img"], batch["visuals"])
        self.prompt_model.set_classes(["object0"], vpe)
        return batch

    def update_metrics(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]) -> None:
        """Remap dynamic object0 predictions to floor classes and retain episode outputs."""
        if not self.all_prompts:
            return super().update_metrics(preds, batch)
        query_idx = batch["query_idx"].long()
        metric_batch = batch.copy()
        metric_batch["cls"] = query_idx[batch["batch_idx"].long(), None].to(batch["cls"].dtype)
        source_files = list(batch["im_file"])
        metric_batch["im_file"] = [
            str(Path(path).with_name(f"{Path(path).stem}__query_{self.query_names[int(query)]}{Path(path).suffix}"))
            for path, query in zip(source_files, query_idx)
        ]
        metric_preds = []
        for sample_index, pred in enumerate(preds):
            remapped = pred.copy()
            remapped["cls"] = torch.full_like(pred["cls"], int(query_idx[sample_index]))
            metric_preds.append(remapped)
            self._record_episode(sample_index, pred, batch, source_files[sample_index])
        super().update_metrics(metric_preds, metric_batch)

    def _record_episode(self, sample_index: int, pred: dict[str, torch.Tensor], batch: dict, source: str) -> None:
        """Store native-coordinate predictions and categorized failures for one target-query episode."""
        query_idx = int(batch["query_idx"][sample_index])
        pbatch = self._prepare_batch(sample_index, batch)
        pred_boxes = ops.scale_boxes(
            pbatch["imgsz"], pred["bboxes"].clone(), pbatch["ori_shape"], pbatch["ratio_pad"]
        ).cpu()
        predictions = [
            {"box": [round(float(x), 3) for x in box], "confidence": round(float(conf), 6)}
            for box, conf in zip(pred_boxes, pred["conf"].cpu())
        ]
        targets = self._load_all_targets(source, pbatch["ori_shape"])
        query_targets = [target for target in targets if target["query_idx"] == query_idx]
        failures = self._classify_failures(query_idx, predictions, targets, query_targets)
        self.episode_records.append(
            {
                "image": str(source),
                "query_idx": query_idx,
                "query": self.query_names[query_idx],
                "query_present": bool(query_targets),
                "gt_boxes": [target["box"] for target in query_targets],
                "predictions": predictions,
                "failures": failures,
                "_targets": targets,
            }
        )

    def _load_all_targets(self, image_path: str, shape: tuple[int, int]) -> list[dict]:
        """Load and cache every labeled floor button in one target image."""
        if image_path in self.target_cache:
            return self.target_cache[image_path]
        height, width = map(int, shape)
        label_path = Path(img2label_paths([image_path])[0])
        targets = []
        if label_path.is_file():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                row = [float(value) for value in line.split()]
                if not row or int(row[0]) != 0:
                    continue
                _, cx, cy, box_width, box_height, slot1, slot2, *_ = row
                query_idx = self.slot_to_query.get((int(slot1), int(slot2)))
                if query_idx is None:
                    continue
                targets.append(
                    {
                        "query_idx": query_idx,
                        "floor": self.query_names[query_idx],
                        "box": [
                            (cx - box_width / 2) * width,
                            (cy - box_height / 2) * height,
                            (cx + box_width / 2) * width,
                            (cy + box_height / 2) * height,
                        ],
                    }
                )
        self.target_cache[image_path] = targets
        return targets

    def _classify_failures(
        self, query_idx: int, predictions: list[dict], targets: list[dict], query_targets: list[dict]
    ) -> list[dict]:
        """Categorize failures using the same greedy IoU matching policy as the elevator validators."""
        selected = [
            index
            for index, pred in enumerate(predictions)
            if pred["confidence"] >= self.args.visual_prompt_fail_conf
        ]
        pred_boxes = torch.tensor([predictions[index]["box"] for index in selected], dtype=torch.float32)
        query_boxes = torch.tensor([target["box"] for target in query_targets], dtype=torch.float32)
        iou_query = (
            box_iou(query_boxes, pred_boxes)
            if len(query_boxes) and len(pred_boxes)
            else torch.zeros((len(query_boxes), len(pred_boxes)))
        )
        candidates = torch.nonzero(iou_query >= self.args.visual_prompt_fail_iou, as_tuple=False)
        candidates = sorted(
            ((float(iou_query[gt, pred]), int(gt), int(pred)) for gt, pred in candidates), reverse=True
        )
        used_targets, used_predictions = set(), set()
        for _, target_index, prediction_index in candidates:
            if target_index not in used_targets and prediction_index not in used_predictions:
                used_targets.add(target_index)
                used_predictions.add(prediction_index)

        failures = [
            {"category": "false_negative", "target_index": index, "severity": 1.0}
            for index in range(len(query_targets))
            if index not in used_targets
        ]
        other_targets = [target for target in targets if target["query_idx"] != query_idx]
        other_boxes = torch.tensor([target["box"] for target in other_targets], dtype=torch.float32)
        iou_other = (
            box_iou(other_boxes, pred_boxes)
            if len(other_boxes) and len(pred_boxes)
            else torch.zeros((len(other_boxes), len(pred_boxes)))
        )
        for local_index, prediction_index in enumerate(selected):
            if local_index in used_predictions:
                continue
            confidence = predictions[prediction_index]["confidence"]
            if len(other_targets) and float(iou_other[:, local_index].max()) >= self.args.visual_prompt_fail_iou:
                target_index = int(iou_other[:, local_index].argmax())
                target = other_targets[target_index]
                overlap = float(iou_other[target_index, local_index])
                failures.append(
                    {
                        "category": "wrong_floor",
                        "prediction_index": prediction_index,
                        "overlapped_floor": target["floor"],
                        "iou": round(overlap, 6),
                        "severity": confidence * overlap,
                    }
                )
            elif len(query_targets) and float(iou_query[:, local_index].max()) >= 0.1:
                overlap = float(iou_query[:, local_index].max())
                failures.append(
                    {
                        "category": "localization",
                        "prediction_index": prediction_index,
                        "iou": round(overlap, 6),
                        "severity": confidence * (1 - overlap),
                    }
                )
            else:
                category = "false_positive_absent" if not query_targets else "false_positive"
                failures.append(
                    {"category": category, "prediction_index": prediction_index, "severity": confidence}
                )
        return failures

    def get_stats(self) -> dict[str, Any]:
        """Finalize standard metrics and save exhaustive prompt reports."""
        stats = super().get_stats()
        if self.all_prompts:
            stats["metrics/mAP75(B)"] = float(self.metrics.box.map75)
            self._save_all_prompt_outputs(stats)
        return stats

    def _save_all_prompt_outputs(self, stats: dict[str, Any]) -> None:
        """Save global metrics, per-floor AP, raw predictions, and optional failure visualizations."""
        report = {
            "prompt_bank": str(self.prompt_bank_path),
            "episodes": len(self.episode_records),
            "evaluated_floors": len(self.metrics.ap_class_index),
            **{key: float(value) for key, value in stats.items()},
        }
        (self.save_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        result_index = {int(class_index): index for index, class_index in enumerate(self.metrics.ap_class_index)}
        rows = []
        for query_idx, floor in self.query_names.items():
            result = result_index.get(query_idx)
            rows.append(
                {
                    "query_idx": query_idx,
                    "floor": floor,
                    "images": sum(record["query_idx"] == query_idx for record in self.episode_records),
                    "instances": int(self.metrics.nt_per_class[query_idx]),
                    "precision": None if result is None else float(self.metrics.box.p[result]),
                    "recall": None if result is None else float(self.metrics.box.r[result]),
                    "AP50": None if result is None else float(self.metrics.box.all_ap[result, 0]),
                    "AP75": None if result is None else float(self.metrics.box.all_ap[result, 5]),
                    "AP50-95": None if result is None else float(self.metrics.box.all_ap[result].mean()),
                }
            )
        with open(self.save_dir / "per_floor.csv", "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        with open(self.save_dir / "predictions.jsonl", "w", encoding="utf-8") as file:
            for record in self.episode_records:
                file.write(json.dumps({key: value for key, value in record.items() if not key.startswith("_")}) + "\n")
        if self.args.visual_prompt_fail_cases:
            self._save_fail_cases()

    def _save_fail_cases(self) -> None:
        """Save severity-ranked failure composites and a structured summary."""
        directory = self.save_dir / "fail_cases"
        cases = defaultdict(list)
        failed_records = []
        for record in self.episode_records:
            if record["failures"]:
                failed_records.append({key: value for key, value in record.items() if not key.startswith("_")})
            for failure in record["failures"]:
                cases[failure["category"]].append((failure["severity"], record, failure))
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "cases.jsonl", "w", encoding="utf-8") as file:
            for record in failed_records:
                file.write(json.dumps(record) + "\n")
        summary = Counter(episodes=len(self.episode_records), failed_episodes=len(failed_records))
        for category, category_cases in cases.items():
            summary[f"{category}_cases"] = len(category_cases)
            summary[f"{category}_images"] = len(
                {(record["image"], record["query_idx"]) for _, record, _ in category_cases}
            )
            for rank, (_, record, failure) in enumerate(
                sorted(category_cases, key=lambda item: item[0], reverse=True)[: self.args.visual_prompt_fail_max]
            ):
                destination = directory / category
                if category == "wrong_floor":
                    destination /= f"{record['query']}_to_{failure['overlapped_floor']}"
                else:
                    destination /= f"floor_{record['query']}"
                destination.mkdir(parents=True, exist_ok=True)
                output = destination / f"{Path(record['image']).stem}__query_{record['query']}__{rank:04d}.jpg"
                self._draw_fail_case(record, output)
        summary.update(
            confidence_threshold=float(self.args.visual_prompt_fail_conf),
            iou_threshold=float(self.args.visual_prompt_fail_iou),
            max_per_category=int(self.args.visual_prompt_fail_max),
        )
        (directory / "summary.json").write_text(json.dumps(dict(summary), indent=2) + "\n", encoding="utf-8")

    def _draw_fail_case(self, record: dict, output: Path) -> None:
        """Render prompt, all target identities, predictions, and missed query targets."""
        image = cv2.imread(record["image"])
        if image is None:
            return
        gt_panel, pred_panel = image.copy(), image.copy()
        query_idx = record["query_idx"]
        targets = record["_targets"]
        thickness = max(round(sum(image.shape[:2]) / 2 * 0.003), 2)
        for target in targets:
            color = (0, 200, 0) if target["query_idx"] == query_idx else (150, 150, 150)
            box = [round(value) for value in target["box"]]
            cv2.rectangle(gt_panel, box[:2], box[2:], color, thickness)
            cv2.putText(gt_panel, target["floor"], (box[0], max(box[1] - 5, 16)), 0, 0.6, color, 2)
        query_targets = [target for target in targets if target["query_idx"] == query_idx]
        selected_predictions = [
            pred for pred in record["predictions"] if pred["confidence"] >= self.args.visual_prompt_fail_conf
        ]
        pred_boxes = torch.tensor([pred["box"] for pred in selected_predictions], dtype=torch.float32)
        query_boxes = torch.tensor([target["box"] for target in query_targets], dtype=torch.float32)
        other_targets = [target for target in targets if target["query_idx"] != query_idx]
        other_boxes = torch.tensor([target["box"] for target in other_targets], dtype=torch.float32)
        iou_query = (
            box_iou(query_boxes, pred_boxes)
            if len(query_boxes) and len(pred_boxes)
            else torch.zeros((len(query_boxes), len(pred_boxes)))
        )
        iou_other = (
            box_iou(other_boxes, pred_boxes)
            if len(other_boxes) and len(pred_boxes)
            else torch.zeros((len(other_boxes), len(pred_boxes)))
        )
        matches = torch.nonzero(iou_query >= self.args.visual_prompt_fail_iou, as_tuple=False)
        matches = sorted(((float(iou_query[gt, pred]), int(gt), int(pred)) for gt, pred in matches), reverse=True)
        matched_targets, matched_predictions = set(), set()
        for _, target_index, prediction_index in matches:
            if target_index not in matched_targets and prediction_index not in matched_predictions:
                matched_targets.add(target_index)
                matched_predictions.add(prediction_index)
        for index, prediction in enumerate(selected_predictions):
            color, suffix = (0, 0, 255), "FP"
            if index in matched_predictions:
                color, suffix = (0, 200, 0), "TP"
            elif len(other_targets) and float(iou_other[:, index].max()) >= self.args.visual_prompt_fail_iou:
                target = other_targets[int(iou_other[:, index].argmax())]
                color, suffix = (0, 165, 255), f"->{target['floor']}"
            elif len(query_targets) and float(iou_query[:, index].max()) >= 0.1:
                color, suffix = (0, 255, 255), f"IoU {float(iou_query[:, index].max()):.2f}"
            box = [round(value) for value in prediction["box"]]
            cv2.rectangle(pred_panel, box[:2], box[2:], color, thickness)
            label = f"q{record['query']} {prediction['confidence']:.2f} {suffix}"
            cv2.putText(pred_panel, label, (box[0], max(box[1] - 5, 16)), 0, 0.55, color, 2)
        for index, target in enumerate(query_targets):
            if index not in matched_targets:
                self._draw_dashed_box(pred_panel, target["box"], f"MISS {record['query']}", thickness)
        gt_panel = self._add_header(gt_panel, f"GT query={record['query']}")
        pred_panel = self._add_header(pred_panel, "PREDICT / FAIL")
        prompt_panel = np.full((gt_panel.shape[0], 240, 3), 245, dtype=np.uint8)
        prompt = cv2.imread(str(self.prompt_specs[query_idx]["image"]))
        if prompt is not None:
            gain = min(200 / prompt.shape[0], 200 / prompt.shape[1])
            size = (round(prompt.shape[1] * gain), round(prompt.shape[0] * gain))
            prompt = cv2.resize(prompt, size)
            x, y = (240 - size[0]) // 2, 70
            prompt_panel[y : y + size[1], x : x + size[0]] = prompt
        cv2.putText(prompt_panel, f"PROMPT {record['query']}", (15, 32), 0, 0.75, (20, 20, 20), 2)
        cv2.putText(prompt_panel, f"present={record['query_present']}", (15, 300), 0, 0.6, (20, 20, 20), 2)
        cv2.imwrite(str(output), cv2.hconcat((prompt_panel, gt_panel, pred_panel)))

    @staticmethod
    def _add_header(image: np.ndarray, title: str) -> np.ndarray:
        """Add a title band above one target panel."""
        panel = cv2.copyMakeBorder(image, 44, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(panel, title, (14, 31), 0, 0.8, (25, 25, 25), 2, cv2.LINE_AA)
        return panel

    @staticmethod
    def _draw_dashed_box(image: np.ndarray, box: list[float], label: str, thickness: int) -> None:
        """Draw one missed target as a red dashed rectangle."""
        x1, y1, x2, y2 = (round(value) for value in box)
        dash = max(thickness * 4, 8)
        for x in range(x1, x2, dash * 2):
            cv2.line(image, (x, y1), (min(x + dash, x2), y1), (0, 0, 255), thickness)
            cv2.line(image, (x, y2), (min(x + dash, x2), y2), (0, 0, 255), thickness)
        for y in range(y1, y2, dash * 2):
            cv2.line(image, (x1, y), (x1, min(y + dash, y2)), (0, 0, 255), thickness)
            cv2.line(image, (x2, y), (x2, min(y + dash, y2)), (0, 0, 255), thickness)
        cv2.putText(image, label, (x1, max(y1 - 5, 16)), 0, 0.6, (0, 0, 255), 2)

    def plot_val_samples(self, batch: dict[str, Any], ni: int) -> None:
        """Skip duplicated all-query batch previews while retaining metric plots."""
        if not self.all_prompts:
            super().plot_val_samples(batch, ni)

    def plot_predictions(
        self, batch: dict[str, Any], preds: list[dict[str, torch.Tensor]], ni: int, max_det: int | None = None
    ) -> None:
        """Skip duplicated all-query batch previews while retaining metric plots."""
        if not self.all_prompts:
            super().plot_predictions(batch, preds, ni, max_det)


class YOLOESegValidator(YOLOEDetectValidator, SegmentationValidator):
    """YOLOE segmentation validator that supports both text and visual prompt embeddings."""
