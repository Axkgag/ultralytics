# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
import random
from copy import copy
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from ultralytics.data import EpochSplitSampler, build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset, get_split_fraction
from ultralytics.engine.results import ElevatorAttributes
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.modules import ElevatorDetect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, LOCAL_RANK, RANK, nms
from ultralytics.utils.ops import xyxy2xywh
from ultralytics.utils.patches import override_configs
from ultralytics.utils.plotting import plot_images, plot_labels
from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model


class DetectionTrainer(BaseTrainer):
    """A class extending the BaseTrainer class for training based on a detection model.

    This trainer specializes in object detection tasks, handling the specific requirements for training YOLO models for
    object detection including dataset building, data loading, preprocessing, and model configuration.

    Attributes:
        model (DetectionModel): The YOLO detection model being trained.
        data (dict): Dictionary containing dataset information including class names and number of classes.
        loss_names (tuple): Names of the loss components, derived from the loss dict returned by the criterion.

    Methods:
        build_dataset: Build YOLO dataset for training or validation.
        get_dataloader: Construct and return dataloader for the specified mode.
        preprocess_batch: Preprocess a batch of images by scaling and converting to float.
        set_model_attributes: Set model attributes based on dataset information.
        get_model: Return a YOLO detection model.
        get_validator: Return a validator for model evaluation.
        progress_string: Return a formatted string of training progress.
        plot_training_samples: Plot training samples with their annotations.
        plot_training_labels: Create a labeled training plot of the YOLO model.
        auto_batch: Calculate optimal batch size based on model memory requirements.

    Examples:
        >>> from ultralytics.models.yolo.detect import DetectionTrainer
        >>> args = dict(model="yolo26n.pt", data="coco8.yaml", epochs=3)
        >>> trainer = DetectionTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        """Initialize a DetectionTrainer object for training YOLO object detection models.

        Args:
            cfg (dict, optional): Default configuration dictionary containing training parameters.
            overrides (dict, optional): Dictionary of parameter overrides for the default configuration.
            _callbacks (dict, optional): Dictionary of callback functions to be executed during training.
        """
        super().__init__(cfg, overrides, _callbacks)
        self.target_loader = None
        self.target_iterator = None
        self.pseudo_stats = torch.zeros(8)
        if self.args.target_data and not self.args.unsup:
            raise ValueError("target_data requires unsup=True")
        if self.args.unsup:
            self._validate_unsup_args()

    def _validate_unsup_args(self) -> None:
        """Validate the intentionally narrow ElevatorDetect UDA configuration."""
        if self.args.task != "detect" or not self.data.get("elevator"):
            raise ValueError("unsup=True is supported only for detect datasets with elevator: true")
        if self.args.distill_model:
            raise ValueError("unsup=True cannot be combined with distill_model")
        if self.args.compile or self.args.rect or self.args.multi_scale:
            raise ValueError("unsup=True requires compile=False, rect=False, and multi_scale=0")
        if not isinstance(self.args.batch, int) or self.args.batch < 2:
            raise ValueError("unsup=True requires an explicit integer batch >= 2")
        if self.args.batch // max(self.world_size, 1) < 2:
            raise ValueError("unsup=True requires a per-rank batch of at least 2")
        if not 0 < self.args.unsup_ratio < 1:
            raise ValueError("unsup_ratio must be strictly between 0 and 1")
        if self.args.unsup_weight < 0:
            raise ValueError("unsup_weight must be non-negative")
        if not 0 <= self.args.unsup_warmup_epochs < self.args.epochs:
            raise ValueError("unsup_warmup_epochs must be in [0, epochs)")
        if not 0 < self.args.uda_teacher_decay < 1:
            raise ValueError("uda_teacher_decay must be strictly between 0 and 1")
        for key in ("pseudo_conf", "pseudo_iou", "pseudo_floor_conf"):
            if not 0 < getattr(self.args, key) <= 1:
                raise ValueError(f"{key} must be in (0, 1]")
        if self.args.pseudo_light_conf < 0.5:
            raise ValueError("pseudo_light_conf must be at least 0.5")
        if not 0.5 < self.args.pseudo_max_floor_ratio <= 1:
            raise ValueError("pseudo_max_floor_ratio must be in (0.5, 1]")

    def _setup_train(self):
        """Build training state and make the UDA teacher a slow, fixed-decay EMA."""
        super()._setup_train()
        if self.args.unsup:
            decay = self.args.uda_teacher_decay
            self.ema.decay = lambda _: decay
            if RANK in {-1, 0}:
                LOGGER.info(f"ElevatorDetect UDA teacher initialized with fixed EMA decay={decay:g}")

    def _domain_batch_sizes(self, batch_size: int) -> tuple[int, int]:
        """Return non-empty source and target batch sizes for one rank."""
        if batch_size < 2:
            raise ValueError("unsup=True requires a per-rank batch of at least 2")
        target = round(batch_size * self.args.unsup_ratio)
        target = min(max(target, 1), batch_size - 1)
        return batch_size - target, target

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Build YOLO Dataset for training or validation.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): 'train' mode or 'val' mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for 'rect' mode.

        Returns:
            (Dataset): YOLO dataset object configured for the specified mode.
        """
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        if self.args.unsup and not self.args.target_data and mode == "train":
            with override_configs(
                self.args, overrides={"mosaic": 0.0, "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0}
            ) as args:
                return build_yolo_dataset(args, img_path, batch, self.data, mode=mode, rect=False, stride=gs)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """Construct and return dataloader for the specified mode.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Number of images per batch.
            rank (int): Process rank for distributed training.
            mode (str): 'train' for training dataloader, 'val' for validation dataloader.

        Returns:
            (DataLoader): PyTorch dataloader object.
        """
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        if self.args.unsup and mode == "train":
            batch_size, _ = self._domain_batch_sizes(batch_size)
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        sampler = None
        if self.args.unsup and not self.args.target_data and mode == "train":
            sampler = EpochSplitSampler(
                dataset,
                self.args.unsup_ratio,
                target=False,
                num_replicas=max(self.world_size, 1),
                rank=max(rank, 0),
                seed=self.args.seed,
            )
            shuffle = False
        if getattr(dataset, "rect", False) and shuffle and not np.all(dataset.batch_shapes == dataset.batch_shapes[0]):
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=shuffle,
            rank=rank,
            drop_last=self.args.compile and mode == "train",
            device=self.device,
            sampler=sampler,
        )

    def preprocess_batch(self, batch: dict) -> dict:
        """Preprocess a batch of images by scaling and converting to float.

        Args:
            batch (dict): Dictionary containing batch data with 'img' tensor.

        Returns:
            (dict): Preprocessed batch with normalized images.
        """
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type not in {"cpu", "mps"})
        batch["img"] = batch["img"].float() / 255
        if self.args.multi_scale > 0.0:
            imgs = batch["img"]
            sz = (
                random.randrange(
                    max(self.stride, int(self.args.imgsz * (1.0 - self.args.multi_scale))),  # min imgsz
                    int(self.args.imgsz * (1.0 + self.args.multi_scale) + self.stride),  # max imgsz
                )
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def _strong_augment(self, images: torch.Tensor) -> torch.Tensor:
        """Apply target-only photometric perturbations without changing pseudo-box geometry."""
        images = images.clone()
        batch = images.shape[0]
        brightness = torch.empty(batch, 1, 1, 1, device=images.device).uniform_(0.75, 1.25)
        contrast = torch.empty(batch, 1, 1, 1, device=images.device).uniform_(0.75, 1.25)
        saturation = torch.empty(batch, 1, 1, 1, device=images.device).uniform_(0.75, 1.25)
        mean = images.mean((2, 3), keepdim=True)
        images = (images - mean) * contrast + mean
        gray = images.mean(1, keepdim=True)
        images = (images - gray) * saturation + gray
        images *= brightness
        gray_mask = torch.rand(batch, 1, 1, 1, device=images.device) < 0.1
        images = torch.where(gray_mask, images.mean(1, keepdim=True).expand_as(images), images)
        blur_mask = torch.rand(batch, 1, 1, 1, device=images.device) < 0.2
        blurred = nn.functional.avg_pool2d(images, kernel_size=3, stride=1, padding=1)
        images = torch.where(blur_mask, blurred, images).clamp_(0, 1)
        for image in images:
            if torch.rand((), device=image.device) < 0.25:
                h, w = image.shape[-2:]
                erase_h, erase_w = max(round(h * 0.1), 1), max(round(w * 0.1), 1)
                y = int(torch.randint(max(h - erase_h + 1, 1), (), device=image.device))
                x = int(torch.randint(max(w - erase_w + 1, 1), (), device=image.device))
                image[:, y : y + erase_h, x : x + erase_w] = image.mean()
        return images

    @staticmethod
    def _split_predictions(predictions, count: int):
        """Split nested ElevatorDetect training predictions along their batch dimension."""
        if isinstance(predictions, torch.Tensor):
            return predictions[:count], predictions[count:]
        if isinstance(predictions, dict):
            split = {key: DetectionTrainer._split_predictions(value, count) for key, value in predictions.items()}
            return ({key: value[0] for key, value in split.items()}, {key: value[1] for key, value in split.items()})
        if isinstance(predictions, (list, tuple)):
            split = [DetectionTrainer._split_predictions(value, count) for value in predictions]
            constructor = type(predictions)
            return constructor(value[0] for value in split), constructor(value[1] for value in split)
        raise TypeError(f"Unsupported prediction container: {type(predictions).__name__}")

    def _build_pseudo_batch(self, images: torch.Tensor) -> dict[str, torch.Tensor] | None:
        """Convert teacher ElevatorDetect output into native detector and elevator targets."""
        teacher = self.ema.ema
        teacher.eval()
        with torch.no_grad():
            output = teacher(images)
            inference = output[0] if isinstance(output, (list, tuple)) else output
            if inference.ndim != 3 or inference.shape[1] != 23:
                raise RuntimeError(f"ElevatorDetect teacher must output BCN with 23 channels, got {inference.shape}")
            detections = nms.non_max_suppression(
                inference.clone(),
                conf_thres=self.args.pseudo_conf,
                iou_thres=self.args.pseudo_iou,
                agnostic=True,
                max_det=self.args.pseudo_max_boxes + 1,
                nc=2,
            )

        height, width = images.shape[-2:]
        accepted_images, boxes, classes, elevators, batch_indices, confidences = [], [], [], [], [], []
        self.pseudo_stats[0] += len(images)
        floor_encodings = torch.tensor(
            [(slot1, slot2) for _, slot1, slot2 in ElevatorAttributes.floor_encodings], device=images.device
        )
        for image_index, detection in enumerate(detections):
            if not len(detection):
                continue
            detection = detection[torch.isfinite(detection).all(1)]
            if not len(detection):
                continue
            detection[:, [0, 2]].clamp_(0, width)
            detection[:, [1, 3]].clamp_(0, height)
            size = detection[:, 2:4] - detection[:, :2]
            detection = detection[(size >= 2).all(1)]
            if not len(detection):
                continue

            class_ids = detection[:, 5].long()
            attributes = ElevatorAttributes(detection[:, 6:23], (height, width))
            floor_scores = attributes.floor_scores
            floor_confidence, floor_index = floor_scores.max(1)
            keep = (class_ids != 0) | (floor_confidence >= self.args.pseudo_floor_conf)
            detection, class_ids, floor_index = detection[keep], class_ids[keep], floor_index[keep]
            if not len(detection):
                continue
            if len(detection) > self.args.pseudo_max_boxes:
                self.pseudo_stats[6] += 1
                continue

            pairs = floor_encodings[floor_index]
            floor_mask = class_ids == 0
            slot1 = torch.where(floor_mask, pairs[:, 0], 0)
            slot2 = torch.where(floor_mask, pairs[:, 1], 0)
            light_probability = detection[:, 22]
            light = (light_probability >= 0.5).long()
            light_valid = (
                (light_probability >= self.args.pseudo_light_conf)
                | (light_probability <= 1 - self.args.pseudo_light_conf)
            ).long()
            xywh = xyxy2xywh(detection[:, :4])
            xywh[:, [0, 2]] /= width
            xywh[:, [1, 3]] /= height

            pseudo_index = len(accepted_images)
            accepted_images.append(image_index)
            boxes.append(xywh)
            classes.append(class_ids[:, None].float())
            elevators.append(torch.stack((slot1, slot2, light, light_valid), 1).float())
            batch_indices.append(torch.full((len(detection),), pseudo_index, device=images.device))
            confidences.append(detection[:, 4])

        if not accepted_images:
            return None
        classes = torch.cat(classes)
        floor_mask = classes.squeeze(1).long() == 0
        min_guard_boxes = max(20, len(accepted_images) * 2)
        if (
            len(classes) >= min_guard_boxes
            and floor_mask.float().mean() > self.args.pseudo_max_floor_ratio
        ):
            self.pseudo_stats[7] += 1
            return None
        elevators = torch.cat(elevators)
        confidences = torch.cat(confidences)
        self.pseudo_stats[1] += len(accepted_images)
        self.pseudo_stats[2] += len(classes)
        self.pseudo_stats[3] += confidences.sum().item()
        self.pseudo_stats[4] += floor_mask.sum().item()
        self.pseudo_stats[5] += elevators[:, 3].sum().item()
        return {
            "img": self._strong_augment(images[accepted_images]),
            "bboxes": torch.cat(boxes),
            "cls": classes,
            "elevator": elevators,
            "batch_idx": torch.cat(batch_indices),
        }

    def _next_target_batch(self) -> dict:
        """Return the next target batch, cycling the target loader when required."""
        try:
            return next(self.target_iterator)
        except StopIteration:
            self.target_iterator = iter(self.target_loader)
            return next(self.target_iterator)

    def _unsup_loss_weight(self) -> float:
        """Return the epoch-level target loss weight after source-only warmup."""
        if self.epoch < self.args.unsup_warmup_epochs:
            return 0.0
        return self.args.unsup_weight * min((self.epoch - self.args.unsup_warmup_epochs + 1) / 5, 1.0)

    def compute_loss(self, batch):
        """Compute source supervision plus filtered ElevatorDetect target pseudo supervision."""
        if not self.args.unsup or self._unsup_loss_weight() == 0:
            return super().compute_loss(batch)
        target_batch = self.preprocess_batch(self._next_target_batch())
        pseudo_batch = self._build_pseudo_batch(target_batch["img"])
        if pseudo_batch is None:
            return super().compute_loss(batch)

        source_count = len(batch["img"])
        predictions = self.model(torch.cat((batch["img"], pseudo_batch["img"])))
        source_predictions, target_predictions = self._split_predictions(predictions, source_count)
        model = unwrap_model(self.model)
        source_loss, source_items = model.loss(batch, source_predictions)
        target_loss, target_items = model.loss(pseudo_batch, target_predictions)
        weight = self._unsup_loss_weight()
        target_scale = source_count / len(pseudo_batch["img"])
        loss = source_loss + weight * target_scale * target_loss
        items = {key: value + weight * target_items[key] for key, value in source_items.items()}
        return loss, items

    def set_model_attributes(self):
        """Set model attributes based on dataset information."""
        # Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data["nc"]  # attach number of classes to model
        self.model.names = self.data["names"]  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        if getattr(self.model, "end2end", False):
            self.model.set_head_attr(max_det=self.args.max_det)
        if self.args.unsup and not isinstance(self.model.model[-1], ElevatorDetect):
            raise ValueError("unsup=True requires an ElevatorDetect model head")

    def set_model_names_for_load(self, model):
        """Set target dataset names before loading weights so cls heads can remap by name."""
        if getattr(self.args, "cls_remap", True) and self.data.get("names"):
            model.names = self.data["names"]
        return model

    def get_class_counts(self):
        """Return per-class instance counts from the training dataset labels."""
        classes = np.concatenate([lb["cls"].flatten() for lb in self.train_loader.dataset.labels], 0)
        return np.bincount(classes.astype(int), minlength=self.data["nc"]).astype(np.float32)

    def compute_class_weights(self, class_counts):
        """Convert class counts to inverse-frequency weights raised to the power of cls_pw."""
        class_counts = np.where(class_counts == 0, 1.0, class_counts)
        return (1.0 / class_counts) ** self.args.cls_pw  # apply power directly

    def set_class_weights(self):
        """Compute and set class weights for handling class imbalance.

        Class weights are computed based on inverse class frequency in the training dataset,
        raised to the power of cls_pw (0 < cls_pw <= 1 dampens; values are restricted to the range [0, 1]).
        Final weights are normalized so their mean equals 1.0.
        """
        assert 0 <= self.args.cls_pw <= 1.0, "cls_pw must be in the range [0, 1]"
        if self.args.cls_pw == 0.0:
            return
        class_counts = self.get_class_counts()
        if not class_counts.any():  # nothing counted (e.g. missing/unreadable masks); keep default weights
            return
        weights = self.compute_class_weights(class_counts)
        weights = weights / weights.mean()  # normalize so mean equals 1.0
        model = unwrap_model(self.model)
        if hasattr(model, "student_model"):
            model = model.student_model  # distillation: the student model builds the loss criterion
        model.class_weights = torch.from_numpy(weights).to(self.device)
        LOGGER.info(f"Class weights: {model.class_weights.cpu().numpy().round(3)}")

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        """Return a YOLO detection model.

        Args:
            cfg (str, optional): Path to model configuration file.
            weights (str, optional): Path to model weights.
            verbose (bool): Whether to display model information.

        Returns:
            (DetectionModel): YOLO detection model.
        """
        model = self.set_model_names_for_load(
            DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return a DetectionValidator for YOLO model validation."""
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def _build_train_pipeline(self):
        """Build the training pipeline and align the default detection limit with observed dataset object counts."""
        if self.target_loader is not None:
            self.target_loader.close()
            self.target_loader = None
        super()._build_train_pipeline()
        if self.args.unsup:
            local_batch = self.batch_size // max(self.world_size, 1)
            _, target_batch = self._domain_batch_sizes(local_batch)
            if self.args.target_data:
                with torch_distributed_zero_first(LOCAL_RANK):
                    target_data = check_det_dataset(self.args.target_data)
                if not target_data.get("elevator"):
                    raise ValueError("target_data must define elevator: true")
                if target_data.get("names") != self.data.get("names"):
                    raise ValueError("target_data must use the same classes as the source ElevatorDetect dataset")
                target_path = target_data.get("train")
            else:
                target_data, target_path = self.data, self.data["train"]
            stride = max(int(unwrap_model(self.model).stride.max()), 32)
            target_fraction = None if self.args.target_data else get_split_fraction(self.args.fraction, "train")
            with torch_distributed_zero_first(LOCAL_RANK):
                target_dataset = build_yolo_dataset(
                    self.args,
                    target_path,
                    target_batch,
                    target_data,
                    mode="target",
                    rect=False,
                    stride=stride,
                    fraction=target_fraction,
                )
            target_sampler = None
            if not self.args.target_data:
                if target_dataset.im_files != self.train_loader.dataset.im_files:
                    raise RuntimeError("internal UDA source and target datasets must contain identical image files")
                target_sampler = EpochSplitSampler(
                    target_dataset,
                    self.args.unsup_ratio,
                    target=True,
                    num_replicas=max(self.world_size, 1),
                    rank=max(LOCAL_RANK, 0),
                    seed=self.args.seed,
                )
            self.target_loader = build_dataloader(
                target_dataset,
                batch=target_batch,
                workers=self.args.workers,
                shuffle=target_sampler is None,
                rank=LOCAL_RANK,
                device=self.device,
                sampler=target_sampler,
            )
            self.target_iterator = iter(self.target_loader)
            LOGGER.info(
                f"ElevatorDetect UDA enabled: source_batch={self.train_loader.batch_size}, "
                f"target_batch={self.target_loader.batch_size}, "
                f"target={'external' if self.args.target_data else 'epoch source split'}"
            )
        if self.args.task in {"detect", "segment", "pose", "obb"}:
            datasets = {"train": self.train_loader.dataset, "val": self.test_loader.dataset}
            yolo.detect.DetectionValidator._check_max_det(self.args, datasets)
            model = unwrap_model(self.model)
            if getattr(model, "end2end", False):
                model.set_head_attr(max_det=self.args.max_det)

    def set_epoch(self, epoch: int) -> None:
        """Advance source and target samplers and discard prefetched internal-split batches."""
        super().set_epoch(epoch)
        if not self.args.unsup or self.target_loader is None:
            return
        target_sampler = self.target_loader.sampler
        if hasattr(target_sampler, "set_epoch"):
            target_sampler.set_epoch(epoch)
        if not self.args.target_data:
            self.train_loader.reset()
        self.target_loader.reset()
        self.target_iterator = iter(self.target_loader)
        self.pseudo_stats.zero_()

    def validate(self):
        """Append target pseudo-label acceptance diagnostics to validation metrics."""
        if not self.args.unsup:
            return super().validate()
        stats = self.pseudo_stats.to(self.device)
        if self.world_size > 1:
            dist.all_reduce(stats)
        seen, accepted, boxes, confidence, floor_boxes, light_valid, dense_rejected, class_rejected = stats.tolist()
        pseudo_metrics = {
            "train/pseudo_boxes": boxes,
            "train/pseudo_images": accepted,
            "train/pseudo_accept_ratio": accepted / max(seen, 1),
            "train/pseudo_mean_conf": confidence / max(boxes, 1),
            "train/pseudo_floor_boxes": floor_boxes,
            "train/pseudo_light_valid_ratio": light_valid / max(boxes, 1),
            "train/pseudo_dense_rejected_images": dense_rejected,
            "train/pseudo_class_guard_rejected_batches": class_rejected,
            "train/unsup_weight": self._unsup_loss_weight(),
        }
        metrics, fitness = super().validate()
        if metrics is not None:
            metrics.update(pseudo_metrics)
        return metrics, fitness

    def progress_string(self):
        """Return a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch: dict[str, Any], ni: int) -> None:
        """Plot training samples with their annotations.

        Args:
            batch (dict[str, Any]): Dictionary containing batch data.
            ni (int): Batch index used for naming the output file.
        """
        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

    def auto_batch(self):
        """Get optimal batch size by calculating memory occupation of model.

        Returns:
            (int): Optimal batch size.
        """
        with override_configs(self.args, overrides={"cache": False}) as self.args:
            train_dataset = self.build_dataset(self.data["train"], mode="train", batch=16)
        max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4  # 4 for mosaic augmentation
        n = len(train_dataset)
        del train_dataset  # free memory
        return super().auto_batch(max_num_obj, dataset_size=n)
