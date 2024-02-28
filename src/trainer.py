import os
import time
import json
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from kornia import augmentation
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from .models.gpt import GPT
from .utils import CfgNode as CN
from .dataset import NeedleDataset
from .logger import Logger


class Trainer:
    def __init__(
        self,
        config: CN,
        model: GPT,
        logger: Logger,
        train_dataset: NeedleDataset,
        test_dataset: NeedleDataset,
        rank: int = 0,
    ):
        self.config = config
        self.model = model
        self.logger = logger
        self.rank = rank
        self.device = self.config.gpu_ids[rank]
        print(f"Rank {rank} is using device {self.device}")

        self.optim_gpt, self.optim_yolox = model.configure_optimizers(config)

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        # Test datasets env ids
        rng = np.random.default_rng(self.config.seed)

        if self.test_dataset is not None:
            image_ids = list(range(len(self.test_dataset)))
            self.test_env_ids = rng.choice(image_ids, size=(self.config.test_samples,))

        if self.train_dataset is not None:
            image_ids = list(range(len(self.train_dataset)))
            self.train_env_ids = rng.choice(image_ids, size=(self.config.test_samples,))

        self.rng = rng

        self.best_metric_history = []

        self.model.to(self.device)

    def ddp_setup(self, rank: int, world_size: int, port: int):
        """Setup distributed training.

        ---
        Args:
            rank: The rank of the current process.
            world_size: The total number of processes.
        """
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)
        init_process_group(backend="nccl", rank=rank, world_size=world_size)

    def save_metrics(self):
        """
        Save metrics of the current checkpoint to a json file
        """
        work_dir = Path(self.config.work_dir)
        log_dir = work_dir / self.config.env_name
        os.makedirs(log_dir, exist_ok=True)

        metrics = {}
        for name, values in self.last_test_metrics.items():
            metrics[name] = np.mean(values).item()
            if np.isnan(metrics[name]):
                metrics[name] = 0.0

        metrics_str = json.dumps(metrics, indent=4)
        print(metrics_str)
        with open(log_dir / "metrics.json", "w") as metrics_file:
            metrics_file.write(metrics_str)

    def save_state(self):
        work_dir = Path(self.config.work_dir)
        log_dir = work_dir / self.config.env_name
        os.makedirs(log_dir, exist_ok=True)

        # Retrieve last best checkpoint and metrics from visdom
        try:
            max_metric = max(self.best_metric_history)

            if max_metric == self.best_metric_history[-1]:
                self.save_checkpoint(log_dir / "checkpoint_best.pt")

                with open(log_dir / "best_model.txt", "w") as best_file:
                    best_file.write("index: %d\n" % (len(self.best_metric_history) - 1))
                    best_file.write("%s: %f\n" % (self.best_metric_name, max_metric))
                print(
                    "Saved best model at index %d with best metric %s=%f"
                    % (
                        len(self.best_metric_history) - 1,
                        self.best_metric_name,
                        max_metric,
                    )
                )

        except Exception as e:
            print("Could not save best model: " + str(e))

        self.save_checkpoint(log_dir / "checkpoint.pt")
        self.logger.save_visdom(log_dir)

    def save_checkpoint(self, ckpt_path: str):
        state = {
            "model": (
                self.model.module.state_dict()
                if isinstance(self.model, DDP)
                else self.model.state_dict()
            ),
            "optimizer-gpt": self.optim_gpt.state_dict(),
            "optimizer-yolox": self.optim_yolox.state_dict(),
        }

        torch.save(
            state,
            ckpt_path,
        )

    def prepare_validation(self):
        """
        Reload best checkpoint to test it over the entire dataset
        """
        # Reload best checkpoint
        work_dir = Path(self.config.work_dir)
        log_dir = work_dir / self.config.env_name
        ckpt_path = log_dir / "checkpoint_best.pt"

        if ckpt_path.exists():
            # TODO function trainer.load_checkpoint?
            print("Loading best checkpoint for validation:", ckpt_path)
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            # Load the model checkpoint and add the DDP wrapper.
            if isinstance(self.model, DDP):
                model_checkpoint = dict()
                for module_name, params in checkpoint["model"].items():
                    model_checkpoint["module." + module_name] = params
            else:
                model_checkpoint = checkpoint["model"]
            self.model.load_state_dict(model_checkpoint)
        else:
            print(
                "Could not reload best checkpoint for final test, using last checkpoint"
            )

        # Change test data to all dataset
        self.test_env_ids = list(range(len(self.test_dataset)))

    # === Detection

    def yolox_model(self):
        # XXX: this may not work with DDP, yolox would need to be separated
        # from decision
        return (
            self.model.module.yolox if isinstance(self.model, DDP) else self.model.yolox
        )

    def init_detection(self):
        self.detection_augment = nn.Sequential(
            augmentation.RandomPlanckianJitter(mode="CIED"),
            augmentation.RandomGrayscale(p=0.2),
            augmentation.RandomGaussianBlur((3, 3), sigma=(0.1, 2.0)),
            augmentation.RandomPlasmaShadow(
                shade_intensity=(-0.2, 0.0), shade_quantity=(0.0, 0.4), p=0.5
            ),
            augmentation.RandomGaussianNoise(mean=0.0, std=0.05, p=0.5),
            augmentation.RandomMotionBlur(3, (-180.0, 180.0), 0.0, p=0.3),
        )

    def compute_detection_metrics(
        outputs: List[Optional[torch.Tensor]],
        targets: List[torch.Tensor],
    ) -> dict:
        """Compute detection metrics.

        ---
        Args:
            outputs: List of predicted bboxes per image. Each tensor is all the
                predicted bboxes for one image, None if no prediction.
                Shape of [n_bboxes, 4 + 1 + ?].
                The last dimension is organized as follows:
                    - [xmin, ymin, xmax, ymax]: The coordinates of the bbox.
                    - score
            targets: The true bboxes to be detected.
                Shape of [n_bboxes, 4 + 1]
        """
        n_bboxes = sum([len(t) for t in targets])

        if n_bboxes == 0:
            # No bbox in the batch => fix the map to 0 (torchmetrics would compute -1).
            metrics["map"] = torch.FloatTensor([0.0]).to(targets.device)
            return metrics

        preds = []
        tgts = []

        for i, image_outputs in enumerate(outputs):
            image_targets = targets[i]

            if image_outputs is None:
                image_outputs = torch.zeros((1, 7), device=image_targets.device)

            prediction = {
                "boxes": image_outputs[:, :4],
                "scores": image_outputs[:, 4],
                "labels": torch.zeros(
                    len(image_outputs),
                    device=image_outputs.device,
                    dtype=torch.long,
                ),
            }

            # Keep only the true bboxes.
            # TODO take class into account
            target = {
                "boxes": image_targets[:, 1:5],
                "labels": torch.zeros(
                    len(image_targets),
                    device=image_targets.device,
                    dtype=torch.long,
                ),
            }

            preds.append(prediction)
            tgts.append(target)

        metrics = dict()
        map = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
        metrics["map"] = map(preds, tgts)["map_50"]
        return metrics

    def patch_bboxes2full_image(
        outputs: List[List[Optional[torch.Tensor]]],
        offsets: torch.Tensor,
        masks: torch.Tensor = None,
    ) -> List[Optional[torch.Tensor]]:
        """
        Args:
            outputs: List of predicted bboxes.
                The first list is the batch.
                The second list is the episode length.
                The tensors are predictions for each patches of the episodes of the batch.
        """
        new_outputs = []
        for i, image_outputs in enumerate(outputs):
            new_img_outputs = []

            for j, patch_outputs in enumerate(image_outputs):
                if masks is not None and not masks[i, j]:
                    continue
                if patch_outputs is not None:
                    patch_outputs = patch_outputs.clone()
                    patch_outputs[:, :2] += offsets[i, j]
                    patch_outputs[:, 2:4] += offsets[i, j]
                    new_img_outputs.append(patch_outputs)

            if len(new_img_outputs) > 0:
                new_outputs.append(torch.cat(new_img_outputs))
            else:
                new_outputs.append(None)

        return new_outputs
