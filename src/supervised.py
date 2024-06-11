import os
import time
from collections import defaultdict
from copy import deepcopy
from typing import List, Optional, Dict, Tuple
from pathlib import Path
from itertools import chain
import json
import traceback

import einops
import torch
import torch.nn as nn
import numpy as np
from torchvision.ops import nms

from einops import rearrange
from kornia import augmentation
from torch.distributed import destroy_process_group, init_process_group
from torch.distributions import Categorical
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_convert
from tqdm import tqdm

from .dataset import NeedleDataset
from .env.simple_env import NeedleSimpleEnv, Action, Position, BBox
from .env import get_actions_info
from .models.gpt import GPT
from .models.yolox import NeedleYOLOX
from .utils import CfgNode as CN, plot_model_prediction
from .trainer import Trainer
from .logger import Logger


class SupervisedTrainer(Trainer):
    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = "auto"
        # dataloder parameters
        C.num_workers = 1
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.epoch_length = 100
        C.learning_rate = 1e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1  # only applied on matmul weights
        C.grad_norm_clip = 1.0
        return C

    def __init__(
        self,
        config: CN,
        model: GPT,
        logger: Logger,
        train_dataset: NeedleDataset,
        test_dataset: NeedleDataset,
        rank: int,
    ):
        super(SupervisedTrainer, self).__init__(
            config, model, logger, train_dataset, test_dataset, rank
        )

        # We do not reduce directly so that we can ignore padding labels dynamically.
        actions_info = get_actions_info(config)
        weight = torch.ones(actions_info[0].nclasses)
        if config.stop_enabled:
            weight[Action.STOP.value] = config.stop_weight
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=weight, reduction="none")
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

        # TODO dont need the history, just save highest value
        self.best_metric_name = "map"

        print("Trainer initialized")

    def create_env(self, sample: Dict) -> Tuple[NeedleSimpleEnv, int]:
        image = sample["image"]
        bboxes = sample["bboxes"]
        env = NeedleSimpleEnv(
            image,
            self.config.patch_size,
            bboxes,
        )
        return env, sample["class_id"]

    def generate_trajectories(
        self, batch: Dict, position: Optional[Position] = None
    ) -> Tuple[dict, List[BBox]]:
        """Build a sample based on a trajectory

        Args
            batch: a dict containing the image & the associated bboxes
            position: optional starting position

        Returns
            patches: Tensor of patches of shape [n_patches, n_channels, height, width].
            current_actions: Tensor of action that has been taken for the current patch.
                Shape of [n_patches,].
            next_actions: Tensor of action to take for the current patch.
                Shape of [n_patches,].
            labels: Tensor of label ids of shape [n_patches,].
            positions: List of consecutive positions of the patches.
        """
        samples = []
        batch_size = len(batch["image"])  # batch["image"].shape[0]

        for i in range(batch_size):
            img_sample = {
                "image": batch["image"][i],
                "bboxes": batch["bboxes"][i],
                "class_id": batch["class_id"][i],
            }
            env, class_id = self.create_env(img_sample)

            sample = env.generate_sample(
                self.config.max_seq_len,
                min_keypoints=self.config.min_keypoints,
                max_keypoints=self.config.max_keypoints,
                position=position,
                binomial_keypoints=self.config.binomial_keypoints,
            )
            sample["class_id"] = torch.tensor(
                class_id, dtype=torch.long, device=sample["patches"].device
            )
            samples.append(sample)

        return NeedleSimpleEnv.collate_fn(samples)

    def compute_metrics(
        self,
        action_logits: torch.Tensor,
        actions: torch.Tensor,
        masks: torch.Tensor,
        yolo_loss: Optional[dict] = None,
    ) -> torch.Tensor:
        """Fill the metrics dictionary.

        ---
        Args
            action_logits: Output tensor from the model predicting actions.
                Shape of [batch_size, seq_len, vocab_size].
            actions: Target tensor containing the action ids.
                Shape of [batch_size, seq_len].
            masks: Masks of padding (1 if not masked, 0 if padding).
                Shape of [batch_size, seq_len].
            yolo_loss: Already computed loss from the yolo model.
                None if not used.

        ---
        Returns:
            loss: The computed loss.
        """
        metrics = defaultdict(list)
        metrics["loss"] = torch.zeros(1, device=action_logits.device)
        self.bce_loss.to(action_logits.device)
        self.cross_entropy_loss.to(action_logits.device)

        action_logits = rearrange(action_logits, "b s v -> (b s) v")
        actions = actions.flatten()
        predicted_actions = action_logits.argmax(dim=1)
        no_padding = (masks == 1).flatten()

        action_loss = self.cross_entropy_loss(action_logits, actions)
        metrics["action_loss"] = action_loss[no_padding].mean()
        metrics["action_accuracy"] = (
            (actions[no_padding] == predicted_actions[no_padding]).float().mean()
        )
        metrics["loss"] = metrics["loss"] + metrics["action_loss"]

        if yolo_loss is not None:
            for metric_name, metric_value in yolo_loss.items():
                if not isinstance(metric_value, torch.Tensor):
                    metric_value = torch.tensor(metric_value)
                metrics[f"yolo_{metric_name}"] = metric_value
            metrics["yolo_loss"] = metrics["yolo_total_loss"]
            metrics["loss"] += metrics["yolo_loss"]

        # Remove some possible NaNs.
        for metric_name in [
            "action_accuracy",
            "bbox_iou",
        ]:
            if metric_name in metrics:
                metric_value = metrics[metric_name]
                if torch.isnan(metric_value):
                    metrics[metric_name] = torch.FloatTensor([0.0]).to(self.device)

        metrics["episode_length"] = masks.sum(dim=1).float().mean()
        return metrics

    def compute_yolo_metrics(
        self,
        outputs: List[List[Optional[torch.Tensor]]],
        targets: torch.Tensor,
    ) -> dict:
        """Compute specific yolo metrics.

        ---
        Args:
            outputs: List of predicted bboxes.
                The first list is the batch.
                The second list is the episode length.
                The tensors are predictions for each patches of the episodes of the batch.
            targets: The true bboxes to be detected.
                Shape of [batch_size, n_tokens, n_bboxes, class_id + 4 + 1].
                The last dimension is organized as follows:
                    - [class_id,]: The bbox's class.
                    - [cx, cy, w, h]: The coordinates of the bbox.
                    - [1,]: Whether the bbox is a true bbox or not (objectiveness).
        """
        metrics = dict()
        map = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
        n_bboxes = targets[..., -1].sum().cpu().item()

        if n_bboxes == 0:
            # No bbox in the batch => fix the map to 0 (torchmetrics would compute -1).
            metrics["map"] = torch.FloatTensor([0.0]).to(targets.device)
            return metrics

        preds = []
        for image_outputs in outputs:
            for patch_outputs in image_outputs:
                if patch_outputs is None:
                    # The model predicted nothing.
                    patch_outputs = torch.zeros((0, 5), device=targets.device)
                prediction = {
                    "boxes": patch_outputs[:, :4],
                    "scores": patch_outputs[:, -1],
                    "labels": torch.zeros(
                        len(patch_outputs),
                        device=patch_outputs.device,
                        dtype=torch.long,
                    ),
                }
                preds.append(prediction)

        tgts = []
        for image_targets in targets:
            for patch_targets in image_targets:
                if patch_targets[:, -1].sum() == 0.0:
                    # print("map: no bbox")
                    tgts.append(
                        {
                            "boxes": torch.zeros((0, 4), device=targets.device),
                            "labels": torch.zeros(
                                (0,), device=targets.device, dtype=torch.long
                            ),
                        }
                    )
                    continue

                # Keep only the true bboxes.
                patch_targets = patch_targets[patch_targets[:, -1] == 1]
                target = {
                    "boxes": box_convert(patch_targets[:, 1:5], "cxcywh", "xyxy"),
                    "labels": torch.zeros(
                        len(patch_targets),
                        device=patch_targets.device,
                        dtype=torch.long,
                    ),
                }
                tgts.append(target)

        metrics["map"] = map(preds, tgts)["map_50"]
        self.yolo_metrics = metrics

        return metrics

    @torch.no_grad()
    def test_model_on_env(
        self,
        env: NeedleSimpleEnv,
        max_ep_len: int,
        class_id: int,
        sample_actions: bool = False,
        position: Optional[Position] = None,
    ) -> tuple:
        """
        Test model on a given image

        Parameters:
            sample_actions whether actions should be selected using max or using weighted sampling
        """
        self.model.eval()
        cpy_env = deepcopy(env)
        labels = {"true": [], "logits": []}
        actions = {"true": [], "logits": []}

        patch, infos = env.reset(position)
        sample = env.init_sample(max_ep_len, self.device)
        perfect_sample = cpy_env.generate_sample(
            max_ep_len=50,
            min_keypoints=0,
            max_keypoints=0,
            position=deepcopy(env.position),
            visited_bbox_patches=deepcopy(env.visited_bbox_patches),
            device="cpu",
        )
        infos["best_action"] = Action(perfect_sample["next_actions"][0].item())
        env.add_to_sample(
            sample,
            action_taken=Action.LEFT,
            patch=patch.to(self.device),
            infos=infos,
            index=0,
        )
        sample["class_id"] = torch.tensor(
            class_id, dtype=torch.long, device=self.device
        )

        if sample_actions:
            select_action = lambda logits: Categorical(logits=logits).sample()
        else:
            select_action = lambda logits: logits.argmax()

        for index in range(1, max_ep_len):
            action_logits = self.model(
                sample["patches"].unsqueeze(0),
                sample["current_actions"].unsqueeze(0),
                classes=sample["class_id"].unsqueeze(0),
                positions=sample["positions"].unsqueeze(0),
            )
            action_logits = action_logits[0, index - 1]

            action = select_action(action_logits)
            action = Action(action.cpu().item())

            patch, infos = env.step(action)

            perfect_sample = cpy_env.generate_sample(
                max_ep_len=50,
                min_keypoints=0,
                max_keypoints=0,
                position=deepcopy(env.position),
                visited_bbox_patches=deepcopy(env.visited_bbox_patches),
                device="cpu",
            )
            infos["best_action"] = Action(perfect_sample["next_actions"][0].item())

            actions["true"].append(infos["best_action"].value)
            actions["logits"].append(action_logits.cpu().tolist())
            labels["true"].append(infos["inside_bbox"])

            env.add_to_sample(
                sample,
                action,
                patch.to(self.device),
                infos,
                index,
            )

            if action == Action.STOP:
                break

        # Compute some metrics to plot.
        # Loss metrics.
        if isinstance(self.model, DDP):
            bbox_outs, _, yolo_loss = self.model.module.yolox(
                sample["patches"].unsqueeze(0),
                sample["local_bboxes"].unsqueeze(0),
            )
        else:
            bbox_outs, _, yolo_loss = self.model.yolox(
                sample["patches"].unsqueeze(0),
                sample["local_bboxes"].unsqueeze(0),
            )
        yolo_loss = {
            name: value.cpu() if isinstance(value, torch.Tensor) else value
            for name, value in yolo_loss.items()
        }
        metrics = self.compute_metrics(
            torch.FloatTensor(actions["logits"]).unsqueeze(0),
            torch.LongTensor(actions["true"]).unsqueeze(0),
            sample["masks"][sample["masks"] == 1]
            .unsqueeze(0)
            .cpu()[:, :-1],  # Remove last mask
            yolo_loss,
        )

        for name, value in metrics.items():
            metrics[name] = value.item()

        # Removes NaNs and converts to float.
        for name, value in metrics.items():
            if type(value) is torch.Tensor:
                if value.isnan():
                    metrics[name] = 0.0
                else:
                    metrics[name] = value.cpu().item()

        metrics["stopped_inside_bbox"] = labels["true"][-1]
        metrics["prop_patches_found"] = (
            infos["number_patches_found"] / len(env.bbox_patches)
            if len(env.bbox_patches) > 0
            else 0.0
        )

        return sample, metrics, bbox_outs[0]

    @torch.no_grad()
    def eval_supervised(self, dataset, env_ids):
        print("Evaluating on supervised trajectories...")
        all_metrics = defaultdict(list)
        subset = torch.utils.data.Subset(
            dataset,
            indices=env_ids,
        )
        loader = DataLoader(
            subset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            sampler=torch.utils.data.SequentialSampler(
                subset,
            ),
            drop_last=False,
            collate_fn=dataset.list_collate_fn,
        )
        device = self.model.device

        for batch in tqdm(loader):
            batch = self.generate_trajectories(batch)
            patches = batch["patches"].to(device)
            current_actions = batch["current_actions"].to(device)
            next_actions = batch["next_actions"].to(device)
            positions = batch["positions"].to(device)
            masks = batch["masks"].to(device)
            classes = batch["class_id"].to(device)
            patches_yolox = batch["patches_yolox"].to(device)
            bboxes_yolox = batch["bboxes_yolox"].to(device)

            action_logits = self.model(
                patches,
                current_actions,
                classes=classes,
                positions=positions,
            )

            if self.config.loss_mode == "on-self-trajectory":
                reference_actions = torch.zeros_like(current_actions).to(device)
                reference_actions[:, :-1] = current_actions[:, 1:]
                for batch_id in range(len(masks)):
                    mask = masks[batch_id]
                    reference_actions[batch_id, mask.sum().long() - 1] = next_actions[
                        batch_id, mask.sum().long() - 1
                    ]
            else:
                reference_actions = next_actions

            if isinstance(self.model, DDP):
                bbox_outs, _, yolo_loss = self.model.module.yolox(
                    patches_yolox.unsqueeze(0), bboxes_yolox.unsqueeze(0)
                )
            else:
                bbox_outs, _, yolo_loss = self.model.yolox(
                    patches_yolox.unsqueeze(0), bboxes_yolox.unsqueeze(0)
                )

            metrics = self.compute_metrics(
                action_logits,
                reference_actions,
                masks,
                yolo_loss,
            )

            yolo_metrics = self.compute_yolo_metrics(
                bbox_outs, bboxes_yolox.unsqueeze(0)
            )

            for metric_name, metric_values in chain(
                metrics.items(), yolo_metrics.items()
            ):
                all_metrics[metric_name].append(metric_values.cpu().item())

        return all_metrics

    def eval_missing_patches(
        self,
        dataset: NeedleDataset,
        env_id: int,
        samples: list,
        bboxes: list,
    ):
        """
        Compute the MAP by accounting for the patches missed by the decision model.

        We count the missing patches as false negatives, as if the model predicted nothing.
        """
        env, _ = self.create_env(dataset[env_id])
        env.reset()
        visited_patches = {
            Position(y=pos[0].cpu().item(), x=pos[1].cpu().item())
            for sample in samples
            for pos in sample["positions"][sample["masks"] == 1]
        }

        targets = {}
        for pos in visited_patches:
            env.position = pos
            targets[pos] = env.local_bboxes()

        predicted = {}
        for sample, bboxes_ in zip(samples, bboxes):
            for pos, bbox in zip(
                sample["positions"][sample["masks"] == 1],
                bboxes_,
            ):
                if bbox is None:
                    continue

                pos = Position(y=pos[0].cpu().item(), x=pos[1].cpu().item())
                if pos not in predicted:
                    predicted[pos] = []
                predicted[pos].append(bbox)

        for pos, bboxes in predicted.items():
            predicted[pos] = torch.concat(bboxes, dim=0)

        list_targets, list_predicted = [], []
        for pos in visited_patches:
            list_targets.append(targets[pos])
            if pos not in predicted:
                list_predicted.append(None)
            else:
                bbox = predicted[pos]
                if isinstance(bbox, torch.Tensor):
                    bbox = bbox.cpu()
                list_predicted.append(bbox)

        targets = torch.stack(list_targets)

        # Add batch dimension.
        targets = targets.unsqueeze(0)
        list_predicted = [list_predicted]

        # Apply NMS to ignore duplicates (avoid false positives).
        for batch_id in range(len(list_predicted)):
            for patch_id in range(len(list_predicted[batch_id])):
                bboxes = list_predicted[batch_id][patch_id]
                if bboxes is None:
                    continue

                bboxes_ids = nms(bboxes[:, :4], bboxes[:, -1], iou_threshold=0.5)
                list_predicted[batch_id][patch_id] = bboxes[bboxes_ids]

        # Add the missing patches as false negatives.
        for pos in env.bbox_patches - visited_patches:
            # We prepend None to the list of predictions because there may be a bug in
            # torchmetrics that discards empty images that are at the end of the batch.
            # This should not be our case though.
            # See: https://github.com/Lightning-AI/torchmetrics/issues/1774
            list_predicted[0].insert(0, None)
            env.position = pos
            bboxes = env.local_bboxes().unsqueeze(0).unsqueeze(0)
            targets = torch.cat([bboxes, targets], dim=1)

        metrics = self.compute_yolo_metrics(list_predicted, targets)
        return metrics

    def metrics_from_multiple_samples(
        self, env: NeedleSimpleEnv, samples: list, bboxes: list
    ) -> dict:
        current_env_pos = env.position
        visited_patches = {
            Position(y=pos[0].cpu().item(), x=pos[1].cpu().item())
            for sample in samples
            for pos in sample["positions"][sample["masks"] == 1]
        }

        targets = {}
        for pos in visited_patches:
            env.position = pos
            targets[pos] = env.local_bboxes()

        predicted = {}
        for sample, bboxes_ in zip(samples, bboxes):
            for pos, bbox in zip(
                sample["positions"][sample["masks"] == 1],
                bboxes_,
            ):
                if bbox is None:
                    continue

                pos = Position(y=pos[0].cpu().item(), x=pos[1].cpu().item())
                if pos not in predicted:
                    predicted[pos] = []
                predicted[pos].append(bbox)

        for pos, bboxes in predicted.items():
            predicted[pos] = torch.concat(bboxes, dim=0)

        list_targets, list_predicted = [], []
        for pos in visited_patches:
            list_targets.append(targets[pos])
            if pos not in predicted:
                list_predicted.append(None)
            else:
                bbox = predicted[pos]
                if isinstance(bbox, torch.Tensor):
                    bbox = bbox.cpu()
                list_predicted.append(bbox)

        targets = torch.stack(list_targets)

        # Add batch dimension.
        targets = targets.unsqueeze(0)
        list_predicted = [list_predicted]

        for batch_id in range(len(list_predicted)):
            for patch_id in range(len(list_predicted[batch_id])):
                bboxes = list_predicted[batch_id][patch_id]
                if bboxes is None:
                    continue

                bboxes_ids = nms(bboxes[:, :4], bboxes[:, -1], iou_threshold=0.5)
                list_predicted[batch_id][patch_id] = bboxes[bboxes_ids]

        metrics = self.compute_yolo_metrics(list_predicted, targets)
        metrics["prop_patches_found"] = torch.tensor(
            len(visited_patches & env.bbox_patches) / len(env.bbox_patches)
            if len(env.bbox_patches) > 0
            else 0.0
        )

        env.position = current_env_pos

        return metrics

    def eval_envs(
        self,
        dataset: NeedleDataset,
        env_ids: list,
        eval_mode: str = "multistart",
        sample_actions: bool = False,
    ):
        """
        Compute and aggregate all metrics in a "metrics" object. Then,
        use logger to dispatch it to visdom & co
        """
        print("Evaluate trajectories with mode: %s" % eval_mode)
        # ids of images that will be visualized
        visual_ids = self.rng.choice(
            np.arange(len(env_ids)), size=min(6, len(env_ids)), replace=False
        )
        prediction_images = {
            "model_images": [],
            "sample_images": [],
        }
        all_metrics = defaultdict(list)

        for loop_id, env_id in enumerate(tqdm(env_ids)):
            # eval function
            seed = loop_id
            env, class_id = self.create_env(
                dataset.__getitem__(env_id, np.random.default_rng(seed))
            )
            env.rng = np.random.default_rng(seed)

            all_samples, all_bboxes = [], []
            if eval_mode == "corners":
                positions = env.corners
            elif eval_mode == "multistart":
                max_starts = 2
                positions = [None] * max_starts
            elif eval_mode == "rollouts":
                max_rollouts = 2
                positions = [env.position] * max_rollouts

            for i, position in enumerate(positions):
                env.reset()
                if position is None:
                    positions[i] = env.position
                    position = env.position

                sample, metrics, bboxes = self.test_model_on_env(
                    env,
                    self.config.test_max_seq_len,
                    class_id,
                    sample_actions,
                    position,
                )
                all_samples.append(sample)
                all_bboxes.append(bboxes)

                for name, value in metrics.items():
                    all_metrics[name].append(value)

            for n_starts in range(1, len(positions) + 1):
                metrics = self.metrics_from_multiple_samples(
                    env, all_samples[:n_starts], all_bboxes[:n_starts]
                )
                suffix = f"_{eval_mode}_{n_starts}" if n_starts != 1 else ""
                for name, value in metrics.items():
                    all_metrics[f"{name}_traj{suffix}"].append(value.cpu().item())

                # take missing patches into account
                metrics_missing_patches = self.eval_missing_patches(
                    dataset, env_id, all_samples[:n_starts], all_bboxes[:n_starts]
                )
                for name, value in metrics_missing_patches.items():
                    all_metrics[f"{name}{suffix}"].append(value.cpu().item())

            if loop_id in visual_ids:
                sample = all_samples[0]
                bboxes = all_bboxes[0]
                position = positions[0]
                env, _ = self.create_env(
                    dataset.__getitem__(env_id, np.random.default_rng(loop_id))
                )
                true_bboxes = NeedleYOLOX.parse_bbox_targets(
                    sample["local_bboxes"].cpu(),
                    sample["positions"].cpu(),
                    env.patch_size,
                )
                predicted_bboxes = NeedleYOLOX.parse_bbox_predictions(
                    bboxes, sample["positions"], env.patch_size
                )
                image = plot_model_prediction(
                    env.image,
                    sample["patches"][sample["masks"] == 1].cpu(),
                    sample["positions"][sample["masks"] == 1].cpu(),
                    true_bboxes=true_bboxes,
                    predicted_bboxes=predicted_bboxes,
                )
                prediction_images["model_images"].append(image)

                sample = env.generate_sample(
                    max_ep_len=env.patch_width
                    * env.patch_height,  # Highest possible value to make sure we don't truncate.
                    min_keypoints=dataset.min_keypoints,
                    max_keypoints=dataset.max_keypoints,
                    position=position,
                    binomial_keypoints=dataset.binomial_keypoints,
                )
                image = plot_model_prediction(
                    env.image,
                    sample["patches"][sample["masks"] == 1],
                    sample["positions"][sample["masks"] == 1],
                    true_bboxes=env.raw_bboxes,
                )
                prediction_images["sample_images"].append(image)

        return all_metrics, prediction_images

    def test(self, sample_actions: bool = False):
        """
        Test current model over datasets.
        Log to visdom
        """
        self.model.eval()

        datasets = [self.test_dataset]
        # ids of test images in the dataset
        datasets_envs_ids = [self.test_env_ids]
        modes = ["test"]

        if self.config.eval_training_set:
            datasets.append(self.train_dataset)
            datasets_envs_ids.append(self.train_env_ids)
            modes.append("train")

        for dataset, envs_ids, mode in zip(datasets, datasets_envs_ids, modes):
            # TODO disable data aug should be cleaner than that (more "safe proof")
            translations, rotations = dataset.translations, dataset.rotations
            dataset.translations, dataset.rotations = False, False

            metrics, images = self.eval_envs(
                dataset, envs_ids, sample_actions=sample_actions
            )

            if self.config.failure_select_rate > 0:
                worst_img_count = int(self.config.failure_select_rate * len(dataset))
                metrics_array = np.array(metrics[self.best_metric_name])
                worst_ids = np.argsort(metrics_array)[:worst_img_count]

                try:
                    # XXX: in a normal training, len(worst id) should not be == 0. Investigate
                    worst_ids = envs_ids[worst_ids]
                    print("Evaluate worst images for plotting")
                    worst_metrics, worst_images = self.eval_envs(dataset, worst_ids)

                    images["worst_images"] = worst_images["model_images"]
                except:
                    print("Could not compute worst images")
                    traceback.print_exc()

            # metrics on supervised trajectories
            supervised_metrics = self.eval_supervised(dataset, envs_ids)
            for name, values in supervised_metrics.items():
                metrics["supervised_" + name] = values

            # TODO this would be overriden by train metrics if train is added
            # Should be out of the "if"
            self.last_test_metrics = metrics
            self.best_metric_history.append(np.mean(metrics[self.best_metric_name]))

            self.logger.log_to_visdom(metrics, images, mode)
            dataset.translations, dataset.rotations = translations, rotations

        self.save_state()
        self.save_metrics()

    def run(self, rank: int, world_size: int, port: int):
        self.ddp_setup(rank, world_size, port)
        self.model = DDP(self.model, device_ids=[self.device])
        model, config = self.model, self.config

        augment = nn.Sequential(
            augmentation.RandomPlanckianJitter(mode="CIED"),
            augmentation.RandomGrayscale(p=0.2),
            augmentation.RandomGaussianBlur((3, 3), sigma=(0.1, 2.0)),
            augmentation.RandomPlasmaShadow(
                shade_intensity=(-0.2, 0.0), shade_quantity=(0.0, 0.4), p=0.5
            ),
            augmentation.RandomGaussianNoise(mean=0.0, std=0.05, p=0.5),
            augmentation.RandomMotionBlur(3, (-180.0, 180.0), 0.0, p=0.3),
        )

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=DistributedSampler(self.train_dataset),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            collate_fn=self.train_dataset.list_collate_fn,
        )

        data_iter = iter(train_loader)
        # pass only necessary stuff, like models, etc?
        self.logger.log_start(model)

        # TODO iter_num = each iter_size iteration
        for self.iter_num in tqdm(range(1, config.max_iters), disable=rank != 0):
            model.train()

            try:
                batch = next(data_iter)
            except StopIteration:
                train_loader.sampler.set_epoch(self.iter_num)
                data_iter = iter(train_loader)
                batch = next(data_iter)

            batch = self.generate_trajectories(batch)

            # print('batch=',batch)
            patches = batch["patches"].to(self.device)
            current_actions = batch["current_actions"].to(self.device)
            next_actions = batch["next_actions"].to(self.device)
            positions = batch["positions"].to(self.device)
            masks = batch["masks"].to(self.device)
            classes = batch["class_id"].to(self.device)

            # Do batched data augmentation on GPU.
            with torch.no_grad():
                batch_size = patches.shape[0]
                patches = einops.rearrange(patches, "b t c h w -> (b t) c h w")
                patches = augment(patches)
                patches = einops.rearrange(
                    patches, "(b t) c h w -> b t c h w", b=batch_size
                )

            action_logits = model(
                patches,
                current_actions,
                classes=classes,
                positions=positions,
            )

            if self.config.loss_mode == "on-self-trajectory":
                reference_actions = torch.zeros_like(current_actions).to(self.device)
                reference_actions[:, :-1] = current_actions[:, 1:]
                for batch_id in range(len(masks)):
                    mask = masks[batch_id]
                    reference_actions[batch_id, mask.sum().long() - 1] = next_actions[
                        batch_id, mask.sum().long() - 1
                    ]
            else:
                reference_actions = next_actions

            # Train YOLOX.
            patches_yolox = batch["patches_yolox"].to(self.device)
            bboxes_yolox = batch["bboxes_yolox"].to(self.device)
            with torch.no_grad():
                patches_yolox = augment(patches_yolox)

            if isinstance(model, DDP):
                _, _, yolo_loss = model.module.yolox(
                    patches_yolox.unsqueeze(0), bboxes_yolox.unsqueeze(0)
                )
            else:
                _, _, yolo_loss = model.yolox(
                    patches_yolox.unsqueeze(0), bboxes_yolox.unsqueeze(0)
                )

            self.last_train_metrics = self.compute_metrics(
                action_logits,
                reference_actions,
                masks,
                yolo_loss,
            )
            loss = self.last_train_metrics["loss"]

            # Backprop and update the parameters.
            loss.backward()
            if self.iter_num % self.config.gradient_accumulation == 0:
                self.optim_gpt.step(), self.optim_yolox.step()
                self.optim_gpt.zero_grad(), self.optim_yolox.zero_grad()

            if self.iter_num % config.test_every == 0 and rank == 0:
                self.test()

        if rank == 0:
            self.prepare_validation()
            self.test()  # final = True

        destroy_process_group()
