from collections import defaultdict
from pathlib import Path
from functools import partial
from typing import Any, Iterator, Tuple, List, Dict
import copy

import numpy as np
import einops
import torch
import torch.nn.functional as F
from kornia.geometry.boxes import Boxes
from torch.distributions import Categorical
from torch.nn.utils import clip_grad
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from .env.general_env import NeedleGeneralEnv
from .trainer import Trainer
from .logger import Logger
from .utils import (
    CfgNode as CN,
    bboxes_to_tensor,
    parse_bbox_predictions,
    parse_bbox_targets,
    plot_model_prediction,
    save_batch,
    merge_boxes_batched,
)
from .models.gpt import GPT
from .dataset import NeedleDataset


class ReinforceTrainer(Trainer):
    """The REINFORCE algorithm.
    It used to train the model, on a batched environment.
    """

    def __init__(
        self,
        config: CN,
        model: GPT,
        logger: Logger,
        train_dataset: NeedleDataset,
        test_dataset: NeedleDataset,
        rank: int = 0,
    ):
        super(ReinforceTrainer, self).__init__(
            config, model, logger, train_dataset, test_dataset, rank
        )

        self.patch_size = model.patch_size
        self.max_ep_len = self.config.max_seq_len
        self.entropy_weight = self.config.entropy_weight
        self.n_glimps_levels = 1  # n_glimps_levels
        self.stop_enabled = self.config.stop_enabled
        self.max_iters = config.max_iters
        self.log_every = config.test_every
        self.device = self.config.gpu_ids[rank]

        self.checkpoint_dir = Path(config.work_dir) / config.env_name
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.best_metric_name = "prop_patches_found"
        # last return values to compute mean and std
        self.last_return_values = []
        self.last_return_mean = 0
        self.last_return_std = 1

    def sample_from_logits(
        self, logits: torch.Tensor, take_best_action: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        take_best_action: always take the best action instead of sampling
        """

        # sample from the last token
        last_action_logits = logits[:, -1, :]
        categorical = Categorical(logits=last_action_logits)
        if take_best_action:
            actions = last_action_logits.argmax(dim=1)
        else:
            actions = categorical.sample()
        logprobs = categorical.log_prob(actions)
        entropies = categorical.entropy()

        return actions, logprobs, entropies

    @torch.no_grad()
    def _compute_last_returns_mean_std(self):
        """
        Compute returns mean and std over last batch and reset last returns values
        """
        all_return_values = torch.cat(self.last_return_values)
        if len(all_return_values) == 0:
            mean, std = 0, 1
        elif len(all_return_values) == 1:
            mean, std = all_return_values[0], 1
        else:
            mean, std = all_return_values.mean(), all_return_values.std()

        self.last_return_mean, self.last_return_std = mean, std
        self.last_return_values = []

    def rollout(
        self,
        env: NeedleGeneralEnv,
        do_detection: bool = False,
        sample_actions: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Do a rollout on the given environment.
        Returns the rewards, returns and logprobs of the rollout.
        """
        # all final shapes are batch_size * max(ep_lengths)
        rewards = []
        logprobs = []
        entropies = []
        masks = []

        actions = torch.zeros(
            (env.batch_size, 1),  # (env.batch_size, 1, 2), # two actions: x & y
            dtype=torch.long,
            device=self.device,
        )
        # TODO classes
        classes = torch.zeros((env.batch_size,), dtype=torch.int64, device=self.device)
        patches, infos = env.reset()
        positions = infos["positions"].unsqueeze(1)
        bboxes = [[] for i in range(env.batch_size)]
        masks.append(
            torch.tensor(
                [True for i in range(env.batch_size)],
                device=self.device,
                dtype=torch.bool,
            )
        )

        if do_detection:
            yolox = self.yolox_model()
            # Add prediction of the first batch
            bbox_out, _, yolo_loss = yolox(patches[0], None)
            for i in range(env.batch_size):
                bboxes[i].append(bbox_out[i])

        embeddings = None

        for step_id in range(env.max_ep_len):
            action_logits, embeddings = self.model(
                patches, actions, classes, positions, embeddings
            )
            new_actions, logprobs_, entropies_ = self.sample_from_logits(
                action_logits, take_best_action=not sample_actions
            )

            new_patches, step_rewards, terminated, truncated, infos = env.step(
                new_actions
            )

            if do_detection:
                # XXX: the patch includes "glimps_level" at dim 1 which
                # is not supported by yolox (and useless?)
                bbox_out, _, yolo_loss = yolox(new_patches[:, 0], None)
                for i in range(env.batch_size):
                    bboxes[i].append(bbox_out[i])

            rewards.append(step_rewards)
            logprobs.append(logprobs_)  # .sum(dim=1)
            entropies.append(entropies_)  # .sum(dim=1)
            masks.append(~terminated)

            # Append for the next iteration of the model
            actions = torch.concat((actions, new_actions.unsqueeze(1)), dim=1)
            patches = torch.concat((patches, new_patches), dim=1)
            positions = torch.concat(
                (positions, infos["positions"].unsqueeze(1)), dim=1
            )

            if torch.all(terminated | truncated):
                # All environments are done.
                # WARNING: if the rollout is stopped early, the tensors shapes will be less than `max_ep_len`.
                break

        rewards = torch.stack(rewards, dim=1)
        logprobs = torch.stack(logprobs, dim=1)
        entropies = torch.stack(entropies, dim=1)
        masks = torch.stack(masks, dim=1)

        # The last terminated state is not counted in the masks,
        # so we need to shift the masks by 1 to make sure we include id.
        logit_masks = torch.roll(masks[:, 1:], shifts=1, dims=(1,))
        logit_masks[:, 0] = True

        backward_rewards = torch.flip(rewards, dims=(1,))
        # shift to right and drop first element -> drop last element
        backward_masks = torch.flip(logit_masks, dims=(1,))
        backward_cumulated_returns = torch.cumsum(
            backward_rewards * backward_masks, dim=1
        )
        cumulated_returns = torch.flip(backward_cumulated_returns, dims=(1,))

        return {
            "rewards": rewards,
            "returns": cumulated_returns,
            "logprobs": logprobs,
            "entropies": entropies,
            "masks": masks,
            # masks for reward and logits
            "logit_masks": logit_masks,
            "positions": positions,
            "bboxes": bboxes,
            "patches": patches,
        }

    def compute_metrics(
        self, rollout: Dict[str, torch.Tensor], env: NeedleGeneralEnv = None
    ) -> Dict[str, torch.Tensor]:
        """Compute the metrics of the given rollout.

        ---
        Args:
            rollout: A dictionary containing the rewards, returns and logprobs.

        ---
        Returns:
            The metrics, containing the loss.
        """
        metrics = dict()
        returns = rollout["returns"]
        masks = rollout["logit_masks"]

        if self.config.reward_norm:
            self.last_return_values.append(returns[masks].clone().detach())
            mean, std = self.last_return_mean, self.last_return_std
            advantages = (returns - mean) / (std + 1e-8)
        else:
            advantages = returns

        # Train loss
        # TODO compute_loss()
        metrics["action_loss"] = (
            -(rollout["logprobs"] * advantages * masks).sum() / masks.sum()
        )
        # print(advantages, metrics["action-loss"])
        metrics["entropy_loss"] = -(rollout["entropies"] * masks).sum() / masks.sum()
        metrics["loss"] = (
            metrics["action_loss"] + self.entropy_weight * metrics["entropy_loss"]
        )
        metrics["returns"] = (rollout["rewards"] * masks).sum(dim=1).mean()
        metrics["episode_length"] = masks.sum(dim=1).float().mean()

        # Test metrics
        if env:
            metrics["prop_patches_found"] = env.prop_patches_found[0]
            metrics["prop_bbox_found"] = env.prop_bboxes_found[0]

            if self.stop_enabled:
                metrics["stop_used"] = env.terminated[0].to(torch.float32)
                metrics["stop_misused"] = (
                    env.terminated[0] and env.prop_patches_found[0] < 1
                ).to(torch.float32)

        return metrics

    def run(self, rank: int, world_size: int, ddp_port: int):
        """Train the model using REINFORCE.
        The logs are sent to Weights & Biases.

        ---
        Args:
            group: The name of the group of experiments.
            config: A dictionary of hyperparameters.
            mode: The mode of the Weights & Biases run.
        """
        self.init_detection()
        self.ddp_setup(rank, world_size, ddp_port)
        # XXX: DDP doesnt work: "one of the variables needed for gradient computation has been modified by an inplace operation"
        # self.model = DDP(self.model, device_ids=[self.device])
        self.model.device = self.device
        model, config = self.model, self.config

        train_loader = DataLoader(
            self.train_dataset,
            sampler=DistributedSampler(self.train_dataset),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            collate_fn=partial(
                self.train_dataset.padded_collate_fn, patch_size=self.patch_size
            ),
        )

        print(f"Launching REINFORCE on device {self.device}")
        self.logger.log_start(model)

        self.model.to(self.device)
        train_iter = iter(train_loader)

        for step_id in tqdm(range(self.max_iters)):
            self.iter_num = step_id + 1
            self.model.train()

            try:
                batch = next(train_iter)
            except StopIteration:
                train_loader.sampler.set_epoch(self.iter_num)
                train_iter = iter(train_loader)
                batch = next(train_iter)

            images, bboxes = batch["image"].to(self.device), batch["bboxes"].to(
                self.device
            )

            env = NeedleGeneralEnv(
                images,
                bboxes,
                self.patch_size,
                self.max_ep_len,
                self.n_glimps_levels,
                self.stop_enabled,
            )

            rollout = self.rollout(env)
            metrics = self.compute_metrics(rollout)
            loss = metrics["loss"]

            if self.config.detection_enabled:
                patches_yolox, bboxes_yolox = env.get_detection_batch()
                with torch.no_grad():
                    patches_yolox = self.detection_augment(patches_yolox)

                yolox = self.yolox_model()
                _, _, yolo_loss = yolox(patches_yolox, bboxes_yolox)

                total_loss = yolo_loss["total_loss"]
                loss += total_loss

            (loss / self.config.gradient_accumulation).backward()

            if self.iter_num % self.config.gradient_accumulation == 0:
                clip_grad.clip_grad_value_(self.model.parameters(), 1)
                self.optim_gpt.step()
                self.optim_gpt.zero_grad()

                if self.config.detection_enabled:
                    self.optim_yolox.step()
                    self.optim_yolox.zero_grad()

                if self.config.reward_norm:
                    self._compute_last_returns_mean_std()

            if self.iter_num % config.test_every == 0 and rank == 0:
                self.test()

        if rank == 0:
            self.prepare_validation()
            self.test()  # final = True

        destroy_process_group()

    @torch.no_grad()
    def test(self, sample_actions: bool = False) -> Dict[str, torch.Tensor]:
        """Test the model on the test dataset."""
        self.model.eval()
        all_metrics = defaultdict(list)
        plot_images = {
            "model_images": [],
        }

        dataset = self.test_dataset
        env_ids = self.test_env_ids

        visual_ids = self.rng.choice(
            np.arange(len(env_ids)), size=min(6, len(env_ids)), replace=False
        )

        translations, rotations = dataset.translations, dataset.rotations
        dataset.translations, dataset.rotations = False, False

        for loop_id, env_id in enumerate(tqdm(env_ids)):
            batch = dataset.__getitem__(env_id)
            plot_traj = loop_id in visual_ids
            metrics, plot_image = self.eval_on_sample(batch, plot_traj, env_id)

            if plot_traj:
                plot_images["model_images"].append(plot_image)

            for key, value in metrics.items():
                all_metrics[key].append(value.cpu())

        # Select worst images
        if self.config.failure_select_rate > 0:
            worst_img_count = int(self.config.failure_select_rate * len(dataset))
            metrics_array = np.array(all_metrics[self.best_metric_name])
            worst_ids = np.argsort(metrics_array)[:worst_img_count]
            plot_images["worst_images"] = []
            print("Evaluate worst images for plotting")

            for worst_id in tqdm(worst_ids):
                worst_env_id = env_ids[worst_id]
                batch = dataset.__getitem__(env_id)
                metrics, plot_image = self.eval_on_sample(batch, True)
                plot_images["worst_images"].append(plot_image)

        dataset.translations, dataset.rotations = translations, rotations
        self.model.train()

        self.last_test_metrics = all_metrics
        self.best_metric_history.append(np.mean(all_metrics[self.best_metric_name]))
        self.logger.log_to_visdom(all_metrics, plot_images, "test")
        self.save_state()
        self.save_metrics()

    def eval_on_sample(self, batch, plot_traj: bool, *args):
        """
        Evaluate an image
        """
        images = batch["image"].unsqueeze(0).to(self.device)
        bboxes = bboxes_to_tensor(batch["bboxes"]).unsqueeze(0).to(self.device)

        env = NeedleGeneralEnv(
            images,
            bboxes,
            self.patch_size,
            self.max_ep_len,
            self.n_glimps_levels,
            self.stop_enabled,
        )

        rollout = self.rollout(
            env, sample_actions=False, do_detection=self.config.detection_enabled
        )
        metrics = self.compute_metrics(rollout, env)

        positions = rollout["positions"]
        masks = rollout["masks"]
        patches = rollout["patches"]
        full_img_targets = [[]]
        full_img_preds = [[]]

        if self.config.detection_enabled:
            # Convert to list
            full_img_targets = env.get_detection_targets()

            # metrics over trajectories
            traj_bbox_preds = rollout["bboxes"]
            # convert position y, x to x, y
            offsets = positions[:, :, [1, 0]] * self.patch_size
            full_img_preds = Trainer.patch_bboxes2full_image(
                traj_bbox_preds, offsets, masks
            )

            if self.config.merge_bboxes:
                full_img_preds = merge_boxes_batched(full_img_preds, target=False)
                full_img_targets = merge_boxes_batched(full_img_targets, target=True)

            traj_yolo_metrics = Trainer.compute_detection_metrics(
                full_img_preds,
                full_img_targets,
            )

            for metric_name, metric_value in traj_yolo_metrics.items():
                metrics[metric_name] = metric_value
            # TODO add yolo false positive metric

            # metrics over full image
            target_patches, target_bboxes = env.get_detection_batch(sample_neg=0)
            yolox = self.yolox_model()
            pred_bboxes, _, yolo_losses = yolox(target_patches)

            yolo_metrics = Trainer.compute_detection_metrics(pred_bboxes, target_bboxes)

            yolo_metrics.update(yolo_losses)

            for metric_name, metric_value in yolo_metrics.items():
                metrics["yolo_" + metric_name] = metric_value

        plot_image = None

        if plot_traj:
            # Generate image
            plot_image = plot_model_prediction(
                env.images[0, 0].cpu(),  # XXX: we have to select the first glimps level
                patches[0].cpu(),  # first of batch
                positions[0][masks[0] == 1].cpu(),
                true_bboxes=parse_bbox_targets(torch.stack(full_img_targets)),
                predicted_bboxes=parse_bbox_predictions(full_img_preds),
            )

        return metrics, plot_image
