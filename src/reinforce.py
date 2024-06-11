from collections import defaultdict
from pathlib import Path
from functools import partial
from typing import Any, Iterator, Tuple, List, Dict

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
from .utils import CfgNode as CN, bboxes_to_tensor, plot_model_prediction
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
        # RL options
        self.max_ep_len = self.config.max_seq_len
        self.entropy_weight = self.config.entropy_weight
        self.n_glimps_levels = 1  # n_glimps_levels
        self.stop_enabled = self.config.stop_enabled
        # ===

        self.max_iters = config.max_iters
        self.log_every = config.test_every
        # self.plot_every = plot_every
        self.device = self.config.gpu_ids[rank]

        self.checkpoint_dir = Path(config.work_dir) / config.env_name
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.best_metric_name = "prop_patches_found"
        # last return values to compute mean and std
        self.last_return_values = []
        self.last_return_mean = 0
        self.last_return_std = 1

    def sample_from_logits(
        self,
        logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # sample from the last token
        last_action_logits = logits[:, -1, :]
        categorical = Categorical(logits=last_action_logits)
        actions = categorical.sample()
        logprobs = categorical.log_prob(actions)
        entropies = categorical.entropy()

        return actions, logprobs, entropies

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

    def rollout(self, env: NeedleGeneralEnv) -> Dict[str, torch.Tensor]:
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

        for step_id in range(env.max_ep_len):
            action_logits = self.model(patches, actions, classes, positions)
            new_actions, logprobs_, entropies_ = self.sample_from_logits(action_logits)
            new_patches, step_rewards, terminated, truncated, infos = env.step(
                new_actions
            )

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
        masks = torch.roll(masks, shifts=1, dims=(1,))
        masks[:, 0] = True

        rewards = torch.flip(rewards, dims=(1,))
        masks = torch.flip(masks, dims=(1,))
        cumulated_returns = torch.cumsum(rewards * masks, dim=1)
        cumulated_returns = torch.flip(cumulated_returns, dims=(1,))
        rewards = torch.flip(rewards, dims=(1,))
        masks = torch.flip(masks, dims=(1,))

        return {
            "rewards": rewards,
            "returns": cumulated_returns,
            "logprobs": logprobs,
            "entropies": entropies,
            "masks": masks,
        }

    def compute_metrics(
        self, rollout: Dict[str, torch.Tensor]
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
        masks = rollout["masks"]

        if self.config.reward_norm:
            self.last_return_values.append(returns[masks].clone().detach())
            mean, std = self.last_return_mean, self.last_return_std
            advantages = (returns - mean) / (std + 1e-8)
        else:
            advantages = returns

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

            (metrics["loss"] / self.config.gradient_accumulation).backward()

            metrics = dict()

            if self.iter_num % self.config.gradient_accumulation == 0:
                clip_grad.clip_grad_value_(self.model.parameters(), 1)
                self.optim_gpt.step()  # , self.optim_yolox.step()
                self.optim_gpt.zero_grad()  # , self.optim_yolox.zero_grad()

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

            rollout = self.rollout(env)
            metrics = self.compute_metrics(rollout)

            # Additional metrics for test
            # TODO put in another method
            metrics["prop_patches_found"] = env.prop_patches_found[0]
            metrics["prop_bbox_found"] = env.prop_bboxes_found[0]

            if self.stop_enabled:
                metrics["stop_used"] = env.terminated[0].to(torch.float32)
                metrics["stop_misused"] = (
                    env.terminated[0] and env.prop_patches_found[0] < 1
                ).to(torch.float32)

            for key, value in metrics.items():
                all_metrics[key].append(value.cpu())

            if loop_id in visual_ids:
                positions, masks, patches = self.predict(env)

                # Generate image
                plot_image = plot_model_prediction(
                    env.images[
                        0, 0
                    ].cpu(),  # XXX: we have to select the first glimps level
                    patches[0].cpu(),  # first of batch
                    positions[0][masks[0] == 1].cpu(),
                    true_bboxes=[],  # TODO ground truth bboxes
                    predicted_bboxes=[],
                )
                plot_images["model_images"].append(plot_image)

        dataset.translations, dataset.rotations = translations, rotations

        self.last_test_metrics = all_metrics
        self.best_metric_history.append(np.mean(all_metrics[self.best_metric_name]))
        self.logger.log_to_visdom(all_metrics, plot_images, "test")
        self.save_state()
        self.save_metrics()

    @torch.no_grad()
    def predict(
        self, env: NeedleGeneralEnv, sample_actions: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluates the model on a batch of images.
        Return a plot of its trajectories on all images.

        ---
        Args:
            env: The environment to evaluate the model on.

        ---
        Returns:
            positions: The positions visited by the model.
                Shape of [batch_size, max_ep_len + 1, 2].
            masks: The masks of the visited positions.
                Shape of [batch_size, max_ep_len + 1].
        """
        self.model.eval()
        classes = torch.zeros((env.batch_size,), dtype=torch.int64, device=self.device)
        actions = torch.zeros((env.batch_size, 1), dtype=torch.long, device=self.device)
        patches, infos = env.reset()
        positions = infos["positions"].unsqueeze(1)

        masks = torch.zeros(
            (env.batch_size, env.max_ep_len + 1),
            dtype=torch.bool,
            device=self.device,
        )
        masks[:, 0] = True

        for step_id in range(env.max_ep_len):
            logits = self.model(patches, actions, classes, positions)
            if sample_actions:
                new_actions, _, _ = self.sample_from_logits(logits)
            else:
                new_actions = logits[:, -1].argmax(dim=-1)

            new_patches, _, terminated, _, infos = env.step(new_actions)
            masks[:, step_id + 1] = ~terminated

            # Append for the next iteration of the model
            actions = torch.concat((actions, new_actions.unsqueeze(1)), dim=1)
            patches = torch.concat((patches, new_patches), dim=1)
            positions = torch.concat(
                (positions, infos["positions"].unsqueeze(1)), dim=1
            )

            if terminated:
                break

        # The last terminated state is not counted in the masks,
        # so we need to shift the masks by 1 to make sure we include id.
        masks = torch.roll(masks, shifts=1, dims=(1,))
        masks[:, 0] = True

        # complete positions so that they are the same size as masks
        positions = F.pad(
            positions, (0, 0, 0, env.max_ep_len - positions.shape[1] + 1), value=0
        )

        return positions, masks, patches
