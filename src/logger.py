import os
import json
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

from .utils import CfgNode
from .visualizer import VisdomPlotter


class Logger:
    def __init__(self, training_config: CfgNode, model_config: CfgNode):
        self.training_config = training_config
        self.model_config = model_config
        self.metrics = defaultdict(list)
        self.visdom = VisdomPlotter(training_config.env_name)

    def model_summary(self, model: torch.nn.Module):
        b_size = self.training_config.batch_size
        seq_len = self.training_config.max_seq_len
        actions = torch.zeros((b_size, seq_len), dtype=torch.long)
        positions = torch.zeros((b_size, seq_len, 2), dtype=torch.long)
        patches = torch.randn(
            (
                b_size,
                seq_len,
                self.model_config.n_channels,
                self.model_config.patch_size,
                self.model_config.patch_size,
            )
        )
        classes = torch.zeros((b_size,), dtype=torch.long)

        summary(
            model,
            input_data=(patches, actions, classes, positions),
            device=model.device,
            depth=2,
        )

    def log_args(self):
        """
        Log config args
        """
        args = " ".join(sys.argv)
        self.visdom.add_text(args, name="Arguments")

        self.visdom.add_table(
            {
                keyname: self.model_config.__dict__[keyname]
                for keyname in [
                    "model_type",
                    "block_size",
                    "n_channels",
                    "dropout",
                    "patch_size",
                    "image_processor",
                    "use_pos_emb",
                    "concat_emb",
                    "decoder_pos_encoding",
                ]
            },
            "Model config",
        )
        self.visdom.add_table(
            {
                keyname: self.training_config.__dict__[keyname]
                for keyname in [
                    "loss_mode",
                    "min_keypoints",
                    "max_keypoints",
                    "binomial_keypoints",
                    "rotations",
                    "translations",
                    "learning_rate",
                    "batch_size",
                    "device",
                    "num_workers",
                    "stop_weight",
                    "weight_decay",
                ]
            },
            "Training config",
        )

    def log_start(self, model: torch.nn.Module):
        """
        Log information at startup
        """
        self.model_summary(model)
        self.log_args()
        self.visdom.update()

    def log_batch_metrics(self, metrics):
        for metric_name, metric_value in metrics.items():
            self.metrics[metric_name].append(metric_value.cpu().item())

    def log_to_visdom(self, metrics: dict, images: dict, dataset_name: str):
        """
        Log given metrics to visdom

        dataset_name = test or train
        """
        # TODO remove NaNs & infs inside of metrics
        legends = {
            "prop_patches_found": "average % of bbox patches found in images",
            "prop_bbox_found": "average % of bboxes found in images",
            "episode_length": "average episode length",
            "stop_used": "% of rollouts stopped by the model",
            "stop_misused": "% of rollouts stopped too early by the model",
        }

        # autoregressive
        for name, values in metrics.items():
            eval_mode = "auto-regressive"
            if name.startswith("supervised_"):
                name = name[len("supervised_") :]
                eval_mode = "on generated trajectories"

            legend = legends[name] if name in legends else name
            if "yolo" in name:
                plot_name = "Yolo losses"
            elif "map" in name:
                plot_name = "map"
            elif (
                name.startswith("stopped_inside_bbox")
                or name.startswith("prop_patches_found")
                or name.startswith("prop_bbox_found")
            ):
                plot_name = f"BBox patches metrics ({dataset_name})"
            elif "episode_length" == name:
                plot_name = "Episode length"

                self.visdom.add_data(
                    np.std(values),
                    f"{plot_name} ({eval_mode})",
                    f"episode length std ({dataset_name})",
                )
            elif "stop" in name:
                plot_name = "Stop action metrics"
            elif "loss" in name:
                plot_name = "loss"
            elif "action" in name:
                plot_name = "action"
            elif "label" in name:
                plot_name = "label"
            else:
                plot_name = name

            self.visdom.add_data(
                np.mean(values),
                f"{plot_name} ({eval_mode})",
                f"{legend} ({dataset_name})",
            )

        self.visdom.add_images(
            torch.stack(images["model_images"]),
            f"Model predictions ({dataset_name})",
        )
        if "sample_images" in images:
            self.visdom.add_images(
                torch.stack(images["sample_images"]),
                f"Generated samples ({dataset_name})",
            )

        if "worst_images" in images and len(images["worst_images"]) != 0:
            self.visdom.add_images(
                torch.stack(images["worst_images"]),
                f"Model failures ({dataset_name})",
            )

        self.visdom.update()

    def save_visdom(self, log_dir):
        self.visdom.save(log_dir)
