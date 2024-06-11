import os
import time
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

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
        image_ids = list(range(len(self.test_dataset)))
        self.test_env_ids = rng.choice(image_ids, size=(self.config.test_samples,))
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
            "model": self.model.module.state_dict()
            if isinstance(self.model, DDP)
            else self.model.state_dict(),
            "optimizer-gpt": self.optim_gpt.state_dict(),
            "optimizer-yolox": self.optim_yolox.state_dict(),
        }

        torch.save(
            state,
            ckpt_path,
        )

    def prepare_validation(self):
        """
        Reload best checkpoint and test it over the entire dataset
        """
        # Reload best checkpoint
        work_dir = Path(self.config.work_dir)
        log_dir = work_dir / self.config.env_name
        ckpt_path = log_dir / "checkpoint_best.pt"

        if ckpt_path.exists():
            # TODO function trainer.load_checkpoint?
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
