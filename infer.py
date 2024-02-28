import argparse
import os
import json
from collections import defaultdict
import time

import torch
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision.utils import save_image
import numpy as np

from main import get_args, args_to_config, load_checkpoint, load_detection_checkpoint
from src.trainer import Trainer
from src.supervised import SupervisedTrainer
from src.reinforce import ReinforceTrainer
from src.models.gpt import GPT
from src.env import get_actions_info
from src.env.general_env import NeedleGeneralEnv
from src.utils import (
    CfgNode as CN,
    parse_bbox_predictions,
    parse_bbox_targets,
    plot_model_prediction,
)


def get_infer_args(args=None):
    parser = argparse.ArgumentParser(
        description="Inference and test script for JoliNeedle"
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Model directory, used to load the model checkpoint",
    )
    parser.add_argument(
        "--input-images", nargs="*", help="Input images to perform inference on"
    )
    parser.add_argument(
        "--dataset", help="Dataset over which perform tests and compute metrics"
    )
    parser.add_argument("--output-dir", help="Output directory for images, metrics...")
    parser.add_argument(
        "--track-object",
        action="store_true",
        help="Initialize model position at the patch where the last bbox was found on the previous image",
    )
    parser.add_argument(
        "--detection-checkpoint",
        type=str,
        help="Load detection model from given checkpoint file (can be different from --resume-training model)",
    )

    return parser.parse_args(args)


def config_from_file(config_path):
    """
    Load configuration from a json file
    """
    with open(config_path) as config_file:
        config_json = json.load(config_file)

    train_config = SupervisedTrainer.get_default_config()
    for key in config_json["train"]:
        setattr(train_config, key, config_json["train"][key])

    model_config = GPT.get_default_config()
    for key in config_json["model"]:
        setattr(model_config, key, config_json["model"][key])

    return train_config, model_config


def load_bboxes(bbox_fname):
    bboxes = []
    with open(bbox_fname) as bbox_file:
        for line in bbox_file:
            line = line.strip().split()
            # cls = int(line[0])
            bbox = [int(i) for i in line[1:5]]
            bboxes.append(bbox)
    return bboxes


def infer(args):
    config_path = os.path.join(args.model_dir, "config.json")
    train_config, model_config = config_from_file(config_path)

    device = "cuda"

    # Information to initialize model heads
    model_config.actions_info = get_actions_info(train_config)
    # Load model
    model = GPT(model_config)
    model.eval()

    # Create pipeline
    trainer = ReinforceTrainer(train_config, model, None, None, None, 0)

    # Load checkpoint
    train_config.resume_training = args.model_dir
    train_config.detection_checkpoint = args.detection_checkpoint

    load_checkpoint(train_config, trainer, best=True)
    if train_config.detection_checkpoint is not None:
        load_detection_checkpoint(train_config, trainer)
        train_config.detection_enabled = True

    # Load dataset
    image_paths = []
    target_paths = []

    if args.dataset:
        with open(args.dataset) as dset_file:
            for line in dset_file:
                line = line.strip().split()
                image_paths.append(line[0])
                target_paths.append(line[1])

    if args.input_images:
        image_paths += args.input_images

    # Do inference
    track_location = None
    all_metrics = defaultdict(list)

    for img_id in range(len(image_paths)):
        image_path = image_paths[img_id]
        print("Processing image %d/%d %s" % (img_id + 1, len(image_paths), image_path))
        image = read_image(image_path).to(device).unsqueeze(0).float() / 255
        # TODO convert image?

        has_targets = img_id < len(target_paths)
        if has_targets:
            bboxes = load_bboxes(target_paths[img_id])
            bboxes = torch.tensor([bboxes], device=device)
        else:
            bboxes = torch.zeros((1, 1, 4), device=device)

        patch_size = train_config.patch_size
        max_seq_len = train_config.max_seq_len

        # Pad image according to patch size
        padded_width = ((image.shape[-2] - 1) // patch_size + 1) * patch_size
        padded_height = ((image.shape[-1] - 1) // patch_size + 1) * patch_size
        image = F.pad(
            image,
            (0, padded_height - image.shape[-1], 0, padded_width - image.shape[-2]),
            value=0,
        )

        env = NeedleGeneralEnv(
            image, bboxes, patch_size, max_seq_len, 1, train_config.stop_enabled
        )
        with torch.inference_mode():
            start_ts = time.perf_counter()
            rollout = trainer.rollout(
                env, do_detection=train_config.detection_enabled, sample_actions=True
            )
            duration = time.perf_counter() - start_ts

        positions = rollout["positions"]
        masks = rollout["masks"]
        patches = rollout["patches"]
        bboxes = rollout["bboxes"]

        # metrics over trajectories
        traj_bbox_preds = rollout["bboxes"]
        # convert position y, x to x, y
        offsets = positions[:, :, [1, 0]] * patch_size
        full_img_preds = Trainer.patch_bboxes2full_image(
            traj_bbox_preds, offsets, masks
        )

        plot_image = plot_model_prediction(
            env.images[0, 0].cpu(),  # XXX: we have to select the first glimps level
            patches[0].cpu(),  # first of batch
            positions[0][masks[0] == 1].cpu(),
            true_bboxes=[],
            predicted_bboxes=parse_bbox_predictions(full_img_preds),
        )

        # Summary
        obj_count = 0 if full_img_preds[0] == None else len(full_img_preds[0])
        print(
            "Found %d objects in %d steps and %0.2fms"
            % (obj_count, positions.shape[1], duration * 1000)
        )

        save_img_path = os.path.join(args.output_dir, "result%d.png" % img_id)
        save_image(plot_image, save_img_path)

        # If dataset:
        # metrics over dataset
        if has_targets:
            metrics = trainer.compute_metrics(rollout, env)

            full_img_targets = env.get_detection_targets()
            traj_yolo_metrics = Trainer.compute_detection_metrics(
                full_img_preds,
                full_img_targets,
            )
            for metric_name, metric_value in traj_yolo_metrics.items():
                metrics[metric_name] = metric_value

            for mname in metrics:
                all_metrics[mname].append(metrics[mname].cpu().item())

    if len(target_paths) > 0:
        for mname, values in all_metrics.items():
            print("%s: %0.3f" % (mname, np.mean(values)))


if __name__ == "__main__":
    args = get_infer_args()
    infer(args)
