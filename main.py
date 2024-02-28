import argparse
import math
import os
import random
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from thop import clever_format, profile

from src.dataset import build_datasets
from src.env import get_actions_info
from src.logger import Logger
from src.models.gpt import GPT
from src.supervised import SupervisedTrainer

from src.reinforce import ReinforceTrainer
from src.visualizer import VisdomPlotter


def get_args(args=None):
    parser = argparse.ArgumentParser(description="MinGPT Needle")

    # Model configs
    parser.add_argument(
        "--training-mode",
        type=str,
        default="supervised",
        choices=["supervised", "reinforce"],
        help="Which algorithm should be used to train the model",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="gpt-mini",
        help="Choose GPT general hyperparameters",
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=32, help="Maximum sequence length"
    )
    parser.add_argument(
        "--test-max-seq-len",
        type=int,
        help="Maximum sequence length for test. Must be less or equals to max-seq-len. Default is same as max-seq-len.",
    )
    parser.add_argument(
        "--patch-size", type=int, default=224, help="Size of a square patch"
    )
    parser.add_argument(
        "--minimum-image-size",
        type=int,
        default=224 * 5,
        help="Minimum size of the image, smaller images will be resized to this size",
    )
    parser.add_argument(
        "--no-detection",
        action="store_false",
        dest="detection_enabled",
        help="Disable detection model, use only decision (only for RL pipeline for now)",
    )
    parser.add_argument(
        "--image-processor",
        type=str,
        default="yolox",
        choices={
            "yolox",
            "yolox-nano",
            "yolox-tiny",
            "yolox-s",
            "yolox-m",
            "yolox-l",
            "yolox-x",
        },
        help="Choose the yolox size, default to nano",
    )
    parser.add_argument(
        "--gpt-backbone",
        type=str,
        choices={
            "yolox-nano",
            "yolox-tiny",
            "yolox-s",
            "yolox-m",
            "yolox-l",
            "yolox-x",
        },
        help="Backbone for GPT, if different from YOLOX",
    )
    parser.add_argument(
        "--freeze-image-processor",
        action="store_true",
        help="Should the yolox backbone processor be frozen",
    )
    parser.add_argument(
        "--detector-conf-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for the detector",
    )
    parser.add_argument(
        "--use-positional-embedding",
        action="store_true",
        help="Should the absolute positions of the patches be used",
    )
    parser.add_argument(
        "--no-patch-embedding",
        action="store_true",
        help="Remove the information from the patch embedder (for ablation studies)",
    )
    parser.add_argument(
        "--concat-embeddings",
        action="store_true",
        help="Should the embeddings be concatenated (averaged otherwise)",
    )
    parser.add_argument(
        "--decoder-pos-encoding",
        action="store_true",
        help="Replace the nn.Embedding positional embeddings with a 1D positional encoding",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability",
    )

    # Training configs
    parser.add_argument(
        "--enable-stop",
        action="store_true",
        help="Enable STOP action for the model to decide to stop early",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay for the optimizer",
    )
    parser.add_argument(
        "--stop-weight",
        type=float,
        default=1.0,
        help="Weight of the STOP action in cross-entropy loss for supervised training",
    )
    parser.add_argument(
        "--no-reward-norm",
        action="store_false",
        dest="reward_norm",
        help="Disable reward normalization in RL pipeline",
    )
    parser.add_argument(
        "--entropy-weight",
        type=float,
        default=0.01,
        help="Entropy loss weight in RL training",
    )
    parser.add_argument(
        "--binomial-keypoints",
        action="store_true",
        help="Should the keypoints be sampled from a binomial distribution",
    )
    parser.add_argument(
        "--min-keypoints",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--max-keypoints",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--merge-bboxes",
        action="store_true",
        help="Merge bounding boxes before computing metrics. This yields more representative mAP (we can directly compare it with a detector mAP).\n\nSince it adds post-processing (= source of confusion) to the results, we might not want to merge bbox in every case. Take extra care when the boxes could be superposed or close to each other.",
    )
    parser.add_argument(
        "--loss",
        choices={"on-self-trajectory", "on-optimal-trajectory"},
        default="on-optimal-trajectory",
        help="Decide if the reference action are based on the trajectory itself or on the optimal trajectory",
    )
    parser.add_argument(
        "--yolo-lr",
        type=float,
        default=1e-4,
        help="Learning rate for the YOLO model",
    )
    parser.add_argument(
        "--augment-rotate",
        action="store_true",
        help="Do augment the dataset with rotations",
    )
    parser.add_argument(
        "--augment-translate",
        action="store_true",
        help="Do augment the dataset with translations.",
    )
    parser.add_argument(
        "--devices", nargs="+", type=int, help="Choose devices to train on"
    )
    parser.add_argument("--port-ddp", type=int, help="Choose ddp port", default=12355)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--max-iters",
        type=int,
        default=1000,
        help="Number of training iterations (in batches)",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--gradient-accumulation", type=int, default=1, help="Gradient accumulation"
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default="test",
        help="Visdom environment name",
    )
    parser.add_argument(
        "--group",
        type=str,
        default="",
        help="Visdom group environment name",
    )
    parser.add_argument(
        "--work-dir", type=str, default="./out/", help="Where to log the results"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.01, help="Fraction of test samples"
    )
    parser.add_argument(
        "--test-samples", type=int, default=100, help="Number of test samples"
    )
    parser.add_argument(
        "--test-pattern", type=str, default="", help="Pattern of test image names"
    )
    parser.add_argument(
        "--test-every",
        type=int,
        default=500,
        help="How many iterations to wait between each test evaluations",
    )
    parser.add_argument(
        "--failure-select-rate",
        type=float,
        default=0.1,
        help="Proportion of model failures (worst metrics at test time) kept for evaluation",
    )
    parser.add_argument(
        "--eval-training-set",
        action="store_true",
        help="Should the model be evaluated on the training set",
    )
    parser.add_argument(
        "--resume-training",
        type=str,
        required=False,
        help="Resume precedent training, give the precedent checkpoint directory as arg",
    )
    parser.add_argument(
        "--detection-checkpoint",
        type=str,
        help="Load detection model from given checkpoint file (can be different from --resume-training model)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help="Path to the dataset",
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Fix the seed of the experiment",
    )
    parser.add_argument(
        "--train-size", type=int, default=-1, help="Set number of training samples"
    )
    parser.add_argument(
        "--num-workers", type=int, default=1, help="Number of workers in the dataloader"
    )
    parser.add_argument(
        "--generated-sample-eval-size",
        type=int,
        default=500,
        help="Number of generated samples to evaluate for each evaluation loop",
    )
    parser.add_argument(
        "--filter-classes",
        action="append",
        help="Filter the classes to train on",
    )
    parser.add_argument("--measure-flops", action="store_true", help="Measure flops")

    # Deprecated options
    parser.add_argument(
        "--no-recurrent-embedding",
        action="store_true",
        help="Normally the embeddings are reused from one patch to another to save resources. This should not have an impact on training, but in some cases the result is different without the optimization",
    )

    return parser.parse_args(args)


def args_to_config(args):
    # TODO reorder options into semantic groups
    train_config = SupervisedTrainer.get_default_config()
    train_config.training_mode = args.training_mode
    train_config.rotations = args.augment_rotate
    train_config.translations = args.augment_translate
    train_config.learning_rate = args.lr
    train_config.max_iters = args.max_iters
    train_config.batch_size = args.batch_size
    train_config.detection_enabled = args.detection_enabled
    train_config.gradient_accumulation = args.gradient_accumulation
    train_config.env_name = args.env_name
    train_config.work_dir = args.work_dir
    train_config.test_size = args.test_size
    train_config.test_samples = args.test_samples
    train_config.test_pattern = args.test_pattern
    train_config.test_every = args.test_every
    train_config.failure_select_rate = args.failure_select_rate
    train_config.eval_training_set = args.eval_training_set
    train_config.resume_training = args.resume_training
    train_config.detection_checkpoint = args.detection_checkpoint
    train_config.merge_bboxes = args.merge_bboxes
    train_config.seed = args.seed
    train_config.train_size = args.train_size
    train_config.num_workers = args.num_workers
    train_config.min_keypoints = args.min_keypoints
    train_config.max_keypoints = args.max_keypoints
    train_config.loss_mode = args.loss
    train_config.yolo_lr = args.yolo_lr
    train_config.binomial_keypoints = args.binomial_keypoints
    train_config.generated_sample_eval_size = args.generated_sample_eval_size
    train_config.weight_decay = args.weight_decay
    train_config.stop_weight = args.stop_weight
    train_config.entropy_weight = args.entropy_weight
    train_config.reward_norm = args.reward_norm
    train_config.minimum_image_size = args.minimum_image_size
    train_config.filter_classes = (
        set(int(c) for c in args.filter_classes)
        if args.filter_classes is not None
        else None
    )
    train_config.port_ddp = args.port_ddp
    train_config.measure_flops = args.measure_flops
    print("Filter classes:", train_config.filter_classes)
    train_config.gpu_ids = args.devices
    train_config.world_size = len(args.devices)
    print("Using GPUs:", train_config.gpu_ids)
    train_config.max_seq_len = args.max_seq_len
    train_config.test_max_seq_len = (
        args.test_max_seq_len if args.test_max_seq_len else args.max_seq_len
    )
    train_config.patch_size = args.patch_size
    train_config.n_channels = 3
    train_config.stop_enabled = args.enable_stop
    train_config.image_cols = math.ceil(
        2064 / train_config.patch_size
    )  # XXX: 2064 is dataset image width

    model_config = GPT.get_default_config()
    model_config.model_type = args.model_type
    model_config.image_processor = args.image_processor  # yolox
    model_config.gpt_backbone = args.gpt_backbone
    model_config.freeze_image_processor = args.freeze_image_processor
    model_config.detector_conf_threshold = args.detector_conf_threshold
    model_config.use_pos_emb = args.use_positional_embedding
    model_config.no_patch_emb = args.no_patch_embedding
    model_config.concat_emb = args.concat_embeddings
    model_config.decoder_pos_encoding = args.decoder_pos_encoding
    model_config.pos_emb_size = train_config.image_cols**2
    model_config.dropout = args.dropout

    model_config.block_size = train_config.max_seq_len
    model_config.n_channels = train_config.n_channels
    model_config.patch_size = train_config.patch_size
    model_config.image_cols = train_config.image_cols

    model_config.no_recurrent_embedding = args.no_recurrent_embedding

    return train_config, model_config


def main(args):
    train_config, model_config = args_to_config(args)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)
    np.random.seed(train_config.seed)

    # TODO do that only once on the rank 0 thread
    dataset_dir = args.dataset_dir
    train_dataset, test_dataset = build_datasets(
        dataset_dir,
        min_keypoints=train_config.min_keypoints,
        max_keypoints=train_config.max_keypoints,
        patch_size=model_config.patch_size,
        max_ep_len=model_config.block_size,
        rotations=train_config.rotations,
        translations=train_config.translations,
        test_size=train_config.test_size,
        test_pattern=train_config.test_pattern,
        seed=train_config.seed,
        train_size=train_config.train_size,
        binomial_keypoints=train_config.binomial_keypoints,
        minimum_image_size=train_config.minimum_image_size,
        filter_classes=train_config.filter_classes,
    )
    if args.group != "":
        train_config.env_name = f"{args.group}_{train_config.env_name}"
    print(
        f"training env_name = {train_config.env_name}",
        f"\nsize of train_dataset = {len(train_dataset)}",
        f"\ttest_dataset = {len(test_dataset)}",
        f"\nUsing positional encoding = {args.use_positional_embedding}",
        f"\nConcatenating embeddings = {args.concat_embeddings}",
    )
    if args.no_patch_embedding:
        print("No patch embeddings!")
    save_config(model_config, train_config)

    mp.spawn(
        launch_ddp_training,
        args=(train_config.world_size, train_config, model_config, dataset_dir),
        nprocs=train_config.world_size,
        join=True,
    )


def save_config(model_config, train_config):
    train_folder = Path(train_config.work_dir) / train_config.env_name
    train_folder.mkdir(parents=True, exist_ok=True)
    config_path = train_folder / "config.json"
    config_json = dict()

    config_json["model"] = vars(model_config)
    train_config = deepcopy(train_config)
    if train_config.filter_classes:
        train_config.filter_classes = list(train_config.filter_classes)
    config_json["train"] = vars(train_config)

    with open(config_path, "w") as config_file:
        json.dump(config_json, config_file, indent=4)


def compute_flops(model, model_config, train_config, device):
    model = deepcopy(model)
    b_size = 1  # train_config.batch_size
    nc = model_config.n_channels
    patch_size = model_config.patch_size
    print(f"Computing MACs for patch size = {patch_size} and batch size = {b_size}")

    # backbone MACs
    one_patch = torch.randn((b_size, nc, patch_size, patch_size), device=device)
    bkb_macs, bkb_params = profile(
        model.yolox.backbone, inputs=[one_patch], verbose=False
    )
    pretty_bkb_macs, pretty_bkb_params = clever_format([bkb_macs, bkb_params], "%.3f")
    print(f"Backbone MACs: {pretty_bkb_macs}, Params: {pretty_bkb_params}")

    # head MACs
    yolo_macs, yolo_params = profile(model.yolox, inputs=[one_patch], verbose=False)
    pretty_head_macs, pretty_head_params = clever_format(
        [yolo_macs - bkb_macs, yolo_params - bkb_params], "%.3f"
    )
    print(f"Yolox Head MACs: {pretty_head_macs}, Params: {pretty_head_params}")

    for seq_len in [1, 2, 4, 8, 16]:
        actions = torch.zeros((b_size, seq_len), dtype=torch.long)
        positions = torch.zeros((b_size, seq_len, 2), dtype=torch.long)
        patches = torch.randn(
            (
                b_size,
                seq_len,
                nc,
                patch_size,
                patch_size,
            )
        )
        classes = torch.zeros((b_size,), dtype=torch.long)

        inputs = (patches, actions, classes, positions)
        inputs = [i.to(device) for i in inputs]
        macs, params = profile(model, inputs=inputs, verbose=False)

        # GPT MACs for the entire sequence
        pretty_gpt_macs, pretty_gpt_params = clever_format(
            [macs - seq_len * yolo_macs, params - yolo_params], "%.3f"
        )
        print(
            f"Seq len: {seq_len}, GPT MACs: {pretty_gpt_macs}, Params: {pretty_gpt_params}"
        )
        # Embedding details for sequence size = 8, yolox s & gpt nano
        # - YOLO 17G
        # - FPN embd 13M
        # - Transformer 260K

        # Total MACs
        macs, params = clever_format([macs, params], "%.3f")
        print(f"Seq len: {seq_len}, Complete Model MACs: {macs}, Params: {params}")

    # Compute for big image
    big_img_size = patch_size * 8
    print(f"Compute Yolox MACs for image of size {big_img_size}x{big_img_size}")
    full_img = torch.randn((b_size, nc, big_img_size, big_img_size), device=device)
    bkb_macs, bkb_params = profile(
        model.yolox.backbone, inputs=[full_img], verbose=False
    )
    pretty_bkb_macs, pretty_bkb_params = clever_format([bkb_macs, bkb_params], "%.3f")
    print(f"Backbone MACs: {pretty_bkb_macs}, Params: {pretty_bkb_params}")

    # head MACs
    yolo_macs, yolo_params = profile(
        model.yolox, inputs=[full_img.unsqueeze(0)], verbose=False
    )
    pretty_head_macs, pretty_head_params = clever_format(
        [yolo_macs - bkb_macs, yolo_params - bkb_params], "%.3f"
    )
    print(f"Yolox Head MACs: {pretty_head_macs}, Params: {pretty_head_params}")
    pretty_yolo_macs, pretty_yolo_params = clever_format(
        [yolo_macs, yolo_params], "%.3f"
    )
    print(f"Yolox total MACS: {pretty_yolo_macs}, Params: {pretty_yolo_params}")


def load_checkpoint(train_config, trainer, visdom=None, best=False):
    print("Resuming from ", train_config.resume_training)
    log_dir = Path(train_config.resume_training)
    if best:
        checkpoint = torch.load(log_dir / "checkpoint_best.pt", map_location="cpu")
    else:
        checkpoint = torch.load(log_dir / "checkpoint.pt", map_location="cpu")

    # Load the model checkpoint and remove the DDP wrapper.
    model_checkpoint = dict()
    for module_name, params in checkpoint["model"].items():
        module_name = module_name.replace("module.", "")
        model_checkpoint[module_name] = params

    trainer.model.load_state_dict(model_checkpoint)
    trainer.optim_gpt.load_state_dict(checkpoint["optimizer-gpt"])
    trainer.optim_yolox.load_state_dict(checkpoint["optimizer-yolox"])

    lr = train_config.learning_rate
    lr_yolo = train_config.yolo_lr
    weight_decay = train_config.weight_decay

    trainer.optim_gpt.lr = lr
    trainer.optim_gpt.weight_decay = weight_decay
    trainer.optim_yolox.lr = lr_yolo
    trainer.optim_yolox.weight_decay = weight_decay

    if visdom:
        visdom = VisdomPlotter.load(log_dir / "visdom.pkl", train_config.env_name)

    return visdom


def load_detection_checkpoint(train_config, trainer):
    print("Load detection checkpoint from", train_config.detection_checkpoint)
    checkpoint = torch.load(train_config.detection_checkpoint, map_location="cpu")

    # TODO allow to load only jit model from dede or pretrained checkpoint
    # Load the model checkpoint and remove the DDP wrapper.
    model_checkpoint = dict()
    for module_name, params in checkpoint["model"].items():
        module_name = module_name.replace("module.", "")

        if module_name.startswith("yolox."):
            module_name = module_name[6:]
            model_checkpoint[module_name] = params

    trainer.yolox_model().load_state_dict(model_checkpoint)
    if "optimize-yolox" in checkpoint:
        trainer.optim_yolox.load_state_dict(checkpoint["optimizer-yolox"])

    trainer.optim_yolox.lr = train_config.yolo_lr
    trainer.optim_yolox.weight_decay = train_config.weight_decay


def launch_ddp_training(rank, world_size, train_config, model_config, dataset_dir):
    train_dataset, test_dataset = build_datasets(
        dataset_dir,
        min_keypoints=train_config.min_keypoints,
        max_keypoints=train_config.max_keypoints,
        patch_size=train_config.patch_size,
        max_ep_len=train_config.max_seq_len,
        rotations=train_config.rotations,
        translations=train_config.translations,
        test_size=train_config.test_size,
        test_pattern=train_config.test_pattern,
        seed=train_config.seed,
        train_size=train_config.train_size,
        binomial_keypoints=train_config.binomial_keypoints,
        minimum_image_size=train_config.minimum_image_size,
        filter_classes=train_config.filter_classes,
    )

    # Information to initialize model heads
    model_config.actions_info = get_actions_info(train_config)

    # TODO allow other models than GPT
    model = GPT(model_config)

    # Init logger
    visdom, logger = None, None
    if rank == 0:
        logger = Logger(train_config, model_config)

    # Init trainer
    if train_config.training_mode == "reinforce":
        trainer = ReinforceTrainer(
            train_config, model, logger, train_dataset, test_dataset, rank
        )
    elif train_config.training_mode == "supervised":
        trainer = SupervisedTrainer(
            train_config, model, logger, train_dataset, test_dataset, rank
        )
    else:
        raise ValueError("Unkown training mode: %s" % train_config.training_mode)

    # Reload checkpoint
    if train_config.resume_training is not None:
        visdom = load_checkpoint(train_config, trainer, visdom=(rank == 0))
        if logger:
            logger.visdom = visdom

    if train_config.detection_checkpoint is not None:
        load_detection_checkpoint(train_config, trainer)

    if rank == 0 and train_config.measure_flops:
        compute_flops(model, model_config, train_config, trainer.device)
        return

    trainer.run(rank, world_size, train_config.port_ddp)


if __name__ == "__main__":
    args = get_args()
    main(args)
