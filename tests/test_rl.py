import pytest

import main
from test_common import *


def test_reinforce_pipeline(work_dir):
    download_dataset(work_dir, TOY_LARD)
    dset_path = os.path.join(work_dir, TOY_LARD["subdir"])

    # TODO allow to disable visdom
    # fmt: off
    cli_args = [
        "--seed", "12345",
        "--port-ddp", "12346",
        "--dataset-dir", dset_path,
        "--training-mode", "reinforce",
        "--work-dir", os.path.join(work_dir, "checkpoints"),
        "--max-iters", "3",
        "--test-every", "2",
        "--generated-sample-eval-size", "4",
        "--test-samples", "4",
        "--env-name", "reinforce",
        "--group", "test",
        "--model-type", "gpt-nano",
        "--gpt-backbone", "yolox-nano",
        "--image-processor", "yolox-s",
        "--concat-embeddings",
        "--decoder-pos-encoding",
        "--use-positional-embedding",
        "--loss", "on-optimal-trajectory",
        "--binomial-keypoints",
        "--max-seq-len", "8",
        "--batch-size", "4",
        "--gradient-accumulation", "4",
        "--max-keypoints", "3",
        "--min-keypoints", "0",
        "--dropout", "0.0",
        "--num-workers", "1",
        "--patch-size", "448",
        "--stop-weight", "0.1",
        "--detector-conf-threshold", "0.50",
        "--lr", "0.0001",
        "--yolo-lr", "0.0001",
        "--devices", "0",
        "--augment-translate",
        "--enable-stop",
    ]
    # fmt: on
    args = main.get_args(cli_args)

    main.main(args)

    # TODO check that files are generated correctly
    # TODO load metrics, best model file...
