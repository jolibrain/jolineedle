import pytest

import torch

from src.trainer import Trainer
from src.env.general_env import NeedleGeneralEnv


def test_map():
    env = NeedleGeneralEnv(
        images=torch.zeros((1, 3, 1792, 2240)),
        bboxes=torch.tensor([[[410, 410, 500, 500], [1500, 1500, 1600, 1600]]]),
        patch_size=448,
        max_ep_len=20,
        n_glimps_levels=1,
    )

    preds1 = [None]
    targets = env.get_detection_targets()

    assert len(targets) == 1
    assert torch.equal(
        targets[0],
        torch.tensor(
            [
                [0, 410, 410, 447, 447],
                [0, 448, 410, 500, 447],
                [0, 410, 448, 447, 500],
                [0, 448, 448, 500, 500],
                [0, 1500, 1500, 1600, 1600],
            ],
            dtype=torch.int64,
        ),
    ), targets[0]

    metrics = Trainer.compute_detection_metrics(preds1, targets)

    assert metrics["map"] == pytest.approx(0.0)

    preds2 = [
        torch.tensor(
            [
                [410, 410, 447, 446, 0.5, 1],
                [448, 410, 500, 447, 0.9, 1],
                [410, 448, 447, 500, 0.8, 1],
                [448, 448, 500, 500, 0.7, 1],
                [1500, 1500, 1600, 1600, 0.6, 1],
            ]
        )
    ]

    metrics = Trainer.compute_detection_metrics(preds2, targets)
    assert metrics["map"] == pytest.approx(1)

    preds3 = [
        torch.tensor(
            [
                [410, 410, 447, 446, 0.5, 1],
                [410, 448, 447, 500, 0.8, 1],
                [448, 448, 500, 500, 0.7, 1],
                [1500, 1500, 1600, 1600, 0.6, 1],
            ]
        )
    ]
    metrics = Trainer.compute_detection_metrics(preds3, targets)
    assert metrics["map"] == pytest.approx(0.8, 0.01)


def test_bbox2full_image():
    patch_boxes = [
        [
            torch.tensor([[20, 40, 30, 100], [40, 60, 100, 90]]),
            torch.tensor([[38, 6, 90, 10]]),
            None,
            torch.tensor([[70, 30, 89, 59]]),
        ]
    ]
    offsets = torch.tensor([[[448, 0], [448, 448], [448, 896], [448, 1344]]])
    masks = torch.tensor([[True, True, True, False]])

    results = Trainer.patch_bboxes2full_image(patch_boxes, offsets, masks)
    expect = [
        torch.tensor([[468, 40, 478, 100], [488, 60, 548, 90], [486, 454, 538, 458]])
    ]

    for i, res in enumerate(results):
        assert torch.all(res == expect[i])
