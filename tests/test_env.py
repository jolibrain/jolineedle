import pytest

import torch

from src.trainer import Trainer
from src.env.general_env import NeedleGeneralEnv
from src.env.common import Action


def test_env():
    images = torch.zeros(1, 3, 1792, 2240)
    images[:, 0, 0:448, 448:896] = 255

    env = NeedleGeneralEnv(
        images=images,
        bboxes=torch.tensor([[[310, 810, 400, 850], [700, 1500, 800, 1600]]]),
        patch_size=448,
        max_ep_len=8,
        n_glimps_levels=1,
    )

    # position is (y, x) because of opencv
    patches, infos = env.reset(torch.tensor([[1, 0]]))
    assert torch.equal(infos["positions"], torch.tensor([[1, 0]]))

    env.step(torch.tensor([Action.RIGHT.value]))
    env.step(torch.tensor([Action.DOWN.value]))
    patches, reward, terminated, truncated, infos = env.step(
        torch.tensor([Action.DOWN.value])
    )
    assert torch.equal(infos["positions"], torch.tensor([[3, 1]]))
