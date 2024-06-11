from enum import Enum
from dataclasses import dataclass

_ACTIONS = [
    "LEFT",
    "RIGHT",
    "UP",
    "DOWN",
    "LEFT_UP",
    "RIGHT_UP",
    "LEFT_DOWN",
    "RIGHT_DOWN",
    "STOP",
]
Action = Enum("Action", _ACTIONS, start=0)

ACTION_DELTAS = {
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT_UP: (-1, -1),
    Action.RIGHT_UP: (-1, 1),
    Action.LEFT_DOWN: (1, -1),
    Action.RIGHT_DOWN: (1, 1),
    Action.STOP: (0, 0),
}

MOVES = [
    Action.LEFT,
    Action.RIGHT,
    Action.UP,
    Action.DOWN,
    Action.LEFT_UP,
    Action.RIGHT_UP,
    Action.LEFT_DOWN,
    Action.RIGHT_DOWN,
]


@dataclass
class ActionInfo:
    action_type: str  # categorical or scalar
    nclasses: int
    # range: Tuple[float]


def get_actions_info(train_config):
    actions = []
    actions.append(
        ActionInfo(
            action_type="categorical",
            nclasses=len(Action) if train_config.stop_enabled else len(Action) - 1,
        )
    )
    return actions
