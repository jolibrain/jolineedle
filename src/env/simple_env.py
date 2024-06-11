import random
from math import floor
from typing import Optional, Set

import einops
import torch
from torch.utils.data import default_collate

from .common import Action, MOVES, ACTION_DELTAS
from ..utils import BBox, Position


def pixel_pos_to_patch_pos(pixel_position: Position, patch_size: int) -> Position:
    """Returns the patch position in the image containing the given pixel position."""
    return Position(
        y=floor(pixel_position.y / patch_size),
        x=floor(pixel_position.x / patch_size),
    )


def closest_bbox_patch(src: Position, bbox: BBox) -> Position:
    """Returns the position of the closest patch being in the bbox."""
    x = max(src.x, bbox.up_left.x)
    x = min(x, bbox.bottom_right.x)

    y = max(src.y, bbox.up_left.y)
    y = min(y, bbox.bottom_right.y)

    return Position(y, x)


def closest_patch(position: Position, patches: Set[Position]) -> Position:
    def dist_to(patch_pos: Position) -> int:
        return abs(position.x - patch_pos.x) + abs(position.y - patch_pos.y)

    distances = [(dist_to(patch_pos), patch_pos) for patch_pos in patches]
    distances = list(sorted(distances, key=lambda el: el[0]))

    min_distance = distances[0][0]
    best_patches = [
        patch_pos for distance, patch_pos in distances if distance == min_distance
    ]
    return random.choice(best_patches)


def iter_corners(bbox: BBox):
    """Yields all corners starting from `up_left` turning anti-clockwise."""
    up_left, bottom_right = bbox
    yield up_left
    yield Position(bottom_right.y, up_left.x)
    yield bottom_right
    yield Position(up_left.y, bottom_right.x)


def get_patch(image: torch.Tensor, patch_size: int, position: Position) -> torch.Tensor:
    """Get the patch in the image at the given location.

    Args
        image: Tensor containing all pixels.
            Shape of [n_channels, height, width].
        patch_size: Size of a patch.
        position: Tuple (y, x) of the position of the patch.

    Returns
        tile: The patch of pixels of shape [n_channels, patch_size, patch_size].
    """
    _, height, width = image.shape
    assert height % patch_size == 0
    assert width % patch_size == 0

    n_patches_height = height // patch_size
    n_patches_width = width // patch_size
    assert 0 <= position[0] < n_patches_height
    assert 0 <= position[1] < n_patches_width

    # Reshape to [n_patches_height, n_patches_width, n_channels, patch_size, patch_size].
    tiled_image = einops.rearrange(
        image, "c (p1 h) (p2 w) -> p1 p2 c h w", p1=n_patches_height, p2=n_patches_width
    )

    return tiled_image[position.y, position.x]


def move_towards(current_position: Position, target_position: Position) -> Action:
    """Find the action to use to go towards the target position from
    the current position.

    Args
        current_position: Tuple (y, x) of the current position.
        target_position: Tuple (y, x) of the target position.

    Returns
        action: The action to apply to get one step closer to the target position.
    """
    gradient = Position(
        target_position.y - current_position.y,
        target_position.x - current_position.x,
    )

    if gradient.y > 0 and gradient.x == 0:
        return Action.DOWN
    elif gradient.y < 0 and gradient.x == 0:
        return Action.UP
    elif gradient.x > 0 and gradient.y == 0:
        return Action.RIGHT
    elif gradient.x < 0 and gradient.y == 0:
        return Action.LEFT
    # elif gradient.y < 0 and gradient.x > 0:
    #     return random.choice([Action.RIGHT, Action.UP])
    # elif gradient.y < 0 and gradient.x < 0:
    #     return random.choice([Action.LEFT, Action.UP])
    # elif gradient.y > 0 and gradient.x > 0:
    #     return random.choice([Action.RIGHT, Action.DOWN])
    # elif gradient.y > 0 and gradient.x < 0:
    #     return random.choice([Action.LEFT, Action.DOWN])
    elif gradient.y < 0 and gradient.x > 0:
        return Action.RIGHT_UP
    elif gradient.y < 0 and gradient.x < 0:
        return Action.LEFT_UP
    elif gradient.y > 0 and gradient.x > 0:
        return Action.RIGHT_DOWN
    elif gradient.y > 0 and gradient.x < 0:
        return Action.LEFT_DOWN

    return Action.STOP


def apply_action(current_position: Position, action: Action) -> Position:
    """Apply action and return the new position.
    Does not check whether we're outside the image!
    """
    delta = Position(*ACTION_DELTAS[action])
    return Position(current_position.y + delta.y, current_position.x + delta.x)


def closest_non_visited_corner(
    position: Position, bbox: BBox, visited_corners: list
) -> Optional[Position]:
    """Return the position of the closest corner from the given position.
    If all corners are visited, it returns None.
    """
    target_pos = None
    min_distance = float("+inf")
    for corner in iter_corners(bbox):
        if corner in visited_corners:
            continue

        distance = (position.x - corner.x) ** 2 + (position.y - corner.y) ** 2

        if distance < min_distance:
            min_distance = distance
            target_pos = corner

    return target_pos


import random
from copy import deepcopy
from itertools import product
from typing import List, Optional, Set, Union

import numpy as np
import torch


class NeedleSimpleEnv:
    def __init__(
        self,
        image: torch.Tensor,
        patch_size: int,
        bboxes: List[BBox],
        seed: Optional[int] = None,
    ):
        """Creates one environment for a specific image and a bbox.

        ---
        Args:
            image: Tensor of the image.
                Shape of [n_channels, height, width].
            patch_size: Size of one patch of the image.
            bbox: The bbox where is located the object in the image.
                The bbox is to be given in pixel space.
        """
        self.image = image
        self.patch_size = patch_size
        self.rng = np.random.default_rng(seed)
        self.raw_bboxes = bboxes
        self.bboxes = [
            BBox(
                pixel_pos_to_patch_pos(bbox.up_left, self.patch_size),
                pixel_pos_to_patch_pos(bbox.bottom_right, self.patch_size),
            )
            for bbox in bboxes
        ]  # In patch space.
        self.position = Position(0, 0)  # In patch space.
        self.infos = dict()

        self.n_channels, self.height, self.width = self.image.shape
        self.patch_height = self.height // self.patch_size
        self.patch_width = self.width // self.patch_size

        self.bbox_patches = set()
        for bbox in bboxes:
            self.bbox_patches = self.bbox_patches | self.bbox_positions(bbox)

        self.visited_bbox_patches = set()

    def gather_infos(self) -> dict:
        """Returns additional information based on the current position.
        A copy of the dictionary is returned to avoid side effects.

        ---
        Returns:
            infos: Dictionary containing additional information:
                * inside_bbox: Whether the agent is inside the bbox.
                * position: The current position of the agent.
                * number_patches_found: The number of patches found.
                * local_bboxes: The local bboxes.
        """
        infos = dict()

        infos["position"] = self.position
        infos["number_patches_found"] = len(self.visited_bbox_patches)
        infos["local_bboxes"] = self.local_bboxes()
        infos["inside_bbox"] = self.position in self.bbox_patches

        self.infos = deepcopy(infos)

        return infos

    def local_bboxes(self, position: Optional[Position] = None) -> torch.Tensor:
        """Return the overlapping bbox between the current patch
        and the raw bbox.

        It is exprimed in the current patch absolute coordinates:
            - The class of the bbox.
            - [cx, cy, width, height]
            - The objectivness of the bbox: whether or not the bbox is valid.
        If the bbox don't overlap, it returns a zero tensor.
        """
        if position is None:
            position = self.position

        local_bboxes = torch.zeros((len(self.bboxes), 6), dtype=torch.float32)
        x1_patch, y1_patch = (
            position.x * self.patch_size,
            position.y * self.patch_size,
        )
        x2_patch, y2_patch = x1_patch + self.patch_size, y1_patch + self.patch_size

        for bbox_id, raw_bbox in enumerate(self.raw_bboxes):
            x1 = max(x1_patch, raw_bbox.up_left.x)
            y1 = max(y1_patch, raw_bbox.up_left.y)
            x2 = min(x2_patch, raw_bbox.bottom_right.x)
            y2 = min(y2_patch, raw_bbox.bottom_right.y)

            if (x1_patch <= x1 < x2 <= x2_patch) and (y1_patch <= y1 < y2 <= y2_patch):
                local_bboxes[bbox_id] = torch.FloatTensor(
                    [
                        0,  # The class id.
                        (x1 + x2) / 2 - x1_patch,
                        (y1 + y2) / 2 - y1_patch,
                        (x2 - x1),
                        (y2 - y1),
                        1,  # The objectivness.
                    ]
                )
        return local_bboxes

    def bbox_positions(
        self, raw_bbox: BBox, area_threshold: float = 0.05
    ) -> Set[Position]:
        bbox_patches = set()
        bbox = BBox(
            up_left=pixel_pos_to_patch_pos(raw_bbox.up_left, self.patch_size),
            bottom_right=pixel_pos_to_patch_pos(raw_bbox.bottom_right, self.patch_size),
        )
        for y, x in product(
            range(bbox.up_left.y, bbox.bottom_right.y + 1),
            range(bbox.up_left.x, bbox.bottom_right.x + 1),
        ):
            position = Position(y, x)

            # Check if the patch contains enough pixels of the object.
            bbox_patch = BBox(
                Position(
                    max(position.y * self.patch_size, raw_bbox.up_left.y),
                    max(position.x * self.patch_size, raw_bbox.up_left.x),
                ),
                Position(
                    min(
                        (position.y + 1) * self.patch_size,
                        raw_bbox.bottom_right.y,
                    ),
                    min(
                        (position.x + 1) * self.patch_size,
                        raw_bbox.bottom_right.x,
                    ),
                ),
            )
            bbox_area = (
                (bbox_patch.bottom_right.y - bbox_patch.up_left.y)
                * (bbox_patch.bottom_right.x - bbox_patch.up_left.x)
                / (self.patch_size**2)
            )
            if bbox_area > area_threshold:
                # Add the patch only if it contains enough pixels of the object.
                bbox_patches.add(position)

        # Makes sure the center at least is visited.
        center_pos = Position(
            y=(raw_bbox.up_left.y + raw_bbox.bottom_right.y) // 2,
            x=(raw_bbox.up_left.x + raw_bbox.bottom_right.x) // 2,
        )
        center_pos = pixel_pos_to_patch_pos(center_pos, self.patch_size)
        bbox_patches.add(center_pos)

        # Remove the patches that are outside the image.
        bbox_patches = {pos for pos in bbox_patches if 0 <= pos.x < self.patch_width}
        bbox_patches = {pos for pos in bbox_patches if 0 <= pos.y < self.patch_height}
        return bbox_patches

    def reset(
        self,
        position: Optional[Position] = None,
        visited_bbox_patches: Optional[Set[Position]] = None,
    ):
        """Samples a new position and returns the observation."""
        if position is None:
            position = Position(
                y=self.rng.integers(low=0, high=self.patch_height),
                x=self.rng.integers(low=0, high=self.patch_width),
            )

        self.position = position
        patch = get_patch(self.image, self.patch_size, self.position)

        if visited_bbox_patches is None:
            visited_bbox_patches = set()
        self.visited_bbox_patches = visited_bbox_patches

        if self.position in self.bbox_patches:
            self.visited_bbox_patches.add(self.position)

        return patch, self.gather_infos()

    def step(self, move: Union[Action, np.ndarray]):
        """Applies the action and returns the new observation.

        ---
        Args:
            move: The move to make.
            label: The label of the current patch (before moving).

        ---
        Returns:
            patch: The new observation.
            reward: The reward obtained.
            infos: Additional information, see `self.gather_infos`.
        """
        # Convert to `Action` if necessary.
        move = Action(move.item()) if type(move) is np.ndarray else move

        self.position = apply_action(self.position, move)
        self.position = Position(
            min(max(self.position.y, 0), self.patch_height - 1),
            min(max(self.position.x, 0), self.patch_width - 1),
        )  # Make sure we do not cross the borders of the image.

        # Update visited patches.
        if self.position in self.bbox_patches:
            self.visited_bbox_patches.add(self.position)

        infos = self.gather_infos()
        patch = get_patch(self.image, self.patch_size, self.position)
        return patch, infos

    def init_sample(self, max_ep_len: int, device: str):
        sample = {
            "patches": torch.zeros(
                (max_ep_len, self.n_channels, self.patch_size, self.patch_size),
                dtype=torch.float,
                device=device,
            ),
            "current_actions": torch.zeros(
                (max_ep_len,), dtype=torch.long, device=device
            ),
            "next_actions": torch.zeros((max_ep_len,), dtype=torch.long, device=device),
            "positions": torch.zeros((max_ep_len, 2), dtype=torch.long, device=device),
            "masks": torch.zeros((max_ep_len,), dtype=torch.float, device=device),
            "labels": torch.zeros((max_ep_len,), dtype=torch.long, device=device),
            "local_bboxes": torch.zeros(
                (max_ep_len, len(self.bboxes), 6), device=device
            ),
        }

        bboxes_positions = set()
        patches_yolox, bboxes_yolox = [], []

        # Find the positions of the bboxes.
        for raw_bbox in self.raw_bboxes:
            for pos in self.bbox_positions(raw_bbox):
                bboxes_positions.add(pos)

        # Add a random empty patch.
        empty_positions = [
            Position(y=y, x=x)
            for y, x in product(range(self.patch_height), range(self.patch_width))
            if Position(y=y, x=x) not in bboxes_positions
        ]
        if len(empty_positions) > 0:
            pos_id = self.rng.choice(len(empty_positions))
            pos = empty_positions[pos_id]
            bboxes_positions.add(pos)

        # Add bboxes.
        for pos in bboxes_positions:
            patches_yolox.append(get_patch(self.image, self.patch_size, pos))
            bboxes_yolox.append(self.local_bboxes(pos))

        if len(patches_yolox) == 0:
            # Fictitious bbox.
            patches_yolox.append(
                torch.zeros(
                    (self.n_channels, self.patch_size, self.patch_size),
                    dtype=torch.float,
                    device=device,
                )
            )
            bboxes_yolox.append(
                torch.zeros(
                    (len(self.bboxes), 6),
                    dtype=torch.float,
                    device=device,
                )
            )

        sample["patches_yolox"] = torch.stack(patches_yolox).to(device)
        sample["bboxes_yolox"] = torch.stack(bboxes_yolox).to(device)

        return sample

    def add_to_sample(
        self,
        sample: dict,
        action_taken: Action,
        patch: torch.Tensor,
        infos: dict,
        index: int,
    ):
        if sample["patches"].shape[0] <= index:
            # Add space to the sample.
            for key in sample:
                if key in ["patches_yolox", "bboxes_yolox"]:
                    continue

                sample[key] = torch.cat(
                    [
                        sample[key],
                        torch.zeros(
                            (
                                sample[key].shape[0],
                                *sample[key].shape[1:],
                            ),
                            dtype=sample[key].dtype,
                            device=sample[key].device,
                        ),
                    ],
                    dim=0,
                )

        sample["patches"][index] = patch
        sample["current_actions"][index] = action_taken.value
        sample["next_actions"][index] = infos["best_action"].value
        sample["positions"][index][0] = infos["position"].y
        sample["positions"][index][1] = infos["position"].x
        sample["masks"][index] = 1.0
        sample["labels"][index] = int(infos["inside_bbox"])
        sample["local_bboxes"][index] = infos["local_bboxes"]

    def generate_sample(
        self,
        max_ep_len: int,
        min_keypoints: int,
        max_keypoints: int,
        binomial_keypoints: bool = False,
        position: Optional[Position] = None,
        visited_bbox_patches: Optional[Set[Position]] = None,
        device: str = "cpu",
    ) -> dict:
        """Generates a sample of an episode, taking into account the
        multiple bounding boxes.
        The episodes starts at the given position, and then visits all
        bounding boxes in a greedy manner (from closest neighbour to
        closest neighbour). It also insert random keypoints in the trajectory.
        The episode ends when the agent visits the center of the last bbox visited,
        or when the maximum episode length is reached.

        ---
        Args:
            max_ep_len: The maximum length of the episode.
            min_keypoints: The minimum number of random keypoints to visit.
            max_keypoints: The maximum number of random keypoints to visit.
            position: The starting position of the agent.
                Optional, if not provided, a random position is chosen.
            device: The device to use for the sample.

        ---
        Returns:
            A dict containing the sample:
                * patches: The patches visited during the episode.
                    Shape of [max_ep_len, n_channels, patch_size, patch_size].
                * current_actions: The actions taken during the episode.
                    Shape of [max_ep_len,].
                * next_actions: The best actions to take at each step.
                    Shape of [max_ep_len,].
                * positions: The positions of the agent at each step.
                    Shape of [max_ep_len, 2].
                * labels: The labels to use for the episode.
                    Shape of [max_ep_len,].
                * masks: The masks that defines the end of the episode.
                    '1' for unmasked steps, '0' for masked steps.
                    Shape of [max_ep_len,].
        """
        sample = self.init_sample(max_ep_len, device)
        patch, infos = self.reset(position, visited_bbox_patches)
        infos["best_action"] = Action.LEFT  # Action.START
        self.add_to_sample(
            sample,
            action_taken=Action.LEFT,
            patch=patch.to(device),
            infos=infos,
            index=0,
        )

        keypoints = self.build_keypoints_trajectory()

        # Sample the number of random keypoints that will be added to the trajectory.
        n_keypoints = self.rng.integers(min_keypoints, max_keypoints + 1)
        # Sample where the random keypoints will be added.
        insert_bbox_visit_at = self.rng.integers(0, len(keypoints), size=n_keypoints)
        # Add the random keypoints to the trajectory.
        insert_bbox_visit_at = list(sorted(insert_bbox_visit_at, reverse=True))

        for keypoint_id, keypoint in enumerate(keypoints):
            # Replace the last action (that should be random) by the best action
            # to reach the next keypoint.
            previous_best_action = move_towards(self.position, keypoint)
            sample_size = sample["masks"].long().sum().item() - 1
            sample["next_actions"][sample_size] = self.remove_stop_action(
                previous_best_action
            ).value

            # Add random keypoints if necessary.
            # We make sure to keep the original keypoint as
            # the true target to visit.
            while keypoint_id in insert_bbox_visit_at:
                if binomial_keypoints:
                    random_keypoint = self.generate_binomial_keypoints(1, keypoint)[0]
                else:
                    random_keypoint = self.generate_keypoints(1)[0]

                self.visit_point(sample, random_keypoint, keypoint, device)
                insert_bbox_visit_at.remove(keypoint_id)

            # Visit the keypoint.
            self.visit_point(sample, keypoint, keypoint, device)

        # Replace the last best action by the STOP action.
        # sample_size = sample["masks"].long().sum().item() - 1
        # sample["next_actions"][sample_size] = Action.STOP.value

        ep_len = sample["masks"].long().sum().item()
        if ep_len > max_ep_len:
            print(
                f"Warning: episode length ({ep_len}) is greater than max_ep_len ({max_ep_len})."
            )
            print("The episode is truncated.")

            # We only take the last steps of the episodes, to make sure
            # the maximum of keypoints is visited.
            for key in sample:
                if key not in ["patches_yolox", "bboxes_yolox"]:
                    sample[key] = sample[key][ep_len - max_ep_len : ep_len]

        assert sample["patches"].shape[0] == max_ep_len

        return sample

    def build_keypoints_trajectory(self) -> List[Position]:
        """Return the keypoints to visit by the agent in order.
        The order is generated in a greedy manner.

        If no keypoints are found, the agent will visit only a random point.
        """
        keypoints = []
        positions_to_visit = set()
        for bbox in self.raw_bboxes:
            positions_to_visit |= self.bbox_positions(bbox)

        for pos in self.visited_bbox_patches:
            positions_to_visit.remove(pos)

        current_pos = self.position
        while positions_to_visit:
            closest_pos, min_dist = [], float("+inf")
            for pos in positions_to_visit:
                dist = abs(pos.x - current_pos.x) + abs(pos.y - current_pos.y)
                if dist < min_dist:
                    min_dist = dist
                    closest_pos = []

                if dist == min_dist:
                    closest_pos.append(pos)

            closest_pos = random.choice(closest_pos)

            keypoints.append(closest_pos)
            positions_to_visit.remove(closest_pos)
            current_pos = closest_pos

        if not keypoints:
            keypoints.append(self.generate_keypoints(1)[0])
            if len(self.visited_bbox_patches) == 0:
                print(
                    "Warning: no keypoints found, either the image is empty or the bbox is outside the bound of the image."
                )

        return keypoints

    def visit_point(
        self, sample: dict, to_visit: Position, true_target: Position, device: str
    ):
        """Visit the point and add the corresponding trajectory to the sample.

        ---
        Args:
            sample: The sample to add the trajectory to.
            to_visit: The position to visit.
            true_target: The true target position, for which the best actions are defined.
            device: The device to use for the sample.
        """
        patch, infos = self.reset(self.position)
        sample_size = sample["masks"].long().sum().item()

        index = sample_size
        while self.position != to_visit:
            action = move_towards(self.position, to_visit)
            patch, infos = self.step(action)

            best_action = move_towards(self.position, true_target)
            # Can't stop here, we need to visit the point.
            infos["best_action"] = self.remove_stop_action(best_action)
            self.reset(self.position)

            self.add_to_sample(
                sample,
                action,
                patch.to(device),
                infos,
                index,
            )

            index += 1

    def generate_keypoints(self, n_keypoints: int) -> list:
        """Generates keypoints that the agent should visit.

        ---
        Args:
            n_keypoints: The number of keypoints to generate.

        ---
        Returns:
            A list of `n_keypoints` keypoints.
        """
        keypoints = []
        for _ in range(n_keypoints):
            y = self.rng.integers(0, self.patch_height)
            x = self.rng.integers(0, self.patch_width)
            keypoints.append(Position(y, x))
        return keypoints

    def generate_binomial_keypoints(
        self, n_keypoints: int, target_pos: Position
    ) -> list:
        """Generates keypoints that the agent should visit.
        The keypoints are sampled using a binomial distribution
        around the center of the bbox. It simulates better a searching behaviour
        around the bbox.

        The keypoints are sampled in the following way:
            1. Compute the maximum number of steps away from the center the agent can go.
            2. Simulate a random horizontal displacement using a distribution where
                each left/right actions have a 1/2 probability of being sampled.
                This leads to a binomial distribution of `Binom(n_steps, 1/2)`.
            3. Do the same for vertical displacement.
            4. Add the displacement to the center of the bbox, wrapping around the map
                if necessary.
            5. Repeat for all keypoints to be sampled.
        """
        keypoints = []
        for _ in range(n_keypoints):
            # Simulate a random horizontal displacement.
            x = self.rng.binomial(self.patch_width, 0.5) - self.patch_width // 2
            # Simulate a random vertical displacement.
            y = self.rng.binomial(self.patch_height, 0.5) - self.patch_height // 2
            # Add the displacement to the center of the bbox.
            y = (target_pos[0] + y) % self.patch_height
            x = (target_pos[1] + x) % self.patch_width
            keypoints.append(Position(y, x))

        return keypoints

    def remove_stop_action(self, action: Action) -> Action:
        if action == Action.STOP:
            return self.rng.choice(MOVES)
        return action

    @staticmethod
    def collate_fn(batch: List[dict]):
        """
        Add padding to the bboxes so that all tensors can be concatenated.
        """
        max_bboxes = max([sample["local_bboxes"].shape[1] for sample in batch])
        for sample in batch:
            local_bboxes = sample["local_bboxes"]
            max_ep_len, n_bboxes, bbox_shape = local_bboxes.shape
            n_diff_bboxes = max_bboxes - n_bboxes
            padding = torch.zeros(
                (max_ep_len, n_diff_bboxes, bbox_shape), dtype=torch.float32
            )
            sample["local_bboxes"] = torch.cat((local_bboxes, padding), dim=1)

        patches_yolox = []
        bboxes_yolox = []

        for sample in batch:
            patches_yolox.append(sample.pop("patches_yolox"))
            bboxes_yolox.append(sample.pop("bboxes_yolox"))

        for bbox_id, bboxes in enumerate(bboxes_yolox):
            n_diff_bboxes = max_bboxes - bboxes.shape[1]
            padding = torch.zeros(
                (bboxes.shape[0], n_diff_bboxes, bboxes.shape[2]), dtype=torch.float32
            )
            bboxes = torch.cat((bboxes, padding), dim=1)
            bboxes_yolox[bbox_id] = bboxes

        batch = default_collate(batch)
        batch["patches_yolox"] = torch.cat(patches_yolox)
        batch["bboxes_yolox"] = torch.cat(bboxes_yolox)

        # # Replace with the bboxes from the path.
        # batch["patches_yolox"] = einops.rearrange(
        #     batch["patches"], "b t c h w -> (b t) c h w"
        # )
        # batch["bboxes_yolox"] = einops.rearrange(
        #     batch["local_bboxes"], "b t n c -> (b t) n c"
        # )

        # save_patches(batch["patches_yolox"], batch["bboxes_yolox"])
        return batch
