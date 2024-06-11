from typing import Tuple, List
import einops
import gymnasium as gym
import torch
import torchvision.transforms.functional as TF

from torch import Tensor
from kornia.geometry.boxes import Boxes

from .common import Action, ACTION_DELTAS


class NeedleGeneralEnv(gym.Env):
    def __init__(
        self,
        images: Tensor,
        bboxes: Tensor,
        patch_size: int,
        max_ep_len: int,
        n_glimps_levels: int,
        stop_enabled: bool = False,
    ):
        """Creates a batched environment for the needle problem.

        ---
        Args:
            images: A batch of images, as uint8 RGB (saves memory).
                Shape of [batch_size, n_channels, height, width].
            bboxes: The bounding boxes of the batch.
                List of length `batch_size`, where each element is
                a tensor of shape [n_bboxes, 4].
            patch_size: The size of the patches.
            max_ep_len: The maximum number of steps in an episode.
            n_glimps_levels: The number of levels of glimpses for each patch.
        """
        assert images.shape[0] == bboxes.shape[0]
        assert len(images.shape) == 4
        assert n_glimps_levels > 0

        self.patch_size = patch_size
        self.max_ep_len = max_ep_len
        self.n_glimps_levels = n_glimps_levels
        self.stop_enabled = stop_enabled

        # Save the batch dimensions.
        self.batch_size, self.n_channels, self.height, self.width = images.shape

        # Make sure that the images are divisible by the patch size.
        assert self.height % self.patch_size == 0
        assert self.width % self.patch_size == 0

        # Number of patches in the images.
        self.n_vertical_patches = self.height // self.patch_size
        self.n_horizontal_patches = self.width // self.patch_size

        # The device is the same as the images.
        self.device = images.device

        # Observation and action spaces.
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(self.batch_size, self.n_channels, self.patch_size, self.patch_size),
        )
        # Vertical and horizontal movements.
        self.action_space = gym.spaces.Tuple(
            (
                gym.spaces.Discrete(self.n_vertical_patches),
                gym.spaces.Discrete(self.n_horizontal_patches),
            )
        )

        # Bounding boxes of the images.
        self.bbox_masks = self.convert_bboxes_to_masks(bboxes)
        self.bboxes = bboxes

        # Initialize the glimps patches.
        self.init_glimps_images(images)

        # Initialize some variables for the environment.
        self.init_env_variables()

    def init_glimps_images(self, images: torch.Tensor):
        """Build the glimps images from the original images.
        They represent progressive lower resolution images.

        The input token of the model is a slice of the glimps images.
        """
        glimps_images = []
        current_glimps = images

        # Progressively adds padding to the glimps and resizes it to the original size.
        for _ in range(self.n_glimps_levels):
            glimps_images.append(current_glimps)

            # Pad all sides of the images of `patch_size` pixels.
            current_glimps = TF.pad(
                current_glimps,
                padding=[self.patch_size] * 4,
                padding_mode="reflect",
                # padding_mode="constant",
                # fill=0,
            )
            # Resize the images to the original size.
            current_glimps = TF.resize(
                current_glimps,
                size=[self.height, self.width],
                antialias=True,
            )

        # Stack the images.
        # Final shape is [batch_size, n_glimps_levels, n_channels, height, width].
        self.images = torch.stack(glimps_images, dim=1)

    def init_env_variables(self):
        # Positions of the agents in the images.
        self.positions = torch.zeros(
            (self.batch_size, 2),
            dtype=torch.long,
            device=self.device,
        )

        # Visited patches by the agents.
        self.visited_patches = torch.zeros(
            (self.batch_size, self.n_vertical_patches, self.n_horizontal_patches),
            dtype=torch.bool,
            device=self.device,
        )

        # Number of steps taken by the agents.
        self.steps = torch.zeros(
            (self.batch_size,),
            dtype=torch.long,
            device=self.device,
        )
        self.has_stopped = torch.zeros(
            (self.batch_size,),
            dtype=torch.bool,
            device=self.device,
        )

    def reset(self) -> Tuple[Tensor, dict]:
        """Reset the environment variables.
        Randomly initialize the positions of the agents.

        ---
        Returns:
            patches: The patches where are the agents.
                Shape of [batch_size, n_channels, patch_size, patch_size].
            infos: Additional infos.
        """
        self.init_env_variables()
        self.positions[:, 0] = torch.randint(
            low=0, high=self.n_vertical_patches, size=(self.batch_size,)
        )
        self.positions[:, 1] = torch.randint(
            low=0, high=self.n_horizontal_patches, size=(self.batch_size,)
        )
        self.visited_patches = self.visited_patches | self.tiles_reached

        infos = {
            "positions": self.positions,
        }
        patches = self.patches / 255
        return patches, infos

    @torch.no_grad()
    def step(self, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, dict]:
        """Apply the actions, compute the rewards and some contextual information
        and return them.

        ---
        Args:
            actions: The actions to apply.
                Shape of [batch_size, 4].
        ---
        Returns:
            patches: The patches where are the agents.
                Shape of [batch_size, n_channels, patch_size, patch_size].
            rewards: The reward of the agents.
                Shape of [batch_size,].
            terminated: Whether the environments are terminated (won or stopped volunteerly).
                Shape of [batch_size,].
            truncated: Whether the environments are truncated (max steps reached).
                Shape of [batch_size,].
            infos: Additional infos.
        """
        # Apply the actions.
        self.apply_movements(actions)
        rewards = self.rewards
        self.visited_patches = self.visited_patches | self.tiles_reached
        self.steps += 1

        # Compute the rewards and terminaisons.
        truncated = self.steps >= self.max_ep_len

        infos = {
            "positions": self.positions,
        }

        patches = self.patches / 255

        return patches, rewards, self.terminated, truncated, infos

    def actions_to_movements(self, actions):
        movements = [ACTION_DELTAS[Action(int(act.item()))] for act in actions]
        movements = torch.tensor(movements, device=actions.device)
        return movements

    def apply_movements(self, actions: Tensor):
        """Apply the movements to the agents.
        Make sure that the agents don't move outside the images.

        ---
        Args:
            movements: The movement action to apply.
                Shape of [batch_size].
        """
        self.positions = self.positions + self.actions_to_movements(actions)
        # XXX Disabled roll over
        # self.positions[:, 0] = self.positions[:, 0] % self.n_vertical_patches
        # self.positions[:, 1] = self.positions[:, 1] % self.n_horizontal_patches
        self.positions[:, 0] = torch.clamp(
            self.positions[:, 0], 0, self.n_vertical_patches - 1
        )
        self.positions[:, 1] = torch.clamp(
            self.positions[:, 1], 0, self.n_horizontal_patches - 1
        )
        self.has_stopped |= actions == Action.STOP.value

    @property
    def terminated(self):
        if self.stop_enabled:
            return self.has_stopped
        else:
            return (
                torch.sum(
                    (self.bbox_masks & self.visited_patches) != self.bbox_masks,
                    dim=(1, 2),
                )
                == 0
            )

    @property
    def tiles_reached(self) -> Tensor:
        """Compute the boolean masks marking each agent's position
        in the images.

        ---
        Returns:
            The boolean masks.
                Shape of [batch_size, n_vertical_patches, n_horizontal_patches].
        """
        # Positions in the form [batch_id, tile_id], where `tile_id` is
        # a one-dimensional index of the flattened two-dimensional coordinates.
        one_dim_positions = (
            self.positions[:, 0] * self.n_horizontal_patches + self.positions[:, 1]
        )
        # Positions in the form [batch_id x tile_id,].
        # Those positions are absolute indices in the batch of patches.
        tiles_per_images = self.n_horizontal_patches * self.n_vertical_patches
        offsets = torch.arange(
            start=0,
            end=self.batch_size * tiles_per_images,
            step=tiles_per_images,
            device=self.device,
        )
        flattened_positions = one_dim_positions + offsets

        # Build the masks.
        masks = torch.zeros(
            (self.batch_size, self.n_vertical_patches, self.n_horizontal_patches),
            dtype=torch.bool,
            device=self.device,
        )
        # Use the absolute positions to index inside the masks.
        masks.flatten()[flattened_positions] = True

        return masks

    @property
    def patches(self) -> Tensor:
        """Fetch the patches of the images that the agents have reached.
        Return all patches of the trajectories.

        ---
        Returns:
            The batch of patches reached.
                Shape of [batch_size, n_channels, patch_size, patch_size].
        """
        # Compute the indices of the pixels of the first patch.
        row_indices = torch.arange(
            start=0,
            end=self.patch_size,
            device=self.device,
        )
        offsets = torch.arange(
            start=0,
            end=self.patch_size * self.width,
            step=self.width,
            device=self.device,
        )
        # This adds the offset to each row index, making it
        # a [patch_size, patch_size] tensor of indices.
        pixel_indices = row_indices.unsqueeze(0) + offsets.unsqueeze(1)

        # Add a starting offset to the indices depending on the
        # agent's position in the images.
        pixel_indices = einops.rearrange(pixel_indices, "h w -> (h w)")
        offsets = (
            self.positions[:, 0] * (self.width * self.patch_size)
            + self.positions[:, 1] * self.patch_size
        )
        # Add the offset of the first pixel index of each agent to the global
        # patch indices, making it a [batch_size, patch_size x patch_size] tensor.
        pixel_indices = pixel_indices.unsqueeze(0) + offsets.unsqueeze(1)

        # Add the channels dimension.
        pixel_indices = einops.repeat(
            pixel_indices, "b p -> b g c p", c=self.n_channels, g=self.n_glimps_levels
        )

        # Finally gather the pixels.
        images = einops.rearrange(self.images, "b g c h w -> b g c (h w)")
        patches = torch.gather(images, dim=3, index=pixel_indices)
        patches = einops.rearrange(
            patches, "b g c (h w) -> b g c h w", h=self.patch_size
        )
        return patches

    @property
    def prop_patches_found(self) -> Tensor:
        visited_bboxes = self.bbox_masks & self.visited_patches
        count = visited_bboxes.sum(dim=(1, 2))
        tot = self.bbox_masks.sum(dim=(1, 2))
        # avoid null
        tot[tot == 0] = 1
        return count / tot

    @property
    def prop_bboxes_found(self) -> Tensor:
        return (self.prop_patches_found > 0).to(torch.float32)

    @property
    def rewards(self) -> Tensor:
        """Compute the rewards of the agents.
        They win 1 point if on a patch with a box that they did not visited
        before.

        ---
        Returns:
            The rewards of the agents for the latest step.
                Shape of [batch_size,].
        """
        # Logical "OR" on the `n_bboxes` dimension.
        batch_ids = torch.arange(self.bbox_masks.shape[0])
        rewards = (
            self.bbox_masks[batch_ids, self.positions[:, 0], self.positions[:, 1]]
        ) & ~self.visited_patches[batch_ids, self.positions[:, 0], self.positions[:, 1]]
        # visited_patches does not include the latest visited patch here

        # Associate cost on negative patches
        cost_weight = -1 / self.max_ep_len
        costs = torch.ones_like(rewards) * cost_weight

        # STOP
        stop_eval = torch.zeros_like(rewards)

        if self.stop_enabled:
            bboxes_found = (self.visited_patches & self.bbox_masks).sum(dim=(1, 2))
            all_bboxes = self.bbox_masks.sum(dim=(1, 2))
            found_all_bboxes = (bboxes_found == all_bboxes).to(torch.int64)

            # sum of all patches with bbox if everything was found
            # negative sum of all missed bbox if the model missed something
            stop_eval = found_all_bboxes * bboxes_found + (1 - found_all_bboxes) * (
                bboxes_found - all_bboxes
            )
            stop_eval *= self.has_stopped

        return rewards + costs + stop_eval

    def convert_bboxes_to_masks(self, bboxes: Tensor) -> Tensor:
        """Convert the bounding boxes to masks.

        ---
        Args:
            bboxes: The bounding boxes of the batch (tensors)

        ---
        Returns:
            masks: The masks of the bounding boxes, to know which patch
                contains at least a bounding box.
                Shape of [batch_size, n_vertical_patches, n_horizontal_patches].
        """
        bboxes = Boxes.from_tensor(bboxes, "xyxy_plus")
        masks = bboxes.to_mask(self.height, self.width).long()
        masks = masks.max(dim=1).values  # Merge the channels.

        # Logical OR of the masks, to reduce to the patch dimensions.
        masks = torch.nn.functional.max_pool2d(masks.float(), self.patch_size)
        return masks.bool()

    def parse_bboxes(self, bboxes: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """Return the bounding boxes of the images as a tensor.
        Each bounding box of an image is given an id, which will serve as an
        index in the dimension `n_bboxes` of the tensors.

        This implementation is slow and not necessary.
        It keeps the original bboxes in the patches, which can be useful
        if you want to train a detector. Since the goal of the agent is only
        to find the patches where there are bounding boxes, it is not necessary.

        ---
        Args:
            bboxes: The bounding boxes of the batch.
                List of length `batch_size`, where each element is
                a tensor of shape [n_bboxes, 4].

        ---
        Returns:
            bboxes: The bounding boxes.
                Shape of [batch_size, n_vertical_patches, n_horizontal_patches, n_bboxes, 4].
            masks: The masks of the bounding boxes, to remove the padding
                at the `n_bboxes` dimension.
                Shape of [batch_size, n_vertical_patches, n_horizontal_patches, n_bboxes].
        """
        n_bboxes = max([bboxes_.shape[0] for bboxes_ in bboxes])
        tensor_bboxes = torch.zeros(
            (
                self.batch_size,
                self.n_vertical_patches,
                self.n_horizontal_patches,
                n_bboxes,
                4,
            ),
            dtype=torch.long,
            device=self.device,
        )
        masks = torch.zeros(
            (
                self.batch_size,
                self.n_vertical_patches,
                self.n_horizontal_patches,
                n_bboxes,
            ),
            dtype=torch.bool,
            device=self.device,
        )

        def place_bbox_recursive(
            bbox: Tensor,
            bbox_id: int,
            bboxes: Tensor,
            masks: Tensor,
            patch_size: int,
        ):
            """Place the bounding box in the tensor.
            If the bounding box is too big to fit in a patch, it will be split
            across the patches, by recursively calling this function.
            """
            # Compute the coordinates of the bounding box inside the patch.
            x1 = bbox[0] % patch_size
            y1 = bbox[1] % patch_size
            x2 = x1 + (bbox[2] - bbox[0])
            y2 = y1 + (bbox[3] - bbox[1])

            # Compute the coordinates of the patch.
            patch_x = bbox[0] // patch_size
            patch_y = bbox[1] // patch_size

            # Make sure the bounding box does not go outside the patch.
            x2_clampled = torch.clamp(x2, max=patch_size - 1)
            y2_clampled = torch.clamp(y2, max=patch_size - 1)

            # Save the bounding box.
            bboxes[patch_y, patch_x, bbox_id] = torch.LongTensor(
                [x1, y1, x2_clampled, y2_clampled]
            ).to(bboxes.device)
            masks[patch_y, patch_x, bbox_id] = True

            # Recursively place the bounding box in the other patches,
            # if the bounding box cross the borders of the current patch.
            if x2 - x2_clampled > 0:
                # The bounding box cross the right border of the patch.
                n_bbox = torch.LongTensor(
                    [
                        (patch_x + 1) * patch_size,
                        bbox[1],
                        bbox[2],
                        patch_y * patch_size + y2_clampled,
                    ]
                )
                place_bbox_recursive(n_bbox, bbox_id, bboxes, masks, patch_size)

            if y2 - y2_clampled > 0:
                # The bounding box cross the bottom border of the patch.
                n_bbox = torch.LongTensor(
                    [
                        bbox[0],
                        (patch_y + 1) * patch_size,
                        patch_x * patch_size + x2_clampled,
                        bbox[3],
                    ]
                )
                place_bbox_recursive(n_bbox, bbox_id, bboxes, masks, patch_size)

            if (x2 - x2_clampled > 0) and (y2 - y2_clampled > 0):
                # The bounding box cross the bottom-right corner of the patch.
                n_bbox = torch.LongTensor(
                    [
                        (patch_x + 1) * patch_size,
                        (patch_y + 1) * patch_size,
                        bbox[2],
                        bbox[3],
                    ]
                )
                place_bbox_recursive(n_bbox, bbox_id, bboxes, masks, patch_size)

        for batch_id, bboxes_ in enumerate(bboxes):
            for bbox_id, bbox in enumerate(bboxes_):
                place_bbox_recursive(
                    bbox,
                    bbox_id,
                    tensor_bboxes[batch_id],
                    masks[batch_id],
                    self.patch_size,
                )

        return tensor_bboxes, masks
