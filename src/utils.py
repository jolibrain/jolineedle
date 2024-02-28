import os
from typing import NamedTuple, List, Optional

import matplotlib.pyplot as plt
from ast import literal_eval
import cv2 as cv
import numpy as np
import torch

Position = NamedTuple("Position", [("y", int), ("x", int)])
# Rename to top_left& bottom_right, or replace by kornia, or use tensors directly
BBox = NamedTuple("BBox", [("up_left", Position), ("bottom_right", Position)])


class CfgNode:
    """a lightweight configuration class inspired by yacs"""

    # TODO: convert to subclass from a dict like in yacs?
    # TODO: implement freezing to prevent shooting of own foot
    # TODO: additional existence/override checks when reading/writing params?

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return self._str_helper(0)

    def _str_helper(self, indent):
        """need to have a helper to support nested indentation for pretty printing"""
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                parts.append("%s:\n" % k)
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append("%s: %s\n" % (k, v))
        parts = [" " * (indent * 4) + p for p in parts]
        return "".join(parts)

    def to_dict(self):
        """return a dict representation of the config"""
        return {
            k: v.to_dict() if isinstance(v, CfgNode) else v
            for k, v in self.__dict__.items()
        }

    def merge_from_dict(self, d):
        self.__dict__.update(d)

    def merge_from_args(self, args):
        """
        update the configuration from a list of strings that is expected
        to come from the command line, i.e. sys.argv[1:].
        The arguments are expected to be in the form of `--arg=value`, and
        the arg can use . to denote nested sub-attributes. Example:
        --model.n_layer=10 --trainer.batch_size=32
        """
        for arg in args:
            keyval = arg.split("=")
            assert len(keyval) == 2, (
                "expecting each override arg to be of form --arg=value, got %s" % arg
            )
            key, val = keyval  # unpack

            # first translate val into a python object
            try:
                val = literal_eval(val)
                """
                need some explanation here.
                - if val is simply a string, literal_eval will throw a ValueError
                - if val represents a thing (like an 3, 3.14, [1,2,3], False, None, etc.) it will get created
                """
            except ValueError:
                pass

            # find the appropriate object to insert the attribute into
            assert key[:2] == "--"
            key = key[2:]  # strip the '--'
            keys = key.split(".")
            obj = self
            for k in keys[:-1]:
                obj = getattr(obj, k)
            leaf_key = keys[-1]

            # ensure that this attribute exists
            assert hasattr(
                obj, leaf_key
            ), f"{key} is not an attribute that exists in the config"

            # overwrite the attribute
            print("command line overwriting config attribute %s with %s" % (key, val))
            setattr(obj, leaf_key, val)


def bboxes_to_tensor(bboxes: List[BBox]):
    return torch.tensor(
        [
            [
                bbox.up_left.x,
                bbox.up_left.y,
                bbox.bottom_right.x,
                bbox.bottom_right.y,
            ]
            for bbox in bboxes
        ]
    )


def parse_bbox_predictions(
    outputs: List[Optional[torch.Tensor]],
    positions: Optional[torch.Tensor] = None,
    patch_size: int = 1,
) -> List[BBox]:
    """Parse the predicted bboxes of a single image, by producing a list of BBoxes.

    ---
    Args:
        outputs: List of predicted bboxes, one tensor for each positions.
            Each tensor is of shape [n_bboxes, 4 + num_classes + 1].
        positions: Tensor of positions. If no positions are provided, bbox coordinates are not modified.
            Shape of [n_patches, 2].
        patch_size: Size of a single patch.

    ---
    Returns:
        The global list of bboxes for the image.
    """
    parsed_bboxes = []
    for i, bboxes in enumerate(outputs):
        if bboxes is None:
            continue  # No bbox predicted for this position.
        bboxes = bboxes[:, :4]  # Get the bbox positions.

        if positions is not None:
            position = positions[i]
            for x_id in [0, 2]:
                bboxes[:, x_id] = bboxes[:, x_id] + position[1] * patch_size
            for y_id in [1, 3]:
                bboxes[:, y_id] = bboxes[:, y_id] + position[0] * patch_size

        for bbox in bboxes:
            bbox = bbox.cpu()
            parsed_bboxes.append(
                BBox(
                    up_left=Position(x=bbox[0].item(), y=bbox[1].item()),
                    bottom_right=Position(x=bbox[2].item(), y=bbox[3].item()),
                )
            )

    return parsed_bboxes


def parse_bbox_targets(
    targets: torch.Tensor, positions: Optional[torch.Tensor] = None, patch_size: int = 1
) -> List[BBox]:
    """

    ---
    Args:
        targets: The target sample.
            Shape of [n_patches, n_bboxes, class_id + 4 + 1].
        positions: Tensor of positions. If no positions are provided, bbox coordinates are not modified.
            Shape of [n_patches, 2].
        patch_size: Size of a single patch.
    """
    # Swap the class and the area predictions.
    targets2 = targets.clone()
    targets[..., :4], targets[..., 4] = targets2[..., 1:5], targets2[..., 0]
    # Remove non-existing bboxes, replacing them by `None`.
    filtered_targets = []
    for bboxes in targets:
        filtered_bboxes = []
        for bbox in bboxes:
            if bbox[-1] == 1:
                filtered_bboxes.append(bbox)
        filtered_bboxes = (
            torch.stack(filtered_bboxes) if len(filtered_bboxes) > 0 else None
        )
        filtered_targets.append(filtered_bboxes)
    # # Add the `n_bboxes` dimension.
    # targets = [t.unsqueeze(0) if t is not None else t for t in targets]
    return parse_bbox_predictions(targets, positions, patch_size)


def merge_boxes_batched(
    batch: List[Optional[torch.Tensor]], threshold: int = 2, target: bool = False
) -> List[torch.Tensor]:
    result = []
    for boxes in batch:
        if boxes is None:
            result.append(None)
        else:
            result.append(merge_boxes(boxes, threshold, target))

    return result


def merge_boxes(
    boxes: torch.Tensor, threshold: int = 2, target: bool = False
) -> torch.Tensor:
    """
    Merge contiguous bboxes obtained from patch prediction.
    If the boxes tensor includes score, the highest score of
    all merged boxes is kept

    target: whether box is in target form (cls, xmin, ymin, xmax, ymax)
    or in prediction form (xmin, ymin, xmax, ymax, obj_conf, cls_confs...)
    """
    off = 1 if target else 0

    def dist(a: torch.Tensor, b: torch.Tensor):
        d1 = abs(b[off + 2] - a[off + 0])
        d2 = abs(a[off + 2] - b[off + 0])
        d3 = abs(b[off + 3] - a[off + 1])
        d4 = abs(a[off + 3] - b[off + 1])
        return min(d1, d2, d3, d4)

    groups = []

    for i in range(len(boxes)):
        if boxes[i] is None:
            continue

        for groupid in range(len(groups)):
            if i in groups[groupid]:
                break
        else:
            groupid = len(groups)
            groups.append([i])

        for j in range(i + 1, len(boxes)):
            if boxes[j] is None:
                continue

            if dist(boxes[i], boxes[j]) <= threshold:
                groups[groupid].append(j)

    merged_boxes = []
    for group in groups:
        to_merge = [boxes[i] for i in group]
        xmin = [box[off + 0] for box in to_merge]
        ymin = [box[off + 1] for box in to_merge]
        xmax = [box[off + 2] for box in to_merge]
        ymax = [box[off + 3] for box in to_merge]
        merged_box = [min(xmin), min(ymin), max(xmax), max(ymax)]
        # confidence
        if target:
            merged_box = [0] + merged_box
        elif boxes.shape[1] > 5:
            conf = max([box[4] * box[5] for box in to_merge])
            merged_box += [conf, 1]

        merged_boxes.append(torch.tensor(merged_box, device=boxes.device))

    return torch.stack(merged_boxes)


# Plotting


def plot_image(axe: plt.Axes, image: np.ndarray, patch_size: int, cmap: str = "gray"):
    # Increase lightness of the image.
    image = image * 0.8 + 0.2
    axe.imshow(image, cmap=cmap, vmin=0, vmax=1)
    axe.set_xticks(np.arange(0, image.shape[1], patch_size))
    axe.set_yticks(np.arange(0, image.shape[0], patch_size))
    axe.grid(visible=True, color="white")


def plot_bbox(axe: plt.Axes, bbox: BBox, color: str = "green"):
    up_left = bbox.up_left
    bottom_right = bbox.bottom_right

    axe.plot(
        [up_left.x, up_left.x],
        [up_left.y, bottom_right.y],
        color=color,
        alpha=0.6,
    )
    axe.plot(
        [up_left.x, bottom_right.x],
        [bottom_right.y, bottom_right.y],
        color=color,
        alpha=0.6,
    )
    axe.plot(
        [bottom_right.x, bottom_right.x],
        [bottom_right.y, up_left.y],
        color=color,
        alpha=0.6,
    )
    axe.plot(
        [bottom_right.x, up_left.x],
        [up_left.y, up_left.y],
        color=color,
        alpha=0.6,
    )


def plot_patches(
    axe: plt.Axes, patches: list, positions: list, height: int, width: int
):
    patch_h, patch_w, n_channels = patches[0].shape
    image = np.zeros((height, width, n_channels))
    for patch, position in zip(patches, positions):
        image[position.y : position.y + patch_h, position.x : position.x + patch_w] = (
            patch
        )
    axe.imshow(image, vmin=0, vmax=1, alpha=0.3)


def plot_model_prediction(
    image: torch.Tensor,
    patches: torch.Tensor,
    positions: torch.Tensor,
    true_bboxes: Optional[List[BBox]] = None,
    predicted_bboxes: Optional[List[BBox]] = None,
):
    """Plot the model predictions onto the image.
    The patches that the model visited are plotted in a progressive red scale.
    If a label is positive, the patch is plotted in green.
    The main bbox of the environment is plotted in green.
    The other bboxes of the environment are plotted in blue.

    Args:
        image: Original image.
            Shape of [n_channels, height, width].
        patches: List of patches visited by the model.
            Shape of [n_patches, n_channels, patch_size, patch_size].
        positions: List of positions of the patches.
            Shape of [n_patches, (y, x)].
        main_bbox: The bounding box of the environment.
        bboxes: List of all bounding boxes of the image.

    Returns:
        image_prediction: Image with the patches in progressive red scale.
            Shape of [n_channels, height, width].
    """
    patch_size = patches.shape[-1]
    figure = plt.figure()
    axe = figure.gca()

    # To numpy.
    image = image.cpu().numpy()
    patches = patches.cpu().numpy()
    positions = positions.cpu().numpy()

    # To matplotlib shape.
    image = image.transpose(1, 2, 0)
    patches = patches.transpose(0, 2, 3, 1)

    # Parse positions.
    parsed_positions = [
        Position(p[0] * patch_size, p[1] * patch_size) for p in positions
    ]

    # To progressive gray scale markers.
    markers = np.ones_like(patches)
    min_range = 0.3
    for marker_id, marker in enumerate(markers):
        coeff = marker_id / len(markers)  # In range [0, 1].
        coeff = min_range + coeff * (1 - min_range)  # In range [min_range, 1].
        markers[marker_id] = marker * coeff

    # To red scale.
    markers[:, :, :, 1] = 0
    markers[:, :, :, 2] = 0

    plot_image(axe, image, patch_size, None)
    plot_patches(axe, markers, parsed_positions, image.shape[0], image.shape[1])

    if predicted_bboxes is not None:
        for bbox in predicted_bboxes:
            plot_bbox(axe, bbox, color="blue")

    if true_bboxes is not None:
        for bbox in true_bboxes:
            plot_bbox(axe, bbox, color="green")

    # Get figure data to numpy array.
    canvas = figure.canvas
    canvas.draw()
    data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    image = data.reshape(canvas.get_width_height()[::-1] + (3,))
    figure.clear()
    plt.close(figure)

    # To valid visdom image tensor.
    image = image / 255  # To [0, 1] range.
    image = torch.FloatTensor(image)
    image = image.permute(2, 0, 1)  # Right dimensions for visdom.

    return image


def save_batch(filename, batch):
    img_batch = (batch * 255).permute((0, 2, 3, 1)).cpu().numpy().astype(np.uint8)

    # create mozaic
    width_patch = 3
    height_patch = ((batch.shape[0] - 1) // width_patch) + 1
    patch_width = batch.shape[2]
    patch_height = batch.shape[3]

    fimg_width = 1 + width_patch * (patch_width + 1)
    fimg_height = 1 + height_patch * (patch_height + 1)
    final_img = np.zeros((fimg_width, fimg_height, 3), dtype=np.uint8)

    for yi in range(height_patch):
        for xi in range(width_patch):
            i = xi + yi * width_patch
            if i >= img_batch.shape[0]:
                break

            xmin = 1 + xi * (patch_width + 1)
            ymin = 1 + yi * (patch_height + 1)
            xmax = xmin + patch_width
            ymax = ymin + patch_height

            final_img[xmin:xmax, ymin:ymax, :] = img_batch[i]

    final_img = cv.cvtColor(final_img, cv.COLOR_RGB2BGR)
    cv.imwrite(filename, final_img)
