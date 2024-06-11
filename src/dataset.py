import os
import random
import uuid
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, default_collate
from torchvision import transforms
from torchvision.transforms import functional as F

from .utils import bboxes_to_tensor, BBox, Position


class NeedleDataset(Dataset):
    def __init__(
        self,
        image_paths: list,
        bbox_paths: list,
        patch_size: int,
        max_ep_len: int,
        rotations: bool,
        translations: bool,
        min_keypoints: int,
        max_keypoints: int,
        binomial_keypoints: bool,
        minimum_image_size: int,
        filter_classes: Optional[set] = None,
    ):
        assert len(image_paths) == len(bbox_paths)
        self.image_paths = image_paths
        self.bbox_paths = bbox_paths
        self.patch_size = patch_size
        self.max_ep_len = max_ep_len
        self.rotations = rotations
        self.translations = translations
        self.min_keypoints = min_keypoints
        self.max_keypoints = max_keypoints
        self.binomial_keypoints = binomial_keypoints
        self.minimum_image_size = minimum_image_size
        self.filter_classes = filter_classes
        if self.filter_classes is not None:
            self.raw_classes_to_ordered_classes = {
                class_id: i for i, class_id in enumerate(sorted(filter_classes))
            }

        self.to_tensor = transforms.ToTensor()
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.image_paths)

    def load_image(self, index: int) -> np.ndarray:
        """
        Output image is of shape [height, width, n_channels].
        """
        image = Image.open(self.image_paths[index])
        image = image.convert("RGB")
        image = np.array(image)
        return image

    def get_all_bboxes(self, index: int) -> tuple:
        bbox_path = self.bbox_paths[index]
        bboxes = []
        classes = []
        with open(bbox_path, "r") as bbox_file:
            for bbox_line in bbox_file:
                points = bbox_line.strip().split(" ")
                points = [int(float(p)) for p in points]
                bbox = BBox(
                    Position(points[2], points[1]),
                    Position(points[4], points[3]),
                )
                classes.append(points[0])
                bboxes.append(bbox)

        if self.filter_classes is not None:
            bboxes = [
                bbox
                for bbox, class_id in zip(bboxes, classes)
                if class_id in self.filter_classes
            ]
            classes = [
                class_id for class_id in classes if class_id in self.filter_classes
            ]
            classes = [
                self.raw_classes_to_ordered_classes[class_id] for class_id in classes
            ]

        return classes, bboxes

    def rotate(
        self,
        image: torch.Tensor,
        bboxes: List[BBox],
        rng: Optional[np.random.Generator] = None,
    ) -> tuple:
        """Randomly rotate the image and bbox."""
        _, image_width, image_height = image.shape
        if rng is not None:
            angles = [0, 90, 180, 270]
            angle_id = rng.choice(np.arange(len(angles)), (1,))[0]
            angle = angles[angle_id]
        else:
            angle = random.choice([0, 90, 180, 270])

        if angle == 0:
            image_aug = image
            bboxes_aug = bboxes
        elif angle == 90:
            image_aug = torch.transpose(image, 1, 2)
            image_aug = torch.flip(image_aug, [2])
            bboxes_aug = [
                BBox(
                    Position(bbox.up_left.x, image_width - bbox.bottom_right.y),
                    Position(bbox.bottom_right.x, image_width - bbox.up_left.y),
                )
                for bbox in bboxes
            ]
        elif angle == 180:
            image_aug = torch.flip(image, [1, 2])
            bboxes_aug = [
                BBox(
                    Position(
                        image_width - bbox.bottom_right.y,
                        image_height - bbox.bottom_right.x,
                    ),
                    Position(
                        image_width - bbox.up_left.y,
                        image_height - bbox.up_left.x,
                    ),
                )
                for bbox in bboxes
            ]
        else:  # angle = 270
            image_aug = torch.transpose(image, 1, 2)
            image_aug = torch.flip(image_aug, [1])
            bboxes_aug = [
                BBox(
                    Position(
                        image_height - bbox.bottom_right.x,
                        bbox.up_left.y,
                    ),
                    Position(
                        image_height - bbox.up_left.x,
                        bbox.bottom_right.y,
                    ),
                )
                for bbox in bboxes
            ]

        return image_aug, bboxes_aug

    def translate(
        self,
        image: torch.Tensor,
        bboxes: List[BBox],
        rng: Optional[np.random.Generator] = None,
    ) -> tuple:
        """Randomly translate the image and bbox."""
        _, img_height, img_width = image.shape
        min_bbox_x = min([bbox.up_left.x for bbox in bboxes])
        min_bbox_y = min([bbox.up_left.y for bbox in bboxes])
        min_bbox_x = max(min_bbox_x, 0)
        min_bbox_y = max(min_bbox_y, 0)
        up_left_margin = Position(
            x=min(img_width // 3, min_bbox_x),
            y=min(img_height // 3, min_bbox_y),
        )
        max_bbox_x = max([bbox.bottom_right.x for bbox in bboxes])
        max_bbox_y = max([bbox.bottom_right.y for bbox in bboxes])
        max_bbox_x = min(max_bbox_x, img_width)
        max_bbox_y = min(max_bbox_y, img_height)
        bottom_right_margin = Position(
            x=min(img_width // 3, img_width - max_bbox_x),
            y=min(img_height // 3, img_height - max_bbox_y),
        )

        if rng is not None:
            if up_left_margin.x == 0 and bottom_right_margin.x == 0:
                translate_x = 0
            else:
                translate_x = rng.integers(
                    -up_left_margin.x, bottom_right_margin.x, (1,)
                )[0]
            if up_left_margin.y == 0 and bottom_right_margin.y == 0:
                translate_y = 0
            else:
                translate_y = rng.integers(
                    -up_left_margin.y, bottom_right_margin.y, (1,)
                )[0]
        else:
            if up_left_margin.x == 0 and bottom_right_margin.y == 0:
                translate_x = 0
            else:
                translate_x = random.randint(-up_left_margin.x, bottom_right_margin.x)
            if up_left_margin.y == 0 and bottom_right_margin.y == 0:
                translate_y = 0
            else:
                translate_y = random.randint(-up_left_margin.y, bottom_right_margin.y)

        image = F.affine(
            image,
            angle=0,
            translate=[translate_x, translate_y],
            scale=1.0,
            shear=0.0,
            fill=0.0,
        )
        bboxes = [
            BBox(
                up_left=Position(
                    x=bbox.up_left.x + translate_x,
                    y=bbox.up_left.y + translate_y,
                ),
                bottom_right=Position(
                    x=bbox.bottom_right.x + translate_x,
                    y=bbox.bottom_right.y + translate_y,
                ),
            )
            for bbox in bboxes
        ]
        return image, bboxes

    @torch.no_grad()
    def transform(
        self,
        image: np.ndarray,
        bboxes: List[BBox],
        rng: Optional[np.random.Generator] = None,
    ) -> tuple:
        """
        Output image is of shape [n_channels, height, width].
        It is normalized in range [0, 1].
        """
        # Channels in the right order and in range [0, 1].
        image = self.to_tensor(image)
        _, height, width = image.shape

        if self.minimum_image_size != 0 and (
            width < self.minimum_image_size or height < self.minimum_image_size
        ):
            ratio = width / height
            original_width = width
            original_height = height
            if width < self.minimum_image_size:
                width = self.minimum_image_size
                height = int(width / ratio)
            if height < self.minimum_image_size:
                height = self.minimum_image_size
                width = int(height * ratio)

            image = F.resize(image, size=[height, width])
            for bbox_id, bbox in enumerate(bboxes):
                up_left = Position(
                    x=bbox.up_left.x * width / original_width,
                    y=bbox.up_left.y * height / original_height,
                )
                bottom_right = Position(
                    x=bbox.bottom_right.x * width / original_width,
                    y=bbox.bottom_right.y * height / original_height,
                )
                bbox = BBox(
                    up_left=up_left,
                    bottom_right=bottom_right,
                )
                bboxes[bbox_id] = bbox

        image = complete_to_patch_size(image, self.patch_size)

        if self.rotations:
            image, bboxes = self.rotate(image, bboxes, rng)

        if self.translations:
            image, bboxes = self.translate(image, bboxes, rng)

        return image, bboxes

    def __getitem__(
        self, index: int, rng: Optional[np.random.Generator] = None
    ) -> dict:
        if rng is None:
            rng = self.rng
        image = self.load_image(index)
        classes, bboxes = self.get_all_bboxes(index)
        class_id = rng.choice(classes)
        bboxes = [bbox for bbox, c in zip(bboxes, classes) if c == class_id]
        image, bboxes = self.transform(image, bboxes, rng)
        return {
            "image": image,
            "bboxes": bboxes,
            "class_id": class_id,
        }

    @staticmethod
    def list_collate_fn(batch: List) -> Dict:
        """
        Don't stack images, just return them as a list.
        """
        keys = batch[0].keys()
        batch = {key: [sample[key] for sample in batch] for key in keys}
        return batch

    @staticmethod
    def padded_collate_fn(batch: List[Dict], patch_size: int) -> Dict:
        """Collate the batch of images and bboxes.

        The images can be of varying sizes. They are padded with zeros to match
        the biggest image. They are also padded to a multiple of `patch_size`.
        """
        batch = NeedleDataset.list_collate_fn(batch)

        # Find the max height and width.
        max_height = max(img.shape[1] for img in batch["image"])
        max_width = max(img.shape[2] for img in batch["image"])
        max_bbox = max(len(bbox) for bbox in batch["bboxes"])

        # Add padding to match a multiple of patch_size.
        delta_h = patch_size - max_height % patch_size
        delta_w = patch_size - max_width % patch_size
        delta_h = delta_h if delta_h != patch_size else 0
        delta_w = delta_w if delta_w != patch_size else 0
        final_height = max_height + delta_h
        final_width = max_width + delta_w

        for i, image in enumerate(batch["image"]):
            height, width = image.shape[1:]
            pad_h = final_height - height
            pad_w = final_width - width
            # pad images
            batch["image"][i] = torch.nn.functional.pad(
                image, (0, pad_w, 0, pad_h), mode="constant", value=0
            )
            # pad bbox
            bbox = bboxes_to_tensor(batch["bboxes"][i])
            batch["bboxes"][i] = torch.nn.functional.pad(
                bbox, (0, 0, 0, max_bbox - bbox.shape[0]), mode="constant", value=0
            )

        return {
            "image": torch.stack(batch["image"], dim=0),
            "bboxes": torch.stack(batch["bboxes"], dim=0),
            "class_id": torch.tensor(batch["class_id"]),
        }


def save_patches(patches: torch.Tensor, bboxes: torch.Tensor):
    """Save the patches and bboxes to disk.

    ---
    Args:
        patches: Tensor of patches of shape [n_patches, n_channels, height, width].
        bboxes: Tensor of bboxes of shape [n_patches, n_bboxes, 6].
    """
    # Use torchvision to draw each bbox on the patch.
    # Then save the patch to disk.
    # The bboxes are in the format [class_id, cx, cy, w, h, objectivness].
    for patch, bbox in zip(patches, bboxes):
        patch = patch.permute(1, 2, 0).cpu().numpy()
        patch = (patch * 255).astype(np.uint8)
        patch = Image.fromarray(patch)

        for bbox in bbox.cpu().numpy():
            _, cx, cy, w, h, _ = bbox
            # cx = int(cx * patch.width)
            # cy = int(cy * patch.height)
            # w = int(w * patch.width)
            # h = int(h * patch.height)
            bbox = (cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2)
            draw = ImageDraw.Draw(patch)
            draw.rectangle(bbox, outline="red")

        patch.save(f"patches/{uuid.uuid4()}.png")


def complete_to_patch_size(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Add padding to the end of the image so that its shape is exactly
    a multiple of the `patch_size`.
    """
    n_channels, height, width = image.shape
    delta_h = patch_size - height % patch_size
    delta_w = patch_size - width % patch_size

    if delta_h == patch_size:
        delta_h = 0
    if delta_w == patch_size:
        delta_w = 0

    image = torch.concat(
        (
            image,
            torch.zeros((n_channels, delta_h, width), dtype=torch.float),
        ),
        dim=1,
    )
    image = torch.concat(
        (
            image,
            torch.zeros((n_channels, height + delta_h, delta_w), dtype=torch.float),
        ),
        dim=2,
    )
    return image


def get_paths(
    dataset_directory: Path, test_pattern="", seed: int = 0, test_size: float = 0.01
) -> tuple:
    def get_images_and_bboxes(file: Path) -> list:
        images, bboxes = [], []
        dir_path = file.parent
        with open(file, "r") as path_file:
            for line in path_file:
                i, b = line.split(" ")
                if not Path(i).is_absolute():
                    i = str(dir_path / i)
                if not Path(b).is_absolute():
                    b = str(dir_path / b)
                images.append(i.strip())
                bboxes.append(b.strip())
        return images, bboxes

    def train_test_paths(dataset_directory: Path):
        images_train, bbox_train = get_images_and_bboxes(
            dataset_directory / "train.txt"
        )
        images_test, bbox_test = get_images_and_bboxes(dataset_directory / "test.txt")
        return images_train, bbox_train, images_test, bbox_test

    def single_paths(
        dataset_directory: Path, test_pattern: str, seed: int, test_size: float
    ):
        images_train, bbox_train = [], []
        images_test, bbox_test = [], []
        filename = (
            "all.txt" if os.path.isfile(dataset_directory / "all.txt") else "paths.txt"
        )
        with open(dataset_directory / filename, "r") as path_file:
            for line in path_file:
                i, b = line.split(" ")
                i = dataset_directory / i.strip()
                b = dataset_directory / b.strip()

                if test_pattern != "" and test_pattern in str(i):
                    images_test.append(i)
                    bbox_test.append(b)
                else:
                    images_train.append(i)
                    bbox_train.append(b)

        if test_pattern == "":
            images_train, images_test, bbox_train, bbox_test = train_test_split(
                images_train, bbox_train, test_size=test_size, random_state=seed
            )
        return images_train, bbox_train, images_test, bbox_test

    # TODO there should be an option to determine if the dataset is splitted or not
    if os.path.isfile(dataset_directory / "train.txt") and os.path.isfile(
        dataset_directory / "test.txt"
    ):
        print("Using native train/test split.")
        return train_test_paths(dataset_directory)

    if os.path.isfile(dataset_directory / "paths.txt") or os.path.isfile(
        dataset_directory / "all.txt"
    ):
        print("Single file containing all paths.")
        return single_paths(dataset_directory, test_pattern, seed, test_size)

    raise RuntimeError(
        "Loading a new dataset, please specify the way it should be loaded."
    )


def filter_images(classes_to_keep: set, image_paths: list, bbox_paths: list) -> tuple:
    filtered_images, filtered_bboxes = [], []
    for image_path, bbox_path in zip(image_paths, bbox_paths):
        with open(bbox_path, "r") as bbox_file:
            classes = set([int(line.strip().split(" ")[0]) for line in bbox_file])
            if classes & classes_to_keep:
                filtered_images.append(image_path)
                filtered_bboxes.append(bbox_path)

    return filtered_images, filtered_bboxes


def build_datasets(
    dataset_directory: Path,
    patch_size: int,
    max_ep_len: int,
    min_keypoints: int,
    max_keypoints: int,
    rotations: bool,
    translations: bool,
    seed: int = 0,
    train_size: int = -1,
    test_size: float = 0.01,
    test_pattern: str = "",
    binomial_keypoints: bool = False,
    minimum_image_size: int = 0,
    filter_classes: Optional[set] = None,
) -> tuple:
    train_images, train_bbox, test_images, test_bbox = get_paths(
        dataset_directory,
        test_pattern,
        test_size=test_size,
        seed=seed,
    )
    if filter_classes is not None:
        train_images, train_bbox = filter_images(
            filter_classes, train_images, train_bbox
        )
        test_images, test_bbox = filter_images(filter_classes, test_images, test_bbox)

    train_size = len(train_images) if train_size == -1 else train_size
    train_size = min(len(train_images), train_size)

    print(f"Train size: {train_size}")
    print("Test size: {}".format(len(test_images)))

    train_dataset = NeedleDataset(
        train_images[:train_size],
        train_bbox[:train_size],
        patch_size,
        max_ep_len,
        rotations,
        translations,
        min_keypoints,
        max_keypoints,
        binomial_keypoints=binomial_keypoints,
        minimum_image_size=minimum_image_size,
        filter_classes=filter_classes,
    )
    test_dataset = NeedleDataset(
        test_images,
        test_bbox,
        patch_size,
        max_ep_len,
        rotations,
        translations,
        min_keypoints,
        max_keypoints,
        binomial_keypoints=binomial_keypoints,
        minimum_image_size=minimum_image_size,
        filter_classes=filter_classes,
    )
    return train_dataset, test_dataset
