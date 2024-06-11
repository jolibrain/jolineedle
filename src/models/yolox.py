from typing import List, Optional

import einops
import torch
from torchvision.ops import box_convert

from yolox.models import YOLOX
from yolox.models.yolo_head import YOLOXHead
from yolox.models.yolo_pafpn import YOLOPAFPN
from yolox.utils import postprocess

from ..utils import BBox, Position


class NeedleYOLOX(YOLOX):
    def __init__(self, backbone: YOLOPAFPN, head: YOLOXHead, conf_threshold: float):
        super().__init__(backbone, head)
        # self.head.decode_in_inference = True
        self.conf_threshold = conf_threshold
        self.head.use_l1 = True

    def forward(self, patches: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """Either predicts the bboxes or computes the loss.

        ---
        Args:
            patches: Consecutive patches of the image.
                Shape of [batch_size, n_tokens, n_channels, patch_size, patch_size].
            targets: The true bboxes to be detected.
                If provided, shape of [batch_size, n_tokens, n_bboxes, class_id + 4 + 1].
                The last dimension is organized as follows:
                    - [class_id,]: The bbox's class.
                    - [cx, cy, w, h]: The coordinates of the bbox.
                    - [1,]: Whether the bbox is a true bbox or not (objectiveness).

        ---
        Returns:
            outputs: The predicted outputs.
            fpn_outs: The pyramidal embeddings outputted by the backbone.
                Tuple of shapes [batch_size, n_channels_i, width_i, height_i].
            losses: Dictionary of losses.
        """
        mode = self.training
        device = str(patches.device)

        # We set the default device before doing the inferences,
        # because somewhere in the model's head, new tensors are allocated
        # using the default device (which is 'cuda:0' by default).
        # Moreover, the argument needs to be a cuda device, so if the current
        # device is not a cuda device, we set it to 'cuda:0'.
        device = torch.device(device if "cuda" in device else "cuda:0")
        with torch.cuda.device(device):
            # Backbone inference, with the current mode of the model.
            x = einops.rearrange(patches, "b t c w h -> (b t) c w h")
            fpn_outs = self.backbone(x)

            # Training mode, to compute the loss.
            if targets is not None:
                self.train()
                targets = einops.rearrange(targets, "b t a n -> (b t) a n")
                loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                    fpn_outs, targets, x
                )
                losses = {
                    "total_loss": loss,
                    "iou_loss": iou_loss,
                    "l1_loss": l1_loss,
                    "conf_loss": conf_loss,
                    "cls_loss": cls_loss,
                    "num_fg": num_fg,
                }
            else:
                losses = {}

            # Evaluation mode, to predict the bboxes.
            self.eval()
            n_tokens = patches.shape[1]
            outputs = self.head(fpn_outs)
            outputs = postprocess(
                outputs,
                num_classes=1,
                class_agnostic=True,
                conf_thre=self.conf_threshold,
            )
            outputs = [
                outputs[i : i + n_tokens] for i in range(0, len(outputs), n_tokens)
            ]
            outputs = [NeedleYOLOX.clamp_outputs(o, patches.shape[-1]) for o in outputs]

        # Restore the model's mode.
        self.train(mode)

        return outputs, fpn_outs, losses

    @staticmethod
    def clamp_outputs(
        outputs: List[Optional[torch.Tensor]], image_size: int
    ) -> List[Optional[torch.Tensor]]:
        """Clamp the predicted bboxes to the image's size.

        ---
        Args:
            outputs: List of predicted bboxes, one tensor for each positions.
                Each tensor is of shape [n_bboxes, 4 + num_classes + 1].
        ---
        Returns:
            The clamped list of bboxes.
        """
        for bboxes in outputs:
            if bboxes is None:
                continue

            bboxes[:, :4].clamp_(min=0, max=image_size - 1)

        return outputs

    @staticmethod
    def parse_bbox_predictions(
        outputs: List[Optional[torch.Tensor]],
        positions: torch.Tensor,
        patch_size: int,
    ) -> List[BBox]:
        """Parse the predicted bboxes of a single image, by producing a list of BBoxes.

        ---
        Args:
            outputs: List of predicted bboxes, one tensor for each positions.
                Each tensor is of shape [n_bboxes, 4 + num_classes + 1].
            positions: Tensor of positions.
                Shape of [n_patches, 2].
            patch_size: Size of a single patch.

        ---
        Returns:
            The global list of bboxes for the image.
        """
        parsed_bboxes = []
        for bboxes, position in zip(outputs, positions):
            if bboxes is None:
                continue  # No bbox predicted for this position.
            bboxes = bboxes[:, :4]  # Get the bbox positions.
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

    @staticmethod
    def parse_bbox_targets(
        targets: torch.Tensor, positions: torch.Tensor, patch_size: int
    ) -> List[BBox]:
        """

        ---
        Args:
            targets: The target sample.
                Shape of [n_patches, n_bboxes, class_id + 4 + 1].
            positions: Tensor of positions.
                Shape of [n_patches, 2].
            patch_size: Size of a single patch.
        """
        # Change the bbox area layout.
        targets[:, :, 1:5] = box_convert(targets[:, :, 1:5], "cxcywh", "xyxy")
        # Swap the class and the area predictions.
        targets[:, :, :4], targets[:, :, 4] = targets[:, :, 1:5], targets[:, :, 0]
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
        return NeedleYOLOX.parse_bbox_predictions(targets, positions, patch_size)

    @classmethod
    def from_base_yolox(
        cls, backbone: YOLOPAFPN, head: YOLOXHead, conf_threshold: float
    ):
        return cls(backbone, head, conf_threshold)
