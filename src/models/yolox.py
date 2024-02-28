from typing import List, Optional

import einops
import torch
from torchvision.ops import box_convert

from yolox.models import YOLOX
from yolox.models.yolo_head import YOLOXHead
from yolox.models.yolo_pafpn import YOLOPAFPN
from yolox.utils import postprocess

from ..utils import BBox, Position

import yolox


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
                Shape of [batch_size, n_channels, patch_size, patch_size].
            targets: The true bboxes to be detected.
                If provided, shape of [batch_size, n_bboxes, class_id + 4 + 1].
                The last dimension is organized as follows:
                    - [class_id,]: The bbox's class.
                    - [xmin, ymin, xmax, ymax]: The coordinates of the bbox.
                    - [1,]: Whether the bbox is a true bbox or not (objectiveness).
        ---
        Returns:
            outputs: The predicted outputs. BBoxes are in xmin, ymin, xmax, ymax format
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
            fpn_outs = self.backbone(patches)

            # Training mode, to compute the loss.
            if targets is not None:
                targets[:, :, 1:5] = box_convert(targets[:, :, 1:5], "xyxy", "cxcywh")

                self.train()
                loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                    fpn_outs, targets, patches
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
            outputs = self.head(fpn_outs)
            outputs = postprocess(
                outputs,
                num_classes=1,
                class_agnostic=True,
                conf_thre=self.conf_threshold,
            )
            outputs = NeedleYOLOX.clamp_outputs(outputs, patches.shape[-1])

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

    # TODO put these functions in utils.
    @classmethod
    def from_base_yolox(
        cls, backbone: YOLOPAFPN, head: YOLOXHead, conf_threshold: float
    ):
        return cls(backbone, head, conf_threshold)
