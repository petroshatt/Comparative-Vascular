from __future__ import annotations

from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete


def build_loss(config: dict):
    return DiceCELoss(
        to_onehot_y=True,
        softmax=True,
    )


def build_metric(config: dict):
    return DiceMetric(
        include_background=False,
        reduction="mean_batch",
        get_not_nans=False,
        ignore_empty=True,
    )


def build_post_transforms(config: dict):
    num_classes = int(config["model"]["out_channels"])
    post_label = AsDiscrete(to_onehot=num_classes)
    post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
    return post_label, post_pred