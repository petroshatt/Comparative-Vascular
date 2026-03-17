from __future__ import annotations

import numpy as np
from monai import transforms


class ModalityAwareScalingd:
    def __init__(self, keys, channel_ranges):
        self.keys = keys
        self.channel_ranges = channel_ranges

    def __call__(self, data):
        d = dict(data)
        image = d[self.keys]

        if image.shape[0] != len(self.channel_ranges):
            raise ValueError(
                f"Expected {len(self.channel_ranges)} channels for key '{self.keys}', "
                f"but got shape {tuple(image.shape)}."
            )

        scaled_channels = []
        for i, (a_min, a_max) in enumerate(self.channel_ranges):
            ch = image[i]
            ch = np.clip(ch, a_min, a_max)
            ch = (ch - a_min) / (a_max - a_min)
            scaled_channels.append(ch)

        d[self.keys] = np.stack(scaled_channels).astype(np.float32)
        return d

    def get_channel_info(self):
        info = {}
        for i, (a_min, a_max) in enumerate(self.channel_ranges):
            info[i] = {
                "min": a_min,
                "max": a_max,
            }
        return info


class CTandZScoreNormalizationd:
    """
    Channel-wise normalization for multi-channel medical images.

    Supported normalization types per channel:
    - "zscore"
    - "ct"
    """

    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = True

    def __init__(self, keys, channel_schemes, seg_key=None, target_dtype=np.float32):
        self.keys = keys
        self.channel_schemes = channel_schemes
        self.seg_key = seg_key
        self.target_dtype = target_dtype

    def _zscore_normalize(self, image, seg=None, use_mask_for_norm=False):
        eps = 1e-8 if self.target_dtype != np.float16 else 1e-4
        image = image.astype(self.target_dtype, copy=False)

        if use_mask_for_norm:
            assert seg is not None, (
                "use_mask_for_norm is True, but no segmentation was provided. "
                "The mask is computed as seg >= 0."
            )

        if seg is not None and use_mask_for_norm:
            mask = seg >= 0
            if np.any(mask):
                mean = image[mask].mean()
                std = image[mask].std()
                image[mask] = (image[mask] - mean) / max(std, eps)
        else:
            mean = image.mean()
            std = image.std()
            image -= mean
            image /= max(std, eps)

        return image

    def _ct_normalize(self, image, intensityproperties):
        if intensityproperties is None:
            raise ValueError("CT normalization requires 'intensityproperties'.")

        eps = 1e-8 if self.target_dtype != np.float16 else 1e-4
        mean_intensity = intensityproperties["mean"]
        std_intensity = intensityproperties["std"]
        lower_bound = intensityproperties["percentile_00_5"]
        upper_bound = intensityproperties["percentile_99_5"]

        image = image.astype(self.target_dtype, copy=False)
        np.clip(image, lower_bound, upper_bound, out=image)
        image -= mean_intensity
        image /= max(std_intensity, eps)

        return image

    def __call__(self, data):
        d = dict(data)
        image = d[self.keys]
        seg = d[self.seg_key] if self.seg_key is not None and self.seg_key in d else None

        if image.shape[0] != len(self.channel_schemes):
            raise ValueError(
                f"Expected {len(self.channel_schemes)} channels for key '{self.keys}', "
                f"but got shape {tuple(image.shape)}."
            )

        normalized_channels = []

        for i, scheme in enumerate(self.channel_schemes):
            ch = image[i]
            norm_type = scheme["type"].lower()

            if norm_type == "zscore":
                ch_seg = None
                if seg is not None:
                    if seg.ndim == image.ndim and seg.shape[0] == image.shape[0]:
                        ch_seg = seg[i]
                    else:
                        ch_seg = seg

                ch = self._zscore_normalize(
                    ch,
                    seg=ch_seg,
                    use_mask_for_norm=scheme.get("use_mask_for_norm", False),
                )

            elif norm_type == "ct":
                ch = self._ct_normalize(
                    ch,
                    intensityproperties=scheme.get("intensityproperties"),
                )

            else:
                raise ValueError(
                    f"Unsupported normalization type '{norm_type}' for channel {i}. "
                    f"Supported types are 'zscore' and 'ct'."
                )

            normalized_channels.append(ch)

        d[self.keys] = np.stack(normalized_channels).astype(self.target_dtype)
        return d

    def get_channel_info(self):
        info = {}
        for i, scheme in enumerate(self.channel_schemes):
            info[i] = dict(scheme)
        return info


def get_train_transforms(config: dict):
    patch_size = tuple(config["transforms"]["patch_size"])

    return transforms.Compose([
        transforms.LoadImaged(keys=["ct", "pet", "label"]),
        transforms.EnsureChannelFirstd(keys=["ct", "pet", "label"]),
        transforms.Orientationd(keys=["ct", "pet", "label"], axcodes="RAS", labels=None),

        transforms.ConcatItemsd(keys=["ct", "pet"], name="image", dim=0),

        transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
        transforms.SpatialPadd(keys=["image", "label"], spatial_size=patch_size),

        ModalityAwareScalingd(
            keys="image",
            channel_ranges=[(-1000.0, 1200.0), (0.0, 5.0)],
        ),

        transforms.RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=2,
            neg=1,
            num_samples=1,
        ),

        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),

        transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.2),
        transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.2),

        transforms.ToTensord(keys=["image", "label"]),
        transforms.DeleteItemsd(keys=["ct", "pet"]),
    ])


def get_val_transforms(config: dict):
    patch_size = tuple(config["transforms"]["patch_size"])

    return transforms.Compose([
        transforms.LoadImaged(keys=["ct", "pet", "label"]),
        transforms.EnsureChannelFirstd(keys=["ct", "pet", "label"]),
        transforms.Orientationd(keys=["ct", "pet", "label"], axcodes="RAS", labels=None),

        transforms.ConcatItemsd(keys=["ct", "pet"], name="image", dim=0),

        transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
        transforms.SpatialPadd(keys=["image", "label"], spatial_size=patch_size),

        ModalityAwareScalingd(
            keys="image",
            channel_ranges=[(-1000.0, 1200.0), (0.0, 5.0)],
        ),

        transforms.ToTensord(keys=["image", "label"]),
        transforms.DeleteItemsd(keys=["ct", "pet"]),
    ])


def get_train_transforms_nnunet_like(config: dict):
    """
    nnU-Net-like preprocessing and augmentation:
    - Load + channel first
    - Orientation to RAS
    - Concat modalities
    - Resample to target spacing
    - Foreground crop
    - Pad to patch size
    - Pos/neg crop
    - Flips + affine
    - nnU-Net-like channel-aware normalization
    - Intensity augmentations
    """
    patch_size = tuple(config["transforms"]["patch_size"])
    target_spacing = tuple(config["transforms"]["target_spacing"])

    return transforms.Compose([
        transforms.LoadImaged(keys=["ct", "pet", "label"]),
        transforms.EnsureChannelFirstd(keys=["ct", "pet", "label"]),
        transforms.Orientationd(keys=["ct", "pet", "label"], axcodes="RAS", labels=None),

        transforms.Spacingd(
            keys=["ct", "pet", "label"],
            pixdim=target_spacing,
            mode=("bilinear", "bilinear", "nearest"),
            align_corners=(True, True, None),
        ),

        transforms.ConcatItemsd(keys=["ct", "pet"], name="image", dim=0),

        transforms.CropForegroundd(
            keys=["image", "label"],
            source_key="image",
            allow_smaller=True,
        ),

        transforms.SpatialPadd(
            keys=["image", "label"],
            spatial_size=patch_size,
            method="symmetric",
        ),

        transforms.RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=1,
            neg=1,
            num_samples=1,
            image_key="image",
            image_threshold=0,
            allow_smaller=False,
        ),

        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),

        transforms.RandAffined(
            keys=["image", "label"],
            prob=0.2,
            rotate_range=(0.1, 0.1, 0.1),
            scale_range=(0.1, 0.1, 0.1),
            mode=("bilinear", "nearest"),
            padding_mode="border",
        ),

        CTandZScoreNormalizationd(
            keys="image",
            seg_key=None,
            channel_schemes=[
                {
                    "type": "ct",
                    "intensityproperties": {
                        "mean": config["transforms"]["ct_mean"],
                        "std": config["transforms"]["ct_std"],
                        "percentile_00_5": config["transforms"]["ct_percentile_00_5"],
                        "percentile_99_5": config["transforms"]["ct_percentile_99_5"],
                    },
                },
                {
                    "type": "zscore",
                    "use_mask_for_norm": False,
                },
            ],
            target_dtype=np.float32,
        ),

        transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.15),
        transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.15),
        transforms.RandGaussianNoised(keys="image", prob=0.1, mean=0.0, std=0.01),
        transforms.RandGaussianSmoothd(
            keys="image",
            prob=0.1,
            sigma_x=(0.5, 1.0),
            sigma_y=(0.5, 1.0),
            sigma_z=(0.5, 1.0),
        ),
        transforms.RandAdjustContrastd(keys="image", prob=0.15, gamma=(0.7, 1.5)),

        transforms.ToTensord(keys=["image", "label"]),
        transforms.DeleteItemsd(keys=["ct", "pet"]),
    ])


def get_val_transforms_nnunet_like(config: dict):
    """
    nnU-Net-like validation preprocessing:
    - Load + channel first
    - Orientation to RAS
    - Resample to target spacing
    - Concat modalities
    - Foreground crop
    - Pad to patch size
    - nnU-Net-like normalization
    """
    patch_size = tuple(config["transforms"]["patch_size"])
    target_spacing = tuple(config["transforms"]["target_spacing"])

    return transforms.Compose([
        transforms.LoadImaged(keys=["ct", "pet", "label"]),
        transforms.EnsureChannelFirstd(keys=["ct", "pet", "label"]),
        transforms.Orientationd(keys=["ct", "pet", "label"], axcodes="RAS", labels=None),

        transforms.Spacingd(
            keys=["ct", "pet", "label"],
            pixdim=target_spacing,
            mode=("bilinear", "bilinear", "nearest"),
            align_corners=(True, True, None),
        ),

        transforms.ConcatItemsd(keys=["ct", "pet"], name="image", dim=0),

        transforms.CropForegroundd(
            keys=["image", "label"],
            source_key="image",
            allow_smaller=True,
        ),

        transforms.SpatialPadd(
            keys=["image", "label"],
            spatial_size=patch_size,
            method="symmetric",
        ),

        CTandZScoreNormalizationd(
            keys="image",
            seg_key=None,
            channel_schemes=[
                {
                    "type": "ct",
                    "intensityproperties": {
                        "mean": config["transforms"]["ct_mean"],
                        "std": config["transforms"]["ct_std"],
                        "percentile_00_5": config["transforms"]["ct_percentile_00_5"],
                        "percentile_99_5": config["transforms"]["ct_percentile_99_5"],
                    },
                },
                {
                    "type": "zscore",
                    "use_mask_for_norm": False,
                },
            ],
            target_dtype=np.float32,
        ),

        transforms.ToTensord(keys=["image", "label"]),
        transforms.DeleteItemsd(keys=["ct", "pet"]),
    ])