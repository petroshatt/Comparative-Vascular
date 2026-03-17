from __future__ import annotations

import inspect

from monai.networks.nets import (
    AttentionUnet,
    SegResNet,
    SwinUNETR,
    UNETR,
    UNet,
)


def _build_swin_unetr(model_cfg: dict, transforms_cfg: dict):
    patch_size = tuple(transforms_cfg["patch_size"])

    common_kwargs = {
        "in_channels": int(model_cfg["in_channels"]),
        "out_channels": int(model_cfg["out_channels"]),
        "feature_size": int(model_cfg["feature_size"]),
        "use_checkpoint": bool(model_cfg.get("use_checkpoint", False)),
        "spatial_dims": int(model_cfg.get("spatial_dims", 3)),
    }

    sig = inspect.signature(SwinUNETR.__init__)

    if "img_size" in sig.parameters:
        return SwinUNETR(
            img_size=patch_size,
            **common_kwargs,
        )

    return SwinUNETR(**common_kwargs)


def _build_unetr(model_cfg: dict, transforms_cfg: dict):
    patch_size = tuple(transforms_cfg["patch_size"])

    return UNETR(
        in_channels=int(model_cfg["in_channels"]),
        out_channels=int(model_cfg["out_channels"]),
        img_size=patch_size,
        feature_size=int(model_cfg.get("feature_size", 16)),
        hidden_size=int(model_cfg.get("hidden_size", 768)),
        mlp_dim=int(model_cfg.get("mlp_dim", 3072)),
        num_heads=int(model_cfg.get("num_heads", 12)),
        proj_type=str(model_cfg.get("proj_type", "conv")),
        norm_name=str(model_cfg.get("norm_name", "instance")),
        res_block=bool(model_cfg.get("res_block", True)),
        dropout_rate=float(model_cfg.get("dropout_rate", 0.0)),
        spatial_dims=int(model_cfg.get("spatial_dims", 3)),
        qkv_bias=bool(model_cfg.get("qkv_bias", False)),
        save_attn=bool(model_cfg.get("save_attn", False)),
    )


def _build_attention_unet(model_cfg: dict):
    return AttentionUnet(
        spatial_dims=int(model_cfg.get("spatial_dims", 3)),
        in_channels=int(model_cfg["in_channels"]),
        out_channels=int(model_cfg["out_channels"]),
        channels=[int(x) for x in model_cfg["channels"]],
        strides=[int(x) for x in model_cfg["strides"]],
        dropout=float(model_cfg.get("dropout", 0.0)),
    )


def _build_unet(model_cfg: dict):
    return UNet(
        spatial_dims=int(model_cfg.get("spatial_dims", 3)),
        in_channels=int(model_cfg["in_channels"]),
        out_channels=int(model_cfg["out_channels"]),
        channels=[int(x) for x in model_cfg["channels"]],
        strides=[int(x) for x in model_cfg["strides"]],
        num_res_units=int(model_cfg.get("num_res_units", 2)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )


def _build_segresnet(model_cfg: dict):
    return SegResNet(
        spatial_dims=int(model_cfg.get("spatial_dims", 3)),
        init_filters=int(model_cfg.get("init_filters", 8)),
        in_channels=int(model_cfg["in_channels"]),
        out_channels=int(model_cfg["out_channels"]),
        dropout_prob=float(model_cfg.get("dropout_prob", 0.0)),
        act=model_cfg.get("act", ("RELU", {"inplace": True})),
        norm=model_cfg.get("norm", ("GROUP", {"num_groups": 8})),
        blocks_down=tuple(int(x) for x in model_cfg.get("blocks_down", (1, 2, 2, 4))),
        blocks_up=tuple(int(x) for x in model_cfg.get("blocks_up", (1, 1, 1))),
        upsample_mode=str(model_cfg.get("upsample_mode", "nontrainable")),
    )


def build_model(config: dict):
    model_cfg = config["model"]
    model_name = str(model_cfg.get("name", "")).lower()

    if model_name == "swin_unetr":
        return _build_swin_unetr(model_cfg, config["transforms"])

    if model_name == "unetr":
        return _build_unetr(model_cfg, config["transforms"])

    if model_name == "attention_unet":
        return _build_attention_unet(model_cfg)

    if model_name == "unet":
        return _build_unet(model_cfg)

    if model_name == "segresnet":
        return _build_segresnet(model_cfg)

    raise ValueError(
        f"Model '{model_name}' is not implemented yet. "
        f"Currently supported: "
        f"['swin_unetr', 'unetr', 'attention_unet', 'unet', 'segresnet']"
    )