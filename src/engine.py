from __future__ import annotations

import math
import os

import torch
from monai.inferers import sliding_window_inference

from .data import rebuild_dataloaders_for_patch_size


def _nanmean(values: list[float]) -> float:
    valid = [v for v in values if not math.isnan(v)]
    if len(valid) == 0:
        return 0.0
    return sum(valid) / len(valid)


def _nan_to_zero(x: float) -> float:
    return 0.0 if math.isnan(x) else x


def _to_binary_indices(
    labels: torch.Tensor,
    foreground_region_channels: list[int] | None = None,
) -> torch.Tensor:
    if foreground_region_channels is None:
        foreground_region_channels = [1, 2, 3, 4]

    if labels.ndim < 2:
        raise ValueError(f"Expected labels with shape [B, 1, ...] or [B, C, ...], got {tuple(labels.shape)}")

    if labels.shape[1] == 1:
        fg = torch.zeros_like(labels, dtype=torch.bool)
        for c in foreground_region_channels:
            fg |= (labels == c)
        return fg.long()

    fg = labels[:, foreground_region_channels].sum(dim=1, keepdim=True) > 0
    return fg.long()


def _pred5_to_pred2(
    outputs: torch.Tensor,
    background_channel: int = 0,
    foreground_region_channels: list[int] | None = None,
) -> torch.Tensor:
    if foreground_region_channels is None:
        foreground_region_channels = [1, 2, 3, 4]

    bg = outputs[:, background_channel : background_channel + 1]
    fg = outputs[:, foreground_region_channels].sum(dim=1, keepdim=True)
    return torch.cat([bg, fg], dim=1)


def _binary_dice_from_indices(pred_bin: torch.Tensor, tgt_bin: torch.Tensor) -> float:
    pred_bin = pred_bin.float()
    tgt_bin = tgt_bin.float()

    inter = (pred_bin * tgt_bin).sum()
    denom = pred_bin.sum() + tgt_bin.sum()

    if float(denom.item()) == 0.0:
        return 1.0

    return float((2.0 * inter / denom).item())


def _get_progressive_patch_size(
    epoch: int,
    curriculum_cfg: dict,
    default_patch_size: tuple[int, ...],
) -> tuple[int, ...]:
    if not bool(curriculum_cfg.get("enabled", False)):
        return default_patch_size

    if curriculum_cfg.get("mode", "") != "progressive_patch_size":
        return default_patch_size

    stages = curriculum_cfg.get("stages", [])
    if len(stages) == 0:
        return default_patch_size

    for stage in stages:
        end_epoch = int(stage["end_epoch"])
        stage_patch_size = tuple(int(x) for x in stage["patch_size"])
        if epoch <= end_epoch:
            return stage_patch_size

    return tuple(int(x) for x in stages[-1]["patch_size"])


def _get_progressive_stage_index(epoch: int, curriculum_cfg: dict) -> int:
    if not bool(curriculum_cfg.get("enabled", False)):
        return 0

    if curriculum_cfg.get("mode", "") != "progressive_patch_size":
        return 0

    stages = curriculum_cfg.get("stages", [])
    if len(stages) == 0:
        return 0

    for i, stage in enumerate(stages):
        if epoch <= int(stage["end_epoch"]):
            return i + 1

    return len(stages)


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_function,
    metric,
    post_pred,
    post_label,
    device,
    config,
    run_dir,
    models_dir,
    wandb_run=None,
    region_metric_names=None,
    mean_metric_name="dice_mean_score",
    best_mean_metric_name="best_dice_mean_score",
):
    max_epochs = int(config["training"]["max_epochs"])
    val_every = int(config["training"].get("val_every", 1))
    default_patch_size = tuple(config["transforms"]["patch_size"])
    sw_batch_size = int(config["training"].get("sw_batch_size", 1))
    infer_overlap = float(config["training"].get("infer_overlap", 0.5))

    curriculum_cfg = config["training"].get("curriculum", {})
    curriculum_enabled = bool(curriculum_cfg.get("enabled", False))
    curriculum_mode = curriculum_cfg.get("mode", "")

    pretrain_epochs = int(curriculum_cfg.get("pretrain_epochs", 0))
    background_channel = int(curriculum_cfg.get("background_channel", 0))
    foreground_region_channels = list(curriculum_cfg.get("foreground_region_channels", [1, 2, 3, 4]))
    binary_auxiliary_loss = bool(curriculum_cfg.get("binary_auxiliary_loss", False))
    binary_auxiliary_weight_max = float(curriculum_cfg.get("binary_auxiliary_weight_max", 0.1))
    binary_auxiliary_warmup_epochs = int(curriculum_cfg.get("binary_auxiliary_warmup_epochs", 50))

    use_binary_curriculum = curriculum_enabled and curriculum_mode == "binary_to_four_region_aorta"
    use_progressive_patch_curriculum = curriculum_enabled and curriculum_mode == "progressive_patch_size"

    best_metric = -1.0
    best_metric_epoch = -1

    current_train_patch_size = None
    current_progressive_stage_index = None

    for epoch in range(1, max_epochs + 1):
        stage_a = use_binary_curriculum and (epoch <= pretrain_epochs)

        patch_size = _get_progressive_patch_size(
            epoch=epoch,
            curriculum_cfg=curriculum_cfg,
            default_patch_size=default_patch_size,
        )
        progressive_stage_index = _get_progressive_stage_index(epoch, curriculum_cfg)

        if use_progressive_patch_curriculum:
            should_rebuild_loaders = (
                current_train_patch_size is None
                or tuple(current_train_patch_size) != tuple(patch_size)
            )

            if should_rebuild_loaders:
                print(f"Rebuilding dataloaders for patch size: {patch_size}")

                train_loader, val_loader, _, _, _, _ = rebuild_dataloaders_for_patch_size(
                    config=config,
                    patch_size=patch_size,
                )
                current_train_patch_size = tuple(patch_size)
                current_progressive_stage_index = progressive_stage_index

        model.train()
        epoch_train_loss = 0.0
        train_steps = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)

            if stage_a:
                train_targets = _to_binary_indices(
                    labels,
                    foreground_region_channels=foreground_region_channels,
                )
                train_outputs = _pred5_to_pred2(
                    outputs,
                    background_channel=background_channel,
                    foreground_region_channels=foreground_region_channels,
                )
                loss = loss_function(train_outputs, train_targets)
            else:
                loss = loss_function(outputs, labels)

                if use_binary_curriculum and binary_auxiliary_loss:
                    binary_targets = _to_binary_indices(
                        labels,
                        foreground_region_channels=foreground_region_channels,
                    )
                    binary_outputs = _pred5_to_pred2(
                        outputs,
                        background_channel=background_channel,
                        foreground_region_channels=foreground_region_channels,
                    )
                    aux_loss = loss_function(binary_outputs, binary_targets)

                    b_epoch = epoch - pretrain_epochs
                    if binary_auxiliary_warmup_epochs > 0:
                        aux_weight = min(
                            binary_auxiliary_weight_max,
                            binary_auxiliary_weight_max * (b_epoch / float(binary_auxiliary_warmup_epochs)),
                        )
                    else:
                        aux_weight = binary_auxiliary_weight_max

                    loss = loss + aux_weight * aux_loss

            loss.backward()
            optimizer.step()

            epoch_train_loss += float(loss.item())
            train_steps += 1

        epoch_train_loss /= max(train_steps, 1)

        print(f"Epoch {epoch}/{max_epochs}")
        print(f"train_loss: {epoch_train_loss:.6f}")

        epoch_log = {
            "epoch": epoch,
            "train_loss": epoch_train_loss,
        }

        if use_binary_curriculum:
            epoch_log["curriculum_stage_a_binary"] = int(stage_a)
            print(f"curriculum_stage: {'binary' if stage_a else 'multi'}")

        if use_progressive_patch_curriculum:
            epoch_log["curriculum_patch_stage"] = progressive_stage_index
            epoch_log["curriculum_patch_x"] = int(patch_size[0])
            epoch_log["curriculum_patch_y"] = int(patch_size[1])
            epoch_log["curriculum_patch_z"] = int(patch_size[2])
            print(f"curriculum_stage: patch_stage_{progressive_stage_index}")
            print(f"curriculum_patch_size: {patch_size}")

        if epoch % val_every == 0:
            model.eval()
            val_loss = 0.0
            val_steps = 0

            metric.reset()
            binary_val_scores = []

            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(device)
                    labels = batch["label"].to(device)

                    outputs = sliding_window_inference(
                        inputs=images,
                        roi_size=patch_size,
                        sw_batch_size=sw_batch_size,
                        predictor=model,
                        overlap=infer_overlap,
                    )

                    if stage_a:
                        val_targets = _to_binary_indices(
                            labels,
                            foreground_region_channels=foreground_region_channels,
                        )
                        val_outputs = _pred5_to_pred2(
                            outputs,
                            background_channel=background_channel,
                            foreground_region_channels=foreground_region_channels,
                        )
                        loss = loss_function(val_outputs, val_targets)
                        val_loss += float(loss.item())
                        val_steps += 1

                        pred_bin = torch.argmax(outputs, dim=1) > 0
                        tgt_bin = val_targets.squeeze(1) > 0
                        binary_val_scores.append(_binary_dice_from_indices(pred_bin, tgt_bin))
                    else:
                        loss = loss_function(outputs, labels)
                        val_loss += float(loss.item())
                        val_steps += 1

                        outputs_list = [post_pred(o) for o in outputs]
                        labels_list = [post_label(l) for l in labels]
                        metric(y_pred=outputs_list, y=labels_list)

            val_loss /= max(val_steps, 1)
            epoch_log["val_loss"] = val_loss
            print(f"val_loss: {val_loss:.6f}")

            if stage_a:
                dice_mean = _nanmean(binary_val_scores)
                epoch_log[mean_metric_name] = dice_mean
                epoch_log["dice_binary_aorta"] = dice_mean

                print(f"dice_binary_aorta: {dice_mean:.6f}")
                print(f"{mean_metric_name}: {dice_mean:.6f}")
            else:
                metric_values = metric.aggregate()
                metric.reset()

                if hasattr(metric_values, "detach"):
                    metric_values = metric_values.detach().cpu().float()

                if metric_values.ndim == 0:
                    class_scores = [float(metric_values.item())]
                else:
                    class_scores = [float(x) for x in metric_values.flatten()]

                dice_mean = _nanmean(class_scores)

                epoch_log[mean_metric_name] = dice_mean
                print(f"{mean_metric_name}: {dice_mean:.6f}")

                if region_metric_names is not None:
                    for label_idx, metric_name in region_metric_names.items():
                        score_index = label_idx - 1
                        raw_score = class_scores[score_index] if score_index < len(class_scores) else float("nan")
                        logged_score = _nan_to_zero(raw_score)
                        epoch_log[metric_name] = logged_score
                        print(f"{metric_name}: {logged_score:.6f}")

                if dice_mean > best_metric:
                    best_metric = dice_mean
                    best_metric_epoch = epoch

                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "best_metric": best_metric,
                            "config": config,
                        },
                        os.path.join(models_dir, "best_model.pt"),
                    )

            epoch_log[best_mean_metric_name] = best_metric if best_metric >= 0.0 else 0.0
            print(f"{best_mean_metric_name}: {epoch_log[best_mean_metric_name]:.6f}")
            print(f"best_metric_epoch: {best_metric_epoch}")

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                },
                os.path.join(models_dir, "last_model.pt"),
            )

        if wandb_run is not None:
            wandb_run.log(epoch_log)