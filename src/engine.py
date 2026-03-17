from __future__ import annotations

import math
import os

import torch
from monai.inferers import sliding_window_inference


def _nanmean(values: list[float]) -> float:
    valid = [v for v in values if not math.isnan(v)]
    if len(valid) == 0:
        return 0.0
    return sum(valid) / len(valid)


def _nan_to_zero(x: float) -> float:
    return 0.0 if math.isnan(x) else x


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
    patch_size = tuple(config["transforms"]["patch_size"])
    sw_batch_size = int(config["training"].get("sw_batch_size", 1))
    infer_overlap = float(config["training"].get("infer_overlap", 0.5))

    best_metric = -1.0
    best_metric_epoch = -1

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_train_loss = 0.0
        train_steps = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = loss_function(outputs, labels)
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

        if epoch % val_every == 0:
            model.eval()
            val_loss = 0.0
            val_steps = 0

            metric.reset()

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

                    loss = loss_function(outputs, labels)
                    val_loss += float(loss.item())
                    val_steps += 1

                    outputs_list = [post_pred(o) for o in outputs]
                    labels_list = [post_label(l) for l in labels]
                    metric(y_pred=outputs_list, y=labels_list)

            val_loss /= max(val_steps, 1)

            metric_values = metric.aggregate()
            metric.reset()

            if hasattr(metric_values, "detach"):
                metric_values = metric_values.detach().cpu().float()

            if metric_values.ndim == 0:
                class_scores = [float(metric_values.item())]
            else:
                class_scores = [float(x) for x in metric_values.flatten()]

            dice_mean = _nanmean(class_scores)

            epoch_log["val_loss"] = val_loss
            epoch_log[mean_metric_name] = dice_mean

            print(f"val_loss: {val_loss:.6f}")
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

            epoch_log[best_mean_metric_name] = best_metric
            print(f"{best_mean_metric_name}: {best_metric:.6f}")
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