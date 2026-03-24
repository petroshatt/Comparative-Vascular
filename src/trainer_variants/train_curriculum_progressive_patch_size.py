from __future__ import annotations

import warnings
import os

warnings.filterwarnings(
    "ignore",
    message=r".*cuda\.cudart module is deprecated.*",
    category=FutureWarning,
)

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
from typing import Any
import torch
import wandb

from ..data import build_dataloaders, save_split_manifest
from ..engine import run_training
from ..losses_and_metrics import build_loss, build_metric, build_post_transforms
from ..models import build_model
from ..utils import (
    copy_file_to_dir,
    count_parameters,
    get_device,
    load_config,
    merge_configs,
    resolve_run_dir,
    set_seed,
    write_text,
)


REGION_METRIC_NAMES = {
    1: "dice_abdominal",
    2: "dice_aortic_arch",
    3: "dice_ascending",
    4: "dice_descending",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train PET/CT segmentation model (progressive patch curriculum)")
    parser.add_argument("--base_config", type=str, required=True)
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--curriculum_config", type=str, required=True)
    return parser.parse_args()


def _to_plain_dict(obj: Any):
    if isinstance(obj, dict):
        return {k: _to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain_dict(v) for v in obj]
    return obj


def _build_wandb_config(config: dict, run_dir: str) -> dict:
    tracked = {
        "seed": config["seed"],
        "run_dir": run_dir,
        "model": config["model"],
        "transforms": config["transforms"],
        "optimization": config["optimization"],
        "training": config["training"],
        "metric_names": {
            "mean": "dice_mean_score",
            "best_mean": "best_dice_mean_score",
            **{str(k): v for k, v in REGION_METRIC_NAMES.items()},
        },
    }
    return _to_plain_dict(tracked)


def _init_wandb(config: dict, run_dir: str):
    wandb_cfg = config.get("wandb", {})
    if not bool(wandb_cfg.get("enabled", False)):
        return None

    return wandb.init(
        project=wandb_cfg.get("project", "petct-segmentation"),
        entity=wandb_cfg.get("entity", None),
        name=wandb_cfg.get("run_name", None),
        dir=run_dir,
        config=_build_wandb_config(config, run_dir),
        mode=wandb_cfg.get("mode", "online"),
    )


def main():
    args = parse_args()

    base_config = load_config(args.base_config)
    model_config = load_config(args.model_config)
    curriculum_config = load_config(args.curriculum_config)

    config = merge_configs(base_config, model_config)
    config = merge_configs(config, curriculum_config)

    set_seed(int(config["seed"]))

    run_dir = resolve_run_dir(config["output_dir"], config.get("run_name", None))
    device = get_device()

    artifacts_dir = os.path.join(run_dir, "artifacts")
    models_dir = os.path.join(run_dir, "models")

    copy_file_to_dir(args.base_config, artifacts_dir, "base_config.yaml")
    copy_file_to_dir(args.model_config, artifacts_dir, "model_config.yaml")
    copy_file_to_dir(args.curriculum_config, artifacts_dir, "curriculum_config.yaml")
    copy_file_to_dir(__file__, artifacts_dir, "train_curriculum_progressive_patch.py")

    train_loader, val_loader, test_loader, train_files, val_files, test_files = build_dataloaders(config)

    save_split_manifest(run_dir, train_files, val_files, test_files)

    model = build_model(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optimization"]["learning_rate"]),
        weight_decay=float(config["optimization"]["weight_decay"]),
    )

    loss_function = build_loss(config)
    metric = build_metric(config)
    post_label, post_pred = build_post_transforms(config)

    wandb_run = _init_wandb(config, run_dir)

    try:
        run_training(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            metric=metric,
            post_pred=post_pred,
            post_label=post_label,
            device=device,
            config=config,
            run_dir=run_dir,
            models_dir=models_dir,
            wandb_run=wandb_run,
            region_metric_names=REGION_METRIC_NAMES,
        )
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()