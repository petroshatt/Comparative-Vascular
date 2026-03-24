from __future__ import annotations

import glob
import json
import os
import random

from monai.data import CacheDataset, DataLoader

from .transforms import (
    get_train_transforms,
    get_val_transforms,
    get_train_transforms_nnunet_like,
    get_val_transforms_nnunet_like,
)


def discover_cases(images_dir: str, labels_dir: str):
    pet_files = sorted(glob.glob(os.path.join(images_dir, "*_0000.nii.gz")))
    if len(pet_files) == 0:
        raise RuntimeError(f"No PET files found in {images_dir}")

    data_dicts = []
    for pet_path in pet_files:
        case_id = os.path.basename(pet_path).replace("_0000.nii.gz", "")
        ct_path = os.path.join(images_dir, f"{case_id}_0001.nii.gz")
        label_path = os.path.join(labels_dir, f"{case_id}.nii.gz")

        if not os.path.exists(ct_path):
            raise FileNotFoundError(f"Missing CT file for case {case_id}: {ct_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Missing label file for case {case_id}: {label_path}")

        data_dicts.append(
            {
                "pet": pet_path,
                "ct": ct_path,
                "label": label_path,
                "case_id": case_id,
            }
        )

    return data_dicts


def split_cases(data_dicts: list[dict], split_seed: int, train_ratio: float, val_ratio: float):
    random.seed(split_seed)
    shuffled = data_dicts[:]
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_files = shuffled[:n_train]
    val_files = shuffled[n_train:n_train + n_val]
    test_files = shuffled[n_train + n_val:]

    return train_files, val_files, test_files


def build_datasets(config: dict):
    data_cfg = config["data"]
    loader_cfg = config.get("dataloader", {})

    data_dicts = discover_cases(
        images_dir=data_cfg["images_dir"],
        labels_dir=data_cfg["labels_dir"],
    )

    train_files, val_files, test_files = split_cases(
        data_dicts=data_dicts,
        split_seed=int(data_cfg.get("split_seed", 123)),
        train_ratio=float(data_cfg.get("train_ratio", 0.7)),
        val_ratio=float(data_cfg.get("val_ratio", 0.1)),
    )

    transform_mode = config.get("transforms", {}).get("mode", "default").lower()

    if transform_mode == "nnunet_like":
        train_transform = get_train_transforms_nnunet_like(config)
        val_transform = get_val_transforms_nnunet_like(config)
    else:
        train_transform = get_train_transforms(config)
        val_transform = get_val_transforms(config)

    train_ds = CacheDataset(
        data=train_files,
        transform=train_transform,
        cache_rate=float(loader_cfg.get("train_cache_rate", 1.0)),
        num_workers=int(loader_cfg.get("train_num_workers", 8)),
    )
    val_ds = CacheDataset(
        data=val_files,
        transform=val_transform,
        cache_rate=float(loader_cfg.get("val_cache_rate", 1.0)),
        num_workers=int(loader_cfg.get("val_num_workers", 4)),
    )
    test_ds = CacheDataset(
        data=test_files,
        transform=val_transform,
        cache_rate=float(loader_cfg.get("test_cache_rate", 1.0)),
        num_workers=int(loader_cfg.get("test_num_workers", 4)),
    )

    return train_ds, val_ds, test_ds, train_files, val_files, test_files


def build_dataloaders(config: dict):
    loader_cfg = config.get("dataloader", {})

    train_ds, val_ds, test_ds, train_files, val_files, test_files = build_datasets(config)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(loader_cfg.get("train_batch_size", 1)),
        shuffle=True,
        num_workers=int(loader_cfg.get("train_loader_workers", 4)),
        pin_memory=bool(loader_cfg.get("pin_memory", True)),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=int(loader_cfg.get("val_batch_size", 1)),
        shuffle=False,
        num_workers=int(loader_cfg.get("val_loader_workers", 2)),
        pin_memory=bool(loader_cfg.get("pin_memory", True)),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=int(loader_cfg.get("test_batch_size", 1)),
        shuffle=False,
        num_workers=int(loader_cfg.get("test_loader_workers", 2)),
        pin_memory=bool(loader_cfg.get("pin_memory", True)),
    )

    return train_loader, val_loader, test_loader, train_files, val_files, test_files


def save_split_manifest(run_dir: str, train_files: list[dict], val_files: list[dict], test_files: list[dict]):
    manifest = {
        "train": [x["case_id"] for x in train_files],
        "val": [x["case_id"] for x in val_files],
        "test": [x["case_id"] for x in test_files],
    }

    out_path = os.path.join(run_dir, "split_manifest.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)