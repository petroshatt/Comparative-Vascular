from __future__ import annotations

import argparse
import glob
import json
import os
import random
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from monai import transforms
from monai.utils import set_determinism


# ----------------------------
# Small utilities
# ----------------------------

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def save_text(path: str, text: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        f.write(text)


def to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def unique_label_counts(label: np.ndarray) -> dict[int, int]:
    values, counts = np.unique(label.astype(np.int32), return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts)}


def format_counts(counts: dict[int, int]) -> str:
    if not counts:
        return "{}"
    return ", ".join([f"{k}: {v}" for k, v in sorted(counts.items(), key=lambda x: x[0])])


def get_spacing_from_meta(item: dict, key: str):
    meta_key = f"{key}_meta_dict"
    if meta_key in item:
        meta = item[meta_key]
        if "pixdim" in meta:
            try:
                return list(np.asarray(meta["pixdim"]).tolist())
            except Exception:
                return meta["pixdim"]
        if "spacing" in meta:
            return meta["spacing"]
        if "affine" in meta:
            affine = np.asarray(meta["affine"])
            if affine.shape == (4, 4):
                spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
                return spacing.tolist()
    return None


def center_slices(vol_3d: np.ndarray):
    z = vol_3d.shape[2] // 2
    y = vol_3d.shape[1] // 2
    x = vol_3d.shape[0] // 2
    axial = vol_3d[:, :, z]
    coronal = vol_3d[:, y, :]
    sagittal = vol_3d[x, :, :]
    return axial, coronal, sagittal


def normalize_for_display(img2d: np.ndarray) -> np.ndarray:
    img2d = img2d.astype(np.float32)
    vmin = np.percentile(img2d, 1)
    vmax = np.percentile(img2d, 99)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    out = np.clip((img2d - vmin) / (vmax - vmin), 0.0, 1.0)
    return out


def overlay_label_on_gray(gray: np.ndarray, label: np.ndarray) -> np.ndarray:
    base = np.stack([gray, gray, gray], axis=-1)
    colors = {
        1: np.array([1.0, 0.1, 0.1], dtype=np.float32),   # red
        2: np.array([0.1, 1.0, 0.1], dtype=np.float32),   # green
        3: np.array([0.1, 0.5, 1.0], dtype=np.float32),   # blue-ish
        4: np.array([1.0, 1.0, 0.1], dtype=np.float32),   # yellow
    }
    alpha = 0.45
    out = base.copy()
    for label_id, color in colors.items():
        mask = label == label_id
        if np.any(mask):
            out[mask] = (1 - alpha) * out[mask] + alpha * color
    return np.clip(out, 0.0, 1.0)


def save_step_figure(
    image: np.ndarray,
    label: np.ndarray,
    out_path: str,
    title: str,
):
    """
    image: [2, H, W, D] with channel 0 = CT, channel 1 = PET
    label: [1, H, W, D] or [H, W, D]
    """
    ensure_dir(os.path.dirname(out_path))

    if label.ndim == 4:
        label = label[0]

    ct = image[0]
    pet = image[1]

    ct_ax, ct_cor, ct_sag = center_slices(ct)
    pet_ax, pet_cor, pet_sag = center_slices(pet)
    lab_ax, lab_cor, lab_sag = center_slices(label)

    ct_ax_n = normalize_for_display(ct_ax)
    ct_cor_n = normalize_for_display(ct_cor)
    ct_sag_n = normalize_for_display(ct_sag)

    pet_ax_n = normalize_for_display(pet_ax)
    pet_cor_n = normalize_for_display(pet_cor)
    pet_sag_n = normalize_for_display(pet_sag)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(title, fontsize=13)

    planes = [
        ("Axial", ct_ax_n, pet_ax_n, lab_ax),
        ("Coronal", ct_cor_n, pet_cor_n, lab_cor),
        ("Sagittal", ct_sag_n, pet_sag_n, lab_sag),
    ]

    row_titles = ["CT + label", "PET + label", "Label"]

    for col, (plane_name, ct_im, pet_im, lab_im) in enumerate(planes):
        axes[0, col].imshow(overlay_label_on_gray(ct_im, lab_im), origin="lower")
        axes[0, col].set_title(f"{plane_name}")
        axes[0, col].axis("off")

        axes[1, col].imshow(overlay_label_on_gray(pet_im, lab_im), origin="lower")
        axes[1, col].axis("off")

        axes[2, col].imshow(lab_im, origin="lower")
        axes[2, col].axis("off")

    for row in range(3):
        axes[row, 0].set_ylabel(row_titles[row], fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Transforms
# ----------------------------

class ModalityAwareScalingd:
    """
    Expected channel order:
      image[0] = CT
      image[1] = PET
    """

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


def build_step_transforms(patch_size: tuple[int, int, int]):
    return [
        (
            "00_loaded",
            transforms.Compose([
                transforms.LoadImaged(keys=["ct", "pet", "label"]),
                transforms.EnsureChannelFirstd(keys=["ct", "pet", "label"]),
            ]),
        ),
        (
            "01_oriented",
            transforms.Compose([
                transforms.LoadImaged(keys=["ct", "pet", "label"]),
                transforms.EnsureChannelFirstd(keys=["ct", "pet", "label"]),
                transforms.Orientationd(keys=["ct", "pet", "label"], axcodes="RAS", labels=None),
            ]),
        ),
        (
            "02_concatenated",
            transforms.Compose([
                transforms.LoadImaged(keys=["ct", "pet", "label"]),
                transforms.EnsureChannelFirstd(keys=["ct", "pet", "label"]),
                transforms.Orientationd(keys=["ct", "pet", "label"], axcodes="RAS", labels=None),
                transforms.ConcatItemsd(keys=["ct", "pet"], name="image", dim=0),
            ]),
        ),
        (
            "03_cropforeground",
            transforms.Compose([
                transforms.LoadImaged(keys=["ct", "pet", "label"]),
                transforms.EnsureChannelFirstd(keys=["ct", "pet", "label"]),
                transforms.Orientationd(keys=["ct", "pet", "label"], axcodes="RAS", labels=None),
                transforms.ConcatItemsd(keys=["ct", "pet"], name="image", dim=0),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            ]),
        ),
        (
            "04_spatialpad",
            transforms.Compose([
                transforms.LoadImaged(keys=["ct", "pet", "label"]),
                transforms.EnsureChannelFirstd(keys=["ct", "pet", "label"]),
                transforms.Orientationd(keys=["ct", "pet", "label"], axcodes="RAS", labels=None),
                transforms.ConcatItemsd(keys=["ct", "pet"], name="image", dim=0),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                transforms.SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
            ]),
        ),
        (
            "05_scaled",
            transforms.Compose([
                transforms.LoadImaged(keys=["ct", "pet", "label"]),
                transforms.EnsureChannelFirstd(keys=["ct", "pet", "label"]),
                transforms.Orientationd(keys=["ct", "pet", "label"], axcodes="RAS", labels=None),
                transforms.ConcatItemsd(keys=["ct", "pet"], name="image", dim=0),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                transforms.SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
                ModalityAwareScalingd(
                    keys="image",
                    channel_ranges=[(-1000.0, 1500.0), (0.0, 4.0)],
                ),
            ]),
        ),
    ]


def build_train_patch_transform(patch_size: tuple[int, int, int]):
    return transforms.Compose([
        transforms.LoadImaged(keys=["ct", "pet", "label"]),
        transforms.EnsureChannelFirstd(keys=["ct", "pet", "label"]),
        transforms.Orientationd(keys=["ct", "pet", "label"], axcodes="RAS", labels=None),
        transforms.ConcatItemsd(keys=["ct", "pet"], name="image", dim=0),
        transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
        transforms.SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
        ModalityAwareScalingd(
            keys="image",
            channel_ranges=[(-1000.0, 1500.0), (0.0, 4.0)],
        ),
        transforms.RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=1,
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


# ----------------------------
# Case discovery
# ----------------------------

@dataclass
class CasePaths:
    case_id: str
    pet: str
    ct: str
    label: str


def discover_cases(images_dir: str, labels_dir: str) -> list[CasePaths]:
    pet_files = sorted(glob.glob(os.path.join(images_dir, "*_0000.nii.gz")))
    if len(pet_files) == 0:
        raise RuntimeError(f"No PET files found in {images_dir}")

    cases: list[CasePaths] = []
    for pet_path in pet_files:
        case_id = os.path.basename(pet_path).replace("_0000.nii.gz", "")
        ct_path = os.path.join(images_dir, f"{case_id}_0001.nii.gz")
        label_path = os.path.join(labels_dir, f"{case_id}.nii.gz")

        if not os.path.exists(ct_path):
            raise FileNotFoundError(f"Missing CT file for case {case_id}: {ct_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Missing label file for case {case_id}: {label_path}")

        cases.append(CasePaths(case_id=case_id, pet=pet_path, ct=ct_path, label=label_path))

    return cases


def choose_case(cases: list[CasePaths], case_id: str | None, index: int | None) -> CasePaths:
    if case_id is not None:
        for c in cases:
            if c.case_id == case_id:
                return c
        raise ValueError(f"Case '{case_id}' not found.")
    if index is None:
        index = 0
    if index < 0 or index >= len(cases):
        raise IndexError(f"Case index {index} is out of range for {len(cases)} cases.")
    return cases[index]


# ----------------------------
# Stats
# ----------------------------

def describe_step(item: dict, step_name: str) -> str:
    lines = [f"Step: {step_name}"]

    for key in ["ct", "pet", "image", "label"]:
        if key not in item:
            continue

        arr = to_numpy(item[key])
        lines.append(f"  {key}:")
        lines.append(f"    shape: {tuple(arr.shape)}")
        lines.append(f"    dtype: {arr.dtype}")

        if key == "label":
            uniq = unique_label_counts(arr)
            lines.append(f"    unique/counts: {format_counts(uniq)}")
        else:
            lines.append(f"    min: {float(arr.min()):.6f}")
            lines.append(f"    max: {float(arr.max()):.6f}")
            lines.append(f"    mean: {float(arr.mean()):.6f}")
            if arr.ndim >= 4:
                for ch in range(arr.shape[0]):
                    ch_arr = arr[ch]
                    lines.append(
                        f"    channel_{ch}: min={float(ch_arr.min()):.6f}, "
                        f"max={float(ch_arr.max()):.6f}, mean={float(ch_arr.mean()):.6f}"
                    )

    for meta_key in ["ct", "pet", "label"]:
        spacing = get_spacing_from_meta(item, meta_key)
        if spacing is not None:
            lines.append(f"  {meta_key} spacing/meta: {spacing}")

    lines.append("")
    return "\n".join(lines)


def describe_patch(item: dict, patch_idx: int) -> str:
    image = to_numpy(item["image"])
    label = to_numpy(item["label"])
    counts = unique_label_counts(label)

    lines = [f"Patch sample {patch_idx}"]
    lines.append(f"  image shape: {tuple(image.shape)}")
    lines.append(f"  label shape: {tuple(label.shape)}")
    lines.append(f"  image dtype: {image.dtype}")
    lines.append(f"  label dtype: {label.dtype}")
    lines.append(f"  CT channel min/max/mean: {float(image[0].min()):.6f} / {float(image[0].max()):.6f} / {float(image[0].mean()):.6f}")
    lines.append(f"  PET channel min/max/mean: {float(image[1].min()):.6f} / {float(image[1].max()):.6f} / {float(image[1].mean()):.6f}")
    lines.append(f"  label unique/counts: {format_counts(counts)}")
    lines.append(f"  contains foreground: {'yes' if np.any(label > 0) else 'no'}")
    lines.append("")
    return "\n".join(lines)


# ----------------------------
# Main inspection
# ----------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Inspect PET/CT preprocessing step by step.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_petct_vascular.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--case-id",
        type=str,
        default=None,
        help="Specific case ID to inspect.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Case index to inspect if case-id is not given.",
    )
    parser.add_argument(
        "--num-random-patches",
        type=int,
        default=3,
        help="How many random training patches to sample and save.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="preprocessing_inspection/output",
        help="Root directory for inspection outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_determinism(seed=args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = load_config(args.config)
    patch_size = tuple(config["transforms"]["patch_size"])

    images_dir = config["data"]["images_dir"]
    labels_dir = config["data"]["labels_dir"]

    cases = discover_cases(images_dir, labels_dir)
    case = choose_case(cases, args.case_id, args.index)

    case_out_dir = ensure_dir(os.path.join(args.output_root, case.case_id))
    figs_dir = ensure_dir(os.path.join(case_out_dir, "figures"))

    base_item = {
        "ct": case.ct,
        "pet": case.pet,
        "label": case.label,
        "case_id": case.case_id,
    }

    stats_blocks = []
    summary = {
        "case_id": case.case_id,
        "ct": case.ct,
        "pet": case.pet,
        "label": case.label,
        "patch_size": patch_size,
        "num_random_patches": args.num_random_patches,
        "seed": args.seed,
    }

    # Deterministic steps
    for step_name, step_transform in build_step_transforms(patch_size):
        out = step_transform(dict(base_item))

        stats_blocks.append(describe_step(out, step_name))

        if "image" in out:
            image = to_numpy(out["image"])
        else:
            ct = to_numpy(out["ct"])
            pet = to_numpy(out["pet"])
            image = np.concatenate([ct, pet], axis=0)

        label = to_numpy(out["label"])
        fig_path = os.path.join(figs_dir, f"{step_name}.png")
        save_step_figure(
            image=image,
            label=label,
            out_path=fig_path,
            title=f"{case.case_id} - {step_name}",
        )

    # Random train-patch samples
    train_patch_transform = build_train_patch_transform(patch_size)

    for patch_idx in range(args.num_random_patches):
        out = train_patch_transform(dict(base_item))
        patch_item = out[0] if isinstance(out, list) else out

        stats_blocks.append(describe_patch(patch_item, patch_idx))

        image = to_numpy(patch_item["image"])
        label = to_numpy(patch_item["label"])

        fig_path = os.path.join(figs_dir, f"06_train_patch_{patch_idx:02d}.png")
        save_step_figure(
            image=image,
            label=label,
            out_path=fig_path,
            title=f"{case.case_id} - train_patch_{patch_idx:02d}",
        )

    with open(os.path.join(case_out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    save_text(os.path.join(case_out_dir, "stats.txt"), "\n".join(stats_blocks))

    print(f"Inspection completed for case: {case.case_id}")
    print(f"Output directory: {case_out_dir}")
    print(f"Figures: {figs_dir}")
    print(f"Stats file: {os.path.join(case_out_dir, 'stats.txt')}")


if __name__ == "__main__":
    main()