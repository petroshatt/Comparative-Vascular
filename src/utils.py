from __future__ import annotations

import os
import random
import shutil
from datetime import datetime

import numpy as np
import torch
import yaml


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_configs(base: dict, override: dict) -> dict:
    merged = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = merge_configs(merged[k], v)
        else:
            merged[k] = v
    return merged


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def write_text(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def resolve_run_dir(base_log_dir: str, run_name: str | None = None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name is None or str(run_name).strip() == "":
        run_dir_name = timestamp
    else:
        run_dir_name = f"{timestamp}_{run_name}"
    run_dir = os.path.join(base_log_dir, run_dir_name)
    ensure_dir(run_dir)
    ensure_dir(os.path.join(run_dir, "models"))
    ensure_dir(os.path.join(run_dir, "artifacts"))
    return run_dir


def copy_file_to_dir(src_path: str, dst_dir: str, dst_name: str | None = None):
    ensure_dir(dst_dir)
    if dst_name is None:
        dst_name = os.path.basename(src_path)
    shutil.copy2(src_path, os.path.join(dst_dir, dst_name))