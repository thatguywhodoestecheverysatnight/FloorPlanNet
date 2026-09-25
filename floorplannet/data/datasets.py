"""PyTorch datasets and loader factory."""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..classes import get_taxonomy
from ..config import TrainConfig
from .cubicasa import load_sample, read_split
from .synthetic import FloorPlanGenerator, SynthConfig
from .transforms import TrainAugment, pad_to, resize_long_side, to_tensor_arrays


class CubiCasa5K(Dataset):
    def __init__(self, root: str, split: str, taxonomy: str = "coarse", image_file: str = "F1_scaled.png",
                 max_side: int = 1024, augment: Optional[TrainAugment] = None, cache: bool = True):
        self.folders = read_split(root, split)
        self.tax = get_taxonomy(taxonomy)
        self.image_file, self.max_side, self.augment, self.cache = image_file, max_side, augment, cache

    def __len__(self) -> int:
        return len(self.folders)

    def __getitem__(self, idx: int):
        img, mask = load_sample(self.folders[idx], self.tax, self.image_file, self.cache)
        img, mask = resize_long_side(img, mask, self.max_side)
        if self.augment is not None:
            img, mask = self.augment(img, mask)
        return torch.from_numpy(to_tensor_arrays(img)), torch.from_numpy(mask.astype(np.int64))


class SyntheticPlans(Dataset):
    """Deterministic per-index procedural samples (seed = base_seed + idx)."""

    def __init__(self, length: int, seed: int = 0, size: int = 512, augment: Optional[TrainAugment] = None):
        self.length, self.seed, self.size, self.augment = length, seed, size, augment

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        img, mask = FloorPlanGenerator(SynthConfig(size=self.size), seed=self.seed + idx).generate()
        img = img[:, :, ::-1].copy()  # generator renders BGR
        if self.augment is not None:
            img, mask = self.augment(img, mask)
        return torch.from_numpy(to_tensor_arrays(img)), torch.from_numpy(mask.astype(np.int64))


class FolderDataset(Dataset):
    """Generic ``<root>/<split>/{images,masks}/<name>.png`` dataset.

    Used for pre-rendered synthetic data and for custom annotated plans. Masks
    are single-channel PNGs holding class indices.
    """

    def __init__(self, root: str, split: str, max_side: int = 1024, augment: Optional[TrainAugment] = None):
        base = Path(root) / split
        self.images = sorted((base / "images").glob("*.png"))
        self.masks = [base / "masks" / p.name for p in self.images]
        if not self.images:
            raise FileNotFoundError(f"no images under {base / 'images'}")
        self.max_side, self.augment = max_side, augment

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = cv2.cvtColor(cv2.imread(str(self.images[idx]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.masks[idx]), cv2.IMREAD_GRAYSCALE)
        img, mask = resize_long_side(img, mask, self.max_side)
        if self.augment is not None:
            img, mask = self.augment(img, mask)
        return torch.from_numpy(to_tensor_arrays(img)), torch.from_numpy(mask.astype(np.int64))


def pad_collate(batch, ignore: int = 255, multiple: int = 32):
    """Pad variable-size eval samples to a shared, stride-aligned shape."""
    h = max(x.shape[1] for x, _ in batch)
    w = max(x.shape[2] for x, _ in batch)
    h, w = -(-h // multiple) * multiple, -(-w // multiple) * multiple
    xs, ys = [], []
    for x, y in batch:
        px = torch.zeros(3, h, w)
        px[:] = torch.tensor([(1 - m) / s for m, s in zip((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]).view(3, 1, 1)
        px[:, : x.shape[1], : x.shape[2]] = x
        py = torch.full((h, w), ignore, dtype=torch.long)
        py[: y.shape[0], : y.shape[1]] = y
        xs.append(px)
        ys.append(py)
    return torch.stack(xs), torch.stack(ys)


def build_loaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader, int]:
    d = cfg.data
    aug = TrainAugment(crop=d.crop_size, ignore=cfg.loss.ignore_index, seed=cfg.seed)
    if d.dataset == "synthetic":
        train = SyntheticPlans(d.synthetic_train, seed=cfg.seed, size=d.max_side, augment=aug)
        val = SyntheticPlans(d.synthetic_val, seed=10_000_000 + cfg.seed, size=d.max_side)
        num_classes = get_taxonomy("coarse").num_classes
    elif d.dataset == "folder":
        train = FolderDataset(d.root, "train", d.max_side, aug)
        val = FolderDataset(d.root, "val", d.max_side)
        num_classes = get_taxonomy(d.taxonomy).num_classes
    elif d.dataset == "cubicasa5k":
        train = CubiCasa5K(d.root, "train", d.taxonomy, d.image_file, d.max_side, aug, d.cache_masks)
        val = CubiCasa5K(d.root, "val", d.taxonomy, d.image_file, d.max_side, None, d.cache_masks)
        num_classes = get_taxonomy(d.taxonomy).num_classes
    else:
        raise ValueError(f"unknown dataset {d.dataset}")

    def worker_init(worker_id: int) -> None:
        info = torch.utils.data.get_worker_info()
        ds = info.dataset
        if getattr(ds, "augment", None) is not None:
            ds.augment.rng = np.random.default_rng(cfg.seed * 1000 + worker_id + int(torch.initial_seed() % 2**31))

    train_loader = DataLoader(train, batch_size=cfg.optim.batch_size, shuffle=True, drop_last=True,
                              num_workers=d.num_workers, pin_memory=torch.cuda.is_available(),
                              persistent_workers=d.num_workers > 0, worker_init_fn=worker_init)
    val_loader = DataLoader(val, batch_size=1, shuffle=False, num_workers=d.num_workers,
                            collate_fn=functools.partial(pad_collate, ignore=cfg.loss.ignore_index))
    return train_loader, val_loader, num_classes


__all__ = ["CubiCasa5K", "FolderDataset", "SyntheticPlans", "build_loaders", "pad_collate", "pad_to"]
