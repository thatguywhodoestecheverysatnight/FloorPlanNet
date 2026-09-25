"""Joint image/mask augmentations (pure OpenCV + NumPy, no extra deps).

Floor plans are orientation-agnostic under the dihedral group D4, so random
90-degree rotations and flips are label-preserving. Thin structures (walls,
openings) are fragile under interpolation, so masks always use nearest
neighbour and geometric scale is bounded.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def resize_long_side(img: np.ndarray, mask: Optional[np.ndarray], max_side: int):
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1.0:
        return img, mask
    size = (max(1, round(w * s)), max(1, round(h * s)))
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    if mask is not None:
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return img, mask


def pad_to(img: np.ndarray, mask: Optional[np.ndarray], h: int, w: int, ignore: int = 255):
    ph, pw = max(0, h - img.shape[0]), max(0, w - img.shape[1])
    if ph == 0 and pw == 0:
        return img, mask
    img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    if mask is not None:
        mask = cv2.copyMakeBorder(mask, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=ignore)
    return img, mask


class TrainAugment:
    def __init__(self, crop: int = 512, scale: Tuple[float, float] = (0.6, 1.4), ignore: int = 255,
                 seed: Optional[int] = None):
        self.crop, self.scale, self.ignore = crop, scale, ignore
        self.rng = np.random.default_rng(seed)

    def __call__(self, img: np.ndarray, mask: np.ndarray):
        r = self.rng
        s = r.uniform(*self.scale)
        if abs(s - 1) > 1e-3:
            size = (max(8, int(img.shape[1] * s)), max(8, int(img.shape[0] * s)))
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
            mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        k = int(r.integers(0, 4))
        if k:
            img, mask = np.rot90(img, k).copy(), np.rot90(mask, k).copy()
        if r.random() < 0.5:
            img, mask = img[:, ::-1].copy(), mask[:, ::-1].copy()
        img, mask = pad_to(img, mask, self.crop, self.crop, self.ignore)
        # foreground-biased crop so small plans are not dominated by padding
        h, w = mask.shape
        fg = np.argwhere((mask != 0) & (mask != self.ignore))
        if len(fg) and r.random() < 0.8:
            cy, cx = fg[r.integers(len(fg))]
            y0 = int(np.clip(cy - self.crop // 2, 0, h - self.crop))
            x0 = int(np.clip(cx - self.crop // 2, 0, w - self.crop))
        else:
            y0 = int(r.integers(0, h - self.crop + 1))
            x0 = int(r.integers(0, w - self.crop + 1))
        img = img[y0:y0 + self.crop, x0:x0 + self.crop]
        mask = mask[y0:y0 + self.crop, x0:x0 + self.crop]
        return self.photometric(img), mask

    def photometric(self, img: np.ndarray) -> np.ndarray:
        r = self.rng
        out = img.astype(np.float32)
        if r.random() < 0.8:
            out = out * r.uniform(0.75, 1.25) + r.uniform(-25, 25)
        if r.random() < 0.3:
            g = out.mean(axis=2, keepdims=True)
            out = np.repeat(g, 3, axis=2)
        if r.random() < 0.3:
            out = out + r.uniform(-15, 15, size=(1, 1, 3))
        if r.random() < 0.2:
            out = cv2.GaussianBlur(out, (3, 3), r.uniform(0.2, 1.2))
        if r.random() < 0.2:
            out = out + r.normal(0, r.uniform(2, 10), out.shape)
        return np.clip(out, 0, 255).astype(np.uint8)


def to_tensor_arrays(img: np.ndarray) -> np.ndarray:
    """HWC uint8 RGB -> CHW float32 ImageNet-normalised."""
    x = img.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))
