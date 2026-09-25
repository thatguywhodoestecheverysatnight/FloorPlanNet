"""Streaming segmentation metrics built on a confusion matrix.

mIoU follows the CubiCasa5K / Cityscapes convention: IoU per class accumulated
over the whole split (not averaged per image), then averaged over classes.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch


class ConfusionMatrix:
    def __init__(self, num_classes: int, ignore_index: int = 255, device: Optional[torch.device] = None):
        self.k, self.ignore = num_classes, ignore_index
        self.mat = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        if pred.dim() == target.dim() + 1:
            pred = pred.argmax(1)
        pred, target = pred.reshape(-1), target.reshape(-1)
        m = target != self.ignore
        idx = target[m].to(torch.int64) * self.k + pred[m].to(torch.int64)
        self.mat += torch.bincount(idx, minlength=self.k ** 2).reshape(self.k, self.k).to(self.mat.device)

    def update_numpy(self, pred: np.ndarray, target: np.ndarray) -> None:
        self.update(torch.from_numpy(np.asarray(pred)), torch.from_numpy(np.asarray(target)))

    def reset(self) -> None:
        self.mat.zero_()

    def all_reduce(self) -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(self.mat)

    def compute(self, class_names: Optional[Sequence[str]] = None) -> Dict[str, object]:
        h = self.mat.double().cpu()
        tp = h.diag()
        gt = h.sum(1)
        pr = h.sum(0)
        union = gt + pr - tp
        iou = tp / union.clamp_min(1)
        acc_c = tp / gt.clamp_min(1)
        present = gt > 0
        freq = gt / gt.sum().clamp_min(1)
        names: List[str] = list(class_names) if class_names else [str(i) for i in range(self.k)]
        return {
            "mIoU": float(iou[present].mean()) if present.any() else 0.0,
            "fwIoU": float((freq * iou).sum()),
            "pixel_acc": float(tp.sum() / h.sum().clamp_min(1)),
            "mean_acc": float(acc_c[present].mean()) if present.any() else 0.0,
            "per_class_iou": {n: float(v) for n, v, p in zip(names, iou, present) if p},
        }


def format_metrics(m: Dict[str, object]) -> str:
    head = f"mIoU {m['mIoU']:.4f} | fwIoU {m['fwIoU']:.4f} | pixAcc {m['pixel_acc']:.4f} | mAcc {m['mean_acc']:.4f}"
    per = "  ".join(f"{k}:{v:.3f}" for k, v in m["per_class_iou"].items())  # type: ignore[union-attr]
    return f"{head}\n  {per}"
