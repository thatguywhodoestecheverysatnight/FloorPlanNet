"""Segmentation losses tuned for thin-structure floor plan parsing.

``FloorPlanLoss = ce_w * boundary-weighted CE + dice_w * soft Dice``

* Boundary-weighted CE up-weights pixels within a few pixels of a label edge.
  Walls and openings are 2-10 px wide, so plain CE under-penalises the errors
  that break vectorisation (gaps in walls, merged rooms).
* Soft Dice is computed per class over the batch and averaged over classes that
  are present, which counters the heavy background/room imbalance.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def edge_weight_map(target: torch.Tensor, ignore_index: int, radius: int = 2) -> torch.Tensor:
    """1.0 on label boundaries (within ``radius`` px), 0.0 elsewhere."""
    t = target.clone().float()
    t[target == ignore_index] = -1
    t = t.unsqueeze(1)
    k = 2 * radius + 1
    mx = F.max_pool2d(t, k, 1, radius)
    mn = -F.max_pool2d(-t, k, 1, radius)
    return (mx != mn).squeeze(1).float()


def soft_dice(logits: torch.Tensor, target: torch.Tensor, ignore_index: int, eps: float = 1.0) -> torch.Tensor:
    k = logits.shape[1]
    valid = (target != ignore_index)
    probs = logits.float().softmax(1) * valid.unsqueeze(1)
    tgt = target.clone()
    tgt[~valid] = 0
    onehot = F.one_hot(tgt, k).permute(0, 3, 1, 2).float() * valid.unsqueeze(1)
    dims = (0, 2, 3)
    inter = (probs * onehot).sum(dims)
    denom = probs.sum(dims) + onehot.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)
    present = onehot.sum(dims) > 0
    return 1 - (dice[present].mean() if present.any() else dice.mean())


class FloorPlanLoss(nn.Module):
    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 0.5, boundary_weight: float = 2.0,
                 class_weights: Optional[Sequence[float]] = None, ignore_index: int = 255,
                 label_smoothing: float = 0.0, aux_weight: float = 0.4):
        super().__init__()
        self.ce_w, self.dice_w, self.bw = ce_weight, dice_weight, boundary_weight
        self.ignore, self.ls, self.aux_w = ignore_index, label_smoothing, aux_weight
        self.register_buffer("cw", torch.tensor(class_weights, dtype=torch.float32) if class_weights else None,
                             persistent=False)

    def _single(self, logits: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        ce = F.cross_entropy(logits.float(), target, weight=self.cw, ignore_index=self.ignore,
                             reduction="none", label_smoothing=self.ls)
        valid = (target != self.ignore).float()
        w = 1.0 + self.bw * edge_weight_map(target, self.ignore)
        ce = (ce * w * valid).sum() / (w * valid).sum().clamp_min(1.0)
        dice = soft_dice(logits, target, self.ignore) if self.dice_w > 0 else logits.new_zeros(())
        return {"ce": ce, "dice": dice, "total": self.ce_w * ce + self.dice_w * dice}

    def forward(self, outputs: Dict[str, torch.Tensor], target: torch.Tensor) -> Dict[str, torch.Tensor]:
        main = self._single(outputs["out"], target)
        if "aux" in outputs and self.aux_w > 0:
            aux = self._single(outputs["aux"], target)
            main["aux"] = aux["total"]
            main["total"] = main["total"] + self.aux_w * aux["total"]
        return main
