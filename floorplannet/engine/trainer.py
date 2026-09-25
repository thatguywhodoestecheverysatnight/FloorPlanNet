"""Training loop: AMP, grad accumulation, warmup + poly LR, EMA, DDP, checkpointing.

Launch single GPU:  ``python -m floorplannet.cli train -c configs/cubicasa5k_deeplabv3_r101.yaml``
Launch multi GPU:   ``torchrun --nproc_per_node 4 -m floorplannet.cli train -c ...``
"""
from __future__ import annotations

import copy
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from ..classes import get_taxonomy
from ..config import TrainConfig
from ..data.datasets import build_loaders
from ..losses import FloorPlanLoss
from ..metrics import ConfusionMatrix, format_metrics
from ..models import build_model, count_params, param_groups


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ModelEMA:
    """Exponential moving average of weights (BN buffers copied)."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        self.updates = 0
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))  # warm start
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


def lr_at(it: int, total: int, warmup: int, power: float = 0.9) -> float:
    if it < warmup:
        return (it + 1) / max(1, warmup)
    t = (it - warmup) / max(1, total - warmup)
    return max(0.0, (1 - t)) ** power


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, num_classes: int, device: torch.device,
             class_names, ignore_index: int = 255, amp: bool = False) -> Dict[str, object]:
    model.eval()
    cm = ConfusionMatrix(num_classes, ignore_index, device)
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
            logits = model(x)["out"]
        cm.update(logits.argmax(1), y)
    cm.all_reduce()
    return cm.compute(class_names)


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
        if self.distributed and not dist.is_initialized():
            dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        self.rank = dist.get_rank() if self.distributed else 0
        self.device = resolve_device(cfg.device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            torch.backends.cudnn.benchmark = True
        seed_everything(cfg.seed + self.rank)

        self.train_loader, self.val_loader, self.k = build_loaders(cfg)
        if self.distributed:
            ds = self.train_loader.dataset
            self.train_loader = DataLoader(ds, batch_size=cfg.optim.batch_size, sampler=DistributedSampler(ds),
                                           num_workers=cfg.data.num_workers, drop_last=True, pin_memory=True)
        self.class_names = list(get_taxonomy(cfg.data.taxonomy).classes)[: self.k]

        m = cfg.model
        self.model = build_model(m.arch, self.k, encoder=m.encoder, pretrained=m.pretrained,
                                 output_stride=m.output_stride, aux_loss=m.aux_loss).to(self.device)
        if self.distributed:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.device.index]
                                                                   if self.device.type == "cuda" else None)
        core = self.model.module if self.distributed else self.model
        self.ema = ModelEMA(core, cfg.optim.ema_decay) if cfg.optim.ema_decay > 0 else None
        lc = cfg.loss
        self.criterion = FloorPlanLoss(lc.ce_weight, lc.dice_weight, lc.boundary_weight, lc.class_weights,
                                       lc.ignore_index, lc.label_smoothing).to(self.device)
        o = cfg.optim
        self.opt = torch.optim.AdamW(param_groups(self.model, o.lr, o.backbone_lr_mult, o.weight_decay), lr=o.lr)
        self.use_amp = o.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.out_dir = Path(cfg.output_dir) / cfg.name
        if self.rank == 0:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            (self.out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
            print(f"[floorplannet] {m.arch}: {count_params(core) / 1e6:.2f}M params | classes={self.class_names} "
                  f"| device={self.device} | amp={self.use_amp}")
        self.best = -1.0
        self.it = 0

    def _log(self, rec: Dict[str, object]) -> None:
        if self.rank != 0:
            return
        with open(self.out_dir / "log.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    def save(self, name: str, metrics: Optional[Dict[str, object]] = None) -> None:
        if self.rank != 0:
            return
        core = self.model.module if self.distributed else self.model
        state = {
            "model": (self.ema.module if self.ema else core).state_dict(),
            "raw_model": core.state_dict(),
            "optimizer": self.opt.state_dict(),
            "iter": self.it,
            "config": self.cfg.to_dict(),
            "num_classes": self.k,
            "class_names": self.class_names,
            "metrics": metrics,
        }
        torch.save(state, self.out_dir / name)

    def resume(self, path: str) -> None:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        core = self.model.module if self.distributed else self.model
        core.load_state_dict(ck["raw_model"])
        if self.ema:
            self.ema.module.load_state_dict(ck["model"])
        if "optimizer" in ck:  # full resume; weights-only checkpoints start a fresh schedule
            self.opt.load_state_dict(ck["optimizer"])
            self.it = ck["iter"]

    def fit(self) -> Dict[str, object]:
        cfg, o = self.cfg, self.cfg.optim
        steps_per_epoch = len(self.train_loader) // o.grad_accum
        total = cfg.max_iters or o.epochs * steps_per_epoch
        epoch, last_metrics = 0, {}
        t0 = time.time()
        while self.it < total:
            if self.distributed:
                self.train_loader.sampler.set_epoch(epoch)  # type: ignore[union-attr]
            self.model.train()
            for step, (x, y) in enumerate(self.train_loader):
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                with torch.autocast(self.device.type, enabled=self.use_amp):
                    out = self.model(x)
                losses = self.criterion(out, y)
                self.scaler.scale(losses["total"] / o.grad_accum).backward()
                if (step + 1) % o.grad_accum:
                    continue
                f = lr_at(self.it, total, o.warmup_iters)
                for g in self.opt.param_groups:
                    g["lr"] = o.lr * g.get("lr_mult", 1.0) * f
                if o.clip_grad:
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), o.clip_grad)
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad(set_to_none=True)
                if self.ema:
                    self.ema.update(self.model.module if self.distributed else self.model)
                self.it += 1
                if self.it % cfg.log_every == 0 and self.rank == 0:
                    rec = {"iter": self.it, "epoch": epoch, "lr": o.lr * f,
                           **{k: float(v.detach()) for k, v in losses.items()}, "elapsed_s": round(time.time() - t0, 1)}
                    print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()),
                          flush=True)
                    self._log(rec)
                if self.it >= total:
                    break
            epoch += 1
            if epoch % cfg.eval_every == 0 or self.it >= total:
                net = self.ema.module if self.ema else (self.model.module if self.distributed else self.model)
                last_metrics = evaluate(net, self.val_loader, self.k, self.device, self.class_names,
                                        cfg.loss.ignore_index, self.use_amp)
                if self.rank == 0:
                    print(f"[eval] epoch {epoch} iter {self.it}: {format_metrics(last_metrics)}", flush=True)
                    self._log({"iter": self.it, "epoch": epoch, "eval": last_metrics})
                    self.save("last.pt", last_metrics)
                    if last_metrics["mIoU"] > self.best:
                        self.best = float(last_metrics["mIoU"])
                        self.save("best.pt", last_metrics)
        if self.distributed:
            dist.destroy_process_group()
        return {"best_mIoU": self.best, "last": last_metrics}
