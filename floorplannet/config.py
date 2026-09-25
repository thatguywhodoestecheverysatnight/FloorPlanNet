"""Typed experiment configuration loaded from YAML."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DataConfig:
    dataset: str = "cubicasa5k"          # cubicasa5k | synthetic
    root: str = "data/cubicasa5k"
    taxonomy: str = "coarse"             # coarse | cubicasa12
    image_file: str = "F1_scaled.png"
    crop_size: int = 512
    max_side: int = 1024                 # long side resize cap before cropping
    num_workers: int = 4
    synthetic_train: int = 4000
    synthetic_val: int = 400
    cache_masks: bool = True             # rasterise SVGs once to <root>/.cache


@dataclass
class ModelConfig:
    arch: str = "deeplabv3_resnet101"    # see floorplannet.models.MODEL_REGISTRY
    encoder: Optional[str] = None        # timm encoder for deeplabv3plus_timm
    pretrained: bool = True
    output_stride: int = 16
    aux_loss: bool = True


@dataclass
class LossConfig:
    ce_weight: float = 1.0
    dice_weight: float = 0.5
    boundary_weight: float = 2.0         # extra CE weight on thin structures (walls/openings)
    label_smoothing: float = 0.0
    class_weights: Optional[List[float]] = None
    ignore_index: int = 255


@dataclass
class OptimConfig:
    lr: float = 1e-3
    backbone_lr_mult: float = 0.1
    weight_decay: float = 1e-4
    epochs: int = 80
    batch_size: int = 8
    grad_accum: int = 1
    warmup_iters: int = 500
    amp: bool = True
    ema_decay: float = 0.999
    clip_grad: float = 1.0


@dataclass
class TrainConfig:
    name: str = "floorplannet"
    output_dir: str = "runs"
    seed: int = 42
    eval_every: int = 1
    log_every: int = 20
    max_iters: Optional[int] = None      # hard stop (useful for smoke tests / CPU)
    device: str = "auto"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _merge(dc: Any, overrides: Dict[str, Any]) -> Any:
    for f in fields(dc):
        if f.name not in overrides:
            continue
        value = overrides[f.name]
        current = getattr(dc, f.name)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(dc, f.name, value)
    return dc


def load_config(path: Optional[str] = None, overrides: Optional[List[str]] = None) -> TrainConfig:
    """Load YAML config, then apply ``key.sub=value`` CLI overrides."""
    cfg = TrainConfig()
    if path:
        with open(Path(path), "r", encoding="utf-8") as fh:
            _merge(cfg, yaml.safe_load(fh) or {})
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node: Dict[str, Any] = {}
        cursor = node
        parts = key.split(".")
        for p in parts[:-1]:
            cursor[p] = {}
            cursor = cursor[p]
        cursor[parts[-1]] = yaml.safe_load(raw)
        _merge(cfg, node)
    return cfg
