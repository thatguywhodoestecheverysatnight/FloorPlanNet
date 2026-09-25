"""DeepLabV3 / DeepLabV3+ model zoo for floor plan parsing.

All models return ``{"out": logits[N,K,H,W], "aux": optional}`` at input
resolution, so the trainer, predictor and ONNX exporter are architecture
agnostic.

Registered architectures
------------------------
deeplabv3_resnet50 / deeplabv3_resnet101
    torchvision DeepLabV3, dilated ResNet (output stride 8), ASPP rates 12/24/36.
deeplabv3plus_timm
    DeepLabV3+ (ASPP + stride-4 decoder) on any ``timm`` encoder
    (``resnet50``, ``convnext_tiny``, ``efficientnet_b3`` ...).
deeplabv3plus_mobilenet (alias: lite)
    DeepLabV3+ on a dilated MobileNetV3-Large; ~3.9M params, runs in the
    browser through onnxruntime-web.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation.deeplabv3 import ASPP


class _Wrap(nn.Module):
    """Adapt torchvision segmentation models (OrderedDict output) to our contract."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        o = self.net(x)
        out = {"out": o["out"]}
        if self.training and "aux" in o:
            out["aux"] = o["aux"]
        return out

    def param_groups(self):
        return self.net.backbone.parameters(), [p for n, p in self.net.named_parameters() if not n.startswith("backbone.")]


class DeepLabV3PlusHead(nn.Module):
    def __init__(self, high_ch: int, low_ch: int, num_classes: int, aspp_ch: int = 256,
                 rates: Sequence[int] = (6, 12, 18), low_proj: int = 48):
        super().__init__()
        self.aspp = ASPP(high_ch, list(rates), aspp_ch)
        self.low = nn.Sequential(nn.Conv2d(low_ch, low_proj, 1, bias=False), nn.BatchNorm2d(low_proj), nn.ReLU(inplace=True))
        self.fuse = nn.Sequential(
            nn.Conv2d(aspp_ch + low_proj, aspp_ch, 3, padding=1, bias=False), nn.BatchNorm2d(aspp_ch), nn.ReLU(inplace=True),
            nn.Conv2d(aspp_ch, aspp_ch, 3, padding=1, bias=False), nn.BatchNorm2d(aspp_ch), nn.ReLU(inplace=True),
        )
        self.cls = nn.Conv2d(aspp_ch, num_classes, 1)

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        y = self.aspp(high)
        y = F.interpolate(y, size=low.shape[-2:], mode="bilinear", align_corners=False)
        y = self.fuse(torch.cat([y, self.low(low)], 1))
        return self.cls(y)


class DeepLabV3Plus(nn.Module):
    """Generic DeepLabV3+; ``encoder(x)`` must return [low(stride 4), high(stride 16)]."""

    def __init__(self, encoder: nn.Module, low_ch: int, high_ch: int, num_classes: int,
                 aspp_ch: int = 256, rates: Sequence[int] = (6, 12, 18), aux: bool = False):
        super().__init__()
        self.encoder = encoder
        self.head = DeepLabV3PlusHead(high_ch, low_ch, num_classes, aspp_ch, rates)
        self.aux = nn.Sequential(nn.Conv2d(high_ch, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128),
                                 nn.ReLU(inplace=True), nn.Conv2d(128, num_classes, 1)) if aux else None

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        size = x.shape[-2:]
        low, high = self.encoder(x)
        out = {"out": F.interpolate(self.head(low, high), size=size, mode="bilinear", align_corners=False)}
        if self.training and self.aux is not None:
            out["aux"] = F.interpolate(self.aux(high), size=size, mode="bilinear", align_corners=False)
        return out

    def param_groups(self):
        enc = list(self.encoder.parameters())
        ids = {id(p) for p in enc}
        return enc, [p for p in self.parameters() if id(p) not in ids]


class _TimmEncoder(nn.Module):
    def __init__(self, name: str, pretrained: bool, output_stride: int = 16):
        super().__init__()
        import timm

        self.body = timm.create_model(name, features_only=True, pretrained=pretrained,
                                      output_stride=output_stride, out_indices=(1, 4))
        self.channels = self.body.feature_info.channels()

    def forward(self, x):
        low, high = self.body(x)
        return [low, high]


class _MobileNetV3Encoder(nn.Module):
    """Dilated MobileNetV3-Large; taps stride-4 (24ch) and stride-16 (160ch) features."""

    def __init__(self, pretrained: bool, small: bool = False):
        super().__init__()
        from torchvision.models import (
            MobileNet_V3_Large_Weights,
            MobileNet_V3_Small_Weights,
            mobilenet_v3_large,
            mobilenet_v3_small,
        )

        if small:
            w = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            feats = mobilenet_v3_small(weights=w, dilated=True).features
            self.low_idx, self.channels = 1, [16, 96]  # stride 4 / stride 16 (dilated)
            self.stages = nn.ModuleList([feats[: self.low_idx + 1], feats[self.low_idx + 1: 12]])
        else:
            w = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
            feats = mobilenet_v3_large(weights=w, dilated=True).features
            self.low_idx, self.channels = 3, [24, 160]
            self.stages = nn.ModuleList([feats[: self.low_idx + 1], feats[self.low_idx + 1: 16]])

    def forward(self, x):
        low = self.stages[0](x)
        return [low, self.stages[1](low)]


def _tv_deeplab(depth: int) -> Callable[..., nn.Module]:
    def build(num_classes: int, pretrained: bool = True, aux_loss: bool = True, **_: object) -> nn.Module:
        from torchvision.models import ResNet50_Weights, ResNet101_Weights
        from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101

        fn = deeplabv3_resnet50 if depth == 50 else deeplabv3_resnet101
        wb = (ResNet50_Weights.IMAGENET1K_V2 if depth == 50 else ResNet101_Weights.IMAGENET1K_V2) if pretrained else None
        return _Wrap(fn(weights=None, weights_backbone=wb, num_classes=num_classes, aux_loss=aux_loss))

    return build


def _timm_plus(num_classes: int, encoder: Optional[str] = None, pretrained: bool = True,
               output_stride: int = 16, aux_loss: bool = True, **_: object) -> nn.Module:
    enc = _TimmEncoder(encoder or "resnet50", pretrained, output_stride)
    rates = (6, 12, 18) if output_stride == 16 else (12, 24, 36)
    return DeepLabV3Plus(enc, enc.channels[0], enc.channels[1], num_classes, 256, rates, aux=aux_loss)


def _mobile_plus(num_classes: int, pretrained: bool = True, aux_loss: bool = False, small: bool = False, **_: object) -> nn.Module:
    enc = _MobileNetV3Encoder(pretrained, small=small)
    return DeepLabV3Plus(enc, enc.channels[0], enc.channels[1], num_classes, aspp_ch=128 if not small else 96,
                         rates=(4, 8, 12), aux=aux_loss)


MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {
    "deeplabv3_resnet50": _tv_deeplab(50),
    "deeplabv3_resnet101": _tv_deeplab(101),
    "deeplabv3plus_timm": _timm_plus,
    "deeplabv3plus_mobilenet": _mobile_plus,
    "lite": _mobile_plus,
    "nano": lambda num_classes, **kw: _mobile_plus(num_classes, small=True, **kw),
}


def build_model(arch: str, num_classes: int, **kwargs) -> nn.Module:
    if arch not in MODEL_REGISTRY:
        raise ValueError(f"unknown arch '{arch}'. available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[arch](num_classes=num_classes, **kwargs)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def param_groups(model: nn.Module, lr: float, backbone_mult: float, wd: float) -> List[dict]:
    core = model.module if hasattr(model, "module") else model
    if hasattr(core, "param_groups"):
        bb, head = core.param_groups()
        bb = list(bb)
    else:  # pragma: no cover
        bb, head = [], list(core.parameters())

    def split(params):
        decay, no_decay = [], []
        for p in params:
            if not p.requires_grad:
                continue
            (no_decay if p.ndim <= 1 else decay).append(p)
        return decay, no_decay

    groups = []
    for params, mult in ((bb, backbone_mult), (head, 1.0)):
        d, nd = split(params)
        if d:
            groups.append({"params": d, "lr": lr * mult, "weight_decay": wd, "lr_mult": mult})
        if nd:
            groups.append({"params": nd, "lr": lr * mult, "weight_decay": 0.0, "lr_mult": mult})
    return groups
