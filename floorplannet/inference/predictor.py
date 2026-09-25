"""Tiled, TTA-capable inference for arbitrarily large floor plans.

Large plans (CubiCasa originals reach 6000 px) are processed with overlapping
windows blended by a 2-D Gaussian importance map, which removes the seam
artefacts that otherwise break wall continuity at tile borders.

Tiling / blending is NumPy-only so the ONNX serving image does not need torch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..data.transforms import resize_long_side, to_tensor_arrays


def gaussian_window(h: int, w: int, sigma_scale: float = 0.25) -> np.ndarray:
    ys = np.arange(h, dtype=np.float32) - (h - 1) / 2
    xs = np.arange(w, dtype=np.float32) - (w - 1) / 2
    g = np.exp(-(ys[:, None] ** 2) / (2 * (sigma_scale * h) ** 2)) * np.exp(-(xs[None] ** 2) / (2 * (sigma_scale * w) ** 2))
    return np.maximum(g / g.max(), 1e-3).astype(np.float32)


def _starts(length: int, tile: int, stride: int) -> List[int]:
    if length <= tile:
        return [0]
    s = list(range(0, length - tile, stride))
    s.append(length - tile)
    return s


def _softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


@dataclass
class PredictorConfig:
    tile: int = 512
    overlap: float = 0.25
    max_side: int = 2048
    tta: bool = False
    batch_tiles: int = 4


class BasePredictor:
    """Subclasses implement ``_logits(batch[N,3,h,w] float32) -> [N,K,h,w]``."""

    k: int
    cfg: PredictorConfig

    def _logits(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def _probs(self, x: np.ndarray) -> np.ndarray:
        p = _softmax(self._logits(x))
        if not self.cfg.tta:
            return p
        p = p + _softmax(self._logits(x[..., ::-1].copy()))[..., ::-1]
        p = p + _softmax(self._logits(x[..., ::-1, :].copy()))[..., ::-1, :]
        if x.shape[2] == x.shape[3]:
            r = np.rot90(x, 1, (2, 3)).copy()
            p = p + np.rot90(_softmax(self._logits(r)), -1, (2, 3))
            return p / 4
        return p / 3

    def predict_proba(self, img_rgb: np.ndarray) -> Tuple[np.ndarray, float]:
        """Return (probs[K,H,W] float32 at working resolution, working/input scale)."""
        h0 = max(img_rgb.shape[:2])
        img, _ = resize_long_side(img_rgb, None, self.cfg.max_side)
        scale = max(img.shape[:2]) / h0
        H, W = img.shape[:2]
        T = self.cfg.tile
        Hp = max(T, H + (-H) % 32) if H < T else H + (-H) % 32
        Wp = max(T, W + (-W) % 32) if W < T else W + (-W) % 32
        canvas = np.full((Hp, Wp, 3), 255, np.uint8)
        canvas[:H, :W] = img
        x = to_tensor_arrays(canvas)[None]
        th, tw = min(T, Hp), min(T, Wp)
        sh, sw = max(1, int(th * (1 - self.cfg.overlap))), max(1, int(tw * (1 - self.cfg.overlap)))
        win = gaussian_window(th, tw)
        acc = np.zeros((self.k, Hp, Wp), np.float32)
        norm = np.zeros((1, Hp, Wp), np.float32)
        coords = [(y, xx) for y in _starts(Hp, th, sh) for xx in _starts(Wp, tw, sw)]
        for i in range(0, len(coords), self.cfg.batch_tiles):
            chunk = coords[i:i + self.cfg.batch_tiles]
            batch = np.concatenate([x[:, :, y:y + th, xx:xx + tw] for y, xx in chunk]).astype(np.float32)
            probs = self._probs(batch)
            for p, (y, xx) in zip(probs, chunk):
                acc[:, y:y + th, xx:xx + tw] += p * win
                norm[:, y:y + th, xx:xx + tw] += win
        return (acc / np.maximum(norm, 1e-6))[:, :H, :W], scale

    def predict(self, img_rgb: np.ndarray, full_resolution: bool = True) -> np.ndarray:
        probs, scale = self.predict_proba(img_rgb)
        label = probs.argmax(0).astype(np.uint8)
        if full_resolution and scale != 1.0:
            label = cv2.resize(label, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        return label


class Predictor(BasePredictor):
    """PyTorch backend."""

    def __init__(self, model, num_classes: int, device: str = "cpu", cfg: Optional[PredictorConfig] = None):
        import torch

        self.torch = torch
        self.model = model.eval().to(device)
        self.k, self.device, self.cfg = num_classes, torch.device(device), cfg or PredictorConfig()

    def _logits(self, x: np.ndarray) -> np.ndarray:
        with self.torch.inference_mode():
            t = self.torch.from_numpy(np.ascontiguousarray(x)).to(self.device)
            return self.model(t)["out"].float().cpu().numpy()


class OnnxPredictor(BasePredictor):
    """onnxruntime backend (used by the Docker API; no torch required)."""

    def __init__(self, onnx_path: str, num_classes: int, cfg: Optional[PredictorConfig] = None,
                 providers: Optional[List[str]] = None):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(onnx_path, so, providers=providers or ort.get_available_providers())
        self.input_name = self.sess.get_inputs()[0].name
        self.k, self.cfg = num_classes, cfg or PredictorConfig()

    def _logits(self, x: np.ndarray) -> np.ndarray:
        return self.sess.run(None, {self.input_name: np.ascontiguousarray(x, np.float32)})[0]


__all__ = ["BasePredictor", "Predictor", "OnnxPredictor", "PredictorConfig", "gaussian_window"]
