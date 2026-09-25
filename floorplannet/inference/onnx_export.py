"""ONNX export (dynamic H/W), parity check, and optional INT8 static quantisation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch


class _LogitsOnly(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["out"]


def export_onnx(model: torch.nn.Module, out_path: str, size: int = 512, fp16: bool = False, opset: int = 17,
                meta: Optional[Dict[str, object]] = None, check: bool = True) -> Dict[str, object]:
    import onnx

    model = model.eval().cpu()
    wrapper = _LogitsOnly(model).eval()
    dummy = torch.randn(1, 3, size, size)
    out_path = str(out_path)
    torch.onnx.export(wrapper, (dummy,), out_path, input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch", 2: "height", 3: "width"},
                                    "logits": {0: "batch", 2: "height", 3: "width"}},
                      opset_version=opset, do_constant_folding=True, dynamo=False)
    m = onnx.load(out_path)
    if fp16:
        from onnxconverter_common import float16  # optional dependency

        m = float16.convert_float_to_float16(m, keep_io_types=True)
    for k, v in (meta or {}).items():
        entry = m.metadata_props.add()
        entry.key, entry.value = k, json.dumps(v)
    onnx.save(m, out_path)
    info: Dict[str, object] = {"path": out_path, "size_mb": round(Path(out_path).stat().st_size / 2 ** 20, 2)}
    if check:
        info["max_abs_diff"] = parity(wrapper, out_path, size)
    return info


def parity(torch_model: torch.nn.Module, onnx_path: str, size: int = 384) -> float:
    import onnxruntime as ort

    x = torch.randn(1, 3, size, size + 64)
    with torch.no_grad():
        ref = torch_model(x).numpy()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"image": x.numpy()})[0]
    return float(np.abs(ref - got).max())


class _Calib:
    def __init__(self, arrays: Iterable[np.ndarray]):
        self.it = iter([{"image": a[None].astype(np.float32)} for a in arrays])

    def get_next(self):
        return next(self.it, None)


def quantize_int8(fp32_path: str, out_path: str, calib_arrays: Iterable[np.ndarray]) -> Dict[str, object]:
    """Static QDQ INT8 (per-channel weights). ~4x smaller; verify mIoU before shipping."""
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.c = _Calib(calib_arrays)

        def get_next(self):
            return self.c.get_next()

    pre = str(Path(out_path).with_suffix(".pre.onnx"))
    quant_pre_process(fp32_path, pre)
    quantize_static(pre, out_path, Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
    Path(pre).unlink(missing_ok=True)
    return {"path": out_path, "size_mb": round(Path(out_path).stat().st_size / 2 ** 20, 2)}
