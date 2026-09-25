import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from floorplannet.data.synthetic import FloorPlanGenerator  # noqa: E402


@pytest.fixture(scope="session")
def synth_samples():
    g = FloorPlanGenerator(seed=123)
    return [g.generate() for _ in range(4)]


@pytest.fixture(scope="session")
def tiny_onnx(tmp_path_factory):
    torch = pytest.importorskip("torch")
    from floorplannet.inference.onnx_export import export_onnx
    from floorplannet.models import build_model

    torch.manual_seed(0)
    model = build_model("nano", 5, pretrained=False, aux_loss=False)
    path = tmp_path_factory.mktemp("onnx") / "tiny.onnx"
    export_onnx(model, str(path), size=128, check=False)
    return str(path)


def rgb(img_bgr: np.ndarray) -> np.ndarray:
    return img_bgr[:, :, ::-1].copy()
