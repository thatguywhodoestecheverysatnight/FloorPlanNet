"""Export a checkpoint for the web app and the API, and write its model card.

    python scripts/export_web_model.py --ckpt runs/synthetic_lite/best.pt [--int8] [--data data/synthetic]

Writes:
    models/floorplannet_lite.onnx                (API image)
    web/public/models/floorplannet_lite.onnx     (browser)
    web/public/models/model-card.json            (shown on the site)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from floorplannet.cli import _load_torch  # noqa: E402
from floorplannet.inference.onnx_export import export_onnx, quantize_int8  # noqa: E402
from floorplannet.models import count_params  # noqa: E402


def evaluate_onnx(path: str, data_root: Path, n: int = 200) -> dict:
    import cv2

    from floorplannet.classes import COARSE
    from floorplannet.inference.predictor import OnnxPredictor, PredictorConfig
    from floorplannet.metrics import ConfusionMatrix

    pred = OnnxPredictor(path, 5, PredictorConfig(tile=512, max_side=512))
    cm = ConfusionMatrix(5)
    imgs = sorted((data_root / "val" / "images").glob("*.png"))[:n]
    for p in imgs:
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        gt = cv2.imread(str(data_root / "val" / "masks" / p.name), cv2.IMREAD_GRAYSCALE)
        cm.update_numpy(pred.predict(img), gt)
    return cm.compute(COARSE.classes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--int8", action="store_true", help="ship the INT8 model if it stays within 1 mIoU point")
    a = ap.parse_args()

    model, ck = _load_torch(a.ckpt)
    out_dir = ROOT / "models"
    out_dir.mkdir(exist_ok=True)
    fp32 = out_dir / "floorplannet_lite.onnx"
    info = export_onnx(model, str(fp32), size=512, meta={"classes": ck["class_names"], "arch": ck["config"]["model"]["arch"]})
    print("fp32:", info)
    data = Path(a.data)
    m32 = evaluate_onnx(str(fp32), data) if (data / "val").exists() else ck.get("metrics")
    chosen, metrics = fp32, m32
    if a.int8 and (data / "train").exists():
        import cv2

        from floorplannet.data.transforms import to_tensor_arrays

        calib = [to_tensor_arrays(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB))
                 for p in sorted((data / "train" / "images").glob("*.png"))[:48]]
        q = out_dir / "floorplannet_lite.int8.onnx"
        print("int8:", quantize_int8(str(fp32), str(q), calib))
        m8 = evaluate_onnx(str(q), data)
        print(f"mIoU fp32 {m32['mIoU']:.4f} | int8 {m8['mIoU']:.4f}")
        if m32["mIoU"] - m8["mIoU"] <= 0.01:
            chosen, metrics = q, m8
    web = ROOT / "web" / "public" / "models"
    web.mkdir(parents=True, exist_ok=True)
    shutil.copy(chosen, web / "floorplannet_lite.onnx")
    if chosen != fp32:
        shutil.copy(chosen, fp32)
    log = [json.loads(line) for line in (Path(a.ckpt).parent / "log.jsonl").read_text().splitlines()] if (Path(a.ckpt).parent / "log.jsonl").exists() else []
    card = {
        "name": "floorplannet-lite",
        "arch": "DeepLabV3+ (ASPP, stride-4 decoder), MobileNetV3-Large dilated encoder",
        "params_m": round(count_params(model) / 1e6, 2),
        "size_mb": round((web / "floorplannet_lite.onnx").stat().st_size / 2 ** 20, 2),
        "precision": "int8" if chosen != fp32 else "fp32",
        "input": "RGB, long side 512 px, ImageNet normalisation",
        "classes": ck["class_names"],
        "trained_on": "procedurally generated floor plans (scripts/make_synthetic.py), no ImageNet pretraining",
        "iterations": ck["iter"],
        "val": {k: metrics[k] for k in ("mIoU", "pixel_acc", "per_class_iou")} if metrics else None,
        "curve": [{"iter": r["iter"], "mIoU": r["eval"]["mIoU"]} for r in log if "eval" in r],
        "notes": [
            "Validated on held-out synthetic plans, not on CubiCasa5K. Expect lower accuracy on real scans until you fine-tune.",
            "For production accuracy train configs/cubicasa5k_deeplabv3_r101.yaml on a GPU and serve it through the API.",
            "Best on clean, roughly axis-aligned plans with dark walls. Very faint or hand-drawn lines may be missed.",
        ],
    }
    (web / "model-card.json").write_text(json.dumps(card, indent=2))
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()
