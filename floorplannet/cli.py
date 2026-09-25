"""FloorPlanNet command line.

    floorplannet train     -c configs/cubicasa5k_deeplabv3_r101.yaml [key.sub=value ...]
    floorplannet evaluate  -c CONFIG --ckpt runs/x/best.pt [--split test] [--tta]
    floorplannet predict   --ckpt best.pt|model.onnx IMAGE [IMAGE ...] --out out/ [--svg --dxf --geojson]
    floorplannet export    --ckpt best.pt --out model.onnx [--fp16] [--size 512]
    floorplannet vectorize MASK.png --out plan.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_torch(ckpt: str, device: str = "cpu"):
    import torch

    from .models import build_model

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = ck["config"]["model"]
    model = build_model(m["arch"], ck["num_classes"], encoder=m.get("encoder"), pretrained=False,
                        output_stride=m.get("output_stride", 16), aux_loss=m.get("aux_loss", False))
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    bad = [k for k in missing if not k.startswith(("aux", "net.aux_classifier"))]
    if bad:
        raise RuntimeError(f"checkpoint mismatch, missing keys: {bad[:5]}")
    return model.eval().to(device), ck


def cmd_train(a) -> None:
    from .config import load_config
    from .engine.trainer import Trainer

    cfg = load_config(a.config, a.overrides)
    tr = Trainer(cfg)
    if a.resume:
        tr.resume(a.resume)
    print(json.dumps(tr.fit(), indent=2))


def cmd_evaluate(a) -> None:
    import torch

    from .classes import get_taxonomy
    from .config import load_config
    from .data.datasets import CubiCasa5K, FolderDataset, SyntheticPlans
    from .inference.predictor import Predictor, PredictorConfig
    from .metrics import ConfusionMatrix, format_metrics

    cfg = load_config(a.config, a.overrides)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = _load_torch(a.ckpt, dev)
    d = cfg.data
    if d.dataset == "cubicasa5k":
        ds = CubiCasa5K(d.root, a.split, d.taxonomy, d.image_file, d.max_side)
    elif d.dataset == "folder":
        ds = FolderDataset(d.root, a.split, d.max_side)
    else:
        ds = SyntheticPlans(d.synthetic_val, seed=10_000_000 + cfg.seed, size=d.max_side)
    tax = get_taxonomy(d.taxonomy)
    pred = Predictor(model, ck["num_classes"], dev, PredictorConfig(tile=a.tile, tta=a.tta, max_side=10 ** 5))
    cm = ConfusionMatrix(ck["num_classes"], cfg.loss.ignore_index)
    from .data.transforms import IMAGENET_MEAN, IMAGENET_STD

    for i in range(len(ds)):
        x, y = ds[i]
        img = ((x.numpy().transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN) * 255).clip(0, 255).astype("uint8")
        cm.update_numpy(pred.predict(img), y.numpy())
    res = cm.compute(list(tax.classes)[: ck["num_classes"]])
    print(format_metrics(res))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2))


def cmd_predict(a) -> None:
    import cv2
    import numpy as np

    from .classes import get_taxonomy
    from .inference.predictor import OnnxPredictor, Predictor, PredictorConfig
    from .vectorize import VectorizeConfig, vectorize

    pc = PredictorConfig(tile=a.tile, tta=a.tta, max_side=a.max_side)
    tax = get_taxonomy(a.taxonomy)
    if a.ckpt.endswith(".onnx"):
        pred = OnnxPredictor(a.ckpt, tax.num_classes, pc)
    else:
        import torch

        model, ck = _load_torch(a.ckpt, "cuda" if torch.cuda.is_available() else "cpu")
        pred = Predictor(model, ck["num_classes"], str(next(model.parameters()).device), pc)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pal = np.array(tax.colors, np.uint8)
    for path in a.images:
        img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        probs, scale = pred.predict_proba(img)
        label = probs.argmax(0).astype(np.uint8)
        geo = vectorize(label, tax, VectorizeConfig(px_per_meter=a.px_per_meter / scale if a.px_per_meter else None),
                        image_size=(img.shape[1], img.shape[0]))
        stem = Path(path).stem
        full = cv2.resize(label, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(out / f"{stem}_mask.png"), full)
        overlay = (0.55 * img + 0.45 * pal[full]).astype(np.uint8)
        cv2.imwrite(str(out / f"{stem}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        (out / f"{stem}.json").write_text(geo.to_json(indent=1))
        if a.svg:
            (out / f"{stem}.svg").write_text(geo.to_svg())
        if a.geojson:
            (out / f"{stem}.geojson").write_text(json.dumps(geo.to_geojson()))
        if a.dxf:
            geo.to_dxf(str(out / f"{stem}.dxf"))
        print(stem, geo.summary())


def cmd_export(a) -> None:
    from .inference.onnx_export import export_onnx

    model, ck = _load_torch(a.ckpt)
    info = export_onnx(model, a.out, size=a.size, fp16=a.fp16, opset=a.opset,
                       meta={"classes": ck["class_names"], "arch": ck["config"]["model"]["arch"],
                             "metrics": ck.get("metrics")})
    print(json.dumps(info, indent=2))


def cmd_vectorize(a) -> None:
    import cv2

    from .classes import get_taxonomy
    from .vectorize import vectorize

    label = cv2.imread(a.mask, cv2.IMREAD_GRAYSCALE)
    geo = vectorize(label, get_taxonomy(a.taxonomy))
    Path(a.out).write_text(geo.to_json(indent=1))
    if a.svg:
        Path(a.out).with_suffix(".svg").write_text(geo.to_svg())
    print(geo.summary())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="floorplannet")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("-c", "--config")
    t.add_argument("--resume")
    t.add_argument("overrides", nargs="*")
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser("evaluate")
    e.add_argument("-c", "--config")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--split", default="test")
    e.add_argument("--tile", type=int, default=512)
    e.add_argument("--tta", action="store_true")
    e.add_argument("--json")
    e.add_argument("overrides", nargs="*")
    e.set_defaults(fn=cmd_evaluate)

    p = sub.add_parser("predict")
    p.add_argument("images", nargs="+")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="predictions")
    p.add_argument("--taxonomy", default="coarse")
    p.add_argument("--tile", type=int, default=512)
    p.add_argument("--max-side", type=int, default=2048)
    p.add_argument("--tta", action="store_true")
    p.add_argument("--px-per-meter", type=float)
    p.add_argument("--svg", action="store_true")
    p.add_argument("--geojson", action="store_true")
    p.add_argument("--dxf", action="store_true")
    p.set_defaults(fn=cmd_predict)

    x = sub.add_parser("export")
    x.add_argument("--ckpt", required=True)
    x.add_argument("--out", default="floorplannet.onnx")
    x.add_argument("--size", type=int, default=512)
    x.add_argument("--opset", type=int, default=17)
    x.add_argument("--fp16", action="store_true")
    x.set_defaults(fn=cmd_export)

    v = sub.add_parser("vectorize")
    v.add_argument("mask")
    v.add_argument("--out", default="plan.json")
    v.add_argument("--taxonomy", default="coarse")
    v.add_argument("--svg", action="store_true")
    v.set_defaults(fn=cmd_vectorize)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main(sys.argv[1:])
