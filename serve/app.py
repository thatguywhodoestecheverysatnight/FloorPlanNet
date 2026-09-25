"""FloorPlanNet inference API (FastAPI).

Env vars
--------
FLOORPLANNET_MODEL     path to .onnx (recommended) or .pt checkpoint   [models/floorplannet_lite.onnx]
FLOORPLANNET_TAXONOMY  coarse | cubicasa12                              [coarse]
FLOORPLANNET_TILE      sliding-window tile size                         [512]
FLOORPLANNET_MAX_SIDE  long-side cap before inference                   [2048]
ALLOWED_ORIGINS        comma separated CORS origins                     [*]

Run:  uvicorn serve.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import base64
import json
import os
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from floorplannet import __version__
from floorplannet.classes import get_taxonomy
from floorplannet.inference.predictor import OnnxPredictor, Predictor, PredictorConfig
from floorplannet.vectorize import FloorPlanGeometry, VectorizeConfig, vectorize

MAX_UPLOAD = 25 * 2 ** 20
app = FastAPI(title="FloorPlanNet API", version=__version__,
              description="Floor plan parsing: DeepLabV3 segmentation + OpenCV vectorisation.")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])


@lru_cache(maxsize=1)
def get_predictor():
    root = Path(__file__).resolve().parents[1]
    default = next((str(p) for p in (root / "models" / "floorplannet_lite.onnx",
                                      root / "web" / "public" / "models" / "floorplannet_lite.onnx") if p.exists()),
                   str(root / "models" / "floorplannet_lite.onnx"))
    path = os.getenv("FLOORPLANNET_MODEL", default)
    tax = get_taxonomy(os.getenv("FLOORPLANNET_TAXONOMY", "coarse"))
    cfg = PredictorConfig(tile=int(os.getenv("FLOORPLANNET_TILE", 512)), max_side=int(os.getenv("FLOORPLANNET_MAX_SIDE", 2048)))
    if not Path(path).exists():
        raise RuntimeError(f"model not found at {path}; set FLOORPLANNET_MODEL")
    if path.endswith(".onnx"):
        pred = OnnxPredictor(path, tax.num_classes, cfg)
    else:
        import torch

        from floorplannet.cli import _load_torch

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model, _ = _load_torch(path, dev)
        pred = Predictor(model, tax.num_classes, dev, cfg)
    return pred, tax, path


def _decode(data: bytes, flags=cv2.IMREAD_COLOR) -> np.ndarray:
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "file too large (25 MB max)")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), flags)
    if img is None:
        raise HTTPException(400, "could not decode image (png / jpg / webp / bmp / tiff supported)")
    return img


def _png_b64(arr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", arr)
    return base64.b64encode(buf.tobytes()).decode()


@app.get("/health")
def health() -> Dict[str, Any]:
    try:
        pred, tax, path = get_predictor()
        return {"status": "ok", "version": __version__, "model": Path(path).name, "classes": list(tax.classes)}
    except RuntimeError as exc:
        return {"status": "degraded", "version": __version__, "error": str(exc)}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), tta: bool = Form(False), max_side: Optional[int] = Form(None),
                  px_per_meter: Optional[float] = Form(None), return_mask: bool = Form(True)) -> JSONResponse:
    try:
        pred, tax, _ = get_predictor()
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    img = cv2.cvtColor(_decode(await file.read()), cv2.COLOR_BGR2RGB)
    pred.cfg.tta = bool(tta)
    if max_side:
        pred.cfg.max_side = int(np.clip(max_side, 256, 4096))
    t0 = time.perf_counter()
    probs, scale = pred.predict_proba(img)
    t1 = time.perf_counter()
    label = probs.argmax(0).astype(np.uint8)
    conf = float(probs.max(0).mean())
    geo = vectorize(label, tax, VectorizeConfig(px_per_meter=px_per_meter / scale if px_per_meter else None),
                    image_size=(img.shape[1], img.shape[0]))
    t2 = time.perf_counter()
    body: Dict[str, Any] = {
        "width": img.shape[1], "height": img.shape[0], "classes": list(tax.classes),
        "geometry": geo.to_dict(), "summary": geo.summary(), "mean_confidence": round(conf, 4),
        "timings_ms": {"segmentation": round((t1 - t0) * 1e3, 1), "vectorization": round((t2 - t1) * 1e3, 1)},
    }
    if return_mask:
        body["mask_png"] = _png_b64(label)  # working resolution; class index per pixel
        body["mask_scale"] = scale
    return JSONResponse(body)


@app.post("/api/vectorize")
async def vectorize_mask(file: UploadFile = File(...), taxonomy: str = Form("coarse"),
                         px_per_meter: Optional[float] = Form(None)) -> JSONResponse:
    label = _decode(await file.read(), cv2.IMREAD_GRAYSCALE)
    geo = vectorize(label, get_taxonomy(taxonomy), VectorizeConfig(px_per_meter=px_per_meter))
    return JSONResponse({"geometry": geo.to_dict(), "summary": geo.summary()})


@app.post("/api/export/{fmt}")
async def export(fmt: str, geometry: Dict[str, Any]) -> Response:
    geo = FloorPlanGeometry.from_dict(geometry)
    if fmt == "svg":
        return Response(geo.to_svg(), media_type="image/svg+xml")
    if fmt == "geojson":
        return Response(json.dumps(geo.to_geojson()), media_type="application/geo+json")
    if fmt == "json":
        return Response(geo.to_json(), media_type="application/json")
    if fmt == "dxf":
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "plan.dxf")
            geo.to_dxf(p)
            data = Path(p).read_bytes()
        return Response(data, media_type="application/dxf",
                        headers={"Content-Disposition": 'attachment; filename="floorplan.dxf"'})
    raise HTTPException(400, "format must be one of svg, geojson, json, dxf")
