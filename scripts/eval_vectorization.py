"""Instance-level evaluation of the full pipeline (segmentation + vectorisation).

Pixel mIoU does not say whether a plan comes out as the right rooms and openings.
This script vectorises both the prediction and the ground-truth mask and reports:

* room detection precision / recall / F1 (greedy matching at polygon IoU >= 0.5)
* mean IoU of matched room polygons, exact room-count rate
* door and window detection F1 (centre within max(width / 2, 2 * wall thickness))

    python scripts/eval_vectorization.py --model models/floorplannet_lite.onnx --data data/synthetic --n 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from floorplannet.inference.predictor import OnnxPredictor, PredictorConfig  # noqa: E402
from floorplannet.vectorize import vectorize  # noqa: E402


def _raster(poly, shape):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(np.array(poly)).astype(np.int32)], 1)
    return m.astype(bool)


def match_rooms(pred, gt, shape, thr=0.5):
    P = [_raster(r.polygon, shape) for r in pred]
    G = [_raster(r.polygon, shape) for r in gt]
    pairs = []
    for i, p in enumerate(P):
        for j, g in enumerate(G):
            inter = (p & g).sum()
            if inter:
                pairs.append((inter / (p | g).sum(), i, j))
    pairs.sort(reverse=True)
    used_p, used_g, ious = set(), set(), []
    for iou, i, j in pairs:
        if iou < thr:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        ious.append(iou)
    return len(ious), len(P), len(G), ious


def match_openings(pred, gt, kind, t):
    P = [o for o in pred if o.kind == kind]
    G = [o for o in gt if o.kind == kind]
    used, tp = set(), 0
    for g in G:
        tol = max(g.width / 2, 2 * t)
        best, bd = None, tol
        for i, p in enumerate(P):
            if i in used:
                continue
            d = np.hypot(p.center[0] - g.center[0], p.center[1] - g.center[1])
            if d <= bd:
                best, bd = i, d
        if best is not None:
            used.add(best)
            tp += 1
    return tp, len(P), len(G)


def f1(tp, np_, ng):
    p = tp / np_ if np_ else 1.0
    r = tp / ng if ng else 1.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r) if p + r else 0.0, 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--json")
    a = ap.parse_args()
    pred = OnnxPredictor(a.model, 5, PredictorConfig(tile=512, max_side=512))
    root = Path(a.data) / a.split
    rooms = [0, 0, 0]
    ious, exact = [], 0
    ops = {"door": [0, 0, 0], "window": [0, 0, 0]}
    imgs = sorted((root / "images").glob("*.png"))[: a.n]
    for k, p in enumerate(imgs):
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        gt_mask = cv2.imread(str(root / "masks" / p.name), cv2.IMREAD_GRAYSCALE)
        g_pred = vectorize(pred.predict(img))
        g_gt = vectorize(gt_mask)
        tp, n_p, n_g, iu = match_rooms(g_pred.rooms, g_gt.rooms, gt_mask.shape)
        rooms = [rooms[0] + tp, rooms[1] + n_p, rooms[2] + n_g]
        ious += iu
        exact += int(n_p == n_g)
        t = float(g_gt.meta["wall_thickness_px"])
        for kind in ops:
            r = match_openings(g_pred.openings, g_gt.openings, kind, t)
            ops[kind] = [x + y for x, y in zip(ops[kind], r)]
        if (k + 1) % 50 == 0:
            print(f"{k + 1}/{len(imgs)}", flush=True)
    res = {
        "plans": len(imgs),
        "rooms": {**f1(*rooms), "matched_mean_iou": round(float(np.mean(ious)) if ious else 0.0, 4),
                  "exact_count_rate": round(exact / max(1, len(imgs)), 4)},
        "doors": f1(*ops["door"]),
        "windows": f1(*ops["window"]),
    }
    print(json.dumps(res, indent=2))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
