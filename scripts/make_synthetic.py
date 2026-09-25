"""Pre-render a synthetic floor plan dataset in FolderDataset layout.

    python scripts/make_synthetic.py --out data/synthetic --train 4000 --val 400 --workers 8
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from floorplannet.data.synthetic import FloorPlanGenerator, SynthConfig  # noqa: E402


def _render(args):
    out, split, idx, seed, size = args
    img, mask = FloorPlanGenerator(SynthConfig(size=size), seed=seed).generate()
    name = f"{idx:06d}.png"
    cv2.imwrite(str(out / split / "images" / name), img)
    cv2.imwrite(str(out / split / "masks" / name), mask)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--train", type=int, default=4000)
    ap.add_argument("--val", type=int, default=400)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out)
    jobs = []
    for split, n, base in (("train", a.train, a.seed), ("val", a.val, a.seed + 10_000_000)):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "masks").mkdir(parents=True, exist_ok=True)
        jobs += [(out, split, i, base + i, a.size) for i in range(n)]
    with ProcessPoolExecutor(a.workers) as ex:
        for i, _ in enumerate(ex.map(_render, jobs, chunksize=16)):
            if (i + 1) % 500 == 0:
                print(f"rendered {i + 1}/{len(jobs)}", flush=True)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
