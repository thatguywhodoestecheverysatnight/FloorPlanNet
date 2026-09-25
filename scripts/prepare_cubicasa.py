"""Rasterise every CubiCasa5K ``model.svg`` to a cached class-index PNG and print class stats.

    python scripts/prepare_cubicasa.py --root data/cubicasa5k --taxonomy coarse --workers 8
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from floorplannet.classes import get_taxonomy  # noqa: E402
from floorplannet.data.cubicasa import iter_folders, load_sample  # noqa: E402


def _one(args):
    folder, tax_name = args
    tax = get_taxonomy(tax_name)
    try:
        _, mask = load_sample(Path(folder), tax, cache=True)
        return np.bincount(mask.ravel(), minlength=tax.num_classes)[: tax.num_classes]
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] {folder}: {exc}")
        return np.zeros(tax.num_classes, np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/cubicasa5k")
    ap.add_argument("--taxonomy", default="coarse")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    tax = get_taxonomy(a.taxonomy)
    folders = [(str(f), a.taxonomy) for f in iter_folders(a.root)]
    print(f"{len(folders)} plans")
    total = np.zeros(tax.num_classes, np.int64)
    with ProcessPoolExecutor(a.workers) as ex:
        for i, c in enumerate(ex.map(_one, folders, chunksize=8)):
            total += c
            if (i + 1) % 500 == 0:
                print(f"{i + 1}/{len(folders)}")
    freq = total / total.sum()
    w = 1 / np.sqrt(freq + 1e-6)
    w = w / w.mean()
    for n, f, wi in zip(tax.classes, freq, w):
        print(f"{n:>14s}  freq={f:.4f}  suggested_weight={wi:.2f}")


if __name__ == "__main__":
    main()
