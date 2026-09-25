"""CubiCasa5K loader.

Dataset layout (after ``scripts/download_cubicasa5k.sh``)::

    data/cubicasa5k/
        train.txt  val.txt  test.txt            # one folder per line, e.g. /high_quality_architectural/2003/
        high_quality/...  high_quality_architectural/...  colorful/...
            <id>/F1_scaled.png   <id>/F1_original.png   <id>/model.svg

``F1_scaled.png`` shares the coordinate frame of ``model.svg`` so polygons can be
rasterised directly. Masks are rasterised once and cached as PNG.
"""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..classes import Taxonomy, room_label

Matrix = np.ndarray  # 3x3 affine


@dataclass
class SvgPolygon:
    kind: str            # wall | door | window | room | railing
    subtype: str         # e.g. Kitchen, External, Swing
    points: np.ndarray   # (N, 2) float32 in image pixels


_NUM = r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?"


def parse_transform(value: Optional[str]) -> Matrix:
    """Parse an SVG ``transform`` attribute into a 3x3 matrix."""
    m = np.eye(3, dtype=np.float64)
    if not value:
        return m
    for name, args in re.findall(r"(\w+)\s*\(([^)]*)\)", value):
        a = [float(x) for x in re.findall(_NUM, args)]
        t = np.eye(3)
        if name == "matrix" and len(a) == 6:
            t = np.array([[a[0], a[2], a[4]], [a[1], a[3], a[5]], [0, 0, 1]])
        elif name == "translate":
            t[0, 2] = a[0]
            t[1, 2] = a[1] if len(a) > 1 else 0.0
        elif name == "scale":
            t[0, 0] = a[0]
            t[1, 1] = a[1] if len(a) > 1 else a[0]
        elif name == "rotate":
            r = math.radians(a[0])
            rot = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
            if len(a) == 3:
                cx, cy = a[1], a[2]
                pre = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
                post = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
                rot = pre @ rot @ post
            t = rot
        m = m @ t
    return m


def _parse_points(raw: str) -> np.ndarray:
    nums = [float(x) for x in re.findall(_NUM, raw)]
    if len(nums) < 6:
        return np.zeros((0, 2), np.float32)
    return np.asarray(nums[: len(nums) // 2 * 2], np.float32).reshape(-1, 2)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _classify(el: ET.Element) -> Optional[Tuple[str, str]]:
    cls = (el.get("class") or "").split()
    gid = el.get("id") or ""
    head = cls[0] if cls else gid
    sub = cls[1] if len(cls) > 1 else ""
    if head == "Wall" or gid == "Wall":
        return "wall", sub
    if head == "Door" or gid == "Door":
        return "door", sub
    if head == "Window" or gid == "Window":
        return "window", sub
    if head == "Space":
        return "room", sub or "Undefined"
    if head == "Railing" or gid == "Railing":
        return "railing", sub
    return None


def parse_cubicasa_svg(svg_path: str | Path) -> List[SvgPolygon]:
    """Extract wall / door / window / room polygons from a CubiCasa ``model.svg``."""
    tree = ET.parse(str(svg_path))
    out: List[SvgPolygon] = []

    def visit(el: ET.Element, ctm: Matrix) -> None:
        ctm = ctm @ parse_transform(el.get("transform"))
        if _local(el.tag) == "g":
            kind = _classify(el)
            # The first direct <polygon> child of a semantic group is its outline;
            # nested decoration groups (panels, glass, icons) never classify.
            # Doors/windows may be nested inside wall groups, so keep recursing.
            if kind is not None:
                for child in el:
                    if _local(child.tag) == "polygon":
                        pts = _parse_points(child.get("points", ""))
                        if len(pts) >= 3:
                            h = np.c_[pts, np.ones(len(pts))] @ ctm.T
                            out.append(SvgPolygon(kind[0], kind[1], h[:, :2].astype(np.float32)))
                        break
        for child in el:
            visit(child, ctm)

    visit(tree.getroot(), np.eye(3))
    return out


def rasterize(polys: Sequence[SvgPolygon], shape: Tuple[int, int], tax: Taxonomy) -> np.ndarray:
    """Paint polygons in z-order rooms -> walls/railings -> windows -> doors."""
    mask = np.zeros(shape, np.uint8)
    order = {"room": 0, "railing": 1, "wall": 2, "window": 3, "door": 4}
    for p in sorted(polys, key=lambda q: order[q.kind]):
        pts = np.round(p.points).astype(np.int32).reshape(-1, 1, 2)
        if p.kind == "room":
            val = room_label(p.subtype, tax)
        elif p.kind == "railing":
            val = room_label("Railing", tax)
        elif p.kind == "wall":
            val = tax.wall
        elif p.kind == "door":
            val = tax.door
        else:
            val = tax.window
        cv2.fillPoly(mask, [pts], int(val))
    return mask


def read_split(root: str | Path, split: str) -> List[Path]:
    root = Path(root)
    lines = (root / f"{split}.txt").read_text().split()
    return [root / ln.strip("/") for ln in lines if ln.strip()]


def load_sample(folder: Path, tax: Taxonomy, image_file: str = "F1_scaled.png",
                cache: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(str(folder / image_file), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(folder / image_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cache_file = folder / f".mask_{tax.name}.png"
    if cache and cache_file.exists():
        mask = cv2.imread(str(cache_file), cv2.IMREAD_GRAYSCALE)
    else:
        mask = rasterize(parse_cubicasa_svg(folder / "model.svg"), img.shape[:2], tax)
        if cache:
            cv2.imwrite(str(cache_file), mask)
    if mask.shape != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    return img, mask


def iter_folders(root: str | Path, splits: Iterable[str] = ("train", "val", "test")) -> Iterable[Path]:
    for s in splits:
        yield from read_split(root, s)
