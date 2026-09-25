"""Raster-to-vector conversion for floor plan segmentation masks.

Pipeline
--------
1. ``clean_mask``     remove speckle components, refill them from nearest valid label.
2. ``extract_walls``  structural mask (wall + openings) -> skeleton -> pixel graph ->
                      polylines -> Douglas-Peucker -> Manhattan snapping -> collinear
                      merge -> T/L junction closure. Thickness from the distance transform.
3. ``extract_rooms``  per-class connected components -> contours (with holes) ->
                      Douglas-Peucker -> rectilinear orthogonalisation.
4. ``extract_openings`` door / window components -> oriented min-area rectangles ->
                      axis snapping -> host wall assignment.

All tolerances scale with the median wall thickness, so the same config works
for a 512 px thumbnail and a 4000 px scan.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi

from ..classes import COARSE, Taxonomy
from .geometry import FloorPlanGeometry, Opening, Point, Room, Wall


@dataclass
class VectorizeConfig:
    min_room_area: int = 200          # px^2 at working resolution
    min_opening_area: int = 10
    min_speckle: int = 12
    rdp_room: float = 0.35            # x wall thickness
    rdp_wall: float = 0.35
    snap_angle_deg: float = 12.0
    merge_offset: float = 0.9         # x thickness: collinear walls closer than this merge
    merge_gap: float = 1.2            # x thickness: bridge gaps shorter than this
    junction_tol: float = 1.6         # x thickness: endpoint-to-wall snapping distance
    spur_len: float = 1.5             # x thickness: prune skeleton spurs shorter than this
    min_wall_len: float = 1.5         # x thickness
    split_radius: float = 1.0         # x thickness: erode rooms this much to cut leaks through doorways
    manhattan: bool = True
    px_per_meter: Optional[float] = None


# ============================================================================ cleanup
def clean_mask(label: np.ndarray, num_classes: int, min_size: int = 12) -> np.ndarray:
    """Drop components smaller than ``min_size`` and fill from the nearest surviving pixel."""
    label = label.copy()
    invalid = np.zeros(label.shape, bool)
    for c in range(num_classes):
        m = label == c
        if not m.any():
            continue
        lab, n = ndi.label(m)
        if n == 0:
            continue
        sizes = ndi.sum(m, lab, index=np.arange(1, n + 1))
        small = np.isin(lab, np.nonzero(sizes < min_size)[0] + 1)
        invalid |= small
    if invalid.any() and (~invalid).any():
        _, (iy, ix) = ndi.distance_transform_edt(invalid, return_indices=True)
        label[invalid] = label[iy[invalid], ix[invalid]]
    return label


def _dist(m: np.ndarray) -> np.ndarray:
    """Distance transform treating the image border as background (avoids inf on full masks)."""
    padded = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    return cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]


def split_regions(mask: np.ndarray, radius: float, min_marker: int = 4) -> Tuple[np.ndarray, int]:
    """Marker-based region splitting.

    Two rooms that touch through a thin channel (a doorway the network labelled
    as room instead of door) are one connected component. Eroding by ``radius``
    cuts channels narrower than ``2 * radius``; the surviving cores become
    markers, which are then grown back over the original mask with a BFS so
    every room keeps its full extent. Returns (labels, n_labels incl. 0).
    """
    r = int(round(radius))
    if r < 1:
        n, lab = cv2.connectedComponents(mask, connectivity=4)
        return lab, n
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r + 1, 2 * r + 1))
    core = cv2.erode(mask, k, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    n, markers, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=4)
    keep = np.zeros(n, np.int32)
    nid = 0
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_marker:
            nid += 1
            keep[i] = nid
    markers = keep[markers]
    # mask components that lost every marker (too thin to erode) keep their own id
    n2, lab2 = cv2.connectedComponents(mask, connectivity=4)
    for i in range(1, n2):
        comp = lab2 == i
        if not (markers[comp] > 0).any():
            nid += 1
            markers[comp] = nid
    # geodesic growth: nearest marker within the mask (4-connected BFS == city-block)
    out = markers.copy()
    H, W = mask.shape
    frontier = np.argwhere(out > 0)
    todo = (mask > 0) & (out == 0)
    while len(frontier) and todo.any():
        nxt = []
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            ys, xs = frontier[:, 0] + dy, frontier[:, 1] + dx
            ok = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
            ys, xs, src = ys[ok], xs[ok], frontier[ok]
            free = todo[ys, xs]
            ys, xs, src = ys[free], xs[free], src[free]
            out[ys, xs] = out[src[:, 0], src[:, 1]]
            todo[ys, xs] = False
            nxt.append(np.stack([ys, xs], 1))
        frontier = np.concatenate(nxt) if nxt else np.zeros((0, 2), int)
    return out, nid + 1


# ============================================================================ skeleton graph
_N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _neighbours(skel: np.ndarray, y: int, x: int) -> List[Tuple[int, int]]:
    """8-neighbours with redundant diagonals removed (a diagonal bridged by an
    orthogonal skeleton pixel is not a separate branch). Keeps degree honest."""
    H, W = skel.shape
    out = []
    for dy, dx in _N8:
        ny, nx = y + dy, x + dx
        if not (0 <= ny < H and 0 <= nx < W) or not skel[ny, nx]:
            continue
        if dy != 0 and dx != 0 and (skel[y + dy, x] or skel[y, x + dx]):
            continue
        out.append((ny, nx))
    return out


def skeleton_paths(skel: np.ndarray) -> List[List[Tuple[int, int]]]:
    """Decompose a 1-px skeleton into node-to-node pixel paths (plus closed loops)."""
    skel = skel.astype(bool)
    pts = np.argwhere(skel)
    nbrs = {(int(y), int(x)): _neighbours(skel, int(y), int(x)) for y, x in pts}
    nodes = {p for p, n in nbrs.items() if len(n) != 2}
    seen_edges = set()
    paths: List[List[Tuple[int, int]]] = []

    def walk(start, first):
        path = [start, first]
        seen_edges.add((start, first))
        seen_edges.add((first, start))
        prev, cur = start, first
        while cur not in nodes:
            nxt = [n for n in nbrs[cur] if n != prev and (cur, n) not in seen_edges]
            if not nxt:
                break
            n = nxt[0]
            seen_edges.add((cur, n))
            seen_edges.add((n, cur))
            path.append(n)
            prev, cur = cur, n
            if cur == start:
                break
        return path

    for node in nodes:
        for n in nbrs[node]:
            if (node, n) not in seen_edges:
                paths.append(walk(node, n))
    # isolated loops (every pixel degree 2)
    for p, ns in nbrs.items():
        if p in nodes:
            continue
        for n in ns:
            if (p, n) not in seen_edges:
                nodes.add(p)
                paths.append(walk(p, n))
                nodes.discard(p)
    return paths


def _prune_spurs(paths, min_len: float):
    """Remove short branches that end in a free endpoint (skeleton hairs)."""
    ends: Dict[Tuple[int, int], int] = {}
    for p in paths:
        for e in (p[0], p[-1]):
            ends[e] = ends.get(e, 0) + 1
    kept = []
    for p in paths:
        free = (ends[p[0]] == 1) + (ends[p[-1]] == 1)
        if free and len(p) < min_len and len(paths) > 1:
            continue
        kept.append(p)
    return kept


# ============================================================================ segment ops
@dataclass
class _Seg:
    x1: float
    y1: float
    x2: float
    y2: float
    t: float

    @property
    def length(self) -> float:
        return math.hypot(self.x2 - self.x1, self.y2 - self.y1)

    @property
    def horizontal(self) -> bool:
        return abs(self.y2 - self.y1) < 1e-6 and abs(self.x2 - self.x1) > 0

    @property
    def vertical(self) -> bool:
        return abs(self.x2 - self.x1) < 1e-6 and abs(self.y2 - self.y1) > 0


def _snap_axis(s: _Seg, max_deg: float) -> _Seg:
    ang = abs(math.degrees(math.atan2(s.y2 - s.y1, s.x2 - s.x1))) % 180
    if min(ang, 180 - ang) <= max_deg:
        y = (s.y1 + s.y2) / 2
        a, b = sorted((s.x1, s.x2))
        return _Seg(a, y, b, y, s.t)
    if abs(ang - 90) <= max_deg:
        x = (s.x1 + s.x2) / 2
        a, b = sorted((s.y1, s.y2))
        return _Seg(x, a, x, b, s.t)
    return s


def _merge_collinear(segs: List[_Seg], off_tol: float, gap_tol: float) -> List[_Seg]:
    hs = [s for s in segs if s.horizontal]
    vs = [s for s in segs if s.vertical]
    others = [s for s in segs if not (s.horizontal or s.vertical)]

    def merge(group: List[_Seg], horiz: bool) -> List[_Seg]:
        # key: fixed coordinate; span: (lo, hi)
        items = sorted(group, key=lambda s: (s.y1 if horiz else s.x1))
        clusters: List[List[_Seg]] = []
        for s in items:
            c = s.y1 if horiz else s.x1
            if clusters:
                last = clusters[-1]
                wsum = sum(q.length for q in last)
                mean = sum((q.y1 if horiz else q.x1) * q.length for q in last) / max(wsum, 1e-6)
                if abs(c - mean) <= off_tol:
                    last.append(s)
                    continue
            clusters.append([s])
        out = []
        for cl in clusters:
            spans = sorted((((q.x1, q.x2, q) if horiz else (q.y1, q.y2, q)) for q in cl), key=lambda z: (z[0], z[1]))
            cur: List[_Seg] = [spans[0][2]]
            lo, hi = spans[0][0], spans[0][1]
            runs = []
            for a, b, q in spans[1:]:
                if a <= hi + gap_tol:
                    hi = max(hi, b)
                    cur.append(q)
                else:
                    runs.append((lo, hi, cur))
                    lo, hi, cur = a, b, [q]
            runs.append((lo, hi, cur))
            for lo, hi, qs in runs:
                w = sum(q.length for q in qs) or 1.0
                c = sum((q.y1 if horiz else q.x1) * q.length for q in qs) / w
                t = sum(q.t * q.length for q in qs) / w
                out.append(_Seg(lo, c, hi, c, t) if horiz else _Seg(c, lo, c, hi, t))
        return out

    return merge(hs, True) + merge(vs, False) + others


def _close_junctions(segs: List[_Seg], tol: float) -> List[_Seg]:
    """Extend / trim H endpoints onto nearby V walls and vice versa (L and T joints)."""
    hs = [s for s in segs if s.horizontal]
    vs = [s for s in segs if s.vertical]
    for _ in range(2):
        for h in hs:
            for end in (0, 1):
                ex = h.x1 if end == 0 else h.x2
                best, bd = None, tol
                for v in vs:
                    if v.y1 - tol <= h.y1 <= v.y2 + tol:
                        d = abs(v.x1 - ex)
                        if d < bd:
                            best, bd = v, d
                if best is not None:
                    if end == 0:
                        h.x1 = best.x1
                    else:
                        h.x2 = best.x1
        for v in vs:
            for end in (0, 1):
                ey = v.y1 if end == 0 else v.y2
                best, bd = None, tol
                for h in hs:
                    if h.x1 - tol <= v.x1 <= h.x2 + tol:
                        d = abs(h.y1 - ey)
                        if d < bd:
                            best, bd = h, d
                if best is not None:
                    if end == 0:
                        v.y1 = best.y1
                    else:
                        v.y2 = best.y1
    others = [s for s in segs if not (s.horizontal or s.vertical)]
    # cluster free endpoints of oblique walls
    ends = [(s.x1, s.y1) for s in hs + vs] + [(s.x2, s.y2) for s in hs + vs]
    for s in others:
        for attr in (("x1", "y1"), ("x2", "y2")):
            px, py = getattr(s, attr[0]), getattr(s, attr[1])
            if ends:
                d = [math.hypot(px - ex, py - ey) for ex, ey in ends]
                i = int(np.argmin(d))
                if d[i] < tol:
                    setattr(s, attr[0], ends[i][0])
                    setattr(s, attr[1], ends[i][1])
    return [s for s in hs + vs + others if s.length > 0]


# ============================================================================ polygons
def orthogonalize(poly: np.ndarray, max_deg: float) -> np.ndarray:
    """Rectify a closed polygon: near-axis edges become exactly axis aligned."""
    if len(poly) < 4:
        return poly
    pts = poly.astype(np.float64)
    n = len(pts)
    kinds = []
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        ang = abs(math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))) % 180
        kinds.append("H" if min(ang, 180 - ang) <= max_deg else "V" if abs(ang - 90) <= max_deg else "D")
    # drop vertices between two collinear axis edges
    keep = [i for i in range(n) if not (kinds[i - 1] == kinds[i] and kinds[i] in "HV")]
    if len(keep) < 3:
        return poly
    pts = pts[keep]
    n = len(pts)
    kinds = []
    fixed = []
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        ang = abs(math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))) % 180
        k = "H" if min(ang, 180 - ang) <= max_deg else "V" if abs(ang - 90) <= max_deg else "D"
        kinds.append(k)
        fixed.append((a[1] + b[1]) / 2 if k == "H" else (a[0] + b[0]) / 2 if k == "V" else None)
    out = pts.copy()
    for i in range(n):
        pk, pf = kinds[i - 1], fixed[i - 1]   # edge ending at vertex i
        nk, nf = kinds[i], fixed[i]           # edge starting at vertex i
        x, y = pts[i]
        for k, f in ((pk, pf), (nk, nf)):
            if k == "H":
                y = f
            elif k == "V":
                x = f
        out[i] = (x, y)
    # remove duplicate consecutive vertices
    dedup = [out[0]]
    for p in out[1:]:
        if np.hypot(*(p - dedup[-1])) > 1e-6:
            dedup.append(p)
    if len(dedup) > 2 and np.hypot(*(dedup[0] - dedup[-1])) < 1e-6:
        dedup.pop()
    return np.asarray(dedup)


def _poly_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def _to_pts(a: np.ndarray) -> List[Point]:
    return [(round(float(x), 2), round(float(y), 2)) for x, y in a]


# ============================================================================ main
class Vectorizer:
    def __init__(self, taxonomy: Taxonomy = COARSE, cfg: Optional[VectorizeConfig] = None):
        self.tax, self.cfg = taxonomy, cfg or VectorizeConfig()

    def __call__(self, label: np.ndarray, image_size: Optional[Tuple[int, int]] = None) -> FloorPlanGeometry:
        """``label``: HxW class map. ``image_size`` (w, h): rescale output coords to this size."""
        cfg, tax = self.cfg, self.tax
        label = clean_mask(label.astype(np.uint8), tax.num_classes, cfg.min_speckle)
        H, W = label.shape
        struct = np.isin(label, [tax.wall, tax.door, tax.window])
        thickness = self._thickness(struct)
        walls, wall_polys = self.extract_walls(struct, label, thickness)
        rooms = self.extract_rooms(label, thickness)
        openings = self.extract_openings(label, walls, thickness)
        geo = FloorPlanGeometry(width=W, height=H, rooms=rooms, walls=walls, openings=openings,
                                wall_polygons=wall_polys, px_per_meter=cfg.px_per_meter,
                                meta={"wall_thickness_px": round(thickness, 2), "taxonomy": tax.name})
        if image_size and (image_size[0] != W or image_size[1] != H):
            geo = rescale(geo, image_size[0] / W, image_size[1] / H)
        if cfg.px_per_meter:
            for r in geo.rooms:
                r.area_m2 = round(r.area_px / cfg.px_per_meter ** 2, 2)
        return geo

    @staticmethod
    def _thickness(struct: np.ndarray) -> float:
        if not struct.any():
            return 4.0
        from skimage.morphology import skeletonize

        dist = _dist(struct.astype(np.uint8))
        sk = skeletonize(struct)
        vals = dist[sk]
        cap = max(4.0, min(struct.shape) / 8)
        return float(np.clip(2 * np.median(vals), 2.0, cap)) if len(vals) else 4.0

    # ------------------------------------------------------------------ walls
    def extract_walls(self, struct: np.ndarray, label: np.ndarray, t: float):
        from skimage.morphology import skeletonize

        cfg = self.cfg
        if not struct.any():
            return [], []
        m = struct.astype(np.uint8)
        k = max(1, int(round(t / 3)))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
        dist = _dist(m)
        skel = skeletonize(m > 0)
        paths = _prune_spurs(skeleton_paths(skel), cfg.spur_len * t)
        segs: List[_Seg] = []
        for p in paths:
            if len(p) < 2:
                continue
            arr = np.array([(x, y) for y, x in p], np.float32).reshape(-1, 1, 2)
            approx = cv2.approxPolyDP(arr, max(1.0, cfg.rdp_wall * t), False).reshape(-1, 2)
            for a, b in zip(approx[:-1], approx[1:]):
                n = max(2, int(np.hypot(*(b - a))))
                xs = np.clip(np.linspace(a[0], b[0], n).round().astype(int), 0, m.shape[1] - 1)
                ys = np.clip(np.linspace(a[1], b[1], n).round().astype(int), 0, m.shape[0] - 1)
                th = float(2 * np.median(dist[ys, xs]))
                s = _Seg(float(a[0]), float(a[1]), float(b[0]), float(b[1]), th)
                segs.append(_snap_axis(s, cfg.snap_angle_deg) if cfg.manhattan else s)
        if cfg.manhattan:
            segs = _merge_collinear(segs, cfg.merge_offset * t, cfg.merge_gap * t)
            segs = _close_junctions(segs, cfg.junction_tol * t)
            segs = _merge_collinear(segs, cfg.merge_offset * t, 0.5)
        segs = [s for s in segs if s.length >= cfg.min_wall_len * t]
        walls = []
        for i, s in enumerate(segs):
            walls.append(Wall(i, (round(s.x1, 2), round(s.y1, 2)), (round(s.x2, 2), round(s.y2, 2)),
                              round(max(1.0, s.t), 2), self._is_exterior(s, label)))
        # editable wall outlines
        cnts, _ = cv2.findContours((label == self.tax.wall).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        polys = []
        for c in cnts:
            if cv2.contourArea(c) < t * t:
                continue
            a = cv2.approxPolyDP(c, max(1.0, 0.25 * t), True).reshape(-1, 2)
            if len(a) >= 3:
                polys.append(_to_pts(orthogonalize(a, cfg.snap_angle_deg) if cfg.manhattan else a))
        return walls, polys

    def _is_exterior(self, s: _Seg, label: np.ndarray) -> bool:
        H, W = label.shape
        dx, dy = s.x2 - s.x1, s.y2 - s.y1
        L = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / L, dx / L
        off = s.t * 1.5 + 2
        bg_hits = 0
        total = 0
        for f in np.linspace(0.2, 0.8, 5):
            cx, cy = s.x1 + dx * f, s.y1 + dy * f
            for sgn in (1, -1):
                x, y = int(round(cx + sgn * nx * off)), int(round(cy + sgn * ny * off))
                if 0 <= x < W and 0 <= y < H:
                    total += 1
                    bg_hits += int(label[y, x] == 0)
                else:
                    total += 1
                    bg_hits += 1
        return bg_hits >= max(2, total // 4)

    # ------------------------------------------------------------------ rooms
    def extract_rooms(self, label: np.ndarray, t: float) -> List[Room]:
        cfg, tax = self.cfg, self.tax
        rooms: List[Room] = []
        rid = 0
        for c in tax.room_ids:
            m = (label == c).astype(np.uint8)
            cc, n = split_regions(m, cfg.split_radius * t) if cfg.split_radius > 0 else cv2.connectedComponents(m, connectivity=4)[::-1]
            areas = np.bincount(cc.ravel(), minlength=n)
            for k in range(1, n):
                if areas[k] < cfg.min_room_area:
                    continue
                comp = (cc == k).astype(np.uint8)
                cnts, hier = cv2.findContours(comp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                if not cnts:
                    continue
                outer_idx = [i for i in range(len(cnts)) if hier[0][i][3] < 0]
                outer_i = max(outer_idx, key=lambda i: cv2.contourArea(cnts[i]))
                eps = max(1.0, cfg.rdp_room * t)
                poly = cv2.approxPolyDP(cnts[outer_i], eps, True).reshape(-1, 2)
                if len(poly) < 3:
                    continue
                if cfg.manhattan:
                    poly = orthogonalize(poly, cfg.snap_angle_deg)
                holes = []
                for i in range(len(cnts)):
                    if hier[0][i][3] == outer_i and cv2.contourArea(cnts[i]) > cfg.min_room_area / 2:
                        h = cv2.approxPolyDP(cnts[i], eps, True).reshape(-1, 2)
                        if len(h) >= 3:
                            holes.append(orthogonalize(h, cfg.snap_angle_deg) if cfg.manhattan else h)
                area = _poly_area(poly) - sum(_poly_area(h) for h in holes)
                ys, xs = np.nonzero(comp)
                name = tax.classes[c]
                rooms.append(Room(rid, name if name != "room" else "room", _to_pts(poly), [_to_pts(h) for h in holes],
                                  round(area, 1), None, (round(float(xs.mean()), 1), round(float(ys.mean()), 1))))
                rid += 1
        return rooms

    # ------------------------------------------------------------------ openings
    def extract_openings(self, label: np.ndarray, walls: Sequence[Wall], t: float) -> List[Opening]:
        cfg, tax = self.cfg, self.tax
        out: List[Opening] = []
        oid = 0
        for kind, c in (("door", tax.door), ("window", tax.window)):
            m = (label == c).astype(np.uint8)
            n, cc, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            for k in range(1, n):
                if stats[k, cv2.CC_STAT_AREA] < cfg.min_opening_area:
                    continue
                ys, xs = np.nonzero(cc == k)
                pts = np.stack([xs, ys], 1).astype(np.float32)
                (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
                w, h = w + 1, h + 1  # pixel-centre -> pixel-edge extent
                if h > w:
                    w, h, ang = h, w, ang + 90
                ang = ((ang + 90) % 180) - 90
                if cfg.manhattan:
                    if abs(ang) <= cfg.snap_angle_deg:
                        ang = 0.0
                    elif abs(abs(ang) - 90) <= cfg.snap_angle_deg:
                        ang = 90.0
                box = cv2.boxPoints(((cx, cy), (w, h), ang))
                wall_id = self._host_wall((cx, cy), walls, 2 * t + 2)
                out.append(Opening(oid, kind, _to_pts(box), (round(float(cx), 2), round(float(cy), 2)),
                                   round(float(w), 2), round(float(h), 2), round(float(ang), 2), wall_id))
                oid += 1
        return out

    @staticmethod
    def _host_wall(p: Point, walls: Sequence[Wall], max_d: float) -> Optional[int]:
        best, bd = None, max_d
        px, py = p
        for w in walls:
            (x1, y1), (x2, y2) = w.p1, w.p2
            dx, dy = x2 - x1, y2 - y1
            L2 = dx * dx + dy * dy
            u = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / L2))
            d = math.hypot(px - (x1 + u * dx), py - (y1 + u * dy))
            if d < bd:
                best, bd = w.id, d
        return best


def rescale(geo: FloorPlanGeometry, sx: float, sy: float) -> FloorPlanGeometry:
    def P(p):
        return (round(p[0] * sx, 2), round(p[1] * sy, 2))

    s = (sx + sy) / 2
    geo.width, geo.height = int(round(geo.width * sx)), int(round(geo.height * sy))
    for r in geo.rooms:
        r.polygon = [P(p) for p in r.polygon]
        r.holes = [[P(p) for p in h] for h in r.holes]
        r.centroid = P(r.centroid)
        r.area_px = round(r.area_px * sx * sy, 1)
    for w in geo.walls:
        w.p1, w.p2, w.thickness = P(w.p1), P(w.p2), round(w.thickness * s, 2)
    for o in geo.openings:
        o.polygon = [P(p) for p in o.polygon]
        o.center = P(o.center)
        o.width, o.depth = round(o.width * s, 2), round(o.depth * s, 2)
    geo.wall_polygons = [[P(p) for p in poly] for poly in geo.wall_polygons]
    geo.meta["wall_thickness_px"] = round(float(geo.meta.get("wall_thickness_px", 0)) * s, 2)
    return geo


def vectorize(label: np.ndarray, taxonomy: Taxonomy = COARSE, cfg: Optional[VectorizeConfig] = None,
              image_size: Optional[Tuple[int, int]] = None) -> FloorPlanGeometry:
    return Vectorizer(taxonomy, cfg)(label, image_size)
