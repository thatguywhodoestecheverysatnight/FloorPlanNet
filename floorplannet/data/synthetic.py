"""Procedural floor plan generator.

Produces (image, mask) pairs in the ``coarse`` taxonomy with realistic nuisance
factors: several wall render styles, door swings, window glazing lines, room
labels, furniture, dimension chains, floor hatching, blur and JPEG artefacts.

Used for (a) the in-browser demo model, (b) unit tests, and (c) synthetic
pre-training before CubiCasa5K fine-tuning.

Geometry is mask-first: a BSP partition assigns a room id to every pixel inside
an (optionally notched) footprint; walls are the dilated label boundaries, so
any footprint yields topologically consistent walls.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..classes import COARSE

BG, WALL, ROOM, DOOR, WINDOW = 0, COARSE.wall, 2, COARSE.door, COARSE.window
ROOM_NAMES = ["LIVING", "KITCHEN", "BED", "BATH", "WC", "HALL", "STORAGE", "DINING",
              "OFFICE", "ENTRY", "CLOSET", "UTILITY", "MASTER BED", "BALCONY"]


@dataclass
class SynthConfig:
    size: int = 512
    min_room: int = 60
    wall_thick: Tuple[int, int] = (4, 9)
    ext_extra: Tuple[int, int] = (1, 5)
    door_width: Tuple[int, int] = (22, 36)
    notch_prob: float = 0.55
    furniture_prob: float = 0.6
    text_prob: float = 0.8
    dim_prob: float = 0.5


def _bsp(rng: np.random.Generator, rect, min_room: int, depth: int = 0) -> List[Tuple[int, int, int, int]]:
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    can_v = w >= 2 * min_room
    can_h = h >= 2 * min_room
    stop_p = 0.0 if depth < 2 else 0.25 + 0.1 * depth
    if (not can_v and not can_h) or rng.random() < stop_p:
        return [rect]
    vertical = can_v and (not can_h or (w > h if rng.random() < 0.8 else rng.random() < 0.5))
    if vertical:
        s = int(rng.integers(x0 + min_room, x1 - min_room + 1))
        return _bsp(rng, (x0, y0, s, y1), min_room, depth + 1) + _bsp(rng, (s, y0, x1, y1), min_room, depth + 1)
    s = int(rng.integers(y0 + min_room, y1 - min_room + 1))
    return _bsp(rng, (x0, y0, x1, s), min_room, depth + 1) + _bsp(rng, (x0, s, x1, y1), min_room, depth + 1)


def _boundaries(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Boolean maps of horizontal-neighbour and vertical-neighbour label changes."""
    hb = np.zeros_like(labels, bool)
    vb = np.zeros_like(labels, bool)
    hb[:, 1:] = labels[:, 1:] != labels[:, :-1]
    vb[1:, :] = labels[1:, :] != labels[:-1, :]
    return hb, vb


def _runs(coords: np.ndarray) -> List[Tuple[int, int]]:
    """Split sorted 1-D integer coords into contiguous runs [start, end]."""
    if len(coords) == 0:
        return []
    breaks = np.where(np.diff(coords) > 1)[0]
    starts = np.r_[coords[0], coords[breaks + 1]]
    ends = np.r_[coords[breaks], coords[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def _shared_segments(labels: np.ndarray, a: int, b: int):
    """Axis-aligned shared boundary segments between region a and b.

    Returns list of (orientation, fixed_coord, start, end) where orientation 'v'
    means a vertical wall at x=fixed spanning y in [start, end].
    """
    segs = []
    left, right = labels[:, :-1], labels[:, 1:]
    m = ((left == a) & (right == b)) | ((left == b) & (right == a))
    ys, xs = np.nonzero(m)
    for x in np.unique(xs):
        for s, e in _runs(np.sort(ys[xs == x])):
            segs.append(("v", int(x) + 1, s, e))
    top, bot = labels[:-1, :], labels[1:, :]
    m = ((top == a) & (bot == b)) | ((top == b) & (bot == a))
    ys, xs = np.nonzero(m)
    for y in np.unique(ys):
        for s, e in _runs(np.sort(xs[ys == y])):
            segs.append(("h", int(y) + 1, s, e))
    return segs


def _carve(mask: np.ndarray, seg, pos: int, length: int, thick: int, value: int) -> Tuple[int, int, int, int]:
    ori, fixed, _, _ = seg
    half = thick // 2 + 1
    if ori == "v":
        x0, x1 = fixed - half, fixed + half
        y0, y1 = pos - length // 2, pos + length // 2
    else:
        y0, y1 = fixed - half, fixed + half
        x0, x1 = pos - length // 2, pos + length // 2
    region = mask[y0:y1, x0:x1]
    region[region == WALL] = value
    return x0, y0, x1, y1


class FloorPlanGenerator:
    def __init__(self, cfg: Optional[SynthConfig] = None, seed: Optional[int] = None):
        self.cfg = cfg or SynthConfig()
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ geometry
    def _layout(self):
        c, rng, S = self.cfg, self.rng, self.cfg.size
        margin = int(rng.integers(int(S * 0.07), int(S * 0.16)))
        x0, y0 = margin + int(rng.integers(-10, 10)), margin + int(rng.integers(-10, 10))
        x1 = S - margin + int(rng.integers(-10, 10))
        y1 = S - margin + int(rng.integers(-10, 10))
        min_room = int(c.min_room * rng.uniform(0.8, 1.3))
        rooms = _bsp(rng, (x0, y0, x1, y1), min_room)
        labels = np.full((S, S), -1, np.int32)
        for i, (a, b, cc, d) in enumerate(rooms):
            labels[b:d, a:cc] = i
        if rng.random() < c.notch_prob:  # carve 1-2 corner notches -> L / U shaped footprints
            for _ in range(int(rng.integers(1, 3))):
                nw = int(rng.uniform(0.2, 0.45) * (x1 - x0))
                nh = int(rng.uniform(0.2, 0.45) * (y1 - y0))
                cx = x0 if rng.random() < 0.5 else x1 - nw
                cy = y0 if rng.random() < 0.5 else y1 - nh
                labels[cy:cy + nh, cx:cx + nw] = -1
        # drop slivers created by the notch
        for rid in np.unique(labels):
            if rid < 0:
                continue
            m = (labels == rid).astype(np.uint8)
            n, cc_lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
            for k in range(1, n):
                w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
                if min(w, h) < min_room * 0.45:
                    labels[cc_lab == k] = -1
        # relabel components so each room id is one connected region
        out = np.full_like(labels, -1)
        nid = 0
        for rid in np.unique(labels):
            if rid < 0:
                continue
            n, cc_lab = cv2.connectedComponents((labels == rid).astype(np.uint8), connectivity=4)
            for k in range(1, n):
                out[cc_lab == k] = nid
                nid += 1
        return out

    def generate(self):
        c, rng, _S = self.cfg, self.rng, self.cfg.size
        labels = self._layout()
        if labels.max() < 0:
            return self.generate()
        t = int(rng.integers(*c.wall_thick))
        te = t + int(rng.integers(*c.ext_extra))
        hb, vb = _boundaries(labels)
        edge = hb | vb
        # exterior boundary: any edge touching the outside label
        outside = labels < 0
        near_out = cv2.dilate(outside.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        ext_edge = edge & near_out
        int_edge = edge & ~near_out
        k_i = cv2.getStructuringElement(cv2.MORPH_RECT, (t, t))
        k_e = cv2.getStructuringElement(cv2.MORPH_RECT, (te, te))
        wall = (cv2.dilate(int_edge.astype(np.uint8), k_i) | cv2.dilate(ext_edge.astype(np.uint8), k_e)) > 0

        mask = np.where(labels >= 0, ROOM, BG).astype(np.uint8)
        mask[wall] = WALL

        doors, windows = [], []
        n_rooms = int(labels.max()) + 1
        # doors between adjacent rooms: spanning-tree-ish (every room gets >= 1)
        has_door = np.zeros(n_rooms, bool)
        pairs = []
        for a in range(n_rooms):
            for b in range(a + 1, n_rooms):
                segs = [s for s in _shared_segments(labels, a, b) if s[3] - s[2] > c.door_width[1] + 2 * te]
                if segs:
                    pairs.append((a, b, segs))
        rng.shuffle(pairs)
        for a, b, segs in pairs:
            if has_door[a] and has_door[b] and rng.random() < 0.55:
                continue
            seg = max(segs, key=lambda s: s[3] - s[2])
            dw = int(rng.integers(*c.door_width))
            lo, hi = seg[2] + dw // 2 + t + 2, seg[3] - dw // 2 - t - 2
            if hi <= lo:
                continue
            pos = int(rng.integers(lo, hi + 1))
            box = _carve(mask, seg, pos, dw, t, DOOR)
            doors.append((seg, pos, dw, t, box))
            has_door[a] = has_door[b] = True
        # exterior: 1 entrance + windows
        ext_segs = []
        for rid in range(n_rooms):
            ext_segs += [(rid, s) for s in _shared_segments(labels, rid, -1) if s[3] - s[2] > 40 + 2 * te]
        rng.shuffle(ext_segs)
        if ext_segs:
            rid, seg = ext_segs[0]
            dw = int(rng.integers(c.door_width[0] + 4, c.door_width[1] + 6))
            lo, hi = seg[2] + dw // 2 + te + 2, seg[3] - dw // 2 - te - 2
            if hi > lo:
                pos = int(rng.integers(lo, hi + 1))
                doors.append((seg, pos, dw, te, _carve(mask, seg, pos, dw, te, DOOR)))
        for _rid, seg in ext_segs[1:]:
            if rng.random() < 0.25:
                continue
            L = seg[3] - seg[2]
            ww = int(L * rng.uniform(0.25, 0.6))
            lo, hi = seg[2] + ww // 2 + te + 3, seg[3] - ww // 2 - te - 3
            if hi <= lo or ww < 14:
                continue
            pos = int(rng.integers(lo, hi + 1))
            windows.append((seg, pos, ww, te, _carve(mask, seg, pos, ww, te, WINDOW)))
        image = self._render(mask, labels, doors, windows)
        return image, mask

    # ------------------------------------------------------------------ rendering
    def _render(self, mask, labels, doors, windows) -> np.ndarray:
        rng, S = self.rng, self.cfg.size
        paper = np.array([rng.integers(238, 256)] * 3, np.int32) - np.array([0, rng.integers(0, 6), rng.integers(0, 12)])
        img = np.ones((S, S, 3), np.uint8) * paper.clip(0, 255).astype(np.uint8)
        ink = int(rng.integers(0, 60))
        ink_c = (ink, ink, ink)
        style = rng.choice(["solid", "solid", "gray", "hatch", "outline"])
        wall = (mask == WALL).astype(np.uint8)

        # floor fills / tiling in some rooms (label: room)
        for rid in range(int(labels.max()) + 1):
            if rng.random() < 0.18:
                rm = (labels == rid) & (mask == ROOM)
                step = int(rng.integers(8, 18))
                yy, xx = np.mgrid[0:S, 0:S]
                tile = ((yy % step == 0) | (xx % step == 0)) & rm
                shade = int(rng.integers(170, 225))
                img[tile] = (shade, shade, shade)
            elif rng.random() < 0.12:
                rm = (labels == rid) & (mask == ROOM)
                tint = rng.integers(225, 250, 3)
                img[rm] = tint

        self._furniture(img, labels, mask)

        if style in ("solid", "gray"):
            v = ink if style == "solid" else int(rng.integers(90, 150))
            img[wall > 0] = (v, v, v)
            if style == "gray":
                cnts, _ = cv2.findContours(wall, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cnts, -1, ink_c, 1)
        elif style == "hatch":
            yy, xx = np.mgrid[0:S, 0:S]
            step = int(rng.integers(4, 7))
            hatch = ((xx + yy) % step == 0) & (wall > 0)
            img[wall > 0] = (235, 235, 235)
            img[hatch] = ink_c
            cnts, _ = cv2.findContours(wall, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, ink_c, 1)
        else:
            cnts, _ = cv2.findContours(wall, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, ink_c, int(rng.integers(1, 3)))

        for seg, pos, dw, _thick, (x0, y0, x1, y1) in doors:
            img[y0:y1, x0:x1] = paper.clip(0, 255)
            self._door_swing(img, seg, pos, dw, labels, ink_c)
        for seg, _pos, _ww, _thick, (x0, y0, x1, y1) in windows:
            img[y0:y1, x0:x1] = paper.clip(0, 255)
            cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), ink_c, 1)
            if seg[0] == "v":
                xm = (x0 + x1) // 2
                cv2.line(img, (xm, y0), (xm, y1 - 1), ink_c, 1)
                if x1 - x0 > 7:
                    cv2.line(img, (xm - 2, y0), (xm - 2, y1 - 1), ink_c, 1)
            else:
                ym = (y0 + y1) // 2
                cv2.line(img, (x0, ym), (x1 - 1, ym), ink_c, 1)
                if y1 - y0 > 7:
                    cv2.line(img, (x0, ym - 2), (x1 - 1, ym - 2), ink_c, 1)

        self._text(img, labels, mask, ink_c)
        self._dimensions(img, mask, ink_c)
        return self._degrade(img)

    def _door_swing(self, img, seg, pos, dw, labels, color):
        """Draw an opened leaf plus a quarter-circle swing (pure decoration, label stays room)."""
        rng = self.rng
        ori, fixed, _, _ = seg
        sign = 1 if rng.random() < 0.5 else -1          # which side of the wall the leaf opens to
        hinge_low = rng.random() < 0.5                  # hinge at the low or high end of the gap
        end = -1 if hinge_low else 1
        if ori == "v":
            hinge = (fixed, pos + end * dw // 2)
            tip = (fixed + sign * dw, hinge[1])
            th_t = 0 if sign > 0 else 180
            th_o = 90 if hinge_low else -90
        else:
            hinge = (pos + end * dw // 2, fixed)
            tip = (hinge[0], fixed + sign * dw)
            th_t = 90 if sign > 0 else -90
            th_o = 0 if hinge_low else 180
        if abs(th_o - th_t) > 180:
            th_o += 360 if th_o < th_t else -360
        cv2.line(img, hinge, tip, color, 1, cv2.LINE_AA)
        cv2.ellipse(img, hinge, (dw, dw), 0, min(th_t, th_o), max(th_t, th_o), color, 1, cv2.LINE_AA)

    def _furniture(self, img, labels, mask):
        rng = self.rng
        for rid in range(int(labels.max()) + 1):
            if rng.random() > self.cfg.furniture_prob:
                continue
            ys, xs = np.nonzero((labels == rid) & (mask == ROOM))
            if len(xs) < 800:
                continue
            x0, x1, y0, y1 = xs.min() + 8, xs.max() - 8, ys.min() + 8, ys.max() - 8
            for _ in range(int(rng.integers(1, 4))):
                w, h = int(rng.integers(12, 45)), int(rng.integers(12, 45))
                if x1 - x0 <= w or y1 - y0 <= h:
                    continue
                px, py = int(rng.integers(x0, x1 - w)), int(rng.integers(y0, y1 - h))
                if not (labels[py:py + h, px:px + w] == rid).all():
                    continue
                shade = tuple(int(v) for v in rng.integers(60, 170, 1).repeat(3))
                kind = rng.integers(0, 4)
                if kind == 0:
                    cv2.rectangle(img, (px, py), (px + w, py + h), shade, 1)
                elif kind == 1:
                    cv2.circle(img, (px + w // 2, py + h // 2), min(w, h) // 2, shade, 1, cv2.LINE_AA)
                elif kind == 2:
                    cv2.rectangle(img, (px, py), (px + w, py + h), shade, 1)
                    cv2.line(img, (px, py), (px + w, py + h), shade, 1)
                else:
                    cv2.ellipse(img, (px + w // 2, py + h // 2), (w // 2, h // 3), 0, 0, 360, shade, 1, cv2.LINE_AA)

    def _text(self, img, labels, mask, color):
        rng = self.rng
        for rid in range(int(labels.max()) + 1):
            if rng.random() > self.cfg.text_prob:
                continue
            ys, xs = np.nonzero(labels == rid)
            if len(xs) < 1500:
                continue
            name = str(rng.choice(ROOM_NAMES))
            scale = float(rng.uniform(0.3, 0.5))
            (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            cx, cy = int(xs.mean()), int(ys.mean())
            org = (cx - tw // 2, cy)
            cv2.putText(img, name, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
            if rng.random() < 0.6:
                area = f"{rng.uniform(4, 30):.1f} m2"
                cv2.putText(img, area, (org[0], cy + th + 4), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.8, color, 1, cv2.LINE_AA)

    def _dimensions(self, img, mask, color):
        rng = self.rng
        if rng.random() > self.cfg.dim_prob:
            return
        ys, xs = np.nonzero(mask != BG)
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        off = int(rng.integers(12, 24))
        y = max(4, y0 - off)
        cv2.line(img, (x0, y), (x1, y), color, 1)
        for x in (x0, x1):
            cv2.line(img, (x - 3, y + 3), (x + 3, y - 3), color, 1)
        cv2.putText(img, f"{rng.uniform(6, 20):.2f}", ((x0 + x1) // 2 - 12, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)
        x = min(img.shape[1] - 4, x1 + off)
        cv2.line(img, (x, y0), (x, y1), color, 1)
        for yy in (y0, y1):
            cv2.line(img, (x - 3, yy + 3), (x + 3, yy - 3), color, 1)

    def _degrade(self, img: np.ndarray) -> np.ndarray:
        rng = self.rng
        if rng.random() < 0.5:
            k = int(rng.choice([3, 5]))
            img = cv2.GaussianBlur(img, (k, k), rng.uniform(0.3, 1.0))
        if rng.random() < 0.5:
            noise = rng.normal(0, rng.uniform(2, 8), img.shape)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        if rng.random() < 0.5:
            q = int(rng.integers(35, 90))
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return img


def generate_dataset(n: int, seed: int = 0, size: int = 512):
    gen = FloorPlanGenerator(SynthConfig(size=size), seed=seed)
    for _ in range(n):
        yield gen.generate()
