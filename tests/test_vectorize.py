import json

import cv2
import numpy as np

from floorplannet.classes import COARSE
from floorplannet.vectorize import FloorPlanGeometry, clean_mask, orthogonalize, skeleton_paths, vectorize


def _rooms_gt(mask, min_area=200):
    n, _, stats, _ = cv2.connectedComponentsWithStats((mask == 2).astype(np.uint8), connectivity=4)
    return sum(stats[k, cv2.CC_STAT_AREA] >= min_area for k in range(1, n))


def _openings_gt(mask, cls, min_area=10):
    n, _, stats, _ = cv2.connectedComponentsWithStats((mask == cls).astype(np.uint8), connectivity=8)
    return sum(stats[k, cv2.CC_STAT_AREA] >= min_area for k in range(1, n))


def test_vectorize_ground_truth_topology(synth_samples):
    for _, mask in synth_samples:
        geo = vectorize(mask)
        assert len(geo.rooms) == _rooms_gt(mask)
        assert sum(o.kind == "door" for o in geo.openings) == _openings_gt(mask, COARSE.door)
        assert sum(o.kind == "window" for o in geo.openings) == _openings_gt(mask, COARSE.window)
        assert len(geo.walls) >= 4
        assert any(w.exterior for w in geo.walls) and any(not w.exterior for w in geo.walls)
        # every opening sits on a wall
        assert all(o.wall_id is not None for o in geo.openings)


def test_room_polygons_reproject(synth_samples):
    """Rasterised room polygons must reproduce the room mask (vectorisation fidelity)."""
    for _, mask in synth_samples:
        geo = vectorize(mask)
        canvas = np.zeros(mask.shape, np.uint8)
        for r in geo.rooms:
            cv2.fillPoly(canvas, [np.round(np.array(r.polygon)).astype(np.int32)], 1)
        gt = mask == 2
        iou = (gt & (canvas > 0)).sum() / (gt | (canvas > 0)).sum()
        assert iou > 0.95, iou


def test_walls_are_manhattan(synth_samples):
    geo = vectorize(synth_samples[0][1])
    for w in geo.walls:
        assert w.p1[0] == w.p2[0] or w.p1[1] == w.p2[1]


def test_orthogonalize_rectifies():
    poly = np.array([[0, 0], [50, 1], [51, 40], [1, 41]], float)
    out = orthogonalize(poly, 10)
    for a, b in zip(out, np.roll(out, -1, 0)):
        assert a[0] == b[0] or a[1] == b[1]


def test_skeleton_paths_cross():
    sk = np.zeros((21, 21), bool)
    sk[10, 2:19] = True
    sk[2:19, 10] = True
    paths = skeleton_paths(sk)
    assert len(paths) == 4  # four arms from the junction


def test_clean_mask_removes_speckle():
    m = np.full((40, 40), 2, np.uint8)
    m[5, 5] = 1
    out = clean_mask(m, 5, min_size=4)
    assert (out == 2).all()


def test_serialisation_roundtrip(synth_samples, tmp_path):
    geo = vectorize(synth_samples[1][1], image_size=(1024, 1024))
    assert geo.width == 1024
    d = json.loads(geo.to_json())
    g2 = FloorPlanGeometry.from_dict(d)
    assert g2.summary() == geo.summary()
    gj = geo.to_geojson()
    assert gj["type"] == "FeatureCollection" and len(gj["features"]) == len(geo.rooms) + len(geo.walls) + len(geo.openings)
    assert geo.to_svg().startswith("<svg")
    p = tmp_path / "plan.dxf"
    geo.to_dxf(str(p))
    assert p.stat().st_size > 1000
