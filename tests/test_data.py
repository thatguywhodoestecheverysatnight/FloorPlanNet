import numpy as np

from floorplannet.classes import COARSE, CUBICASA12, room_label
from floorplannet.data.cubicasa import parse_cubicasa_svg, parse_transform, rasterize
from floorplannet.data.synthetic import FloorPlanGenerator
from floorplannet.data.transforms import TrainAugment, to_tensor_arrays

SVG = """<?xml version="1.0"?>
<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100">
  <g class="Model"><g class="Floor"><g class="Floorplan" transform="translate(10,0)">
    <g id="Space1" class="Space Kitchen"><polygon points="0,0 90,0 90,100 0,100"/></g>
    <g id="Space2" class="Space Bedroom"><polygon points="90,0 180,0 180,100 90,100"/></g>
    <g id="Wall" class="Wall External"><polygon points="85,0 95,0 95,100 85,100"/>
      <g id="Door" class="Door Swing Beside"><polygon points="85,40 95,40 95,60 85,60"/>
        <g class="Panel"><polygon points="0,0 1,0 1,1"/></g></g>
    </g>
    <g id="Window" class="Window Regular"><polygon points="20,0 60,0 60,4 20,4"/></g>
  </g></g></g>
</svg>"""


def test_transform_parsing():
    m = parse_transform("translate(10,5) scale(2)")
    p = m @ np.array([1.0, 1.0, 1.0])
    assert np.allclose(p[:2], [12, 7])
    r = parse_transform("rotate(90)") @ np.array([1.0, 0.0, 1.0])
    assert np.allclose(r[:2], [0, 1], atol=1e-9)


def test_cubicasa_svg_roundtrip(tmp_path):
    f = tmp_path / "model.svg"
    f.write_text(SVG)
    polys = parse_cubicasa_svg(f)
    kinds = sorted(p.kind for p in polys)
    assert kinds == ["door", "room", "room", "wall", "window"]  # decoration panel ignored, nested door found
    mask = rasterize(polys, (100, 200), COARSE)
    assert mask[50, 30] == 2            # kitchen interior -> room
    assert mask[20, 100] == COARSE.wall  # wall shifted by translate(10,0)
    assert mask[50, 100] == COARSE.door  # door painted over wall
    assert mask[2, 50] == COARSE.window
    m12 = rasterize(polys, (100, 200), CUBICASA12)
    assert m12[50, 30] == 3 and m12[50, 150] == 5  # kitchen / bedroom


def test_room_label_mapping():
    assert room_label("Kitchen", CUBICASA12) == 3
    assert room_label("NotARoom", CUBICASA12) == 11
    assert room_label("Bedroom", COARSE) == 2


def test_synthetic_deterministic_and_valid():
    a_img, a_mask = FloorPlanGenerator(seed=5).generate()
    b_img, b_mask = FloorPlanGenerator(seed=5).generate()
    assert np.array_equal(a_mask, b_mask) and np.array_equal(a_img, b_img)
    assert a_img.shape == (512, 512, 3) and a_mask.dtype == np.uint8
    present = set(np.unique(a_mask).tolist())
    assert {0, 1, 2}.issubset(present) and present <= {0, 1, 2, 3, 4}


def test_augment_shapes(synth_samples):
    img, mask = synth_samples[0]
    aug = TrainAugment(crop=256, seed=0)
    for _ in range(5):
        i2, m2 = aug(img, mask)
        assert i2.shape == (256, 256, 3) and m2.shape == (256, 256)
        assert set(np.unique(m2)) <= {0, 1, 2, 3, 4, 255}
    x = to_tensor_arrays(i2)
    assert x.shape == (3, 256, 256) and x.dtype == np.float32
