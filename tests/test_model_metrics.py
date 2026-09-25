import numpy as np
import pytest

torch = pytest.importorskip("torch")

from floorplannet.inference.predictor import Predictor, PredictorConfig  # noqa: E402
from floorplannet.losses import FloorPlanLoss, edge_weight_map  # noqa: E402
from floorplannet.metrics import ConfusionMatrix  # noqa: E402
from floorplannet.models import build_model, count_params  # noqa: E402


@pytest.mark.parametrize("arch", ["lite", "nano"])
def test_forward_shapes(arch):
    m = build_model(arch, 5, pretrained=False, aux_loss=True).train()
    out = m(torch.randn(2, 3, 96, 128))
    assert out["out"].shape == (2, 5, 96, 128) and out["aux"].shape == (2, 5, 96, 128)
    m.eval()
    assert "aux" not in m(torch.randn(1, 3, 64, 64))
    assert count_params(m) < 5e6


def test_torchvision_deeplab():
    m = build_model("deeplabv3_resnet50", 5, pretrained=False, aux_loss=True).eval()
    assert m(torch.randn(1, 3, 64, 64))["out"].shape == (1, 5, 64, 64)


def test_loss_finite_and_decreases():
    torch.manual_seed(0)
    m = build_model("nano", 5, pretrained=False, aux_loss=True)
    x = torch.randn(2, 3, 64, 64)
    y = torch.randint(0, 5, (2, 64, 64))
    y[:, :4] = 255
    crit = FloorPlanLoss()
    opt = torch.optim.Adam(m.parameters(), 1e-2)
    losses = []
    for _ in range(8):
        loss = crit(m(x), y)["total"]
        assert torch.isfinite(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]


def test_edge_weight_map():
    y = torch.zeros(1, 20, 20, dtype=torch.long)
    y[:, :, 10:] = 1
    w = edge_weight_map(y, 255, radius=1)
    assert w[0, 5, 9] == 1 and w[0, 5, 10] == 1 and w[0, 5, 2] == 0


def test_confusion_matrix_miou():
    cm = ConfusionMatrix(3, ignore_index=255)
    t = np.array([[0, 0, 1, 1], [2, 2, 255, 255]])
    p = np.array([[0, 1, 1, 1], [2, 0, 0, 0]])
    cm.update_numpy(p, t)
    r = cm.compute(["a", "b", "c"])
    # iou a = 1/3, b = 2/3, c = 1/2
    assert abs(r["mIoU"] - (1 / 3 + 2 / 3 + 1 / 2) / 3) < 1e-6
    assert abs(r["pixel_acc"] - 4 / 6) < 1e-6


def test_predictor_tiling_matches_shape():
    m = build_model("nano", 5, pretrained=False, aux_loss=False)
    pred = Predictor(m, 5, "cpu", PredictorConfig(tile=128, overlap=0.25, max_side=300, tta=True))
    img = (np.random.rand(200, 330, 3) * 255).astype(np.uint8)
    probs, scale = pred.predict_proba(img)
    assert probs.shape[0] == 5 and max(probs.shape[1:]) == 300
    assert np.allclose(probs.sum(0), 1, atol=1e-4)
    assert pred.predict(img).shape == (200, 330)
