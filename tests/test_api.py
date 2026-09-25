import cv2
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client(tiny_onnx, monkeypatch):
    monkeypatch.setenv("FLOORPLANNET_MODEL", tiny_onnx)
    monkeypatch.setenv("FLOORPLANNET_TILE", "256")
    from serve import app as appmod

    appmod.get_predictor.cache_clear()
    return TestClient(appmod.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_and_export(client, synth_samples):
    img, _ = synth_samples[0]
    ok, buf = cv2.imencode(".png", img)
    r = client.post("/api/predict", files={"file": ("plan.png", buf.tobytes(), "image/png")},
                    data={"max_side": "384"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["width"] == 512 and "geometry" in body and "mask_png" in body
    for fmt, ctype in (("svg", "image/svg+xml"), ("geojson", "application/geo+json"), ("dxf", "application/dxf")):
        e = client.post(f"/api/export/{fmt}", json=body["geometry"])
        assert e.status_code == 200 and e.headers["content-type"].startswith(ctype)


def test_vectorize_endpoint(client, synth_samples):
    _, mask = synth_samples[1]
    ok, buf = cv2.imencode(".png", mask)
    r = client.post("/api/vectorize", files={"file": ("m.png", buf.tobytes(), "image/png")})
    assert r.status_code == 200 and r.json()["summary"]["rooms"] > 0


def test_bad_upload(client):
    r = client.post("/api/predict", files={"file": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 400
