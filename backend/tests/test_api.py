import importlib
import time

import pytest
from fastapi.testclient import TestClient

from .fixtures import make_geotiff, make_obb_model


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("GEOAI_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("GEOAI_MODELS_DIR", str(tmp_path / "models"))
    import geoai.config
    import geoai.api

    importlib.reload(geoai.config)
    for mod in ("geoai.models", "geoai.jobs", "geoai.api"):
        importlib.reload(importlib.import_module(mod))
    return TestClient(geoai.api.app)


def test_end_to_end(client, tmp_path):
    assert client.get("/api/health").json()["status"] == "ok"

    model = make_obb_model(tmp_path / "planes.onnx")
    with open(model, "rb") as fh:
        r = client.post("/api/models", files={"file": ("planes.onnx", fh)}, data={"name": "Mock planes"})
    assert r.status_code == 200, r.text
    m = r.json()
    assert m["id"] == "custom-planes" and m["task"] == "obb" and m["names"] == ["plane", "ship"]

    tif = make_geotiff(tmp_path / "scene.tif", width=900, height=700)
    with open(tif, "rb") as fh:
        r = client.post("/api/rasters", files={"file": ("scene.tif", fh)})
    assert r.status_code == 200, r.text
    raster = r.json()
    assert raster["epsg"] == 32638 and len(raster["preview_corners"]) == 4
    assert client.get(f"/api/rasters/{raster['id']}/preview.png").headers["content-type"] == "image/png"

    r = client.post("/api/jobs", json={"raster_id": raster["id"], "model_id": m["id"], "classes": [0]})
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]
    for _ in range(100):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] not in ("queued", "running"):
            break
        time.sleep(0.05)
    assert job["status"] == "done", job
    assert set(job["summary"]["counts"]) == {"plane"}

    fc = client.get(f"/api/jobs/{job_id}/result.geojson").json()
    assert fc["features"] and fc["features"][0]["geometry"]["type"] == "Polygon"
    csv = client.get(f"/api/jobs/{job_id}/export?format=csv").text
    assert csv.splitlines()[0].startswith("id,") and "POLYGON" in csv


def test_rejects_bad_input(client, tmp_path):
    assert client.get("/api/rasters/../../etc").status_code == 404
    assert client.get("/api/jobs/not-an-id").status_code == 400
    bad = tmp_path / "x.txt"
    bad.write_text("hi")
    with open(bad, "rb") as fh:
        assert client.post("/api/rasters", files={"file": ("x.txt", fh)}).status_code == 400
