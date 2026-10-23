import importlib
import io
import json
import math
import time
import zipfile

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from PIL import Image
from pyproj import Transformer
from rasterio.transform import from_origin

from .fixtures import make_geotiff, make_obb_model


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("GEOAI_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("GEOAI_MODELS_DIR", str(tmp_path / "models"))
    import geoai.api
    import geoai.config

    importlib.reload(geoai.config)
    for mod in ("geoai.models", "geoai.jobs", "geoai.api"):
        importlib.reload(importlib.import_module(mod))
    return TestClient(geoai.api.app)


def _upload(client, *paths):
    files = [("files", (p.name, open(p, "rb"))) for p in paths]
    try:
        return client.post("/api/rasters", files=files)
    finally:
        for _, (_, fh) in files:
            fh.close()


def _wait(client, job_id):
    for _ in range(200):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_end_to_end(client, tmp_path):
    assert client.get("/api/health").json()["status"] == "ok"

    model = make_obb_model(tmp_path / "planes.onnx")
    with open(model, "rb") as fh:
        r = client.post("/api/models", files={"file": ("planes.onnx", fh)}, data={"name": "Mock planes"})
    assert r.status_code == 200, r.text
    m = r.json()
    assert m["id"] == "custom-planes" and m["task"] == "obb" and m["names"] == ["plane", "ship"]

    r = _upload(client, make_geotiff(tmp_path / "scene.tif", width=900, height=700))
    assert r.status_code == 200, r.text
    raster = r.json()
    assert raster["status"] == "ready" and raster["epsg"] == 32638
    assert raster["crs_info"]["utm_zone"] == "38N" and raster["maxzoom"] >= 18

    r = client.post("/api/jobs", json={"raster_id": raster["id"], "model_id": m["id"], "classes": [0]})
    assert r.status_code == 200, r.text
    job = _wait(client, r.json()["id"])
    assert job["status"] == "done", job
    assert set(job["summary"]["counts"]) == {"plane"}

    fc = client.get(f"/api/jobs/{job['id']}/result.geojson").json()
    assert fc["features"] and fc["features"][0]["geometry"]["type"] == "Polygon"
    csv = client.get(f"/api/jobs/{job['id']}/export?format=csv").text
    assert csv.splitlines()[0].startswith("id,") and "POLYGON" in csv

    # auto annotation from the job, then a YOLO-OBB export
    r = client.post(f"/api/rasters/{raster['id']}/annotations/from-job", json={"job_id": job["id"]})
    assert r.status_code == 200, r.text
    doc = r.json()["doc"]
    assert r.json()["imported"] == len(fc["features"])
    assert all(i["source"] == "auto" and i["type"] == "polygon" for i in doc["items"])
    r = client.post(f"/api/rasters/{raster['id']}/annotations/export", json={"format": "yolo-obb", "chip_size": 640})
    z = zipfile.ZipFile(io.BytesIO(r.content))
    labels = [n for n in z.namelist() if n.startswith("labels/")]
    assert labels and "task: obb" in z.read("data.yaml").decode()
    line = next(z.read(n).decode() for n in labels if z.read(n))
    assert len(line.splitlines()[0].split()) == 9


def test_window_limits_processing(client, tmp_path):
    with open(make_obb_model(tmp_path / "planes.onnx"), "rb") as fh:
        m = client.post("/api/models", files={"file": ("planes.onnx", fh)}).json()
    raster = _upload(client, make_geotiff(tmp_path / "scene.tif", width=1500, height=1500)).json()
    r = client.post("/api/jobs", json={"raster_id": raster["id"], "model_id": m["id"], "window": [0, 0, 300, 300]})
    job = _wait(client, r.json()["id"])
    assert job["summary"]["tiles"] == 1 and job["summary"]["counts"] == {"plane": 1}


def test_tiles_are_exactly_aligned(client, tmp_path):
    """A marked pixel must land where Web Mercator says it is, at sub-pixel accuracy."""
    path = tmp_path / "mark.tif"
    W = H = 3000
    data = np.full((3, H, W), 40, np.uint8)
    px, py = 2217, 1311
    data[:, py, px] = 255
    with rasterio.open(path, "w", driver="GTiff", width=W, height=H, count=3, dtype="uint8", crs="EPSG:32640",
                       transform=from_origin(230000, 2710000, 2.0, 2.0)) as ds:
        ds.write(data)
        cx, cy = ds.transform @ (px + 0.5, py + 0.5)
    raster = _upload(client, path).json()
    z = raster["maxzoom"]
    mx, my = Transformer.from_crs(32640, 3857, always_xy=True).transform(cx, cy)
    size = 2 * 20037508.342789244 / 2 ** z
    tx, ty = (mx + 20037508.342789244) / size, (20037508.342789244 - my) / size
    r = client.get(f"/api/rasters/{raster['id']}/tiles/{z}/{int(tx)}/{int(ty)}.png")
    assert r.status_code == 200
    img = np.asarray(Image.open(io.BytesIO(r.content)))[..., 0].astype(float)
    yy, xx = np.unravel_index(img.argmax(), img.shape)
    ex, ey = (tx % 1) * 256, (ty % 1) * 256
    assert math.hypot(xx + 0.5 - ex, yy + 0.5 - ey) < 1.5
    # outside the raster → 204, cached
    assert client.get(f"/api/rasters/{raster['id']}/tiles/{z}/0/0.png").status_code == 204


def test_crs_diagnostics_and_repair(client, tmp_path):
    # degrees written with a UTM CRS: the classic wrong-CRS mistake
    path = tmp_path / "wrong.tif"
    with rasterio.open(path, "w", driver="GTiff", width=200, height=200, count=3, dtype="uint8", crs="EPSG:32638",
                       transform=from_origin(44.30, 33.40, 0.0001, 0.0001)) as ds:
        ds.write(np.zeros((3, 200, 200), np.uint8))
    meta = _upload(client, path).json()
    assert meta["status"] == "needs_crs"
    assert any(w["code"] == "degrees_as_projected" for w in meta["warnings"])
    assert client.get(f"/api/rasters/{meta['id']}/tiles/10/0/0.png").status_code == 409
    fixed = client.post(f"/api/rasters/{meta['id']}/crs", json={"crs": "4326"}).json()
    assert fixed["status"] == "ready" and fixed["epsg"] == 4326
    assert fixed["bounds_wgs84"][0] == pytest.approx(44.30, abs=1e-6)


def test_world_file_and_prj_sidecars(client, tmp_path):
    img = tmp_path / "photo.jpg"
    Image.fromarray(np.full((100, 120, 3), 128, np.uint8)).save(img)
    (tmp_path / "photo.jgw").write_text("0.5\n0\n0\n-0.5\n440000.25\n3700000.25\n")
    (tmp_path / "photo.prj").write_text(rasterio.crs.CRS.from_epsg(32638).to_wkt())
    meta = _upload(client, img, tmp_path / "photo.jgw", tmp_path / "photo.prj").json()
    assert meta["status"] == "ready" and meta["epsg"] == 32638, meta
    assert meta["bounds"][0] == pytest.approx(440000.0) and meta["sidecars"] == ["photo.jgw", "photo.prj"]
    # without the world file the image cannot be placed
    r = _upload(client, img)
    assert r.status_code == 400 and "world file" in r.json()["detail"]


def test_annotations_crud_and_exports(client, tmp_path):
    raster = _upload(client, make_geotiff(tmp_path / "scene.tif", width=800, height=800, res=0.5)).json()
    rid = raster["id"]
    doc = client.get(f"/api/rasters/{rid}/annotations").json()
    assert doc["version"] == 0 and doc["classes"][0]["name"] == "object"
    doc["classes"] = [{"id": 1, "name": "tree", "color": "#3fb68b"}, {"id": 2, "name": "building", "color": "#e5534b"}]
    doc["items"] = [
        {"id": 1, "cls": 1, "type": "bbox", "pts": [[150, 100], [50, 40]]},  # corners normalised on save
        {"id": 2, "cls": 2, "type": "polygon", "pts": [[300, 300], [400, 300], [400, 380], [300, 380]]},
        {"id": 3, "cls": 1, "type": "point", "pts": [[600, 600]]},
        {"id": 4, "cls": 1, "type": "stroke", "pts": [[500, 100], [700, 100]], "width": 20},
        {"id": 5, "cls": 1, "type": "stroke", "pts": [[600, 80], [600, 120]], "width": 10, "erase": True},
    ]
    saved = client.put(f"/api/rasters/{rid}/annotations", json=doc).json()
    assert saved["version"] == 1 and saved["items"][0]["pts"] == [[50, 40], [150, 100]]
    assert client.put(f"/api/rasters/{rid}/annotations", json=doc).status_code == 409  # stale version

    def export(fmt, **kw):
        r = client.post(f"/api/rasters/{rid}/annotations/export", json={"format": fmt, "chip_size": 800, **kw})
        assert r.status_code == 200, r.text
        return zipfile.ZipFile(io.BytesIO(r.content))

    z = export("yolo-detect", val_ratio=0)
    lines = z.read([n for n in z.namelist() if n.startswith("labels/train/")][0]).decode().split("\n")
    # bbox 50..150 × 40..100 in an 800 px chip → centre (100, 70), size (100, 60)
    assert "0 0.125000 0.087500 0.125000 0.075000" in lines
    assert any(l.startswith("1 0.437500 0.425000 0.125000 0.100000") for l in lines)
    assert any(l.startswith("0 0.750000 0.750000 0.020000 0.020000") for l in lines)  # point → 16 px box
    assert "names:\n  0: tree\n  1: building" in z.read("data.yaml").decode()

    coco = json.loads(export("coco").read("annotations.json"))
    assert {c["name"] for c in coco["categories"]} == {"tree", "building"}
    seg = next(a for a in coco["annotations"] if a["category_id"] == 2)
    assert seg["bbox"] == [300, 300, 100, 80] and seg["area"] == 8000

    z = export("semantic")
    mask = np.asarray(Image.open(io.BytesIO(z.read([n for n in z.namelist() if n.startswith("masks/")][0]))))
    assert mask[70, 100] == 1 and mask[340, 350] == 2 and mask[100, 550] == 1
    assert mask[100, 600] == 0  # erased by the eraser stroke
    assert mask[600, 600] == 0  # points are not part of semantic masks

    gj = json.loads(export("geojson").read("annotations.geojson"))
    assert len(gj["features"]) == 5
    poly = next(f for f in gj["features"] if f["properties"]["class"] == "building")
    assert poly["properties"]["area_m2"] == pytest.approx(100 * 80 * 0.25, rel=0.01)

    # GeoJSON round trip lands on the same pixels
    bio = io.BytesIO(json.dumps(gj).encode())
    r = client.post(f"/api/rasters/{rid}/annotations/import", files={"file": ("a.geojson", bio)})
    items = r.json()["doc"]["items"]
    again = [i for i in items if i["id"] > 5 and i["type"] == "polygon" and i["cls"] == 2]
    assert len(items) == 10 and np.allclose(again[-1]["pts"][0], [300, 300], atol=1e-4)

    # semantic mask preview tile
    z_ = raster["maxzoom"]
    lon, lat = (raster["bounds_wgs84"][0] + raster["bounds_wgs84"][2]) / 2, (raster["bounds_wgs84"][1] + raster["bounds_wgs84"][3]) / 2
    n = 2 ** z_
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    found = False
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            r = client.get(f"/api/rasters/{rid}/annotations/mask/{z_}/{x + dx}/{y + dy}.png")
            found |= r.status_code == 200
    assert found


def test_yolo_import(client, tmp_path):
    raster = _upload(client, make_geotiff(tmp_path / "scene.tif", width=1000, height=500)).json()
    labels = io.BytesIO(b"0 0.5 0.5 0.1 0.2\n1 0.1 0.1 0.2 0.1 0.2 0.3\n")
    r = client.post(f"/api/rasters/{raster['id']}/annotations/import", files={"file": ("l.txt", labels)},
                    data={"names": "car,pool"})
    doc = r.json()["doc"]
    car = next(i for i in doc["items"] if i["type"] == "bbox")
    assert car["pts"] == [[450, 200], [550, 300]]
    assert {c["name"] for c in doc["classes"]} >= {"car", "pool"}


def test_rejects_bad_input(client, tmp_path):
    assert client.get("/api/rasters/../../etc").status_code == 404
    assert client.get("/api/jobs/not-an-id").status_code == 400
    bad = tmp_path / "x.txt"
    bad.write_text("hi")
    assert _upload(client, bad).status_code == 400
