import numpy as np
import pytest
import rasterio

from geoai.models import ModelSpec, OnnxModel
from geoai.pipeline import DetectionParams, run_detection
from geoai.postprocess import decode, nms
from geoai.tiling import tile_grid

from .fixtures import make_geotiff, make_obb_e2e_model, make_obb_model, make_seg_model


def _model(path, task):
    spec = ModelSpec(id="t", name="t", task=task, file=str(path))
    return OnnxModel(spec)


def test_tile_grid_covers_raster():
    grid = tile_grid(1500, 1000, 640, 0.2)
    cols = sorted({c for c, _ in grid})
    rows = sorted({r for _, r in grid})
    assert cols[0] == 0 and cols[-1] + 640 == 1500
    assert rows[0] == 0 and rows[-1] + 640 == 1000
    assert all(b - a <= 640 for a, b in zip(cols, cols[1:]))


def test_small_raster_single_tile():
    assert tile_grid(300, 200, 1024, 0.2) == [(0, 0)]


def test_decode_and_rotated_nms(tmp_path):
    m = _model(make_obb_model(tmp_path / "m.onnx"), "obb")
    det = m.predict(np.zeros((640, 640, 3), np.uint8), conf=0.25, iou=0.45)
    assert len(det) == 2  # duplicate suppressed, low score filtered
    assert sorted(det.classes.tolist()) == [0, 1]
    assert np.allclose(det.boxes[0, :4], (100, 100, 60, 20))


def test_end2end_layout_matches_raw(tmp_path):
    raw = _model(make_obb_model(tmp_path / "raw.onnx"), "obb")
    e2e = _model(make_obb_e2e_model(tmp_path / "e2e.onnx"), "obb")
    img = np.zeros((640, 640, 3), np.uint8)
    a, b = raw.predict(img), e2e.predict(img)
    assert np.allclose(a.boxes, b.boxes) and np.allclose(a.scores, b.scores)


def test_class_filter(tmp_path):
    m = _model(make_obb_model(tmp_path / "m.onnx"), "obb")
    det = m.predict(np.zeros((640, 640, 3), np.uint8), classes=[1])
    assert det.classes.tolist() == [1]


def test_letterbox_scales_back_to_tile(tmp_path):
    m = _model(make_obb_model(tmp_path / "m.onnx"), "obb")
    det = m.predict(np.zeros((320, 320, 3), np.uint8))  # upscaled ×2 into a 640 input
    assert np.allclose(det.boxes[0, :4], (50, 50, 30, 10))


def test_segment_polygon(tmp_path):
    m = _model(make_seg_model(tmp_path / "s.onnx"), "segment")
    det = m.predict(np.zeros((640, 640, 3), np.uint8))
    poly = det.polygons[0]
    assert poly is not None and len(poly) >= 4
    assert abs(poly[:, 0].min() - 160) <= 4 and abs(poly[:, 0].max() - 240) <= 4
    assert abs(poly[:, 1].min() - 180) <= 4 and abs(poly[:, 1].max() - 220) <= 4


def test_pipeline_georeferences_tiles(tmp_path):
    tif = make_geotiff(tmp_path / "img.tif", width=1500, height=1500, res=0.5)
    m = _model(make_obb_model(tmp_path / "m.onnx"), "obb")
    fc = run_detection(tif, m, DetectionParams(overlap=0.2))
    grid = tile_grid(1500, 1500, 640, 0.2)
    assert fc["metadata"]["tiles_processed"] == len(grid)
    # every tile yields one plane and one ship at distinct raster positions
    assert fc["metadata"]["counts"] == {"plane": len(grid), "ship": len(grid)}
    plane = next(f for f in fc["features"] if f["properties"]["class_name"] == "plane")
    p = plane["properties"]
    assert p["length_m"] == pytest.approx(30, rel=0.01)  # 60 px × 0.5 m
    assert p["width_m"] == pytest.approx(10, rel=0.01)
    assert p["area_m2"] == pytest.approx(300, rel=0.02)
    with rasterio.open(tif) as ds:
        from pyproj import Transformer
        lon, lat = Transformer.from_crs(ds.crs, "EPSG:4326", always_xy=True).transform(*(ds.transform @ (100, 100)))
    assert any(abs(f["properties"]["center_lon"] - lon) < 1e-6 and abs(f["properties"]["center_lat"] - lat) < 1e-6
               for f in fc["features"])


def test_pipeline_resample_scales_back_to_raster(tmp_path):
    # scale=2: each 640 px model tile covers 320 raster px, so the 60×20 px model box is 30×10 raster px
    tif = make_geotiff(tmp_path / "img.tif", width=1000, height=1000, res=0.5)
    m = _model(make_obb_model(tmp_path / "m.onnx"), "obb")
    fc = run_detection(tif, m, DetectionParams(overlap=0.2, scale=2.0))
    assert fc["metadata"]["tile_size"] == 320
    assert fc["metadata"]["tiles_processed"] == len(tile_grid(1000, 1000, 320, 0.2))
    plane = next(f["properties"] for f in fc["features"] if f["properties"]["class_name"] == "plane")
    assert plane["length_m"] == pytest.approx(15, rel=0.01)  # 30 raster px × 0.5 m
    assert plane["width_m"] == pytest.approx(5, rel=0.01)


def test_seam_cut_objects_dropped_when_neighbour_sees_them():
    from geoai.pipeline import _drop_seam_cuts
    from geoai.postprocess import Detections

    boxes = np.array([[5, 300, 20, 20, 0], [320, 320, 20, 20, 0], [630, 300, 20, 20, 0]], np.float32)
    det = Detections(boxes, np.ones(3, np.float32), np.zeros(3, np.int32))
    # interior tile: all four edges are shared with neighbours
    kept = _drop_seam_cuts(det, col=512, row=512, tile=640, width=4000, height=4000, overlap_px=128)
    assert kept.boxes[:, 0].tolist() == [320]
    # first tile: left edge is the raster boundary, so the left object stays
    kept = _drop_seam_cuts(det, col=0, row=0, tile=640, width=4000, height=4000, overlap_px=128)
    assert kept.boxes[:, 0].tolist() == [5, 320]


def test_nms_rotated_vs_axis_aligned():
    boxes = np.array([[50, 50, 100, 10, 0.0], [50, 50, 100, 10, np.pi / 2]], np.float32)
    scores = np.array([0.9, 0.8], np.float32)
    cls = np.zeros(2, np.int32)
    assert len(nms(boxes, scores, cls, 0.3, rotated=True)) == 2  # a cross: low IoU
    assert len(nms(boxes, scores, cls, 0.3, rotated=False)) == 1  # angle ignored


def test_decode_raw_shapes():
    out = np.zeros((1, 4 + 3, 10), np.float32)
    out[0, :4, 0] = (5, 5, 2, 2)
    out[0, 6, 0] = 0.9
    boxes, scores, cls, coeffs = decode(out, "detect", 3, 0.5)
    assert boxes.shape == (1, 5) and cls.tolist() == [2] and coeffs is None
