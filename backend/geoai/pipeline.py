"""Tiled, georeferenced inference over a whole raster → GeoJSON FeatureCollection."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np
import rasterio
from pyproj import Geod, Transformer

from .models import OnnxModel
from .postprocess import Detections, nms, obb_corners
from .raster import compute_stretch, default_bands, read_rgb
from .tiling import tile_grid

GEOD = Geod(ellps="WGS84")
EDGE_MARGIN_PX = 2


class Cancelled(Exception):
    pass


@dataclass
class DetectionParams:
    conf: float = 0.25
    iou: float = 0.45
    classes: list[int] | None = None
    overlap: float = 0.2
    bands: list[int] | None = None
    simplify_px: float = 1.0
    class_agnostic_nms: bool = False
    min_valid_fraction: float = 0.01
    # >1 zooms in (each model pixel covers 1/scale raster pixels); use when objects look
    # smaller than in the training data, e.g. coarse imagery or downsampled exports.
    scale: float = 1.0


def _drop_seam_cuts(det: Detections, col, row, tile, width, height, overlap_px) -> Detections:
    """Remove objects touching an interior tile edge: a neighbouring tile sees them whole.
    Objects bigger than the overlap can't be whole in any tile, so they are kept."""
    if not len(det) or overlap_px <= 0:
        return det
    if det.polygons is not None and any(p is not None for p in det.polygons):
        pts = [p if p is not None else obb_corners(det.boxes[i : i + 1])[0] for i, p in enumerate(det.polygons)]
        x0 = np.array([p[:, 0].min() for p in pts]); x1 = np.array([p[:, 0].max() for p in pts])
        y0 = np.array([p[:, 1].min() for p in pts]); y1 = np.array([p[:, 1].max() for p in pts])
    else:
        c = obb_corners(det.boxes)
        x0, x1, y0, y1 = c[..., 0].min(1), c[..., 0].max(1), c[..., 1].min(1), c[..., 1].max(1)
    m = EDGE_MARGIN_PX
    cut = np.zeros(len(det), bool)
    if col > 0:
        cut |= x0 <= m
    if row > 0:
        cut |= y0 <= m
    if col + tile < width:
        cut |= x1 >= tile - m
    if row + tile < height:
        cut |= y1 >= tile - m
    small = np.maximum(x1 - x0, y1 - y0) < overlap_px
    return det.take(np.flatnonzero(~(cut & small)))


def run_detection(raster_path, model: OnnxModel, params: DetectionParams, progress=None, cancelled=None) -> dict:
    t0 = time.time()
    model_tile = model.input_hw[0]
    tile = max(32, int(round(model_tile / params.scale)))  # tile footprint in raster pixels
    overlap_px = int(round(tile * params.overlap))
    with rasterio.open(raster_path) as ds:
        bands = params.bands or default_bands(ds.count)
        stretch = compute_stretch(ds, bands)
        grid = tile_grid(ds.width, ds.height, tile, params.overlap)
        parts, processed = [], 0
        for i, (col, row) in enumerate(grid):
            if cancelled and cancelled():
                raise Cancelled()
            rgb, valid = read_rgb(ds, col, row, tile, bands, stretch, out_size=model_tile)
            if valid.mean() >= params.min_valid_fraction:
                processed += 1
                det = model.predict(rgb, params.conf, params.iou, params.classes, params.simplify_px)
                if len(det):
                    cx = np.clip(det.boxes[:, 0].astype(int), 0, model_tile - 1)
                    cy = np.clip(det.boxes[:, 1].astype(int), 0, model_tile - 1)
                    det = det.take(np.flatnonzero(valid[cy, cx]))
                    if model_tile != tile:  # model pixels → raster pixels
                        k = tile / model_tile
                        det.boxes[:, :4] *= k
                        if det.polygons is not None:
                            det.polygons = [p * k if p is not None else None for p in det.polygons]
                    det = _drop_seam_cuts(det, col, row, tile, ds.width, ds.height, overlap_px)
                    det.boxes[:, 0] += col
                    det.boxes[:, 1] += row
                    if det.polygons is not None:
                        det.polygons = [p + (col, row) if p is not None else None for p in det.polygons]
                    parts.append(det)
            if progress:
                progress((i + 1) / len(grid), f"Tile {i + 1}/{len(grid)}")
        merged = Detections.concat(parts)
        keep = nms(merged.boxes, merged.scores, merged.classes, params.iou, rotated=model.task == "obb",
                   class_agnostic=params.class_agnostic_nms)
        merged = merged.take(keep)
        features = to_features(merged, model, ds.transform, ds.crs)
        meta_raster = {"width": ds.width, "height": ds.height, "crs": ds.crs.to_string(), "res": list(ds.res)}

    counts = Counter(f["properties"]["class_name"] for f in features)
    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "model": model.spec.id,
            "task": model.task,
            "classes": model.names,
            "params": asdict(params),
            "raster": meta_raster,
            "tiles_total": len(grid),
            "tiles_processed": processed,
            "tile_size": tile,
            "counts": dict(counts.most_common()),
            "elapsed_s": round(time.time() - t0, 2),
        },
    }


def _pix_to_lonlat(pts: np.ndarray, transform, to_wgs84) -> np.ndarray:
    a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
    xw = a * pts[:, 0] + b * pts[:, 1] + c
    yw = d * pts[:, 0] + e * pts[:, 1] + f
    if to_wgs84 is not None:
        xw, yw = to_wgs84.transform(xw, yw)
    return np.stack([xw, yw], 1)


def to_features(det: Detections, model: OnnxModel, transform, crs) -> list[dict]:
    to_wgs84 = None if crs.to_epsg() == 4326 else Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    corners = obb_corners(det.boxes) if len(det) else np.zeros((0, 4, 2))
    features = []
    for i in range(len(det)):
        poly_px = det.polygons[i] if det.polygons is not None and det.polygons[i] is not None else corners[i]
        ring = _pix_to_lonlat(np.asarray(poly_px, np.float64), transform, to_wgs84)
        area, perim = GEOD.polygon_area_perimeter(ring[:, 0], ring[:, 1])
        cls = int(det.classes[i])
        center = _pix_to_lonlat(det.boxes[i : i + 1, :2].astype(np.float64), transform, to_wgs84)[0]
        props = {
            "id": i + 1,
            "class_id": cls,
            "class_name": model.names[cls] if cls < len(model.names) else str(cls),
            "score": round(float(det.scores[i]), 4),
            "area_m2": round(abs(area), 2),
            "perimeter_m": round(perim, 2),
            "center_lon": round(float(center[0]), 7),
            "center_lat": round(float(center[1]), 7),
        }
        if model.task == "obb":
            c = _pix_to_lonlat(corners[i].astype(np.float64), transform, to_wgs84)
            az1, _, s1 = GEOD.inv(c[0, 0], c[0, 1], c[1, 0], c[1, 1])
            az2, _, s2 = GEOD.inv(c[1, 0], c[1, 1], c[2, 0], c[2, 1])
            length, width, az = (s1, s2, az1) if s1 >= s2 else (s2, s1, az2)
            props.update(length_m=round(length, 2), width_m=round(width, 2), heading_deg=round(az % 180, 1))
        coords = ring.round(8).tolist()
        coords.append(coords[0])
        features.append({"type": "Feature", "id": i + 1, "properties": props,
                         "geometry": {"type": "Polygon", "coordinates": [coords]}})
    return features
