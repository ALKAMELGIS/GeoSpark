"""GeoTIFF inspection, display stretch, tile reading and web previews."""

from __future__ import annotations

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import Window


def default_bands(count: int) -> list[int]:
    return [1, 2, 3] if count >= 3 else [1, 1, 1]


def inspect(path) -> dict:
    with rasterio.open(path) as ds:
        if ds.crs is None:
            raise ValueError("Raster is not georeferenced (no CRS).")
        bounds_wgs84 = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds, densify_pts=21)
        return {
            "driver": ds.driver,
            "width": ds.width,
            "height": ds.height,
            "count": ds.count,
            "dtype": ds.dtypes[0],
            "crs": ds.crs.to_string(),
            "epsg": ds.crs.to_epsg(),
            "is_geographic": ds.crs.is_geographic,
            "res": list(ds.res),
            "nodata": ds.nodata,
            "bounds": list(ds.bounds),
            "bounds_wgs84": list(bounds_wgs84),
            "default_bands": default_bands(ds.count),
            "band_descriptions": [d or f"Band {i + 1}" for i, d in enumerate(ds.descriptions)],
        }


def compute_stretch(ds, bands: list[int], lo_pct=2.0, hi_pct=98.0) -> list[tuple[float, float]]:
    """Per-band (low, high) display range. uint8 imagery passes through untouched,
    because detection models are trained on unstretched 8-bit RGB."""
    if ds.dtypes[0] == "uint8":
        return [(0.0, 255.0)] * len(bands)
    scale = max(ds.width, ds.height) / 1024
    shape = (max(1, int(ds.height / max(scale, 1))), max(1, int(ds.width / max(scale, 1))))
    out = []
    for b in bands:
        arr = ds.read(b, out_shape=shape, resampling=Resampling.nearest, masked=True)
        vals = arr.compressed()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            out.append((0.0, 1.0))
            continue
        lo, hi = np.percentile(vals, [lo_pct, hi_pct])
        out.append((float(lo), float(hi) if hi > lo else float(lo) + 1.0))
    return out


def to_uint8(arr: np.ndarray, stretch: list[tuple[float, float]]) -> np.ndarray:
    """(bands, H, W) array → (H, W, bands) uint8 using per-band ranges."""
    if arr.dtype == np.uint8 and all(s == (0.0, 255.0) for s in stretch):
        return np.ascontiguousarray(np.moveaxis(arr, 0, -1))
    out = np.empty(arr.shape[1:] + (arr.shape[0],), dtype=np.uint8)
    for i, (lo, hi) in enumerate(stretch):
        band = (arr[i].astype(np.float32) - lo) / (hi - lo)
        out[..., i] = (np.clip(np.nan_to_num(band), 0, 1) * 255).astype(np.uint8)
    return out


def read_rgb(ds, col_off: int, row_off: int, size: int, bands, stretch, out_size: int | None = None):
    """Read a square tile (zero-padded past raster edges), optionally resampled to `out_size` pixels.
    Returns (rgb uint8 HxWx3, valid mask HxW)."""
    window = Window(col_off, row_off, size, size)
    out_size = out_size or size
    kw = {} if out_size == size else {"out_shape": (len(bands), out_size, out_size), "resampling": Resampling.bilinear}
    arr = ds.read(bands, window=window, boundless=True, fill_value=0, **kw)
    mkw = {} if out_size == size else {"out_shape": (out_size, out_size), "resampling": Resampling.nearest}
    valid = ds.read_masks(1, window=window, boundless=True, **mkw) > 0
    return to_uint8(arr, stretch), valid


def render_preview(path, out_png, max_size: int = 2048, bands=None) -> list[list[float]]:
    """Warp the raster to Web Mercator, write an RGBA PNG and return its four
    lon/lat corners (TL, TR, BR, BL) for a MapLibre image source."""
    from pyproj import Transformer

    with rasterio.open(path) as src:
        bands = bands or default_bands(src.count)
        stretch = compute_stretch(src, bands)
        with WarpedVRT(src, crs="EPSG:3857", resampling=Resampling.bilinear) as vrt:
            scale = max(vrt.width, vrt.height) / max_size
            scale = max(scale, 1.0)
            h, w = max(1, int(vrt.height / scale)), max(1, int(vrt.width / scale))
            arr = vrt.read(bands, out_shape=(len(bands), h, w), resampling=Resampling.bilinear)
            mask = vrt.read_masks(1, out_shape=(h, w), resampling=Resampling.nearest)
            left, bottom, right, top = vrt.bounds
    rgb = to_uint8(arr, stretch)
    rgba = np.dstack([rgb, mask.astype(np.uint8)])
    Image.fromarray(rgba, "RGBA").save(out_png, optimize=True)
    t = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    corners = [(left, top), (right, top), (right, bottom), (left, bottom)]
    return [list(t.transform(x, y)) for x, y in corners]
