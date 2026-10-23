"""Raster reading: display stretch, inference tiles and exact Web Mercator map tiles."""

from __future__ import annotations

import math

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform
from rasterio.windows import Window

WEB_MERCATOR = "EPSG:3857"
ORIGIN = 20037508.342789244  # half the Web Mercator world width, metres
TILE = 256


def default_bands(count: int) -> list[int]:
    return [1, 2, 3] if count >= 3 else [1, 1, 1]


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


def to_uint8(arr: np.ndarray, stretch) -> np.ndarray:
    """(bands, H, W) array → (H, W, bands) uint8 using per-band ranges."""
    if arr.dtype == np.uint8 and all(tuple(s) == (0.0, 255.0) for s in stretch):
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


# ---------------------------------------------------------------- Web Mercator tiles

def tile_bounds(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """(left, bottom, right, top) of an XYZ tile in EPSG:3857 metres."""
    size = 2 * ORIGIN / (1 << z)
    left = -ORIGIN + x * size
    top = ORIGIN - y * size
    return left, top - size, left + size, top


def mercator_resolution(ds) -> float:
    """Native pixel size of the raster once projected to Web Mercator (metres/pixel)."""
    transform, _, _ = calculate_default_transform(ds.crs, WEB_MERCATOR, ds.width, ds.height, *ds.bounds)
    return abs(transform.a)


def zoom_range(ds) -> tuple[int, int]:
    """(minzoom, maxzoom): maxzoom shows native pixels; minzoom shows the raster ~64 px wide."""
    res = mercator_resolution(ds)
    native = math.ceil(math.log2(2 * ORIGIN / (TILE * res)))
    span = max(ds.width, ds.height)
    minzoom = max(0, math.floor(native - math.log2(span / 64)))
    return minzoom, min(max(native, 0), 24)


def render_tile(path, z: int, x: int, y: int, bands, stretch, size: int = TILE) -> bytes | None:
    """Render one XYZ tile as RGBA PNG, or None when the tile holds no data.

    The warped VRT is defined on exactly the tile's bounds, so the output is placed by
    the tile grid with no rounding; GDAL reads the source overviews for low zooms.
    """
    left, bottom, right, top = tile_bounds(z, x, y)
    with rasterio.open(path) as src:
        res = mercator_resolution(src)
        n = int(min(max(size, round((right - left) / res)), size * 512))
        transform = from_bounds(left, bottom, right, top, n, n)
        # without nodata, a destination alpha band is what marks the area outside the raster
        add_alpha = src.nodata is None
        with WarpedVRT(src, crs=WEB_MERCATOR, transform=transform, width=n, height=n,
                       resampling=Resampling.bilinear, add_alpha=add_alpha) as vrt:
            mask = vrt.read_masks(1, out_shape=(size, size), resampling=Resampling.nearest)
            if not mask.any():
                return None
            arr = vrt.read(bands, out_shape=(len(bands), size, size), resampling=Resampling.bilinear)
    rgb = to_uint8(arr, stretch)
    bgra = np.dstack([rgb[..., ::-1], mask.astype(np.uint8)])
    ok, buf = cv2.imencode(".png", bgra, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return buf.tobytes()


def build_overviews(path) -> list[int]:
    """Add internal overviews (factors 2, 4, 8, …) so low-zoom tiles read little data."""
    with rasterio.open(path, "r+") as ds:
        factors, f = [], 2
        while max(ds.width, ds.height) / f > TILE:
            factors.append(f)
            f *= 2
        if factors and not ds.overviews(1):
            ds.build_overviews(factors, Resampling.average)
            ds.update_tags(ns="rio_overview", resampling="average")
        return factors
