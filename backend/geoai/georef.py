"""Raster ingest, CRS diagnostics and CRS repair.

Common projection problems this module detects or fixes:
  * georeferencing stored in sidecars (.tfw/.jgw world files, .prj, .aux.xml) that were not uploaded
  * GCP-only imagery (no geotransform): warped to a regular grid on ingest
  * missing CRS: the raster is kept and flagged until the user assigns one
  * wrong CRS: degrees labelled as metres (or the reverse), eastings outside the UTM zone,
    local datums whose transformation to WGS 84 is approximate
"""

from __future__ import annotations

import math
import shutil
import warnings
from pathlib import Path

import rasterio
import rasterio.shutil
from pyproj import CRS, Transformer
from pyproj.transformer import TransformerGroup
from rasterio.errors import NotGeoreferencedWarning
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

from .raster import build_overviews, compute_stretch, default_bands, mercator_resolution, zoom_range

RASTER_EXT = {".tif", ".tiff", ".jp2", ".img", ".jpg", ".jpeg", ".png"}
SIDECAR_EXT = {".tfw", ".tifw", ".tfwx", ".jgw", ".jpgw", ".pgw", ".pngw", ".j2w", ".wld", ".prj", ".xml", ".ovr",
               ".rpb", ".imd"}


def split_upload(names: list[str]) -> tuple[str, dict[str, str]]:
    """Pick the main raster among uploaded file names and map each sidecar to its stored suffix
    (e.g. 'scene.tif.aux.xml' → '.tif.aux.xml', 'scene.tfw' → '.tfw')."""
    mains = [n for n in names if Path(n).suffix.lower() in RASTER_EXT]
    if not mains:
        raise ValueError("No raster file found. Upload a GeoTIFF (optionally with its .tfw/.prj/.aux.xml files).")
    main = max(mains, key=lambda n: Path(n).suffix.lower() in (".tif", ".tiff"))
    stem = Path(main).stem.lower()
    sidecars = {}
    for n in names:
        if n == main:
            continue
        low = n.lower()
        ext = Path(low).suffix
        if ext not in SIDECAR_EXT:
            raise ValueError(f"Unsupported file in upload: {n}")
        sidecars[n] = low[len(stem):] if low.startswith(stem + ".") else ext
    return main, sidecars


def _crs_from_prj(src_path: Path) -> CRS | None:
    prj = src_path.with_suffix(".prj")
    if not prj.exists():
        return None
    try:
        return CRS.from_user_input(prj.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None


def ingest(src_path: Path, out_path: Path) -> Path:
    """Normalise an upload into a tiled, compressed GeoTIFF with overviews. Returns the stored path."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(src_path) as ds:
            gcps, gcp_crs = ds.gcps
            georef = not ds.transform.is_identity
            if not georef and not gcps:
                raise ValueError("The image has no georeferencing. Include its world file (.tfw/.jgw) and .prj, "
                                 "or upload a GeoTIFF.")
            crs = ds.crs or _crs_from_prj(src_path)
            tiled = ds.profile.get("tiled", False)
            convert = (ds.driver != "GTiff" or not georef or (not tiled and max(ds.width, ds.height) > 2048)
                       or (ds.crs is None and crs is not None))
            opts = dict(driver="GTiff", tiled=True, blockxsize=512, blockysize=512, compress="deflate",
                        predictor=2 if ds.dtypes[0] in ("uint8", "uint16", "int16") else 1, BIGTIFF="IF_SAFER")
            if not georef:  # GCP-only imagery → regular grid in the GCP CRS
                with WarpedVRT(ds, src_crs=gcp_crs, crs=gcp_crs) as vrt:
                    rasterio.shutil.copy(vrt, out_path, **opts)
            elif convert:
                rasterio.shutil.copy(ds, out_path, **opts)
            else:
                shutil.copyfile(src_path, out_path)
    if crs is not None:
        with rasterio.open(out_path, "r+") as ds:
            if ds.crs is None:
                ds.crs = rasterio.crs.CRS.from_wkt(crs.to_wkt())
    try:
        build_overviews(out_path)
    except Exception:  # read-only formats or unusual layouts: tiles still render, just slower
        pass
    return out_path


def set_crs(path: Path, crs_text: str) -> None:
    """Assign (or correct) the CRS without moving pixels: coordinates are reinterpreted."""
    crs = CRS.from_user_input(crs_text)
    with rasterio.open(path, "r+") as ds:
        ds.crs = rasterio.crs.CRS.from_wkt(crs.to_wkt())


def crs_info(crs) -> dict | None:
    if crs is None:
        return None
    c = CRS.from_user_input(crs.to_wkt())
    unit = c.axis_info[0].unit_name if c.axis_info else ""
    return {
        "name": c.name,
        "epsg": c.to_epsg(),
        "type": "geographic" if c.is_geographic else "projected" if c.is_projected else c.type_name,
        "datum": c.datum.name if c.datum else None,
        "ellipsoid": c.ellipsoid.name if c.ellipsoid else None,
        "units": unit,
        "utm_zone": c.utm_zone,
    }


def _datum_accuracy(crs) -> tuple[float | None, bool]:
    """Best available accuracy (m) of the CRS → WGS 84 transformation and whether a better
    grid-based one exists but is not installed locally."""
    try:
        group = TransformerGroup(crs, 4326, always_xy=True)
    except Exception:
        return None, False
    acc = group.transformers[0].accuracy if group.transformers else None
    return (acc if acc is not None and acc >= 0 else None), bool(group.unavailable_operations)


def diagnose(ds) -> list[dict]:
    """Warnings about likely projection mistakes. Each: {level, code, message}."""
    out = []

    def warn(code, message, level="warning"):
        out.append({"level": level, "code": code, "message": message})

    if ds.crs is None:
        warn("no_crs", "No coordinate reference system. Assign one (for example the image's UTM zone EPSG code) "
                       "before it can be placed on the map.", "error")
        return out
    crs = CRS.from_user_input(ds.crs.to_wkt())
    left, bottom, right, top = ds.bounds
    looks_degrees = -180.5 <= left <= 180.5 and -180.5 <= right <= 180.5 and -90.5 <= bottom <= 90.5 \
        and -90.5 <= top <= 90.5 and abs(ds.res[0]) < 0.1
    if crs.is_projected and looks_degrees:
        warn("degrees_as_projected", f"Coordinates look like degrees but the CRS is projected ({crs.name}). "
                                     "The CRS is probably EPSG:4326.", "error")
    if crs.is_geographic and (abs(left) > 180.5 or abs(right) > 180.5 or abs(bottom) > 90.5 or abs(top) > 90.5):
        warn("metres_as_geographic", "Coordinates look like metres but the CRS is geographic (degrees). "
                                     "Assign the projected CRS the image was produced in (e.g. its UTM zone).",
             "error")
    if crs.utm_zone:
        if not (100_000 <= left <= 900_000 and 100_000 <= right <= 900_000):
            warn("utm_easting", f"Eastings {left:,.0f}–{right:,.0f} m fall outside the normal range of "
                                f"UTM zone {crs.utm_zone}. The zone may be wrong.")
        south = crs.utm_zone.upper().endswith("S")
        if (not south and bottom < -1) or (south and top > 10_000_001):
            warn("utm_hemisphere", f"Northings do not fit hemisphere of UTM zone {crs.utm_zone}.", "error")
    if ds.transform.b != 0 or ds.transform.d != 0:
        warn("rotated", "The geotransform is rotated or skewed. It is handled, but check the source if "
                        "the image looks tilted.", "info")
    if not any(w["level"] == "error" for w in out):
        try:
            lon0, lat0, lon1, lat1 = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds, densify_pts=21)
            if not all(map(math.isfinite, (lon0, lat0, lon1, lat1))) or lat1 > 85.06 or lat0 < -85.06:
                warn("mercator_limits", "Part of the raster lies beyond ±85° latitude and cannot be shown "
                                        "on a Web Mercator basemap.")
        except Exception as e:
            warn("transform_failed", f"Cannot transform the bounds to WGS 84: {e}", "error")
        datum = (crs.datum.name if crs.datum else "") or ""
        wgs84_like = any(k in datum for k in ("WGS 84", "World Geodetic System 1984", "ITRF", "WGS84"))
        if not wgs84_like:
            acc, better = _datum_accuracy(ds.crs)
            msg = f"Datum '{datum}' is converted to WGS 84 for the basemap"
            msg += f" with ~{acc:g} m accuracy." if acc else " with unknown accuracy."
            if better:
                msg += " A more accurate grid-based transformation exists but its PROJ grid is not installed."
            warn("datum_shift", msg, "info" if acc and acc <= 1 else "warning")
    return out


def inspect(path) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path) as ds:
            meta = {
                "driver": ds.driver,
                "width": ds.width,
                "height": ds.height,
                "count": ds.count,
                "dtype": ds.dtypes[0],
                "crs": ds.crs.to_string() if ds.crs else None,
                "epsg": ds.crs.to_epsg() if ds.crs else None,
                "crs_info": crs_info(ds.crs),
                "is_geographic": bool(ds.crs and ds.crs.is_geographic),
                "res": list(ds.res),
                "nodata": ds.nodata,
                "bounds": list(ds.bounds),
                "transform": list(ds.transform)[:6],
                "overviews": ds.overviews(1),
                "default_bands": default_bands(ds.count),
                "band_descriptions": [d or f"Band {i + 1}" for i, d in enumerate(ds.descriptions)],
                "warnings": diagnose(ds),
            }
            meta["status"] = "ready"
            if any(w["level"] == "error" for w in meta["warnings"]):
                meta["status"] = "needs_crs"
                return meta
            lon0, lat0, lon1, lat1 = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds, densify_pts=21)
            meta["bounds_wgs84"] = [max(lon0, -180), max(lat0, -85.05), min(lon1, 180), min(lat1, 85.05)]
            meta["footprint"] = footprint(ds)
            meta["minzoom"], meta["maxzoom"] = zoom_range(ds)
            meta["mercator_res"] = mercator_resolution(ds)
            bands = default_bands(ds.count)
            meta["stretch"] = {"-".join(map(str, bands)): compute_stretch(ds, bands)}
            return meta


def footprint(ds, n: int = 16) -> list[list[float]]:
    """Raster outline in lon/lat (densified, so curved edges are drawn correctly)."""
    w, h = ds.width, ds.height
    edge = [(w * i / n, 0) for i in range(n)] + [(w, h * i / n) for i in range(n)] + \
           [(w - w * i / n, h) for i in range(n)] + [(0, h - h * i / n) for i in range(n)]
    xs, ys = zip(*[ds.transform @ p for p in edge])
    lons, lats = Transformer.from_crs(ds.crs, 4326, always_xy=True).transform(xs, ys)
    ring = [[round(a, 8), round(b, 8)] for a, b in zip(lons, lats)]
    return ring + [ring[0]]


def control_grid(path, n: int = 11) -> dict:
    """Pixel ↔ lon/lat control points for the web client's local polynomial georeferencing."""
    with rasterio.open(path) as ds:
        to_ll = Transformer.from_crs(ds.crs, 4326, always_xy=True)
        pts = []
        for j in range(n):
            for i in range(n):
                px, py = ds.width * i / (n - 1), ds.height * j / (n - 1)
                x, y = ds.transform @ (px, py)
                lon, lat = to_ll.transform(x, y)
                pts.append([px, py, lon, lat])
        return {"width": ds.width, "height": ds.height, "points": pts, "res_m": ground_resolution(ds)}


def ground_resolution(ds) -> float:
    """Approximate pixel size on the ground in metres."""
    if ds.crs.is_geographic:
        lat = math.radians((ds.bounds.bottom + ds.bounds.top) / 2)
        return abs(ds.res[0]) * 111_320 * math.cos(lat)
    factor = CRS.from_user_input(ds.crs.to_wkt()).axis_info[0].unit_conversion_factor or 1.0
    return abs(ds.res[0]) * factor
