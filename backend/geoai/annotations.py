"""Label annotation: storage, georeferencing, import/export and semantic-mask rendering.

Geometry is stored in the raster's own pixel grid (x right, y down, pixel corners at integers),
which is exact for training exports. Geographic views (GeoJSON, map tiles) are derived from it
with the raster's affine transform and CRS.

Task modes and their geometry types:
  detection — bbox, point            → YOLO detect / YOLO OBB / COCO bbox
  instance  — polygon, freehand, bbox → YOLO seg / COCO segmentation
  semantic  — polygon, freehand, stroke (brush), eraser → class-index masks
"""

from __future__ import annotations

import io
import json
import os
import random
import tempfile
import zipfile
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import rasterio
from PIL import Image
from pydantic import BaseModel, Field, field_validator
from pyproj import Geod, Transformer
from rasterio.transform import Affine, from_bounds
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window
from shapely.geometry import LineString, Point, Polygon, box
from shapely.validation import make_valid

from .raster import WEB_MERCATOR, compute_stretch, default_bands, read_rgb, tile_bounds
from .tiling import tile_grid

PALETTE = ["#3fb68b", "#e5534b", "#4c9aff", "#f2b84b", "#b97cf3", "#2ec4d6", "#f47b3c", "#9bd25b", "#ef6fa8",
           "#c9a36b", "#5ad1a8", "#7b8cff", "#e9d45b", "#ff8f7a", "#7fc8f8", "#d38fff", "#a3e635", "#f59e0b"]
GEOD = Geod(ellps="WGS84")
ItemType = Literal["bbox", "polygon", "freehand", "point", "stroke"]
FILL_TYPES = ("bbox", "polygon", "freehand")


class AnnClass(BaseModel):
    id: int
    name: str = Field(min_length=1, max_length=80)
    color: str = Field(pattern=r"^#[0-9a-fA-F]{6}$")


class AnnItem(BaseModel):
    id: int
    cls: int
    type: ItemType
    pts: list[tuple[float, float]]
    width: float | None = None  # brush diameter in raster pixels (stroke)
    erase: bool = False  # semantic eraser: paints background
    source: str = "manual"
    score: float | None = None

    @field_validator("pts")
    @classmethod
    def _finite(cls, v):
        if not v or not all(np.isfinite(c) for p in v for c in p):
            raise ValueError("pts must be finite coordinates")
        return v

    def model_post_init(self, _):
        need = {"bbox": 2, "polygon": 3, "freehand": 3, "point": 1, "stroke": 1}[self.type]
        if len(self.pts) < need:
            raise ValueError(f"{self.type} needs at least {need} points")
        if self.type == "bbox":
            (x0, y0), (x1, y1) = self.pts[0], self.pts[-1]
            self.pts = [(min(x0, x1), min(y0, y1)), (max(x0, x1), max(y0, y1))]
        if self.type == "stroke" and not (self.width and self.width > 0):
            raise ValueError("stroke needs a positive width")


class AnnDoc(BaseModel):
    task: Literal["detection", "instance", "semantic"] = "detection"
    classes: list[AnnClass] = []
    items: list[AnnItem] = []
    version: int = 0


# ---------------------------------------------------------------- storage

def doc_path(raster_dir: Path) -> Path:
    return raster_dir / "annotations.json"


def load_doc(raster_dir: Path) -> AnnDoc:
    p = doc_path(raster_dir)
    if not p.exists():
        return AnnDoc(classes=[AnnClass(id=1, name="object", color=PALETTE[0])])
    return AnnDoc.model_validate_json(p.read_text(encoding="utf-8"))


def save_doc(raster_dir: Path, doc: AnnDoc) -> AnnDoc:
    ids = {c.id for c in doc.classes}
    doc.items = [i for i in doc.items if i.cls in ids or i.erase]
    doc.version += 1
    fd, tmp = tempfile.mkstemp(dir=raster_dir, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(doc.model_dump_json())
    os.replace(tmp, doc_path(raster_dir))
    return doc


def ensure_class(doc: AnnDoc, name: str) -> int:
    for c in doc.classes:
        if c.name.lower() == name.lower():
            return c.id
    cid = max([c.id for c in doc.classes], default=0) + 1
    doc.classes.append(AnnClass(id=cid, name=name, color=PALETTE[(cid - 1) % len(PALETTE)]))
    return cid


def _next_id(doc: AnnDoc) -> int:
    return max([i.id for i in doc.items], default=0) + 1


# ---------------------------------------------------------------- geometry

def item_geometry(item: AnnItem, point_box: float = 16.0):
    """Shapely geometry in pixel coordinates."""
    p = item.pts
    if item.type == "bbox":
        return box(p[0][0], p[0][1], p[1][0], p[1][1])
    if item.type == "point":
        h = point_box / 2
        return box(p[0][0] - h, p[0][1] - h, p[0][0] + h, p[0][1] + h)
    if item.type == "stroke":
        line = LineString(p) if len(p) > 1 else Point(p[0])
        return line.buffer(item.width / 2)
    g = Polygon(p)
    return g if g.is_valid else make_valid(g)


def _polygons(geom) -> list[Polygon]:
    if geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    return [g for g in getattr(geom, "geoms", []) if g.geom_type == "Polygon" and g.area > 0]


class Georef:
    """Pixel ↔ lon/lat for one raster."""

    def __init__(self, ds):
        self.transform = ds.transform
        self.inv = ~ds.transform
        self.to_ll = Transformer.from_crs(ds.crs, 4326, always_xy=True)
        self.from_ll = Transformer.from_crs(4326, ds.crs, always_xy=True)

    def to_lonlat(self, pts) -> list[list[float]]:
        a = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        t = self.transform
        x = t.a * a[:, 0] + t.b * a[:, 1] + t.c
        y = t.d * a[:, 0] + t.e * a[:, 1] + t.f
        lon, lat = self.to_ll.transform(x, y)
        return np.round(np.stack([lon, lat], 1), 9).tolist()

    def to_pixel(self, lonlat) -> list[tuple[float, float]]:
        a = np.asarray(lonlat, dtype=np.float64).reshape(-1, 2)
        x, y = self.from_ll.transform(a[:, 0], a[:, 1])
        t = self.inv
        px = t.a * x + t.b * y + t.c
        py = t.d * x + t.e * y + t.f
        return [(float(u), float(v)) for u, v in zip(px, py)]


# ---------------------------------------------------------------- GeoJSON

def to_geojson(doc: AnnDoc, ds) -> dict:
    geo = Georef(ds)
    names = {c.id: c for c in doc.classes}
    feats = []
    for it in doc.items:
        c = names.get(it.cls)
        props = {"id": it.id, "class_id": it.cls, "class": c.name if c else "erase", "type": it.type,
                 "source": it.source, "score": it.score}
        if it.type == "point":
            geom = {"type": "Point", "coordinates": geo.to_lonlat(it.pts)[0]}
        elif it.type == "stroke":
            props["width_px"] = it.width
            props["erase"] = it.erase
            pts = it.pts if len(it.pts) > 1 else [it.pts[0], it.pts[0]]
            geom = {"type": "LineString", "coordinates": geo.to_lonlat(pts)}
        else:
            ring = [it.pts[0], (it.pts[1][0], it.pts[0][1]), it.pts[1], (it.pts[0][0], it.pts[1][1])] \
                if it.type == "bbox" else it.pts
            coords = geo.to_lonlat(ring)
            coords.append(coords[0])
            geom = {"type": "Polygon", "coordinates": [coords]}
            area, _ = GEOD.polygon_area_perimeter([p[0] for p in coords], [p[1] for p in coords])
            props["area_m2"] = round(abs(area), 2)
            props["erase"] = it.erase
        feats.append({"type": "Feature", "id": it.id, "properties": props, "geometry": geom})
    return {"type": "FeatureCollection", "features": feats,
            "metadata": {"task": doc.task, "classes": [c.model_dump() for c in doc.classes],
                         "raster": {"width": ds.width, "height": ds.height, "crs": ds.crs.to_string(),
                                    "transform": list(ds.transform)[:6]}}}


# ---------------------------------------------------------------- import

def _class_from_props(props: dict) -> str:
    for k in ("class", "class_name", "label", "name", "category"):
        if props.get(k):
            return str(props[k])
    return "object"


def import_geojson(doc: AnnDoc, ds, data: dict) -> int:
    crs_name = (data.get("crs") or {}).get("properties", {}).get("name")
    geo = Georef(ds)
    if crs_name and "CRS84" not in crs_name and "4326" not in crs_name:
        to_raster = Transformer.from_crs(crs_name, ds.crs, always_xy=True)
        inv = ~ds.transform

        def conv(coords):
            a = np.asarray(coords, dtype=np.float64)[:, :2]
            x, y = to_raster.transform(a[:, 0], a[:, 1])
            return [inv @ (u, v) for u, v in zip(x, y)]
    else:
        def conv(coords):
            return geo.to_pixel(np.asarray(coords, dtype=np.float64)[:, :2])

    feats = data["features"] if data.get("type") == "FeatureCollection" else [data]
    n, nid = 0, _next_id(doc)
    for f in feats:
        g, props = f.get("geometry"), f.get("properties") or {}
        if not g:
            continue
        cid = ensure_class(doc, _class_from_props(props))
        src = props.get("source", "import")
        parts = []
        if g["type"] == "Point":
            parts.append(("point", conv([g["coordinates"]]), None))
        elif g["type"] == "MultiPoint":
            parts += [("point", conv([c]), None) for c in g["coordinates"]]
        elif g["type"] in ("LineString", "MultiLineString"):
            lines = [g["coordinates"]] if g["type"] == "LineString" else g["coordinates"]
            parts += [("stroke", conv(c), float(props.get("width_px") or 10)) for c in lines]
        elif g["type"] in ("Polygon", "MultiPolygon"):
            polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
            for rings in polys:
                pts = conv(rings[0])[:-1] if rings[0][0] == rings[0][-1] else conv(rings[0])
                if props.get("type") == "bbox":
                    xs, ys = zip(*pts)
                    parts.append(("bbox", [(min(xs), min(ys)), (max(xs), max(ys))], None))
                else:
                    parts.append(("polygon", pts, None))
        for typ, pts, width in parts:
            try:
                doc.items.append(AnnItem(id=nid, cls=cid, type=typ, pts=pts, width=width, source=src,
                                         score=props.get("score"), erase=bool(props.get("erase", False))))
            except ValueError:
                continue
            nid += 1
            n += 1
    return n


def import_yolo(doc: AnnDoc, ds, text: str, names: list[str]) -> int:
    """YOLO labels normalised to the full raster: 'c cx cy w h' or 'c x1 y1 x2 y2 …' (seg / OBB)."""
    W, H = ds.width, ds.height
    n, nid = 0, _next_id(doc)
    for line in text.splitlines():
        v = line.split()
        if len(v) < 5:
            continue
        k = int(float(v[0]))
        cid = ensure_class(doc, names[k] if k < len(names) else f"class_{k}")
        nums = list(map(float, v[1:]))
        if len(nums) == 4:
            cx, cy, w, h = nums
            item = AnnItem(id=nid, cls=cid, type="bbox", source="import",
                           pts=[((cx - w / 2) * W, (cy - h / 2) * H), ((cx + w / 2) * W, (cy + h / 2) * H)])
        else:
            pts = [(nums[i] * W, nums[i + 1] * H) for i in range(0, len(nums) - 1, 2)]
            item = AnnItem(id=nid, cls=cid, type="polygon", pts=pts, source="import")
        doc.items.append(item)
        nid += 1
        n += 1
    return n


def import_coco(doc: AnnDoc, ds, data: dict) -> int:
    images = data.get("images", [])
    img = next((i for i in images if i.get("width") == ds.width and i.get("height") == ds.height),
               images[0] if images else None)
    if img is None:
        raise ValueError("COCO file has no images")
    sx, sy = ds.width / img["width"], ds.height / img["height"]
    cats = {c["id"]: c["name"] for c in data.get("categories", [])}
    n, nid = 0, _next_id(doc)
    for a in data.get("annotations", []):
        if a.get("image_id") != img["id"]:
            continue
        cid = ensure_class(doc, cats.get(a["category_id"], f"class_{a['category_id']}"))
        seg = a.get("segmentation")
        if isinstance(seg, list) and seg:
            for poly in seg:
                pts = [(poly[i] * sx, poly[i + 1] * sy) for i in range(0, len(poly) - 1, 2)]
                if len(pts) >= 3:
                    doc.items.append(AnnItem(id=nid, cls=cid, type="polygon", pts=pts, source="import"))
                    nid += 1
                    n += 1
        elif a.get("bbox"):
            x, y, w, h = a["bbox"]
            doc.items.append(AnnItem(id=nid, cls=cid, type="bbox", source="import",
                                     pts=[(x * sx, y * sy), ((x + w) * sx, (y + h) * sy)]))
            nid += 1
            n += 1
    return n


def import_detections(doc: AnnDoc, ds, fc: dict, min_score: float = 0.0, replace_auto: bool = False) -> int:
    """Turn a detection job's GeoJSON into editable annotations (auto annotation)."""
    if replace_auto:
        doc.items = [i for i in doc.items if i.source != "auto"]
    geo = Georef(ds)
    task = fc.get("metadata", {}).get("task", "detect")
    n, nid = 0, _next_id(doc)
    for f in fc["features"]:
        p = f["properties"]
        if p.get("score", 1) < min_score:
            continue
        ring = f["geometry"]["coordinates"][0][:-1]
        pts = geo.to_pixel(ring)
        cid = ensure_class(doc, p.get("class_name", "object"))
        if task == "detect":
            xs, ys = zip(*pts)
            item = AnnItem(id=nid, cls=cid, type="bbox", pts=[(min(xs), min(ys)), (max(xs), max(ys))],
                           source="auto", score=p.get("score"))
        else:
            item = AnnItem(id=nid, cls=cid, type="polygon", pts=pts, source="auto", score=p.get("score"))
        doc.items.append(item)
        nid += 1
        n += 1
    return n


# ---------------------------------------------------------------- semantic rasterisation

def rasterize(doc: AnnDoc, x0: float, y0: float, w: int, h: int, scale: float = 1.0) -> np.ndarray:
    """Class-index mask (0 = unlabelled, k = k-th class, 1-based) for the pixel window starting at
    (x0, y0), sampled every `scale` raster pixels. Later items paint over earlier ones."""
    index = {c.id: i + 1 for i, c in enumerate(doc.classes)}
    mask = np.zeros((h, w), np.uint8)
    x1, y1 = x0 + w * scale, y0 + h * scale
    shift = 4
    k = (1 << shift) / scale
    for it in doc.items:
        if it.type == "point":
            continue
        a = np.asarray(it.pts, dtype=np.float64)
        pad = (it.width or 0) / 2
        if a[:, 0].max() + pad < x0 or a[:, 0].min() - pad > x1 or a[:, 1].max() + pad < y0 or a[:, 1].min() - pad > y1:
            continue
        value = 0 if it.erase else index.get(it.cls)
        if value is None:
            continue
        if it.type == "bbox":
            a = np.array([a[0], [a[1][0], a[0][1]], a[1], [a[0][0], a[1][1]]])
        pts = np.round((a - (x0, y0)) * k).astype(np.int32)
        if it.type == "stroke":
            thick = max(1, int(round(it.width / scale)))
            if len(pts) == 1:
                cv2.circle(mask, tuple(pts[0]), max(1, thick // 2) << shift, int(value), -1, cv2.LINE_8, shift)
            else:
                cv2.polylines(mask, [pts], False, int(value), thick, cv2.LINE_8, shift)
        else:
            cv2.fillPoly(mask, [pts], int(value), cv2.LINE_8, shift)
    return mask


def _rgba_lut(doc: AnnDoc, alpha: int = 150) -> np.ndarray:
    lut = np.zeros((256, 4), np.uint8)
    for i, c in enumerate(doc.classes, start=1):
        r, g, b = (int(c.color[j:j + 2], 16) for j in (1, 3, 5))
        lut[i] = (b, g, r, alpha)  # BGRA for OpenCV
    return lut


def render_mask_tile(doc: AnnDoc, raster_path, z: int, x: int, y: int, size: int = 256) -> bytes | None:
    """Semantic mask as an exact Web Mercator tile, for WYSIWYG preview on the map."""
    if not any(i.type != "point" for i in doc.items):
        return None
    left, bottom, right, top = tile_bounds(z, x, y)
    with rasterio.open(raster_path) as ds:
        t = Transformer.from_crs(WEB_MERCATOR, ds.crs, always_xy=True)
        u = np.linspace(0, 1, 9)
        ex = np.concatenate([left + (right - left) * u, np.full(9, right), right - (right - left) * u, np.full(9, left)])
        ey = np.concatenate([np.full(9, top), top - (top - bottom) * u, np.full(9, bottom), bottom + (top - bottom) * u])
        xs, ys = t.transform(ex, ey)
        inv = ~ds.transform
        cols, rows = zip(*[inv @ (a, b) for a, b in zip(xs, ys)])
        px0, px1 = max(0.0, min(cols) - 2), min(float(ds.width), max(cols) + 2)
        py0, py1 = max(0.0, min(rows) - 2), min(float(ds.height), max(rows) + 2)
        if px1 <= px0 or py1 <= py0:
            return None
        scale = max(1.0, (px1 - px0) / (size * 2), (py1 - py0) / (size * 2))
        w, h = int(np.ceil((px1 - px0) / scale)), int(np.ceil((py1 - py0) / scale))
        mask = rasterize(doc, px0, py0, w, h, scale)
        if not mask.any():
            return None
        src_transform = ds.transform @ Affine.translation(px0, py0) @ Affine.scale(scale)
        out = np.zeros((size, size), np.uint8)
        reproject(mask, out, src_transform=src_transform, src_crs=ds.crs,
                  dst_transform=from_bounds(left, bottom, right, top, size, size), dst_crs=WEB_MERCATOR,
                  resampling=Resampling.nearest, src_nodata=0, dst_nodata=0)
    if not out.any():
        return None
    ok, buf = cv2.imencode(".png", _rgba_lut(doc)[out])
    return buf.tobytes()


def write_mask_geotiff(doc: AnnDoc, raster_path, out_path, block: int = 2048) -> None:
    """Full-resolution class-index mask aligned to the raster grid, with a colour table."""
    with rasterio.open(raster_path) as ds:
        profile = dict(driver="GTiff", width=ds.width, height=ds.height, count=1, dtype="uint8", crs=ds.crs,
                       transform=ds.transform, compress="deflate", tiled=True, blockxsize=512, blockysize=512,
                       nodata=0, BIGTIFF="IF_SAFER")
    with rasterio.open(out_path, "w", **profile) as dst:
        for r in range(0, profile["height"], block):
            for c in range(0, profile["width"], block):
                w, h = min(block, profile["width"] - c), min(block, profile["height"] - r)
                dst.write(rasterize(doc, c, r, w, h), 1, window=Window(c, r, w, h))
        cmap = {0: (0, 0, 0, 0)}
        for i, cl in enumerate(doc.classes, start=1):
            cmap[i] = tuple(int(cl.color[j:j + 2], 16) for j in (1, 3, 5)) + (255,)
        dst.write_colormap(1, cmap)
        dst.update_tags(1, **{f"class_{i}": cl.name for i, cl in enumerate(doc.classes, start=1)})


# ---------------------------------------------------------------- dataset export

class ExportOptions(BaseModel):
    format: Literal["geojson", "yolo-detect", "yolo-seg", "yolo-obb", "coco", "semantic", "mask-geotiff"]
    chip_size: int = Field(640, ge=64, le=4096)
    overlap: float = Field(0.1, ge=0.0, le=0.5)
    include_empty: bool = False
    image_format: Literal["jpg", "png"] = "jpg"
    point_box: float = Field(16.0, gt=0, le=512)
    min_visibility: float = Field(0.3, ge=0.0, le=1.0)
    val_ratio: float = Field(0.2, ge=0.0, le=0.9)
    seed: int = 0


def _norm(v, size):
    return f"{min(max(v / size, 0.0), 1.0):.6f}"


def _labels_for_chip(doc, geoms, cls_index, chip, opts):
    """Per-chip labels: list of (class_index, clipped polygons, item)."""
    c0, r0, s = chip
    window = box(c0, r0, c0 + s, r0 + s)
    out = []
    for it, g in geoms:
        if it.type == "stroke" or it.erase or it.cls not in cls_index:
            continue
        if not g.intersects(window):
            continue
        clipped = g.intersection(window)
        if g.area > 0 and clipped.area / g.area < opts.min_visibility:
            continue
        polys = _polygons(clipped)
        if polys:
            out.append((cls_index[it.cls], polys, it))
    return out


def export_dataset(doc: AnnDoc, raster_path, opts: ExportOptions, stretch=None) -> Path:
    """Write a training dataset (ZIP) of image chips and labels. Returns the ZIP path."""
    fd, zpath = tempfile.mkstemp(suffix=".zip")
    os.close(fd)
    zpath = Path(zpath)
    cls_index = {c.id: i for i, c in enumerate(doc.classes)}
    names = [c.name for c in doc.classes]
    with rasterio.open(raster_path) as ds, zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        if opts.format == "geojson":
            zf.writestr("annotations.geojson", json.dumps(to_geojson(doc, ds)))
            return zpath
        if opts.format == "mask-geotiff":
            tmp = zpath.with_suffix(".tif")
            write_mask_geotiff(doc, raster_path, tmp)
            zf.write(tmp, "mask.tif")
            tmp.unlink()
            zf.writestr("classes.json", json.dumps([{"value": i + 1, **c.model_dump()} for i, c in enumerate(doc.classes)], indent=2))
            return zpath
        bands = default_bands(ds.count)
        stretch = stretch or compute_stretch(ds, bands)
        geoms = [(it, item_geometry(it, opts.point_box)) for it in doc.items]
        if opts.format != "semantic":
            geoms = [(it, g) for it, g in geoms if it.type != "stroke"]
            if opts.format != "yolo-detect":  # segmentation formats need real outlines
                geoms = [(it, g) for it, g in geoms if it.type != "point" or opts.format == "coco"]
        rng = random.Random(opts.seed)
        coco = {"images": [], "annotations": [], "categories": [{"id": i + 1, "name": n} for i, n in enumerate(names)]}
        bounds = np.array([g.bounds for _, g in geoms]) if geoms else np.zeros((0, 4))
        s = opts.chip_size
        stem = Path(raster_path).parent.name[:8]
        count = 0
        for c0, r0 in tile_grid(ds.width, ds.height, s, opts.overlap):
            hit = (bounds[:, 2] > c0) & (bounds[:, 0] < c0 + s) & (bounds[:, 3] > r0) & (bounds[:, 1] < r0 + s) \
                if len(bounds) else np.zeros(0, bool)
            if not hit.any() and not opts.include_empty:
                continue
            local = [geoms[i] for i in np.flatnonzero(hit)]
            rgb, valid = read_rgb(ds, c0, r0, s, bands, stretch)
            if valid.mean() < 0.05:
                continue
            split = "val" if rng.random() < opts.val_ratio else "train"
            name = f"{stem}_{c0}_{r0}"
            img_name = f"{name}.{opts.image_format}"
            ext = ".jpg" if opts.image_format == "jpg" else ".png"
            ok, buf = cv2.imencode(ext, rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95] if ext == ".jpg" else [])
            zf.writestr(f"images/{split}/{img_name}", buf.tobytes())
            count += 1
            if opts.format == "semantic":
                mask = rasterize(doc, c0, r0, s, s)
                pal = Image.fromarray(mask, "P")
                palette = [0, 0, 0]
                for cl in doc.classes:
                    palette += [int(cl.color[j:j + 2], 16) for j in (1, 3, 5)]
                pal.putpalette(palette + [0] * (768 - len(palette)))
                bio = io.BytesIO()
                pal.save(bio, "PNG")
                zf.writestr(f"masks/{split}/{name}.png", bio.getvalue())
                continue
            labels = _labels_for_chip(doc, local, cls_index, (c0, r0, s), opts)
            if opts.format == "coco":
                img_id = len(coco["images"]) + 1
                coco["images"].append({"id": img_id, "file_name": f"{split}/{img_name}", "width": s, "height": s})
                for k, polys, it in labels:
                    for poly in polys:
                        x0, y0, x1, y1 = poly.bounds
                        seg = [] if it.type == "point" else \
                            [round(v, 2) for xy in list(poly.exterior.coords)[:-1] for v in (xy[0] - c0, xy[1] - r0)]
                        coco["annotations"].append({
                            "id": len(coco["annotations"]) + 1, "image_id": img_id, "category_id": k + 1,
                            "bbox": [round(x0 - c0, 2), round(y0 - r0, 2), round(x1 - x0, 2), round(y1 - y0, 2)],
                            "area": round(poly.area, 2), "segmentation": [seg] if seg else [], "iscrowd": 0,
                            "score": it.score})
                continue
            lines = []
            for k, polys, it in labels:
                if opts.format == "yolo-detect":
                    x0, y0, x1, y1 = (min(p.bounds[0] for p in polys), min(p.bounds[1] for p in polys),
                                      max(p.bounds[2] for p in polys), max(p.bounds[3] for p in polys))
                    lines.append(f"{k} {_norm((x0 + x1) / 2 - c0, s)} {_norm((y0 + y1) / 2 - r0, s)} "
                                 f"{_norm(x1 - x0, s)} {_norm(y1 - y0, s)}")
                elif opts.format == "yolo-obb":
                    pts = np.vstack([np.asarray(p.exterior.coords) for p in polys]).astype(np.float32)
                    rect = cv2.boxPoints(cv2.minAreaRect(pts))
                    lines.append(f"{k} " + " ".join(f"{_norm(x - c0, s)} {_norm(y - r0, s)}" for x, y in rect))
                else:  # yolo-seg
                    for p in polys:
                        coords = list(p.exterior.coords)[:-1]
                        lines.append(f"{k} " + " ".join(f"{_norm(x - c0, s)} {_norm(y - r0, s)}" for x, y in coords))
            zf.writestr(f"labels/{split}/{name}.txt", "\n".join(lines) + ("\n" if lines else ""))
        if opts.format == "coco":
            zf.writestr("annotations.json", json.dumps(coco))
        elif opts.format == "semantic":
            zf.writestr("classes.json", json.dumps([{"value": 0, "name": "background"}] +
                                                   [{"value": i + 1, **c.model_dump()} for i, c in enumerate(doc.classes)], indent=2))
        else:
            yaml = "path: .\ntrain: images/train\nval: images/val\n"
            if opts.format == "yolo-obb":
                yaml += "task: obb\n"
            elif opts.format == "yolo-seg":
                yaml += "task: segment\n"
            yaml += "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names))
            zf.writestr("data.yaml", yaml)
        zf.writestr("README.txt", f"Exported {count} chips of {s}px from a {ds.width}x{ds.height} raster "
                                  f"({ds.crs}). Chip file names are <raster>_<col>_<row> pixel offsets.\n")
    return zpath
