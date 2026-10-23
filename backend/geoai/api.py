"""REST API and static web client."""

from __future__ import annotations

import csv
import io
import json
import re
import shutil
import uuid
from pathlib import Path

import rasterio
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__, annotations as ann, config
from .georef import control_grid, ingest, inspect, set_crs, split_upload
from .jobs import JobManager
from .models import ModelManager, read_onnx_metadata
from .pipeline import DetectionParams
from .raster import compute_stretch, render_tile

ID_RE = re.compile(r"^[0-9a-f]{32}$")
BANDS_RE = re.compile(r"^\d{1,3}-\d{1,3}-\d{1,3}$")

app = FastAPI(title="AI Platform — GeoAI", version=__version__)
models = ModelManager()
jobs = JobManager(models)


def _check_id(value: str) -> str:
    if not ID_RE.match(value):
        raise HTTPException(400, "Invalid id")
    return value


def _raster_dir(raster_id: str) -> Path:
    d = config.RASTER_DIR / _check_id(raster_id)
    if not d.exists():
        raise HTTPException(404, "Raster not found")
    return d


def _raster_meta(raster_id: str) -> dict:
    return json.loads((_raster_dir(raster_id) / "meta.json").read_text(encoding="utf-8"))


def _save_meta(raster_id: str, meta: dict):
    (_raster_dir(raster_id) / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def _raster_source(raster_id: str) -> Path:
    return _raster_dir(raster_id) / _raster_meta(raster_id)["stored_name"]


def _ready(raster_id: str) -> dict:
    meta = _raster_meta(raster_id)
    if meta.get("status") != "ready":
        raise HTTPException(409, "Raster needs a valid CRS first (see its warnings).")
    return meta


# ---------------------------------------------------------------- health / models

@app.get("/api/health")
def health():
    import onnxruntime as ort

    return {"status": "ok", "version": __version__, "providers": ort.get_available_providers()}


@app.get("/api/models")
def list_models():
    return [s.to_dict() for s in models.specs().values()]


@app.post("/api/models")
async def upload_model(file: UploadFile = File(...), name: str = Form(""), task: str = Form(""),
                       names: str = Form(""), imgsz: int = Form(0)):
    """Register a custom Ultralytics ONNX model. Metadata embedded in the file is used when present."""
    if not file.filename.lower().endswith(".onnx"):
        raise HTTPException(400, "Expected an .onnx file")
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(file.filename).stem)[:60] or "model"
    dest = config.CUSTOM_MODEL_DIR / f"{stem}.onnx"
    with dest.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)
    try:
        meta = read_onnx_metadata(dest)
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, f"Not a valid ONNX model: {e}")
    info = {"name": name or stem, **meta}
    if task:
        info["task"] = task
    if names:
        info["names"] = [n.strip() for n in names.split(",") if n.strip()]
    if imgsz:
        info["imgsz"] = imgsz
    if info.get("task") not in ("detect", "obb", "segment"):
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "Task must be detect, obb or segment (not found in model metadata).")
    dest.with_suffix(".json").write_text(json.dumps(info), encoding="utf-8")
    model_id = f"custom-{stem}"
    models.evict(model_id)
    return models.specs()[model_id].to_dict()


# ---------------------------------------------------------------- rasters

@app.post("/api/rasters")
async def upload_raster(files: list[UploadFile] = File(...)):
    """Upload a raster with optional sidecars (.tfw/.jgw world file, .prj, .aux.xml, .ovr)."""
    try:
        main, sidecars = split_upload([f.filename for f in files])
    except ValueError as e:
        raise HTTPException(400, str(e))
    rid = uuid.uuid4().hex
    d = config.RASTER_DIR / rid
    up = d / "upload"
    up.mkdir(parents=True)
    by_name = {f.filename: f for f in files}
    src = up / f"upload{Path(main).suffix.lower()}"
    with src.open("wb") as fh:
        shutil.copyfileobj(by_name[main].file, fh, length=8 * 1024 * 1024)
    for name, suffix in sidecars.items():
        with (up / f"upload{suffix}").open("wb") as fh:
            shutil.copyfileobj(by_name[name].file, fh)
    try:
        stored = ingest(src, d / "source.tif")
        meta = inspect(stored)
    except Exception as e:
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(400, f"Could not read raster: {e}")
    shutil.rmtree(up, ignore_errors=True)
    meta.update(id=rid, name=Path(main).name, stored_name=stored.name, size_bytes=stored.stat().st_size,
                sidecars=sorted(Path(n).name for n in sidecars))
    _save_meta(rid, meta)
    return meta


@app.get("/api/rasters")
def list_rasters():
    out = []
    for d in sorted(config.RASTER_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        f = d / "meta.json"
        if f.exists():
            out.append(json.loads(f.read_text(encoding="utf-8")))
    return out


@app.get("/api/rasters/{raster_id}")
def get_raster(raster_id: str):
    return _raster_meta(raster_id)


class CrsRequest(BaseModel):
    crs: str = Field(min_length=3, max_length=20000, description="EPSG code (e.g. EPSG:32638) or WKT")


@app.post("/api/rasters/{raster_id}/crs")
def assign_crs(raster_id: str, req: CrsRequest):
    """Assign or correct the CRS. Pixel coordinates are kept and reinterpreted in the new CRS."""
    d = _raster_dir(raster_id)
    old = _raster_meta(raster_id)
    text = req.crs.strip()
    if text.isdigit():
        text = f"EPSG:{text}"
    try:
        set_crs(d / old["stored_name"], text)
        meta = inspect(d / old["stored_name"])
    except Exception as e:
        raise HTTPException(400, f"Invalid CRS: {e}")
    shutil.rmtree(d / "tiles", ignore_errors=True)
    for k in ("id", "name", "stored_name", "size_bytes", "sidecars"):
        meta[k] = old.get(k)
    _save_meta(raster_id, meta)
    return meta


@app.get("/api/rasters/{raster_id}/tiles/{z}/{x}/{y}.png")
def raster_tile(raster_id: str, z: int, x: int, y: int, bands: str = "1-2-3"):
    """Exact Web Mercator XYZ tiles rendered from the raster, cached on disk."""
    if not (0 <= z <= 24 and 0 <= x < 2 ** z and 0 <= y < 2 ** z) or not BANDS_RE.match(bands):
        raise HTTPException(400, "Bad tile request")
    meta = _ready(raster_id)
    b = [int(v) for v in bands.split("-")]
    if not all(1 <= v <= meta["count"] for v in b):
        b = meta["default_bands"]
        bands = "-".join(map(str, b))
    d = _raster_dir(raster_id)
    cache = d / "tiles" / bands / str(z) / str(x) / f"{y}.png"
    if cache.exists():
        data = cache.read_bytes()
        return Response(data, media_type="image/png") if data else Response(status_code=204)
    stretch = meta.get("stretch", {}).get(bands)
    if stretch is None:
        with rasterio.open(d / meta["stored_name"]) as ds:
            stretch = compute_stretch(ds, b)
        meta.setdefault("stretch", {})[bands] = stretch
        _save_meta(raster_id, meta)
    png = render_tile(d / meta["stored_name"], z, x, y, b, stretch)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(png or b"")
    return Response(png, media_type="image/png") if png else Response(status_code=204)


@app.get("/api/rasters/{raster_id}/georef")
def raster_georef(raster_id: str):
    _ready(raster_id)
    return control_grid(_raster_source(raster_id))


@app.delete("/api/rasters/{raster_id}")
def delete_raster(raster_id: str):
    shutil.rmtree(_raster_dir(raster_id))
    return {"deleted": raster_id}


# ---------------------------------------------------------------- annotations

@app.get("/api/rasters/{raster_id}/annotations")
def get_annotations(raster_id: str):
    return ann.load_doc(_raster_dir(raster_id))


@app.put("/api/rasters/{raster_id}/annotations")
def put_annotations(raster_id: str, doc: ann.AnnDoc):
    d = _raster_dir(raster_id)
    current = ann.load_doc(d)
    if doc.version != current.version:
        raise HTTPException(409, f"Annotations changed elsewhere (server version {current.version}). Reload.")
    return ann.save_doc(d, doc)


@app.get("/api/rasters/{raster_id}/annotations/mask/{z}/{x}/{y}.png")
def annotation_mask_tile(raster_id: str, z: int, x: int, y: int):
    if not (0 <= z <= 24 and 0 <= x < 2 ** z and 0 <= y < 2 ** z):
        raise HTTPException(400, "Bad tile request")
    _ready(raster_id)
    d = _raster_dir(raster_id)
    png = ann.render_mask_tile(ann.load_doc(d), _raster_source(raster_id), z, x, y)
    headers = {"Cache-Control": "no-store"}
    return Response(png, media_type="image/png", headers=headers) if png else Response(status_code=204, headers=headers)


@app.post("/api/rasters/{raster_id}/annotations/import")
async def import_annotations(raster_id: str, file: UploadFile = File(...), names: str = Form("")):
    """Merge labels from GeoJSON, YOLO (.txt, normalised to the whole raster) or COCO (.json)."""
    _ready(raster_id)
    d = _raster_dir(raster_id)
    doc = ann.load_doc(d)
    raw = (await file.read()).decode("utf-8-sig", errors="replace")
    fname = file.filename.lower()
    with rasterio.open(_raster_source(raster_id)) as ds:
        try:
            if fname.endswith(".txt"):
                n = ann.import_yolo(doc, ds, raw, [s.strip() for s in names.split(",") if s.strip()])
            else:
                data = json.loads(raw)
                n = ann.import_coco(doc, ds, data) if "annotations" in data and "images" in data \
                    else ann.import_geojson(doc, ds, data)
        except (ValueError, KeyError, TypeError) as e:
            raise HTTPException(400, f"Could not import labels: {e}")
    doc = ann.save_doc(d, doc)
    return {"imported": n, "doc": doc}


class FromJobRequest(BaseModel):
    job_id: str
    min_score: float = Field(0.0, ge=0.0, le=1.0)
    replace_auto: bool = True


@app.post("/api/rasters/{raster_id}/annotations/from-job")
def annotations_from_job(raster_id: str, req: FromJobRequest):
    """Auto annotation: convert a finished detection job into editable labels."""
    _ready(raster_id)
    job = _job(req.job_id)
    if job.raster_id != raster_id:
        raise HTTPException(400, "Job belongs to another raster")
    fc = json.loads(_result(req.job_id).read_text(encoding="utf-8"))
    d = _raster_dir(raster_id)
    doc = ann.load_doc(d)
    with rasterio.open(_raster_source(raster_id)) as ds:
        n = ann.import_detections(doc, ds, fc, req.min_score, req.replace_auto)
    doc = ann.save_doc(d, doc)
    return {"imported": n, "doc": doc}


@app.post("/api/rasters/{raster_id}/annotations/export")
def export_annotations(raster_id: str, opts: ann.ExportOptions, background: BackgroundTasks):
    meta = _ready(raster_id)
    d = _raster_dir(raster_id)
    doc = ann.load_doc(d)
    if not doc.items:
        raise HTTPException(400, "There are no annotations to export.")
    stretch = next(iter(meta.get("stretch", {}).values()), None)
    zpath = ann.export_dataset(doc, _raster_source(raster_id), opts, stretch)
    background.add_task(zpath.unlink, missing_ok=True)
    stem = Path(meta["name"]).stem[:40] or "raster"
    return FileResponse(zpath, media_type="application/zip", filename=f"{stem}_{opts.format}.zip")


# ---------------------------------------------------------------- jobs

class JobRequest(BaseModel):
    raster_id: str
    model_id: str
    conf: float = Field(0.25, ge=0.01, le=1.0)
    iou: float = Field(0.45, ge=0.05, le=0.95)
    classes: list[int] | None = None
    overlap: float = Field(0.2, ge=0.0, le=0.5)
    bands: list[int] | None = None
    simplify_px: float = Field(1.0, ge=0.0, le=10.0)
    class_agnostic_nms: bool = False
    scale: float = Field(1.0, ge=0.25, le=4.0)
    window: list[float] | None = Field(None, min_length=4, max_length=4)


@app.post("/api/jobs")
def create_job(req: JobRequest):
    meta = _ready(req.raster_id)
    path = _raster_source(req.raster_id)
    if req.bands and (len(req.bands) != 3 or not all(1 <= b <= meta["count"] for b in req.bands)):
        raise HTTPException(400, f"bands must be three band numbers between 1 and {meta['count']}")
    params = DetectionParams(**req.model_dump(exclude={"raster_id", "model_id"}))
    try:
        job = jobs.submit(req.raster_id, path, req.model_id, params)
    except KeyError:
        raise HTTPException(404, "Model not found")
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(400, str(e))
    return job.to_dict()


@app.get("/api/jobs")
def list_jobs():
    return sorted((j.to_dict() for j in jobs.jobs.values()), key=lambda j: -j["created"])


def _job(job_id: str):
    job = jobs.jobs.get(_check_id(job_id))
    if job is None:
        raise HTTPException(404, "Job not found")
    return job


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    return _job(job_id).to_dict()


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    jobs.cancel(_job(job_id).id)
    return {"cancelling": job_id}


def _result(job_id: str) -> Path:
    job = _job(job_id)
    path = jobs.result_path(job.id)
    if job.status != "done" or not path.exists():
        raise HTTPException(409, f"Job is {job.status}")
    return path


@app.get("/api/jobs/{job_id}/result.geojson")
def job_result(job_id: str):
    return FileResponse(_result(job_id), media_type="application/geo+json")


def _wkt(geom) -> str:
    ring = ", ".join(f"{x} {y}" for x, y in geom["coordinates"][0])
    return f"POLYGON (({ring}))"


@app.get("/api/jobs/{job_id}/export")
def export_job(job_id: str, format: str = "geojson"):
    path = _result(job_id)
    name = f"geoai_{job_id[:8]}"
    if format == "geojson":
        return FileResponse(path, media_type="application/geo+json", filename=f"{name}.geojson")
    if format == "csv":
        fc = json.loads(path.read_text(encoding="utf-8"))
        cols = sorted({k for f in fc["features"] for k in f["properties"]}, key=lambda k: (k != "id", k))
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(cols + ["wkt"])
        for f in fc["features"]:
            w.writerow([f["properties"].get(c, "") for c in cols] + [_wkt(f["geometry"])])
        return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                                 headers={"Content-Disposition": f'attachment; filename="{name}.csv"'})
    raise HTTPException(400, "format must be geojson or csv")


# ---------------------------------------------------------------- web client

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


app.mount("/", StaticFiles(directory=config.FRONTEND_DIR, html=True), name="web")
