"""REST API and static web client."""

from __future__ import annotations

import csv
import io
import json
import re
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__, config
from .jobs import JobManager
from .models import ModelManager, read_onnx_metadata
from .pipeline import DetectionParams
from .raster import inspect, render_preview

RASTER_EXT = {".tif", ".tiff", ".jp2", ".img", ".vrt"}
ID_RE = re.compile(r"^[0-9a-f]{32}$")

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


def _raster_source(raster_id: str) -> Path:
    meta = _raster_meta(raster_id)
    return _raster_dir(raster_id) / meta["stored_name"]


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
async def upload_raster(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in RASTER_EXT:
        raise HTTPException(400, f"Unsupported raster type '{ext}'. Use GeoTIFF.")
    rid = uuid.uuid4().hex
    d = config.RASTER_DIR / rid
    d.mkdir(parents=True)
    src = d / f"source{ext}"
    with src.open("wb") as fh:
        shutil.copyfileobj(file.file, fh, length=8 * 1024 * 1024)
    try:
        meta = inspect(src)
        meta["preview_corners"] = render_preview(src, d / "preview.png", config.PREVIEW_MAX_SIZE)
    except Exception as e:
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(400, f"Could not read raster: {e}")
    meta.update(id=rid, name=file.filename, stored_name=src.name, size_bytes=src.stat().st_size)
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
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


@app.get("/api/rasters/{raster_id}/preview.png")
def raster_preview(raster_id: str):
    return FileResponse(_raster_dir(raster_id) / "preview.png", media_type="image/png")


@app.delete("/api/rasters/{raster_id}")
def delete_raster(raster_id: str):
    shutil.rmtree(_raster_dir(raster_id))
    return {"deleted": raster_id}


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


@app.post("/api/jobs")
def create_job(req: JobRequest):
    path = _raster_source(req.raster_id)
    meta = _raster_meta(req.raster_id)
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
