"""Background inference jobs with progress, cancellation and on-disk results."""

from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

from . import config
from .models import ModelManager
from .pipeline import Cancelled, DetectionParams, run_detection


@dataclass
class Job:
    id: str
    raster_id: str
    model_id: str
    params: dict
    status: str = "queued"  # queued | running | done | error | cancelled
    progress: float = 0.0
    message: str = ""
    created: float = field(default_factory=time.time)
    finished: float | None = None
    summary: dict | None = None
    error: str | None = None

    def to_dict(self):
        return asdict(self)


class JobManager:
    def __init__(self, models: ModelManager, workers: int = 1):
        self.models = models
        self.jobs: dict[str, Job] = {}
        self._cancel: set[str] = set()
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="geoai")
        self._load_existing()

    def _load_existing(self):
        for f in config.JOB_DIR.glob("*.json"):
            try:
                job = Job(**json.loads(f.read_text(encoding="utf-8")))
            except (TypeError, ValueError):
                continue
            if job.status in ("queued", "running"):
                job.status, job.error = "error", "Server restarted before the job finished."
            self.jobs[job.id] = job

    def result_path(self, job_id: str):
        return config.JOB_DIR / f"{job_id}.geojson"

    def _save(self, job: Job):
        (config.JOB_DIR / f"{job.id}.json").write_text(json.dumps(job.to_dict()), encoding="utf-8")

    def submit(self, raster_id: str, raster_path, model_id: str, params: DetectionParams) -> Job:
        self.models.get(model_id)  # fail fast on a missing or invalid model
        job = Job(id=uuid.uuid4().hex, raster_id=raster_id, model_id=model_id, params=asdict(params))
        with self._lock:
            self.jobs[job.id] = job
        self._save(job)
        self._pool.submit(self._run, job, raster_path, params)
        return job

    def cancel(self, job_id: str):
        self._cancel.add(job_id)

    def _run(self, job: Job, raster_path, params: DetectionParams):
        job.status, job.message = "running", "Loading model"

        def progress(frac, msg):
            job.progress, job.message = round(frac, 4), msg

        try:
            model = self.models.get(job.model_id)
            fc = run_detection(raster_path, model, params, progress, lambda: job.id in self._cancel)
            fc["metadata"]["raster_id"] = job.raster_id
            self.result_path(job.id).write_text(json.dumps(fc), encoding="utf-8")
            meta = fc["metadata"]
            job.summary = {"count": len(fc["features"]), "counts": meta["counts"], "elapsed_s": meta["elapsed_s"],
                           "tiles": meta["tiles_processed"], "task": meta["task"]}
            job.status, job.progress, job.message = "done", 1.0, f"{len(fc['features'])} objects"
        except Cancelled:
            job.status, job.message = "cancelled", "Cancelled"
        except Exception as e:  # surface any failure to the client
            job.status, job.error = "error", f"{type(e).__name__}: {e}"
            traceback.print_exc()
        finally:
            job.finished = time.time()
            self._cancel.discard(job.id)
            self._save(job)
