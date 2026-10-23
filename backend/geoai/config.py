import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("GEOAI_DATA_DIR", ROOT / "data"))
MODELS_DIR = Path(os.environ.get("GEOAI_MODELS_DIR", ROOT / "models"))
FRONTEND_DIR = ROOT / "frontend"

RASTER_DIR = DATA_DIR / "rasters"
JOB_DIR = DATA_DIR / "jobs"
CUSTOM_MODEL_DIR = MODELS_DIR / "custom"

PREVIEW_MAX_SIZE = int(os.environ.get("GEOAI_PREVIEW_MAX_SIZE", 2048))

for d in (RASTER_DIR, JOB_DIR, CUSTOM_MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)
