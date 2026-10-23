# AI Platform — GeoAI & Aerial Object Detection

A web application for running AI object detection and segmentation on aerial and
satellite GeoTIFFs. Results come back as georeferenced GIS layers. Inference runs
**locally with ONNX Runtime**, so no internet connection is needed once models are installed.

It re-implements the GeoAI capabilities described by [EarthMapper](https://github.com/AliAlkazraji/EarthMapper)
as a browser-based platform. See [docs/PHASES.md](docs/PHASES.md) for the analysis and roadmap.

## Status

| Capability | Status |
|---|---|
| Aerial oriented-box detection (DOTA: planes, ships, tanks, vehicles, bridges, harbors, airports, helipads…) | ✅ Phase 1 |
| Tiled native-resolution inference, overlap handling, cross-tile rotated NMS, class selection, progress/cancel | ✅ Phase 1 |
| Custom Ultralytics ONNX models (detect / obb / segment), auto-configured from metadata | ✅ Phase 1 |
| GIS output: georeferenced polygons, geodesic area/length/width/heading, GeoJSON + CSV export | ✅ Phase 1 |
| Exact Web Mercator tiles from GeoTIFF (native resolution, overviews), CRS diagnostics and repair, sidecar upload | ✅ Phase 1.5 |
| Label annotation: detection / instance / semantic modes, editing, undo/redo, auto annotation, YOLO / COCO / mask / GeoJSON export, import | ✅ Phase 1.5 |
| Tree detection, building segmentation (engine ready; model packaging + GSD resampling) | 🔜 Phase 2 |
| SAM 2 point-prompt segmentation | 🔜 Phase 3 |
| SAM 3 text-prompt segmentation (optional) | 🔜 Phase 4 |
| GPKG/SHP/KML export, AOI, symbology, spatial analysis | 🔜 Phase 5 |

## Quick start (Windows)

Double-click **`run.bat`** in the `AI_Platfrom` folder (or run `C:\path\to\AI_Platfrom\run.ps1` from any
directory). The first run creates the Python environment. The launcher skips ports already used by other
programs (Docker often holds 8000), prints the address, and opens the browser. Keep its window open while
you use the app: closing it stops the server, and the page then shows "Server not reachable".

```powershell
.venv\Scripts\python scripts\export_models.py yolo26s-obb-dota2   # 39 MB built-in aerial model
```

Then drop a GeoTIFF into the sidebar, choose a model and classes, and press **Run detection**.
To try a plain JPG/PNG, georeference it first:

```powershell
.venv\Scripts\python scripts\georef_image.py photo.jpg photo.tif --lon 44.36 --lat 33.31 --gsd 0.3
```

### Other models

```powershell
pip install ultralytics huggingface_hub
python scripts\export_models.py yolo11n-obb-dota1        # Ultralytics DOTA v1 (AGPL-3.0)
python scripts\export_models.py building-yolov8n-seg     # keremberke building segmentation
python scripts\export_models.py tree-yolov8 --weights C:\path\to\TreeSensing\best.pt
```

You can also upload any Ultralytics-exported `.onnx` from the UI (**Add custom ONNX model**).

## Layout

```
backend/geoai/   raster.py  georef.py  tiling.py  models.py  postprocess.py  pipeline.py  jobs.py
                 annotations.py  api.py
backend/tests/   unit + end-to-end tests (generated ONNX models, synthetic GeoTIFFs)
frontend/        static MapLibre client (vendored, works offline): app.js (detect), annotate.js (labelling)
models/          registry.json + downloaded/exported .onnx files (git-ignored)
scripts/         export_models.py, georef_image.py
data/            uploaded rasters and job results (git-ignored)
```

## API

| Method | Path | |
|---|---|---|
| GET | `/api/models` | Registry with availability, task, classes, license |
| POST | `/api/models` | Upload custom ONNX (`file`, `name`, `task`, `names`) |
| POST | `/api/rasters` | Upload a raster (`files`: GeoTIFF + optional .tfw/.prj/.aux.xml) → metadata, CRS warnings |
| GET | `/api/rasters/{id}/tiles/{z}/{x}/{y}.png?bands=1-2-3` | Exact Web Mercator tiles |
| POST | `/api/rasters/{id}/crs` | Assign or correct the CRS (`{"crs": "EPSG:32638"}`) |
| GET/PUT | `/api/rasters/{id}/annotations` | Label document (pixel coordinates, versioned) |
| POST | `/api/rasters/{id}/annotations/export` | `{format: yolo-detect\|yolo-obb\|yolo-seg\|coco\|semantic\|mask-geotiff\|geojson, chip_size, …}` → ZIP |
| POST | `/api/rasters/{id}/annotations/import` | GeoJSON, YOLO `.txt` or COCO `.json` |
| POST | `/api/rasters/{id}/annotations/from-job` | Auto annotation from a detection job |
| POST | `/api/jobs` | `{raster_id, model_id, conf, iou, classes, overlap, bands, scale, window, class_agnostic_nms}` |
| GET | `/api/jobs/{id}` | Status / progress / summary |
| POST | `/api/jobs/{id}/cancel` | Cancel |
| GET | `/api/jobs/{id}/export?format=geojson\|csv` | Download results |

Interactive docs: `http://127.0.0.1:8000/docs`.

## Tests

```powershell
cd backend; ..\.venv\Scripts\python -m pytest
```

## Licensing

Platform code: choose your license. Third-party frameworks, models and datasets keep their own
licenses. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Model weights are not bundled.
