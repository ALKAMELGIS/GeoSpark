# AI Platform — GeoAI Web Application: Phased Plan

This plan rebuilds the GeoAI and aerial object-detection capabilities of
[EarthMapper](https://github.com/AliAlkazraji/EarthMapper) as a web application.

## 0. Reverse-engineering findings (what EarthMapper is)

| Finding | Evidence |
|---|---|
| The GitHub repo contains **only `README.md`**; no source code is published. | `git ls-files` → 1 file |
| The app ships as a closed-source **.NET / WPF** Windows installer (v1.0.0 → v1.3.0, ~115 MB). | GitHub Releases API |
| The map UI is web-based (**Leaflet + MapLibre GL** inside **WebView2**). | README "Platform", "Acknowledgements" |
| GeoAI runs **locally through ONNX Runtime** with a built-in OBB model, or a user's own ONNX model. | README "GeoAI and Spatial Analysis" |
| Processing is **native-resolution and tiled**, with class selection, overlap handling, progress and **georeferenced output**. | README |
| The built-in model is **YOLO26s-OBB trained on DOTA v2.0** (18 classes, imgsz 1024, MIT, `best.onnx` in release v1.0). | `JustinVladut/YOLO26-DOTA-v2` → `src/dota2_obb.yaml`, `train_args.yaml` |
| Tree detection follows **TreeSensing** (YOLOv8 detect, 1 class `tree`, imgsz 640). **No license declared** upstream. | `flyingvaibhav/TreeSensing` |
| Building segmentation follows **keremberke/yolov8n-building-segmentation** (YOLOv8n-seg, imgsz 640, `.pt` only). | Hugging Face model card / `config.json` |
| Interactive segmentation uses **SAM 2** (point prompts) and optionally **SAM 3** (text prompts). | README |

**Approach:** this is a *clean-room* re-implementation from public specifications
(the Ultralytics ONNX output format, the GeoTIFF/affine georeferencing model, the
SAM prompt interface). It does not decompile or copy EarthMapper binaries. The
reusable intellectual property is in the open upstream projects, and this platform
consumes them as pluggable ONNX models.

### Inferred GeoAI pipeline (implemented in `backend/geoai`)

```
GeoTIFF ──► raster.py  (metadata, band selection, percentile stretch → uint8 RGB)
        ──► tiling.py  (native-resolution tile grid, imgsz, overlap %)
        ──► models.py  (ONNX Runtime session, letterbox, Ultralytics metadata)
        ──► postprocess.py (decode detect | obb | segment, raw or end-to-end layouts,
                            per-tile NMS, prototype masks → polygons)
        ──► pipeline.py (tile→raster offsets, global cross-tile NMS (rotated for OBB),
                         pixel → CRS → WGS84, geodesic area/length/width)
        ──► GeoJSON layer ──► web map (symbology, attribute table, export)
```

## Phases

### Phase 0 — Foundations ✅
- Repository layout, Python venv, FastAPI service, static web client.
- Model registry (`models/registry.json`) with task, classes, imgsz, license and source per model.
- `THIRD_PARTY_NOTICES.md` with licensing and attribution.

### Phase 1 — GeoAI core + aerial OBB detection ✅ (this delivery)
- GeoTIFF upload, inspection (CRS, bounds, resolution, bands, dtype) and Web Mercator preview.
- Tiled ONNX inference with overlap, class filter, confidence/IoU, band mapping, progress, cancel.
- Decoders for Ultralytics **detect / OBB / segment** (raw and NMS-free end-to-end exports).
- Global rotated NMS across tile seams; georeferenced polygons with attributes.
- Web map: basemaps (OSM, Esri imagery, none/offline), raster overlay with opacity,
  class-coloured detection layer, popups, per-class summary, attribute table, zoom-to.
- Export GeoJSON and CSV (WKT). Custom ONNX upload, auto-configured from embedded metadata.
- Unit and end-to-end tests with a generated ONNX test model.

#### Phase 1 verification with the real model (2026-10-23)
- `yolo26s-obb-dota2.onnx` (Ultralytics 8.4.40 export): input `[1,3,1024,1024]`, RGB/255; output
  `[1,300,7]` end-to-end `[cx, cy, w, h, score, class, angle]`, `end2end=True`. Matches the decoder.
- Clean aerial test image (Ultralytics `boats.jpg`, 1920×1080): **185 ships + 7 harbors** across 6 tiles in
  ~7 s on CPU, top scores 0.94–0.95. Rotated boxes align with the vessels on the map.
- The upstream README example JPGs are downsized renders with predictions already drawn on them, so
  the model detects few objects in them at native scale. This is expected and confirms the need for scale control:
  a **Resample ×** parameter (`scale`) now zooms imagery before inference.
- The embedded ONNX metadata declares **AGPL-3.0** for the weights (the repository code itself is MIT).

### Phase 2 — Tree detection & building segmentation
- Register TreeSensing-style tree model and keremberke building-seg model (`scripts/export_models.py`).
- Automatic GSD-aware resampling: pick `scale` from the raster resolution and each model's training GSD
  (manual `scale` already exists).
- Building polygon regularisation (orthogonalisation, minimum area, hole handling).
- Tree count / density statistics per AOI and point-centroid output mode.

### Phase 3 — SAM 2 interactive segmentation
- SAM 2 ONNX encoder/decoder (Hiera-T/S) in `geoai/sam.py`; cache image embeddings per view window.
- Map tool: left-click = positive point, right-click = negative point, box prompt; live mask preview.
- Accept, refine or undo masks → polygons → editable layer.

### Phase 4 — SAM 3 text-prompted segmentation (optional)
- Optional PyTorch worker (GPU recommended) behind the same job API; prompt "solar panels", "pools", etc.
- Gated by the user's acceptance of the Meta SAM License; model weights are never bundled.

### Phase 5 — GIS integration
- Exports: GeoPackage, Shapefile, KML/KMZ, FlatGeobuf (pyogrio/GDAL); reprojection to a chosen CRS.
- AOI drawing to limit processing; layer manager; categorized/graduated symbology; labels.
- Attribute queries, buffers, spatial joins, statistics reports (HTML export).
- Vendored map assets and an offline tile cache for disconnected operation.

### Phase 6 — Production hardening
- Persistent job queue (SQLite/Redis), concurrent workers, GPU (CUDA/DirectML) providers.
- Cloud-Optimized GeoTIFF / tiled raster serving (TiTiler-style) for large rasters.
- AuthN/Z, upload quotas, Docker packaging, CI, benchmark suite.
