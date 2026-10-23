# Third-Party Notices

This platform's own code does not include any third-party model weights. Models are fetched or
exported by the user (`scripts/export_models.py`) and keep their own licenses. **Verify the
provenance and redistribution rights of any weights you supply.** Acknowledgments below do not
imply endorsement by, or affiliation with, the original developers.

## Frameworks and libraries

| Component | Use | License |
|---|---|---|
| [Microsoft ONNX Runtime](https://onnxruntime.ai/) | Local neural-network inference | MIT |
| [Ultralytics](https://www.ultralytics.com/) | YOLO export format, training framework of the supported models | AGPL-3.0 / Enterprise |
| [rasterio](https://rasterio.readthedocs.io/) / GDAL | GeoTIFF I/O, warping | BSD-3-Clause / MIT |
| [pyproj](https://pyproj4.github.io/pyproj/) / PROJ | CRS transforms, geodesic measures | MIT |
| [OpenCV](https://opencv.org/) | Rotated NMS, contour extraction | Apache-2.0 |
| [FastAPI](https://fastapi.tiangolo.com/) | Web API | MIT |
| [MapLibre GL JS](https://maplibre.org/) | Web map (vendored in `frontend/vendor`) | BSD-3-Clause |

## Models and datasets

| Model / dataset | Role | License / terms |
|---|---|---|
| [DOTA](https://captain-whu.github.io/DOTA/) | Aerial oriented-object training data | Academic use; see dataset terms |
| [YOLO26-DOTA-v2](https://github.com/JustinVladut/YOLO26-DOTA-v2) by JustinVladut | Built-in aerial OBB model (18 classes) | Repository code MIT; the exported weights declare **AGPL-3.0** (Ultralytics) in their ONNX metadata |
| Ultralytics YOLO11n-OBB | Lightweight DOTA v1 OBB alternative | AGPL-3.0 |
| [TreeSensing](https://github.com/flyingvaibhav/TreeSensing) by flyingvaibhav | Tree detection reference | **No license declared** (all rights reserved). Training data: Roboflow tree-counting, CC BY 4.0 |
| [YOLOv8n Building Segmentation](https://huggingface.co/keremberke/yolov8n-building-segmentation) by keremberke | Building instance segmentation | Ultralytics AGPL-3.0; check model-card terms |
| [SAM 2](https://github.com/facebookresearch/sam2) by Meta AI | Point-prompt segmentation (Phase 3) | Apache-2.0 |
| [SAM 3](https://github.com/facebookresearch/sam3) by Meta AI | Text-prompt segmentation (Phase 4, optional) | Meta SAM License |

## Basemaps

Esri World Imagery (© Esri and its data providers) and OpenStreetMap (© OpenStreetMap
contributors, ODbL) are online services, subject to their providers' terms. The application
works offline without them.

## Feature-parity reference

Capabilities follow the public description of [EarthMapper](https://github.com/AliAlkazraji/EarthMapper)
by Ali Alkazraji. No EarthMapper code or binaries are included. Third-party imagery, data, models and
services keep their own licenses; free availability does not make every use of them non-commercial.
