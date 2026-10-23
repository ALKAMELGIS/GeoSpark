"""Model registry and ONNX Runtime inference for Ultralytics-exported YOLO models."""

from __future__ import annotations

import ast
import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from . import config
from .postprocess import Detections, decode, masks_to_polygons, nms

INFERENCE_TASKS = {"detect", "obb", "segment"}
MAX_CANDIDATES = 30000


@dataclass
class ModelSpec:
    id: str
    name: str
    task: str
    file: str | None = None
    imgsz: int = 640
    names: list[str] = field(default_factory=list)
    category: str = "custom"
    description: str = ""
    license: str = ""
    source: str = ""
    download_url: str | None = None
    layout: str = "auto"
    planned: bool = False

    @property
    def path(self) -> Path | None:
        return (config.MODELS_DIR / self.file) if self.file else None

    @property
    def available(self) -> bool:
        return not self.planned and self.path is not None and self.path.exists()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["available"] = self.available
        return d


def read_onnx_metadata(path) -> dict:
    """Ultralytics embeds task, names, imgsz (and end2end for NMS-free heads) in the ONNX file."""
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return _parse_metadata(sess)


def _parse_metadata(sess) -> dict:
    meta = sess.get_modelmeta().custom_metadata_map
    out = {}
    if "task" in meta:
        out["task"] = meta["task"]
    if "names" in meta:
        names = ast.literal_eval(meta["names"])
        out["names"] = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    if "imgsz" in meta:
        sz = ast.literal_eval(meta["imgsz"])
        out["imgsz"] = int(sz[0] if isinstance(sz, (list, tuple)) else sz)
    if meta.get("end2end", "").lower() == "true":
        out["layout"] = "end2end"
    return out


def load_registry() -> dict[str, ModelSpec]:
    specs: dict[str, ModelSpec] = {}
    reg = config.MODELS_DIR / "registry.json"
    if reg.exists():
        for item in json.loads(reg.read_text(encoding="utf-8"))["models"]:
            specs[item["id"]] = ModelSpec(**item)
    for f in sorted(config.CUSTOM_MODEL_DIR.glob("*.onnx")):
        mid = f"custom-{f.stem}"
        if mid in specs:
            continue
        sidecar = f.with_suffix(".json")
        info = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
        specs[mid] = ModelSpec(
            id=mid, name=info.get("name", f.stem), task=info.get("task", "detect"),
            file=f"custom/{f.name}", imgsz=info.get("imgsz", 640), names=info.get("names", []),
            category="custom", description="User-supplied ONNX model", license=info.get("license", "User-supplied"),
            layout=info.get("layout", "auto"),
        )
    return specs


def letterbox(img: np.ndarray, size: tuple[int, int]):
    h, w = img.shape[:2]
    ih, iw = size
    r = min(ih / h, iw / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = img if (nh, nw) == (h, w) else cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    px, py = (iw - nw) / 2, (ih - nh) / 2
    out = np.full((ih, iw, 3), 114, np.uint8)
    top, left = int(round(py - 0.1)), int(round(px - 0.1))
    out[top : top + nh, left : left + nw] = resized
    return out, (r, float(left), float(top))


class OnnxModel:
    def __init__(self, spec: ModelSpec):
        if not spec.available:
            raise FileNotFoundError(f"Model file not found for '{spec.id}': {spec.path}")
        providers = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider") if p in ort.get_available_providers()]
        self.session = ort.InferenceSession(str(spec.path), providers=providers + ["CPUExecutionProvider"])
        meta = _parse_metadata(self.session)
        self.spec = spec
        self.task = meta.get("task", spec.task)
        self.names = meta.get("names") or spec.names
        self.layout = meta.get("layout", spec.layout)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        h, w = inp.shape[2], inp.shape[3]
        imgsz = meta.get("imgsz", spec.imgsz)
        self.input_hw = (h if isinstance(h, int) else imgsz, w if isinstance(w, int) else imgsz)
        if self.task not in INFERENCE_TASKS:
            raise ValueError(f"Unsupported task '{self.task}'")
        if not self.names:
            self.names = [f"class_{i}" for i in range(self._infer_nc())]
        self._lock = threading.Lock()

    def _infer_nc(self) -> int:
        """Class count from a raw-layout output shape, for ONNX files without metadata."""
        shape = self.session.get_outputs()[0].shape
        feat = shape[1] if isinstance(shape[1], int) else None
        if feat is None or self.layout == "end2end":
            raise ValueError("Cannot infer class names; supply them in the model's JSON sidecar.")
        extra = {"detect": 4, "obb": 5, "segment": 4 + 32}[self.task]
        return feat - extra

    @property
    def nc(self) -> int:
        return len(self.names)

    def predict(self, rgb: np.ndarray, conf=0.25, iou=0.45, classes=None, simplify_px=1.0,
                max_det=1000) -> Detections:
        """Run on one RGB uint8 tile. Returns detections in tile pixel coordinates."""
        img, (r, px, py) = letterbox(rgb, self.input_hw)
        x = np.ascontiguousarray(img.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        with self._lock:
            outs = self.session.run(None, {self.input_name: x})
        boxes, scores, cls, coeffs = decode(outs[0], self.task, self.nc, conf, classes, self.layout)
        if len(scores) > MAX_CANDIDATES:
            top = np.argsort(-scores)[:MAX_CANDIDATES]
            boxes, scores, cls = boxes[top], scores[top], cls[top]
            coeffs = coeffs[top] if coeffs is not None else None
        keep = nms(boxes, scores, cls, iou, rotated=self.task == "obb")[:max_det]
        boxes, scores, cls = boxes[keep], scores[keep], cls[keep]
        polygons = None
        if self.task == "segment" and len(outs) > 1 and len(keep):
            polygons = masks_to_polygons(coeffs[keep], outs[1][0], boxes, self.input_hw, (r, px, py), simplify_px)
        boxes = boxes.copy()
        boxes[:, 0] = (boxes[:, 0] - px) / r
        boxes[:, 1] = (boxes[:, 1] - py) / r
        boxes[:, 2:4] /= r
        if self.task == "segment" and polygons is None:
            polygons = [None] * len(keep)
        return Detections(boxes, scores, cls, polygons)


class ModelManager:
    def __init__(self):
        self._cache: dict[str, OnnxModel] = {}
        self._lock = threading.Lock()

    def specs(self) -> dict[str, ModelSpec]:
        return load_registry()

    def get(self, model_id: str) -> OnnxModel:
        with self._lock:
            if model_id not in self._cache:
                spec = self.specs().get(model_id)
                if spec is None:
                    raise KeyError(model_id)
                self._cache[model_id] = OnnxModel(spec)
            return self._cache[model_id]

    def evict(self, model_id: str):
        with self._lock:
            self._cache.pop(model_id, None)
