"""Decoding of Ultralytics ONNX outputs (detect / obb / segment) and NMS.

Two output layouts are supported:
  * raw     — (1, F, N) with N anchors (≈8400 at 640, ≈21504 at 1024); needs NMS.
              detect: [cx, cy, w, h, cls_0..cls_nc]
              obb:    [cx, cy, w, h, cls_0..cls_nc, angle]
              segment:[cx, cy, w, h, cls_0..cls_nc, mask_0..mask_nm]
  * end2end — (1, N, F) with N = max_det (YOLO26 / YOLOv10 NMS-free exports).
              detect: [x1, y1, x2, y2, score, cls]
              obb:    [cx, cy, w, h, score, cls, angle]
              segment:[x1, y1, x2, y2, score, cls, mask_0..mask_nm]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Detections:
    """Boxes are (cx, cy, w, h, angle_rad) in pixel coordinates."""

    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 5), np.float32))
    scores: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    classes: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    polygons: list | None = None  # per detection (K, 2) pixel coords, segment only

    def __len__(self):
        return len(self.scores)

    def take(self, idx) -> "Detections":
        idx = np.asarray(idx, dtype=np.int64)
        polys = [self.polygons[i] for i in idx] if self.polygons is not None else None
        return Detections(self.boxes[idx], self.scores[idx], self.classes[idx], polys)

    @staticmethod
    def concat(items: list["Detections"]) -> "Detections":
        items = [d for d in items if len(d)]
        if not items:
            return Detections()
        polys = None
        if any(d.polygons is not None for d in items):
            polys = [p for d in items for p in (d.polygons or [None] * len(d))]
        return Detections(
            np.concatenate([d.boxes for d in items]),
            np.concatenate([d.scores for d in items]),
            np.concatenate([d.classes for d in items]),
            polys,
        )


def detect_layout(shape, nc: int, task: str) -> str:
    _, a, b = shape
    if a < b:
        return "raw"
    # end2end rows are short (6/7/6+nm); raw rows are long. Prefer the declared width.
    return "end2end" if b in (6, 7) or task == "segment" else "raw"


def _xyxy_to_cxcywh(b):
    return np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1)


def decode(output: np.ndarray, task: str, nc: int, conf: float, classes=None, layout: str = "auto"):
    """Return (boxes[n,5], scores[n], classes[n], mask_coeffs[n,nm] | None) in model-input pixels."""
    if layout == "auto":
        layout = detect_layout(output.shape, nc, task)
    if layout == "raw":
        p = output[0].T  # (N, F)
        cls_scores = p[:, 4 : 4 + nc]
        cls = cls_scores.argmax(1).astype(np.int32)
        score = cls_scores[np.arange(len(p)), cls]
        boxes = p[:, :4]
        angle = p[:, 4 + nc] if task == "obb" else np.zeros(len(p), np.float32)
        coeffs = p[:, 4 + nc :] if task == "segment" else None
    else:
        p = output[0]
        score, cls = p[:, 4], p[:, 5].astype(np.int32)
        if task == "obb":
            boxes, angle, coeffs = p[:, :4], p[:, 6], None
        else:
            boxes, angle = _xyxy_to_cxcywh(p[:, :4]), np.zeros(len(p), np.float32)
            coeffs = p[:, 6:] if task == "segment" else None

    keep = score >= conf
    if classes is not None:
        keep &= np.isin(cls, np.asarray(list(classes), dtype=np.int32))
    boxes = np.concatenate([boxes[keep], angle[keep, None]], 1).astype(np.float32)
    return boxes, score[keep].astype(np.float32), cls[keep], (coeffs[keep] if coeffs is not None else None)


def nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou: float, rotated: bool,
        class_agnostic: bool = False) -> np.ndarray:
    """Per-class NMS. Rotated boxes use OpenCV's rotated-rectangle IoU."""
    if len(scores) == 0:
        return np.zeros(0, np.int64)
    groups = [np.arange(len(scores))] if class_agnostic else [np.flatnonzero(classes == c) for c in np.unique(classes)]
    keep = []
    for g in groups:
        b, s = boxes[g], scores[g].tolist()
        if rotated:
            rects = [((float(x), float(y)), (float(w), float(h)), float(np.degrees(a))) for x, y, w, h, a in b]
            idx = cv2.dnn.NMSBoxesRotated(rects, s, 0.0, iou)
        else:
            xywh = [[float(x - w / 2), float(y - h / 2), float(w), float(h)] for x, y, w, h, _ in b]
            idx = cv2.dnn.NMSBoxes(xywh, s, 0.0, iou)
        keep.extend(g[np.asarray(idx, dtype=np.int64).reshape(-1)])
    keep = np.asarray(keep, dtype=np.int64)
    return keep[np.argsort(-scores[keep])]


def obb_corners(boxes: np.ndarray) -> np.ndarray:
    """(n,5) → (n,4,2) corner points, clockwise in image space."""
    cx, cy, w, h, a = boxes.T
    cos, sin = np.cos(a), np.sin(a)
    dx = np.stack([w / 2, -w / 2, -w / 2, w / 2], 1)
    dy = np.stack([h / 2, h / 2, -h / 2, -h / 2], 1)
    x = cx[:, None] + dx * cos[:, None] - dy * sin[:, None]
    y = cy[:, None] + dx * sin[:, None] + dy * cos[:, None]
    return np.stack([x, y], -1)


def masks_to_polygons(coeffs, protos, boxes_in, input_hw, letterbox, simplify_px: float = 1.0,
                      mask_thr: float = 0.5):
    """Prototype masks → one polygon per detection, in original-tile pixels.

    boxes_in are (cx,cy,w,h,·) in model-input pixels; letterbox is (ratio, pad_x, pad_y).
    Only the box region of each mask is upsampled, which keeps memory flat for dense scenes.
    """
    r, px, py = letterbox
    nm, mh, mw = protos.shape
    ih, iw = input_hw
    sx, sy = mw / iw, mh / ih
    flat = protos.reshape(nm, -1)
    polygons = []
    for c, (cx, cy, w, h, _) in zip(coeffs, boxes_in):
        x1, y1 = int(max(0, np.floor((cx - w / 2) * sx))), int(max(0, np.floor((cy - h / 2) * sy)))
        x2, y2 = int(min(mw, np.ceil((cx + w / 2) * sx))), int(min(mh, np.ceil((cy + h / 2) * sy)))
        if x2 <= x1 or y2 <= y1:
            polygons.append(None)
            continue
        logits = (c @ flat).reshape(mh, mw)[y1:y2, x1:x2]
        prob = 1 / (1 + np.exp(-logits))
        # crop origin and size in original-tile pixels
        ox, oy = (x1 / sx - px) / r, (y1 / sy - py) / r
        tw, th = max(1, int(round((x2 - x1) / sx / r))), max(1, int(round((y2 - y1) / sy / r)))
        m = (cv2.resize(prob.astype(np.float32), (tw, th), interpolation=cv2.INTER_LINEAR) > mask_thr).astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            polygons.append(None)
            continue
        cnt = max(contours, key=cv2.contourArea)
        if simplify_px > 0:
            cnt = cv2.approxPolyDP(cnt, simplify_px, True)
        pts = cnt.reshape(-1, 2).astype(np.float32) + np.array([ox, oy], np.float32)
        polygons.append(pts if len(pts) >= 3 else None)
    return polygons
