"""Synthetic GeoTIFFs and tiny ONNX models that mimic Ultralytics output layouts."""

import numpy as np
import onnx
import rasterio
from onnx import TensorProto, helper, numpy_helper
from rasterio.transform import from_origin

OBB_DETS = [  # cx, cy, w, h, angle, class, score  (model-input pixels)
    (100, 100, 60, 20, 0.3, 0, 0.90),
    (102, 101, 60, 20, 0.3, 0, 0.80),  # duplicate → removed by NMS
    (300, 200, 40, 40, 0.0, 1, 0.70),
    (500, 500, 30, 30, 0.0, 1, 0.10),  # below confidence → filtered
]


def _model(outputs: dict, imgsz: int, task: str, names: list[str], path):
    nodes, outs = [], []
    for name, arr in outputs.items():
        nodes.append(helper.make_node("Constant", [], [name], value=numpy_helper.from_array(arr.astype(np.float32))))
        outs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, list(arr.shape)))
    inp = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, imgsz, imgsz])
    graph = helper.make_graph(nodes, "mock", [inp], outs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    for k, v in {"task": task, "names": str(dict(enumerate(names))), "imgsz": f"[{imgsz}, {imgsz}]"}.items():
        model.metadata_props.append(onnx.StringStringEntryProto(key=k, value=v))
    onnx.save(model, str(path))
    return path


def make_obb_model(path, imgsz=640, n_anchors=64):
    names = ["plane", "ship"]
    out = np.zeros((1, 4 + len(names) + 1, n_anchors), np.float32)
    for i, (cx, cy, w, h, a, c, s) in enumerate(OBB_DETS):
        out[0, :4, i] = (cx, cy, w, h)
        out[0, 4 + c, i] = s
        out[0, 4 + len(names), i] = a
    return _model({"output0": out}, imgsz, "obb", names, path)


def make_obb_e2e_model(path, imgsz=640, max_det=300):
    out = np.zeros((1, max_det, 7), np.float32)
    for i, (cx, cy, w, h, a, c, s) in enumerate(OBB_DETS):
        out[0, i] = (cx, cy, w, h, s, c, a)
    return _model({"output0": out}, imgsz, "obb", ["plane", "ship"], path)


def make_seg_model(path, imgsz=640, n_anchors=64, nm=32):
    out = np.zeros((1, 4 + 1 + nm, n_anchors), np.float32)
    out[0, :4, 0] = (200, 200, 80, 40)
    out[0, 4, 0] = 0.9
    out[0, 5, 0] = 1.0  # coefficient on prototype 0
    protos = np.zeros((1, nm, imgsz // 4, imgsz // 4), np.float32)
    protos[0, 0] = 10.0  # mask = sigmoid(10) inside the box
    return _model({"output0": out, "output1": protos}, imgsz, "segment", ["building"], path)


def make_geotiff(path, width=1500, height=1500, res=0.5, crs="EPSG:32638", origin=(440000.0, 3700000.0), dtype="uint8"):
    rng = np.random.default_rng(0)
    data = rng.integers(40, 200, (3, height, width)).astype(dtype)
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height, count=3, dtype=dtype, crs=crs,
                       transform=from_origin(origin[0], origin[1], res, res)) as ds:
        ds.write(data)
    return path
