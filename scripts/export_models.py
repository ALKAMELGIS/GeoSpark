"""Fetch or export the registry's ONNX models into ./models.

    python scripts/export_models.py yolo26s-obb-dota2          # downloads the published ONNX (MIT)
    python scripts/export_models.py yolo11n-obb-dota1           # needs: pip install ultralytics
    python scripts/export_models.py building-yolov8n-seg        # needs: pip install ultralytics huggingface_hub
    python scripts/export_models.py tree-yolov8 --weights path/to/best.pt

Weights are not bundled with this repository. Check each model's license
(models/registry.json and THIRD_PARTY_NOTICES.md) before use or redistribution.
"""

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def spec(model_id):
    for m in json.loads((MODELS / "registry.json").read_text(encoding="utf-8"))["models"]:
        if m["id"] == model_id:
            return m
    sys.exit(f"Unknown model id: {model_id}")


def export_pt(weights, imgsz, dest):
    from ultralytics import YOLO

    out = YOLO(str(weights)).export(format="onnx", imgsz=imgsz, opset=17, simplify=True, dynamic=False)
    shutil.move(out, dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id")
    ap.add_argument("--weights", help="local .pt weights to export instead of the default source")
    args = ap.parse_args()
    m = spec(args.model_id)
    dest = MODELS / m["file"]
    if args.weights:
        export_pt(args.weights, m["imgsz"], dest)
    elif m.get("download_url"):
        print(f"Downloading {m['download_url']}")
        urllib.request.urlretrieve(m["download_url"], dest)
    elif m["id"] == "yolo11n-obb-dota1":
        export_pt("yolo11n-obb.pt", m["imgsz"], dest)
    elif m["id"] == "building-yolov8n-seg":
        from huggingface_hub import hf_hub_download

        export_pt(hf_hub_download("keremberke/yolov8n-building-segmentation", "best.pt"), m["imgsz"], dest)
    else:
        sys.exit("No automatic source for this model; pass --weights.")
    print(f"Saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
