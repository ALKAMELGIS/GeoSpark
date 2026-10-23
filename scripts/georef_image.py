"""Turn a plain image (JPG/PNG) into a GeoTIFF, or create a synthetic test GeoTIFF.

    python scripts/georef_image.py photo.jpg out.tif --lon 44.36 --lat 33.31 --gsd 0.3
    python scripts/georef_image.py --synthetic out.tif

Useful for trying models on sample images that carry no georeferencing. The image is
placed in the local UTM zone with its top-left corner at (lon, lat) and square pixels of `gsd` metres.
"""

import argparse

import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.transform import from_origin


def utm_epsg(lon, lat):
    return (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?")
    ap.add_argument("dst")
    ap.add_argument("--lon", type=float, default=44.3661)
    ap.add_argument("--lat", type=float, default=33.3152)
    ap.add_argument("--gsd", type=float, default=0.3, help="metres per pixel")
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()

    if a.synthetic:
        rng = np.random.default_rng(1)
        arr = rng.integers(60, 180, (2048, 2048, 3), dtype=np.uint8)
    else:
        arr = np.asarray(Image.open(a.src).convert("RGB"))
    epsg = utm_epsg(a.lon, a.lat)
    x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(a.lon, a.lat)
    h, w = arr.shape[:2]
    with rasterio.open(a.dst, "w", driver="GTiff", width=w, height=h, count=3, dtype="uint8", crs=f"EPSG:{epsg}",
                       transform=from_origin(x, y, a.gsd, a.gsd), compress="deflate", tiled=True) as ds:
        ds.write(np.moveaxis(arr, -1, 0))
    print(f"Wrote {a.dst}: {w}x{h} px, EPSG:{epsg}, {a.gsd} m/px")


if __name__ == "__main__":
    main()
