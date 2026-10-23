"""Native-resolution tile grid with overlap."""


def _starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] + tile < length:
        starts.append(length - tile)  # last tile flush with the raster edge
    return starts


def tile_grid(width: int, height: int, tile: int, overlap: float) -> list[tuple[int, int]]:
    """(col_off, row_off) of every tile. `overlap` is a fraction of the tile size (0–0.5)."""
    overlap = min(max(overlap, 0.0), 0.5)
    stride = max(1, int(round(tile * (1 - overlap))))
    return [(c, r) for r in _starts(height, tile, stride) for c in _starts(width, tile, stride)]
