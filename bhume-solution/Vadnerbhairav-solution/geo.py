"""
bhume/geo.py
------------
Geometric helpers: patch extraction from rasters, IoU, centroid error,
boundary raster sampling, and shift application.
"""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.windows import from_bounds as window_from_bounds
from shapely.affinity import translate
from shapely.geometry import MultiPolygon, Polygon, box


# ── Coordinate helpers ─────────────────────────────────────────────────────

def bounds_with_buffer(geom, buffer_m: float = 60.0):
    """Return (minx, miny, maxx, maxy) of geometry with a buffer in metres."""
    b = geom.buffer(buffer_m).bounds
    return b  # (minx, miny, maxx, maxy)


def apply_shift(geom, dx: float, dy: float):
    """Translate a geometry by (dx, dy) in its native CRS units (metres)."""
    return translate(geom, xoff=dx, yoff=dy)


# ── Raster patch extraction ────────────────────────────────────────────────

def extract_patch(
    src: rasterio.DatasetReader,
    bounds,           # (minx, miny, maxx, maxy) in src CRS
    target_size: int = 128,
):
    """
    Extract a raster patch covering `bounds` and resize to target_size×target_size.

    Returns:
        patch : np.ndarray  shape (bands, H, W) uint8
        win_transform : rasterio.Affine  transform of the extracted window
    """
    minx, miny, maxx, maxy = bounds
    # Clip to raster bounds
    rb = src.bounds
    minx = max(minx, rb.left)
    miny = max(miny, rb.bottom)
    maxx = min(maxx, rb.right)
    maxy = min(maxy, rb.top)
    if minx >= maxx or miny >= maxy:
        return None, None

    win = window_from_bounds(minx, miny, maxx, maxy, src.transform)
    win = win.round_lengths().round_offsets()

    try:
        data = src.read(
            out_shape=(src.count, target_size, target_size),
            window=win,
            resampling=rasterio.enums.Resampling.bilinear,
        )
    except Exception:
        return None, None

    win_transform = src.window_transform(win)
    return data, win_transform


def extract_boundary_patch(
    bsrc: rasterio.DatasetReader,
    bounds,
    target_size: int = 128,
):
    """Extract boundary hint patch (single band, 0/255 binary mask)."""
    patch, t = extract_patch(bsrc, bounds, target_size)
    if patch is None:
        return None
    return patch[0]  # single band


# ── Geometry ↔ binary mask ─────────────────────────────────────────────────

def geom_to_mask(geom, bounds, size: int = 128) -> np.ndarray:
    """
    Rasterise `geom` into a (size×size) binary mask covering `bounds`.

    Returns uint8 mask where 1 = inside polygon.
    """
    from rasterio.transform import from_bounds as tfrom_bounds
    from rasterio.features import rasterize

    minx, miny, maxx, maxy = bounds
    transform = tfrom_bounds(minx, miny, maxx, maxy, size, size)
    mask = rasterize(
        [(geom.__geo_interface__, 1)],
        out_shape=(size, size),
        transform=transform,
        fill=0,
        dtype="uint8",
    )
    return mask


# ── IoU and centroid error ─────────────────────────────────────────────────

def iou(geom_a, geom_b) -> float:
    """Intersection over Union between two Shapely geometries."""
    try:
        inter = geom_a.intersection(geom_b).area
        union = geom_a.union(geom_b).area
        if union <= 0:
            return 0.0
        return inter / union
    except Exception:
        return 0.0


def centroid_error_m(geom_a, geom_b) -> float:
    """Euclidean distance between centroids of two geometries (metres, assumes metric CRS)."""
    ca = geom_a.centroid
    cb = geom_b.centroid
    return ((ca.x - cb.x) ** 2 + (ca.y - cb.y) ** 2) ** 0.5


# ── Overlap score between parcel mask and boundary hint ───────────────────

def boundary_overlap_score(geom, bounds, boundary_patch: np.ndarray) -> float:
    """
    Compute what fraction of the parcel's PERIMETER pixels overlap with
    boundary hint pixels (from the boundary raster).

    Higher = parcel edge coincides with real field edges.
    """
    size = boundary_patch.shape[0]
    mask = geom_to_mask(geom, bounds, size)

    # Erode mask to get perimeter ring only
    from skimage.morphology import erosion, square as sk_square
    eroded = erosion(mask, sk_square(3))
    perimeter = mask.astype(np.int16) - eroded.astype(np.int16)
    perimeter = np.clip(perimeter, 0, 1).astype(np.uint8)

    boundary_binary = (boundary_patch > 128).astype(np.uint8)

    perimeter_px = perimeter.sum()
    if perimeter_px == 0:
        return 0.0
    overlap = (perimeter & boundary_binary).sum()
    return float(overlap) / float(perimeter_px)
