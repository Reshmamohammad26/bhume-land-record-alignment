"""
bhume/io.py
-----------
Input / output helpers for the BhuMe pipeline.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.crs import CRS
from shapely.geometry import mapping, shape


# ── Reading ────────────────────────────────────────────────────────────────

def read_parcels(path: str | Path) -> gpd.GeoDataFrame:
    """Load input.geojson and project to EPSG:3857 (metres)."""
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf.to_crs("EPSG:3857")


def read_example_truths(path: str | Path) -> Optional[gpd.GeoDataFrame]:
    """Load example_truths.geojson. Returns None if file doesn't exist."""
    p = Path(path)
    if not p.exists():
        warnings.warn(f"example_truths.geojson not found at {p}")
        return None
    gdf = gpd.read_file(p)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf.to_crs("EPSG:3857")


def open_raster(path: str | Path):
    """Return an open rasterio DatasetReader (caller must close or use as context manager)."""
    return rasterio.open(path)


# ── Writing ────────────────────────────────────────────────────────────────

def write_predictions(predictions: list[dict], out_path: str | Path) -> None:
    """
    Write predictions to GeoJSON.

    Each dict must have:
      plot_number, status, confidence, method_note, geometry (Shapely)
    Geometry is converted back to EPSG:4326 for output.
    """
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    features = []
    for pred in predictions:
        geom = pred["geometry"]
        # Reproject geometry coords lon/lat
        if geom is not None:
            geom_4326 = _reproject_geometry(geom, transformer)
        else:
            geom_4326 = None

        features.append(
            {
                "type": "Feature",
                "properties": {
                    "plot_number": pred["plot_number"],
                    "status": pred["status"],
                    "confidence": round(float(pred["confidence"]), 4),
                    "method_note": pred["method_note"],
                },
                "geometry": mapping(geom_4326) if geom_4326 is not None else None,
            }
        )

    fc = {"type": "FeatureCollection", "features": features}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(fc, f, indent=2)
    print(f"[IO] Wrote {len(features)} predictions → {out_path}")


def _reproject_geometry(geom, transformer):
    """Reproject a Shapely geometry using a pyproj Transformer."""
    from shapely.ops import transform as shp_transform
    return shp_transform(
        lambda x, y: transformer.transform(x, y),
        geom,
    )
