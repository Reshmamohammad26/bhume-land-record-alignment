"""
bhume/baseline.py
-----------------
Replicates the BhuMe starter-kit baseline:
  1. Reads example_truths.geojson
  2. Computes per-example shift from (official → truth)
  3. Takes the median (dx, dy)
  4. Applies that single global shift to every parcel
  5. Marks all as "corrected" with confidence 0.5

This is used for benchmarking only.
"""

from __future__ import annotations

import numpy as np
import geopandas as gpd
from shapely.affinity import translate
from typing import Optional


def compute_global_shift(
    input_gdf: gpd.GeoDataFrame,
    truths_gdf: gpd.GeoDataFrame,
) -> tuple[float, float]:
    """
    Match example_truths to input parcels by plot_number.
    Return median (dx, dy) shift in metres (CRS must already be metric).
    """
    shifts = []
    input_indexed = input_gdf.set_index("plot_number")

    for _, truth_row in truths_gdf.iterrows():
        pn = truth_row["plot_number"]
        if pn not in input_indexed.index:
            continue
        official_geom = input_indexed.loc[pn, "geometry"]
        truth_geom = truth_row["geometry"]

        # Shift = truth_centroid - official_centroid
        oc = official_geom.centroid
        tc = truth_geom.centroid
        shifts.append((tc.x - oc.x, tc.y - oc.y))

    if not shifts:
        return 0.0, 0.0

    shifts = np.array(shifts)
    dx = float(np.median(shifts[:, 0]))
    dy = float(np.median(shifts[:, 1]))
    print(f"[Baseline] Global shift: dx={dx:.2f} m, dy={dy:.2f} m  (from {len(shifts)} examples)")
    return dx, dy


def run_baseline(
    input_gdf: gpd.GeoDataFrame,
    truths_gdf: Optional[gpd.GeoDataFrame],
) -> list[dict]:
    """Run the global-shift baseline. Returns list of prediction dicts."""
    if truths_gdf is not None:
        dx, dy = compute_global_shift(input_gdf, truths_gdf)
    else:
        dx, dy = 0.0, 0.0
        print("[Baseline] No example_truths — using zero shift")

    predictions = []
    for _, row in input_gdf.iterrows():
        shifted = translate(row.geometry, xoff=dx, yoff=dy)
        predictions.append(
            {
                "plot_number": str(row["plot_number"]),
                "status": "corrected",
                "confidence": 0.5,
                "method_note": f"global_shift dx={dx:.2f}m dy={dy:.2f}m",
                "geometry": shifted,
            }
        )
    return predictions
