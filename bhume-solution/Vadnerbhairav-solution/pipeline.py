"""
bhume/pipeline.py  — optimized pipeline with preloaded rasters
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
import geopandas as gpd

from bhume.io import read_parcels, read_example_truths, write_predictions
from bhume.baseline import compute_global_shift
from bhume.aligner import align_parcel, init_rasters, CONFIDENCE_THRESHOLD
from bhume.geo import apply_shift


def run_pipeline(
    data_dir: str | Path,
    out_path: str | Path = "predictions.geojson",
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    max_parcels: Optional[int] = None,
    verbose: bool = True,
) -> list[dict]:
    data_dir = Path(data_dir)
    t0 = time.time()

    # ── 1. Load inputs ─────────────────────────────────────────────────────
    if verbose:
        print("\n[Pipeline] Loading inputs …")

    parcels = read_parcels(data_dir / "input.geojson")
    if verbose:
        print(f"  {len(parcels)} parcels (EPSG:3857)")

    truths = read_example_truths(data_dir / "example_truths.geojson")
    if truths is not None and verbose:
        print(f"  {len(truths)} example truths")

    # ── 2. Preload rasters into memory ────────────────────────────────────
    if verbose:
        print("  Preloading rasters …")

    with rasterio.open(data_dir / "imagery.tif") as isrc:
        img_array = isrc.read()           # (3, H, W)
        img_transform = isrc.transform

    with rasterio.open(data_dir / "boundaries.tif") as bsrc:
        bnd_array = bsrc.read(1)          # (H, W)
        bnd_transform = bsrc.transform

    init_rasters(img_array, img_transform, bnd_array, bnd_transform)
    if verbose:
        print(f"  Rasters loaded in {time.time()-t0:.1f}s "
              f"(img={img_array.nbytes//1e6:.0f}MB, "
              f"bnd={bnd_array.nbytes//1e6:.0f}MB)")

    # ── 3. Prior shift from example_truths ────────────────────────────────
    if truths is not None:
        prior_dx, prior_dy = compute_global_shift(parcels, truths)
    else:
        prior_dx, prior_dy = 0.0, 0.0
        if verbose:
            print("  No example_truths — prior=(0,0)")

    # ── 4. Per-parcel alignment ───────────────────────────────────────────
    n = len(parcels) if max_parcels is None else min(max_parcels, len(parcels))
    if verbose:
        print(f"\n[Pipeline] Aligning {n} parcels (threshold={confidence_threshold}) …")

    predictions = []
    for i, (_, row) in enumerate(parcels.iloc[:n].iterrows()):
        if verbose and i % 300 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (n - i) / rate if rate > 0 else 0
            print(f"  [{i}/{n}]  {elapsed:.0f}s elapsed  ETA {eta:.0f}s …")

        geom     = row.geometry
        area_sqm = float(row.get("map_area_sqm", geom.area))

        result     = align_parcel(geom, prior_dx, prior_dy, area_sqm)
        dx         = result["dx"]
        dy         = result["dy"]
        confidence = result["confidence"]
        method     = result["method_note"]

        if confidence >= confidence_threshold:
            status     = "corrected"
            final_geom = apply_shift(geom, dx, dy)
        else:
            status     = "flagged"
            # Apply global prior as best-effort for flagged parcels
            final_geom = apply_shift(geom, prior_dx, prior_dy)
            method     = f"flagged(conf={confidence:.3f}) prior_applied | {method}"

        predictions.append({
            "plot_number": str(row["plot_number"]),
            "status":      status,
            "confidence":  confidence,
            "method_note": method,
            "geometry":    final_geom,
        })

    # ── 5. Summary ────────────────────────────────────────────────────────
    if verbose:
        corrected = sum(1 for p in predictions if p["status"] == "corrected")
        flagged   = len(predictions) - corrected
        confs     = [p["confidence"] for p in predictions]
        print(f"\n[Pipeline] Results ({time.time()-t0:.1f}s total):")
        print(f"  Corrected : {corrected} ({100*corrected/len(predictions):.1f}%)")
        print(f"  Flagged   : {flagged}   ({100*flagged/len(predictions):.1f}%)")
        print(f"  Confidence: mean={np.mean(confs):.3f}  "
              f"median={np.median(confs):.3f}  "
              f"min={np.min(confs):.3f}  max={np.max(confs):.3f}")

    # ── 6. Write output ───────────────────────────────────────────────────
    write_predictions(predictions, out_path)
    return predictions
