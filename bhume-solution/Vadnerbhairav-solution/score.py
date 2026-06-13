"""
bhume/score.py
--------------
Evaluation metrics: IoU, centroid error, and confidence calibration.
Used to compare our pipeline vs the baseline.
"""

from __future__ import annotations

import numpy as np
import geopandas as gpd
from shapely.geometry import shape

from bhume.geo import iou, centroid_error_m


def evaluate(
    predictions: list[dict],
    ground_truth_gdf: gpd.GeoDataFrame,
    label: str = "Model",
) -> dict:
    """
    Evaluate predictions against ground-truth GeoDataFrame.

    ground_truth_gdf must have 'plot_number' and 'geometry' columns (EPSG:3857).
    Returns dict of metrics.
    """
    gt_indexed = ground_truth_gdf.set_index("plot_number")

    ious, centroid_errs, conf_correct, conf_wrong = [], [], [], []
    n_matched = 0

    for pred in predictions:
        pn = pred["plot_number"]
        if pn not in gt_indexed.index:
            continue
        gt_geom = gt_indexed.loc[pn, "geometry"]
        pred_geom = pred["geometry"]
        conf = pred["confidence"]

        n_matched += 1
        iou_val = iou(pred_geom, gt_geom)
        ce_val = centroid_error_m(pred_geom, gt_geom)
        ious.append(iou_val)
        centroid_errs.append(ce_val)

        # Confidence calibration: is high confidence associated with correct results?
        if iou_val > 0.6:
            conf_correct.append(conf)
        else:
            conf_wrong.append(conf)

    metrics = {
        "label": label,
        "n_matched": n_matched,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "median_iou": float(np.median(ious)) if ious else 0.0,
        "iou_gt_50": int(np.sum(np.array(ious) > 0.5)) if ious else 0,
        "iou_gt_70": int(np.sum(np.array(ious) > 0.7)) if ious else 0,
        "mean_centroid_err_m": float(np.mean(centroid_errs)) if centroid_errs else 999.0,
        "median_centroid_err_m": float(np.median(centroid_errs)) if centroid_errs else 999.0,
        "mean_conf_correct": float(np.mean(conf_correct)) if conf_correct else 0.0,
        "mean_conf_wrong": float(np.mean(conf_wrong)) if conf_wrong else 0.0,
        "calibration_gap": (
            float(np.mean(conf_correct)) - float(np.mean(conf_wrong))
            if conf_correct and conf_wrong else 0.0
        ),
    }

    print(f"\n{'='*50}")
    print(f"  Evaluation: {label}")
    print(f"{'='*50}")
    print(f"  Matched parcels    : {n_matched}")
    print(f"  Mean IoU           : {metrics['mean_iou']:.4f}")
    print(f"  Median IoU         : {metrics['median_iou']:.4f}")
    print(f"  IoU > 0.50         : {metrics['iou_gt_50']}")
    print(f"  IoU > 0.70         : {metrics['iou_gt_70']}")
    print(f"  Mean centroid err  : {metrics['mean_centroid_err_m']:.2f} m")
    print(f"  Median centroid err: {metrics['median_centroid_err_m']:.2f} m")
    print(f"  Avg conf (correct) : {metrics['mean_conf_correct']:.4f}")
    print(f"  Avg conf (wrong)   : {metrics['mean_conf_wrong']:.4f}")
    print(f"  Calibration gap    : {metrics['calibration_gap']:+.4f} (higher = better)")
    print(f"{'='*50}")
    return metrics


def compare(metrics_list: list[dict]) -> None:
    """Print a side-by-side comparison table."""
    print(f"\n{'Model':<20} {'MeanIoU':>9} {'MedIoU':>9} {'MedCE(m)':>10} {'CalGap':>8}")
    print("-" * 60)
    for m in metrics_list:
        print(
            f"  {m['label']:<18} {m['mean_iou']:>9.4f} {m['median_iou']:>9.4f}"
            f" {m['median_centroid_err_m']:>10.2f} {m['calibration_gap']:>+8.4f}"
        )
    print("-" * 60)
