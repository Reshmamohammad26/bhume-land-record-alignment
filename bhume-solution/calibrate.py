#!/usr/bin/env python3
"""
Tune flag_threshold on example truths (directional check only — do not overfit).

Usage:
    python calibrate.py data/34855_vadnerbhairav_chandavad_nashik
"""

from __future__ import annotations

import sys
from pathlib import Path

from bhume import load, score
from bhume.predict import predict_village

DEFAULT_VILLAGE = 'data/34855_vadnerbhairav_chandavad_nashik'


def main() -> None:
    village_dir = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VILLAGE)
    village = load(village_dir)

    if village.example_truths is None:
        print('Need example_truths.geojson for calibration sweep')
        sys.exit(1)

    print(f'Sweeping flag_threshold on {village.slug} ({len(village.example_truths)} truths)\n')
    print(f'{"threshold":>10}  {"corrected":>9}  {"flagged":>7}  {"med IoU":>8}  {"Spearman":>8}  {"AUC":>6}')
    print('-' * 60)

    best = None
    for thresh in [0.35, 0.40, 0.45, 0.50, 0.52, 0.55, 0.60, 0.65, 0.70]:
        preds = predict_village(village, flag_threshold=thresh)
        sc = score(preds, village)
        spearman = sc.spearman_conf_vs_iou if sc.spearman_conf_vs_iou is not None else 0.0
        auc = sc.auc_accurate_vs_conf if sc.auc_accurate_vs_conf is not None else 0.0
        iou = sc.median_iou_pred if sc.median_iou_pred is not None else 0.0

        print(
            f'{thresh:10.2f}  {sc.n_corrected:9d}  {sc.n_flagged:7d}  '
            f'{iou:8.3f}  {spearman:8.3f}  {auc:6.3f}'
        )

        # Prefer high Spearman + AUC with reasonable coverage.
        cal_score = spearman + auc
        if best is None or cal_score > best[0]:
            best = (cal_score, thresh, sc)

    if best:
        _, best_t, best_sc = best
        print(f'\nSuggested starting threshold: {best_t:.2f}')
        print(best_sc)


if __name__ == '__main__':
    main()
