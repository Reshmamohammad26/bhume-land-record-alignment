#!/usr/bin/env python3
"""
Generate predictions.geojson using edge-based local alignment + calibrated confidence.

Usage:
    uv run predict.py data/34855_vadnerbhairav_chandavad_nashik
    uv run predict.py data/34855_vadnerbhairav_chandavad_nashik --compare-baseline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bhume import load, score, write_predictions
from bhume.baseline import global_median_shift
from bhume.predict import predict_village

DEFAULT_VILLAGE = 'data/34855_vadnerbhairav_chandavad_nashik'


def main() -> None:
    parser = argparse.ArgumentParser(description='BhuMe boundary correction pipeline')
    parser.add_argument('village_dir', nargs='?', default=DEFAULT_VILLAGE)
    parser.add_argument(
        '--compare-baseline', action='store_true',
        help='Also score the naive global_median_shift baseline',
    )
    parser.add_argument(
        '--flag-threshold', type=float, default=0.52,
        help='Confidence below this → flagged (default: 0.52)',
    )
    args = parser.parse_args()

    village_dir = Path(args.village_dir)
    if not (village_dir / 'input.geojson').exists():
        print(f'ERROR: No village bundle at {village_dir}/')
        print('Download a bundle from the BhuMe site and unzip into data/')
        sys.exit(1)

    village = load(village_dir)
    n_truth = 0 if village.example_truths is None else len(village.example_truths)
    print(f'Loaded {village.slug}: {len(village.plots)} plots, {n_truth} example truths')

    print('Running edge-alignment pipeline...')
    preds = predict_village(village, flag_threshold=args.flag_threshold)

    n_corrected = (preds['status'] == 'corrected').sum()
    n_flagged = (preds['status'] == 'flagged').sum()
    print(f'  {n_corrected} corrected, {n_flagged} flagged')

    out = write_predictions(village_dir / 'predictions.geojson', preds)
    print(f'  wrote → {out}')

    if village.example_truths is not None:
        print()
        print('=== Edge-alignment score ===')
        print(score(preds, village))

        if args.compare_baseline:
            baseline = global_median_shift(village)
            print()
            print('=== Baseline (global_median_shift) score ===')
            print(score(baseline, village))
    else:
        print('(No example_truths.geojson — skipping self-score)')


if __name__ == '__main__':
    main()
