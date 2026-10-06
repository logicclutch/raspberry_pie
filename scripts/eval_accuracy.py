#!/usr/bin/env python3
"""Accuracy gate (WORKFLOW.md A6). Exit code 0 = PASS, 1 = FAIL.

  python scripts/eval_accuracy.py --config config.yaml --images data/test/labels.csv
  python scripts/eval_accuracy.py --config config.yaml --videos data/test/videos.csv

images CSV: image,plate      (empty plate = image with no readable plate)
videos CSV: video,plates     (plates separated by ';')
Paths in the CSV are relative to the CSV file.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anpr.config import load_config  # noqa: E402
from anpr.evaluate import eval_images, eval_videos, read_labels  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--images", type=Path)
    mode.add_argument("--videos", type=Path)
    ap.add_argument("--min-precision", type=float, default=0.995)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    from anpr.detector import make_detector
    from anpr.ocr import FastPlateOcr

    cfg = load_config(args.config)
    detector, ocr = make_detector(cfg.detector), FastPlateOcr(cfg.ocr)
    if args.images:
        score = eval_images(cfg, read_labels(args.images), detector, ocr)
    else:
        score = eval_videos(cfg, read_labels(args.videos), detector, ocr)
    print(score.report(args.min_precision))
    return 0 if score.precision >= args.min_precision else 1


if __name__ == "__main__":
    sys.exit(main())
