"""Turn a deployed Pi's saved plate events into labelable training crops - so the fine-tuning dataset
builds itself from real camera traffic (especially the ingest / camera-push mode).

Every event the engine saves already keeps the OCR-ready plate crop and the full snapshot in the Pi's
data folder. This copies those crops into the training dataset as `unverified` rows (the voted plate is
kept only as a hint, never as the answer), so you can label them in `training.labeler` and fine-tune.

Typical flow (vendor side, on the Mac):
  1. Copy a Pi's data folder over, e.g.  rsync -a pi@PI:/opt/anpr/data/ ./pi-data/
  2. python -m training.import_events --db pi-data/anpr.db --images pi-data/images --source gvd_cam
  3. python -m training.labeler --data training/data     # type the correct plate for each crop
  4. training/train.sh --name gvd-v1                      # fine-tune + gated eval

Re-running on the same database adds nothing twice (dedup by source + event id).
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

import cv2

from training.dataset import IMAGES_DIR, SOURCE_NAME_RE, append_new_rows

# Columns we read from the Pi's events table (older DBs may lack camera/hsrp).
_BASE = "id, plate, kind, confidence, votes, last_seen, crop_path, snapshot_path"


def _guess_two_line(crop_path: Path) -> str:
    """A 2-line plate crop is much squarer than a 1-line one. Best-effort from the crop's aspect ratio."""
    img = cv2.imread(str(crop_path))
    if img is None or img.shape[0] == 0:
        return "0"
    h, w = img.shape[:2]
    return "1" if (w / h) < 2.4 else "0"


def import_events(db: Path, images: Path, data_dir: Path, source: str, min_conf: float) -> tuple[int, int]:
    if not SOURCE_NAME_RE.fullmatch(source):
        raise SystemExit(f"--source must be letters/digits/_/-, up to 40 chars (got {source!r})")
    if not db.exists():
        raise SystemExit(f"database not found: {db}")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute(f"SELECT {_BASE} FROM events ORDER BY id").fetchall()
    finally:
        con.close()

    out_dir = data_dir / IMAGES_DIR / source
    out_dir.mkdir(parents=True, exist_ok=True)
    video_tag = f"ingest:{db.stem}"  # stable pseudo-source so re-imports dedup by event id
    new_rows = []
    skipped_noconf = skipped_nocrop = 0
    for r in rows:
        eid, plate, kind, conf, votes, last_seen, crop_rel, snap_rel = r
        if conf is not None and float(conf) < min_conf:
            skipped_noconf += 1
            continue
        if not crop_rel:
            skipped_nocrop += 1
            continue
        src_crop = images / crop_rel
        if not src_crop.exists():
            skipped_nocrop += 1
            continue
        rel_crop = f"{IMAGES_DIR}/{source}/evt_{int(eid):08d}.png"
        shutil.copyfile(src_crop, data_dir / rel_crop)
        rel_raw = rel_crop
        if snap_rel and (images / snap_rel).exists():
            rel_raw = f"{IMAGES_DIR}/{source}/evt_{int(eid):08d}_snap.png"
            shutil.copyfile(images / snap_rel, data_dir / rel_raw)
        two_line = _guess_two_line(data_dir / rel_crop)
        new_rows.append(
            {
                "image_path": rel_crop,
                "raw_image_path": rel_raw,
                "source": source,
                "video": video_tag,
                "frame": str(int(eid)),  # event id -> unique, stable dedup key
                "track": str(int(eid)),  # one event = one vehicle pass
                "box": "",
                "width_px": "",
                "two_line": two_line,
                "prep": "ingest",
                "sharpness": "",
                "ocr_text": plate or "",  # the device's voted read: a hint only
                "ocr_conf": f"{float(conf):.4f}" if conf is not None else "",
                "ocr_valid": "1" if plate else "0",
                "ocr_plate": plate or "",
                "plate_text": "",  # a human types the real plate in training.labeler
                "status": "unverified",
                "split": "",
                "labeler": "",
                "labeled_at": "",
            }
        )
    added, dup = append_new_rows(data_dir, new_rows)
    return added, dup


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m training.import_events",
        description="Import a Pi's saved plate events as labelable training crops.",
    )
    ap.add_argument("--db", type=Path, required=True, help="the Pi's anpr.db (copied over)")
    ap.add_argument("--images", type=Path, required=True, help="the Pi's data/images folder (copied over)")
    ap.add_argument("--source", required=True, help="a name for this camera, e.g. gvd_cam")
    ap.add_argument("--data", type=Path, default=Path("training/data"), help="training dataset folder")
    ap.add_argument("--min-conf", type=float, default=0.0, help="skip events below this confidence")
    args = ap.parse_args(argv)
    added, dup = import_events(args.db, args.images, args.data, args.source, args.min_conf)
    ts = datetime.now(UTC).astimezone().isoformat(timespec="seconds")
    print(f"[{ts}] imported {added} new crop(s) ({dup} already present) into {args.data}/labels.csv")
    print(f"next: label them  ->  .venv/bin/python -m training.labeler --data {args.data}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
