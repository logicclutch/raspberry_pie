"""labels.csv: the one table shared by harvest, labeler and split.

Every writer takes the same lock file, re-reads the CSV, changes only its own rows/columns and
replaces the file atomically (temp file + os.replace), so a crash never leaves a half-written CSV
and two writers (two labeler tabs, a harvest run) never lose each other's changes.
"""

from __future__ import annotations

import contextlib
import csv
import fcntl
import os
import re
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path

LABELS_CSV = "labels.csv"
IMAGES_DIR = "images"

COLUMNS: tuple[str, ...] = (
    "image_path",  # OCR-ready crop (same pixels the runtime OCR sees), relative to the data dir
    "raw_image_path",  # padded crop straight from the frame (no de-shear / CLAHE)
    "source",  # --source-name, e.g. gate_cam
    "video",  # absolute path of the video it came from
    "frame",  # 0-based frame index in the video
    "track",  # tracker id (restarts per video)
    "box",  # detector box "x1 y1 x2 y2" in full-frame pixels
    "width_px",  # box width
    "two_line",  # 1 = looks like a 2-line plate (guess from the de-shear analysis), else 0
    "prep",  # which path made the OCR crop: deshear | plain | deskew
    "sharpness",  # Laplacian variance of the unpadded crop
    "ocr_text",  # raw OCR output (suggestion only, never ground truth)
    "ocr_conf",  # mean per-character confidence
    "ocr_valid",  # 1 = strict Indian validator accepted the read, else 0
    "ocr_plate",  # validator's canonical text when ocr_valid = 1
    "plate_text",  # HUMAN label (empty until someone types / confirms it)
    "status",  # unverified | verified | skip | unreadable
    "split",  # train | val | test (set by training.split), empty otherwise
    "labeler",  # who labelled it
    "labeled_at",  # ISO-8601 local time with offset
)
STATUSES: tuple[str, ...] = ("unverified", "verified", "skip", "unreadable")
HUMAN_COLUMNS: tuple[str, ...] = ("plate_text", "status", "labeler", "labeled_at")

PLATE_TEXT_RE = re.compile(r"[A-Z0-9]{1,12}")
SOURCE_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}")

Row = dict[str, str]

_thread_lock = threading.Lock()


def row_key(row: Row) -> tuple[str, str, str, str]:
    """Identity of a harvested crop: the same video frame + track is never added twice."""
    return (row["source"], row["video"], str(row["frame"]), str(row["track"]))


def track_key(row: Row) -> tuple[str, str, str]:
    """One vehicle pass: all crops of a track show the same plate."""
    return (row["source"], row["video"], str(row["track"]))


def normalize_plate(text: str) -> str:
    """Uppercase and keep only A-Z / 0-9 (what a human label is stored as)."""
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def labels_path(data_dir: Path) -> Path:
    return Path(data_dir) / LABELS_CSV


@contextlib.contextmanager
def locked(data_dir: Path) -> Iterator[None]:
    """Exclusive lock for a read-modify-write of labels.csv (across processes and threads)."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    with _thread_lock, open(data_dir / (LABELS_CSV + ".lock"), "a+") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def read_rows(data_dir: Path) -> list[Row]:
    """All rows (missing file = empty). Unknown extra columns are dropped, missing ones are blank."""
    path = labels_path(data_dir)
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [{c: (r.get(c) or "") for c in COLUMNS} for r in csv.DictReader(f)]


def write_rows(data_dir: Path, rows: list[Row]) -> None:
    """Atomically replace labels.csv. Call inside `locked()`."""
    path = labels_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".labels.", suffix=".csv.tmp", dir=path.parent)
    try:
        # mkstemp makes the file 0600: keep the old file's mode (a new file gets 0644)
        try:
            mode = path.stat().st_mode & 0o777
        except FileNotFoundError:
            mode = 0o644
        os.chmod(tmp, mode)
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in COLUMNS})
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def append_new_rows(data_dir: Path, new_rows: list[Row]) -> tuple[int, int]:
    """Add rows whose key is not in the CSV yet. Existing rows (and their human labels) are never
    touched. Returns (added, already_present)."""
    with locked(data_dir):
        rows = read_rows(data_dir)
        seen = {row_key(r) for r in rows}
        added = 0
        for r in new_rows:
            k = row_key(r)
            if k in seen:
                continue
            seen.add(k)
            rows.append({c: str(r.get(c, "")) for c in COLUMNS})
            added += 1
        if added:
            write_rows(data_dir, rows)
    return added, len(new_rows) - added
