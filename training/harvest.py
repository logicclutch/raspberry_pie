"""Harvest plate crops for OCR fine-tuning from recorded videos.

    .venv/bin/python -m training.harvest --config config.yaml --video gate.mp4 --source-name gate_cam \
        --out training/data [--every N] [--max-per-track K]

Every frame (or every Nth) goes through the REAL runtime parts: detector -> IoU tracker ->
prepare_plate (same CropConfig as the engine: de-shear, CLAHE ...) -> OCR -> strict validator. So the
saved crops are exactly what the OCR sees on the station. For every track the K most useful crops are
kept: sharpest, spread over the track's lifetime, no near-duplicates. The padded raw crop (no
de-shear/CLAHE) is saved next to it for trainers that do their own preprocessing.

OCR text is only a SUGGESTION (status "unverified", plate_text empty): a human labels it in
training.labeler. Re-running on the same video adds nothing twice and never touches human labels.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from hashlib import sha1
from pathlib import Path

import cv2
import numpy as np

from anpr.config import AppConfig, CropConfig, load_config
from anpr.preprocess import TWO_LINE_ASPECT, analyse_plate, crop_plate, prepare_plate, sharpness
from anpr.tracker import IouTracker
from anpr.types import Box, PlateDetector, PlateOcr
from anpr.validator import PlateValidator
from training.dataset import IMAGES_DIR, SOURCE_NAME_RE, Row, append_new_rows, read_rows, row_key

DEFAULT_MAX_PER_TRACK = 4
DEFAULT_MIN_DIFF = 6.0  # mean abs grey difference (0-255) below which two crops count as duplicates
MAX_CANDIDATES = 240  # per track; beyond this every other candidate is dropped (bounded memory)
DUP_SIZE = (64, 32)  # crops are compared at this size

Prepare = Callable[[np.ndarray, Box, CropConfig], "np.ndarray | None"]


@dataclass(slots=True)
class Candidate:
    frame: int
    box: Box
    crop: np.ndarray  # OCR-ready crop (prepare_plate output)
    raw: np.ndarray  # padded crop, untouched
    sharp: float
    two_line: bool
    prep: str
    thumb: np.ndarray = field(repr=False)


@dataclass(slots=True)
class VideoStats:
    video: str
    frames: int = 0
    processed: int = 0
    tracks: int = 0
    tracks_kept: int = 0
    crops: int = 0
    added: int = 0
    existing: int = 0
    two_line: int = 0
    ocr_valid: int = 0
    seconds: float = 0.0


# ---- helpers ------------------------------------------------------------------------------------


def slug(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_-")[:max_len].strip("_-")
    return s or "video"


def video_tag(video: Path) -> str:
    """Folder name for one video's crops: readable stem + short hash of the full path (unique)."""
    return f"{slug(video.stem, 32)}_{sha1(str(video).encode()).hexdigest()[:6]}"


def looks_two_line(frame: np.ndarray, box: Box) -> bool:
    """Same test deshear_plate uses to hand a plate back to the plain path (2-line plate)."""
    fh, fw = frame.shape[:2]
    inner = frame[max(0, box.y1) : min(fh, box.y2), max(0, box.x1) : min(fw, box.x2)]
    if inner.size == 0:
        return False
    gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY) if inner.ndim == 3 else inner
    _t, band = analyse_plate(gray)
    if band is None:
        return False
    bx0, bx1, by0, by1 = band
    return bool(bx1 - bx0 < TWO_LINE_ASPECT * (by1 - by0))


def prep_kind(cfg: CropConfig, two_line: bool) -> str:
    if cfg.deshear and not two_line:
        return "deshear"
    return "deskew" if cfg.deskew else "plain"


def thumb(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.resize(gray, DUP_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)


def is_duplicate(c: Candidate, kept: list[Candidate], min_diff: float) -> bool:
    return any(float(np.abs(c.thumb - k.thumb).mean()) < min_diff for k in kept)


def select_diverse(cands: list[Candidate], k: int, min_diff: float = DEFAULT_MIN_DIFF) -> list[Candidate]:
    """Up to k crops: the sharpest of each of k time segments of the track, then the sharpest of the
    rest; a crop too similar to one already kept is skipped. Returned in frame order."""
    if k <= 0 or not cands:
        return []
    ordered = sorted(cands, key=lambda c: c.frame)
    kept: list[Candidate] = []
    for seg in np.array_split(np.arange(len(ordered)), min(k, len(ordered))):
        for i in sorted(seg.tolist(), key=lambda j: -ordered[j].sharp):
            if not is_duplicate(ordered[i], kept, min_diff):
                kept.append(ordered[i])
                break
    for c in sorted(ordered, key=lambda c: -c.sharp):
        if len(kept) >= k:
            break
        if all(c is not x for x in kept) and not is_duplicate(c, kept, min_diff):
            kept.append(c)
    return sorted(kept, key=lambda c: c.frame)


def write_png(path: Path, img: np.ndarray) -> None:
    """Lossless, atomic (a crash never leaves a truncated image)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.png")
    if not cv2.imwrite(str(tmp), img):
        raise OSError(f"could not write {path}")
    try:
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
        raise


# ---- core ---------------------------------------------------------------------------------------


def harvest_video(
    video: Path,
    source: str,
    out_dir: Path,
    cfg: AppConfig,
    detector: PlateDetector,
    ocr: PlateOcr,
    validator: PlateValidator,
    *,
    every: int = 1,
    max_per_track: int = DEFAULT_MAX_PER_TRACK,
    min_diff: float = DEFAULT_MIN_DIFF,
    prepare: Prepare = prepare_plate,
    log: Callable[[str], None] | None = None,
) -> VideoStats:
    """Replay one video, save the kept crops and append their rows to labels.csv."""
    if not SOURCE_NAME_RE.fullmatch(source):
        raise ValueError(f"source name must match {SOURCE_NAME_RE.pattern}: {source!r}")
    video = Path(video).resolve()
    out_dir = Path(out_dir)
    t0 = time.monotonic()
    stats = VideoStats(video=str(video))
    cap = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not (math.isfinite(fps) and 0.5 <= fps <= 1000.0):
        fps = float(cfg.camera.fps)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    tag = video_tag(video)
    existing = {row_key(r) for r in read_rows(out_dir)}
    tracker = IouTracker(cfg.tracking)
    pending: dict[int, list[Candidate]] = {}
    rows: list[Row] = []

    def finish(tid: int) -> None:
        cands = pending.pop(tid, [])
        stats.tracks += 1
        kept = select_diverse(cands, max_per_track, min_diff)
        if kept:
            stats.tracks_kept += 1
        for c in kept:
            stats.crops += 1
            stats.two_line += int(c.two_line)
            row: Row = {
                "source": source,
                "video": str(video),
                "frame": str(c.frame),
                "track": str(tid),
            }
            if row_key(row) in existing:
                stats.existing += 1
                continue
            res = ocr.read(c.crop)
            valid = validator.validate(res) if res is not None else None
            stats.ocr_valid += int(valid is not None)
            name = f"t{tid:04d}_f{c.frame:06d}"
            rel = f"{IMAGES_DIR}/{source}/{tag}/{name}.png"
            rel_raw = f"{IMAGES_DIR}/{source}/{tag}/{name}_raw.png"
            write_png(out_dir / rel, c.crop)
            write_png(out_dir / rel_raw, c.raw)
            row.update(
                image_path=rel,
                raw_image_path=rel_raw,
                box=f"{c.box.x1} {c.box.y1} {c.box.x2} {c.box.y2}",
                width_px=str(c.box.width),
                two_line="1" if c.two_line else "0",
                prep=c.prep,
                sharpness=f"{c.sharp:.1f}",
                ocr_text=res.text if res else "",
                ocr_conf=f"{sum(res.char_confs) / len(res.char_confs):.4f}" if res else "",
                ocr_valid="1" if valid else "0",
                ocr_plate=valid.text if valid else "",
                plate_text="",
                status="unverified",
                split="",
                labeler="",
                labeled_at="",
            )
            rows.append(row)

    idx = -1
    try:
        while True:
            idx += 1
            if idx % every:
                if not cap.grab():
                    break
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            stats.processed += 1
            assigned, expired = tracker.update(detector.detect(frame), idx / fps)
            for tr in expired:
                finish(tr.id)
            for tid, box in assigned:
                crop = prepare(frame, box, cfg.crop)
                if crop is None:
                    continue
                raw = crop_plate(frame, box, cfg.crop.padding)
                inner = crop_plate(frame, box, 0.0)
                if raw is None or inner is None:
                    continue
                two = looks_two_line(frame, box)
                cands = pending.setdefault(tid, [])
                cands.append(
                    Candidate(
                        frame=idx,
                        box=box,
                        crop=crop.copy(),
                        raw=raw.copy(),
                        sharp=sharpness(inner),
                        two_line=two,
                        prep=prep_kind(cfg.crop, two),
                        thumb=thumb(crop),
                    )
                )
                if len(cands) > MAX_CANDIDATES:
                    pending[tid] = cands[::2]
            if log and stats.processed % 500 == 0:
                log(f"  {video.name}: frame {idx}/{n_frames or '?'} tracks={stats.tracks + len(pending)}")
    finally:
        cap.release()
    for tid in sorted(pending):
        finish(tid)
    stats.frames = idx
    stats.added, dup = append_new_rows(out_dir, rows)
    stats.existing += dup
    stats.seconds = time.monotonic() - t0
    return stats


def format_summary(all_stats: list[tuple[str, VideoStats]], seconds: float) -> str:
    lines = [
        f"{'source':<12} {'frames':>7} {'tracks':>6} {'w/crops':>7} {'crops':>6} {'new':>5} "
        f"{'had':>5} {'2-line':>6} {'ocr ok':>6} {'time':>7}  video"
    ]
    tot = VideoStats(video="TOTAL")
    for source, s in all_stats:
        lines.append(
            f"{source:<12} {s.processed:>7} {s.tracks:>6} {s.tracks_kept:>7} {s.crops:>6} {s.added:>5} "
            f"{s.existing:>5} {s.two_line:>6} {s.ocr_valid:>6} {s.seconds:>6.1f}s  {Path(s.video).name}"
        )
        for f in (
            "processed",
            "tracks",
            "tracks_kept",
            "crops",
            "added",
            "existing",
            "two_line",
            "ocr_valid",
        ):
            setattr(tot, f, getattr(tot, f) + getattr(s, f))
    lines.append(
        f"{'TOTAL':<12} {tot.processed:>7} {tot.tracks:>6} {tot.tracks_kept:>7} {tot.crops:>6} "
        f"{tot.added:>5} {tot.existing:>5} {tot.two_line:>6} {tot.ocr_valid:>6} {seconds:>6.1f}s"
    )
    lines.append(
        "(new = rows added now, had = already in labels.csv; ocr ok = new crops the validator accepted)"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m training.harvest", description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--video", action="append", required=True, type=Path, help="repeat for several videos")
    ap.add_argument("--source-name", help="camera/source name (default: the video's file name)")
    ap.add_argument("--out", type=Path, default=Path("training/data"))
    ap.add_argument("--every", type=int, default=1, help="process every Nth frame (default 1 = all)")
    ap.add_argument("--max-per-track", type=int, default=DEFAULT_MAX_PER_TRACK)
    ap.add_argument("--min-diff", type=float, default=DEFAULT_MIN_DIFF, help="near-duplicate threshold")
    args = ap.parse_args(argv)
    if args.every < 1 or args.max_per_track < 1:
        ap.error("--every and --max-per-track must be >= 1")
    if args.source_name and not SOURCE_NAME_RE.fullmatch(args.source_name):
        ap.error(f"--source-name must match {SOURCE_NAME_RE.pattern}")
    for v in args.video:
        if not v.is_file():
            ap.error(f"video not found: {v}")

    from anpr.detector import make_detector
    from anpr.ocr import FastPlateOcr

    cfg = load_config(args.config)
    detector = make_detector(cfg.detector)
    ocr = FastPlateOcr(cfg.ocr)
    validator = PlateValidator(cfg.validation)
    t0 = time.monotonic()
    results: list[tuple[str, VideoStats]] = []
    for v in args.video:
        source = args.source_name or slug(v.stem)
        print(f"harvest {v} -> {args.out} (source {source})", file=sys.stderr, flush=True)
        s = harvest_video(
            v,
            source,
            args.out,
            cfg,
            detector,
            ocr,
            validator,
            every=args.every,
            max_per_track=args.max_per_track,
            min_diff=args.min_diff,
            log=lambda m: print(m, file=sys.stderr, flush=True),
        )
        results.append((source, s))
    print(format_summary(results, time.monotonic() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
