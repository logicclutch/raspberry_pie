"""Check a camera stream (or a recording of it) before relying on it for number plates.

    .venv/bin/python -m scripts.probe_camera URL [URL ...] [--seconds 10] [--save-dir /tmp/probe]
    .venv/bin/python -m scripts.probe_camera \\
        rtsp://user:pass@192.168.5.53:554/avstream/channel=1/stream=1.sdp --guess-main --seconds 20

For each address (RTSP link or video file) it reports: codec, resolution, frame rate (declared and
measured), bitrate, keyframe interval and read errors. Then it decodes a few frames per second, runs
the real plate detector on them and reports how WIDE the plates are in pixels and how SHARP the plate
crops are, with plain hints (use the main stream, H.264, 10-15 fps, faster shutter, zoom in ...).
Targets: 1-line plate >= 120 px wide, 2-line plate >= 80 px, crop sharpness >= crop.min_sharpness.
Plates are measured in the picture the engine will see: wider streams are shrunk to --output-width
(default: camera.output_width from config.yaml, else 1280), like camera.output_width does.
Frames are checked one at a time, so a long run (--seconds 60) needs no more memory than a short one.

--guess-main also tries the usual MAIN stream next to a sub-stream address, for two well-known URL
patterns only: .../avstream/channel=N/stream=1.sdp -> stream=0.sdp, and Hikvision-style
/Streaming/Channels/102 -> 101. Passwords are shown as *** everywhere. Needs ffmpeg + ffprobe.
"""

from __future__ import annotations

import argparse
import heapq
import logging
import math
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from anpr.camera import (
    FFMPEG_BIN,
    FFPROBE_BIN,
    StreamInfo,
    _read_exact,
    _terminate,
    ffmpeg_input,
    ffmpeg_select_expr,
    input_options,
    mask_text,
    parse_probe,
    probe_command,
    scaled_size,
)
from anpr.sources import mask_source

ONE_LINE_MIN_PX = 120  # plate width targets (see docs/CAMERA_SETUP.md)
TWO_LINE_MIN_PX = 80
ONE_LINE_ASPECT = 3.0  # box width / height: 1-line ~4.2, 2-line ~1.7
MAX_SAVED_FRAMES = 3
MAX_SAVED_CROPS = 12

_AVSTREAM_SUB = re.compile(r"(/avstream/channel=\d+/stream=)1(\.sdp)", re.IGNORECASE)
_HIK_SUB = re.compile(r"(/Streaming/Channels/\d*?)02(?=$|[/?#])", re.IGNORECASE)


# -- addresses -------------------------------------------------------------------------------------------


def guess_main_urls(url: str) -> list[str]:
    """Likely MAIN-stream addresses for a SUB-stream address (only the two well-known patterns)."""
    out = []
    for pattern, repl in ((_AVSTREAM_SUB, r"\g<1>0\g<2>"), (_HIK_SUB, r"\g<1>01")):
        guess = pattern.sub(repl, url, count=1)
        if guess != url and guess not in out:
            out.append(guess)
    return out


def candidates(urls: list[str], guess_main: bool) -> list[tuple[str, str | None]]:
    """(address, guessed-from address or None), in order, without duplicates."""
    seen: set[str] = set()
    out: list[tuple[str, str | None]] = []
    for url in urls:
        items: list[tuple[str, str | None]] = [(url, None)]
        if guess_main:
            items += [(g, url) for g in guess_main_urls(url)]
        for addr, origin in items:
            if addr not in seen:
                seen.add(addr)
                out.append((addr, origin))
    return out


# -- report ----------------------------------------------------------------------------------------------


@dataclass(slots=True)
class PlateSample:
    width: int  # detector box width in px (what crop.min_width_px is compared with)
    height: int
    sharpness: float  # variance of the Laplacian of the plate crop (the pipeline's crop.min_sharpness)
    score: float = 1.0
    kind: str = ""  # "1-line" | "2-line"; "" = guess from the box shape
    slant_deg: float | None = None  # text baseline angle; side-mounted cameras give 25-40 deg

    def __post_init__(self) -> None:
        if not self.kind:
            self.kind = "1-line" if self.width >= ONE_LINE_ASPECT * max(1, self.height) else "2-line"


@dataclass(slots=True)
class StreamReport:
    address: str  # masked
    guessed_from: str | None = None  # masked
    error: str | None = None
    info: StreamInfo | None = None
    packets: int = 0
    measured_fps: float | None = None
    bitrate_kbps: float | None = None
    keyframe_every_frames: float | None = None
    keyframe_every_s: float | None = None
    read_errors: int = 0
    frames_sampled: int = 0
    plates: list[PlateSample] = field(default_factory=list)
    saved: list[str] = field(default_factory=list)
    detector_used: bool = True
    min_sharpness: float = 60.0
    analysed_size: tuple[int, int] | None = None  # (w, h) the plates were measured in
    hints: list[str] = field(default_factory=list)


def _fmt(value: float | None, spec: str = ".1f", unit: str = "") -> str:
    return "?" if value is None else f"{value:{spec}}{unit}"


def _stats(values: list[float]) -> str:
    if not values:
        return "-"
    return f"min {min(values):.0f} / median {statistics.median(values):.0f} / max {max(values):.0f}"


def make_hints(r: StreamReport) -> list[str]:
    hints: list[str] = []
    info = r.info
    if info is None:
        return hints
    if info.codec != "h264":
        hints.append(
            f"codec is {info.codec.upper()}: set the camera to H.264 (the Pi 3B+ hardware-decodes only H.264)"
        )
    if info.width < 1280:
        hints.append(
            f"{info.width}x{info.height} is low: this is probably the SUB-stream; use the MAIN stream "
            "(1280x720, or 1920x1080 with camera.output_width 1280)"
        )
    fps = r.measured_fps or info.fps
    if fps is not None and fps < 9.5:
        hints.append(f"{fps:.1f} fps is too few frames per vehicle: set 10-15 fps on the camera")
    elif fps is not None and fps > 16:
        hints.append(f"{fps:.0f} fps: set 10-15 (camera.max_fps drops the rest, but they still get decoded)")
    if r.keyframe_every_s is not None and r.keyframe_every_s > 2.5:
        hints.append(f"keyframe every {r.keyframe_every_s:.1f} s: set the I-frame interval = fps (1 s)")
    if r.bitrate_kbps is not None and info.width >= 1280 and r.bitrate_kbps < 1500:
        hints.append(
            f"bitrate {r.bitrate_kbps:.0f} kbit/s is low for {info.width}x{info.height}: CBR 2-4 Mbit/s"
        )
    if r.read_errors:
        hints.append(f"{r.read_errors} read errors: check the network/cable and the camera bitrate")
    if not (r.detector_used and r.frames_sampled):
        return hints
    if not r.plates:
        hints.append("no plate seen in the sample: run longer (--seconds 60) while vehicles pass")
        return hints
    for kind, target in (("1-line", ONE_LINE_MIN_PX), ("2-line", TWO_LINE_MIN_PX)):
        widths = [p.width for p in r.plates if p.kind == kind]
        if widths and statistics.median(widths) < target:
            med = statistics.median(widths)
            hints.append(
                f"{kind} plates are {med:.0f} px wide (median): zoom in / main stream, >= {target} px"
            )
    sharp = statistics.median(p.sharpness for p in r.plates)
    if sharp < r.min_sharpness:
        hints.append(
            f"plate crops are blurry (median sharpness {sharp:.0f} < {r.min_sharpness:.0f}): "
            "faster shutter (1/1000 s), check focus"
        )
    slants = [abs(p.slant_deg) for p in r.plates if p.slant_deg is not None]
    if slants and statistics.median(slants) > 20:
        hints.append(
            f"plate text is slanted {statistics.median(slants):.0f} deg (median): aim the camera more along "
            "the lane (< 20 deg); keep crop.deshear true meanwhile"
        )
    return hints


def format_report(r: StreamReport) -> str:
    lines = [f"== {r.address}"]
    if r.guessed_from:
        lines.append(f"   (guessed main stream of {r.guessed_from})")
    if r.error:
        lines.append(f"   ERROR: {r.error}")
        return "\n".join(lines)
    info = r.info
    assert info is not None
    declared_br = f" (declared {info.bit_rate / 1000:.0f})" if info.bit_rate else ""
    kf = f"{_fmt(r.keyframe_every_frames, '.0f', ' frames')}, {_fmt(r.keyframe_every_s, '.1f', ' s')}"
    lines += [
        f"   codec            {info.codec}",
        f"   resolution       {info.width}x{info.height}",
        f"   fps              declared {_fmt(info.fps)}, measured {_fmt(r.measured_fps)}"
        f" ({r.packets} frames)",
        f"   bitrate          {_fmt(r.bitrate_kbps, '.0f', ' kbit/s')}{declared_br}",
        f"   keyframe every   {kf}",
        f"   read errors      {r.read_errors}",
    ]
    if not r.detector_used:
        lines.append(f"   plates           not checked (--no-detect), {r.frames_sampled} frames sampled")
    else:
        one = [p.width for p in r.plates if p.kind == "1-line"]
        two = [p.width for p in r.plates if p.kind == "2-line"]
        slants = [abs(p.slant_deg) for p in r.plates if p.slant_deg is not None]
        sharp = [p.sharpness for p in r.plates]
        at = f" at {r.analysed_size[0]}x{r.analysed_size[1]}" if r.analysed_size else ""
        lines += [
            f"   plates           {len(r.plates)} boxes in {r.frames_sampled} sampled frames{at}",
            f"     1-line width   {_stats(one)} px (target >= {ONE_LINE_MIN_PX}, {len(one)} boxes)",
            f"     2-line width   {_stats(two)} px (target >= {TWO_LINE_MIN_PX}, {len(two)} boxes)",
            f"     sharpness      {_stats(sharp)} (pipeline minimum {r.min_sharpness:.0f}; compare settings)",
            f"     text slant     {_stats(slants)} deg (target < 20)",
        ]
    if r.saved:
        crops = sum("_plate" in Path(p).name for p in r.saved)
        folder = Path(r.saved[0]).parent
        lines.append(f"   saved            {len(r.saved) - crops} frames + {crops} plate crops in {folder}")
    hints = r.hints or ["looks good for plate reading"]
    lines += [f"   -> {h}" for h in hints]
    return "\n".join(lines)


# -- measuring ---------------------------------------------------------------------------------------------


def _run(cmd: list[str], timeout: float) -> tuple[int | None, bytes, bytes]:
    """Run with a wall-clock limit; on timeout keep what was produced so far."""
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
    return proc.returncode, out, err


def parse_packets(csv_text: str, declared_fps: float | None) -> dict[str, float | int | None]:
    """ffprobe `-show_entries packet=pts_time,dts_time,size,flags -of csv=p=0` -> timing statistics."""
    times: list[float] = []
    sizes: list[int] = []
    keys: list[int] = []
    for line in csv_text.splitlines():
        parts = line.strip().split(",")
        if len(parts) < 4:
            continue
        pts, dts, size, flags = parts[0], parts[1], parts[2], parts[3]
        t = None
        for v in (pts, dts):
            try:
                t = float(v)
                break
            except ValueError:
                continue
        try:
            sizes.append(int(size))
        except ValueError:
            continue
        if t is not None:
            times.append(t)
        if "K" in flags:
            keys.append(len(sizes) - 1)
    n = len(sizes)
    span = (max(times) - min(times)) if len(times) >= 2 else 0.0
    fps = (len(times) - 1) / span if span > 0 else None
    frame_s = 1.0 / (fps or declared_fps or 0) if (fps or declared_fps) else 0.0
    duration = span + frame_s
    bitrate = sum(sizes) * 8 / duration / 1000 if duration > 0 and n else None
    gap = (keys[-1] - keys[0]) / (len(keys) - 1) if len(keys) >= 2 else None
    rate = fps or declared_fps
    return {
        "packets": n,
        "measured_fps": fps,
        "bitrate_kbps": bitrate,
        "keyframe_every_frames": gap,
        "keyframe_every_s": gap / rate if gap is not None and rate else None,
    }


def measure_stream(src: str, seconds: float, start: float, ffprobe: str = FFPROBE_BIN) -> tuple[dict, int]:
    interval = f"{start}%+{seconds}" if "://" not in src else f"%+{seconds}"
    cmd = [
        ffprobe, "-v", "error", "-hide_banner", *input_options(src), "-select_streams", "v:0",
        "-read_intervals", interval, "-show_entries", "packet=pts_time,dts_time,size,flags",
        "-of", "csv=p=0", ffmpeg_input(src),
    ]  # fmt: skip
    _code, out, err = _run(cmd, timeout=seconds + 25.0)
    errors = [ln for ln in err.decode("utf-8", "replace").splitlines() if ln.strip()]
    return parse_packets(out.decode("utf-8", "replace"), None), len(errors)


def iter_frames(
    src: str,
    size: tuple[int, int],
    seconds: float,
    start: float,
    sample_fps: float,
    ffmpeg: str = FFMPEG_BIN,
) -> Iterator[np.ndarray]:
    """Decoded BGR frames of `size`, `sample_fps` per second for `seconds`, ONE AT A TIME (a 1080p
    frame is 6 MB: collecting a minute of them would not fit in the Pi's 1 GB)."""
    w, h = size
    cmd = [ffmpeg, "-hide_banner", "-nostats", "-loglevel", "error", "-nostdin", *input_options(src)]
    if "://" not in src and start > 0:
        cmd += ["-ss", str(start)]
    cmd += ["-i", ffmpeg_input(src), "-t", str(seconds), "-map", "0:v:0", "-an", "-sn", "-dn"]
    vf = f"select='{ffmpeg_select_expr(sample_fps)}',scale={w}:{h}:flags=area"
    cmd += ["-vf", vf, "-fps_mode", "passthrough", "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0
    )
    assert proc.stdout is not None
    deadline = time.monotonic() + seconds + 60.0
    try:
        while True:
            buf = bytearray(w * h * 3)
            try:
                if not _read_exact(proc.stdout, buf, max(0.1, deadline - time.monotonic())):
                    return  # end of the sample (a partial last frame is dropped)
            except TimeoutError:
                return
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        proc.stdout.close()
        _terminate(proc)


def check_plates(
    r: StreamReport, frames: Iterable[np.ndarray], detector, save_dir: Path | None, tag: str
) -> None:  # noqa: ANN001
    """Detect plates frame by frame; only the MAX_SAVED_FRAMES best frames are kept in memory."""
    from anpr.preprocess import TWO_LINE_ASPECT, analyse_plate, crop_plate, sharpness

    best: list[tuple[int, int, np.ndarray, list]] = []  # min-heap: (widest plate, -index, frame, boxes)
    first: np.ndarray | None = None
    crops = 0
    for i, frame in enumerate(frames):
        r.frames_sampled += 1
        if first is None:
            first = frame
        boxes = detector.detect(frame)
        for b in boxes:
            inner = crop_plate(frame, b, 0.0)
            if inner is None:
                continue
            # Plate text band after de-shearing: tells 1-line from 2-line even on a slanted plate
            # (whose axis-aligned box is almost square), and gives the slant angle.
            slope, band = analyse_plate(cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY))
            kind = ""
            if band is not None:
                bx0, bx1, by0, by1 = band
                kind = "1-line" if bx1 - bx0 >= TWO_LINE_ASPECT * (by1 - by0) else "2-line"
            sample = PlateSample(
                b.width, b.height, sharpness(inner), float(b.score), kind, math.degrees(math.atan(slope))
            )
            r.plates.append(sample)
            if save_dir is not None and crops < MAX_SAVED_CROPS:
                p = save_dir / f"{tag}_plate{crops:02d}_{sample.kind}_w{b.width}_s{sample.sharpness:.0f}.png"
                cv2.imwrite(str(p), inner)
                r.saved.append(str(p))
                crops += 1
        if boxes and save_dir is not None:
            item = (max(b.width for b in boxes), -i, frame, boxes)
            if len(best) < MAX_SAVED_FRAMES:
                heapq.heappush(best, item)
            elif item[:2] > best[0][:2]:
                heapq.heapreplace(best, item)
    if save_dir is None:
        return
    # widest plate first; among equal widths the earliest frame
    chosen = [(w, -neg, f, b) for w, neg, f, b in sorted(best, key=lambda x: (-x[0], -x[1]))]
    if not chosen and first is not None:
        chosen = [(0, 0, first, [])]
    for _w, i, frame, boxes in chosen:
        img = frame.copy()
        for b in boxes:
            cv2.rectangle(img, (b.x1, b.y1), (b.x2 - 1, b.y2 - 1), (0, 200, 255), 2)
            cv2.putText(
                img,
                f"{b.width}px",
                (b.x1, max(12, b.y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 200, 255),
                1,
            )
        p = save_dir / f"{tag}_frame{i:03d}.jpg"
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        r.saved.append(str(p))


def probe_one(
    src: str,
    guessed_from: str | None,
    *,
    seconds: float,
    start: float,
    sample_fps: float,
    detector,  # noqa: ANN001 - PlateDetector or None
    min_sharpness: float,
    save_dir: Path | None,
    tag: str,
    output_width: int | None = None,
) -> StreamReport:
    r = StreamReport(
        address=mask_source(src),
        guessed_from=mask_source(guessed_from) if guessed_from else None,
        detector_used=detector is not None,
        min_sharpness=min_sharpness,
    )
    code, out, err = _run(probe_command(src), timeout=25.0)
    if code != 0:
        msg = (
            " | ".join(err.decode("utf-8", "replace").strip().splitlines()[-2:])
            or f"ffprobe exit code {code}"
        )
        r.error = mask_text(msg, src)
        return r
    try:
        r.info = parse_probe(out)
    except (ValueError, KeyError, TypeError) as e:
        r.error = str(e)
        return r
    stats, r.read_errors = measure_stream(src, seconds, start)
    r.packets = int(stats["packets"] or 0)
    r.measured_fps = stats["measured_fps"]  # type: ignore[assignment]
    r.bitrate_kbps = stats["bitrate_kbps"]  # type: ignore[assignment]
    r.keyframe_every_frames = stats["keyframe_every_frames"]  # type: ignore[assignment]
    if stats["keyframe_every_frames"] is not None:
        rate = r.measured_fps or r.info.fps
        r.keyframe_every_s = stats["keyframe_every_frames"] / rate if rate else None  # type: ignore[operator]
    r.analysed_size = scaled_size(r.info.width, r.info.height, output_width)
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    frames = iter_frames(src, r.analysed_size, seconds, start, sample_fps)
    if detector is not None:
        check_plates(r, frames, detector, save_dir, tag)
    else:
        for i, frame in enumerate(frames):
            r.frames_sampled += 1
            if save_dir is not None and i < MAX_SAVED_FRAMES:
                p = save_dir / f"{tag}_frame{i:03d}.jpg"
                cv2.imwrite(str(p), frame)
                r.saved.append(str(p))
    r.hints = make_hints(r)
    return r


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m scripts.probe_camera",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("urls", nargs="+", metavar="URL", help="rtsp:// link or video file")
    ap.add_argument("--seconds", type=float, default=10.0, help="how long to measure and sample (default 10)")
    ap.add_argument("--start", type=float, default=0.0, help="video files: start at this second")
    ap.add_argument("--sample-fps", type=float, default=2.0, help="frames per second checked for plates (2)")
    ap.add_argument("--save-dir", type=Path, help="save a few frames and plate crops here")
    ap.add_argument("--guess-main", action="store_true", help="also try the main stream of a sub-stream link")
    ap.add_argument("--config", default="config.yaml", help="detector + crop settings (config.yaml)")
    ap.add_argument("--no-detect", action="store_true", help="skip the plate detector")
    ap.add_argument(
        "--output-width",
        type=int,
        help="measure plates in pictures shrunk to this width, like camera.output_width "
        "(default: config.yaml's camera.output_width, else 1280; 0 = the stream's own size)",
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    missing = [b for b in (FFMPEG_BIN, FFPROBE_BIN) if shutil.which(b) is None]
    if missing:
        print(f"needs {' and '.join(missing)} (Pi: sudo apt install ffmpeg; Mac: brew install ffmpeg)")
        return 2
    if args.seconds <= 0 or args.sample_fps <= 0 or not math.isfinite(args.seconds):
        ap.error("--seconds and --sample-fps must be > 0")
    if args.output_width is not None and 0 < args.output_width < 64 or (args.output_width or 0) < 0:
        ap.error("--output-width must be 0 (no shrinking) or at least 64")

    from anpr.config import load_config

    cfg = load_config(args.config if Path(args.config).exists() else None)
    out_w = args.output_width if args.output_width is not None else (cfg.camera.output_width or 1280)
    detector = None
    if not args.no_detect:
        from anpr.detector import make_detector

        detector = make_detector(cfg.detector)

    todo = candidates(args.urls, args.guess_main)
    if args.guess_main:
        for addr, origin in todo:
            if origin:
                print(f"trying {mask_source(addr)}  (main stream guess for {mask_source(origin)})")
        if not any(origin for _a, origin in todo):
            print("--guess-main: no known sub-stream pattern (stream=1.sdp, /Channels/102) in the address")
    reports = []
    for n, (addr, origin) in enumerate(todo):
        r = probe_one(
            addr,
            origin,
            seconds=args.seconds,
            start=args.start,
            sample_fps=args.sample_fps,
            detector=detector,
            min_sharpness=cfg.crop.min_sharpness,
            save_dir=args.save_dir,
            tag=f"s{n}",
            output_width=out_w or None,
        )
        reports.append(r)
        print(format_report(r), flush=True)
    return 0 if any(r.error is None for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
