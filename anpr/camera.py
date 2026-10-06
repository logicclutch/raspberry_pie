"""Frame sources (WORKFLOW.md step 1). Both implement anpr.types.FrameSource and expose `camera_ok`.

- OpenCVSource: webcam index ("0"), rtsp:// / http:// stream, or a video file.
- FfmpegSource: RTSP stream or video file decoded by the system `ffmpeg` program in a child process
  (camera.backend "ffmpeg"): hardware H.264 decoding on the Pi, frames dropped and downscaled inside
  ffmpeg, raw BGR frames on stdout. The pip OpenCV wheel's own FFmpeg can't use the Pi's decoder.
- RpiCamSource: Pi Camera Module through the `rpicam-vid` CLI (raw I420 on stdout). We deliberately do
  not use the picamera2 Python library: it needs the apt numpy, which clashes with the venv's numpy.

A background daemon thread grabs frames and keeps ONLY the latest one, so the slow inference loop always
gets a fresh frame and never falls behind. `read()` returns each frame at most once (sequence counter).
"""

from __future__ import annotations

import collections
import contextlib
import json
import logging
import math
import platform
import re
import select
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO
from urllib.parse import unquote, urlsplit

import cv2
import numpy as np

from anpr.config import CameraConfig
from anpr.sources import mask_source
from anpr.types import FrameSource

log = logging.getLogger(__name__)

BACKOFF_MIN_S = 1.0
BACKOFF_MAX_S = 10.0


class _LatestSlot:
    """Thread-safe single-slot mailbox: the producer overwrites, the consumer takes each item once."""

    def __init__(self) -> None:
        self.cond = threading.Condition()
        self.item: object | None = None
        self.ts = 0.0
        self.seq = 0  # sequence number of the item in the slot
        self.taken = 0  # sequence number of the last item handed to the consumer
        self.eof = False  # producer will never publish again
        self.closed = False  # stop() was called

    def publish(self, item: object, ts: float) -> None:
        with self.cond:
            self.item = item
            self.ts = ts
            self.seq += 1
            self.cond.notify_all()

    def wait_taken(self) -> bool:
        """Block until the consumer took the current item (lossless mode). False if closed."""
        with self.cond:
            self.cond.wait_for(lambda: self.taken == self.seq or self.closed)
            return not self.closed

    def take(self, timeout: float) -> tuple[object, float] | None:
        deadline = time.monotonic() + max(0.0, timeout)
        with self.cond:
            if self.closed:
                return None  # after stop() nothing is delivered, not even a pending frame
            while self.seq == self.taken:
                if self.eof or self.closed:
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.cond.wait(remaining)
            item, ts = self.item, self.ts
            self.item = None  # drop our reference; the consumer owns it now
            self.taken = self.seq
            self.cond.notify_all()
            return item, ts

    def set_eof(self) -> None:
        with self.cond:
            self.eof = True
            self.cond.notify_all()

    def close(self) -> None:
        with self.cond:
            self.closed = True
            self.item = None  # release the pending frame buffer
            self.cond.notify_all()

    @property
    def drained(self) -> bool:
        with self.cond:
            return self.eof and self.seq == self.taken


def _next_backoff(current: float, maximum: float) -> float:
    return min(maximum, current * 2.0)


class RateGate:
    """Frame-rate cap that only DROPS frames, never repeats them (camera.max_fps).

    A frame at time `t` passes when it is at most a quarter period early for its slot; the next slot
    then starts one period later. A 25 fps camera capped at 10 gives 10 fps, a 10 fps camera with
    arrival jitter keeps every frame, a 5 fps camera is untouched. If time goes BACKWARDS (an IP
    camera resetting its stream timestamps) the schedule restarts; otherwise every frame would be
    dropped until the new timestamps caught up with the old ones (a frozen stream for minutes).
    `ffmpeg_select_expr` is the same rule for ffmpeg's `select` filter (st/ld keep the state).
    """

    def __init__(self, max_fps: float) -> None:
        self.period = 1.0 / max_fps if max_fps > 0 else 0.0
        self.tolerance = self.period / 4.0
        self._next = -math.inf
        self._last = -math.inf  # time of the last frame let through

    def ready(self, t: float) -> bool:
        if self.period <= 0:
            return True
        if t < self._last - self.tolerance:
            self._next = -math.inf  # clock went backwards: start a new schedule
        elif t < self._next - self.tolerance:
            return False
        self._next = max(self._next + self.period, t + self.period - self.tolerance)
        self._last = t
        return True


def ffmpeg_select_expr(max_fps: float) -> str:
    """`select` expression equivalent to RateGate (ld(0) = next slot, ld(1) = started, ld(2) = time
    of the last frame let through). ffmpeg evaluates `a+b+c` left to right, so st(0, ...) still sees
    the previous ld(2)."""
    p = 1.0 / max_fps
    tol = p / 4.0
    first = f"t+{p - tol:.6g}"
    back = f"lt(t,ld(2)-{tol:.6g})"  # time went backwards
    nxt = f"if(ld(1)*not({back}),max(ld(0)+{p:.6g},{first}),{first})"
    keep = f"not(ld(1))+gte(t,ld(0)-{tol:.6g})+ld(1)*{back}"
    return f"if(isnan(t),1,if({keep},1+0*st(0,{nxt})+0*st(2,t)+0*st(1,1),0))"


def scaled_size(width: int, height: int, output_width: int | None) -> tuple[int, int]:
    """Output size for `camera.output_width`: only ever smaller, aspect ratio kept, even numbers."""
    if output_width is None or output_width >= width:
        return width, height
    w = max(2, output_width - output_width % 2)
    h = max(2, round(height * w / width / 2) * 2)
    return w, h


def _resize_to(frame: np.ndarray, output_width: int | None) -> np.ndarray:
    h, w = frame.shape[:2]
    nw, nh = scaled_size(w, h, output_width)
    if (nw, nh) == (w, h):
        return frame
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)


class OpenCVSource:
    """Webcam index, RTSP/HTTP URL or video file via cv2.VideoCapture.

    Live sources (webcam/stream): latest frame only, timestamp = time.time() at grab, reconnect with
    backoff (1 s .. 10 s) when reads fail. Video files: synthetic clock start_ts + index / file_fps;
    `cfg.realtime` True paces at the file FPS and drops frames the consumer didn't take (like a live
    camera); False delivers every frame in order (accuracy evaluation). `finished` once the file is
    exhausted and its last frame was read.

    `cfg.max_fps` caps live streams and realtime files: surplus frames are grabbed (decoded) but never
    converted to BGR. `cfg.output_width` shrinks wider frames (same size rule as FfmpegSource).
    """

    def __init__(self, cfg: CameraConfig) -> None:
        self.cfg = cfg
        src = cfg.source.strip()
        self._src = src
        self._shown = mask_source(src)  # for logs: never the RTSP password
        self.is_webcam = src.isdigit()
        self.is_stream = "://" in src
        self.is_file = not (self.is_webcam or self.is_stream)
        self.backoff_min = BACKOFF_MIN_S
        self.backoff_max = BACKOFF_MAX_S
        self._slot = _LatestSlot()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._camera_ok = False
        self.file_fps: float | None = None

    # -- FrameSource ---------------------------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        if self.is_file and not Path(self._src).is_file():
            raise FileNotFoundError(f"video file not found: {self._src}")
        target = self._run_file if self.is_file else self._run_live
        self._thread = threading.Thread(target=target, name="anpr-camera", daemon=True)
        self._thread.start()

    def read(self, timeout: float = 1.0) -> tuple[np.ndarray, float] | None:
        got = self._slot.take(timeout)
        if got is None:
            return None
        frame, ts = got
        return frame, ts  # type: ignore[return-value]

    @property
    def finished(self) -> bool:
        return self._slot.drained

    @property
    def camera_ok(self) -> bool:
        return self._camera_ok

    def stop(self) -> None:
        self._stop.set()
        self._slot.close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._camera_ok = False

    # -- internals -------------------------------------------------------------------------------
    def _open(self) -> cv2.VideoCapture:
        if self.is_webcam:
            cap = cv2.VideoCapture(int(self._src))
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
                cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        if self.is_stream:
            # Bounded open/read timeouts so a dead camera can't hang the grabber for ~30 s.
            params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000]
            cap = cv2.VideoCapture(self._src, cv2.CAP_FFMPEG, params + self._decode_params())
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        return self._open_file()

    def _decode_params(self) -> list[int]:
        """FFmpeg thread limit: fewer decoder threads = far fewer full-size frame buffers held in RAM."""
        n = self.cfg.decode_threads
        return [cv2.CAP_PROP_N_THREADS, n] if n > 0 else []

    def _open_file(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self._src, cv2.CAP_FFMPEG, self._decode_params())
        if cap.isOpened():
            return cap
        cap.release()
        return cv2.VideoCapture(self._src)  # not FFmpeg-decodable: let OpenCV pick another backend

    def _run_live(self) -> None:
        backoff = self.backoff_min
        cap: cv2.VideoCapture | None = None
        fails = 0
        gate = RateGate(self.cfg.max_fps)
        try:
            while not self._stop.is_set():
                if cap is None:
                    cap = self._open()
                    if not cap.isOpened():
                        cap.release()
                        cap = None
                        self._camera_ok = False
                        log.warning("camera %s: open failed, retry in %.0f s", self._shown, backoff)
                        self._stop.wait(backoff)
                        backoff = _next_backoff(backoff, self.backoff_max)
                        continue
                    log.info("camera %s: opened", self._shown)
                    fails = 0
                if cap.grab():
                    fails = 0
                    backoff = self.backoff_min
                    self._camera_ok = True
                    if not gate.ready(time.monotonic()):
                        continue  # over max_fps: decoded, but not worth a BGR conversion
                    ok, frame = cap.retrieve()
                    if ok and frame is not None:
                        self._slot.publish(_resize_to(frame, self.cfg.output_width), time.time())
                        continue
                fails += 1
                if fails >= 3:  # a single glitch is tolerated; persistent failure -> reconnect
                    self._camera_ok = False
                    log.warning("camera %s: read failed, reconnecting in %.0f s", self._shown, backoff)
                    cap.release()
                    cap = None
                    self._stop.wait(backoff)
                    backoff = _next_backoff(backoff, self.backoff_max)
                else:
                    self._stop.wait(0.05)
        finally:
            if cap is not None:
                cap.release()
            self._camera_ok = False
            self._slot.set_eof()

    def _run_file(self) -> None:
        cap = self._open_file()
        try:
            if not cap.isOpened():
                log.error("video %s: cannot open", self._shown)
                return
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not (math.isfinite(fps) and 0.5 <= fps <= 1000.0):
                fps = float(self.cfg.fps)
            self.file_fps = fps
            self._camera_ok = True
            start_ts = time.time()
            t0 = time.monotonic()
            index = 0  # frames delivered in total (all loops): drives pacing and timestamps
            loops = 0
            gate = RateGate(self.cfg.max_fps if self.cfg.realtime else 0.0)
            while not self._stop.is_set():
                if self.cfg.realtime:
                    delay = t0 + index / fps - time.monotonic()
                    if delay > 0 and self._stop.wait(delay):
                        break
                if not cap.grab():
                    if not (self.cfg.loop and index > 0):
                        break
                    cap.release()  # end of file: play it again from the start
                    cap = self._open_file()
                    loops += 1
                    log.info("video %s: loop %d", self._shown, loops)
                    if not cap.isOpened() or not cap.grab():
                        break
                index += 1
                if not gate.ready((index - 1) / fps):
                    continue  # realtime + max_fps: skip without converting
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    break
                if not self.cfg.realtime and not self._slot.wait_taken():
                    break  # closed while waiting for the consumer
                self._slot.publish(_resize_to(frame, self.cfg.output_width), start_ts + (index - 1) / fps)
            log.info("video %s: finished after %d frames", self._shown, index)
        finally:
            cap.release()
            self._camera_ok = False
            self._slot.set_eof()


# -- system ffmpeg (camera.backend: ffmpeg) -------------------------------------------------------------

FFMPEG_BIN = "ffmpeg"
FFPROBE_BIN = "ffprobe"
PROBE_TIMEOUT_S = 20.0
NET_IO_TIMEOUT_US = 5_000_000  # ffmpeg -timeout: socket I/O timeout in microseconds
STDERR_TAIL_LINES = 20
# Live streams: restart ffmpeg when it is running but no whole frame arrives for this long. ffmpeg's
# own -timeout only catches a silent socket; this also catches a hung hardware decoder, a stream that
# sends data but no pictures, or a select filter stuck on bad timestamps.
FIRST_FRAME_TIMEOUT_S = 30.0  # connect + stream analysis + first keyframe
STALL_TIMEOUT_S = 10.0
_URL_PASSWORD = re.compile(r"(://[^:/@\s]*:)[^@\s]*@")


def input_options(src: str) -> list[str]:
    """ffmpeg/ffprobe options placed before `-i` for this address."""
    scheme = src.split("://", 1)[0].lower() if "://" in src else ""
    if scheme in ("rtsp", "rtsps"):
        # TCP: no lost UDP packets (grey smears across plates). Bounded socket timeout: a dead camera
        # ends the process and we reconnect, instead of hanging forever.
        return ["-rtsp_transport", "tcp", "-timeout", str(NET_IO_TIMEOUT_US)]
    if scheme in ("http", "https"):
        return ["-timeout", str(NET_IO_TIMEOUT_US)]
    return []


def ffmpeg_input(src: str) -> str:
    """The `-i` argument: files get the `file:` prefix so no name can be mistaken for an option/URL."""
    return src if "://" in src else "file:" + src


def mask_text(text: str, src: str) -> str:
    """Hide the password of `src` (and of any user:password@ URL) in text such as ffmpeg messages."""
    s = src.strip()
    if s and "://" in s:
        text = text.replace(s, mask_source(s))
        try:
            pw = urlsplit(s).password
        except ValueError:
            pw = None
        if pw:
            for secret in {pw, unquote(pw)}:
                if len(secret) >= 4:  # shorter ones would blank out ordinary numbers/words
                    text = text.replace(secret, "***")
    return _URL_PASSWORD.sub(r"\1***@", text)


@dataclass(frozen=True, slots=True)
class StreamInfo:
    """What ffprobe says about the first video stream."""

    codec: str
    width: int
    height: int
    fps: float | None  # declared frame rate (avg_frame_rate, else r_frame_rate)
    bit_rate: int | None = None  # bits/s, if the container declares it
    duration_s: float | None = None  # files only


def parse_rate(value: object) -> float | None:
    """ffprobe rate such as "25/1" or "4993/1000" -> fps; None if missing or absurd."""
    try:
        if isinstance(value, str) and "/" in value:
            num, den = value.split("/", 1)
            fps = float(num) / float(den)
        else:
            fps = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return fps if math.isfinite(fps) and 0.1 <= fps <= 1000.0 else None


def _opt_number(value: object, kind: type) -> float | int | None:
    try:
        v = kind(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def _rotation(st: dict) -> int:
    """Display rotation in degrees (phone videos): side data (ffmpeg >= 5), else the old `rotate` tag."""
    for sd in st.get("side_data_list") or []:
        if isinstance(sd, dict) and "rotation" in sd:
            value = sd["rotation"]
            break
    else:
        value = (st.get("tags") or {}).get("rotate", 0)
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return 0


def parse_probe(data: str | bytes | dict) -> StreamInfo:
    """`ffprobe -show_streams -show_format -of json` output -> StreamInfo. ValueError if no video.

    Width/height are the size ffmpeg DELIVERS: it auto-rotates, so a 90/270 degree video is swapped
    (otherwise the pinned scale filter would squash the upright picture)."""
    d = json.loads(data) if isinstance(data, str | bytes) else data
    fmt = d.get("format") or {}
    for st in d.get("streams") or []:
        if st.get("codec_type", "video") != "video" or not st.get("width") or not st.get("height"):
            continue
        width, height = int(st["width"]), int(st["height"])
        if _rotation(st) % 180 == 90:
            width, height = height, width
        return StreamInfo(
            codec=str(st.get("codec_name") or "unknown"),
            width=width,
            height=height,
            fps=parse_rate(st.get("avg_frame_rate")) or parse_rate(st.get("r_frame_rate")),
            bit_rate=_opt_number(st.get("bit_rate"), int) or _opt_number(fmt.get("bit_rate"), int),  # type: ignore[arg-type]
            duration_s=_opt_number(st.get("duration"), float) or _opt_number(fmt.get("duration"), float),
        )
    raise ValueError("no video stream found")


def probe_command(src: str, ffprobe: str = FFPROBE_BIN) -> list[str]:
    return [
        ffprobe, "-v", "error", "-hide_banner", *input_options(src),
        "-select_streams", "v:0", "-show_streams", "-show_format", "-of", "json", ffmpeg_input(src),
    ]  # fmt: skip


_DECODERS: dict[str, frozenset[str]] = {}
_DECODERS_LOCK = threading.Lock()


def ffmpeg_decoders(ffmpeg: str = FFMPEG_BIN) -> frozenset[str]:
    """Decoder names compiled into this ffmpeg (a listed hardware decoder may still lack its device).
    `ffmpeg -decoders` runs once per binary (10 s limit); a failed run is not cached, so a busy boot
    doesn't disable hardware decoding until the next engine restart."""
    with _DECODERS_LOCK:
        cached = _DECODERS.get(ffmpeg)
        if cached is not None:
            return cached
        try:
            out = subprocess.run(
                [ffmpeg, "-hide_banner", "-decoders"],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=10.0,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return frozenset()
        names = set()
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 2 and len(parts[0]) == 6 and parts[0][0] in "VAS" and parts[1] != "=":
                names.add(parts[1])
        if names:
            _DECODERS[ffmpeg] = frozenset(names)
        return frozenset(names)


def decoder_args(
    mode: str,
    codec: str,
    ffmpeg: str = FFMPEG_BIN,
    *,
    system: str | None = None,
    machine: str | None = None,
) -> tuple[list[str], str]:
    """camera.hw_decode -> (ffmpeg options before -i, label "v4l2m2m" | "videotoolbox" | "software").

    The Pi 3B+ decoder (bcm2835-codec, /dev/video10) does H.264 only, so any other codec is decoded in
    software even when v4l2m2m is asked for.
    """
    system = system or platform.system()
    machine = (machine or platform.machine()).lower()
    if mode == "off":
        return [], "software"
    if mode == "videotoolbox" or (mode == "auto" and system == "Darwin"):
        return ["-hwaccel", "videotoolbox"], "videotoolbox"
    if mode == "v4l2m2m" or (mode == "auto" and system == "Linux" and machine in ("aarch64", "arm64")):
        if codec == "h264" and "h264_v4l2m2m" in ffmpeg_decoders(ffmpeg):
            return ["-c:v", "h264_v4l2m2m"], "v4l2m2m"
        if mode == "v4l2m2m":
            log.warning("hw_decode v4l2m2m: not possible for %s with this ffmpeg; using software", codec)
    return [], "software"


def _terminate(proc: subprocess.Popen[bytes] | None) -> None:
    """terminate -> wait -> kill -> wait: the child is always reaped (no zombie)."""
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:  # stuck in the kernel: reaped by the next wait/poll
            log.warning("child process %s did not exit after SIGKILL", proc.pid)


def _read_exact(stream: IO[bytes], buf: bytearray, timeout: float | None = None) -> bool:
    """Fill `buf` completely. False at end of stream (a partial last frame is dropped). With `timeout`,
    TimeoutError if the whole frame did not arrive within that many seconds."""
    view = memoryview(buf)
    got = 0
    n = len(buf)
    deadline = None if timeout is None else time.monotonic() + timeout
    while got < n:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([stream], [], [], remaining)[0]:
                raise TimeoutError
        k = stream.readinto(view[got:])  # type: ignore[attr-defined]
        if not k:
            return False
        got += k
    return True


def _grow_pipe(stream: IO[bytes]) -> None:
    """Linux: 1 MB pipe instead of 64 KB -> ~20x fewer wake-ups per 2.7 MB frame. Best effort."""
    if not sys.platform.startswith("linux"):
        return
    try:
        import fcntl

        fcntl.fcntl(stream.fileno(), 1031, 1 << 20)  # F_SETPIPE_SZ
    except (OSError, ValueError, ImportError):
        pass


class FfmpegSource:
    """RTSP stream or video file decoded by the system `ffmpeg` program (camera.backend: ffmpeg).

    Every (re)start first runs ffprobe (codec, size, fps), then
        ffmpeg -hide_banner -nostats -loglevel warning -nostdin [-rtsp_transport tcp -timeout 5000000] [-re]
               [-c:v h264_v4l2m2m | -hwaccel videotoolbox | -threads N] -i SRC -map 0:v:0 -an -sn -dn
               -vf "select='<max_fps gate>',scale=W:H:flags=area" -fps_mode passthrough
               -pix_fmt bgr24 -f rawvideo pipe:1
    and reads exact W*H*3-byte frames from stdout.

    - Output size: ffprobe's size, shrunk to `output_width` (aspect kept, even numbers). The scale
      filter always pins it, so a camera that changes resolution mid-stream gives rescaled frames,
      never misaligned bytes.
    - Rate: a `select` filter (RateGate's rule) + `-fps_mode passthrough` instead of `fps=N`: `fps=10`
      REPEATS every frame of a 5 fps camera (the voter would count each read twice). Dropped frames
      are never scaled or converted, and never cross the pipe.
    - Live: latest frame only, wall-clock timestamps, restart with backoff (1 s .. 10 s). ffmpeg is also
      restarted when it keeps running without delivering a frame (FIRST_FRAME_TIMEOUT_S /
      STALL_TIMEOUT_S). A hardware decoder that yields no picture (error or hang) is swapped for
      software decoding (and retried later if software fails too, i.e. the camera was the problem).
    - Files: `-re` + max_fps when realtime, every frame in order otherwise; timestamps start_ts +
      index / fps like OpenCVSource; `finished` at the end of the file.
    Passwords never reach the log: ffmpeg's stderr is masked line by line. `ffmpeg`/`ffprobe` let
    tests substitute fakes. Needs ffmpeg >= 5.1 (`-fps_mode`; Bookworm ships 5.1).
    """

    def __init__(self, cfg: CameraConfig, ffmpeg: str = FFMPEG_BIN, ffprobe: str = FFPROBE_BIN) -> None:
        src = cfg.source.strip()
        if not src or src.isdigit() or src.lower() == "picamera":
            raise ValueError("the ffmpeg backend reads RTSP streams and video files only")
        self.cfg = cfg
        self._src = src
        self._shown = mask_source(src)
        self.is_file = "://" not in src
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe
        self.backoff_min = BACKOFF_MIN_S
        self.backoff_max = BACKOFF_MAX_S
        self.first_frame_timeout_s = FIRST_FRAME_TIMEOUT_S
        self.stall_timeout_s = STALL_TIMEOUT_S
        self._slot = _LatestSlot()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._proc_lock = threading.Lock()
        self._camera_ok = False
        self._hw_failed = False  # hardware decoding gave no picture: use software for now
        self._warned_codec = False
        self.restarts = 0  # ffmpeg (re)starts
        self.info: StreamInfo | None = None
        self.frame_size: tuple[int, int] | None = None  # (width, height) delivered
        self.decoder = "software"
        self.file_fps: float | None = None
        self.last_command: list[str] = []  # last ffmpeg command, password masked

    @property
    def rate_limited(self) -> bool:
        return self.cfg.max_fps > 0 and (not self.is_file or self.cfg.realtime)

    # -- FrameSource ---------------------------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        if self.is_file and not Path(self._src).is_file():
            raise FileNotFoundError(f"video file not found: {self._src}")
        self._thread = threading.Thread(target=self._run, name="anpr-ffmpeg", daemon=True)
        self._thread.start()

    def read(self, timeout: float = 1.0) -> tuple[np.ndarray, float] | None:
        got = self._slot.take(timeout)
        if got is None:
            return None
        frame, ts = got
        return frame, ts  # type: ignore[return-value]

    @property
    def finished(self) -> bool:
        return self._slot.drained

    @property
    def camera_ok(self) -> bool:
        return self._camera_ok

    def stop(self) -> None:
        self._stop.set()
        self._slot.close()
        self._kill_proc()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._kill_proc()  # in case the thread started a process while we were stopping
        self._camera_ok = False

    # -- command -------------------------------------------------------------------------------
    def build_command(self, width: int, height: int, hw_args: list[str]) -> list[str]:
        cmd = [self.ffmpeg, "-hide_banner", "-nostats", "-loglevel", "warning", "-nostdin"]
        cmd += input_options(self._src)
        if self.is_file and self.cfg.realtime:
            cmd.append("-re")  # read at the file's own speed, like a camera
        cmd += hw_args
        if not hw_args and self.cfg.decode_threads > 0:
            cmd += ["-threads", str(self.cfg.decode_threads)]  # fewer threads = fewer frame buffers
        cmd += ["-i", ffmpeg_input(self._src), "-map", "0:v:0", "-an", "-sn", "-dn"]
        filters = []
        if self.rate_limited:
            filters.append(f"select='{ffmpeg_select_expr(self.cfg.max_fps)}'")
        filters.append(f"scale={width}:{height}:flags=area")
        cmd += ["-vf", ",".join(filters), "-fps_mode", "passthrough"]
        cmd += ["-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]
        return cmd

    # -- internals -------------------------------------------------------------------------------
    def _set_proc(self, proc: subprocess.Popen[bytes] | None) -> None:
        with self._proc_lock:
            self._proc = proc

    def _kill_proc(self) -> None:
        with self._proc_lock:
            proc = self._proc
        _terminate(proc)

    def _clean(self, line: str) -> str:
        return mask_text(line, self._src)

    def _probe(self) -> StreamInfo | None:
        cmd = probe_command(self._src, self.ffprobe)
        try:
            proc = subprocess.Popen(
                cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        except OSError as e:
            log.error("camera %s: cannot run %s: %s", self._shown, self.ffprobe, e)
            return None
        self._set_proc(proc)
        if self._stop.is_set():
            self._kill_proc()
        try:
            out, err = proc.communicate(timeout=PROBE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            log.warning("camera %s: ffprobe got no answer in %.0f s", self._shown, PROBE_TIMEOUT_S)
            return None
        finally:
            self._set_proc(None)
        if self._stop.is_set():
            return None
        if proc.returncode != 0:
            msg = " | ".join(err.decode("utf-8", "replace").strip().splitlines()[-3:])
            log.warning(
                "camera %s: ffprobe failed (code %s): %s", self._shown, proc.returncode, self._clean(msg)
            )
            return None
        try:
            info = parse_probe(out)
        except (ValueError, KeyError, TypeError) as e:
            log.warning("camera %s: ffprobe: %s", self._shown, e)
            return None
        self.info = info
        return info

    def _decoder(self, codec: str) -> tuple[list[str], str]:
        if self._hw_failed:
            return [], "software"
        args, label = decoder_args(self.cfg.hw_decode, codec, self.ffmpeg)
        if (
            label == "software"
            and self.cfg.hw_decode != "off"
            and sys.platform.startswith("linux")
            and not self._warned_codec
        ):
            self._warned_codec = True
            log.warning(
                "camera %s: %s is decoded in software (the Pi 3B+ hardware decodes H.264 only, and needs "
                "h264_v4l2m2m in ffmpeg): set the camera to H.264, see docs/CAMERA_SETUP.md",
                self._shown,
                codec,
            )
        return args, label

    def _run(self) -> None:
        backoff = self.backoff_min
        try:
            while not self._stop.is_set():
                info = self._probe()
                if self._stop.is_set():
                    break
                if info is None:
                    self._camera_ok = False
                    if self.is_file:
                        log.error("video %s: cannot read it with ffprobe", self._shown)
                        break
                    log.warning("camera %s: not reachable, retry in %.0f s", self._shown, backoff)
                    self._stop.wait(backoff)
                    backoff = _next_backoff(backoff, self.backoff_max)
                    continue
                hw_args, label = self._decoder(info.codec)
                frames, code, tail = self._run_ffmpeg(info, hw_args, label)
                if self._stop.is_set():
                    break
                if frames:
                    backoff = self.backoff_min
                if frames == 0 and hw_args:
                    self._hw_failed = True
                    log.warning(
                        "camera %s: %s hardware decoding gave no picture (code %s: %s); trying software",
                        self._shown,
                        label,
                        code,
                        " | ".join(tail[-3:]),
                    )
                    continue
                if frames == 0 and self._hw_failed:
                    self._hw_failed = (
                        False  # software failed too: the source was the problem, not the decoder
                    )
                if self.is_file:
                    if code:
                        log.error(
                            "video %s: ffmpeg exited with code %s after %d frames: %s",
                            self._shown,
                            code,
                            frames,
                            " | ".join(tail),
                        )
                    else:
                        log.info("video %s: finished after %d frames", self._shown, frames)
                    break
                log.warning(
                    "camera %s: ffmpeg exited (code %s) after %d frames; restart in %.0f s. stderr tail: %s",
                    self._shown,
                    code,
                    frames,
                    backoff,
                    " | ".join(tail),
                )
                self._stop.wait(backoff)
                backoff = _next_backoff(backoff, self.backoff_max)
        finally:
            self._kill_proc()
            self._camera_ok = False
            self._slot.set_eof()

    def _run_ffmpeg(
        self, info: StreamInfo, hw_args: list[str], label: str
    ) -> tuple[int, int | None, list[str]]:
        """One ffmpeg run. Returns (frames delivered, exit code, masked stderr tail)."""
        width, height = scaled_size(info.width, info.height, self.cfg.output_width)
        src_fps = info.fps or float(self.cfg.fps)
        out_fps = min(src_fps, self.cfg.max_fps) if self.rate_limited else src_fps
        cmd = self.build_command(width, height, hw_args)
        self.last_command = [mask_text(a, self._src) for a in cmd]  # for display: no password
        self.frame_size = (width, height)
        self.decoder = label
        if self.is_file:
            self.file_fps = src_fps
        try:
            proc = subprocess.Popen(
                cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0
            )
        except OSError as e:
            return 0, None, [f"cannot start {self.ffmpeg}: {e}"]
        self._set_proc(proc)
        self.restarts += 1
        tail: collections.deque[str] = collections.deque(maxlen=STDERR_TAIL_LINES)
        drain = threading.Thread(
            target=_drain_stderr, args=(proc, tail, self._clean), name="anpr-ffmpeg-err", daemon=True
        )
        drain.start()
        log.info(
            "camera %s: %s %dx%d @ %s fps -> %dx%d BGR, %s decoding, %s",
            self._shown,
            info.codec,
            info.width,
            info.height,
            f"{info.fps:.1f}" if info.fps else "?",
            width,
            height,
            label,
            f"max {self.cfg.max_fps:g} fps" if self.rate_limited else "every frame",
        )
        frames = 0
        eof = False
        frame_bytes = width * height * 3
        assert proc.stdout is not None
        _grow_pipe(proc.stdout)
        start_ts = time.time()
        try:
            while not self._stop.is_set():
                buf = bytearray(frame_bytes)  # fresh buffer: the consumer may still hold the last one
                # Live: watchdog against a running ffmpeg that stopped delivering pictures. Files: none
                # (in lossless mode ffmpeg is paused by the consumer on purpose).
                limit = (
                    None if self.is_file else (self.stall_timeout_s if frames else self.first_frame_timeout_s)
                )
                try:
                    if not _read_exact(proc.stdout, buf, limit):
                        eof = True
                        break
                except TimeoutError:
                    log.warning(
                        "camera %s: no picture from ffmpeg for %g s (%s decoding); restarting it",
                        self._shown,
                        limit,
                        label,
                    )
                    break
                frame = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
                if self.is_file:
                    if not self.cfg.realtime and not self._slot.wait_taken():
                        break  # closed while waiting for the consumer
                    ts = start_ts + frames / out_fps
                else:
                    ts = time.time()
                frames += 1
                self._camera_ok = True
                self._slot.publish(frame, ts)
        finally:
            self._camera_ok = False
            if eof:
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=2.0)  # ended by itself: keep its real exit code
            # Close our end first: an ffmpeg blocked writing a frame ignores SIGTERM until the write
            # returns; with the pipe closed the write fails at once and ffmpeg exits.
            proc.stdout.close()
            self._kill_proc()
            self._set_proc(None)
            drain.join(timeout=1.0)
        return frames, proc.returncode, list(tail)


class RpiCamSource:
    """Pi Camera Module via `rpicam-vid ... --codec yuv420 -o -` (raw I420 frames on stdout).

    The reader thread reads exact W*H*3/2-byte frames and keeps only the latest raw buffer; conversion
    to BGR happens lazily in `read()`, so dropped frames cost no conversion. The subprocess is
    restarted with backoff (1 s .. 10 s) if it dies. `command` lets tests substitute a fake generator.

    Note: width should be a multiple of 64 (1280, 640, ...) — for other widths libcamera may pad
    each row (stride > width) and the raw stream would not be tightly packed.
    """

    def __init__(self, cfg: CameraConfig, command: list[str] | None = None) -> None:
        if cfg.width % 2 or cfg.height % 2:
            raise ValueError(f"I420 needs even width/height, got {cfg.width}x{cfg.height}")
        if cfg.width % 64:
            log.warning("rpicam width %d is not a multiple of 64; rows may be padded", cfg.width)
        self.cfg = cfg
        self.width = cfg.width
        self.height = cfg.height
        self.frame_bytes = cfg.width * cfg.height * 3 // 2
        self.command = list(command) if command is not None else self.default_command(cfg)
        self.backoff_min = BACKOFF_MIN_S
        self.backoff_max = BACKOFF_MAX_S
        self._slot = _LatestSlot()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._proc_lock = threading.Lock()
        self._camera_ok = False
        self.restarts = 0  # number of times the subprocess was (re)started

    @staticmethod
    def default_command(cfg: CameraConfig) -> list[str]:
        cmd = [
            "rpicam-vid",
            "-t", "0",
            "-n",
            "--codec", "yuv420",
            "--width", str(cfg.width),
            "--height", str(cfg.height),
            "--framerate", str(cfg.fps),
        ]  # fmt: skip
        if cfg.exposure_us is not None:
            cmd += ["--shutter", str(cfg.exposure_us)]  # analogue gain stays automatic
        cmd += ["--flush", "-o", "-"]
        return cmd

    # -- FrameSource ---------------------------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="anpr-rpicam", daemon=True)
        self._thread.start()

    def read(self, timeout: float = 1.0) -> tuple[np.ndarray, float] | None:
        got = self._slot.take(timeout)
        if got is None:
            return None
        raw, ts = got
        yuv = np.frombuffer(raw, dtype=np.uint8).reshape(self.height * 3 // 2, self.width)  # type: ignore[arg-type]
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420), ts

    @property
    def finished(self) -> bool:
        return False  # a live camera never finishes

    @property
    def camera_ok(self) -> bool:
        return self._camera_ok

    def stop(self) -> None:
        self._stop.set()
        self._slot.close()
        self._kill_proc()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._camera_ok = False

    # -- internals -------------------------------------------------------------------------------
    def _kill_proc(self) -> None:
        with self._proc_lock:
            proc = self._proc
        _terminate(proc)

    _read_exact = staticmethod(_read_exact)

    def _run(self) -> None:
        backoff = self.backoff_min
        try:
            while not self._stop.is_set():
                try:
                    proc = subprocess.Popen(
                        self.command,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        bufsize=0,
                    )
                except OSError as e:
                    self._camera_ok = False
                    log.error("rpicam: cannot start %s: %s; retry in %.0f s", self.command[0], e, backoff)
                    self._stop.wait(backoff)
                    backoff = _next_backoff(backoff, self.backoff_max)
                    continue
                with self._proc_lock:
                    self._proc = proc
                self.restarts += 1
                if self._stop.is_set():  # stop() raced with the start
                    self._kill_proc()
                    break
                stderr_tail: collections.deque[str] = collections.deque(maxlen=20)
                drain = threading.Thread(
                    target=_drain_stderr, args=(proc, stderr_tail), name="anpr-rpicam-err", daemon=True
                )
                drain.start()
                frames = 0
                assert proc.stdout is not None
                while not self._stop.is_set():
                    buf = bytearray(self.frame_bytes)  # fresh buffer: the consumer may hold the last one
                    if not self._read_exact(proc.stdout, buf):
                        break
                    frames += 1
                    backoff = self.backoff_min
                    self._camera_ok = True
                    self._slot.publish(buf, time.time())
                self._camera_ok = False
                self._kill_proc()
                proc.stdout.close()
                drain.join(timeout=1.0)
                if self._stop.is_set():
                    break
                log.warning(
                    "rpicam exited (code %s) after %d frames; restart in %.0f s. stderr tail: %s",
                    proc.returncode,
                    frames,
                    backoff,
                    " | ".join(stderr_tail),
                )
                self._stop.wait(backoff)
                backoff = _next_backoff(backoff, self.backoff_max)
        finally:
            self._kill_proc()
            self._camera_ok = False
            self._slot.set_eof()


def _drain_stderr(
    proc: subprocess.Popen[bytes], tail: collections.deque[str], clean: Callable[[str], str] | None = None
) -> None:
    """Keep the stderr pipe empty (rpicam-vid is chatty) and remember the last lines for diagnostics.
    `clean` masks each line before it is stored (ffmpeg may echo the RTSP address with its password)."""
    if proc.stderr is None:
        return
    try:
        for line in proc.stderr:
            text = line.decode("utf-8", "replace").rstrip()
            tail.append(clean(text) if clean is not None else text)
    except (OSError, ValueError):
        pass
    finally:
        proc.stderr.close()


def make_source(cfg: CameraConfig) -> FrameSource:
    """ "picamera" -> RpiCamSource (rpicam-vid). RTSP streams and video files -> FfmpegSource when
    camera.backend is "ffmpeg" and the ffmpeg + ffprobe programs exist, else OpenCVSource. Webcam
    indexes always use OpenCVSource."""
    src = cfg.source.strip()
    if src.lower() == "picamera":
        return RpiCamSource(cfg)
    if cfg.backend == "ffmpeg" and src and not src.isdigit():
        missing = [b for b in (FFMPEG_BIN, FFPROBE_BIN) if shutil.which(b) is None]
        if not missing:
            return FfmpegSource(cfg)
        log.error(
            "camera.backend is ffmpeg but %s not installed (Pi: sudo apt install ffmpeg); using OpenCV",
            " and ".join(missing),
        )
    return OpenCVSource(cfg)
