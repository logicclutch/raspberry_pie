"""Ingest ("push") mode: an ANPR camera (e.g. GVD) POSTs vehicle images to this Pi. The camera sends
SEVERAL images per vehicle; we GROUP them and VOTE across the group using the full video pipeline
(2-line / handwritten rescue, re-read, series-letter guard, multi-frame voting, HSRP), so each vehicle
gets ONE best result - no accuracy loss versus a live camera. The result is handed to the sender
(anpr.push), which POSTs it to the client API. No RTSP/video decoding, so it is the lightest mode.

Grouping handles both shapes:
  - Case A: one POST carries several images (a list)      -> that POST is one vehicle's burst.
  - Case B: one POST per image                            -> grouped by the camera's vehicle id if it
                                                             sends one, else by sender + a short time gap.

With ingest.respond_with_plate = true the reply carries the read plate ("plate"/"plates") instead of just
an ack: each POST is then one whole vehicle, read+voted inline (Case A only). The plate is sent to the
client API (anpr.push) in BOTH modes.

Run:  python -m anpr.ingest --config config.yaml      (needs `ingest:` and usually `push:` in the config)
The HTTP server is Python's standard library (no web framework), to stay small on the Pi.
"""
from __future__ import annotations

import base64
import binascii
import contextlib
import dataclasses
import hmac
import json
import logging
import queue
import re
import signal
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import cv2
import numpy as np

from anpr.config import AppConfig
from anpr.types import Box, PlateEvent

log = logging.getLogger("anpr.ingest")

_DATA_URI = re.compile(r"^data:image/[^;]+;base64,")
# A JSON field whose NAME looks like an already-cropped plate (e.g. GVD "PlateImg", "lp_crop",
# "plate_thumb"). When a POST carries BOTH a full scene image and a crop, we read the full image
# first (the primary vote) and keep the crop as an extra vote - see find_all_images().
_CROP_KEY = re.compile(r"plate|crop|thumb|lpimg|lp_img|snaplp", re.IGNORECASE)
EXIT_LICENCE = 4


def _is_crop_key(key: str) -> bool:
    return bool(key) and _CROP_KEY.search(key) is not None


def _event_to_dict(e: PlateEvent, duplicate: bool = False) -> dict:
    """The voted result as plain JSON for the HTTP response (respond_with_plate mode)."""
    return {
        "plate": e.plate,
        "confidence": round(float(e.confidence), 3),
        "kind": e.kind,  # "standard" | "bh"
        "hsrp": e.hsrp,  # "hsrp" | "non_hsrp" | "unsure" | None
        "votes": e.votes,
        # true = read correctly, but the same plate was already recorded within vote.dedupe_window_s,
        # so it was not stored or sent again (e.g. the camera re-sent the vehicle)
        "duplicate": duplicate,
    }


_UNREACHABLE = "cannot reach the server"
PUSH_OFF_MESSAGE = "sending to the client API is off (push.enabled is false or push.url is empty)"


def _api_host(url: str) -> str:
    """'http://192.168.1.72:21300/lane/anpr' -> '192.168.1.72:21300' (the address the Pi connects to)."""
    try:
        return urlsplit(url).netloc or url
    except ValueError:
        return url


def _push_error(message: str, url: str, plate: str | None, queued: bool) -> str:
    """One readable sentence for a push that did not deliver: what went wrong, the API URL, the plate."""
    what = f"plate {plate}" if plate else "the plate"
    if _UNREACHABLE in message:
        # "cannot reach the server: <reason>" or "queued: cannot reach the server: <reason>; retrying in N s"
        reason = message.split(_UNREACHABLE, 1)[1].lstrip(": ").split("; retrying", 1)[0].strip()
        err = f"API is not connecting: {url} (host {_api_host(url)})" + (f": {reason}" if reason else "")
    else:
        err = f"API at {url} (host {_api_host(url)}) did not accept {what}: {message}"
    what = what[0].upper() + what[1:]
    if queued:
        return f"{err}. {what} is saved on the Pi and is sent automatically when the API accepts it."
    return f"{err}. {what} is not sent again."


def _push_result_json(
    out: Any, enabled: bool, message: str = "nothing new to send", *, url: str = "",
    plates: list[str] | tuple[str, ...] = (), cause: str = "",
) -> dict:
    """The outcome of pushing THIS vehicle's plate to the client API, for the HTTP response
    (respond_with_plate mode). `out` is an anpr.push.PushOutcome or None (nothing to send). `url` is the
    client API address and `plates` the plate(s) this POST read; both are echoed back. `cause` (optional)
    is the sender's last error, used for the "error" text when this plate is only waiting behind it. Shape:
      {"enabled": false, "message": ...}                   -> sending to the client API is off/unconfigured
      {"enabled": true, "ok": null, "queued": false, ...}  -> nothing to send (no plate, or a duplicate)
      {"enabled": true, "ok": true, "status": 200, ...}    -> the client API accepted it ("reply" = answer)
      {"enabled": true, "ok": false, "queued": true, "error": "API is not connecting: <url> ...", ...}
                                                           -> not delivered yet (API down/erroring, or behind
                                                              earlier plates); it is stored and the background
                                                              sender retries until the API accepts it
      {"enabled": true, "ok": false, "queued": false, "error": ...}
                                                           -> the API refused this plate for good (bad data)
    """
    if not enabled:
        return {"enabled": False, "message": PUSH_OFF_MESSAGE}
    plate = ", ".join(dict.fromkeys(p for p in plates if p)) or None
    base = {"enabled": True, "url": url, "host": _api_host(url) if url else "", "plate": plate}
    if out is None:
        return {**base, "ok": None, "queued": False, "status": None, "message": message, "reply": ""}
    res = out.result
    d = {
        **base,
        "ok": out.delivered,
        "queued": out.queued,
        "status": res.status if res is not None else None,
        "message": out.message,
        "reply": res.reply if res is not None else "",
    }
    if not out.delivered:
        why = out.message
        if out.queued and cause and _UNREACHABLE in cause and _UNREACHABLE not in why:
            why = cause  # waiting behind plates the API cannot take: the real problem is the connection
        d["error"] = _push_error(why, url, plate, out.queued)
    return d


def _push_failed_json(url: str = "", plates: list[str] | tuple[str, ...] = ()) -> dict:
    """The push step itself crashed: the plate is already stored, the background sender retries it."""
    plate = ", ".join(dict.fromkeys(p for p in plates if p)) or None
    msg = "internal error while sending; the plate is stored and will be retried"
    where = f" to {url} (host {_api_host(url)})" if url else ""
    what = f"plate {plate}" if plate else "the plate"
    err = f"internal error while sending {what}{where}; it is saved on the Pi and will be retried."
    return {"enabled": True, "url": url, "host": _api_host(url) if url else "", "plate": plate,
            "ok": False, "queued": True, "status": None, "message": msg, "reply": "",
            "error": err}


# ---- finding and decoding the image(s) in the camera's JSON -----------------------------------------


def _dig(obj: Any, dotted: str) -> Any:
    """Follow a dotted path like 'data.image' into nested dicts. Returns None if any step is missing."""
    cur = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _decode_b64_image(s: Any) -> bytes | None:
    """Decode a string if it is base64 of a JPEG or PNG, else None."""
    if not isinstance(s, str) or len(s) < 500:
        return None
    s = _DATA_URI.sub("", s.strip())
    try:
        raw = base64.b64decode(s, validate=False)
    except (binascii.Error, ValueError):
        return None
    if raw[:3] == b"\xff\xd8\xff" or raw[:8] == b"\x89PNG\r\n\x1a\n":
        return raw
    return None


def find_all_images(body: Any, field: str = "", limit: int = 64) -> list[bytes]:
    """Up to `limit` vehicle images in the POST as raw bytes. With `field` set, read that field (a base64
    string or a list of them); otherwise collect every base64 JPEG/PNG anywhere in the JSON (so it works
    before we know the camera's exact field name, and handles one-POST-many-images). The limit bounds how
    much we decode from one (possibly hostile) request.

    Ordering (auto-detect): a FULL/scene image is returned BEFORE an already-cropped plate, so the pipeline
    reads the full image first (the primary vote) and treats a crop as an extra vote. This makes the GVD
    shape ("BgImg" = full vehicle, "PlateImg" = plate crop) read the full image first regardless of the
    order the camera lists the fields in. Images under the same kind keep the order they appear in."""
    out: list[bytes] = []
    if field:
        val = _dig(body, field)
        items = val if isinstance(val, list) else [val]
        for it in items:
            raw = _decode_b64_image(it)
            if raw is not None:
                out.append(raw)
                if len(out) >= limit:
                    break
        return out
    fulls: list[bytes] = []  # full/scene images (e.g. BgImg) -> primary votes
    crops: list[bytes] = []  # already-cropped plates (e.g. PlateImg) -> extra votes
    dq: deque[tuple[str, Any]] = deque([("", body)])
    seen = 0
    # Breadth-first so images are discovered in document order; bound the work for a hostile body.
    # Keep scanning for full images even once `limit` crops are found (a full image listed after many
    # crops is still the primary vote); extra crops are skipped without decoding them.
    while dq and len(fulls) < limit and seen < 100_000:
        key, cur = dq.popleft()
        seen += 1
        if isinstance(cur, dict):
            dq.extend(cur.items())
        elif isinstance(cur, list):
            dq.extend((key, v) for v in cur)
        else:
            is_crop = _is_crop_key(key)
            if is_crop and len(crops) >= limit:
                continue
            raw = _decode_b64_image(cur)
            if raw is not None:
                (crops if is_crop else fulls).append(raw)
    out = (fulls + crops)[:limit]
    return out


def find_image_bytes(body: Any, field: str = "") -> bytes | None:
    """The first vehicle image (kept for callers/tests that want a single image)."""
    imgs = find_all_images(body, field)
    return imgs[0] if imgs else None


def decode_image(raw: bytes) -> np.ndarray | None:
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img if img is not None and img.size else None


def _is_image_bytes(raw: bytes) -> bool:
    return raw[:3] == b"\xff\xd8\xff" or raw[:8] == b"\x89PNG\r\n\x1a\n"


_DISP_NAME = re.compile(rb'name="([^"]*)"')
_DISP_FILE = re.compile(rb'filename="([^"]*)"')
_VID_KEYS = ("vehicleid", "vehicle_id", "plateid", "eventid", "sessionid", "id")
_CAM_KEYS = ("camera", "cameraid", "deviceid", "device", "channel", "lane")


def _parse_multipart(body: bytes, boundary: bytes, limit: int) -> tuple[list[bytes], dict[str, str]]:
    """Pull image parts (by image magic or an image filename) and text fields out of a multipart body."""
    images: list[bytes] = []
    fields: dict[str, str] = {}
    sep = b"--" + boundary
    for part in body.split(sep):
        part = part.strip(b"\r\n")
        if not part or part == b"--":
            continue
        head, _, content = part.partition(b"\r\n\r\n")
        if not content:
            continue
        content = content.rstrip(b"\r\n")
        name_m = _DISP_NAME.search(head)
        is_file = _DISP_FILE.search(head) is not None or b"image/" in head.lower()
        if (is_file or _is_image_bytes(content)) and _is_image_bytes(content):
            if len(images) < limit:
                images.append(content)
        elif name_m and len(content) < 256:
            with contextlib.suppress(UnicodeDecodeError):
                fields[name_m.group(1).decode().lower()] = content.decode("utf-8", "replace")
    return images, fields


def _pick(fields: dict[str, str], configured: str, common: tuple[str, ...]) -> str | None:
    if configured and configured.lower() in fields:
        return fields[configured.lower()]
    for k in common:
        if k in fields and fields[k]:
            return fields[k]
    return None


def extract_request(ctype: str, body: bytes, ic) -> tuple[list[bytes], str | None, str | None]:
    """Get (images, vehicle_id, camera) from a POST in ANY common camera format: JSON+base64,
    multipart/form-data with image part(s), or a raw JPEG/PNG body."""
    ctype = ctype or ""
    low = ctype.lower()  # for case-insensitive checks; the boundary itself is case-SENSITIVE
    limit = ic.max_images_per_vehicle
    # raw image body
    if "image/" in low or _is_image_bytes(body[:8]):
        return ([body] if _is_image_bytes(body[:8]) else []), None, None
    # multipart/form-data
    if "multipart/form-data" in low:
        m = re.search(r"boundary=([^;]+)", ctype)  # original case
        if m:
            imgs, fields = _parse_multipart(body, m.group(1).strip().strip('"').encode(), limit)
            vid = _pick(fields, ic.vehicle_id_field, _VID_KEYS)
            cam = _pick(fields, ic.camera_field, _CAM_KEYS)
            return imgs, vid, cam
    # JSON (default)
    try:
        obj = json.loads(body.decode("utf-8", "replace"))
    except (json.JSONDecodeError, RecursionError, ValueError):
        return [], None, None
    vid = _dig(obj, ic.vehicle_id_field) if ic.vehicle_id_field else None
    cam = _dig(obj, ic.camera_field) if ic.camera_field else None
    return find_all_images(obj, ic.image_field, limit=limit), vid, cam


# ---- single-vehicle grouping: force a whole burst into one voting track -----------------------------


def _best_box(boxes: list[Box]) -> Box | None:
    """The clearest plate in one image (biggest, then highest score) - one vehicle per image is assumed."""
    return max(boxes, key=lambda b: (b.width * b.height, b.score)) if boxes else None


class _SingleTrack:
    """A tracker stand-in for a KNOWN single-vehicle burst: every image's best plate box is assigned to
    one track (id 1), so all reads vote together. No expiry during the burst; the Engine's flush() ends
    it. This reuses the full read+vote+HSRP pipeline without depending on IoU linking across the burst."""

    def __init__(self) -> None:
        self._tracks: dict[int, Any] = {}

    @property
    def tracks(self) -> dict[int, Any]:
        return self._tracks

    def has_active(self) -> bool:
        return bool(self._tracks)

    def update(self, boxes: list[Box], ts: float):
        best = _best_box(boxes)
        if best is None:
            return [], []
        tr = self._tracks.get(1)
        if tr is None:
            from anpr.tracker import Track

            self._tracks[1] = Track(id=1, box=best, first_seen=ts, last_seen=ts)
        else:
            tr.box = best
            tr.last_seen = ts
            tr.hits += 1
        return [(1, best)], []


def process_group(
    engine: Any,
    images: list[bytes],
    camera: str | None,
    *,
    dt: float = 0.12,
    duplicates: list[PlateEvent] | None = None,
) -> list[PlateEvent]:
    """Read and vote across one vehicle's images with the full pipeline, writing the voted event(s) to
    the store (the sender then forwards them). Returns the saved events. If `duplicates` is given, plates
    read correctly but not saved (same plate within the dedupe window) are appended to it. The images are
    decoded one at a time and released, so peak memory is one frame plus the engine's small state."""
    engine.reset()
    engine.tracker = _SingleTrack()
    engine.camera = camera or ""
    events: list[PlateEvent] = []
    ts = time.time()
    for raw in images:
        img = decode_image(raw)
        if img is None:
            continue
        events += engine.process(img, ts)
        ts += dt
    events += engine.flush()
    dups = list(getattr(engine, "duplicates", []))
    if not events and not dups:
        events, dups = _zoom_retry(engine, images, ts, dt)
    if duplicates is not None:
        duplicates.extend(dups)
    # Release the full-resolution frame the Engine keeps for the (unused here) live preview, so a
    # ~MB image is not pinned between vehicles.
    engine._last_frame = None
    engine._last_boxes = []
    return events


# ---- second look: zoomed tiles, every candidate box -------------------------------------------------


def _iou(a: Box, b: Box) -> float:
    ix = max(0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


def zoom_candidates(detector: Any, img: np.ndarray, *, frac: float = 0.6, limit: int = 4) -> list[Box]:
    """Plate boxes from the whole image plus 2x2 overlapping tiles (each `frac` of the image), mapped
    back to full-image coordinates, overlaps merged (best score kept), best score first."""
    h, w = img.shape[:2]
    th, tw = max(1, int(h * frac)), max(1, int(w * frac))
    found: list[Box] = list(detector.detect(img))
    for y in (0, h - th):
        for x in (0, w - tw):
            for b in detector.detect(img[y : y + th, x : x + tw]):
                found.append(Box(b.x1 + x, b.y1 + y, b.x2 + x, b.y2 + y, b.score))
    kept: list[Box] = []
    for b in sorted(found, key=lambda b: -b.score):
        if all(_iou(b, k) < 0.5 for k in kept):
            kept.append(b)
    return kept[:limit]


class _FixedBoxes:
    """Detector stand-in that 'finds' one given box, so the zoom retry reuses the full read pipeline."""

    def __init__(self, box: Box) -> None:
        self.box = box

    def detect(self, _frame: np.ndarray) -> list[Box]:
        return [self.box]


def _zoom_retry(engine: Any, images: list[bytes], ts: float, dt: float) -> tuple[list[PlateEvent], list]:
    """The normal pass gave nothing: look again at each image, zoomed, and read every plate box found
    (one box at a time, so a readable small plate is not hidden behind a bigger unreadable one). Stops at
    the first box that gives a plate. The engine's own detector is restored afterwards."""
    ic = getattr(getattr(engine, "cfg", None), "ingest", None)
    if ic is None or not ic.zoom_fallback or not hasattr(engine, "detector"):
        return [], []
    real = engine.detector
    camera = getattr(engine, "camera", "")
    try:
        for raw in images:
            img = decode_image(raw)
            if img is None:
                continue
            for box in zoom_candidates(real, img, limit=ic.zoom_max_candidates):
                engine.reset()
                engine.tracker = _SingleTrack()
                engine.camera = camera
                engine.detector = _FixedBoxes(box)
                events = engine.process(img, ts) + engine.flush()
                dups = list(getattr(engine, "duplicates", []))
                if events or dups:
                    log.info("zoom retry found %s at %s", [e.plate for e in events + dups],
                             (box.x1, box.y1, box.x2, box.y2))
                    return events, dups
                ts += dt
    finally:
        engine.detector = real
    return [], []


# ---- buffering the burst (reaper closes a vehicle's group; a worker reads+votes it) -----------------


@dataclasses.dataclass
class _Group:
    images: list[bytes]
    camera: str | None
    vehicle_id: str | None
    last_ts: float


_MAX_OPEN_GROUPS = 500  # bound memory if a flood of distinct vehicle_ids / IPs arrives
_MAX_QUEUED_GROUPS = 200  # bound the backlog if voting can't keep up (load shedding)


class BurstBuffer:
    """Collects a vehicle's images, then hands the closed group to a single worker that reads+votes it.
    Groups close when idle for group_timeout_s, when they hit max_images_per_vehicle, or on shutdown.
    Both the number of open groups and the processing backlog are bounded, so a hostile or overloaded
    camera cannot exhaust the Pi's memory."""

    def __init__(self, cfg: AppConfig, process_fn) -> None:
        self.cfg = cfg
        self.process_fn = process_fn
        self._groups: dict[str, _Group] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue[_Group] = queue.Queue(maxsize=_MAX_QUEUED_GROUPS)
        self._stop = threading.Event()  # tells the reaper to stop and flush
        self._draining = threading.Event()  # tells the worker it may exit once the queue is empty
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        self._threads = [
            threading.Thread(target=self.reaper, name="ingest-reaper", daemon=True),
            threading.Thread(target=self.worker, name="ingest-worker", daemon=True),
        ]
        for t in self._threads:
            t.start()

    def add(self, key: str, images: list[bytes], camera: str | None, vehicle_id: str | None) -> int:
        cap = self.cfg.ingest.max_images_per_vehicle
        now = time.time()
        with self._lock:
            g = self._groups.get(key)
            if g is None:
                # Bound the number of concurrent open groups: if full, close the oldest to make room.
                if len(self._groups) >= _MAX_OPEN_GROUPS:
                    oldest = min(self._groups, key=lambda k: self._groups[k].last_ts)
                    self._close_locked(oldest)
                g = _Group(images=[], camera=camera, vehicle_id=vehicle_id, last_ts=now)
                self._groups[key] = g
            room = cap - len(g.images)
            if room > 0:
                g.images.extend(images[:room])
            g.last_ts = now
            if camera and not g.camera:
                g.camera = camera
            n = len(g.images)
            if n >= cap:
                self._close_locked(key)
            return n

    def _close_locked(self, key: str) -> None:
        g = self._groups.pop(key, None)
        if g and g.images:
            try:
                self._queue.put_nowait(g)
            except queue.Full:  # voting is behind; shed this burst rather than run out of memory
                log.warning("ingest backlog full, dropping a burst of %d image(s)", len(g.images))

    def reaper(self) -> None:
        to = self.cfg.ingest.group_timeout_s
        while not self._stop.is_set():
            now = time.time()
            with self._lock:
                for k in [k for k, g in self._groups.items() if now - g.last_ts >= to]:
                    self._close_locked(k)
            self._stop.wait(0.3)
        with self._lock:  # flush everything still open on shutdown
            for k in list(self._groups):
                self._close_locked(k)

    def worker(self) -> None:
        while True:
            try:
                g = self._queue.get(timeout=0.3)
            except queue.Empty:
                # Only exit once draining is signalled (i.e. the reaper has flushed and the queue has
                # been fully processed) - never before, or a late-flushed burst would be lost.
                if self._draining.is_set():
                    return
                continue
            try:
                self.process_fn(g)
            except Exception:  # noqa: BLE001 - one bad group must not kill the worker
                log.exception("group processing failed")
            finally:
                self._queue.task_done()

    def stop(self) -> None:
        """Stop cleanly WITHOUT losing bursts: stop the reaper (it flushes every open group to the
        queue), wait for the worker to process the whole queue, then let the worker exit. Call this
        BEFORE the store is closed so every voted plate is written and handed to the sender."""
        self._stop.set()
        if not self._threads:
            return
        reaper, worker = self._threads
        reaper.join(timeout=10)  # its shutdown-flush enqueues every still-open group
        self._queue.join()  # the worker (still running) processes every queued burst
        self._draining.set()  # now the worker may exit on an empty queue
        worker.join(timeout=10)


# ---- HTTP receiver ---------------------------------------------------------------------------------


@dataclasses.dataclass
class _Ctx:
    cfg: AppConfig
    buffer: BurstBuffer
    # Set in respond_with_plate mode: read+vote one POST's images inline and return the events. It locks
    # the (single, shared) engine, so concurrent POSTs are voted one at a time.
    vote_fn: Any = None
    # Set in respond_with_plate mode: push the just-voted plate(s) to the client API now and return a JSON
    # dict describing the outcome (see _push_result_json). Takes the voted events.
    push_fn: Any = None
    # respond_with_plate mode: bounds how many POSTs are held in memory at once (each waits for the one
    # shared engine, holding its body). Taken BEFORE the body is read; see _SYNC_MAX_INFLIGHT.
    slots: threading.BoundedSemaphore | None = None
    received: int = 0
    images_in: int = 0


# respond_with_plate: at most this many POSTs are read+held at once (about 4 x 12 MB worst case on a 1 GB
# Pi); a POST that cannot get a slot within _SYNC_SLOT_WAIT_S is answered 503 so the camera retries.
_SYNC_MAX_INFLIGHT = 4
_SYNC_SLOT_WAIT_S = 10.0
# The whole body must arrive within this time (a LAN camera sends ~200 KB in milliseconds).
_BODY_DEADLINE_S = 20.0


def _make_handler(ctx: _Ctx) -> type[BaseHTTPRequestHandler]:
    ic = ctx.cfg.ingest
    max_body = int(ic.max_body_mb * 1024 * 1024)

    class Handler(BaseHTTPRequestHandler):
        # HTTP/1.0: one request per connection (how ANPR cameras POST one event at a time).
        protocol_version = "HTTP/1.0"
        # Drop a slow/stalled client instead of pinning a thread forever (slowloris guard).
        timeout = 15

        def log_message(self, *a):
            pass

        def _reply(self, code: int, obj: dict) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            with contextlib.suppress(BrokenPipeError, ConnectionResetError, OSError):
                self.wfile.write(data)

        def do_GET(self):
            if self.path.split("?")[0] == "/health":
                self._reply(200, {"ok": True, "received": ctx.received, "images": ctx.images_in})
            else:
                self._reply(404, {"ok": False, "error": "not found"})

        def do_POST(self):
            if self.path.split("?")[0] != ic.path:
                self._reply(404, {"ok": False, "error": "not found"})
                return
            if ic.token and not hmac.compare_digest(self.headers.get("X-Ingest-Token", ""), ic.token):
                self._reply(401, {"ok": False, "error": "bad or missing X-Ingest-Token"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0) or 0)
            except ValueError:
                self._reply(400, {"ok": False, "error": "bad Content-Length"})
                return
            if length <= 0 or length > max_body:
                self._reply(413, {"ok": False, "error": f"body must be 1..{ic.max_body_mb} MB"})
                return
            slots = ctx.slots if ic.respond_with_plate else None
            if slots is not None and not slots.acquire(timeout=_SYNC_SLOT_WAIT_S):
                self._discard(length)  # read it off the socket (not kept) so the client sees the reply
                self._reply(503, {"ok": False, "error": "busy, retry shortly"})
                return
            try:
                self._handle_body(length)
            finally:
                if slots is not None:
                    slots.release()

        def _discard(self, length: int) -> None:
            left = length
            with contextlib.suppress(OSError, ValueError):
                while left > 0:
                    chunk = self.rfile.read(min(left, 64 * 1024))
                    if not chunk:
                        break
                    left -= len(chunk)

        def _read_body(self, length: int) -> bytes | None:
            """The body, or None if it does not arrive within _BODY_DEADLINE_S in total. `timeout` only
            limits each recv, so a client trickling bytes could otherwise hold a sync slot forever."""
            deadline = time.monotonic() + _BODY_DEADLINE_S
            parts, left = [], length
            while left > 0:
                if time.monotonic() > deadline:
                    return None
                chunk = self.rfile.read1(min(left, 256 * 1024))
                if not chunk:
                    return None
                parts.append(chunk)
                left -= len(chunk)
            return b"".join(parts)

        def _handle_body(self, length: int) -> None:
            try:
                body = self._read_body(length)
            except (OSError, ValueError):
                body = None
            if body is None:
                self.close_connection = True
                self._reply(408, {"ok": False, "error": "the request body did not arrive in time"})
                return
            # Accept whatever the camera sends: JSON+base64, multipart/form-data, or a raw JPEG/PNG body.
            try:
                images, vehicle_id, camera = extract_request(
                    self.headers.get("Content-Type", ""), body, ic
                )
            except Exception:  # noqa: BLE001 - a malformed body must not kill the handler
                self._reply(400, {"ok": False, "error": "could not parse the request body"})
                return
            if not images:
                self._reply(422, {"ok": False,
                                  "error": "no image found (send JSON base64, multipart file, or JPEG body)"})
                return

            ctx.received += 1
            ctx.images_in += len(images)
            vid = str(vehicle_id) if vehicle_id not in (None, "") else None
            cam = str(camera) if camera not in (None, "") else None
            # Synchronous mode: this POST is one whole vehicle. Read+vote inline, push to the client API,
            # and return the plate(s) and the push outcome.
            if ic.respond_with_plate:
                try:
                    res = ctx.vote_fn(images, cam, vid)
                except Exception:  # noqa: BLE001 - a bad image must not kill the handler
                    log.exception("inline voting failed")
                    self._reply(500, {"ok": False, "error": "voting failed"})
                    return
                # vote_fn returns the saved events, or (saved events, duplicates).
                events, dups = res if isinstance(res, tuple) else (res, [])
                plates = [_event_to_dict(e) for e in events] + [_event_to_dict(e, True) for e in dups]
                if ctx.push_fn is None:
                    push_json = {"enabled": False}
                else:
                    try:
                        push_json = ctx.push_fn(events)
                    except Exception:  # noqa: BLE001 - the camera must always get its reply
                        log.exception("inline push failed")
                        push_json = _push_failed_json()
                    if not events and dups and push_json.get("enabled"):
                        push_json = _push_result_json(
                            None, enabled=True, url=push_json.get("url", ""),
                            plates=[e.plate for e in dups], message=(
                                "duplicate: already recorded within the last "
                                f"{ctx.cfg.vote.dedupe_window_s:.0f} s, not stored or sent again"))
                reply = {"ok": True, "images": len(images),
                         "plate": plates[0]["plate"] if plates else None,
                         "plates": plates,
                         "push": push_json}
                # The plate WAS read and saved (so "ok" stays true and HTTP stays 200: a camera that sees an
                # error status would re-send the same vehicle). A push that did not deliver is reported
                # here at the top level too, with the API URL and the plate, so it cannot be missed.
                if push_json.get("error"):
                    reply["error"] = push_json["error"]
                log.info("camera request from %s: %d image(s) -> reply: %s", self.client_address[0],
                         len(images), json.dumps(reply, separators=(",", ":")))
                self._reply(200, reply)
                return
            # Group by the camera's vehicle id when present, else by the sender (one open group per sender).
            key = f"vid:{vid}" if vid else f"ip:{self.client_address[0]}"
            n = ctx.buffer.add(key, images, cam, vid)
            self._reply(200, {"ok": True, "received": True, "images": len(images),
                              "group_size": n, "vehicle_id": vid})

    return Handler


def make_server(ctx: _Ctx) -> ThreadingHTTPServer:
    ic = ctx.cfg.ingest
    srv = ThreadingHTTPServer((ic.host, ic.port), _make_handler(ctx))
    srv.daemon_threads = True
    return srv


def _build_engine(cfg: AppConfig, store: Any, detector: Any, ocr: Any) -> Any:
    from anpr.engine import Engine

    # Ingest tuning (every image is a deliberate capture, not a video frame):
    #  - motion=always-on: run the detector on every image.
    #  - close_ratio=0 and settle_s=0: let EVERY valid read vote. The video pipeline keeps only the
    #    largest ("close-up") reads because in a video closeness == clarity; in a discrete camera burst
    #    the biggest frame can be the blurriest, so that gate could filter out the only readable frames
    #    and drop the plate. Majority voting (min_agreement) still rejects the odd wrong read. This is
    #    what keeps "no accuracy loss" true for grouped bursts.
    #  - HSRP: one marked image is enough. The video thresholds (min_marked=2, min_clean=3) assume
    #    many frames per vehicle; a camera POST gives 1-2 reads, so a plate whose hologram is plainly
    #    visible came out "unsure" and was then reported as non-HSRP.
    vote = cfg.vote.model_copy(update={"close_ratio": 0.0, "settle_s": 0.0})
    hsrp = cfg.hsrp.model_copy(update={"min_marked": 1, "min_clean": 1})
    cfg = cfg.model_copy(update={"vote": vote, "hsrp": hsrp})
    return Engine(cfg, detector, ocr, store, motion=lambda _f: True)


def run_ingest(
    cfg: AppConfig, store: Any, detector: Any, ocr: Any, *, stop=lambda: False, pusher: Any = None
) -> int:
    """Start the receiver + grouping worker and block until stop() turns true. Returns 0.
    `pusher` (respond_with_plate mode) is the anpr.push.Pusher whose background thread main() runs; the
    inline push shares it so a plate is never sent twice."""
    engine = _build_engine(cfg, store, detector, ocr)
    engine.collect_duplicates = True  # report re-sent vehicles as duplicates, not as failed reads
    sync = cfg.ingest.respond_with_plate
    # One shared engine, so every read+vote runs under this lock (the async worker is already single, but
    # in sync mode several HTTP threads could call in at once). The GVD camera POSTs one event at a time.
    engine_lock = threading.Lock()

    def vote(
        images: list[bytes], camera: str | None, vehicle_id: str | None
    ) -> tuple[list[PlateEvent], list[PlateEvent]]:
        dups: list[PlateEvent] = []
        with engine_lock:
            events = process_group(engine, images, camera, duplicates=dups)
        plates = ", ".join(sorted({e.plate for e in events})) or "(none)"
        if dups:
            plates += " (duplicate: " + ", ".join(sorted({e.plate for e in dups})) + ")"
        log.info("vehicle %s: %d image(s) -> %s", vehicle_id or camera or "?", len(images), plates)
        return events, dups

    def process(g: _Group) -> None:  # async worker: vote the closed burst (result goes to the store/push)
        vote(g.images, g.camera, g.vehicle_id)

    # Sync mode: try to deliver this vehicle's plate now (short, bounded wait) and report what happened.
    # The background sender (main() runs it on the SAME Pusher, in every mode) does all retrying with
    # backoff, so a plate that misses the inline attempt is still sent when the API is back - even if no
    # other vehicle comes - and after a restart.
    push_fn = None
    if sync:
        from anpr import push as push_mod

        if pusher is None:
            pusher = push_mod.Pusher(store, cfg.web.station_name)

        def do_push(events: list[PlateEvent]) -> dict:
            s = push_mod.load_settings(store)
            if not (s.enabled and s.url):
                return _push_result_json(None, enabled=False)
            plates = [e.plate for e in events]
            ids = [int(e.id) for e in events if e.id]
            if not ids:
                return _push_result_json(None, enabled=True, url=s.url, plates=plates)
            try:
                out = pusher.push_now(ids, s)
            except Exception:  # noqa: BLE001 - the camera must always get its reply
                log.exception("inline push failed")
                return _push_failed_json(s.url, plates)
            cause = "" if out.delivered else (push_mod.load_state(store).last_error or "")
            res = _push_result_json(out, enabled=True, url=s.url, plates=plates, cause=cause)
            if res.get("error"):
                log.warning("push: %s", res["error"])
            return res

        push_fn = do_push

    buf = BurstBuffer(cfg, process)
    slots = threading.BoundedSemaphore(_SYNC_MAX_INFLIGHT) if sync else None
    ctx = _Ctx(cfg, buf, vote_fn=vote, push_fn=push_fn, slots=slots)
    srv = make_server(ctx)
    http = threading.Thread(
        target=srv.serve_forever, kwargs={"poll_interval": 0.5}, name="ingest-http", daemon=True
    )
    mode = "read+reply per POST" if sync else "group+vote per vehicle"
    log.info("ingest server on http://%s:%d%s  (camera POSTs vehicle images here; %s)",
             cfg.ingest.host, cfg.ingest.port, cfg.ingest.path, mode)
    if not sync:  # the grouping buffer is unused in sync mode (each POST is one whole vehicle)
        buf.start()
    http.start()
    try:
        while not stop():
            time.sleep(0.2)
    finally:
        # Order matters: stop new POSTs first, THEN flush/drain the buffer (so no burst is lost). The
        # caller closes the store only after this returns, so every voted plate is written and queued.
        srv.shutdown()
        srv.server_close()
        if not sync:
            buf.stop()
    log.info("ingest stopped: requests=%d images=%d", ctx.received, ctx.images_in)
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    from anpr import licence
    from anpr.config import load_config
    from anpr.detector import make_detector
    from anpr.ocr import FastPlateOcr
    from anpr.push import Pusher, apply_push_config, start_pusher
    from anpr.storage import SqliteEventStore

    ap = argparse.ArgumentParser(prog="python -m anpr.ingest", description="ANPR push/ingest receiver")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--licence", help=f"licence file (default: {licence.LICENCE_FILE} next to the config)")
    ap.add_argument("--machine-id", action="store_true", help="print this device's ID, exit")
    ap.add_argument("--check-licence", action="store_true", help="check the licence for this device, exit")
    ap.add_argument("--activate", metavar="KEY", help="activate with a licence key (or file, or -), exit")
    args = ap.parse_args(argv)

    if args.machine_id:
        try:
            print(licence.machine_id())
        except licence.LicenceError as e:
            print(f"error: {e}", file=sys.stderr)
            return EXIT_LICENCE
        return 0
    licence_path = (
        Path(args.licence) if args.licence else Path(args.config).resolve().parent / licence.LICENCE_FILE
    )
    if args.activate is not None:
        text = args.activate
        if text == "-":
            text = sys.stdin.read()
        elif not text.strip().upper().startswith(licence.KEY_PREFIX.upper()) and Path(text).is_file():
            text = Path(text).read_text(encoding="utf-8-sig")
        try:
            lic = licence.install_licence(text, licence_path, licence.machine_id())
        except licence.LicenceError as e:
            print(f"NOT activated: {e}")
            return EXIT_LICENCE
        print(f"activated: licence {lic.licence_id} for {lic.customer}")
        return 0

    cfg = load_config(args.config)
    logging.basicConfig(level=cfg.runtime.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cv2.setNumThreads(1)
    if cfg.ingest is None or not cfg.ingest.enabled:
        log.error("ingest is not enabled in %s (add an `ingest:` section with enabled: true)", args.config)
        return 2

    if args.check_licence:
        try:
            ro = licence.ReadOnlySettings(cfg.storage.db_path)
            lic = licence.LicenceGuard.open(licence_path, ro).licence
        except licence.LicenceError as e:
            print(f"licence NOT valid: {e}")
            return EXIT_LICENCE
        print(f"licence OK: {lic.licence_id} for {lic.customer}")
        return 0

    stopping = False

    def _on_signal(signum, _frame):  # noqa: ANN001
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    store = SqliteEventStore(cfg.storage.db_path, cfg.storage.image_dir, cfg.storage.snapshot_width)
    guard = licence.wait_for_licence(licence_path, store, stop=lambda: stopping, wait=True)
    if guard is None:
        store.close()
        return 0 if stopping else EXIT_LICENCE
    if cfg.push is not None:
        apply_push_config(store, cfg.push)
    # The background sender always runs (retries with backoff, catches up after a restart). In
    # respond_with_plate mode the inline push shares this same Pusher (one lock, one cursor), so a plate
    # is never sent twice.
    pusher = Pusher(store, cfg.web.station_name)
    sender, sender_stop = start_pusher(store, cfg.web.station_name, pusher=pusher)
    try:
        detector = make_detector(cfg.detector)
        ocr = FastPlateOcr(cfg.ocr)
        return run_ingest(
            cfg, store, detector, ocr, stop=lambda: stopping or guard.expired(), pusher=pusher
        )
    finally:
        if sender_stop is not None:
            sender_stop.set()
            sender.join(timeout=15)
        store.close()


if __name__ == "__main__":
    sys.exit(main())
