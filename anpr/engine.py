"""The live pipeline (WORKFLOW.md steps 1-8): capture -> motion -> detect -> track -> crop -> OCR
-> strict check -> vote -> save. Runs as its own process (`python -m anpr.engine`)."""

from __future__ import annotations

import contextlib
import logging
import math
import os
import re
import resource
import signal
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from anpr.config import AppConfig
from anpr.tracker import IouTracker, Track
from anpr.types import (
    Box,
    EngineStatus,
    EventStore,
    FrameSource,
    OcrResult,
    PlateDetector,
    PlateEvent,
    PlateOcr,
    ValidPlate,
)
from anpr.validator import STATE_CODES, PlateValidator
from anpr.vote import Decision, PlateVoter

log = logging.getLogger("anpr.engine")

# 2-line plates: a read of just the bottom line (series + number) and of just the top line
# (state + district). Only reads of these shapes trigger / accept the top-line rescue.
_BOTTOM_LINE = re.compile(r"[A-Z0-9]{1,3}[0-9O]{3,4}")
_TOP_LINE = re.compile(r"[A-Z]{2}[0-9]{1,2}")  # state + district
# The top line may also carry series letter(s): "HR38A" over "A3075" = HR38AA3075. Accepted only
# when the track's whole-plate reads agree on that many series letters (see _letters_ok): an extra
# letter can also be the bottom line bleeding into the top crop ("TN36B" + "BC8199").
_TOP_LINE_SERIES = re.compile(r"[A-Z]{2}[0-9]{2}[A-Z]{1,2}")
# Start of a whole-plate read: state + 2-digit district ("TN36"). Seen while the vehicle is far
# away and the box still covers both lines of a 2-line plate; remembered per track (see _note_head).
_HEAD = re.compile(r"[A-Z]{2}[0-9]{2}")
HEAD_MIN_SEEN = 2  # a remembered head must come from at least this many reads before it is used


def _leading_letters(text: str) -> int:
    """How many letters `text` starts with (the series letters of "BC8199" -> 2)."""
    n = 0
    while n < len(text) and text[n].isalpha():
        n += 1
    return n


PURGE_EVERY_S = 6 * 3600.0
HSRP_MAX_LOOKS = 12  # HSRP check: looks kept per vehicle (~4 ms each on a Mac); enough to decide
EXIT_MEMORY = 3  # systemd restarts us
EXIT_LICENCE = 4  # no valid licence for this device (or it ended while running)
PREVIEW_TRACKED = (32, 176, 255)  # BGR amber (--signal): plate being read
PREVIEW_CONFIRMED = (140, 211, 55)  # BGR green (--ok): plate confirmed


def current_rss_mb() -> float:
    """Resident memory now: /proc on Linux/Pi, `ps` on macOS (dev only). Peak RSS as a last resort.

    Current, not peak: the memory limit must not trip forever on a spike that has since been freed.
    """
    try:
        with open("/proc/self/status", encoding="ascii") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True, timeout=2.0
            ).stdout
            return int(out.strip()) / 1024.0
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024.0 * 1024.0) if sys.platform == "darwin" else peak / 1024.0


@dataclass(slots=True)
class Stats:
    frames: int = 0
    detected_frames: int = 0
    ocr_reads: int = 0
    valid_reads: int = 0
    events: int = 0
    duplicates: int = 0
    rereads: int = 0  # second-chance reads from a wider crop
    reread_fixes: int = 0  # ... that then passed the format check
    top_line_reads: int = 0  # 2-line plates: extra reads of the line above the box
    top_line_fixes: int = 0  # ... that then passed the format check
    head_fixes: int = 0  # 2-line plates: bottom line joined with the track's earlier state+district


class Engine:
    def __init__(
        self,
        cfg: AppConfig,
        detector: PlateDetector,
        ocr: PlateOcr,
        store: EventStore,
        motion: Callable[[np.ndarray], bool] | None = None,
        prepare: Callable[..., np.ndarray | None] | None = None,
        validator: PlateValidator | None = None,
        reread: Callable[..., np.ndarray | None] | None = None,
        hsrp_look: Callable[[np.ndarray, Box], str | None] | None = None,
        camera: str = "",
    ) -> None:
        self.cfg = cfg
        self.detector = detector
        self.ocr = ocr
        self.store = store
        # Multi-camera: the label saved on each plate (and sent in the JSON). "" = single camera.
        self.camera = camera
        self._own_motion = motion is None
        if motion is None:
            from anpr.motion import MotionGate

            motion = MotionGate(cfg.motion).update
        if prepare is None:
            from anpr.preprocess import prepare_plate, reread_plate

            prepare = prepare_plate
            reread = reread if reread is not None else reread_plate
        self.motion = motion
        self.prepare = prepare
        # Second-chance crop for a read that failed the format check (None = no retry). Only on by
        # default with the real prepare_plate, so tests that inject their own crops are unchanged.
        self.reread = reread
        self.validator = validator or PlateValidator(cfg.validation)
        self.tracker = IouTracker(cfg.tracking)
        self.voter = PlateVoter(cfg.vote)
        self.stats = Stats()
        self._last_emitted: dict[str, float] = {}
        # Plates confirmed but NOT saved because they repeat one within the dedupe window (since the
        # last reset). Ingest reports these back so a re-sent vehicle doesn't look like a failed read.
        # Off by default (the video loop never resets, so the list would only grow); ingest switches it on.
        self.collect_duplicates = False
        self.duplicates: list[PlateEvent] = []
        # For the dashboard live preview: the last frame seen, the tracked boxes on it and the plate
        # each confirmed track was read as. Only references are kept; nothing is copied per frame.
        self._last_frame: np.ndarray | None = None
        self._last_boxes: list[tuple[int, Box]] = []
        self._confirmed: dict[int, str] = {}
        # Per track: state+district heads read so far -> [reads, confidence sum, best confs, crop].
        self._heads: dict[int, dict[str, list]] = {}
        # HSRP check: per track, (plate width, look "hsrp" / "non_hsrp" / None) per judged read.
        if hsrp_look is None and cfg.hsrp.enabled:
            from functools import partial

            from anpr.hsrp import look

            hsrp_look = partial(look, two_line=cfg.hsrp.two_line)
        self.hsrp_look = hsrp_look
        self._hsrp: dict[int, list[tuple[int, str | None]]] = {}

    def reset(self) -> None:
        """New stream: forget tracks, pending votes and the motion reference (a different scene).
        The duplicate window is kept, so the same car on a replayed video isn't saved twice."""
        self.tracker = IouTracker(self.cfg.tracking)
        self.voter = PlateVoter(self.cfg.vote)
        self._last_frame = None
        self._last_boxes = []
        self._confirmed = {}
        self._heads = {}
        self._hsrp = {}
        self.duplicates = []
        if self._own_motion:
            from anpr.motion import MotionGate

            self.motion = MotionGate(self.cfg.motion).update

    def render_preview(self, width: int) -> np.ndarray | None:
        """The last processed frame, shrunk to `width`, with the last frame's tracked plates drawn
        on it: amber = still reading, green + text = confirmed. None before the first frame."""
        frame = self._last_frame
        if frame is None:
            return None
        h, w = frame.shape[:2]
        if w > width:
            scale = width / w
            # Mild shrink (e.g. the 704-wide toll camera -> 640): bilinear looks the same and costs
            # a quarter of INTER_AREA (0.7 vs 2.6 ms on the Mac). Big shrinks need INTER_AREA.
            interp = cv2.INTER_LINEAR if scale >= 0.5 else cv2.INTER_AREA
            img = cv2.resize(frame, (width, max(1, round(h * scale))), interpolation=interp)
        else:
            scale = 1.0
            img = frame.copy()  # never draw on the camera's frame
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        ih, iw = img.shape[:2]
        for tid, b in self._last_boxes:
            x1, y1 = round(b.x1 * scale), round(b.y1 * scale)
            x2, y2 = max(x1 + 1, round(b.x2 * scale) - 1), max(y1 + 1, round(b.y2 * scale) - 1)
            plate = self._confirmed.get(tid)
            color = PREVIEW_CONFIRMED if plate else PREVIEW_TRACKED
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            if plate:
                (tw, th), base = cv2.getTextSize(plate, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                ty = y1 - 4 if y1 - th - base - 4 >= 0 else min(ih - base - 1, y2 + th + 4)
                tx = min(max(0, x1), max(0, iw - tw - 6))
                cv2.rectangle(img, (tx, ty - th - 3), (tx + tw + 6, ty + base), color, -1)
                cv2.putText(
                    img, plate, (tx + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
                )
        return img

    def _retry_read(
        self, frame: np.ndarray, box: Box, tid: int, text: str, crop: np.ndarray
    ) -> tuple[ValidPlate | None, np.ndarray]:
        """One more read from a wider crop after a read failed the format check. Returns the valid
        plate (or None) and the crop that goes with it. The voter still needs several agreeing
        frames, so a single lucky second read cannot confirm a plate on its own."""
        wide = self.reread(frame, box, self.cfg.crop)
        if wide is None:
            return None, crop
        self.stats.rereads += 1
        second = self.ocr.read(wide)
        valid = self.validator.validate(second) if second is not None else None
        if valid is None:
            return None, crop
        self.stats.reread_fixes += 1
        log.debug("re-read track=%d %r -> %r", tid, text, valid.text)
        return valid, wide

    def _top_line_read(
        self, frame: np.ndarray, box: Box, tid: int, bottom: OcrResult
    ) -> tuple[ValidPlate | None, np.ndarray | None]:
        """The read looks like only the bottom line of a 2-line plate: read the line above the box
        and validate the two joined. Returns (valid plate, whole-plate crop) or (None, None)."""
        if not _BOTTOM_LINE.fullmatch(bottom.text):
            return None, None
        from anpr.preprocess import in_box_top_crops, top_line_crops

        # The top line is either inside a box that covers the whole 2-line plate (the reader then
        # returns just the bottom line: "A3075" for HR38AA3075) or above a box on the bottom line.
        # A box shaped like a whole 2-line plate has nothing of the plate above it: one read only.
        crops = in_box_top_crops(frame, box, self.cfg.crop)
        where = "in box"
        if crops is None:
            crops, where = top_line_crops(frame, box, self.cfg.crop), "above"
        if crops is not None:
            top_crop, whole = crops
            top = self.ocr.read(top_crop)
            self.stats.top_line_reads += 1
            series = top is not None and _TOP_LINE_SERIES.fullmatch(top.text) is not None
            if top is not None and (series or _TOP_LINE.fullmatch(top.text)):
                joined = OcrResult(top.text + bottom.text, top.char_confs + bottom.char_confs)
                valid = self.validator.validate(joined)
                if valid is not None and not self._letters_ok(tid, valid.text, required=series):
                    log.debug(
                        "2-line read track=%d %r + %r: series letters disagree", tid, top.text, bottom.text
                    )
                    valid = None
            else:
                valid = None
            if valid is not None:
                self.stats.top_line_fixes += 1
                log.debug(
                    "2-line read track=%d %r (%s) + %r -> %r", tid, top.text, where, bottom.text, valid.text
                )
                return valid, whole
        return self._head_read(tid, bottom)

    def _letters_ok(self, tid: int, plate: str, required: bool) -> bool:
        """Does a joined 2-line plate have as many series letters after its state+district as this
        track's whole-plate reads show ("HR38AA307" -> 2)? A tie between counts is not agreement.
        With no whole-plate read of that head showing a series letter: `required` decides."""
        seen = self._heads.get(tid, {}).get(plate[:4])
        if seen is None or not seen[4]:
            return not required
        top = seen[4].most_common(2)
        if len(top) > 1 and top[0][1] == top[1][1]:
            return False
        return top[0][0] == _leading_letters(plate[4:])

    def _note_head(self, tid: int, result: OcrResult, crop: np.ndarray) -> None:
        """Remember the state+district a whole-plate read of this track starts with ("TN36" from
        "TN36BC819"), if those 4 characters are confident and a real state code."""
        head = result.text[:4]
        if len(result.text) < 6 or not _HEAD.fullmatch(head) or head[:2] not in STATE_CODES:
            return
        confs = result.char_confs[:4]
        if min(confs) < self.cfg.validation.min_char_conf:
            return
        seen = self._heads.setdefault(tid, {}).setdefault(head, [0, 0.0, confs, crop, Counter()])
        seen[0] += 1
        seen[1] += sum(confs) / 4
        if min(confs) > min(seen[2]):
            seen[2], seen[3] = confs, crop
        # A read with no series letter after the head is the reader losing them far away
        # ("TN36981" for TN36BC8199), not a plate without letters: it says nothing about the count.
        letters = _leading_letters(result.text[4:])
        if letters:
            seen[4][letters] += 1

    def _head_read(self, tid: int, bottom: OcrResult) -> tuple[ValidPlate | None, np.ndarray | None]:
        """2-line plate close up, box on the bottom line only: join it with the state+district this
        same track showed while it was farther away (the most-read head, seen >= HEAD_MIN_SEEN times)."""
        heads = self._heads.get(tid)
        if not heads:
            return None, None
        head, (n, _, confs, crop, _letters) = max(heads.items(), key=lambda kv: (kv[1][0], kv[1][1]))
        if n < HEAD_MIN_SEEN or bottom.text.startswith(head[:2]):
            return None, None  # "TN8199" is a cut-short whole-plate read, not a bottom line
        valid = self.validator.validate(OcrResult(head + bottom.text, confs + bottom.char_confs))
        # Whole-plate reads with more series letters than the join has mean the top line holds a
        # letter too ("HR38A" over "A3075"): "HR38" + "A3075" would save a wrong plate (HR38A3075).
        if valid is None or not self._letters_ok(tid, valid.text, required=True):
            return None, None
        self.stats.head_fixes += 1
        log.debug("2-line read track=%d head %r (x%d) + %r -> %r", tid, head, n, bottom.text, valid.text)
        return valid, crop

    def _snapshot(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        sw = self.cfg.storage.snapshot_width
        if w <= sw:
            return frame.copy()
        return cv2.resize(frame, (sw, max(1, round(h * sw / w))), interpolation=cv2.INTER_AREA)

    def _is_duplicate(self, plate: str, ts: float) -> bool:
        window = self.cfg.vote.dedupe_window_s
        last = self._last_emitted.get(plate)
        if last is not None and ts - last < window:
            return True
        return self.store.recent_plate_seen(plate, window, ts)

    def _save(
        self, tid: int, decision: Decision, ts: float, first_seen: float, last_seen: float, rule: str
    ) -> PlateEvent | None:
        """Store a confirmed plate (unless it is a duplicate within the dedupe window)."""
        self._confirmed[tid] = decision.plate  # green in the live preview, saved or duplicate
        if self._is_duplicate(decision.plate, ts):
            self.stats.duplicates += 1
            log.info("duplicate %s within %.0fs, not saved", decision.plate, self.cfg.vote.dedupe_window_s)
            if self.collect_duplicates:
                self.duplicates.append(
                    PlateEvent(
                        plate=decision.plate,
                        kind=decision.kind,
                        confidence=decision.confidence,
                        votes=decision.votes,
                        track_id=tid,
                        first_seen=first_seen,
                        last_seen=last_seen,
                        hsrp=self._hsrp_result(tid),
                        camera=self.camera or None,
                    )
                )
            return None
        best_crop, best_snap = self.voter.evidence(tid, decision.plate)
        event = self.store.add_event(
            PlateEvent(
                plate=decision.plate,
                kind=decision.kind,
                confidence=decision.confidence,
                votes=decision.votes,
                track_id=tid,
                first_seen=first_seen,
                last_seen=last_seen,
                hsrp=self._hsrp_result(tid),
                camera=self.camera or None,
            ),
            crop=best_crop,
            snapshot=best_snap,
        )
        self._last_emitted[decision.plate] = ts
        self.stats.events += 1
        log.info(
            "PLATE %s conf=%.3f votes=%d agree=%.2f track=%d hsrp=%s (%s)",
            event.plate,
            decision.confidence,
            decision.votes,
            decision.agreement,
            tid,
            event.hsrp,
            rule,
        )
        return event

    def _hsrp_observe(self, tid: int, frame: np.ndarray, box: Box) -> None:
        """Judge this frame for the HSRP check. Keeps the looks from the HSRP_MAX_LOOKS biggest
        (closest) views of the plate: far-away frames are the blurriest, and this bounds the cost."""
        if self.hsrp_look is None:
            return
        looks = self._hsrp.setdefault(tid, [])
        if len(looks) >= HSRP_MAX_LOOKS:
            smallest = min(range(len(looks)), key=lambda i: looks[i][0])
            if box.width <= looks[smallest][0]:
                return
            del looks[smallest]
        try:
            look = self.hsrp_look(frame, box)
        except Exception:  # noqa: BLE001 - an extra check must never cost the plate read
            log.debug("hsrp check failed track=%d", tid, exc_info=True)
            look = None
        looks.append((box.width, look))

    def _hsrp_result(self, tid: int) -> str | None:
        """HSRP / non-HSRP / not sure from the track's looks so far; None when the check is off."""
        if self.hsrp_look is None:
            return None
        from anpr.hsrp import decide

        c = self.cfg.hsrp
        looks = [look for _, look in self._hsrp.get(tid, [])]
        result = decide(looks, min_marked=c.min_marked, min_clean=c.min_clean)
        if result == "unsure" and c.unsure_as_non_hsrp:
            return "non_hsrp"
        return result

    def _end_track(self, tr: Track, ts: float) -> PlateEvent | None:
        """A vehicle left: last chance with the stricter end-of-track rule, then forget the track."""
        self._confirmed.pop(tr.id, None)
        self._heads.pop(tr.id, None)
        event = None
        late = self.voter.finish(tr.id)
        if late is not None:
            event = self._save(tr.id, late, ts, tr.first_seen, tr.last_seen, "end of track")
        self._hsrp.pop(tr.id, None)
        reads = self.voter.drop(tr.id)
        if reads and log.isEnabledFor(logging.DEBUG):
            log.debug("track %d ended: %s", tr.id, [(r.text, round(r.confidence, 2)) for r in reads])
        return event

    def flush(self) -> list[PlateEvent]:
        """The stream ended or is being replaced: end every open track now, so the last vehicle of a
        video (or the one in view at a switch) still gets the end-of-track rule instead of being lost."""
        tracks = list(self.tracker.tracks.values())
        self.tracker.tracks.clear()
        events = []
        for tr in tracks:
            event = self._end_track(tr, tr.last_seen)
            if event is not None:
                events.append(event)
        return events

    def process(self, frame: np.ndarray, ts: float) -> list[PlateEvent]:
        """Run one frame through steps 2-8. Returns the events saved for this frame."""
        self.stats.frames += 1
        self._last_frame, self._last_boxes = frame, []
        run_detector = self.motion(frame) or self.tracker.has_active()
        boxes = self.detector.detect(frame) if run_detector else []
        if run_detector:
            self.stats.detected_frames += 1
        assigned, expired = self.tracker.update(boxes, ts)
        self._last_boxes = assigned

        events: list[PlateEvent] = []
        for tr in expired:
            event = self._end_track(tr, ts)
            if event is not None:
                events.append(event)

        for tid, b in assigned:
            self.voter.observe(tid, b.width, ts)

        # OCR only tracks that still need an answer, biggest (closest) plates first.
        todo = [(tid, b) for tid, b in assigned if not self.voter.is_reported(tid)]
        todo.sort(key=lambda p: -p[1].width)
        snapshot: np.ndarray | None = None
        for tid, box in todo[: self.cfg.ocr.max_plates_per_frame]:
            crop = self.prepare(frame, box, self.cfg.crop)
            if crop is None:
                continue
            result = self.ocr.read(crop)
            self.stats.ocr_reads += 1
            if result is None:
                continue
            self._note_head(tid, result, crop)
            valid = self.validator.validate(result)
            if valid is None:
                valid, whole = self._top_line_read(frame, box, tid, result)
                if whole is not None:
                    crop = whole
            if valid is None and self.reread is not None:
                valid, crop = self._retry_read(frame, box, tid, result.text, crop)
            if valid is None:
                log.debug("rejected read track=%d text=%r", tid, result.text)
                continue
            self.stats.valid_reads += 1
            self._hsrp_observe(tid, frame, box)
            if snapshot is None:
                snapshot = self._snapshot(frame)
            decision = self.voter.add(tid, valid, crop, snapshot, width=box.width)
            if decision is None:
                continue
            event = self._save(tid, decision, ts, self.tracker.tracks[tid].first_seen, ts, "vote")
            if event is not None:
                events.append(event)

        # Keep the dedupe map small.
        cutoff = ts - self.cfg.vote.dedupe_window_s
        if len(self._last_emitted) > 256:
            self._last_emitted = {p: t for p, t in self._last_emitted.items() if t >= cutoff}
        return events


@dataclass(slots=True)
class StreamControl:
    """Managed mode (the service): the dashboard can switch streams while the engine runs, and a
    finished video leaves the engine waiting for the next stream instead of exiting."""

    open: Callable[[str], FrameSource]  # source address -> not yet started FrameSource
    default: str  # config camera.source; an empty dashboard request means "back to this"
    applied_rev: int = 0  # dashboard request already applied (or deliberately ignored)
    check_every_s: float = 1.0
    playlist: tuple[str, ...] = ()  # config camera.playlist: videos played one after another
    loop_playlist: bool = False  # after the last video, start the playlist again
    playlist_pos: int | None = None  # playlist item playing now; None = not playing the playlist

    def target(self, requested: str) -> str:
        """Address to open for a dashboard request. "" = back to the config default: the start of
        the playlist when there is one. Any other address leaves the playlist."""
        if requested or not self.playlist:
            self.playlist_pos = None
            return requested or self.default
        self.playlist_pos = 0
        return self.playlist[0]

    def next_video(self) -> str | None:
        """The playlist video to start after the current one ended, or None (wait for the dashboard)."""
        if self.playlist_pos is None:
            return None
        pos = self.playlist_pos + 1
        if pos >= len(self.playlist):
            if not self.loop_playlist:
                return None
            pos = 0
        self.playlist_pos = pos
        return self.playlist[pos]


def run(
    cfg: AppConfig,
    source: FrameSource | None,
    engine: Engine,
    stop: Callable[[], bool] = lambda: False,
    max_frames: int | None = None,
    *,
    source_name: str = "",
    control: StreamControl | None = None,
) -> int:
    """Main loop. Returns a process exit code. Without `control` the loop ends when the source
    finishes (evaluation, tests); with it the engine keeps running until `stop()`."""
    from anpr.sources import mask_source

    store = engine.store
    current: FrameSource | None = None
    name = mask_source(source_name)
    source_state = "running"
    last_status = 0.0
    last_purge = 0.0
    last_check = time.monotonic()
    window_start, window_frames = time.monotonic(), 0
    fps = 0.0
    last_error: str | None = None
    code = 0
    preview_every = cfg.runtime.preview_interval_s
    last_preview = -math.inf  # monotonic time of the last preview attempt
    preview_ts: float | None = None  # wall time of the last preview written
    preview_failing = False

    def start(src: FrameSource | None) -> None:
        nonlocal current, source_state, last_error
        current = src
        if src is None:
            return
        try:
            src.start()
            source_state = "running"
        except Exception as e:
            if control is None:
                raise
            current = None
            source_state = "error"
            last_error = f"cannot open {name}: {e}"
            log.error("%s", last_error)

    def switch(address: str) -> None:
        nonlocal current, name, last_error, source_state, preview_ts, last_preview
        assert control is not None
        if current is not None:
            current.stop()
            current = None
        engine.flush()  # the vehicle in view on the old stream still gets its end-of-track check
        name = mask_source(address)
        last_error = None
        engine.reset()
        # The old stream's picture must never pass for the new one: drop it, and show the new
        # stream's first frame right away instead of up to one interval later.
        store.clear_preview()
        preview_ts, last_preview = None, -math.inf
        log.info("switching stream to %s", name)
        try:
            src = control.open(address)
        except Exception as e:  # bad address from an old database row etc.
            source_state = "error"
            last_error = f"cannot open {name}: {e}"
            log.error("%s", last_error)
            return
        start(src)

    def write_status(fps_now: float, err: str | None) -> float:
        rss = current_rss_mb()
        store.write_status(
            EngineStatus(
                ts=time.time(),
                fps=round(fps_now, 2),
                camera_ok=current is not None and bool(getattr(current, "camera_ok", True)),
                frames=engine.stats.frames,
                events=engine.stats.events,
                rss_mb=round(rss, 1),
                last_error=err,
                source=name or None,
                source_state=source_state,
                source_rev=control.applied_rev if control is not None else 0,
                preview_ts=preview_ts,
            )
        )
        return rss

    def write_preview() -> None:
        """Dashboard live view. Only called right after a new frame from an open stream, so the
        preview's age tells the web when the picture stopped. Failures never stop the engine."""
        nonlocal preview_ts, preview_failing
        try:
            img = engine.render_preview(cfg.runtime.preview_width)
            if img is None:
                return
            preview_ts = store.write_preview(img, cfg.runtime.preview_quality)
        except Exception as e:  # noqa: BLE001 - disk full, permissions, ... keep reading plates
            if not preview_failing:
                log.warning("cannot write the live preview (will keep trying quietly): %s", e)
            preview_failing = True
            return
        if preview_failing:
            log.info("live preview written again")
            preview_failing = False

    store.clear_preview()  # a picture left by an earlier run is not this stream's
    start(source)
    try:
        while not stop():
            now = time.time()
            if now - last_purge > PURGE_EVERY_S:
                n = store.purge_older_than(cfg.storage.retention_days, now)
                if n:
                    log.info(
                        "retention: deleted %d events older than %.0f days", n, cfg.storage.retention_days
                    )
                last_purge = now

            if control is not None and time.monotonic() - last_check >= control.check_every_s:
                last_check = time.monotonic()
                req = store.read_source_request()
                if req is not None and req[0] != control.applied_rev:
                    control.applied_rev = req[0]
                    switch(control.target(req[1]))

            if current is None:
                time.sleep(0.2)  # waiting for the dashboard to pick a stream
            else:
                item = current.read(timeout=1.0)
                if item is not None:
                    frame, ts = item
                    try:
                        engine.process(frame, ts)
                    except Exception as e:  # keep running; one bad frame must not kill the service
                        last_error = f"{type(e).__name__}: {e}"
                        log.exception("frame processing failed")
                    window_frames += 1
                    if preview_every > 0 and time.monotonic() - last_preview >= preview_every:
                        last_preview = time.monotonic()
                        write_preview()
                    if max_frames is not None and engine.stats.frames >= max_frames:
                        break
                elif current.finished:
                    log.info("source finished")
                    engine.flush()  # last vehicle of the video
                    if control is None:
                        break
                    nxt = control.next_video()
                    if nxt is not None:
                        log.info("playlist: next video")
                        switch(nxt)
                    else:
                        current.stop()
                        current = None
                        source_state = "ended"
                        log.info("waiting for a new stream from the dashboard")

            mono = time.monotonic()
            if mono - window_start >= cfg.runtime.status_interval_s:
                fps = window_frames / (mono - window_start)
                window_start, window_frames = mono, 0
            if time.time() - last_status >= cfg.runtime.status_interval_s:
                rss = write_status(fps, last_error)
                last_status = time.time()
                if rss > cfg.runtime.max_rss_mb:
                    log.error(
                        "memory %.0f MB > limit %.0f MB, exiting for restart", rss, cfg.runtime.max_rss_mb
                    )
                    code = EXIT_MEMORY
                    break
    finally:
        if current is not None:
            current.stop()
        write_status(0.0, last_error)  # final state, so the dashboard doesn't show stale numbers
    log.info(
        "stopped: frames=%d detected=%d ocr=%d valid=%d rereads=%d/%d events=%d duplicates=%d",
        engine.stats.frames,
        engine.stats.detected_frames,
        engine.stats.ocr_reads,
        engine.stats.valid_reads,
        engine.stats.reread_fixes,
        engine.stats.rereads,
        engine.stats.events,
        engine.stats.duplicates,
    )
    return code


def _sleep_or_stop(stop: Callable[[], bool], seconds: float) -> bool:
    """Sleep up to `seconds`, waking early if stop() turns true. Returns True if we should stop."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if stop():
            return True
        time.sleep(min(0.2, max(0.0, end - time.monotonic())))
    return stop()


def run_cameras(
    cfg: AppConfig,
    feeds: list,
    store: EventStore,
    *,
    stop: Callable[[], bool] = lambda: False,
    max_frames: int | None = None,
    exit_at_end: bool = False,
    make_engine: Callable[[AppConfig, str], Engine] | None = None,
) -> int:
    """Run several cameras in ONE process, sharing `store` (and the loaded libraries, to fit the Pi's
    1 GB). Each camera gets its own identical pipeline (its own detector, reader, tracker and voter), so
    a plate seen by one camera is processed exactly as it would be on a single-camera device - there is
    no per-frame accuracy loss from sharing. The only shared, finite resource is CPU time: when both
    cameras are busy in the same instant they take turns, which is why this is meant for quiet/medium
    gates. Each saved plate carries its camera's name, which goes into the JSON sent to the API.

    Returns EXIT_MEMORY if the process grows past runtime.max_rss_mb (systemd then restarts it), else 0.
    """
    import threading

    from anpr.camera import make_source
    from anpr.detector import make_detector
    from anpr.ocr import FastPlateOcr
    from anpr.sources import mask_source

    pipelines: list[tuple[str, Engine, AppConfig]] = []
    for i, feed in enumerate(feeds):
        name = (getattr(feed, "name", "") or f"cam{i + 1}").strip()
        cam = cfg.camera.model_copy(update={"source": feed.source, "name": name})
        cfg_i = cfg.model_copy(update={"camera": cam})
        if make_engine is not None:
            eng = make_engine(cfg_i, name)
        else:
            eng = Engine(cfg_i, make_detector(cfg.detector), FastPlateOcr(cfg.ocr), store, camera=name)
        pipelines.append((name, eng, cfg_i))
    names = [n for n, _, _ in pipelines]
    if len(set(names)) != len(names):
        raise ValueError(f"camera names must be unique, got {names}")

    code = 0
    done = threading.Event()  # a camera hit max_frames / the memory limit: stop them all

    def should_stop() -> bool:
        return stop() or done.is_set()

    def worker(name: str, eng: Engine, cfg_i: AppConfig) -> None:
        src_name = mask_source(cfg_i.camera.source)
        while not should_stop():
            try:
                src = make_source(cfg_i.camera)
                src.start()
            except Exception as e:  # noqa: BLE001 - bad address / camera down: wait and retry
                log.error("camera %s: cannot open %s: %s", name, src_name, e)
                if _sleep_or_stop(should_stop, 3.0) or exit_at_end:
                    return
                continue
            log.info("camera %s started: source=%s", name, src_name)
            try:
                while not should_stop():
                    item = src.read(timeout=1.0)
                    if item is not None:
                        frame, ts = item
                        try:
                            eng.process(frame, ts)
                        except Exception:  # noqa: BLE001 - one bad frame must not kill the camera
                            log.exception("camera %s: frame processing failed", name)
                        if max_frames is not None and eng.stats.frames >= max_frames:
                            done.set()
                            break
                    elif src.finished:
                        eng.flush()  # the last vehicle of the file
                        break
            finally:
                with contextlib.suppress(Exception):
                    src.stop()
                eng.flush()
            if exit_at_end or _sleep_or_stop(should_stop, 1.0):
                return  # file mode: stop; live mode: reconnect after a short pause

    threads = [
        threading.Thread(target=worker, args=(n, e, c), name=f"cam-{n}", daemon=True)
        for n, e, c in pipelines
    ]
    log.info("multi-camera: starting %d cameras: %s", len(threads), ", ".join(names))
    for t in threads:
        t.start()

    last_status = 0.0
    last_purge = 0.0
    try:
        while not should_stop() and any(t.is_alive() for t in threads):
            now = time.time()
            if now - last_purge > PURGE_EVERY_S:
                store.purge_older_than(cfg.storage.retention_days, now)
                last_purge = now
            if now - last_status >= cfg.runtime.status_interval_s:
                rss = current_rss_mb()
                frames = sum(e.stats.frames for _, e, _ in pipelines)
                events = sum(e.stats.events for _, e, _ in pipelines)
                store.write_status(
                    EngineStatus(
                        ts=now,
                        fps=0.0,
                        camera_ok=any(t.is_alive() for t in threads),
                        frames=frames,
                        events=events,
                        rss_mb=round(rss, 1),
                        source=", ".join(names),
                        source_state="running",
                    )
                )
                last_status = now
                if rss > cfg.runtime.max_rss_mb:
                    log.error(
                        "memory %.0f MB > limit %.0f MB, exiting for restart", rss, cfg.runtime.max_rss_mb
                    )
                    code = EXIT_MEMORY
                    done.set()
                    break
            time.sleep(0.2)
    finally:
        done.set()
        for t in threads:
            t.join(timeout=5)
    log.info(
        "multi-camera stopped: frames=%d events=%d",
        sum(e.stats.frames for _, e, _ in pipelines),
        sum(e.stats.events for _, e, _ in pipelines),
    )
    return code


def main(argv: list[str] | None = None) -> int:
    import argparse

    from anpr import licence
    from anpr.camera import make_source
    from anpr.config import load_config
    from anpr.detector import make_detector
    from anpr.ocr import FastPlateOcr
    from anpr.push import apply_push_config, start_pusher
    from anpr.sources import mask_source, source_kind
    from anpr.storage import SqliteEventStore

    ap = argparse.ArgumentParser(prog="python -m anpr.engine", description="Run the ANPR pipeline")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--source", help="override camera.source (webcam index, video path, rtsp://, picamera)")
    ap.add_argument("--no-realtime", action="store_true", help="video files: process every frame")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument(
        "--exit-at-end",
        action="store_true",
        help="exit when a video file ends (default: keep running and wait for a new stream)",
    )
    ap.add_argument(
        "--licence", help=f"licence file (default: {licence.LICENCE_FILE} next to the config file)"
    )
    ap.add_argument("--machine-id", action="store_true", help="print this device's ID for the licence, exit")
    ap.add_argument("--check-licence", action="store_true", help="check the licence for this device, exit")
    ap.add_argument(
        "--activate",
        metavar="KEY",
        help="activate this device with a licence key (or a licence file path, or - to read it), exit",
    )
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
        print(
            f"activated: licence {lic.licence_id} for {lic.customer}, "
            f"{'valid until ' + str(lic.expires) if lic.expires else 'no end date'} (saved to {licence_path})"
        )
        return 0
    if args.check_licence:
        # Same answer as the running engine: it also uses the newest time the engine has seen. The
        # database is only read (this may run as root: it must not create files the service can't use).
        try:
            db = load_config(args.config).storage.db_path
        except Exception:  # noqa: BLE001 - no/old config: check the file and the clock only
            db = None
        try:
            lic = licence.LicenceGuard.open(licence_path, licence.ReadOnlySettings(db)).licence
        except licence.LicenceError as e:
            print(f"licence NOT valid: {e}")
            return EXIT_LICENCE
        print(
            f"licence OK: {lic.licence_id} for {lic.customer}, "
            f"{'valid until ' + str(lic.expires) if lic.expires else 'no end date'}"
        )
        return 0

    cfg = load_config(args.config)
    cam = cfg.camera
    if args.no_realtime:
        cam = cam.model_copy(update={"realtime": False})
    cfg = cfg.model_copy(update={"camera": cam})

    logging.basicConfig(
        level=cfg.runtime.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cv2.setNumThreads(1)  # the models own the cores; keep OpenCV from oversubscribing the Pi

    stopping = False

    def _on_signal(signum, _frame):  # noqa: ANN001
        nonlocal stopping
        log.info("signal %d received, stopping", signum)
        stopping = True

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    playlist = tuple(str(Path(p).resolve()) if source_kind(p) == "file" else p for p in cfg.camera.playlist)

    def open_source(address: str) -> FrameSource:
        cam = cfg.camera.model_copy(update={"source": address})
        if address in playlist:  # a playlist video plays once; `loop` repeats the whole playlist
            cam = cam.model_copy(update={"loop": False})
        return make_source(cam)

    store = SqliteEventStore(cfg.storage.db_path, cfg.storage.image_dir, cfg.storage.snapshot_width)
    # Nothing runs (no camera, no plates, no sending) without a valid licence for this device. As a
    # service it waits and checks again every minute, so a licence file copied in later is picked up.
    guard = licence.wait_for_licence(licence_path, store, stop=lambda: stopping, wait=not args.exit_at_end)
    if guard is None:
        store.close()
        return 0 if stopping else EXIT_LICENCE
    # Sends new plates to the client's API. When config.yaml has a `push:` section it is the source of
    # truth, applied here on every start ("file wins"); otherwise sending is managed from the dashboard.
    if cfg.push is not None:
        apply_push_config(store, cfg.push)
    sender, sender_stop = start_pusher(store, cfg.web.station_name)
    try:
        if cfg.cameras:
            # Two or more fixed cameras in one process (headless gate installs). --source is ignored.
            if args.source is not None:
                log.warning("--source is ignored when config.yaml lists multiple cameras")
            code = run_cameras(
                cfg,
                list(cfg.cameras),
                store,
                stop=lambda: stopping or guard.expired(),
                max_frames=args.max_frames,
                exit_at_end=args.exit_at_end,
            )
            if guard.error is not None:
                log.error("stopped: licence no longer valid (%s)", guard.error)
                return EXIT_LICENCE
            return code
        engine = Engine(cfg, make_detector(cfg.detector), FastPlateOcr(cfg.ocr), store)
        control = StreamControl(
            open=open_source, default=cfg.camera.source, playlist=playlist, loop_playlist=cfg.camera.loop
        )
        # Which stream first: --source wins; else the last one picked on the dashboard (kept across
        # restarts/reboots); else config.yaml. Requests already in the database count as applied.
        saved = store.read_source_request()
        if saved is not None:
            control.applied_rev = saved[0]
        if args.source is not None:
            address = args.source
            if source_kind(address) == "file" and Path(address).exists():
                address = str(Path(address).resolve())
        elif saved is not None and saved[1]:
            address = saved[1]
        else:
            address = control.target("")  # the playlist's first video, or config camera.source
        try:
            source: FrameSource | None = open_source(address)
        except Exception as e:  # noqa: BLE001 - reported through status, the engine keeps waiting
            log.error("cannot open %s: %s", mask_source(address), e)
            source = None
        log.info("engine started: source=%s detector=%s", mask_source(address), cfg.detector.backend)
        code = run(
            cfg,
            source,
            engine,
            stop=lambda: stopping or guard.expired(),  # the date is checked again every hour
            max_frames=args.max_frames,
            source_name=address,
            control=None if args.exit_at_end else control,
        )
        if guard.error is not None:
            log.error("stopped: licence no longer valid (%s)", guard.error)
            return EXIT_LICENCE
        return code
    finally:
        sender_stop.set()
        sender.join(timeout=15)  # a request in flight times out within 10 s
        store.close()


if __name__ == "__main__":
    sys.exit(main())
