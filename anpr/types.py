"""Shared data types and interfaces. Every module codes against these — change with care."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

PlateKind = Literal["standard", "bh"]


@dataclass(frozen=True, slots=True)
class Box:
    """Plate box in FULL-FRAME pixel coordinates (x2/y2 exclusive)."""

    x1: int
    y1: int
    x2: int
    y2: int
    score: float

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


@dataclass(frozen=True, slots=True)
class OcrResult:
    """Raw OCR output: text (pad chars removed) and one confidence per character in `text`."""

    text: str
    char_confs: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ValidPlate:
    """A read that passed the strict Indian-format validator."""

    text: str  # canonical, no spaces, e.g. "MH12AB1234" / "22BH1234AB"
    kind: PlateKind
    confidence: float  # 0..1, mean per-character confidence after repair penalties


@dataclass(frozen=True, slots=True)
class PlateEvent:
    """One confirmed vehicle plate (after multi-frame voting)."""

    plate: str
    kind: PlateKind
    confidence: float
    votes: int
    track_id: int
    first_seen: float  # unix epoch seconds
    last_seen: float  # unix epoch seconds (the event time)
    crop_path: str | None = None  # relative to storage image_dir
    snapshot_path: str | None = None  # relative to storage image_dir
    id: int | None = None  # set by the store
    # "hsrp" | "non_hsrp" | "unsure" (see anpr.hsrp); None = not checked (check off, or older event)
    hsrp: str | None = None
    # Which camera saw it (multi-camera installs). "" / None = the single default camera.
    camera: str | None = None


@dataclass(frozen=True, slots=True)
class EngineStatus:
    """Heartbeat the engine writes; the web process reads it for /health."""

    ts: float
    fps: float  # processed frames per second (recent average)
    camera_ok: bool
    frames: int  # total frames processed since start
    events: int  # total events emitted since start
    rss_mb: float
    last_error: str | None = None
    source: str | None = None  # masked (no RTSP password), see anpr.sources.mask_source
    # "running" | "ended" (video finished) | "error" (could not open) | None (older engines)
    source_state: str | None = None
    source_rev: int = 0  # the dashboard stream request this engine has applied
    preview_ts: float | None = None  # unix time the live preview JPEG was last written (None = never)


class FrameSource(Protocol):
    def start(self) -> None: ...

    def read(self, timeout: float = 1.0) -> tuple[np.ndarray, float] | None:
        """Return (BGR frame, capture unix ts) for a frame not returned before, or None on timeout."""

    @property
    def finished(self) -> bool:
        """True once a finite source (video file) is exhausted."""

    def stop(self) -> None: ...


class PlateDetector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Box]:
        """Detect plates in a BGR frame. Boxes are in the frame's pixel coordinates."""


class PlateOcr(Protocol):
    def read(self, crop: np.ndarray) -> OcrResult | None:
        """Read one BGR plate crop. None if the model output is unusable."""


class EventStore(Protocol):
    """Persistence shared by the engine (writer) and the web API (reader), backed by SQLite."""

    def add_event(
        self,
        event: PlateEvent,
        crop: np.ndarray | None = None,
        snapshot: np.ndarray | None = None,
    ) -> PlateEvent:
        """Persist the event (+ JPEGs if given) and return it with id and image paths set."""

    def get_event(self, event_id: int) -> PlateEvent | None: ...

    def list_events(
        self,
        q: str | None = None,
        since: float | None = None,
        until: float | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[PlateEvent], int]:
        """Newest first. `q` is a case/space-insensitive substring of the plate. Returns (page, total)."""

    def events_after(self, event_id: int, limit: int = 100) -> list[PlateEvent]:
        """Events with id > event_id, oldest first (for live push)."""

    def recent_plate_seen(self, plate: str, within_s: float, now: float) -> bool: ...

    def write_status(self, status: EngineStatus) -> None: ...

    def read_status(self) -> EngineStatus | None: ...

    def write_preview(self, image: np.ndarray | bytes, quality: int = 70) -> float:
        """Atomically replace the live preview JPEG. Returns the time it was written."""

    def clear_preview(self) -> None:
        """Remove the live preview JPEG (no-op if there is none). Never raises."""

    def purge_older_than(self, days: float, now: float) -> int:
        """Delete events (and their image files) older than `days`. Returns rows deleted."""

    def close(self) -> None: ...
