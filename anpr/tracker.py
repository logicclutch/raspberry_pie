"""Light plate tracker: gives each plate a stable track id across frames (pure Python/NumPy).

At 2-5 detector frames per second a moving plate can jump further than its own width between
frames, so boxes with no overlap are still matched when their centres are close and their sizes
are similar.
"""

from __future__ import annotations

from dataclasses import dataclass

from anpr.config import TrackingConfig
from anpr.types import Box

_CENTER_GATE = 2.0  # max centre distance, in units of the larger box width
_SIZE_RATIO = 2.0  # max width ratio between matched boxes
# A centre match that jumps more than _JUMP box widths must also land within _OFF_PATH widths of
# where the track's own motion puts it. Painted "GOODS" under a plate the detector missed for a
# frame is 1.2 widths away and 1.25 off the plate's path; real plates in the gate videos land
# <= 0.23 off it (0.76 jump / 1.2 off after a 0.8 s gap, kept by the _JUMP condition).
_JUMP = 1.0
_OFF_PATH = 0.6


@dataclass(slots=True)
class Track:
    id: int
    box: Box
    first_seen: float
    last_seen: float
    hits: int = 1
    velocity: tuple[float, float] | None = None  # centre motion, px/s, from the last two matches


def _centre(b: Box) -> tuple[float, float]:
    return (b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2


def iou(a: Box, b: Box) -> float:
    ix = max(0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    if inter == 0:
        return 0.0
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


def _match_score(tr: Track, d: Box, iou_match: float, ts: float) -> float:
    """>0 means the pair may be matched; higher is better. IoU matches always beat centre matches."""
    t = tr.box
    ov = iou(t, d)
    if ov >= iou_match:
        return 1.0 + ov
    wt, wd = max(t.width, 1), max(d.width, 1)
    if max(wt, wd) / min(wt, wd) > _SIZE_RATIO:
        return 0.0
    (tx, ty), (cx, cy) = _centre(t), _centre(d)
    dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
    w = max(wt, wd)
    gate = _CENTER_GATE * w
    if dist >= gate:
        return 0.0
    if tr.velocity is not None and dist > _JUMP * w:
        dt = ts - tr.last_seen
        px, py = tx + tr.velocity[0] * dt, ty + tr.velocity[1] * dt
        if ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 > _OFF_PATH * w:
            return 0.0  # a big jump against the track's motion: something else, not this plate
    return 1.0 - dist / gate  # in (0, 1]


class IouTracker:
    def __init__(self, cfg: TrackingConfig) -> None:
        self._cfg = cfg
        self._tracks: dict[int, Track] = {}
        self._next_id = 1

    @property
    def tracks(self) -> dict[int, Track]:
        return self._tracks

    def has_active(self) -> bool:
        return bool(self._tracks)

    def update(self, boxes: list[Box], ts: float) -> tuple[list[tuple[int, Box]], list[Track]]:
        """Match detections to tracks. Returns ([(track_id, box)] for this frame, expired tracks)."""
        pairs: list[tuple[float, int, int]] = []
        for tid, tr in self._tracks.items():
            for j, b in enumerate(boxes):
                s = _match_score(tr, b, self._cfg.iou_match, ts)
                if s > 0:
                    pairs.append((s, tid, j))
        pairs.sort(key=lambda p: -p[0])  # greedy, best pairs first

        used_t: set[int] = set()
        used_d: set[int] = set()
        out: list[tuple[int, Box]] = []
        for _s, tid, j in pairs:
            if tid in used_t or j in used_d:
                continue
            used_t.add(tid)
            used_d.add(j)
            tr = self._tracks[tid]
            dt = ts - tr.last_seen
            if dt > 0:
                (ox, oy), (nx, ny) = _centre(tr.box), _centre(boxes[j])
                tr.velocity = ((nx - ox) / dt, (ny - oy) / dt)
            tr.box, tr.last_seen, tr.hits = boxes[j], ts, tr.hits + 1
            out.append((tid, boxes[j]))

        for j, b in enumerate(boxes):
            if j not in used_d:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = Track(id=tid, box=b, first_seen=ts, last_seen=ts)
                out.append((tid, b))

        expired = [t for t in self._tracks.values() if ts - t.last_seen > self._cfg.max_age_s]
        for t in expired:
            del self._tracks[t.id]
        return out, expired
