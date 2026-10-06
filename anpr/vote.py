"""Multi-frame voting: a plate is reported only when several frames agree with high confidence."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from anpr.config import VoteConfig
from anpr.types import PlateKind, ValidPlate


@dataclass(frozen=True, slots=True)
class Decision:
    plate: str
    kind: PlateKind
    votes: int
    confidence: float  # mean confidence of the winning reads
    agreement: float  # winning reads / all valid reads on the track


def decide(reads: list[ValidPlate], cfg: VoteConfig) -> Decision | None:
    """The strict acceptance rule. None = not sure yet (or never)."""
    if not reads:
        return None
    counts = Counter(r.text for r in reads).most_common()
    text, votes = counts[0]
    if len(counts) > 1 and counts[1][1] == votes:  # tie -> not sure
        return None
    if votes < cfg.min_votes:
        return None
    agreement = votes / len(reads)
    if agreement < cfg.min_agreement:
        return None
    winners = [r for r in reads if r.text == text]
    conf = sum(r.confidence for r in winners) / len(winners)
    if conf < cfg.min_avg_conf:
        return None
    return Decision(plate=text, kind=winners[0].kind, votes=votes, confidence=conf, agreement=agreement)


@dataclass(slots=True)
class _Evidence:
    """Best image evidence for one plate text on one track (kept small for the Pi's RAM)."""

    conf: float
    crop: np.ndarray
    snapshot: np.ndarray | None
    key: tuple[float, float] = (0.0, 0.0)


@dataclass(slots=True)
class _TrackVotes:
    reads: list[ValidPlate] = field(default_factory=list)
    widths: list[float | None] = field(default_factory=list)  # plate width of each read (px)
    best: dict[str, _Evidence] = field(default_factory=dict)
    reported: bool = False
    seen: list[tuple[float, float]] = field(default_factory=list)  # (ts, box width) of every sighting
    max_width: float = 0.0  # how close the plate got: largest median of 3 consecutive widths
    recent: list[float] = field(default_factory=list)  # last _WIDTH_WINDOW widths


# The close-up floor uses the median of this many consecutive plate widths, so a box that briefly
# swallows nearby painted lettering (NL01AC0056: 113 px -> 144 px for 3 frames when the "C" of
# "CARRIER" joined the box) cannot raise the floor above every real read. A vehicle that really
# comes close stays big for many frames, so the median still follows it (TN33BY9603).
_WIDTH_WINDOW = 3


def _one_char_apart(a: str, b: str) -> bool:
    """Same length, exactly one position different (one misread character)."""
    return len(a) == len(b) and sum(x != y for x, y in zip(a, b, strict=True)) == 1


def _grow(tv: _TrackVotes, width: float) -> None:
    tv.recent.append(width)
    if len(tv.recent) > _WIDTH_WINDOW:
        del tv.recent[0]
    ordered = sorted(tv.recent)
    tv.max_width = max(tv.max_width, ordered[(len(ordered) - 1) // 2])  # lower median


class PlateVoter:
    """Collects validated reads per track and decides when to report."""

    # Cap per-track history so a car parked in view for hours can't grow memory without bound.
    MAX_READS = 60

    def __init__(self, cfg: VoteConfig) -> None:
        self._cfg = cfg
        self._tracks: dict[int, _TrackVotes] = {}

    def is_reported(self, track_id: int) -> bool:
        tv = self._tracks.get(track_id)
        return tv is not None and tv.reported

    def observe(self, track_id: int, width: float, ts: float) -> None:
        """Record a sighting of the track's plate (every frame it is detected, read or not). Sets how
        close the vehicle got (the close-up floor) even when the close-up frames give no valid read,
        so far misreads cannot vote then (TN33BY9603 read as TN33BT9603 far away)."""
        tv = self._tracks.setdefault(track_id, _TrackVotes())
        tv.seen.append((ts, float(width)))
        _grow(tv, float(width))
        keep_s = max(self._cfg.settle_s, 1.0) * 3
        while len(tv.seen) > 2 and tv.seen[1][0] < ts - keep_s:  # keep the first sample (track age)
            del tv.seen[1]

    def _settled(self, tv: _TrackVotes) -> bool:
        """The plate stopped growing: no new maximum (beyond settle_growth) in the last settle_s."""
        s = self._cfg.settle_s
        if s <= 0 or not tv.seen:
            return True
        now = tv.seen[-1][0]
        if now - tv.seen[0][0] < s:
            return False  # too young to know whether it is still approaching
        before = max((w for t, w in tv.seen if t <= now - s), default=0.0)
        recent = max((w for t, w in tv.seen if t > now - s), default=0.0)
        return before > 0 and recent <= before * (1.0 + self._cfg.settle_growth)

    def _close_reads(self, tv: _TrackVotes) -> list[ValidPlate]:
        r = self._cfg.close_ratio
        if r <= 0 or tv.max_width <= 0:
            return list(tv.reads)
        floor = r * tv.max_width
        return [rd for rd, w in zip(tv.reads, tv.widths, strict=True) if w is None or w >= floor]

    def add(
        self,
        track_id: int,
        read: ValidPlate,
        crop: np.ndarray,
        snapshot: np.ndarray | None = None,
        width: float | None = None,
    ) -> Decision | None:
        """Add a read (width = its plate width in px). Returns a Decision the first time the track
        becomes confident: several agreeing close-up reads, once the plate has stopped growing."""
        tv = self._tracks.setdefault(track_id, _TrackVotes())
        tv.reads.append(read)
        tv.widths.append(width)
        if width is not None and not tv.seen:  # no sightings recorded: size the floor from reads
            _grow(tv, float(width))
        if len(tv.reads) > self.MAX_READS:
            del tv.reads[0]
            del tv.widths[0]
        ev = tv.best.get(read.text)
        # Evidence: the biggest (closest) crop of each text, then the most confident.
        key = ((width or 0.0), read.confidence)
        if ev is None or key > ev.key:
            tv.best[read.text] = _Evidence(read.confidence, crop, snapshot, key)
        if tv.reported or not self._settled(tv):
            return None
        d = decide(self._close_reads(tv), self._cfg)
        if d is not None:
            tv.reported = True
        return d

    def evidence(self, track_id: int, plate: str) -> tuple[np.ndarray | None, np.ndarray | None]:
        tv = self._tracks.get(track_id)
        ev = tv.best.get(plate) if tv else None
        return (ev.crop, ev.snapshot) if ev else (None, None)

    def finish(self, track_id: int) -> Decision | None:
        """The vehicle left without a report: last chance. First the normal rule on the close-up reads
        (several agreeing reads), failing that the end-of-track rules (`_end_rule`). Marks the track
        reported."""
        tv = self._tracks.get(track_id)
        if tv is None or tv.reported:
            return None
        close = self._close_reads(tv)
        d = decide(close, self._cfg) or self._end_rule(tv, close)
        if d is None:
            return None
        tv.reported = True
        return d

    def _end_rule(self, tv: _TrackVotes, close: list[ValidPlate]) -> Decision | None:
        """Stricter rules for a vehicle that gave too few reads for the normal rule (`end_min_votes`
        = 0 disables them):

        1. Two-thirds of the close-up reads, one disputed character: the leading close-up text has
           at least `end_min_votes` reads and at least two-thirds of all close-up reads, every other
           close-up read differs from it in exactly one character, and its mean confidence >=
           `end_min_conf`. With no other reads this is "all close-up reads identical". RJ14UN8156:
           N, N and W at the same position -> N. A 3-against-2 split stays unsure: the KA03MI0352
           Suzuki was read KA03M1035 x3 / KA03M1005 x2, both wrong.
        2. Nothing disagrees at any distance: every valid read on the track is the same text, at least
           `min_votes` of them, at least one is a close-up, mean confidence >= `min_avg_conf` (the
           normal rule's bar). The close-up filter only stops far reads outvoting close ones; with no
           rival there is nothing to outvote. DL12CS4288: 3 identical reads at 75, 80 and 98 px.
        """
        cfg = self._cfg
        if cfg.end_min_votes <= 0 or not close:
            return None
        counts = Counter(r.text for r in close).most_common()
        text, votes = counts[0]
        if (
            votes >= cfg.end_min_votes
            and 3 * votes >= 2 * len(close)
            and all(_one_char_apart(other, text) for other, _ in counts[1:])
        ):
            winners = [r for r in close if r.text == text]
            conf = sum(r.confidence for r in winners) / len(winners)
            if conf >= cfg.end_min_conf:
                agreement = votes / len(close)
                return Decision(
                    plate=text, kind=winners[0].kind, votes=votes, confidence=conf, agreement=agreement
                )
        reads = tv.reads
        if len(reads) >= cfg.min_votes and len({r.text for r in reads}) == 1:
            conf = sum(r.confidence for r in reads) / len(reads)
            if conf >= cfg.min_avg_conf:
                return Decision(
                    plate=reads[0].text, kind=reads[0].kind, votes=len(reads), confidence=conf, agreement=1.0
                )
        return None

    def drop(self, track_id: int) -> list[ValidPlate]:
        """Forget a finished track. Returns its reads (for logging unreported vehicles)."""
        tv = self._tracks.pop(track_id, None)
        return tv.reads if tv else []
