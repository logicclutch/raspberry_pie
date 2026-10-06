"""SQLite (WAL) event store + JPEG files. Shared by the engine (writer) and the web API (reader)."""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import re
import sqlite3
import stat
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from anpr.types import EngineStatus, PlateEvent

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    plate         TEXT    NOT NULL,
    kind          TEXT    NOT NULL,
    confidence    REAL    NOT NULL,
    votes         INTEGER NOT NULL,
    track_id      INTEGER NOT NULL,
    first_seen    REAL    NOT NULL,
    last_seen     REAL    NOT NULL,
    crop_path     TEXT,
    snapshot_path TEXT,
    hsrp          TEXT,
    camera        TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_last_seen ON events(last_seen);
CREATE INDEX IF NOT EXISTS idx_events_plate ON events(plate);
CREATE TABLE IF NOT EXISTS status (
    id   INTEGER PRIMARY KEY CHECK (id = 1),
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_COLS = (
    "id, plate, kind, confidence, votes, track_id, first_seen, last_seen, "
    "crop_path, snapshot_path, hsrp, camera"
)
_NON_ALNUM = re.compile(r"[^A-Z0-9]")
_JPEG_QUALITY = 85
PREVIEW_NAME = "live.jpg"
# RAM-backed folder for the live preview (rewritten every second: ~5 GB/day that must not go to the
# Pi's SD card). Linux only; elsewhere, or if it is unusable, the preview sits next to the image folder.
PREVIEW_RAM_ROOT: Path | None = Path("/dev/shm") if sys.platform.startswith("linux") else None


def normalize_query(q: str) -> str:
    return _NON_ALNUM.sub("", q.upper())


def _row_to_event(row: tuple) -> PlateEvent:
    (eid, plate, kind, conf, votes, track_id, first_seen, last_seen, crop, snap, hsrp, camera) = row
    return PlateEvent(
        plate=plate,
        kind=kind,
        confidence=conf,
        votes=votes,
        track_id=track_id,
        first_seen=first_seen,
        last_seen=last_seen,
        crop_path=crop,
        snapshot_path=snap,
        id=eid,
        hsrp=hsrp,
        camera=camera,
    )


class SqliteEventStore:
    """Thread-safe within one process; safe across processes thanks to WAL + busy timeout."""

    def __init__(self, db_path: Path, image_dir: Path, snapshot_width: int = 640) -> None:
        self._image_dir = Path(image_dir)
        self._snapshot_width = snapshot_width
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._image_dir.mkdir(parents=True, exist_ok=True)
        # Names this station's RAM preview folder (see preview_path).
        self._preview_key = hashlib.sha256(str(self._image_dir.resolve()).encode()).hexdigest()[:16]
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), timeout=5.0, check_same_thread=False)
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.executescript(_SCHEMA)
            cols = {r[1] for r in self._conn.execute("PRAGMA table_info(events)")}
            for col in ("hsrp", "camera"):  # databases from before these columns existed
                if col not in cols:
                    try:
                        self._conn.execute(f"ALTER TABLE events ADD COLUMN {col} TEXT")
                    except sqlite3.OperationalError as exc:  # the other process (engine/web) was first
                        if "duplicate column" not in str(exc):
                            raise
            self._conn.commit()

    @property
    def image_dir(self) -> Path:
        return self._image_dir

    # ---- live preview -------------------------------------------------------------------------
    # One fixed file, never under the image folder (not served by /images/, which is cached for a
    # day; never touched by retention). On Linux (the Pi) it lives in RAM, /dev/shm/anpr-<uid>-<id>/:
    # it is replaced every second and would otherwise wear out the SD card. Elsewhere it sits next
    # to the image folder (data/images -> data/live.jpg). Engine and web derive the same place from
    # the same config, as long as both run as the same user (the systemd units do).

    def _ram_preview_dir(self) -> Path | None:
        root = PREVIEW_RAM_ROOT
        if root is None:
            return None
        uid = os.geteuid()
        d = root / f"anpr-{uid}-{self._preview_key}"
        try:
            with contextlib.suppress(FileExistsError):
                d.mkdir(mode=0o700)
            st = os.lstat(d)
        except OSError:
            return None
        # The RAM root is world-writable: only use a real folder that is ours and closed to others
        # (never a symlink or a folder someone else created first).
        if not stat.S_ISDIR(st.st_mode) or st.st_uid != uid or st.st_mode & 0o077:
            return None
        return d

    def preview_path(self) -> Path:
        ram = self._ram_preview_dir()
        return (ram if ram is not None else self._image_dir.parent) / PREVIEW_NAME

    def clear_preview(self) -> None:
        """Remove the live preview (new stream: the old stream's picture must not pass for it)."""
        with contextlib.suppress(OSError):
            self.preview_path().unlink(missing_ok=True)

    def write_preview(self, image: np.ndarray | bytes, quality: int = 70) -> float:
        """Replace the live preview with `image` (BGR/gray array, or ready JPEG bytes). Atomic
        (tmp file + rename), so the web never serves half a picture. Returns the write time."""
        if isinstance(image, bytes | bytearray | memoryview):
            data = bytes(image)
        else:
            ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
            if not ok:
                raise ValueError("JPEG encoding failed")
            data = buf.tobytes()
        path = self.preview_path()
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
        return time.time()

    def read_preview(self) -> tuple[bytes, float] | None:
        """(JPEG bytes, unix time it was written) or None if there is no preview."""
        try:
            with open(self.preview_path(), "rb") as f:
                return f.read(), os.fstat(f.fileno()).st_mtime  # same inode: bytes and time match
        except OSError:  # missing, a folder, no permission, ... -> "no picture" (404), not a 500
            return None

    def preview_mtime(self) -> float | None:
        try:
            return self.preview_path().stat().st_mtime
        except OSError:
            return None

    # ---- images -------------------------------------------------------------------------------

    def _write_jpeg(self, rel: str, img: np.ndarray) -> str | None:
        path = self._image_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
        if not ok:
            return None
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(buf.tobytes())
        os.replace(tmp, path)  # atomic: the web never serves half-written files
        return rel

    def _shrink(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if w <= self._snapshot_width:
            return img
        scale = self._snapshot_width / w
        return cv2.resize(img, (self._snapshot_width, max(1, round(h * scale))), interpolation=cv2.INTER_AREA)

    # ---- events -------------------------------------------------------------------------------

    def add_event(
        self,
        event: PlateEvent,
        crop: np.ndarray | None = None,
        snapshot: np.ndarray | None = None,
    ) -> PlateEvent:
        day = time.strftime("%Y-%m-%d", time.localtime(event.last_seen))
        stem = f"{day}/{int(event.last_seen * 1000)}_{event.plate}_{event.track_id}"
        crop_path = self._write_jpeg(f"{stem}_crop.jpg", crop) if crop is not None and crop.size else None
        snap_path = (
            self._write_jpeg(f"{stem}_snap.jpg", self._shrink(snapshot))
            if snapshot is not None and snapshot.size
            else None
        )
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO events (plate, kind, confidence, votes, track_id, first_seen, last_seen,"
                " crop_path, snapshot_path, hsrp, camera) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    event.plate,
                    event.kind,
                    float(event.confidence),
                    int(event.votes),
                    int(event.track_id),
                    float(event.first_seen),
                    float(event.last_seen),
                    crop_path,
                    snap_path,
                    event.hsrp,
                    event.camera,
                ),
            )
            self._conn.commit()
            eid = cur.lastrowid
        return dataclasses.replace(event, id=eid, crop_path=crop_path, snapshot_path=snap_path)

    def get_event(self, event_id: int) -> PlateEvent | None:
        with self._lock:
            row = self._conn.execute(f"SELECT {_COLS} FROM events WHERE id = ?", (event_id,)).fetchone()
        return _row_to_event(row) if row else None

    def list_events(
        self,
        q: str | None = None,
        since: float | None = None,
        until: float | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[PlateEvent], int]:
        where: list[str] = []
        args: list[object] = []
        if q:
            nq = normalize_query(q)
            if nq:
                where.append("plate LIKE ? ESCAPE '\\'")
                args.append(f"%{nq}%")  # nq is only A-Z0-9, so nothing to escape
        if since is not None:
            where.append("last_seen >= ?")
            args.append(float(since))
        if until is not None:
            where.append("last_seen <= ?")
            args.append(float(until))
        clause = f" WHERE {' AND '.join(where)}" if where else ""
        limit = max(0, int(limit))
        offset = max(0, int(offset))
        with self._lock:
            total = self._conn.execute(f"SELECT COUNT(*) FROM events{clause}", args).fetchone()[0]
            rows = self._conn.execute(
                f"SELECT {_COLS} FROM events{clause} ORDER BY last_seen DESC, id DESC LIMIT ? OFFSET ?",
                [*args, limit, offset],
            ).fetchall()
        return [_row_to_event(r) for r in rows], int(total)

    def events_after(self, event_id: int, limit: int = 100) -> list[PlateEvent]:
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {_COLS} FROM events WHERE id > ? ORDER BY id ASC LIMIT ?",
                (int(event_id), int(limit)),
            ).fetchall()
        return [_row_to_event(r) for r in rows]

    def latest_id(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COALESCE(MAX(id), 0) FROM events").fetchone()
        return int(row[0])

    def stats(self, since: float, until: float, bucket_s: float = 3600.0, top_n: int = 5) -> dict:
        """Counts for the dashboard over [since, until): total, unique plates, mean confidence,
        per-bucket counts (oldest first) and the most frequently seen plates."""
        if until <= since or bucket_s <= 0:
            raise ValueError("need since < until and bucket_s > 0")
        n_buckets = max(1, int(-(-(until - since) // bucket_s)))
        rng = (float(since), float(until))
        with self._lock:
            total, unique, avg = self._conn.execute(
                "SELECT COUNT(*), COUNT(DISTINCT plate), AVG(confidence) FROM events "
                "WHERE last_seen >= ? AND last_seen < ?",
                rng,
            ).fetchone()
            rows = self._conn.execute(
                "SELECT CAST((last_seen - ?) / ? AS INTEGER) AS b, COUNT(*) FROM events "
                "WHERE last_seen >= ? AND last_seen < ? GROUP BY b",
                (float(since), float(bucket_s), *rng),
            ).fetchall()
            top = self._conn.execute(
                "SELECT plate, kind, COUNT(*) AS n, MAX(last_seen) FROM events "
                "WHERE last_seen >= ? AND last_seen < ? GROUP BY plate, kind "
                "ORDER BY n DESC, MAX(last_seen) DESC LIMIT ?",
                (*rng, int(top_n)),
            ).fetchall()
        buckets = [0] * n_buckets
        for b, n in rows:
            if 0 <= b < n_buckets:
                buckets[b] = int(n)
        return {
            "total": int(total),
            "unique": int(unique),
            "avg_confidence": None if avg is None else float(avg),
            "buckets": buckets,
            "top": [
                {"plate": pl, "kind": k, "count": int(n), "last_seen": float(ls)} for pl, k, n, ls in top
            ],
        }

    def recent_plate_seen(self, plate: str, within_s: float, now: float) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM events WHERE plate = ? AND last_seen >= ? LIMIT 1",
                (plate, now - within_s),
            ).fetchone()
        return row is not None

    # ---- status -------------------------------------------------------------------------------

    def write_status(self, status: EngineStatus) -> None:
        data = json.dumps(dataclasses.asdict(status))
        with self._lock:
            self._conn.execute(
                "INSERT INTO status (id, data) VALUES (1, ?)"
                " ON CONFLICT(id) DO UPDATE SET data=excluded.data",
                (data,),
            )
            self._conn.commit()

    def read_status(self) -> EngineStatus | None:
        with self._lock:
            row = self._conn.execute("SELECT data FROM status WHERE id = 1").fetchone()
        if not row:
            return None
        try:
            data = json.loads(row[0])
            known = {f.name for f in dataclasses.fields(EngineStatus)}
            return EngineStatus(**{k: v for k, v in data.items() if k in known})
        except (TypeError, ValueError, AttributeError):
            return None

    # ---- stream request (dashboard -> engine) ---------------------------------------------------

    def request_source(self, source: str, now: float | None = None) -> int:
        """Ask the engine to switch to `source` ("" = config camera). Returns the request number;
        every call gets a new one, so re-sending the same video restarts it."""
        with self._lock, self._conn:  # one transaction: read the counter and bump it
            row = self._conn.execute("SELECT value FROM settings WHERE key = 'source'").fetchone()
            rev = 0
            if row:
                with contextlib.suppress(TypeError, ValueError):
                    rev = int(json.loads(row[0]).get("rev", 0))
            rev += 1
            data = json.dumps({"source": source, "rev": rev, "ts": time.time() if now is None else now})
            self._conn.execute(
                "INSERT INTO settings (key, value) VALUES ('source', ?)"
                " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (data,),
            )
        return rev

    def read_source_request(self) -> tuple[int, str] | None:
        """(request number, source) of the latest dashboard request, or None if there never was one."""
        with self._lock:
            row = self._conn.execute("SELECT value FROM settings WHERE key = 'source'").fetchone()
        if not row:
            return None
        try:
            d = json.loads(row[0])
            return int(d["rev"]), str(d["source"])
        except (TypeError, ValueError, KeyError):
            return None

    # ---- plain settings (JSON values) -----------------------------------------------------------

    def read_setting(self, key: str) -> object | None:
        """The JSON value saved under `key`, or None if there is none (or it is unreadable)."""
        with self._lock:
            row = self._conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        if not row:
            return None
        try:
            return json.loads(row[0])
        except ValueError:
            return None

    def write_setting(self, key: str, value: object) -> None:
        data = json.dumps(value)
        with self._lock:
            self._conn.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, data),
            )
            self._conn.commit()

    def count_after(self, event_id: int) -> int:
        """How many events have an id above `event_id`."""
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM events WHERE id > ?", (int(event_id),)).fetchone()
        return int(row[0])

    # ---- retention ----------------------------------------------------------------------------

    def purge_older_than(self, days: float, now: float) -> int:
        cutoff = now - days * 86400.0
        with self._lock:
            rows = self._conn.execute(
                "SELECT crop_path, snapshot_path FROM events WHERE last_seen < ?", (cutoff,)
            ).fetchall()
            self._conn.execute("DELETE FROM events WHERE last_seen < ?", (cutoff,))
            self._conn.commit()
        for rel in (p for r in rows for p in r if p):
            with contextlib.suppress(OSError):
                (self._image_dir / rel).unlink(missing_ok=True)
        for d in self._image_dir.iterdir():  # drop empty day folders
            if d.is_dir():
                with contextlib.suppress(OSError):
                    d.rmdir()  # only succeeds when empty
        return len(rows)

    def resolve_image(self, rel: str) -> Path | None:
        """Absolute path for a stored image, or None if `rel` escapes image_dir / doesn't exist."""
        base = self._image_dir.resolve()
        path = (base / rel).resolve()
        if not path.is_relative_to(base) or not path.is_file():
            return None
        return path

    def close(self) -> None:
        with self._lock:
            self._conn.close()
