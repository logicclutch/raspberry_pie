import numpy as np
import pytest

from anpr.storage import SqliteEventStore, normalize_query
from anpr.types import EngineStatus, PlateEvent

T0 = 1_780_000_000.0


def ev(plate: str, ts: float, track: int = 1) -> PlateEvent:
    return PlateEvent(
        plate=plate,
        kind="standard",
        confidence=0.93,
        votes=4,
        track_id=track,
        first_seen=ts - 2,
        last_seen=ts,
    )


@pytest.fixture
def store(tmp_path):
    s = SqliteEventStore(tmp_path / "db" / "anpr.db", tmp_path / "images", snapshot_width=64)
    yield s
    s.close()


def test_add_and_get_with_images(store):
    crop = np.full((40, 120, 3), 200, np.uint8)
    snap = np.zeros((360, 640, 3), np.uint8)
    saved = store.add_event(ev("MH12AB1234", T0), crop=crop, snapshot=snap)
    assert saved.id is not None
    assert saved.crop_path and saved.snapshot_path
    assert store.resolve_image(saved.crop_path) is not None
    import cv2

    snap_img = cv2.imread(str(store.resolve_image(saved.snapshot_path)))
    assert snap_img.shape[1] == 64  # shrunk to snapshot_width
    assert store.get_event(saved.id) == saved


def test_list_search_paging_and_time_filter(store):
    for i, p in enumerate(["MH12AB1234", "KA01MJ0001", "MH14CD5678"]):
        store.add_event(ev(p, T0 + i))
    items, total = store.list_events()
    assert total == 3 and [e.plate for e in items] == ["MH14CD5678", "KA01MJ0001", "MH12AB1234"]
    items, total = store.list_events(q="mh 12")
    assert total == 1 and items[0].plate == "MH12AB1234"
    items, total = store.list_events(q="MH", limit=1, offset=1)
    assert total == 2 and items[0].plate == "MH12AB1234"
    items, total = store.list_events(since=T0 + 1, until=T0 + 1)
    assert total == 1 and items[0].plate == "KA01MJ0001"


def test_like_wildcards_are_neutralised(store):
    store.add_event(ev("MH12AB1234", T0))
    assert store.list_events(q="%")[1] == 1  # '%' stripped -> empty query -> no filter
    assert store.list_events(q="_H12")[1] == 1


def test_events_after_and_latest_id(store):
    a = store.add_event(ev("MH12AB1234", T0))
    b = store.add_event(ev("KA01MJ0001", T0 + 1))
    assert [e.id for e in store.events_after(0)] == [a.id, b.id]
    assert [e.id for e in store.events_after(a.id)] == [b.id]
    assert store.latest_id() == b.id


def test_recent_plate_seen(store):
    store.add_event(ev("MH12AB1234", T0))
    assert store.recent_plate_seen("MH12AB1234", 60, T0 + 30)
    assert not store.recent_plate_seen("MH12AB1234", 60, T0 + 61)
    assert not store.recent_plate_seen("KA01MJ0001", 60, T0 + 1)


def test_status_roundtrip(store):
    assert store.read_status() is None
    st = EngineStatus(ts=T0, fps=3.2, camera_ok=True, frames=10, events=1, rss_mb=210.0)
    store.write_status(st)
    store.write_status(st)
    assert store.read_status() == st


def test_purge_removes_rows_and_files(store):
    crop = np.full((40, 120, 3), 200, np.uint8)
    old = store.add_event(ev("MH12AB1234", T0), crop=crop)
    new = store.add_event(ev("KA01MJ0001", T0 + 40 * 86400), crop=crop)
    path = store.resolve_image(old.crop_path)
    assert store.purge_older_than(30, T0 + 40 * 86400) == 1
    assert not path.exists()
    assert store.get_event(old.id) is None and store.get_event(new.id) is not None


def test_resolve_image_blocks_traversal(store):
    assert store.resolve_image("../db/anpr.db") is None
    assert store.resolve_image("/etc/passwd") is None


def test_second_connection_sees_writes(tmp_path):
    a = SqliteEventStore(tmp_path / "anpr.db", tmp_path / "img")
    b = SqliteEventStore(tmp_path / "anpr.db", tmp_path / "img")
    a.add_event(ev("MH12AB1234", T0))
    assert b.list_events()[1] == 1
    a.close()
    b.close()


def test_normalize_query():
    assert normalize_query(" mh-12 ab%_") == "MH12AB"


def test_source_requests_count_up_and_persist(tmp_path):
    s = SqliteEventStore(tmp_path / "a.db", tmp_path / "img")
    assert s.read_source_request() is None
    assert s.request_source("/v/a.mp4") == 1
    assert s.request_source("/v/a.mp4") == 2  # same video again = a new request (replay)
    s.close()
    s = SqliteEventStore(tmp_path / "a.db", tmp_path / "img")
    assert s.read_source_request() == (2, "/v/a.mp4")
    assert s.request_source("") == 3
    assert s.read_source_request() == (3, "")
    s.close()


def test_status_from_an_older_or_newer_engine_still_reads(store):
    import json

    store._conn.execute(
        "INSERT INTO status (id, data) VALUES (1, ?)",
        (json.dumps({"ts": 1, "fps": 2, "camera_ok": True, "frames": 1, "events": 0, "rss_mb": 9, "x": 1}),),
    )
    store._conn.commit()
    st = store.read_status()
    assert st is not None and st.frames == 1 and st.source is None and st.source_rev == 0


# ---- live preview -----------------------------------------------------------------------------------


def test_preview_write_read_and_atomic_replace(store, tmp_path):
    import cv2

    assert store.read_preview() is None and store.preview_mtime() is None
    path = store.preview_path()
    assert path == tmp_path / "live.jpg"  # next to the image folder, not inside it
    assert not path.is_relative_to(store.image_dir)
    ts = store.write_preview(np.full((48, 64, 3), 200, np.uint8), quality=70)
    data, mtime = store.read_preview()
    assert data[:2] == b"\xff\xd8" and abs(mtime - ts) < 5
    assert cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR).shape == (48, 64, 3)
    inode = path.stat().st_ino
    store.write_preview(b"\xff\xd8second\xff\xd9")  # ready JPEG bytes are written as-is
    assert store.read_preview()[0] == b"\xff\xd8second\xff\xd9"
    assert path.stat().st_ino != inode  # replaced by rename, never rewritten in place
    assert sorted(p.name for p in tmp_path.iterdir() if p.is_file()) == ["live.jpg"]  # no .tmp left


def test_preview_survives_retention_purge(store):
    store.add_event(ev("MH12AB1234", T0), crop=np.zeros((10, 30, 3), np.uint8))
    store.write_preview(np.zeros((8, 8, 3), np.uint8))
    assert store.purge_older_than(1, T0 + 10 * 86400) == 1
    assert store.read_preview() is not None


def test_clear_preview(store):
    store.clear_preview()  # nothing there: no error
    store.write_preview(np.zeros((8, 8, 3), np.uint8))
    store.clear_preview()
    assert store.read_preview() is None and store.preview_mtime() is None


def test_preview_lives_in_ram_folder_on_linux(tmp_path, monkeypatch):
    import os
    import stat

    import anpr.storage

    ram = tmp_path / "shm"
    ram.mkdir()
    monkeypatch.setattr(anpr.storage, "PREVIEW_RAM_ROOT", ram)
    a = SqliteEventStore(tmp_path / "a.db", tmp_path / "data" / "images")
    web = SqliteEventStore(tmp_path / "a.db", tmp_path / "data" / "images")  # the other process
    other = SqliteEventStore(tmp_path / "b.db", tmp_path / "other" / "images")
    try:
        a.write_preview(np.zeros((8, 8, 3), np.uint8))
        path = a.preview_path()
        assert path.parent.parent == ram and path.parent.name.startswith(f"anpr-{os.geteuid()}-")
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
        assert web.preview_path() == path and web.read_preview() is not None  # same config, same file
        assert other.preview_path() != path and other.read_preview() is None  # another station
        assert not (tmp_path / "data" / "live.jpg").exists()  # nothing written to the SD card
    finally:
        for s in (a, web, other):
            s.close()


def test_preview_refuses_a_ram_folder_it_does_not_own_exclusively(tmp_path, monkeypatch):
    import anpr.storage

    ram = tmp_path / "shm"
    ram.mkdir()
    monkeypatch.setattr(anpr.storage, "PREVIEW_RAM_ROOT", ram)
    s = SqliteEventStore(tmp_path / "a.db", tmp_path / "data" / "images")
    try:
        d = s.preview_path().parent
        d.rmdir()
        elsewhere = tmp_path / "attacker"
        elsewhere.mkdir()
        d.symlink_to(elsewhere)  # planted before the engine starts
        assert s.preview_path() == tmp_path / "data" / "live.jpg"  # falls back to disk
        d.unlink()
        d.mkdir(mode=0o777)
        d.chmod(0o777)  # open to other users
        assert s.preview_path() == tmp_path / "data" / "live.jpg"
        s.write_preview(np.zeros((8, 8, 3), np.uint8))
        assert not list(elsewhere.iterdir()) and not list(d.iterdir())
    finally:
        s.close()


def test_status_without_preview_ts_still_parses(store):
    import json

    old = {"ts": T0, "fps": 5.0, "camera_ok": True, "frames": 1, "events": 0, "rss_mb": 100.0}
    with store._lock:
        store._conn.execute("INSERT INTO status (id, data) VALUES (1, ?)", (json.dumps(old),))
        store._conn.commit()
    st = store.read_status()
    assert st is not None and st.preview_ts is None
    store.write_status(
        EngineStatus(ts=T0, fps=5, camera_ok=True, frames=1, events=0, rss_mb=1, preview_ts=T0)
    )
    assert store.read_status().preview_ts == T0


def test_hsrp_result_round_trips(store):
    import dataclasses

    saved = store.add_event(dataclasses.replace(ev("MH12AB1234", T0), hsrp="non_hsrp"))
    assert store.get_event(saved.id).hsrp == "non_hsrp"
    assert store.get_event(store.add_event(ev("KA01MX0001", T0 + 1)).id).hsrp is None  # not checked


def test_database_from_before_the_hsrp_check_is_upgraded(tmp_path):
    import sqlite3

    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, plate TEXT NOT NULL, kind TEXT NOT NULL,"
        " confidence REAL NOT NULL, votes INTEGER NOT NULL, track_id INTEGER NOT NULL,"
        " first_seen REAL NOT NULL, last_seen REAL NOT NULL, crop_path TEXT, snapshot_path TEXT)"
    )
    con.execute(
        "INSERT INTO events (plate, kind, confidence, votes, track_id, first_seen, last_seen)"
        " VALUES ('MH12AB1234', 'standard', 0.9, 3, 1, 1.0, 2.0)"
    )
    con.commit()
    con.close()
    s = SqliteEventStore(db, tmp_path / "img")
    try:
        assert s.get_event(1).plate == "MH12AB1234" and s.get_event(1).hsrp is None
        import dataclasses

        new = s.add_event(dataclasses.replace(ev("KA01MX0001", T0), hsrp="hsrp"))
        assert s.get_event(new.id).hsrp == "hsrp"
        SqliteEventStore(db, tmp_path / "img").close()  # a second opener (web + engine) is fine
    finally:
        s.close()
