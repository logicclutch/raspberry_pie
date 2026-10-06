import csv
import io
import re
import sqlite3
import time
from datetime import datetime

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from anpr.config import AppConfig
from anpr.storage import SqliteEventStore
from anpr.types import EngineStatus, PlateEvent
from anpr.validator import format_display
from anpr.web.app import create_app

# 2026-05-28 local noon-ish; events are spaced one hour apart going back from T0.
T0 = datetime(2026, 5, 28, 12, 0, 0).timestamp()
TOKEN = "s3cret-token"


def ev(plate: str, ts: float, kind: str = "standard", track: int = 1) -> PlateEvent:
    return PlateEvent(
        plate=plate,
        kind=kind,
        confidence=0.91,
        votes=4,
        track_id=track,
        first_seen=ts - 2,
        last_seen=ts,
    )


def make_cfg(tmp_path, token: str | None = None) -> AppConfig:
    return AppConfig.model_validate(
        {
            "storage": {"db_path": str(tmp_path / "db" / "anpr.db"), "image_dir": str(tmp_path / "images")},
            "web": {"auth_token": token},
        }
    )


@pytest.fixture
def store(tmp_path):
    s = SqliteEventStore(tmp_path / "db" / "anpr.db", tmp_path / "images", snapshot_width=64)
    yield s
    s.close()


@pytest.fixture
def seeded(store):
    """5 events: ids 1..5, newest (id 5) at T0, each earlier one an hour before."""
    plates = ["MH12AB1234", "KA01MX0001", "MH14CD5678", "22BH1234AB", "DL3CAB1234"]
    saved = []
    for i, p in enumerate(plates):
        kind = "bh" if "BH" in p else "standard"
        saved.append(store.add_event(ev(p, T0 - (len(plates) - 1 - i) * 3600, kind=kind, track=i)))
    return saved


def client_for(tmp_path, store, token=None, **kw) -> TestClient:
    app = create_app(make_cfg(tmp_path, token), store, ws_poll_s=0.02, **kw)
    return TestClient(app)


@pytest.fixture
def client(tmp_path, store, seeded):
    with client_for(tmp_path, store) as c:
        yield c


# ---- list / search ------------------------------------------------------------------------------


def test_list_newest_first_with_meta(client):
    r = client.get("/api/v1/plates")
    assert r.status_code == 200
    body = r.json()
    assert [e["id"] for e in body["data"]] == [5, 4, 3, 2, 1]
    assert body["meta"] == {"total": 5, "page": 1, "per_page": 50, "total_pages": 1}
    first = body["data"][0]
    assert first["plate"] == "DL3CAB1234"
    assert first["plate_display"] == format_display("DL3CAB1234", "standard")
    assert first["last_seen_iso"].startswith("2026-05-28T12:00:00")
    assert first["crop_url"] is None


def test_search_is_substring_and_space_case_insensitive(client):
    r = client.get("/api/v1/plates", params={"q": "mh 12"})
    assert [e["plate"] for e in r.json()["data"]] == ["MH12AB1234"]
    r = client.get("/api/v1/plates", params={"q": "1234"})
    assert {e["plate"] for e in r.json()["data"]} == {"MH12AB1234", "22BH1234AB", "DL3CAB1234"}
    bh = client.get("/api/v1/plates", params={"q": "bh"}).json()["data"][0]
    assert bh["kind"] == "bh" and bh["plate_display"] == "22 BH 1234 AB"


def test_date_filters_iso_datetime_date_and_epoch(client):
    # id 3 is at T0-2h, id 4 at T0-1h
    r = client.get("/api/v1/plates", params={"from": "2026-05-28T09:30:00", "to": "2026-05-28T11:30"})
    assert [e["id"] for e in r.json()["data"]] == [4, 3]
    r = client.get("/api/v1/plates", params={"from": str(int(T0 - 3600)), "to": str(T0)})
    assert [e["id"] for e in r.json()["data"]] == [5, 4]
    # A bare `to` date covers the whole day; a bare `from` date starts at local midnight.
    r = client.get("/api/v1/plates", params={"from": "2026-05-28", "to": "2026-05-28"})
    assert r.json()["meta"]["total"] == 5
    r = client.get("/api/v1/plates", params={"to": "2026-05-27"})
    assert r.json()["meta"]["total"] == 0


def test_pagination(client):
    r = client.get("/api/v1/plates", params={"per_page": 2, "page": 2})
    body = r.json()
    assert [e["id"] for e in body["data"]] == [3, 2]
    assert body["meta"] == {"total": 5, "page": 2, "per_page": 2, "total_pages": 3}
    r = client.get("/api/v1/plates", params={"per_page": 2, "page": 9})
    assert r.json()["data"] == [] and r.json()["meta"]["total"] == 5


@pytest.mark.parametrize(
    "params",
    [
        {"from": "yesterday"},
        {"to": "2026-13-01"},
        {"from": "2026-05-28", "to": "2026-05-01"},
        {"per_page": 201},
        {"per_page": 0},
        {"page": 0},
        {"page": "x"},
    ],
)
def test_validation_errors_are_422_with_error_envelope(client, params):
    r = client.get("/api/v1/plates", params=params)
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "validation_error" and err["message"]


# ---- get by id ----------------------------------------------------------------------------------


def test_get_by_id_and_404(client):
    r = client.get("/api/v1/plates/2")
    assert r.status_code == 200
    assert r.json()["data"]["plate"] == "KA01MX0001"
    r = client.get("/api/v1/plates/999")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"
    assert client.get("/api/v1/plates/abc").status_code == 422
    assert client.get("/api/v1/nope").status_code == 404


# ---- CSV export ---------------------------------------------------------------------------------


def test_csv_export_content_and_route_precedence(client):
    r = client.get("/api/v1/plates/export.csv", params={"q": "1234"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    assert "attachment" in r.headers["content-disposition"]
    rows = list(csv.reader(io.StringIO(r.text)))
    assert rows[0] == [
        "id",
        "plate",
        "plate_display",
        "kind",
        "confidence",
        "votes",
        "first_seen_iso",
        "last_seen_iso",
        "crop_path",
        "snapshot_path",
        "hsrp",
    ]
    assert [row[0] for row in rows[1:]] == ["5", "4", "1"]
    assert rows[1][1:6] == ["DL3CAB1234", format_display("DL3CAB1234", "standard"), "standard", "0.9100", "4"]
    assert rows[1][7].startswith("2026-05-28T12:00:00")


def test_csv_export_validates_filters(client):
    assert client.get("/api/v1/plates/export.csv", params={"from": "nope"}).status_code == 422


# ---- images -------------------------------------------------------------------------------------


def test_image_serving_and_traversal_blocked(tmp_path, store):
    crop = np.full((40, 120, 3), 200, np.uint8)
    saved = store.add_event(ev("MH12AB1234", T0), crop=crop)
    (tmp_path / "secret.jpg").write_bytes(b"\xff\xd8 not for you")
    with client_for(tmp_path, store) as c:
        item = c.get(f"/api/v1/plates/{saved.id}").json()["data"]
        r = c.get(item["crop_url"])
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/jpeg"
        assert r.content[:2] == b"\xff\xd8"
        assert c.get("/images/does/not/exist.jpg").status_code == 404
        for bad in ("/images/../secret.jpg", "/images/%2e%2e/secret.jpg", "/images/..%2Fsecret.jpg"):
            assert c.get(bad).status_code == 404, bad
        assert c.get(f"/images/{tmp_path / 'secret.jpg'}").status_code == 404


# ---- auth ---------------------------------------------------------------------------------------


def test_auth_required_when_token_set(tmp_path, store, seeded):
    with client_for(tmp_path, store, token=TOKEN) as c:
        r = c.get("/api/v1/plates")
        assert r.status_code == 401
        assert r.json()["error"]["code"] == "unauthorized"
        assert c.get("/api/v1/plates", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert c.get("/api/v1/plates", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200
        assert c.get("/api/v1/plates", params={"token": TOKEN}).status_code == 200
        assert c.get("/api/v1/plates/export.csv").status_code == 401
        assert c.get("/api/v1/plates/1").status_code == 401
        assert c.get("/images/x.jpg").status_code == 401
        # Health and the dashboard shell stay open (monitoring + token prompt).
        assert c.get("/health").status_code in (200, 503)
        assert c.get("/").status_code == 200

        with pytest.raises(WebSocketDisconnect) as exc, c.websocket_connect("/ws/plates") as ws:
            ws.receive_json()
        assert exc.value.code == 1008
        with c.websocket_connect(f"/ws/plates?token={TOKEN}") as ws:
            assert ws.receive_json()["type"] == "hello"


def test_auth_off_by_default(client):
    assert client.get("/api/v1/plates").status_code == 200


# ---- health -------------------------------------------------------------------------------------


def test_health_no_engine_stale_and_fresh(tmp_path, store):
    with client_for(tmp_path, store, stale_after_s=30) as c:
        r = c.get("/health")
        assert r.status_code == 503 and r.json()["status"] == "no_engine"

        store.write_status(
            EngineStatus(ts=time.time() - 120, fps=4.0, camera_ok=True, frames=10, events=1, rss_mb=200)
        )
        r = c.get("/health")
        body = r.json()
        assert r.status_code == 503
        assert body["status"] == "stale" and body["engine_ok"] is False and body["engine_age_s"] >= 119

        store.write_status(
            EngineStatus(ts=time.time(), fps=4.5, camera_ok=True, frames=99, events=3, rss_mb=210)
        )
        r = c.get("/health")
        body = r.json()
        assert r.status_code == 200
        assert body["status"] == "ok" and body["ok"] is True and body["fps"] == 4.5
        assert body["disk_free_mb"] > 0

        store.write_status(
            EngineStatus(ts=time.time(), fps=0.0, camera_ok=False, frames=99, events=3, rss_mb=210)
        )
        r = c.get("/health")
        assert r.status_code == 503 and r.json()["status"] == "camera_down"


# ---- websocket ----------------------------------------------------------------------------------


def test_ws_receives_event_added_after_connect(tmp_path, store, seeded):
    with client_for(tmp_path, store) as c, c.websocket_connect("/ws/plates") as ws:
        hello = ws.receive_json()
        assert hello == {"type": "hello", "data": {"since_id": 5}}
        assert ws.receive_json()["type"] == "status"
        new = store.add_event(ev("GJ05ZZ9999", T0 + 10))
        t0 = time.monotonic()
        msg = ws.receive_json()
        assert time.monotonic() - t0 < 1.0
        assert msg["type"] == "plate"
        assert msg["data"]["id"] == new.id
        assert msg["data"]["plate_display"] == "GJ 05 ZZ 9999"


def test_ws_since_id_replays_missed_events(tmp_path, store, seeded):
    with client_for(tmp_path, store) as c, c.websocket_connect("/ws/plates?since_id=3") as ws:
        assert ws.receive_json()["data"] == {"since_id": 3}
        assert ws.receive_json()["type"] == "status"
        assert [ws.receive_json()["data"]["id"] for _ in range(2)] == [4, 5]


def test_ws_bad_since_id_rejected(client):
    with (
        pytest.raises(WebSocketDisconnect) as exc,
        client.websocket_connect("/ws/plates?since_id=abc") as ws,
    ):
        ws.receive_json()
    assert exc.value.code == 1008


# ---- dashboard + lifecycle ----------------------------------------------------------------------


def test_dashboard_and_static_assets(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "/static/app.js" in r.text
    for asset in ("/static/app.js", "/static/theme.js", "/static/style.css", "/static/logicclutch-logo.png"):
        assert client.get(asset).status_code == 200
    # nothing loaded from the internet (an SVG xmlns inside a data: URI is not a fetch)
    assert not re.search(r'(src|href)="(https?:)?//', r.text)


def test_app_opens_and_closes_its_own_store(tmp_path):
    cfg = make_cfg(tmp_path)
    app = create_app(cfg)
    with TestClient(app) as c:
        assert c.get("/api/v1/plates").json()["meta"]["total"] == 0
    with pytest.raises(sqlite3.ProgrammingError):  # closed on shutdown
        app.state.store.latest_id()


# ---- stats / info ---------------------------------------------------------------------------------


def test_store_stats_buckets_unique_and_top(tmp_path):
    s = SqliteEventStore(tmp_path / "s.db", tmp_path / "img")
    t0 = 1_780_000_000.0
    for plate, dt in [("MH12AB1234", 10), ("MH12AB1234", 3700), ("KA01MJ0001", 3800), ("MH12AB1234", 7300)]:
        s.add_event(
            PlateEvent(
                plate=plate,
                kind="standard",
                confidence=0.9,
                votes=3,
                track_id=1,
                first_seen=t0 + dt,
                last_seen=t0 + dt,
            )
        )
    st = s.stats(t0, t0 + 3 * 3600)
    assert st["total"] == 4 and st["unique"] == 2
    assert st["buckets"] == [1, 2, 1]
    assert st["avg_confidence"] == pytest.approx(0.9)
    assert st["top"][0]["plate"] == "MH12AB1234" and st["top"][0]["count"] == 3
    assert s.stats(t0 + 10 * 3600, t0 + 11 * 3600)["total"] == 0
    with pytest.raises(ValueError):
        s.stats(t0, t0)
    s.close()


def test_stats_endpoint_counts_recent_events(tmp_path, store):
    now = time.time()
    for plate in ("MH12AB1234", "MH12AB1234", "DL2CAZ2022"):
        store.add_event(
            PlateEvent(
                plate=plate,
                kind="standard",
                confidence=0.95,
                votes=3,
                track_id=1,
                first_seen=now - 60,
                last_seen=now - 60,
            )
        )
    with client_for(tmp_path, store) as c:
        r = c.get("/api/v1/stats", params={"hours": 6})
        assert r.status_code == 200
        d = r.json()["data"]
        assert len(d["buckets"]) == 6 and sum(b["count"] for b in d["buckets"]) == 3
        assert d["last_hour_total"] == 3 and d["window_unique"] == 2
        assert d["top"][0]["plate_display"] == "MH 12 AB 1234"
        assert {t["plate_display"] for t in d["top"]} >= {"DL 2C AZ 2022"}
        assert c.get("/api/v1/stats", params={"hours": 0}).status_code == 422
        info = c.get("/api/v1/info").json()["data"]
        assert info["station"] == "Gate 1" and info["min_votes"] == 3


def test_stats_buckets_align_to_local_hours(client):
    d = client.get("/api/v1/stats", params={"hours": 3}).json()["data"]
    for b in d["buckets"]:
        t = datetime.fromtimestamp(b["start"])
        assert t.minute == 0 and t.second == 0
    assert d["window"]["end"] > time.time()


# ---- stream switching (Add stream) --------------------------------------------------------------


def test_source_post_validates_and_stores(tmp_path, store):
    video = tmp_path / "gate.mp4"
    video.write_bytes(b"\x00")
    with client_for(tmp_path, store) as c:
        r = c.post("/api/v1/source", json={"source": str(video)})
        assert r.status_code == 202
        assert r.json()["data"] == {"rev": 1, "source": str(video.resolve()), "kind": "file"}
        assert store.read_source_request() == (1, str(video.resolve()))

        r = c.post("/api/v1/source", json={"source": "rtsp://admin:hunter2@192.168.1.64:554/ch1"})
        assert r.status_code == 202
        assert r.json()["data"]["source"] == "rtsp://admin:***@192.168.1.64:554/ch1"
        assert store.read_source_request()[1].endswith("hunter2@192.168.1.64:554/ch1")  # engine needs it

        r = c.post("/api/v1/source", json={"source": ""})
        assert r.status_code == 202 and r.json()["data"]["kind"] == "webcam"  # config default "0"

        for bad in ["/no/such/file.mp4", "http://x/y", "relative.mp4"]:
            r = c.post("/api/v1/source", json={"source": bad})
            assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"
        assert c.post("/api/v1/source", json={"source": "0", "extra": 1}).status_code == 422
        assert (
            c.post(
                "/api/v1/source",
                content=b"source=0",
                headers={"content-type": "application/x-www-form-urlencoded"},
            ).status_code
            == 422
        )
        assert store.read_source_request()[0] == 3  # nothing stored for rejected requests


def test_source_post_refuses_cross_site(tmp_path, store):
    with client_for(tmp_path, store) as c:
        r = c.post("/api/v1/source", json={"source": "0"}, headers={"origin": "http://evil.example"})
        assert r.status_code == 403
        r = c.post("/api/v1/source", json={"source": "0"}, headers={"origin": "http://testserver"})
        assert r.status_code == 202


def test_source_requires_token_when_set(tmp_path, store):
    with client_for(tmp_path, store, token=TOKEN) as c:
        assert c.post("/api/v1/source", json={"source": "0"}).status_code == 401
        assert c.get("/api/v1/source").status_code == 401
        h = {"authorization": f"Bearer {TOKEN}"}
        assert c.post("/api/v1/source", json={"source": "0"}, headers=h).status_code == 202


def test_source_view_pending_until_engine_applies(tmp_path, store):
    with client_for(tmp_path, store) as c:
        d = c.get("/api/v1/source").json()["data"]
        assert d["current"] is None and d["pending"] is False and d["default"] == "0"
        c.post("/api/v1/source", json={"source": "rtsp://u:pw@cam/x"})
        d = c.get("/api/v1/source").json()["data"]
        assert d["pending"] is True and d["engine_running"] is False
        assert d["requested"] == "rtsp://u:***@cam/x"
        store.write_status(
            EngineStatus(
                ts=time.time(),
                fps=5,
                camera_ok=True,
                frames=1,
                events=0,
                rss_mb=1,
                source="rtsp://u:***@cam/x",
                source_state="running",
                source_rev=1,
            )
        )
        d = c.get("/api/v1/source").json()["data"]
        assert d["pending"] is False and d["engine_running"] is True
        assert d["current"] == "rtsp://u:***@cam/x"
        assert d["current_kind"] == "rtsp" and d["state"] == "running"


def test_health_reports_ended_video_without_the_address(tmp_path, store):
    store.write_status(
        EngineStatus(
            ts=time.time(),
            fps=0,
            camera_ok=False,
            frames=9,
            events=1,
            rss_mb=1,
            source="/secret/place/gate.mp4",
            source_state="ended",
            source_rev=4,
        )
    )
    with client_for(tmp_path, store) as c:
        r = c.get("/health")
        body = r.json()
        assert r.status_code == 503 and body["status"] == "source_ended"
        assert body["source_state"] == "ended" and body["source_rev"] == 4
        assert "secret" not in r.text


# ---- live view ----------------------------------------------------------------------------------------


def test_live_jpg_404_then_200_no_store(tmp_path, store):
    with client_for(tmp_path, store) as c:
        r = c.get("/api/v1/live.jpg")
        assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
        assert c.get("/api/v1/source").json()["data"]["preview_fresh"] is False
        before = time.time()
        store.write_preview(np.full((36, 64, 3), 90, np.uint8))
        r = c.get("/api/v1/live.jpg")
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
        assert r.headers["cache-control"] == "no-store"
        assert float(r.headers["x-captured-at"]) >= before - 2
        assert float(r.headers["x-preview-age"]) < 5
        assert r.content[:2] == b"\xff\xd8"
        d = c.get("/api/v1/source").json()["data"]
        assert d["preview_fresh"] is True and d["preview_age_s"] < 5 and d["preview_interval_s"] == 1.0
        h = c.get("/health").json()
        assert h["preview_fresh"] is True and h["preview_ts"] is not None
        assert c.get("/api/v1/info").json()["data"]["preview_interval_s"] == 1.0


def test_live_jpg_old_picture_is_not_fresh(tmp_path, store):
    import os

    store.write_preview(np.zeros((8, 8, 3), np.uint8))
    old = time.time() - 120
    os.utime(store.preview_path(), (old, old))
    with client_for(tmp_path, store) as c:
        r = c.get("/api/v1/live.jpg")
        assert r.status_code == 200 and float(r.headers["x-preview-age"]) >= 119
        d = c.get("/api/v1/source").json()["data"]
        assert d["preview_fresh"] is False and d["preview_age_s"] >= 119


def test_live_jpg_requires_token_when_set(tmp_path, store):
    store.write_preview(np.zeros((8, 8, 3), np.uint8))
    with client_for(tmp_path, store, token=TOKEN) as c:
        assert c.get("/api/v1/live.jpg").status_code == 401
        assert c.get("/api/v1/live.jpg", params={"token": "wrong"}).status_code == 401
        assert c.get("/api/v1/live.jpg", params={"token": TOKEN}).status_code == 200
        r = c.get("/api/v1/live.jpg", headers={"authorization": f"Bearer {TOKEN}"})
        assert r.status_code == 200 and r.headers["cache-control"] == "no-store"
        assert c.get("/images/../live.jpg", params={"token": TOKEN}).status_code == 404  # not an image


def test_read_only_dashboard_refuses_stream_changes(tmp_path, store):
    cfg = AppConfig.model_validate(
        {
            "storage": {"db_path": str(tmp_path / "db" / "anpr.db"), "image_dir": str(tmp_path / "images")},
            "web": {"read_only": True},
        }
    )
    with TestClient(create_app(cfg, store)) as c:
        r = c.post("/api/v1/source", json={"source": "0"})
        assert r.status_code == 403 and "view-only" in r.json()["error"]["message"]
        assert c.get("/api/v1/info").json()["data"]["read_only"] is True
        assert store.read_source_request() is None


def test_hsrp_result_in_api_and_csv(tmp_path, store):
    import dataclasses

    store.add_event(dataclasses.replace(ev("MH12AB1234", 1_790_000_000.0), hsrp="hsrp"))
    store.add_event(ev("KA01MX0001", 1_790_000_100.0))  # saved before the check existed
    with client_for(tmp_path, store) as c:
        data = c.get("/api/v1/plates").json()["data"]
        assert [(d["plate"], d["hsrp"]) for d in data] == [("KA01MX0001", None), ("MH12AB1234", "hsrp")]
        rows = list(csv.reader(io.StringIO(c.get("/api/v1/plates/export.csv").text)))
        assert [r[-1] for r in rows[1:]] == ["", "hsrp"]


def test_info_shows_the_licence(tmp_path, store):
    with client_for(tmp_path, store) as c:
        assert c.get("/api/v1/info").json()["data"]["licence"] is None  # engine has not checked yet
        store.write_setting(
            "licence",
            {
                "ok": True,
                "error": None,
                "customer": "Acme",
                "expires": "2027-09-30",
                "days_left": 12,
                "licence_id": "LIC-1",
                "device_id": "a1b2c3d4",
                "checked_at": 1.0,
            },
        )
        lic = c.get("/api/v1/info").json()["data"]["licence"]
        assert lic == {
            "ok": True,
            "error": None,
            "customer": "Acme",
            "expires": "2027-09-30",
            "days_left": 12,
            "licence_id": "LIC-1",
        }
