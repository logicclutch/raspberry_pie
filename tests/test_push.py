"""Sending plates to the client's API: settings, JSON body, retries, and the dashboard endpoints."""

import base64
import json
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
from fastapi.testclient import TestClient

from anpr import push
from anpr.config import AppConfig
from anpr.push import Pusher, PushSettings, SendResult
from anpr.storage import SqliteEventStore
from anpr.types import EngineStatus, PlateEvent
from anpr.web.app import create_app

T0 = datetime(2026, 10, 1, 12, 0, 0).timestamp()
URL = "https://client.example.com/api/anpr/events"


def ev(plate: str, ts: float = T0, hsrp: str | None = "hsrp") -> PlateEvent:
    return PlateEvent(
        plate=plate,
        kind="standard",
        confidence=0.97,
        votes=4,
        track_id=1,
        first_seen=ts - 1,
        last_seen=ts,
        hsrp=hsrp,
    )


@pytest.fixture
def store(tmp_path):
    s = SqliteEventStore(tmp_path / "anpr.db", tmp_path / "images", snapshot_width=64)
    yield s
    s.close()


class FakePost:
    """Records requests; answers with the queued results (then with the last one)."""

    def __init__(self, *results: SendResult) -> None:
        self.results = list(results) or [SendResult(True, 200, False, "accepted (HTTP 200)")]
        self.calls: list[dict] = []

    def __call__(self, url, header_name, header_value, body, timeout=10.0):  # noqa: ANN001, ARG002
        self.calls.append({"url": url, "header": (header_name, header_value), "body": body})
        return self.results.pop(0) if len(self.results) > 1 else self.results[0]

    def plates(self) -> list[list[str]]:
        return [[e["plate"] for e in c["body"]["events"]] for c in self.calls]


OK = SendResult(True, 200, False, "accepted (HTTP 200)")
DOWN = SendResult(False, None, True, "cannot reach the server: refused")
BAD = SendResult(False, 422, False, "the server refused the data (HTTP 422)")


def switch_on(store, **kw) -> PushSettings:
    """What the dashboard's Save does when sending is switched on."""
    old = push.load_settings(store)
    s = PushSettings(
        enabled=True,
        url=URL,
        header_value="Bearer k1",
        device_id="PI-012",
        start_rev=old.start_rev + 1,
        start_after_id=store.latest_id(),
        **kw,
    )
    push.save_settings(store, s)
    return s


def drain(p: Pusher, rounds: int = 20) -> None:
    for _ in range(rounds):
        if p.step() != 0.0:
            break


# ---- sender -------------------------------------------------------------------------------------


def test_nothing_is_sent_while_switched_off(store):
    store.add_event(ev("MH12AB1234"))
    post = FakePost()
    p = Pusher(store, "Gate 1", post=post)
    drain(p)
    assert post.calls == []


def test_only_plates_after_switching_on_are_sent_in_order(store):
    store.add_event(ev("OLD0001"))  # before switching on: not sent
    switch_on(store)
    for i in range(3):
        store.add_event(ev(f"MH12AB123{i}", T0 + i))
    post = FakePost(OK)
    p = Pusher(store, "Gate 1", post=post, batch=2)
    drain(p)
    assert post.plates() == [["MH12AB1230", "MH12AB1231"], ["MH12AB1232"]]
    st = push.load_state(store)
    assert st.cursor == store.latest_id() and st.sent_total == 3 and st.last_error is None
    assert post.calls[0]["url"] == URL and post.calls[0]["header"] == ("Authorization", "Bearer k1")


def test_body_follows_the_documented_format(store):
    switch_on(store)
    crop = np.full((20, 60, 3), 200, np.uint8)
    saved = store.add_event(ev("MH12AB1234"), crop=crop, snapshot=np.zeros((48, 64, 3), np.uint8))
    post = FakePost(OK)
    drain(Pusher(store, "Gate 12", post=post))
    body = post.calls[0]["body"]
    assert body["schema_version"] == "1.0"
    assert body["device"] == {
        "device_id": "PI-012",
        "station_name": "Gate 12",
        "software_version": push.__version__,
    }
    e = body["events"][0]
    assert e["event_id"] == f"PI-012-{saved.id:06d}"
    assert (e["plate"], e["plate_display"], e["plate_type"], e["hsrp"]) == (
        "MH12AB1234",
        "MH 12 AB 1234",
        "standard",
        "hsrp",
    )
    assert e["confidence"] == 0.97 and e["votes"] == 4
    assert e["detected_at"] == datetime.fromtimestamp(T0).astimezone().isoformat(timespec="seconds")
    assert base64.b64decode(e["images"]["plate_crop_jpeg_base64"])[:2] == b"\xff\xd8"  # a JPEG
    assert base64.b64decode(e["images"]["full_snapshot_jpeg_base64"])[:2] == b"\xff\xd8"
    json.dumps(body)  # serialisable


def test_photos_can_be_left_out(store):
    switch_on(store, include_images=False)
    store.add_event(ev("MH12AB1234"), crop=np.full((20, 60, 3), 200, np.uint8))
    post = FakePost(OK)
    drain(Pusher(store, "Gate 1", post=post))
    assert post.calls[0]["body"]["events"][0]["images"] == {
        "plate_crop_jpeg_base64": None,
        "full_snapshot_jpeg_base64": None,
    }


def test_server_down_keeps_the_plates_and_backs_off(store):
    switch_on(store)
    store.add_event(ev("MH12AB1234"))
    post = FakePost(DOWN, DOWN, OK)
    p = Pusher(store, "Gate 1", post=post)
    assert p.step() == 5.0
    assert p.step() == 10.0  # growing pause
    st = push.load_state(store)
    assert st.sent_total == 0 and "cannot reach" in st.last_error and store.count_after(st.cursor) == 1
    p.step()  # back up: sent, error cleared
    st = push.load_state(store)
    assert st.sent_total == 1 and st.last_error is None and store.count_after(st.cursor) == 0
    assert post.plates() == [["MH12AB1234"]] * 3


def test_backoff_is_capped(store):
    switch_on(store)
    store.add_event(ev("MH12AB1234"))
    p = Pusher(store, "Gate 1", post=FakePost(DOWN))
    waits = [p.step() for _ in range(12)]
    assert max(waits) == push.MAX_BACKOFF_S


def test_a_refused_plate_is_found_and_skipped_the_rest_go_through(store):
    switch_on(store)
    for p_ in ("MH12AB0001", "BADPLATE", "MH12AB0003"):
        store.add_event(ev(p_))

    def post(url, hn, hv, body, timeout=10.0):  # noqa: ANN001, ARG001
        plates = [e["plate"] for e in body["events"]]
        calls.append(plates)
        return BAD if "BADPLATE" in plates else OK

    calls: list[list[str]] = []
    drain(Pusher(store, "Gate 1", post=post, batch=25))
    assert calls[0] == ["MH12AB0001", "BADPLATE", "MH12AB0003"]  # whole batch refused
    assert calls[1:] == [["MH12AB0001"], ["BADPLATE"], ["MH12AB0003"]]  # then one by one
    st = push.load_state(store)
    assert st.sent_total == 2 and st.skipped_total == 1  # (the error is cleared by the next good send)
    assert store.count_after(st.cursor) == 0


def test_changing_the_address_starts_from_now_but_a_new_key_keeps_the_backlog(store):
    switch_on(store)
    store.add_event(ev("MH12AB0001"))
    post = FakePost(DOWN)
    p = Pusher(store, "Gate 1", post=post)
    p.step()
    # New key, same address: the waiting plate is still sent.
    s = push.load_settings(store)
    push.save_settings(store, PushSettings(**{**s.to_dict(), "header_value": "Bearer k2"}))
    post.results = [OK]
    p.step()
    assert post.calls[-1]["header"] == ("Authorization", "Bearer k2")
    assert post.plates()[-1] == ["MH12AB0001"]
    # New address (the dashboard bumps start_rev): plates from before are not sent to it.
    store.add_event(ev("MH12AB0002"))
    s = push.load_settings(store)
    push.save_settings(
        store,
        PushSettings(
            **{
                **s.to_dict(),
                "url": "https://new.example.com/x",
                "start_rev": s.start_rev + 1,
                "start_after_id": store.latest_id(),
            }
        ),
    )
    n = len(post.calls)
    drain(p)
    assert len(post.calls) == n
    store.add_event(ev("MH12AB0003"))
    drain(p)
    assert post.calls[-1]["url"] == "https://new.example.com/x" and post.plates()[-1] == ["MH12AB0003"]


def test_the_sender_continues_after_a_restart(store):
    switch_on(store)
    store.add_event(ev("MH12AB0001"))
    drain(Pusher(store, "Gate 1", post=FakePost(OK)))
    store.add_event(ev("MH12AB0002"))
    post = FakePost(OK)
    drain(Pusher(store, "Gate 1", post=post))  # a new engine process
    assert post.plates() == [["MH12AB0002"]]


def test_settings_saved_while_waiting_are_tried_right_away(store):
    switch_on(store)
    store.add_event(ev("MH12AB0001"))
    post = FakePost(DOWN, DOWN, OK)
    stop = threading.Event()
    p = Pusher(store, "Gate 1", post=post, poll_s=0.02)
    t = threading.Thread(target=p.run, args=(stop,))
    t.start()
    try:
        deadline = time.time() + 3
        while len(post.calls) < 2 and time.time() < deadline:  # second call is 5 s away...
            time.sleep(0.02)
            if len(post.calls) == 1:
                s = push.load_settings(store)  # ...unless the settings change
                push.save_settings(store, PushSettings(**{**s.to_dict(), "header_value": "Bearer new"}))
        assert len(post.calls) >= 2
    finally:
        stop.set()
        t.join(5)
    assert not t.is_alive()


def test_bad_saved_values_fall_back_to_defaults():
    s = PushSettings.from_dict(
        {"enabled": "yes", "url": 5, "start_rev": True, "include_images": False, "x": 1}
    )
    assert s == PushSettings(include_images=False)
    assert PushSettings.from_dict(None) == PushSettings()


@pytest.mark.parametrize(
    ("url", "ok"),
    [
        ("https://a.example.com/api", True),
        ("http://192.168.1.20:8080/in", True),
        ("ftp://a.example.com", False),
        ("a.example.com/api", False),
        ("https://", False),
        ("https://a.example.com/a b", False),
    ],
)
def test_url_check(url, ok):
    if ok:
        assert push.clean_url(url) == url
    else:
        with pytest.raises(push.PushError):
            push.clean_url(url)


def test_key_and_header_checks():
    with pytest.raises(push.PushError):
        push.clean_header_value("Bearer a\r\nX-Evil: 1")
    with pytest.raises(push.PushError):
        push.clean_header_name("Bad Header")
    assert push.clean_header_name("X-API-Key") == "X-API-Key"
    assert push.secret_hint("Bearer abcdefgh1234") == "…1234"
    assert push.secret_hint("short") == "…"  # short keys: nothing shown


# ---- real HTTP ----------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_proxy_for_local(monkeypatch):
    """The test servers are local: never send them through a proxy set in the environment."""
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")


class _Server:
    def __init__(
        self, status: int = 200, reply: bytes = b'{"accepted":[],"rejected":[]}', headers=None
    ) -> None:
        self.got: list[tuple[dict, dict]] = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                n = int(self.headers.get("Content-Length", 0))
                outer.got.append((self.headers, json.loads(self.rfile.read(n))))  # case-insensitive
                self.send_response(status)
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(reply)

            def log_message(self, *a):  # noqa: ANN002
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/api/anpr/events"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.mark.parametrize(
    ("status", "ok", "retry"),
    [
        (200, True, False),
        (201, True, False),
        (401, False, True),
        (404, False, True),
        (500, False, True),
        (422, False, False),
    ],
)
def test_post_json_status_handling(status, ok, retry):
    srv = _Server(status)
    try:
        r = push.post_json(srv.url, "X-API-Key", "k-123", {"events": []})
    finally:
        srv.close()
    assert (r.ok, r.status, r.retry) == (ok, status, retry)
    hdrs, body = srv.got[0]
    assert (
        hdrs["X-API-Key"] == "k-123" and hdrs["Content-Type"] == "application/json" and body == {"events": []}
    )


def test_post_json_reports_rejected_and_redirects():
    srv = _Server(200, b'{"accepted":["a"],"rejected":[{"event_id":"b","reason":"duplicate"}]}')
    try:
        r = push.post_json(srv.url, "", "", {})
    finally:
        srv.close()
    assert r.ok and r.rejected == (("b", "duplicate"),)
    srv = _Server(302, b"", {"Location": "https://elsewhere.example.com/in"})
    try:
        r = push.post_json(srv.url, "", "", {})
    finally:
        srv.close()
    assert not r.ok and r.retry and "elsewhere.example.com" in r.message


def test_post_json_no_server():
    r = push.post_json("http://127.0.0.1:9/none", "", "", {}, timeout=2)
    assert not r.ok and r.retry and r.status is None


# ---- dashboard endpoints ------------------------------------------------------------------------


def make_client(tmp_path, store, read_only=False) -> TestClient:
    cfg = AppConfig.model_validate(
        {
            "storage": {"db_path": str(tmp_path / "anpr.db"), "image_dir": str(tmp_path / "images")},
            "web": {"read_only": read_only, "station_name": "Gate 12"},
        }
    )
    return TestClient(create_app(cfg, store))


FORM = {
    "enabled": True,
    "url": URL,
    "header_name": "Authorization",
    "header_value": "Bearer secret-key-1234",
    "device_id": "PI-012",
    "include_images": True,
}


def test_settings_save_and_read_back_without_the_key(tmp_path, store):
    store.add_event(ev("MH12AB0001"))
    with make_client(tmp_path, store) as c:
        d = c.get("/api/v1/push").json()["data"]
        assert d["settings"]["enabled"] is False and d["settings"]["header_value_set"] is False
        r = c.put("/api/v1/push", json=FORM)
        assert r.status_code == 200
        d = r.json()["data"]
        assert "secret-key-1234" not in r.text
        assert d["settings"] == {
            "enabled": True,
            "url": URL,
            "header_name": "Authorization",
            "header_value_set": True,
            "header_value_hint": "…1234",
            "device_id": "PI-012",
            "device_id_default": push.default_device_id(),
            "include_images": True,
        }
        assert d["status"]["waiting"] == 0  # the plate from before switching on is not sent
        s = push.load_settings(store)
        assert s.header_value == "Bearer secret-key-1234" and s.start_rev == 1 and s.start_after_id == 1
        # Saving again with the key field empty keeps the key and does not restart from now.
        store.add_event(ev("MH12AB0002"))
        r = c.put("/api/v1/push", json={**FORM, "header_value": None, "include_images": False})
        s = push.load_settings(store)
        assert s.header_value == "Bearer secret-key-1234" and s.start_rev == 1 and s.include_images is False
        assert r.json()["data"]["status"]["waiting"] == 1
        # A new address restarts from now.
        r = c.put("/api/v1/push", json={**FORM, "url": "https://b.example.com/in"})
        assert push.load_settings(store).start_rev == 2 and r.json()["data"]["status"]["waiting"] == 0
        # No header name = no key.
        c.put("/api/v1/push", json={**FORM, "header_name": ""})
        assert push.load_settings(store).header_value == ""


@pytest.mark.parametrize(
    ("change", "msg"),
    [
        ({"url": "client.example.com"}, "https://"),
        ({"url": ""}, "address"),
        ({"header_name": "Bad Name"}, "header name"),
        ({"header_value": "a\nb"}, "one line"),
        ({"device_id": "PI 12"}, "device ID"),
    ],
)
def test_settings_are_checked(tmp_path, store, change, msg):
    with make_client(tmp_path, store) as c:
        r = c.put("/api/v1/push", json={**FORM, **change})
    assert r.status_code == 422 and msg in r.json()["error"]["message"]
    assert push.load_settings(store) == PushSettings()


def test_view_only_dashboard_cannot_see_or_change_settings(tmp_path, store):
    push.save_settings(store, PushSettings(enabled=True, url=URL, header_value="Bearer x"))
    with make_client(tmp_path, store, read_only=True) as c:
        assert c.get("/api/v1/push").status_code == 403
        assert c.put("/api/v1/push", json=FORM).status_code == 403
        assert c.post("/api/v1/push/test", json=FORM).status_code == 403


def test_cross_site_save_is_refused(tmp_path, store):
    with make_client(tmp_path, store) as c:
        r = c.put("/api/v1/push", json=FORM, headers={"Origin": "https://evil.example.com"})
    assert r.status_code == 403


def test_send_test_posts_a_marked_sample_with_the_form_values(tmp_path, store):
    srv = _Server(201, b'{"ok":true}')
    try:
        with make_client(tmp_path, store) as c:
            r = c.post(
                "/api/v1/push/test",
                json={**FORM, "url": srv.url, "header_name": "X-API-Key", "header_value": "k9"},
            )
    finally:
        srv.close()
    d = r.json()["data"]
    assert d["ok"] is True and d["status"] == 201 and d["reply"] == '{"ok":true}'
    hdrs, body = srv.got[0]
    assert hdrs["X-API-Key"] == "k9"
    assert body["test"] is True and body["device"]["station_name"] == "Gate 12"
    assert body["events"][0]["event_id"].startswith("PI-012-TEST-")
    assert push.load_settings(store) == PushSettings()  # a test saves nothing


def test_send_test_uses_the_saved_key_when_the_field_is_empty(tmp_path, store):
    push.save_settings(store, PushSettings(url=URL, header_value="Bearer saved"))
    srv = _Server(500, b"boom")
    try:
        with make_client(tmp_path, store) as c:
            d = c.post("/api/v1/push/test", json={**FORM, "url": srv.url, "header_value": None}).json()[
                "data"
            ]
    finally:
        srv.close()
    assert d["ok"] is False and d["status"] == 500 and "server error" in d["message"] and d["reply"] == "boom"
    assert srv.got[0][0]["Authorization"] == "Bearer saved"


def test_status_shows_errors_and_engine_state(tmp_path, store):
    with make_client(tmp_path, store) as c:
        c.put("/api/v1/push", json=FORM)
        store.add_event(ev("MH12AB0001"))
        Pusher(store, "Gate 12", post=FakePost(DOWN)).step()
        st = c.get("/api/v1/push").json()["data"]["status"]
        assert st["engine_running"] is False
        assert st["waiting"] == 1 and "cannot reach" in st["last_error"]
        store.write_status(
            EngineStatus(ts=time.time(), fps=5.0, frames=1, events=1, camera_ok=True, rss_mb=100.0)
        )
        assert c.get("/api/v1/push").json()["data"]["status"]["engine_running"] is True


@pytest.mark.parametrize(
    ("reply", "ok"),
    [
        (b'{"errorCode":"1","errorMessage":"success","data":"null"}', True),  # IntelliParks success
        (b'{"errorCode":"2","errorMessage":"error","data":"null"}', False),  # IntelliParks error, HTTP 200
        (b'{"ErrorCode":2,"ErrorMessage":"Invalid plate"}', False),
        (b'{"success":false,"message":"duplicate"}', False),
        (b'{"status":"failed"}', False),
        (b'{"accepted":["a"],"rejected":[]}', True),
        (b"OK", True),
        (b"", True),
    ],
)
def test_http_200_with_an_error_in_the_body_is_not_counted_as_sent(reply, ok):
    srv = _Server(200, reply)
    try:
        r = push.post_json(srv.url, "", "", {"events": []})
    finally:
        srv.close()
    assert r.ok is ok
    if not ok:
        assert r.retry and r.status == 200 and "answered an error" in r.message


def test_intelliparks_error_keeps_the_plate_waiting(store):
    switch_on(store)
    store.add_event(ev("MH12AB1234"))
    bad = SendResult(False, 200, True, "the server answered an error: error (errorCode 2)")
    p = Pusher(store, "Gate 1", post=FakePost(bad, OK))
    assert p.step() == 5.0
    st = push.load_state(store)
    assert st.sent_total == 0 and "errorCode 2" in st.last_error and store.count_after(st.cursor) == 1
    p.step()
    assert push.load_state(store).sent_total == 1


def test_last_api_reply_is_kept_and_shown_in_settings(tmp_path, store):
    switch_on(store)
    store.add_event(ev("MH12AB1234"))
    reply = '{"errorCode":"1","errorMessage":"success","data":"null"}'
    Pusher(store, "Gate 1", post=FakePost(SendResult(True, 200, False, "accepted (HTTP 200)", reply))).step()
    st = push.load_state(store)
    assert st.last_reply == reply and st.last_reply_status == 200 and st.last_reply_ts is not None
    with make_client(tmp_path, store) as c:
        s = c.get("/api/v1/push").json()["data"]["status"]
    assert s["last_reply"] == reply and s["last_reply_status"] == 200
    store.add_event(ev("MH12AB1235"))
    err = '{"errorCode":"2","errorMessage":"error","data":"null"}'
    Pusher(store, "Gate 1", post=FakePost(SendResult(False, 200, True, "x", err))).step()
    assert push.load_state(store).last_reply == err  # failures are shown too


def test_config_push_section_is_the_source_of_truth(store):
    """config.yaml `push:` is applied on start ("file wins"): it sets the device, re-applying the same
    config does not restart the stream, changing the address does, and it overrides a dashboard edit."""
    from anpr.config import PushConfig

    url = "https://demo.intelliparks.in/api/deviceAPI/ANPRLogInsert"
    push.apply_push_config(store, PushConfig(enabled=True, url=url, header_name="", header_value=""))
    s = push.load_settings(store)
    assert s.enabled and s.url == url and s.start_rev == 1

    push.apply_push_config(store, PushConfig(enabled=True, url=url, header_name="", header_value=""))
    assert push.load_settings(store).start_rev == 1  # unchanged config => no restart of the stream

    push.apply_push_config(store, PushConfig(enabled=True, url="https://new.example/api"))
    s2 = push.load_settings(store)
    assert s2.url == "https://new.example/api" and s2.start_rev == 2  # new address restarts

    # a dashboard edit is overridden by the file on the next start
    push.save_settings(store, push.PushSettings(**{**s2.to_dict(), "url": "https://dash.example/api"}))
    push.apply_push_config(store, PushConfig(enabled=True, url="https://new.example/api"))
    assert push.load_settings(store).url == "https://new.example/api"


# ---- inline push (ingest respond_with_plate) shares the Pusher with the background sender ------------


class TimedPost(FakePost):
    """FakePost that also records the timeout each request was given."""

    def __init__(self, *results: SendResult) -> None:
        super().__init__(*results)
        self.timeouts: list[float] = []

    def __call__(self, url, header_name, header_value, body, timeout=10.0):  # noqa: ANN001
        self.timeouts.append(timeout)
        return super().__call__(url, header_name, header_value, body, timeout)


def test_push_now_delivers_this_plate_with_a_short_timeout(store):
    switch_on(store)
    e = store.add_event(ev("MH12AB0001"))
    post = TimedPost(OK)
    out = Pusher(store, "Gate 1", post=post).push_now([e.id])
    assert out.delivered and not out.queued and out.result == OK
    assert post.plates() == [["MH12AB0001"]]
    assert post.timeouts == [push.INLINE_TIMEOUT_S]


def test_push_now_api_down_queues_and_backs_off_without_hammering(store):
    switch_on(store)
    p = Pusher(store, "Gate 1", post=FakePost(DOWN))
    e1 = store.add_event(ev("MH12AB0001"))
    out1 = p.push_now([e1.id])
    assert not out1.delivered and out1.queued and "cannot reach" in out1.message
    e2 = store.add_event(ev("MH12AB0002"))
    out2 = p.push_now([e2.id])  # the API failed a moment ago: no new request, answered at once
    assert not out2.delivered and out2.queued and out2.message.startswith("queued:")
    assert out2.result is None
    assert len(p._post.calls) == 1
    assert push.load_state(store).cursor == 0  # nothing lost: both plates still pending


def test_push_now_sends_a_small_backlog_together_with_this_plate(store):
    switch_on(store)
    for i in range(3):
        store.add_event(ev(f"MH12AB000{i}"))
    e = store.add_event(ev("MH12AB0009"))
    post = FakePost(OK)
    out = Pusher(store, "Gate 1", post=post).push_now([e.id])
    assert out.delivered
    assert post.plates() == [["MH12AB0000", "MH12AB0001", "MH12AB0002", "MH12AB0009"]]


def test_push_now_does_not_report_an_older_batch_as_this_plate(store):
    """Backlog bigger than one batch: the request would carry older plates only, so it is left to the
    background sender (in order) and this plate is reported as queued, not as sent."""
    switch_on(store)
    for i in range(3):
        store.add_event(ev(f"MH12AB000{i}"))
    e = store.add_event(ev("MH12AB0009"))
    post = FakePost(OK)
    out = Pusher(store, "Gate 1", post=post, batch=2).push_now([e.id])
    assert not out.delivered and out.queued
    assert post.calls == []


def test_push_now_after_the_background_sender_already_sent_it(store):
    switch_on(store)
    e = store.add_event(ev("MH12AB0001"))
    post = FakePost(SendResult(True, 200, False, "accepted (HTTP 200)", '{"errorCode":"1"}'))
    p = Pusher(store, "Gate 1", post=post)
    p.step()
    out = p.push_now([e.id])
    assert out.delivered and not out.queued
    assert len(post.calls) == 1  # never sent twice
    assert out.result.reply == '{"errorCode":"1"}'


def test_push_now_waits_only_briefly_when_the_sender_is_busy(store):
    switch_on(store)
    e = store.add_event(ev("MH12AB0001"))
    post = FakePost(OK)
    p = Pusher(store, "Gate 1", post=post)
    p.lock.acquire()
    try:
        t0 = time.monotonic()
        out = p.push_now([e.id], lock_wait=0.05)
        assert time.monotonic() - t0 < 1.0
    finally:
        p.lock.release()
    assert not out.delivered and out.queued and post.calls == []


def test_push_now_nothing_to_send_and_switched_off(store):
    p = Pusher(store, "Gate 1", post=FakePost(OK))
    assert p.push_now([]).message == "nothing new to send"
    e = store.add_event(ev("MH12AB0001"))
    out = p.push_now([e.id])  # sending is off
    assert not out.delivered and not out.queued and p._post.calls == []


def test_failed_inline_push_holds_back_the_background_sender(store):
    switch_on(store)
    p = Pusher(store, "Gate 1", post=FakePost(DOWN))
    e = store.add_event(ev("MH12AB0001"))
    p.push_now([e.id])
    assert p._next_at > time.time() + 1  # the background round waits out the backoff too


def test_background_sender_and_inline_push_never_send_a_plate_twice(store):
    switch_on(store)
    post = FakePost(OK)
    p = Pusher(store, "Gate 1", post=post, poll_s=0.005)
    stop = threading.Event()
    t = threading.Thread(target=p.run, args=(stop,))
    t.start()
    plates = [f"MH12AB{i:04d}" for i in range(60)]
    try:
        for plate in plates:
            e = store.add_event(ev(plate))
            p.push_now([e.id])
        deadline = time.time() + 5
        while push.load_state(store).cursor < store.latest_id() and time.time() < deadline:
            time.sleep(0.01)
    finally:
        stop.set()
        t.join(5)
    sent = [x for batch in post.plates() for x in batch]
    assert sorted(sent) == sorted(plates)  # every plate exactly once
    assert sent == plates  # and in order


def test_push_now_plate_listed_as_rejected_is_not_reported_as_accepted(store):
    s = switch_on(store)
    e = store.add_event(ev("MH12AB0001"))
    key = f"{s.device}-{e.id:06d}"
    res = SendResult(True, 200, False, "accepted (HTTP 200)", "{}", ((key, "empty plate"),))
    out = Pusher(store, "Gate 1", post=FakePost(res)).push_now([e.id])
    assert not out.delivered and not out.queued and "empty plate" in out.message


def test_push_now_plate_refused_for_good_is_not_reported_as_queued(store):
    switch_on(store)
    e = store.add_event(ev("MH12AB0001"))
    out = Pusher(store, "Gate 1", post=FakePost(BAD)).push_now([e.id])
    assert not out.delivered and not out.queued and "for good" in out.message
    assert push.load_state(store).skipped_total == 1


def test_push_now_after_the_background_sender_refused_it(store):
    switch_on(store)
    e = store.add_event(ev("MH12AB0001"))
    p = Pusher(store, "Gate 1", post=FakePost(BAD))
    p.step()  # the background sender got it first and the API refused it
    out = p.push_now([e.id])
    assert not out.delivered and not out.queued and out.result.status == 422


def test_a_stale_settings_snapshot_never_rewinds_the_cursor(store):
    switch_on(store)
    for i in range(5):
        store.add_event(ev(f"MH12AB000{i}"))
    post = FakePost(OK)
    p = Pusher(store, "Gate 1", post=post)
    drain(p)
    old = push.load_settings(store)  # e.g. the background thread read this, then waited for the lock
    switch_on(store)  # dashboard saves a new address (new rev, starts after plate 5)
    e = store.add_event(ev("MH12AB0009"))
    assert p.push_now([e.id]).delivered
    p.step(old)  # the stale snapshot must not rewind the cursor to the old stream
    sent = [x for batch in post.plates() for x in batch]
    assert sent == ["MH12AB0000", "MH12AB0001", "MH12AB0002", "MH12AB0003", "MH12AB0004", "MH12AB0009"]
    assert push.load_state(store).start_rev == push.load_settings(store).start_rev
