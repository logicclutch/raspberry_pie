"""Ingest/push mode: an ANPR camera POSTs several images per vehicle. We group them and VOTE across the
group (full pipeline) so each vehicle yields ONE best result, stored for the sender to forward."""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request

import cv2
import numpy as np
import pytest

from anpr import ingest
from anpr.config import AppConfig, IngestConfig, ValidationConfig
from anpr.engine import Engine
from anpr.storage import SqliteEventStore
from anpr.types import Box, OcrResult, PlateEvent
from anpr.validator import PlateValidator


def _jpeg_bytes(w=220, h=90) -> bytes:
    img = np.full((h, w, 3), 200, np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _jpeg_b64(w=220, h=90) -> str:
    return base64.b64encode(_jpeg_bytes(w, h)).decode()


class FakeDetector:
    def detect(self, frame):
        return [Box(10, 10, 200, 80, 0.95)]  # one constant, "close-up" plate box


class SeqOcr:
    """Returns the given plate texts in order, then repeats the last (one high-confidence read each)."""

    def __init__(self, texts):
        self._texts = list(texts)
        self._i = 0

    def read(self, crop):
        t = self._texts[min(self._i, len(self._texts) - 1)]
        self._i += 1
        return OcrResult(t, tuple([0.97] * len(t)))


@pytest.fixture(autouse=True)
def _easy_crop(monkeypatch):
    # The fake OCR supplies the text; skip the real crop-quality gate so the fakes flow through.
    monkeypatch.setattr(
        "anpr.preprocess.prepare_plate", lambda frame, box, cfg: np.zeros((64, 128, 3), np.uint8)
    )


@pytest.fixture
def store(tmp_path):
    s = SqliteEventStore(tmp_path / "anpr.db", tmp_path / "images", snapshot_width=64)
    yield s
    s.close()


def _engine(cfg, store, ocr):
    # hsrp off in tests (keeps the gray image from adding noise); motion always on (read every image)
    cfg = cfg.model_copy(update={"hsrp": cfg.hsrp.model_copy(update={"enabled": False})})
    return Engine(cfg, FakeDetector(), ocr, store, motion=lambda _f: True)


def _all_events(store):
    out, i = [], 0
    while True:
        e = store.get_event(i := i + 1)
        if e is None and i > 20:
            break
        if e is not None:
            out.append(e)
    return out


# ---- grouping + voting (the heart of it) -----------------------------------------------------------


def test_group_of_identical_reads_votes_one_event(store):
    cfg = AppConfig(ingest=IngestConfig(enabled=True))
    eng = _engine(cfg, store, SeqOcr(["MH12AB1234"]))
    imgs = [_jpeg_bytes() for _ in range(4)]
    ingest.process_group(eng, imgs, camera="gate-1")
    evs = _all_events(store)
    assert len(evs) == 1
    assert evs[0].plate == "MH12AB1234" and evs[0].camera == "gate-1"


def test_voting_fixes_one_wrong_read(store):
    cfg = AppConfig(ingest=IngestConfig(enabled=True))
    # 3 correct + 1 one-character-off (both valid); the majority must win.
    eng = _engine(cfg, store, SeqOcr(["MH12AB1234", "MH12AB1234", "MH12AB1284", "MH12AB1234"]))
    ingest.process_group(eng, [_jpeg_bytes() for _ in range(4)], camera=None)
    evs = _all_events(store)
    assert len(evs) == 1 and evs[0].plate == "MH12AB1234"


def test_find_all_images_handles_one_post_many_images():
    b = [_jpeg_b64(), _jpeg_b64(), _jpeg_b64()]
    # case A: a list of images in one POST
    assert len(ingest.find_all_images({"images": b})) == 3
    # nested, odd field name, single image: still found by auto-detect
    assert len(ingest.find_all_images({"data": {"picBase64": _jpeg_b64()}})) == 1
    # explicit field
    assert len(ingest.find_all_images({"d": {"imgs": b}}, field="d.imgs")) == 3
    assert ingest.decode_image(ingest.find_all_images({"x": _jpeg_b64()})[0]) is not None


def test_gvd_shape_reads_full_image_first_and_ignores_metadata():
    # The GVD camera nests data.ai_snap_picture.PlateInfo[] with PlateImg (crop), BgImg (full vehicle)
    # and ImageAllInfo (a base64 BINARY metadata blob - not an image).
    bg = _jpeg_b64(260, 120)  # full scene (distinct size so we can tell them apart)
    plate = _jpeg_b64(140, 48)  # plate crop
    meta = base64.b64encode(b"\x18\x00\x00\x00not-a-jpeg-metadata-blob" * 40).decode()
    body = {
        "data": {
            "ai_snap_picture": {
                "PlateInfo": [
                    {
                        "SnapId": "HR26FD7796",
                        "StrChn": "CH1",
                        "ImageAllInfo": meta,
                        "PlateImg": plate,
                        "BgImg": bg,
                    }
                ]
            }
        }
    }
    imgs = ingest.find_all_images(body)
    assert len(imgs) == 2  # BgImg + PlateImg; ImageAllInfo ignored
    # the FULL image (BgImg) is read first (primary vote), the crop (PlateImg) second (extra vote)
    assert imgs[0] == base64.b64decode(bg)
    assert imgs[1] == base64.b64decode(plate)
    # order of the fields in the JSON must not change the result
    body["data"]["ai_snap_picture"]["PlateInfo"][0] = {
        "BgImg": bg,
        "PlateImg": plate,
        "ImageAllInfo": meta,
    }
    imgs2 = ingest.find_all_images(body)
    assert imgs2[0] == base64.b64decode(bg) and imgs2[1] == base64.b64decode(plate)


# ---- buffering (closing a vehicle's group) ---------------------------------------------------------


def test_buffer_closes_group_on_timeout_then_processes(store):
    cfg = AppConfig(ingest=IngestConfig(enabled=True, group_timeout_s=0.2))
    seen = []
    buf = ingest.BurstBuffer(cfg, lambda g: seen.append(g))
    buf.start()
    buf.add("vid:7", [_jpeg_bytes(), _jpeg_bytes()], "cam", "7")
    buf.add("vid:7", [_jpeg_bytes()], "cam", "7")  # same vehicle -> same group
    time.sleep(0.6)
    buf.stop()  # flushes + drains + joins
    assert len(seen) == 1 and len(seen[0].images) == 3 and seen[0].vehicle_id == "7"


def test_buffer_closes_group_when_cap_reached(store):
    cfg = AppConfig(ingest=IngestConfig(enabled=True, group_timeout_s=30, max_images_per_vehicle=3))
    seen = []
    buf = ingest.BurstBuffer(cfg, lambda g: seen.append(g))
    buf.start()
    buf.add("vid:1", [_jpeg_bytes(), _jpeg_bytes()], None, "1")
    buf.add("vid:1", [_jpeg_bytes(), _jpeg_bytes()], None, "1")  # pushes over the cap -> closes now
    time.sleep(0.4)
    buf.stop()
    assert len(seen) == 1 and len(seen[0].images) == 3  # capped at 3


def test_shutdown_drains_queued_bursts(store):
    # A burst closed just before stop() must still be processed (no data loss on shutdown).
    cfg = AppConfig(ingest=IngestConfig(enabled=True, group_timeout_s=30))
    seen = []
    buf = ingest.BurstBuffer(cfg, lambda g: seen.append(g))
    buf.start()
    buf.add("vid:9", [_jpeg_bytes()], None, "9")  # still open (long timeout)
    buf.stop()  # must flush the open group and process it before returning
    assert len(seen) == 1 and seen[0].vehicle_id == "9"


# ---- the HTTP receiver ------------------------------------------------------------------------------


def _post(port, path, obj, token=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Ingest-Token"] = token
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=json.dumps(obj).encode(), headers=headers, method="POST"
    )
    try:
        r = urllib.request.urlopen(req, timeout=5)
        return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _serve(cfg):
    added = []
    buf = ingest.BurstBuffer(cfg, lambda g: None)
    buf.add = lambda key, images, camera, vid: added.append((key, len(images), vid)) or len(images)
    ctx = ingest._Ctx(cfg, buf)
    srv = ingest.make_server(ctx)
    import threading

    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True).start()
    return srv, added


def test_post_is_accepted_and_buffered():
    cfg = AppConfig(
        ingest=IngestConfig(enabled=True, host="127.0.0.1", port=8731, vehicle_id_field="vehicleId")
    )
    srv, added = _serve(cfg)
    try:
        code, body = _post(8731, "/api/v1/ingest", {"vehicleId": "V9", "images": [_jpeg_b64(), _jpeg_b64()]})
    finally:
        srv.shutdown()
        srv.server_close()
    assert code == 200 and body["ok"] and body["images"] == 2
    assert added and added[0][0] == "vid:V9" and added[0][1] == 2


def test_missing_image_is_422():
    cfg = AppConfig(ingest=IngestConfig(enabled=True, host="127.0.0.1", port=8732))
    srv, _ = _serve(cfg)
    try:
        code, body = _post(8732, "/api/v1/ingest", {"hello": "world"})
    finally:
        srv.shutdown()
        srv.server_close()
    assert code == 422 and not body["ok"]


def test_token_and_path_guards():
    cfg = AppConfig(ingest=IngestConfig(enabled=True, host="127.0.0.1", port=8733, token="secret"))
    srv, _ = _serve(cfg)
    try:
        assert _post(8733, "/api/v1/ingest", {"img": _jpeg_b64()})[0] == 401
        assert _post(8733, "/api/v1/ingest", {"img": _jpeg_b64()}, token="secret")[0] == 200
        assert _post(8733, "/nope", {"img": _jpeg_b64()}, token="secret")[0] == 404
    finally:
        srv.shutdown()
        srv.server_close()


# ---- accept any camera format: JSON base64, multipart, raw JPEG ------------------------------------


def test_extract_request_handles_json_multipart_and_raw():
    from anpr.config import IngestConfig

    ic = IngestConfig(enabled=True, vehicle_id_field="", camera_field="")
    raw = _jpeg_bytes()
    b64 = base64.b64encode(raw).decode()

    # JSON + base64
    import json as _j

    imgs, vid, cam = ingest.extract_request(
        "application/json", _j.dumps({"fullImage": b64, "vehicleId": "V1"}).encode(), ic
    )
    assert len(imgs) == 1

    # raw JPEG body
    imgs, vid, cam = ingest.extract_request("image/jpeg", raw, ic)
    assert len(imgs) == 1 and ingest.decode_image(imgs[0]) is not None

    # multipart/form-data with an image file part + text fields
    boundary = "X123"
    parts = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="vehicleId"\r\n\r\nV9\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="deviceId"\r\n\r\ngate1\r\n'
        ).encode()
        + (
            f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="p.jpg"\r\n'
            f"Content-Type: image/jpeg\r\n\r\n"
        ).encode()
        + raw
        + f"\r\n--{boundary}--\r\n".encode()
    )
    imgs, vid, cam = ingest.extract_request(f"multipart/form-data; boundary={boundary}", parts, ic)
    assert len(imgs) == 1 and vid == "V9" and cam == "gate1"


def test_post_raw_jpeg_body_is_accepted():
    cfg = AppConfig(ingest=IngestConfig(enabled=True, host="127.0.0.1", port=8741))
    added = []
    buf = ingest.BurstBuffer(cfg, lambda g: None)
    buf.add = lambda key, images, camera, vid: added.append((key, len(images))) or len(images)
    ctx = ingest._Ctx(cfg, buf)
    srv = ingest.make_server(ctx)
    import threading

    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True).start()
    try:
        import urllib.request

        req = urllib.request.Request(
            "http://127.0.0.1:8741/api/v1/ingest",
            data=_jpeg_bytes(),
            headers={"Content-Type": "image/jpeg"},
            method="POST",
        )
        r = urllib.request.urlopen(req, timeout=5)
        body = json.loads(r.read())
    finally:
        srv.shutdown()
        srv.server_close()
    assert r.status == 200 and body["images"] == 1 and added


def test_respond_with_plate_returns_the_read_in_the_response():
    """respond_with_plate: the POST is voted inline and the reply carries the plate (not just an ack)."""
    cfg = AppConfig(
        ingest=IngestConfig(
            enabled=True, host="127.0.0.1", port=8742, respond_with_plate=True, vehicle_id_field="vehicleId"
        )
    )
    seen = {}

    def fake_vote(images, camera, vehicle_id):
        seen["n"] = len(images)
        seen["vid"] = vehicle_id
        now = time.time()
        return [
            PlateEvent(
                plate="UP83DT0718",
                kind="standard",
                confidence=0.988,
                votes=1,
                track_id=1,
                first_seen=now,
                last_seen=now,
                hsrp="non_hsrp",
            )
        ]

    pushed = {}

    def fake_push(events):
        pushed["n"] = len(events)
        return {
            "enabled": True,
            "ok": True,
            "status": 200,
            "message": "accepted (HTTP 200)",
            "reply": '{"errorCode":"1"}',
        }

    buf = ingest.BurstBuffer(cfg, lambda g: None)
    ctx = ingest._Ctx(cfg, buf, vote_fn=fake_vote, push_fn=fake_push)
    srv = ingest.make_server(ctx)
    import threading

    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True).start()
    try:
        code, body = _post(
            8742, "/api/v1/ingest", {"vehicleId": "V7", "BgImg": _jpeg_b64(), "PlateImg": _jpeg_b64(140, 48)}
        )
    finally:
        srv.shutdown()
        srv.server_close()
    assert code == 200 and body["ok"]
    assert body["plate"] == "UP83DT0718"
    assert body["plates"][0]["confidence"] == 0.988
    assert body["plates"][0]["hsrp"] == "non_hsrp"
    assert body["images"] == 2
    assert "vehicle_id" not in body  # removed from the response
    # the voted plate was pushed to the client API and its outcome is reported in the reply
    assert body["push"] == {
        "enabled": True,
        "ok": True,
        "status": 200,
        "message": "accepted (HTTP 200)",
        "reply": '{"errorCode":"1"}',
    }
    assert pushed == {"n": 1}
    # both images of the one vehicle were voted together, inline
    assert seen == {"n": 2, "vid": "V7"}


def test_respond_with_plate_null_when_no_plate_read():
    cfg = AppConfig(ingest=IngestConfig(enabled=True, host="127.0.0.1", port=8743, respond_with_plate=True))
    buf = ingest.BurstBuffer(cfg, lambda g: None)
    ctx = ingest._Ctx(cfg, buf, vote_fn=lambda images, camera, vid: [])
    srv = ingest.make_server(ctx)
    import threading

    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True).start()
    try:
        code, body = _post(8743, "/api/v1/ingest", {"img": _jpeg_b64()})
    finally:
        srv.shutdown()
        srv.server_close()
    assert code == 200 and body["ok"] and body["plate"] is None and body["plates"] == []


def test_push_result_json_shapes():
    from anpr.push import PushOutcome, SendResult

    url = "http://192.168.1.72:21300/lane/anpr"
    where = {"url": url, "host": "192.168.1.72:21300", "plate": "UP83DT0718"}
    # sending to the client API is off / not configured
    assert ingest._push_result_json(None, enabled=False) == {
        "enabled": False,
        "message": ingest.PUSH_OFF_MESSAGE,
    }
    # enabled but there was nothing to send
    r = ingest._push_result_json(None, enabled=True, url=url)
    assert r["enabled"] and r["ok"] is None and r["queued"] is False and "error" not in r
    assert r["url"] == url and r["plate"] is None
    # the client API could not be reached -> not delivered, kept and retried; the error names URL + plate
    down = SendResult(False, None, True, "cannot reach the server: [Errno 61] Connection refused")
    d = ingest._push_result_json(
        PushOutcome(False, True, down.message, down), enabled=True, url=url, plates=["UP83DT0718"]
    )
    assert d == {
        **where,
        "enabled": True,
        "ok": False,
        "queued": True,
        "status": None,
        "message": "cannot reach the server: [Errno 61] Connection refused",
        "reply": "",
        "error": (
            "API is not connecting: http://192.168.1.72:21300/lane/anpr "
            "(host 192.168.1.72:21300): [Errno 61] Connection refused. Plate UP83DT0718 is "
            "saved on the Pi and is sent automatically when the API accepts it."
        ),
    }
    # in backoff (no request made now): still reported as not connecting, with the reason
    b = ingest._push_result_json(
        PushOutcome(False, True, "queued: cannot reach the server: timed out; retrying in 7 s"),
        enabled=True,
        url=url,
        plates=["UP83DT0718"],
    )
    assert b["error"].startswith(f"API is not connecting: {url} (host 192.168.1.72:21300): timed out.")
    assert "UP83DT0718" in b["error"]
    # waiting behind an earlier plate while the API is down: the error names the connection, not "busy"
    busy = ingest._push_result_json(
        PushOutcome(False, True, "queued: the sender is busy with earlier plates; sent shortly"),
        enabled=True,
        url=url,
        plates=["HR69D5319"],
        cause="cannot reach the server: timed out",
    )
    assert busy["message"].startswith("queued: the sender is busy")
    assert busy["error"].startswith(f"API is not connecting: {url} (host 192.168.1.72:21300): timed out.")
    assert "HR69D5319" in busy["error"]
    # the client API accepted THIS plate, and its answer is echoed back in "reply"; no error
    ok = SendResult(True, 200, False, "accepted (HTTP 200)", '{"errorCode":"1"}')
    assert ingest._push_result_json(
        PushOutcome(True, False, ok.message, ok), enabled=True, url=url, plates=["UP83DT0718"]
    ) == {
        **where,
        "enabled": True,
        "ok": True,
        "queued": False,
        "status": 200,
        "message": "accepted (HTTP 200)",
        "reply": '{"errorCode":"1"}',
    }
    # the API answered with an error -> named as an API error (not "not connecting"), plate included
    bad = SendResult(False, 500, True, "server error (HTTP 500)", "oops")
    e = ingest._push_result_json(
        PushOutcome(False, True, bad.message, bad), enabled=True, url=url, plates=["UP83DT0718"]
    )
    assert e["status"] == 500 and e["error"].startswith(
        f"API at {url} (host 192.168.1.72:21300) did not accept plate UP83DT0718: server error (HTTP 500)"
    )
    # refused for good -> not re-sent
    gone = ingest._push_result_json(
        PushOutcome(False, False, "refused"), enabled=True, url=url, plates=["UP83DT0718"]
    )
    assert gone["error"].endswith("Plate UP83DT0718 is not sent again.")


def test_respond_with_plate_reports_api_not_connecting():
    """End to end through the real HTTP handler: the push could not connect, so the camera's reply still
    carries the plate (ok: true, HTTP 200) plus a top-level error naming the API URL and the plate."""
    from anpr.push import PushOutcome, SendResult

    url = "http://192.168.1.72:21300/lane/anpr"
    cfg = AppConfig(ingest=IngestConfig(enabled=True, host="127.0.0.1", port=8753, respond_with_plate=True))

    def push_down(events):
        down = SendResult(False, None, True, "cannot reach the server: timed out")
        return ingest._push_result_json(
            PushOutcome(False, True, down.message, down),
            enabled=True,
            url=url,
            plates=[e.plate for e in events],
        )

    ctx = ingest._Ctx(
        cfg,
        ingest.BurstBuffer(cfg, lambda g: None),
        vote_fn=lambda images, cam, vid: ([_ev("UP83DT0718")], []),
        push_fn=push_down,
    )
    srv = _serve_ctx(ctx)
    try:
        code, body = _post(8753, "/api/v1/ingest", {"BgImg": _jpeg_b64()})
    finally:
        srv.shutdown()
        srv.server_close()
    assert code == 200 and body["ok"] is True and body["plate"] == "UP83DT0718"
    assert body["push"]["ok"] is False and body["push"]["url"] == url
    assert body["push"]["plate"] == "UP83DT0718" and body["push"]["host"] == "192.168.1.72:21300"
    assert body["error"] == body["push"]["error"]
    assert body["error"].startswith(f"API is not connecting: {url} (host 192.168.1.72:21300): timed out")
    assert "UP83DT0718" in body["error"]


# ---- review fixes: duplicates, crashes in push, bounded concurrency, image search -------------------


def _serve_ctx(ctx):
    import threading

    srv = ingest.make_server(ctx)
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    return srv


def _ev(plate):
    now = time.time()
    return PlateEvent(
        plate=plate,
        kind="standard",
        confidence=0.99,
        votes=1,
        track_id=1,
        first_seen=now,
        last_seen=now,
        hsrp="non_hsrp",
    )


def test_engine_reports_duplicates_only_when_asked(store):
    cfg = AppConfig(ingest=IngestConfig(enabled=True))
    eng = _engine(cfg, store, SeqOcr(["MH12AB1234"]))
    eng.collect_duplicates = True
    first: list = []
    assert [e.plate for e in ingest.process_group(eng, [_jpeg_bytes()] * 4, None, duplicates=first)] == [
        "MH12AB1234"
    ]
    assert first == []
    again: list = []
    assert ingest.process_group(eng, [_jpeg_bytes()] * 4, None, duplicates=again) == []  # within 60 s
    assert [e.plate for e in again] == ["MH12AB1234"] and again[0].id is None
    # off (the video loop): never collected, so the list cannot grow forever
    eng2 = _engine(cfg, store, SeqOcr(["MH12AB1234"]))
    ingest.process_group(eng2, [_jpeg_bytes()] * 4, None)
    assert eng2.duplicates == []


def test_respond_with_plate_reports_a_duplicate_instead_of_null():
    cfg = AppConfig(ingest=IngestConfig(enabled=True, host="127.0.0.1", port=8751, respond_with_plate=True))
    pushed = []

    def fake_push(events):
        pushed.append(events)
        return ingest._push_result_json(None, enabled=True)

    ctx = ingest._Ctx(
        cfg,
        ingest.BurstBuffer(cfg, lambda g: None),
        vote_fn=lambda images, cam, vid: ([], [_ev("UP83DT0718")]),
        push_fn=fake_push,
    )
    srv = _serve_ctx(ctx)
    try:
        code, body = _post(8751, "/api/v1/ingest", {"BgImg": _jpeg_b64()})
    finally:
        srv.shutdown()
        srv.server_close()
    assert code == 200 and body["plate"] == "UP83DT0718"
    assert body["plates"][0]["duplicate"] is True
    assert body["push"]["ok"] is None and body["push"]["message"].startswith("duplicate")


def test_respond_with_plate_still_replies_when_the_push_crashes():
    cfg = AppConfig(ingest=IngestConfig(enabled=True, host="127.0.0.1", port=8752, respond_with_plate=True))

    def boom(events):
        raise RuntimeError("database is locked")

    ctx = ingest._Ctx(
        cfg,
        ingest.BurstBuffer(cfg, lambda g: None),
        vote_fn=lambda images, cam, vid: [_ev("UP83DT0718")],
        push_fn=boom,
    )
    srv = _serve_ctx(ctx)
    try:
        code, body = _post(8752, "/api/v1/ingest", {"BgImg": _jpeg_b64()})
    finally:
        srv.shutdown()
        srv.server_close()
    assert code == 200 and body["plate"] == "UP83DT0718"
    assert body["plates"][0]["duplicate"] is False
    assert body["push"]["ok"] is False and body["push"]["queued"] is True


def test_respond_with_plate_sheds_load_with_503_when_all_slots_are_busy(monkeypatch):
    import threading

    monkeypatch.setattr(ingest, "_SYNC_SLOT_WAIT_S", 0.05)
    cfg = AppConfig(ingest=IngestConfig(enabled=True, host="127.0.0.1", port=8753, respond_with_plate=True))
    slots = threading.BoundedSemaphore(1)
    ctx = ingest._Ctx(
        cfg,
        ingest.BurstBuffer(cfg, lambda g: None),
        vote_fn=lambda images, cam, vid: [_ev("UP83DT0718")],
        slots=slots,
    )
    srv = _serve_ctx(ctx)
    try:
        slots.acquire()  # another POST holds the only slot
        code, body = _post(8753, "/api/v1/ingest", {"BgImg": _jpeg_b64()})
        assert code == 503 and not body["ok"]
        slots.release()
        code, body = _post(8753, "/api/v1/ingest", {"BgImg": _jpeg_b64()})
        assert code == 200 and body["plate"] == "UP83DT0718"
    finally:
        srv.shutdown()
        srv.server_close()


def test_full_image_after_many_crops_is_still_found_first():
    full = _jpeg_b64(300, 200)
    body = {"PlateImg": [_jpeg_b64(140, 48) for _ in range(5)], "zz": {"BgImg": full}}
    imgs = ingest.find_all_images(body, limit=3)
    assert len(imgs) == 3
    assert imgs[0] == base64.b64decode(full)  # the full image is the primary vote


def test_a_trickled_body_is_cut_off_and_frees_its_slot(monkeypatch):
    import socket
    import threading

    monkeypatch.setattr(ingest, "_BODY_DEADLINE_S", 0.3)
    cfg = AppConfig(ingest=IngestConfig(enabled=True, host="127.0.0.1", port=8754, respond_with_plate=True))
    slots = threading.BoundedSemaphore(1)
    ctx = ingest._Ctx(
        cfg,
        ingest.BurstBuffer(cfg, lambda g: None),
        vote_fn=lambda images, cam, vid: [_ev("UP83DT0718")],
        slots=slots,
    )
    srv = _serve_ctx(ctx)
    try:
        s = socket.create_connection(("127.0.0.1", 8754))
        s.sendall(
            b"POST /api/v1/ingest HTTP/1.0\r\nContent-Type: application/json\r\n"
            b"Content-Length: 100000\r\n\r\n{"
        )
        s.settimeout(5)
        t0 = time.time()
        reply = b""
        while time.time() - t0 < 4:
            try:
                s.sendall(b" ")  # one byte at a time, each well inside the per-recv timeout
            except OSError:
                break
            time.sleep(0.1)
            s.setblocking(False)
            try:
                reply += s.recv(4096)
            except BlockingIOError:
                pass
            except OSError:
                break
            finally:
                s.setblocking(True)
            if reply:
                break
        s.close()
        assert b" 408 " in reply.split(b"\r\n")[0]
        code, body = _post(8754, "/api/v1/ingest", {"BgImg": _jpeg_b64()})  # the slot was released
        assert code == 200 and body["plate"] == "UP83DT0718"
    finally:
        srv.shutdown()
        srv.server_close()


def test_ingest_engine_judges_hsrp_from_a_single_image():
    """A camera POST gives 1-2 reads: one image showing the hologram must give "hsrp", not the
    video rule's "unsure" (saved as "not sure: ...")."""
    from anpr.hsrp import decide

    cfg = AppConfig()
    seen = {}

    class _E:
        def __init__(self, c, *a, **k):
            seen["cfg"] = c

    import anpr.engine as engine_mod

    orig = engine_mod.Engine
    engine_mod.Engine = _E
    try:
        ingest._build_engine(cfg, None, None, None)
    finally:
        engine_mod.Engine = orig
    h = seen["cfg"].hsrp
    assert (h.min_marked, h.min_clean) == (1, 1)
    assert decide(["hsrp"], h.min_marked, h.min_clean) == "hsrp"
    assert decide(["non_hsrp"], h.min_marked, h.min_clean) == "non_hsrp"
    assert cfg.hsrp.min_marked == 2  # the video pipeline keeps its stricter rule


# ---- zoom retry: a small readable plate next to a bigger unreadable one -----------------------------


class _TileDetector:
    """Whole 400x400 image: only the big (unreadable) plate. A 240x240 tile (60%): a small plate at tile
    coords (20, 30)-(120, 70) - i.e. only visible when zoomed in."""

    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        if frame.shape[0] == 400:
            return [Box(200, 300, 380, 390, 0.9)]
        return [Box(20, 30, 120, 70, 0.6)]


class _BoxOcr:
    """Reads by box: the crop stand-in carries box.x1 in its first pixel (see the fixture below)."""

    def __init__(self, texts):  # x1 -> text
        self.texts = texts
        self.seen = []

    def read(self, crop):
        x1 = int(crop[0, 0, 0])
        self.seen.append(x1)
        t = self.texts.get(x1, "XX")
        return OcrResult(t, tuple([0.97] * len(t)))


@pytest.fixture
def _crop_by_box(monkeypatch):
    monkeypatch.setattr(
        "anpr.preprocess.prepare_plate", lambda frame, box, cfg: np.full((64, 128, 3), box.x1 % 256, np.uint8)
    )


def _tile_engine(store, ocr, **ingest_kw):
    cfg = AppConfig(ingest=IngestConfig(enabled=True, **ingest_kw))
    cfg = cfg.model_copy(
        update={
            "hsrp": cfg.hsrp.model_copy(update={"enabled": False}),
            "vote": cfg.vote.model_copy(update={"min_votes": 1, "end_min_votes": 1}),  # one image per vehicle
        }
    )
    return Engine(cfg, _TileDetector(), ocr, store, motion=lambda _f: True)


def test_zoom_candidates_maps_tiles_back_and_merges():
    det = _TileDetector()
    boxes = ingest.zoom_candidates(det, np.zeros((400, 400, 3), np.uint8), limit=8)
    assert det.calls == 5  # whole image + 2x2 tiles
    # tiles start at x/y = 0 and 160 (400 - 240): the small plate appears 4 times at different places
    assert boxes[0] == Box(200, 300, 380, 390, 0.9)
    assert {(b.x1, b.y1) for b in boxes[1:]} == {(20, 30), (180, 30), (20, 190), (180, 190)}


def test_zoom_retry_reads_small_plate_when_big_one_fails(store, _crop_by_box):
    ocr = _BoxOcr({20: "DL1MA1846"})  # the big plate (x1=200) reads as junk
    eng = _tile_engine(store, ocr)
    evs = ingest.process_group(eng, [_jpeg_bytes(400, 400)], camera="gate-1")
    assert [e.plate for e in evs] == ["DL1MA1846"] and evs[0].camera == "gate-1"
    assert isinstance(eng.detector, _TileDetector)  # the real detector is restored


def test_zoom_retry_off(store, _crop_by_box):
    eng = _tile_engine(store, _BoxOcr({20: "DL1MA1846"}), zoom_fallback=False)
    assert ingest.process_group(eng, [_jpeg_bytes(400, 400)], camera=None) == []
    assert eng.detector.calls == 1


def test_zoom_retry_skipped_when_first_pass_reads(store, _crop_by_box):
    eng = _tile_engine(store, _BoxOcr({200: "MH12AB1234", 20: "DL1MA1846"}))
    assert [e.plate for e in ingest.process_group(eng, [_jpeg_bytes(400, 400)], None)] == ["MH12AB1234"]
    assert eng.detector.calls == 1


@pytest.mark.parametrize("readable_x1", [200, 20], ids=["first-pass", "zoom-retry"])
def test_zoom_retry_keeps_whole_vehicle_vote(store, _crop_by_box, readable_x1):
    """The zoom retry must use the same whole-vehicle vote rules as the first pass: with the default vote
    settings (min_votes 3), one clear read is kept at the end of the vehicle, whichever look found it."""
    cfg = AppConfig(ingest=IngestConfig(enabled=True))
    cfg = cfg.model_copy(update={"hsrp": cfg.hsrp.model_copy(update={"enabled": False})})
    assert cfg.vote.min_votes == 3 and cfg.ingest.whole_vehicle and cfg.ingest.zoom_fallback
    eng = Engine(cfg, _TileDetector(), _BoxOcr({readable_x1: "DL1MA1846"}), store, motion=lambda _f: True)
    assert [e.plate for e in ingest.process_group(eng, [_jpeg_bytes(400, 400)], None)] == ["DL1MA1846"]


def test_ingest_exits_cleanly_above_the_memory_limit(tmp_path, monkeypatch):
    """Heavy load on the Pi: above runtime.max_rss_mb the receiver stops itself (drains, returns
    EXIT_MEMORY so systemd restarts it) instead of being killed by MemoryMax mid-write."""
    import anpr.engine as engine_mod
    from anpr.config import RuntimeConfig, StorageConfig

    cfg = AppConfig(
        ingest=IngestConfig(enabled=True, host="127.0.0.1", port=8761),
        storage=StorageConfig(db_path=tmp_path / "a.db", image_dir=tmp_path / "img"),
        runtime=RuntimeConfig(max_rss_mb=100.0, status_interval_s=0.1),
    )
    store = SqliteEventStore(cfg.storage.db_path, cfg.storage.image_dir)
    monkeypatch.setattr(engine_mod, "current_rss_mb", lambda: 999.0)
    monkeypatch.setattr(ingest, "_build_engine", lambda *a: __import__("types").SimpleNamespace())
    try:
        t0 = time.monotonic()
        code = ingest.run_ingest(cfg, store, None, None, stop=lambda: time.monotonic() - t0 > 10)
        assert code == engine_mod.EXIT_MEMORY
        assert time.monotonic() - t0 < 5
    finally:
        store.close()


def test_image_size_reads_the_header():
    img = np.zeros((300, 500, 3), np.uint8)
    for ext in (".jpg", ".png"):
        raw = cv2.imencode(ext, img)[1].tobytes()
        assert ingest.image_size(raw) == (500, 300)
    assert ingest.image_size(b"not an image") is None


@pytest.mark.parametrize(("w", "h"), [(4000, 3000), (3000, 4000), (1280, 720), (8000, 600)])
def test_big_pictures_are_decoded_small(w, h):
    """A 12 MP phone photo must not take 36 MB per POST on the Pi: long side capped at 1920."""
    img = np.full((h, w, 3), 128, np.uint8)
    cv2.putText(img, "TEST", (w // 3, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), 10)
    out = ingest.decode_image(cv2.imencode(".jpg", img)[1].tobytes())
    assert max(out.shape[:2]) == min(max(w, h), ingest.MAX_IMAGE_SIDE)
    assert abs(out.shape[1] / out.shape[0] - w / h) < 0.02  # same shape, just smaller


# ---- fast_ack: reply at once, read from a disk queue ----


def test_spool_keeps_vehicles_until_read(tmp_path):
    sp = ingest.Spool(tmp_path / "spool", max_items=2)
    a = sp.put([b"img-a1", b"img-a2"], "entry", "V1")
    sp.put([b"img-b"], None, None)
    assert a and len(sp) == 2
    assert sp.put([b"img-c"], None, None) is None  # full -> the camera gets 503 and re-sends
    first = sp.pending()[0]
    assert sp.load(first) == ([b"img-a1", b"img-a2"], "entry", "V1")
    # a restart: a new Spool on the same folder still has both (nothing acknowledged is lost)
    assert len(ingest.Spool(tmp_path / "spool", max_items=2)) == 2
    sp.done(first)
    assert len(sp) == 1 and len(sp.pending()) == 1


def test_spool_drops_half_written_posts(tmp_path):
    (tmp_path / "spool" / "123-000001.tmp").mkdir(parents=True)  # crash while writing: never acked
    sp = ingest.Spool(tmp_path / "spool", max_items=10)
    assert len(sp) == 0 and not list((tmp_path / "spool").glob("*.tmp"))


def test_spool_worker_reads_in_order_and_retries_then_parks_bad_posts(tmp_path):
    import threading

    sp = ingest.Spool(tmp_path / "spool", max_items=10)
    for name in ("one", "bad", "two"):
        sp.put([name.encode()], None, None)
    seen = []

    def vote(images, camera, vid):
        seen.append(images[0])
        if images[0] == b"bad":
            raise ValueError("broken image")

    stop = threading.Event()
    t = threading.Thread(target=ingest._spool_worker, args=(sp, vote, stop), daemon=True)
    t.start()
    deadline = time.time() + 5
    while len(sp) and time.time() < deadline:
        time.sleep(0.02)
    stop.set()
    t.join(2)
    assert seen[0] == b"one" and seen.count(b"bad") == ingest.Spool.MAX_TRIES and b"two" in seen
    assert len(sp) == 0 and len(list((tmp_path / "spool" / "failed").iterdir())) == 1


def test_fast_ack_replies_at_once_and_reads_in_the_background(tmp_path, monkeypatch):
    import threading

    from anpr.config import StorageConfig

    cfg = AppConfig(
        ingest=IngestConfig(
            enabled=True, host="127.0.0.1", port=8762, fast_ack=True, respond_with_plate=True
        ),
        storage=StorageConfig(db_path=tmp_path / "a.db", image_dir=tmp_path / "img"),
    )
    read = []

    def slow_vote(engine, images, camera, duplicates=None):  # reading a plate takes ~1 s on a Pi
        time.sleep(0.5)
        read.append(len(images))
        return []

    monkeypatch.setattr(ingest, "_build_engine", lambda *a: __import__("types").SimpleNamespace())
    monkeypatch.setattr(ingest, "process_group", slow_vote)
    store = SqliteEventStore(cfg.storage.db_path, cfg.storage.image_dir)
    stop = threading.Event()
    t = threading.Thread(
        target=ingest.run_ingest, args=(cfg, store, None, None), kwargs={"stop": stop.is_set}
    )
    t.start()
    try:
        for _ in range(50):
            try:
                urllib.request.urlopen("http://127.0.0.1:8762/health", timeout=1)
                break
            except OSError:
                time.sleep(0.1)
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            code, body = _post(8762, "/api/v1/ingest", {"BgImg": _jpeg_b64()})
            times.append(time.perf_counter() - t0)
            assert code == 200 and body["queued"] is True and body["id"]
        assert max(times) < 0.1  # the camera is answered without waiting for the read
        deadline = time.time() + 5
        while len(read) < 3 and time.time() < deadline:
            time.sleep(0.05)
        assert read == [1, 1, 1]  # every queued vehicle was read, nothing lost
        assert not list((tmp_path / "spool").glob("*.post"))
    finally:
        stop.set()
        t.join(10)
        store.close()


# ---- second plate check: an HSRP on the grille, a painted copy on the bumper ------------------------


class _TwoPlateDetector:
    """Whole 400x400 image: only the painted bumper plate (40,300)-(200,380). Zoomed tiles also show the
    HSRP plate above it, at (60,120)-(220,200) in whole-image coordinates."""

    def __init__(self, upper=(60, 120, 220, 200)):
        self.upper = upper

    def detect(self, frame):
        if frame.shape[0] == 400:
            return [Box(40, 300, 200, 380, 0.9)]
        x1, y1, x2, y2 = self.upper
        return [Box(x1, y1, x2, y2, 0.6)] if frame.shape[0] == 240 and y2 <= 240 else []


class _ConfOcr(_BoxOcr):
    """_BoxOcr with a confidence per text."""

    def __init__(self, texts, confs):
        super().__init__(texts)
        self.confs = confs

    def read(self, crop):
        r = super().read(crop)
        return OcrResult(r.text, tuple([self.confs.get(r.text, 0.97)] * len(r.text)))


def _plates_engine(store, ocr, detector=None, **ingest_kw):
    cfg = AppConfig(ingest=IngestConfig(enabled=True, **ingest_kw))
    cfg = cfg.model_copy(
        update={
            "hsrp": cfg.hsrp.model_copy(update={"enabled": True, "min_marked": 1, "min_clean": 1}),
            "vote": cfg.vote.model_copy(update={"min_votes": 1, "end_min_votes": 1, "dedupe_window_s": 0.0}),
        }
    )
    # HSRP look stand-in: the upper (grille) plate is HSRP, the bumper plate is not
    look = lambda frame, box: ("hsrp", "") if box.y1 < 250 else ("non_hsrp", "")  # noqa: E731
    return Engine(cfg, detector or _TwoPlateDetector(), ocr, store, motion=lambda _f: True, hsrp_look=look)


@pytest.fixture
def _steady_variants(monkeypatch):
    # the variant reads (trims, padding, contrast) see the same crop: they agree with the first read
    monkeypatch.setattr("anpr.preprocess.read_variants", lambda crop: [crop] * 5)


def test_second_plate_hsrp_marks_the_vehicle_and_a_surer_reading_wins(store, _crop_by_box, _steady_variants):
    ocr = _ConfOcr({40: "DL1MQ1846", 60: "DL1MA1846"}, {"DL1MQ1846": 0.88, "DL1MA1846": 0.97})
    evs = ingest.process_group(_plates_engine(store, ocr), [_jpeg_bytes(400, 400)], camera=None)
    assert [(e.plate, e.hsrp) for e in evs] == [("DL1MA1846", "hsrp")]


def test_the_number_comes_from_the_hsrp_plate(store, _crop_by_box, monkeypatch):
    # the variant reads disagree (junk): an unconfirmed reading of the HSRP plate
    monkeypatch.setattr("anpr.preprocess.read_variants", lambda crop: [np.zeros_like(crop)] * 5)
    ocr = _ConfOcr({40: "DL1MQ1846", 60: "DL1MA1846"}, {"DL1MQ1846": 0.88, "DL1MA1846": 0.97})
    # whole-vehicle mode (default): the number always comes from the plate judged HSRP
    evs = ingest.process_group(_plates_engine(store, ocr), [_jpeg_bytes(400, 400)], camera=None)
    assert [(e.plate, e.hsrp) for e in evs] == [("DL1MA1846", "hsrp")]
    # video-style voting: only a confirmed, surer reading may overrule the vote - here it is not confirmed
    eng = _plates_engine(store, ocr, whole_vehicle=False)
    evs = ingest.process_group(eng, [_jpeg_bytes(400, 400)], camera=None)
    assert [(e.plate, e.hsrp) for e in evs] == [("DL1MQ1846", "hsrp")]


def test_second_plate_with_a_different_digit_is_another_vehicle(store, _crop_by_box, _steady_variants):
    ocr = _ConfOcr({40: "DL1MA1846", 60: "DL1MA1847"}, {})  # a fleet's next truck, not this one
    evs = ingest.process_group(_plates_engine(store, ocr), [_jpeg_bytes(400, 400)], camera=None)
    assert [(e.plate, e.hsrp) for e in evs] == [("DL1MA1846", "non_hsrp")]


def test_second_plate_check_off(store, _crop_by_box, _steady_variants):
    ocr = _ConfOcr({40: "DL1MQ1846", 60: "DL1MA1846"}, {"DL1MQ1846": 0.88})
    eng = _plates_engine(store, ocr, second_plate_check=False)
    evs = ingest.process_group(eng, [_jpeg_bytes(400, 400)], camera=None)
    assert [(e.plate, e.hsrp) for e in evs] == [("DL1MQ1846", "non_hsrp")]
    assert eng.second_plate_hsrp is None


def test_second_plate_far_away_or_overlapping_is_not_this_vehicle(store, _crop_by_box, _steady_variants):
    for upper in ((0, 0, 30, 20), (50, 290, 210, 375)):  # tiny far plate / the bumper plate again (shifted)
        ocr = _ConfOcr({40: "DL1MA1846", upper[0]: "DL1MA1846"}, {})
        eng = _plates_engine(store, ocr, detector=_TwoPlateDetector(upper))
        assert [e.hsrp for e in ingest.process_group(eng, [_jpeg_bytes(400, 400)], None)] == ["non_hsrp"]


def test_same_number_allows_two_letters_never_a_digit():
    assert ingest._same_number("DL1MA1846", "DL1MA1846")
    assert ingest._same_number("DL1MA1846", "DL1MQ1846")  # worn painted A read as Q
    assert ingest._same_number("DL1MR1846", "DL1HA1846")  # painted copy vs a 50 px HSRP: 2 letters
    assert not ingest._same_number("DL1MR1846", "DH1HQ1846")  # 3 letters
    assert not ingest._same_number("DL1MA1846", "DL1MA1847")  # consecutive fleet numbers
    assert not ingest._same_number("DL1MA1846", "DL1MQ1847")
    assert not ingest._same_number("DL1MA1846", "DL1A1846")


def test_same_number_inputs_are_validator_normalised():
    # Both texts reach _same_number already cleaned by the validator: a digit slot always holds a
    # digit, so a misread B/O/S in the number part is fixed (B -> 8) before the two are compared.
    r = PlateValidator(ValidationConfig()).validate(OcrResult("MH12AB12B4", (0.99,) * 10))
    assert r is not None and r.text == "MH12AB1284"
    assert ingest._same_number("MH12AB1284", r.text)


def test_small_plate_is_judged_over_shifted_boxes():
    boxes = ingest._shifted(Box(584, 69, 634, 102, 0.7))
    assert len(boxes) == 18 and all(isinstance(v, int) for b in boxes for v in (b.x1, b.y1, b.x2, b.y2))
    assert Box(584, 69, 634, 102, 0.7) in boxes


def test_vehicle_saved_under_its_hsrp_reading_is_still_a_duplicate_by_its_painted_one(
    store, _crop_by_box, _steady_variants
):
    ocr = _ConfOcr({40: "DL1MQ1846", 60: "DL1MA1846"}, {"DL1MQ1846": 0.88, "DL1MA1846": 0.97})
    eng = _plates_engine(store, ocr)
    eng.cfg = eng.cfg.model_copy(update={"vote": eng.cfg.vote.model_copy(update={"dedupe_window_s": 60.0})})
    eng.collect_duplicates = True
    assert [e.plate for e in ingest.process_group(eng, [_jpeg_bytes(400, 400)], None)] == ["DL1MA1846"]
    eng.detector = _TwoPlateDetector(upper=(60, 300, 220, 380))  # next request: only the painted plate seen
    dups = []
    assert ingest.process_group(eng, [_jpeg_bytes(400, 400)], None, duplicates=dups) == []
    assert [d.plate for d in dups] == ["DL1MQ1846"]


# ---- whole vehicle: every image read, decided once; HSRP when one image is sure of it ---------------


class _SeqDetector:
    """One box per image, in order: image 1 sees the painted plate, image 2 the HSRP plate."""

    def __init__(self, boxes):
        self.boxes, self.i = list(boxes), 0

    def detect(self, frame):
        if frame.shape[0] != 400:  # zoom tiles: nothing extra
            return []
        b = self.boxes[min(self.i, len(self.boxes) - 1)]
        self.i += 1
        return [b]


class _PJudge:
    """HSRP stand-in with the classifier's judge(): (look, why, p) by box.x1."""

    def __init__(self, by_x1):
        self.by_x1 = by_x1

    def __call__(self, frame, box):
        return self.judge(frame, box)[:2]

    def judge(self, frame, box):
        look, p = self.by_x1[int(box.x1)]
        return look, "", p


def _vehicle_engine(store, ocr, boxes, judge, **ingest_kw):
    cfg = AppConfig(ingest=IngestConfig(enabled=True, second_plate_check=False, **ingest_kw))
    cfg = cfg.model_copy(
        update={
            "hsrp": cfg.hsrp.model_copy(update={"enabled": True, "min_marked": 1, "min_clean": 1}),
            "vote": cfg.vote.model_copy(
                update={
                    "min_votes": 1,
                    "end_min_votes": 1,
                    "dedupe_window_s": 0.0,
                    "close_ratio": 0.0,
                    "settle_s": 0.0,
                }
            ),  # as _build_engine
        }
    )
    return Engine(cfg, _SeqDetector(boxes), ocr, store, motion=lambda _f: True, hsrp_look=judge)


_PAINTED, _HSRP = Box(40, 300, 200, 380, 0.9), Box(60, 120, 220, 200, 0.9)


def test_one_sure_hsrp_image_makes_the_vehicle_hsrp_and_its_reading_wins(store, _crop_by_box):
    ocr = _ConfOcr({40: "DL1MR1846", 60: "DL1MA1846"}, {"DL1MR1846": 0.88, "DL1MA1846": 0.99})
    for order in ([_PAINTED, _HSRP], [_HSRP, _PAINTED]):
        eng = _vehicle_engine(store, ocr, order, _PJudge({40: ("non_hsrp", 0.02), 60: ("hsrp", 0.99)}))
        evs = ingest.process_group(eng, [_jpeg_bytes(400, 400)] * 2, None)
        assert [(e.plate, e.hsrp) for e in evs] == [("DL1MA1846", "hsrp")]


def test_a_weak_hsrp_image_against_another_image_is_not_sure(store, _crop_by_box):
    ocr = _ConfOcr({40: "DL1MA1846", 60: "DL1MA1846"}, {})
    judge = _PJudge({40: ("non_hsrp", 0.05), 60: ("hsrp", 0.7)})
    eng = _vehicle_engine(store, ocr, [_PAINTED, _HSRP], judge)
    evs = ingest.process_group(eng, [_jpeg_bytes(400, 400)] * 2, None)
    assert [e.hsrp for e in evs] == ["not sure: images disagree"]


def test_whole_vehicle_off_keeps_the_first_confident_read(store, _crop_by_box):
    ocr = _ConfOcr({40: "DL1MR1846", 60: "DL1MA1846"}, {"DL1MR1846": 0.88, "DL1MA1846": 0.99})
    judge = _PJudge({40: ("non_hsrp", 0.02), 60: ("hsrp", 0.99)})
    eng = _vehicle_engine(store, ocr, [_PAINTED, _HSRP], judge, whole_vehicle=False)
    evs = ingest.process_group(eng, [_jpeg_bytes(400, 400)] * 2, None)
    assert [(e.plate, e.hsrp) for e in evs] == [("DL1MR1846", "non_hsrp")]


def test_images_of_two_different_numbers_give_no_plate_rather_than_a_guess(store, _crop_by_box):
    ocr = _ConfOcr({40: "DL1MA1846", 60: "HR26TC5656"}, {})
    judge = _PJudge({40: ("non_hsrp", 0.05), 60: ("hsrp", 0.99)})
    eng = _vehicle_engine(store, ocr, [_PAINTED, _HSRP], judge, zoom_fallback=False)
    assert ingest.process_group(eng, [_jpeg_bytes(400, 400)] * 2, None) == []


def test_a_sure_hsrp_image_gives_the_number_even_when_less_confident(store, _crop_by_box):
    # the painted plate reads more confidently (0.99) than the HSRP plate (0.90): the HSRP plate's number wins
    ocr = _ConfOcr({40: "DL1MR1846", 60: "DL1MA1846"}, {"DL1MR1846": 0.99, "DL1MA1846": 0.90})
    judge = _PJudge({40: ("non_hsrp", 0.02), 60: ("hsrp", 0.99)})
    eng = _vehicle_engine(store, ocr, [_PAINTED, _HSRP], judge)
    evs = ingest.process_group(eng, [_jpeg_bytes(400, 400)] * 2, None)
    assert [(e.plate, e.hsrp) for e in evs] == [("DL1MA1846", "hsrp")]
