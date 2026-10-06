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
from anpr.config import AppConfig, IngestConfig
from anpr.engine import Engine
from anpr.storage import SqliteEventStore
from anpr.types import Box, OcrResult, PlateEvent


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
    body = {"data": {"ai_snap_picture": {"PlateInfo": [{
        "SnapId": "HR26FD7796", "StrChn": "CH1",
        "ImageAllInfo": meta, "PlateImg": plate, "BgImg": bg,
    }]}}}
    imgs = ingest.find_all_images(body)
    assert len(imgs) == 2  # BgImg + PlateImg; ImageAllInfo ignored
    # the FULL image (BgImg) is read first (primary vote), the crop (PlateImg) second (extra vote)
    assert imgs[0] == base64.b64decode(bg)
    assert imgs[1] == base64.b64decode(plate)
    # order of the fields in the JSON must not change the result
    body["data"]["ai_snap_picture"]["PlateInfo"][0] = {
        "BgImg": bg, "PlateImg": plate, "ImageAllInfo": meta,
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
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(obj).encode(),
                                 headers=headers, method="POST")
    try:
        r = urllib.request.urlopen(req, timeout=5)
        return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _serve(cfg):
    added = []
    buf = ingest.BurstBuffer(cfg, lambda g: None)
    buf.add = lambda key, images, camera, vid: (added.append((key, len(images), vid)) or len(images))
    ctx = ingest._Ctx(cfg, buf)
    srv = ingest.make_server(ctx)
    import threading
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True).start()
    return srv, added


def test_post_is_accepted_and_buffered():
    cfg = AppConfig(ingest=IngestConfig(enabled=True, host="127.0.0.1", port=8731,
                                        vehicle_id_field="vehicleId"))
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
    imgs, vid, cam = ingest.extract_request("application/json",
                                            _j.dumps({"fullImage": b64, "vehicleId": "V1"}).encode(), ic)
    assert len(imgs) == 1

    # raw JPEG body
    imgs, vid, cam = ingest.extract_request("image/jpeg", raw, ic)
    assert len(imgs) == 1 and ingest.decode_image(imgs[0]) is not None

    # multipart/form-data with an image file part + text fields
    boundary = "X123"
    parts = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"vehicleId\"\r\n\r\nV9\r\n"
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"deviceId\"\r\n\r\ngate1\r\n"
    ).encode() + (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"p.jpg\"\r\n"
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode() + raw + f"\r\n--{boundary}--\r\n".encode()
    imgs, vid, cam = ingest.extract_request(f"multipart/form-data; boundary={boundary}", parts, ic)
    assert len(imgs) == 1 and vid == "V9" and cam == "gate1"


def test_post_raw_jpeg_body_is_accepted():
    cfg = AppConfig(ingest=IngestConfig(enabled=True, host="127.0.0.1", port=8741))
    added = []
    buf = ingest.BurstBuffer(cfg, lambda g: None)
    buf.add = lambda key, images, camera, vid: (added.append((key, len(images))) or len(images))
    ctx = ingest._Ctx(cfg, buf)
    srv = ingest.make_server(ctx)
    import threading
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True).start()
    try:
        import urllib.request
        req = urllib.request.Request("http://127.0.0.1:8741/api/v1/ingest", data=_jpeg_bytes(),
                                     headers={"Content-Type": "image/jpeg"}, method="POST")
        r = urllib.request.urlopen(req, timeout=5)
        body = json.loads(r.read())
    finally:
        srv.shutdown()
        srv.server_close()
    assert r.status == 200 and body["images"] == 1 and added


def test_respond_with_plate_returns_the_read_in_the_response():
    """respond_with_plate: the POST is voted inline and the reply carries the plate (not just an ack)."""
    cfg = AppConfig(ingest=IngestConfig(enabled=True, host="127.0.0.1", port=8742,
                                        respond_with_plate=True, vehicle_id_field="vehicleId"))
    seen = {}

    def fake_vote(images, camera, vehicle_id):
        seen["n"] = len(images)
        seen["vid"] = vehicle_id
        now = time.time()
        return [PlateEvent(plate="UP83DT0718", kind="standard", confidence=0.988, votes=1,
                           track_id=1, first_seen=now, last_seen=now, hsrp="non_hsrp")]

    pushed = {}

    def fake_push(events):
        pushed["n"] = len(events)
        return {"enabled": True, "ok": True, "status": 200, "message": "accepted (HTTP 200)",
                "reply": '{"errorCode":"1"}'}

    buf = ingest.BurstBuffer(cfg, lambda g: None)
    ctx = ingest._Ctx(cfg, buf, vote_fn=fake_vote, push_fn=fake_push)
    srv = ingest.make_server(ctx)
    import threading
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True).start()
    try:
        code, body = _post(8742, "/api/v1/ingest",
                           {"vehicleId": "V7", "BgImg": _jpeg_b64(), "PlateImg": _jpeg_b64(140, 48)})
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
    assert body["push"] == {"enabled": True, "ok": True, "status": 200,
                            "message": "accepted (HTTP 200)", "reply": '{"errorCode":"1"}'}
    assert pushed == {"n": 1}
    # both images of the one vehicle were voted together, inline
    assert seen == {"n": 2, "vid": "V7"}


def test_respond_with_plate_null_when_no_plate_read():
    cfg = AppConfig(ingest=IngestConfig(enabled=True, host="127.0.0.1", port=8743,
                                        respond_with_plate=True))
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
        "enabled": False, "message": ingest.PUSH_OFF_MESSAGE}
    # enabled but there was nothing to send
    r = ingest._push_result_json(None, enabled=True, url=url)
    assert r["enabled"] and r["ok"] is None and r["queued"] is False and "error" not in r
    assert r["url"] == url and r["plate"] is None
    # the client API could not be reached -> not delivered, kept and retried; the error names URL + plate
    down = SendResult(False, None, True, "cannot reach the server: [Errno 61] Connection refused")
    d = ingest._push_result_json(PushOutcome(False, True, down.message, down), enabled=True, url=url,
                                 plates=["UP83DT0718"])
    assert d == {**where, "enabled": True, "ok": False, "queued": True, "status": None,
                 "message": "cannot reach the server: [Errno 61] Connection refused", "reply": "",
                 "error": ("API is not connecting: http://192.168.1.72:21300/lane/anpr "
                           "(host 192.168.1.72:21300): [Errno 61] Connection refused. Plate UP83DT0718 is "
                           "saved on the Pi and is sent automatically when the API accepts it.")}
    # in backoff (no request made now): still reported as not connecting, with the reason
    b = ingest._push_result_json(
        PushOutcome(False, True, "queued: cannot reach the server: timed out; retrying in 7 s"),
        enabled=True, url=url, plates=["UP83DT0718"])
    assert b["error"].startswith(f"API is not connecting: {url} (host 192.168.1.72:21300): timed out.")
    assert "UP83DT0718" in b["error"]
    # waiting behind an earlier plate while the API is down: the error names the connection, not "busy"
    busy = ingest._push_result_json(
        PushOutcome(False, True, "queued: the sender is busy with earlier plates; sent shortly"),
        enabled=True, url=url, plates=["HR69D5319"], cause="cannot reach the server: timed out")
    assert busy["message"].startswith("queued: the sender is busy")
    assert busy["error"].startswith(f"API is not connecting: {url} (host 192.168.1.72:21300): timed out.")
    assert "HR69D5319" in busy["error"]
    # the client API accepted THIS plate, and its answer is echoed back in "reply"; no error
    ok = SendResult(True, 200, False, "accepted (HTTP 200)", '{"errorCode":"1"}')
    assert ingest._push_result_json(PushOutcome(True, False, ok.message, ok), enabled=True, url=url,
                                    plates=["UP83DT0718"]) == {
        **where, "enabled": True, "ok": True, "queued": False, "status": 200,
        "message": "accepted (HTTP 200)", "reply": '{"errorCode":"1"}'}
    # the API answered with an error -> named as an API error (not "not connecting"), plate included
    bad = SendResult(False, 500, True, "server error (HTTP 500)", "oops")
    e = ingest._push_result_json(PushOutcome(False, True, bad.message, bad), enabled=True, url=url,
                                 plates=["UP83DT0718"])
    assert e["status"] == 500 and e["error"].startswith(
        f"API at {url} (host 192.168.1.72:21300) did not accept plate UP83DT0718: server error (HTTP 500)")
    # refused for good -> not re-sent
    gone = ingest._push_result_json(PushOutcome(False, False, "refused"), enabled=True, url=url,
                                    plates=["UP83DT0718"])
    assert gone["error"].endswith("Plate UP83DT0718 is not sent again.")


def test_respond_with_plate_reports_api_not_connecting():
    """End to end through the real HTTP handler: the push could not connect, so the camera's reply still
    carries the plate (ok: true, HTTP 200) plus a top-level error naming the API URL and the plate."""
    from anpr.push import PushOutcome, SendResult

    url = "http://192.168.1.72:21300/lane/anpr"
    cfg = AppConfig(ingest=IngestConfig(enabled=True, host="127.0.0.1", port=8753,
                                        respond_with_plate=True))

    def push_down(events):
        down = SendResult(False, None, True, "cannot reach the server: timed out")
        return ingest._push_result_json(PushOutcome(False, True, down.message, down), enabled=True,
                                        url=url, plates=[e.plate for e in events])

    ctx = ingest._Ctx(cfg, ingest.BurstBuffer(cfg, lambda g: None),
                      vote_fn=lambda images, cam, vid: ([_ev("UP83DT0718")], []), push_fn=push_down)
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
    return PlateEvent(plate=plate, kind="standard", confidence=0.99, votes=1, track_id=1,
                      first_seen=now, last_seen=now, hsrp="non_hsrp")


def test_engine_reports_duplicates_only_when_asked(store):
    cfg = AppConfig(ingest=IngestConfig(enabled=True))
    eng = _engine(cfg, store, SeqOcr(["MH12AB1234"]))
    eng.collect_duplicates = True
    first: list = []
    assert [e.plate for e in ingest.process_group(eng, [_jpeg_bytes()] * 4, None, duplicates=first)] == [
        "MH12AB1234"]
    assert first == []
    again: list = []
    assert ingest.process_group(eng, [_jpeg_bytes()] * 4, None, duplicates=again) == []  # within 60 s
    assert [e.plate for e in again] == ["MH12AB1234"] and again[0].id is None
    # off (the video loop): never collected, so the list cannot grow forever
    eng2 = _engine(cfg, store, SeqOcr(["MH12AB1234"]))
    ingest.process_group(eng2, [_jpeg_bytes()] * 4, None)
    assert eng2.duplicates == []


def test_respond_with_plate_reports_a_duplicate_instead_of_null():
    cfg = AppConfig(ingest=IngestConfig(enabled=True, host="127.0.0.1", port=8751,
                                        respond_with_plate=True))
    pushed = []

    def fake_push(events):
        pushed.append(events)
        return ingest._push_result_json(None, enabled=True)

    ctx = ingest._Ctx(cfg, ingest.BurstBuffer(cfg, lambda g: None),
                      vote_fn=lambda images, cam, vid: ([], [_ev("UP83DT0718")]), push_fn=fake_push)
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
    cfg = AppConfig(ingest=IngestConfig(enabled=True, host="127.0.0.1", port=8752,
                                        respond_with_plate=True))

    def boom(events):
        raise RuntimeError("database is locked")

    ctx = ingest._Ctx(cfg, ingest.BurstBuffer(cfg, lambda g: None),
                      vote_fn=lambda images, cam, vid: [_ev("UP83DT0718")], push_fn=boom)
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
    cfg = AppConfig(ingest=IngestConfig(enabled=True, host="127.0.0.1", port=8753,
                                        respond_with_plate=True))
    slots = threading.BoundedSemaphore(1)
    ctx = ingest._Ctx(cfg, ingest.BurstBuffer(cfg, lambda g: None),
                      vote_fn=lambda images, cam, vid: [_ev("UP83DT0718")], slots=slots)
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
    cfg = AppConfig(ingest=IngestConfig(enabled=True, host="127.0.0.1", port=8754,
                                        respond_with_plate=True))
    slots = threading.BoundedSemaphore(1)
    ctx = ingest._Ctx(cfg, ingest.BurstBuffer(cfg, lambda g: None),
                      vote_fn=lambda images, cam, vid: [_ev("UP83DT0718")], slots=slots)
    srv = _serve_ctx(ctx)
    try:
        s = socket.create_connection(("127.0.0.1", 8754))
        s.sendall(b"POST /api/v1/ingest HTTP/1.0\r\nContent-Type: application/json\r\n"
                  b"Content-Length: 100000\r\n\r\n{")
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
    video rule's "unsure" (which unsure_as_non_hsrp then reports as non-HSRP)."""
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
    monkeypatch.setattr("anpr.preprocess.prepare_plate",
                        lambda frame, box, cfg: np.full((64, 128, 3), box.x1 % 256, np.uint8))


def _tile_engine(store, ocr, **ingest_kw):
    cfg = AppConfig(ingest=IngestConfig(enabled=True, **ingest_kw))
    cfg = cfg.model_copy(update={
        "hsrp": cfg.hsrp.model_copy(update={"enabled": False}),
        "vote": cfg.vote.model_copy(update={"min_votes": 1, "end_min_votes": 1}),  # one image per vehicle
    })
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
