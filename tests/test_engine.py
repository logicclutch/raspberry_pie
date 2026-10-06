"""Engine logic with fake camera/detector/OCR (no models needed)."""

import logging
import time

import cv2
import numpy as np
import pytest

from anpr.config import AppConfig, RuntimeConfig, StorageConfig, VoteConfig
from anpr.engine import Engine, StreamControl, run
from anpr.storage import SqliteEventStore
from anpr.types import Box, OcrResult

FRAME = np.zeros((720, 1280, 3), np.uint8)


class FakeDetector:
    def __init__(self, boxes_per_frame):
        self.boxes = list(boxes_per_frame)
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return self.boxes.pop(0) if self.boxes else []


class FakeOcr:
    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = 0

    def read(self, crop):
        self.calls += 1
        t = self.texts.pop(0) if self.texts else "MH12AB1234"
        return None if t is None else OcrResult(t, tuple([0.95] * len(t)))


class FakeSource:
    def __init__(self, n):
        self.n = n
        self.i = 0
        self.stopped = False
        self.camera_ok = True

    def start(self):
        pass

    def read(self, timeout=1.0):
        if self.i >= self.n:
            return None
        self.i += 1
        return FRAME, 1000.0 + self.i * 0.3

    @property
    def finished(self):
        return self.i >= self.n

    def stop(self):
        self.stopped = True


def pbox(x=400):
    return Box(x, 400, x + 200, 450, 0.9)


# Most engine tests check the pipeline wiring with the vote deciding as soon as reads agree
# (settle_s 0, every read votes). The close-up / settling rule has its own tests at the end.
IMMEDIATE = VoteConfig(settle_s=0.0, close_ratio=0.0)


@pytest.fixture
def cfg(tmp_path):
    return AppConfig(
        storage=StorageConfig(db_path=tmp_path / "a.db", image_dir=tmp_path / "img"), vote=IMMEDIATE
    )


@pytest.fixture
def store(cfg):
    s = SqliteEventStore(cfg.storage.db_path, cfg.storage.image_dir)
    yield s
    s.close()


def make(cfg, store, det, ocr, motion=True):
    crop = np.full((50, 200, 3), 255, np.uint8)
    return Engine(
        cfg,
        det,
        ocr,
        store,
        motion=lambda f: motion,
        prepare=lambda frame, box, c: crop,
    )


def test_three_agreeing_frames_make_one_event(cfg, store):
    det = FakeDetector([[pbox(400)], [pbox(410)], [pbox(420)], [pbox(430)], [pbox(440)]])
    ocr = FakeOcr(["MH12AB1234"] * 5)
    eng = make(cfg, store, det, ocr)
    events = [e for i in range(5) for e in eng.process(FRAME, 1000 + i * 0.3)]
    assert [e.plate for e in events] == ["MH12AB1234"]
    assert events[0].votes == 3 and events[0].crop_path and events[0].snapshot_path
    assert ocr.calls == 3  # no OCR once the track is reported
    assert store.list_events()[1] == 1


def test_invalid_and_unreadable_reads_never_reported(cfg, store):
    det = FakeDetector([[pbox()]] * 6)
    ocr = FakeOcr(["XX12AB1234", None, "HELLO", "MH12AB1234", "MH12AB1234", "XX00"])
    eng = make(cfg, store, det, ocr)
    events = [e for i in range(6) for e in eng.process(FRAME, 1000 + i * 0.3)]
    assert events == []
    assert eng.stats.valid_reads == 2


def test_disagreeing_reads_are_not_reported(cfg, store):
    det = FakeDetector([[pbox()]] * 6)
    ocr = FakeOcr(["MH12AB1234", "MH12AB1284"] * 3)
    eng = make(cfg, store, det, ocr)
    assert [e for i in range(6) for e in eng.process(FRAME, 1000 + i * 0.3)] == []


def test_same_plate_again_within_window_is_deduplicated(cfg, store):
    # Vehicle passes, leaves (track expires), comes back 10 s later on a new track.
    frames = [[pbox()]] * 3 + [[]] * 1 + [[pbox(900)]] * 3
    times = [1000, 1000.3, 1000.6, 1005, 1010, 1010.3, 1010.6]
    det = FakeDetector(frames)
    eng = make(cfg, store, det, FakeOcr(["MH12AB1234"] * 6))
    events = [e for t in times for e in eng.process(FRAME, t)]
    assert len(events) == 1 and eng.stats.duplicates == 1


def test_detector_skipped_without_motion_or_tracks(cfg, store):
    det = FakeDetector([[pbox()]])
    eng = make(cfg, store, det, FakeOcr([]), motion=False)
    eng.process(FRAME, 1000)
    assert det.calls == 0


def test_detector_keeps_running_while_a_track_is_alive(cfg, store):
    det = FakeDetector([[pbox()], [pbox()], [pbox()]])
    moving = iter([True, False, False])
    eng = Engine(
        cfg,
        det,
        FakeOcr(["MH12AB1234"] * 3),
        store,
        motion=lambda f: next(moving),
        prepare=lambda frame, box, c: np.zeros((50, 200, 3), np.uint8),
    )
    events = [e for i in range(3) for e in eng.process(FRAME, 1000 + i * 0.3)]
    assert det.calls == 3  # a stopped car at a gate still gets read
    assert len(events) == 1


def test_run_loop_writes_status_and_stops_source(cfg, store):
    det = FakeDetector([[pbox()]] * 5)
    eng = make(cfg, store, det, FakeOcr(["MH12AB1234"] * 5))
    src = FakeSource(5)
    assert run(cfg, src, eng) == 0
    assert src.stopped
    st = store.read_status()
    assert st is not None and st.camera_ok and st.frames == 5 and st.events == 1


def test_run_loop_survives_processing_errors(cfg, store):
    class Boom:
        def detect(self, frame):
            raise RuntimeError("bad frame")

    eng = make(cfg, store, Boom(), FakeOcr([]))
    assert run(cfg, FakeSource(3), eng) == 0
    assert "bad frame" in store.read_status().last_error


# ---- managed mode: dashboard stream switching ----------------------------------------------------


@pytest.fixture
def fast_cfg(tmp_path):
    return AppConfig(
        storage=StorageConfig(db_path=tmp_path / "a.db", image_dir=tmp_path / "img"),
        runtime=RuntimeConfig(status_interval_s=0.01),
        vote=IMMEDIATE,
    )


class Opener:
    """StreamControl.open stand-in: records addresses, hands out short fake videos."""

    def __init__(self, frames=2, fail=()):
        self.frames = frames
        self.fail = set(fail)
        self.opened = []

    def __call__(self, address):
        if address in self.fail:
            raise OSError("no such camera")
        src = FakeSource(self.frames)
        src.address = address
        self.opened.append(src)
        return src


def test_managed_mode_waits_after_a_video_then_switches(fast_cfg, store):
    eng = make(fast_cfg, store, FakeDetector([]), FakeOcr([]))
    opener = Opener(frames=2)
    control = StreamControl(open=opener, default="0", check_every_s=0.0)
    first = FakeSource(3)
    seen = {"idle_status": None}

    def stop():
        if first.finished and not opener.opened:
            st = store.read_status()
            if st is not None and st.source_state == "ended":  # still alive, waiting for a stream
                seen["idle_status"] = st
                store.request_source("/videos/b.mp4")
        return bool(opener.opened) and opener.opened[0].stopped

    assert run(fast_cfg, first, eng, stop=stop, source_name="/videos/a.mp4", control=control) == 0
    assert first.stopped
    idle = seen["idle_status"]
    assert idle.source == "/videos/a.mp4" and not idle.camera_ok and idle.source_rev == 0
    assert [s.address for s in opener.opened] == ["/videos/b.mp4"]
    st = store.read_status()
    assert st.source == "/videos/b.mp4" and st.source_state == "ended" and st.source_rev == 1
    assert st.frames == 5  # counters keep running across streams


def test_switch_resets_tracks_and_votes(fast_cfg, store):
    eng = make(fast_cfg, store, FakeDetector([]), FakeOcr([]))
    tracker, voter = eng.tracker, eng.voter
    store.request_source("")  # "" -> back to the config camera
    opener = Opener(frames=1)
    control = StreamControl(open=opener, default="0", check_every_s=0.0)
    run(fast_cfg, FakeSource(10**6), eng, stop=lambda: bool(opener.opened), control=control)
    assert opener.opened[0].address == "0"
    assert eng.tracker is not tracker and eng.voter is not voter


def test_unopenable_stream_is_reported_not_fatal(fast_cfg, store):
    eng = make(fast_cfg, store, FakeDetector([]), FakeOcr([]))
    opener = Opener(fail={"rtsp://u:pw@10.0.0.9/x"})
    control = StreamControl(open=opener, default="0", check_every_s=0.0)
    store.request_source("rtsp://u:pw@10.0.0.9/x")
    calls = iter(range(10**6))
    run(fast_cfg, None, eng, stop=lambda: next(calls) > 3, control=control)
    st = store.read_status()
    assert st.source_state == "error" and not st.camera_ok and st.source_rev == 1
    assert "no such camera" in st.last_error
    assert "pw" not in st.source and "pw" not in st.last_error  # password never written out


def test_start_failure_managed_vs_unmanaged(fast_cfg, store):
    class Missing(FakeSource):
        def start(self):
            raise FileNotFoundError("video file not found: /x.mp4")

    eng = make(fast_cfg, store, FakeDetector([]), FakeOcr([]))
    with pytest.raises(FileNotFoundError):
        run(fast_cfg, Missing(1), eng)  # evaluation/tests: fail loudly
    control = StreamControl(open=Opener(), default="0", check_every_s=60.0)
    calls = iter(range(10**6))
    run(fast_cfg, Missing(1), eng, stop=lambda: next(calls) > 2, source_name="/x.mp4", control=control)
    st = store.read_status()
    assert st.source_state == "error" and "not found" in st.last_error


def test_unmanaged_run_still_ends_with_the_video(fast_cfg, store):
    eng = make(fast_cfg, store, FakeDetector([]), FakeOcr([]))
    src = FakeSource(4)
    assert run(fast_cfg, src, eng, source_name="/v.mp4") == 0
    assert src.stopped and store.read_status().frames == 4


# ---- live preview (dashboard "Live" view) ----------------------------------------------------------


def preview_cfg(tmp_path, interval=1e-6, **runtime):
    return AppConfig(
        storage=StorageConfig(db_path=tmp_path / "a.db", image_dir=tmp_path / "img"),
        runtime=RuntimeConfig(status_interval_s=0.01, preview_interval_s=interval, **runtime),
        vote=IMMEDIATE,
    )


class CountingWrites:
    """Wraps store.write_preview to count calls (and optionally fail)."""

    def __init__(self, store, fail=False):
        self.store, self.fail, self.calls, self.images = store, fail, 0, []
        self._orig = store.write_preview

    def __call__(self, image, quality=70):
        self.calls += 1
        self.images.append(image)
        if self.fail:
            raise OSError("disk full")
        return self._orig(image, quality)


def test_render_preview_draws_tracked_amber_and_confirmed_green(tmp_path, store):
    cfg = preview_cfg(tmp_path)
    det = FakeDetector([[pbox(400)], [pbox(400)], [pbox(400)]])
    eng = make(cfg, store, det, FakeOcr(["MH12AB1234"] * 3))
    assert eng.render_preview(640) is None  # nothing seen yet
    eng.process(FRAME, 1000.0)
    img = eng.render_preview(640)
    assert img.shape == (360, 640, 3)  # 1280x720 shrunk to 640 wide
    assert tuple(img[200, 250]) == (32, 176, 255)  # top edge of the box, scaled by 0.5: amber
    assert not FRAME.any()  # the camera frame itself is never drawn on
    eng.process(FRAME, 1000.3)
    eng.process(FRAME, 1000.6)  # third agreeing read -> confirmed
    img = eng.render_preview(640)
    assert tuple(img[224, 250]) == (140, 211, 55)  # bottom edge now green
    assert (img[185:199, 200:300] == (140, 211, 55)).all(axis=2).any()  # plate label above the box
    eng.reset()
    assert eng.render_preview(640) is None


def test_preview_keeps_small_frames_and_gray(tmp_path, store):
    eng = make(preview_cfg(tmp_path), store, FakeDetector([]), FakeOcr([]))
    eng.process(np.zeros((120, 160), np.uint8), 1000.0)
    img = eng.render_preview(640)
    assert img.shape == (120, 160, 3)


def test_run_writes_preview_every_frame_at_tiny_interval(tmp_path, store):
    cfg = preview_cfg(tmp_path)
    counter = CountingWrites(store)
    store.write_preview = counter
    eng = make(cfg, store, FakeDetector([[pbox()]] * 4), FakeOcr(["MH12AB1234"] * 4))
    before = time.time()
    assert run(cfg, FakeSource(4), eng) == 0
    assert counter.calls == 4
    got = store.read_preview()
    assert got is not None
    img = cv2.imdecode(np.frombuffer(got[0], np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (360, 640, 3)
    st = store.read_status()
    assert st.preview_ts is not None and st.preview_ts >= before - 1


def test_run_respects_preview_interval_and_zero_disables(tmp_path, store):
    cfg = preview_cfg(tmp_path, interval=3600.0)
    counter = CountingWrites(store)
    store.write_preview = counter
    run(cfg, FakeSource(5), make(cfg, store, FakeDetector([]), FakeOcr([])))
    assert counter.calls == 1  # the first frame, then not again within the hour

    cfg0 = preview_cfg(tmp_path, interval=0.0)
    counter0 = CountingWrites(store)
    store.write_preview = counter0
    run(cfg0, FakeSource(5), make(cfg0, store, FakeDetector([]), FakeOcr([])))
    assert counter0.calls == 0


def test_no_preview_without_an_open_stream(tmp_path, store):
    cfg = preview_cfg(tmp_path)
    counter = CountingWrites(store)
    store.write_preview = counter
    eng = make(cfg, store, FakeDetector([]), FakeOcr([]))
    control = StreamControl(open=Opener(fail={"rtsp://cam/x"}), default="0", check_every_s=0.0)
    store.request_source("rtsp://cam/x")
    calls = iter(range(10**6))
    run(cfg, None, eng, stop=lambda: next(calls) > 4, control=control)
    assert counter.calls == 0 and store.read_preview() is None
    assert store.read_status().preview_ts is None


def test_no_preview_after_the_video_ended(tmp_path, store):
    cfg = preview_cfg(tmp_path)
    counter = CountingWrites(store)
    store.write_preview = counter
    eng = make(cfg, store, FakeDetector([]), FakeOcr([]))
    control = StreamControl(open=Opener(), default="0", check_every_s=60.0)
    src = FakeSource(3)
    calls = iter(range(10**6))
    run(cfg, src, eng, stop=lambda: next(calls) > 8, control=control)  # keeps idling after the end
    assert src.finished and counter.calls == 3
    st = store.read_status()
    assert st.source_state == "ended" and st.preview_ts is not None  # web sees how old it is


def test_preview_write_failure_is_logged_once_and_not_fatal(tmp_path, store, caplog):
    cfg = preview_cfg(tmp_path)
    counter = CountingWrites(store, fail=True)
    store.write_preview = counter
    eng = make(cfg, store, FakeDetector([[pbox()]] * 6), FakeOcr(["MH12AB1234"] * 6))
    with caplog.at_level(logging.WARNING, logger="anpr.engine"):
        assert run(cfg, FakeSource(6), eng) == 0
    assert counter.calls == 6 and eng.stats.frames == 6 and eng.stats.events == 1
    assert sum("live preview" in r.getMessage() for r in caplog.records) == 1
    assert store.read_status().last_error is None  # not an engine error


def test_switch_drops_the_old_picture_and_shows_the_new_stream_at_once(tmp_path, store):
    cfg = preview_cfg(tmp_path, interval=3600.0)
    store.write_preview(np.zeros((8, 8, 3), np.uint8))  # left over from an earlier run
    counter = CountingWrites(store)
    store.write_preview = counter
    had_picture = []
    clear = store.clear_preview

    def counting_clear():
        had_picture.append(store.read_preview() is not None)
        clear()

    store.clear_preview = counting_clear
    eng = make(cfg, store, FakeDetector([]), FakeOcr([]))
    opener = Opener(frames=2)
    control = StreamControl(open=opener, default="0", check_every_s=0.0)
    first = FakeSource(3)
    seen = {}

    def stop():
        if first.finished and not opener.opened and "rev" not in seen:
            seen["rev"] = store.request_source("/videos/b.mp4")
        return bool(opener.opened) and opener.opened[0].finished

    run(cfg, first, eng, stop=stop, control=control)
    # Cleared at start (the earlier run's picture) and at the switch (the first video's picture).
    assert had_picture == [True, True]
    # One picture per stream: the new stream's first frame did not wait out the 1 h interval.
    assert counter.calls == 2
    assert store.read_preview() is not None


def test_current_rss_is_current_not_peak() -> None:
    import mmap

    from anpr.engine import current_rss_mb

    before = current_rss_mb()
    assert 5.0 < before < 4096.0
    # Anonymous mmap: returned to the OS on close (malloc may keep freed blocks cached on macOS).
    size = 200 * 1024 * 1024
    block = mmap.mmap(-1, size)
    for i in range(0, size, mmap.PAGESIZE):
        block[i] = 1  # touch every page so it is resident
    during = current_rss_mb()
    block.close()
    after = current_rss_mb()
    assert during > before + 100.0
    assert after < during - 100.0  # peak-based reading would stay high forever


def test_vehicle_leaving_with_two_identical_reads_is_saved_once(cfg, store):
    det = FakeDetector([[pbox()], [pbox()]])  # seen on 2 frames only, then gone
    ocr = FakeOcr(["HR51CX6945", "HR51CX6945"])
    cfg = cfg.model_copy(update={"vote": cfg.vote.model_copy(update={"end_min_conf": 0.9})})
    eng = make(cfg, store, det, ocr)
    events = []
    for i in range(20):
        events += eng.process(FRAME, 1000.0 + i * 0.3)
    assert [e.plate for e in events] == ["HR51CX6945"]
    assert events[0].votes == 2 and events[0].first_seen == 1000.0 and events[0].last_seen == 1000.3
    assert store.list_events()[1] == 1
    assert eng.stats.events == 1


def test_vehicle_leaving_with_disagreeing_reads_is_not_saved(cfg, store):
    det = FakeDetector([[pbox()], [pbox()]])
    ocr = FakeOcr(["HR51CX6945", "HR51CX6946"])
    cfg = cfg.model_copy(update={"vote": cfg.vote.model_copy(update={"end_min_conf": 0.9})})
    eng = make(cfg, store, det, ocr)
    events = []
    for i in range(20):
        events += eng.process(FRAME, 1000.0 + i * 0.3)
    assert events == [] and store.list_events()[1] == 0


def test_flush_gives_the_last_vehicle_its_end_of_track_check(cfg, store):
    det = FakeDetector([[pbox()], [pbox()]])  # video ends while the car is still in view
    ocr = FakeOcr(["HR51CX6945", "HR51CX6945"])
    cfg = cfg.model_copy(update={"vote": cfg.vote.model_copy(update={"end_min_conf": 0.9})})
    eng = make(cfg, store, det, ocr)
    assert eng.process(FRAME, 1000.0) == [] and eng.process(FRAME, 1000.3) == []
    events = eng.flush()
    assert [e.plate for e in events] == ["HR51CX6945"] and events[0].last_seen == 1000.3
    assert not eng.tracker.has_active()
    assert eng.flush() == []  # nothing left, never saved twice


def test_run_flushes_when_the_video_ends(cfg, store):
    det = FakeDetector([[pbox()], [pbox()]])
    ocr = FakeOcr(["HR51CX6945", "HR51CX6945"])
    cfg = cfg.model_copy(update={"vote": cfg.vote.model_copy(update={"end_min_conf": 0.9})})
    eng = make(cfg, store, det, ocr)
    assert run(cfg, FakeSource(2), eng) == 0
    assert [e.plate for e in store.list_events()[0]] == ["HR51CX6945"]


# ---- second-chance read from a wider crop ------------------------------------------------------


def make_reread(cfg, store, det, ocr, wide=True):
    crop = np.full((50, 200, 3), 255, np.uint8)
    wider = np.full((50, 240, 3), 250, np.uint8)
    calls = []

    def reread(frame, box, c):
        calls.append(box)
        return wider if wide else None

    eng = Engine(cfg, det, ocr, store, motion=lambda f: True, prepare=lambda f, b, c: crop, reread=reread)
    return eng, calls


def test_failed_read_is_rescued_by_the_wider_crop(cfg, store):
    det = FakeDetector([[pbox(400 + 10 * i)] for i in range(3)])
    ocr = FakeOcr(["MH12AB123X", "MH12AB1234"] * 3)  # first read fails the format check, re-read is fine
    eng, calls = make_reread(cfg, store, det, ocr)
    events = [e for i in range(3) for e in eng.process(FRAME, 1000 + i * 0.3)]
    assert [e.plate for e in events] == ["MH12AB1234"] and events[0].votes == 3
    assert len(calls) == 3 and eng.stats.rereads == 3 and eng.stats.reread_fixes == 3
    assert ocr.calls == 6 and eng.stats.ocr_reads == 3  # ocr_reads counts first reads only


def test_valid_first_read_never_rereads(cfg, store):
    det = FakeDetector([[pbox(400 + 10 * i)] for i in range(3)])
    eng, calls = make_reread(cfg, store, det, FakeOcr(["MH12AB1234"] * 3))
    assert [e.plate for i in range(3) for e in eng.process(FRAME, 1000 + i * 0.3)] == ["MH12AB1234"]
    assert calls == [] and eng.stats.rereads == 0


def test_reread_that_also_fails_saves_nothing(cfg, store):
    det = FakeDetector([[pbox(400 + 10 * i)] for i in range(3)])
    eng, calls = make_reread(cfg, store, det, FakeOcr(["HELLO"] * 6))
    assert [e for i in range(3) for e in eng.process(FRAME, 1000 + i * 0.3)] == []
    assert eng.stats.rereads == 3 and eng.stats.reread_fixes == 0 and eng.stats.valid_reads == 0


def test_no_wider_crop_means_no_second_read(cfg, store):
    det = FakeDetector([[pbox(400)]])
    ocr = FakeOcr(["HELLO"])
    eng, calls = make_reread(cfg, store, det, ocr, wide=False)
    assert eng.process(FRAME, 1000.0) == []
    assert len(calls) == 1 and ocr.calls == 1 and eng.stats.rereads == 0


def test_injected_prepare_has_no_reread_by_default(cfg, store):
    eng = make(cfg, store, FakeDetector([]), FakeOcr([]))
    assert eng.reread is None


# ---- close-up voting: an approaching vehicle is decided from its close-up reads -------------------


def growing(x, w):
    return Box(x, 400, x + w, 400 + w // 4, 0.9)


def test_far_misreads_do_not_vote_once_the_vehicle_is_close(tmp_path):
    """TN33BY9603 case: 3 blurred far reads agree on a wrong text, close reads are right."""
    cfg = AppConfig(storage=StorageConfig(db_path=tmp_path / "a.db", image_dir=tmp_path / "img"))
    store = SqliteEventStore(cfg.storage.db_path, cfg.storage.image_dir)
    widths = [120, 122, 124, 126] + [160 + 15 * i for i in range(8)] + [270] * 8
    det = FakeDetector([[growing(400, w)] for w in widths])
    ocr = FakeOcr(["TN33BT9603"] * 4 + ["TN33BY9603"] * 16)
    eng = make(cfg, store, det, ocr)
    events = [e for i in range(len(widths)) for e in eng.process(FRAME, 1000 + i * 0.2)]
    assert [e.plate for e in events] == ["TN33BY9603"]
    store.close()


def test_nothing_is_decided_while_the_plate_is_still_growing(tmp_path):
    cfg = AppConfig(storage=StorageConfig(db_path=tmp_path / "a.db", image_dir=tmp_path / "img"))
    store = SqliteEventStore(cfg.storage.db_path, cfg.storage.image_dir)
    widths = [100 + 20 * i for i in range(10)]  # keeps approaching for 2 s
    eng = make(cfg, store, FakeDetector([[growing(400, w)] for w in widths]), FakeOcr(["MH12AB1234"] * 10))
    assert [e for i in range(10) for e in eng.process(FRAME, 1000 + i * 0.2)] == []
    # ...then it leaves: decided at the end of the track from the close-up reads
    events = [e for i in range(10, 40) for e in eng.process(FRAME, 1000 + i * 0.2)]
    assert [e.plate for e in events] == ["MH12AB1234"]
    store.close()


def test_a_stopped_vehicle_is_decided_without_waiting_for_it_to_leave(tmp_path):
    cfg = AppConfig(storage=StorageConfig(db_path=tmp_path / "a.db", image_dir=tmp_path / "img"))
    store = SqliteEventStore(cfg.storage.db_path, cfg.storage.image_dir)
    widths = [150, 180, 210] + [230] * 12  # arrives, then stands at the barrier
    eng = make(cfg, store, FakeDetector([[growing(400, w)] for w in widths]), FakeOcr(["MH12AB1234"] * 15))
    events = [(i, e) for i in range(15) for e in eng.process(FRAME, 1000 + i * 0.2)]
    assert [e.plate for _, e in events] == ["MH12AB1234"]
    assert events[0][0] <= 10  # within ~1.4 s of stopping, while it is still in view
    store.close()


def test_close_reads_that_disagree_save_nothing(tmp_path):
    """KA plate case: small plate, close reads split between two texts -> not sure."""
    cfg = AppConfig(storage=StorageConfig(db_path=tmp_path / "a.db", image_dir=tmp_path / "img"))
    store = SqliteEventStore(cfg.storage.db_path, cfg.storage.image_dir)
    widths = [116, 128, 133, 140, 124, 150, 155]
    det = FakeDetector([[growing(400, w)] for w in widths])
    ocr = FakeOcr(["KA03M1035", "KA03M1035", "KA03M1035", "KA03M1005", "KA03M1005", "KA03M1005", "KA03M1035"])
    eng = make(cfg, store, det, ocr)
    assert [e for i in range(40) for e in eng.process(FRAME, 1000 + i * 0.2)] == []
    store.close()


def test_far_misreads_do_not_vote_when_the_close_ups_never_read(tmp_path):
    """The vehicle came close but no close-up frame gave a valid read: the plate sizes seen close up
    still set the floor, so the 4 agreeing far misreads (TN33BT9603) are not saved."""
    cfg = AppConfig(storage=StorageConfig(db_path=tmp_path / "a.db", image_dir=tmp_path / "img"))
    store = SqliteEventStore(cfg.storage.db_path, cfg.storage.image_dir)
    widths = [120, 122, 124, 126] + [160 + 15 * i for i in range(8)] + [270] * 8
    det = FakeDetector([[growing(400, w)] for w in widths])
    eng = make(cfg, store, det, FakeOcr(["TN33BT9603"] * 4 + ["TN33BY96O"] * 16))
    events = [e for i in range(len(widths) + 20) for e in eng.process(FRAME, 1000 + i * 0.2)]
    assert events == []
    store.close()


def two_line_box(x=400):
    # 70 x 60: a whole 2-line plate, at the top of the frame so there is no band above it to read
    return Box(x, 10, x + 70, 70, 0.9)


def test_two_line_plate_boxed_whole_reads_its_top_half(cfg, store):
    """HR38AA3075 case: the box covers both lines but the reader returns only the bottom line;
    the upper half of the box gives the top line, which may carry a series letter ("HR38A"), as
    whole-plate reads of the same track (cut short: "HR38AA307") confirm."""
    det = FakeDetector([[two_line_box(400 + 2 * i)] for i in range(5)])
    eng = make(cfg, store, det, FakeOcr(["HR38AA307", "HR38AA307"] + ["A3075", "HR38A"] * 3))
    events = [e for i in range(5) for e in eng.process(FRAME, 1000 + i * 0.2)]
    assert [e.plate for e in events] == ["HR38AA3075"]
    assert eng.stats.top_line_fixes == 3


def test_remembered_head_joins_the_bottom_line(cfg, store):
    """TN36BC8199 case: far away the whole plate reads "TN36BC819"; close up only "BC8199"."""
    det = FakeDetector([[pbox(400 + 5 * i)] for i in range(5)])
    # far: 2 cut-short whole reads; close: bottom line, and nothing readable above the box
    eng = make(cfg, store, det, FakeOcr(["TN36BC819", "TN36BC819"] + ["BC8199", None] * 3))
    events = [e for i in range(5) for e in eng.process(FRAME, 1000 + i * 0.2)]
    assert [e.plate for e in events] == ["TN36BC8199"]


def test_far_reads_that_lost_the_series_letters_do_not_veto_the_join(cfg, store):
    """TN36BC8199 in the WhatsApp video: far away the reader often drops "BC" ("TN36981", 3 reads)
    and only sometimes keeps it ("TN36BC819", 2 reads). Reads with no series letter say nothing
    about how many the plate has, so "TN36" + "BC8199" is still joined and saved."""
    det = FakeDetector([[pbox(400 + 5 * i)] for i in range(8)])
    eng = make(cfg, store, det, FakeOcr(["TN36981"] * 3 + ["TN36BC819"] * 2 + ["BC8199", None] * 3))
    events = [e for i in range(8) for e in eng.process(FRAME, 1000 + i * 0.2)]
    assert [e.plate for e in events] == ["TN36BC8199"]


def test_remembered_head_is_not_joined_when_it_would_drop_a_series_letter(cfg, store):
    """Whole reads show 2 series letters after "HR38" ("HR38AA307"); the bottom line "A3075" has
    only 1, so the top line holds the other. "HR38" + "A3075" = HR38A3075 would be a wrong plate."""
    det = FakeDetector([[pbox(400 + 5 * i)] for i in range(5)])
    eng = make(cfg, store, det, FakeOcr(["HR38AA307", "HR38AA307"] + ["A3075", None] * 3))
    assert [e for i in range(5) for e in eng.process(FRAME, 1000 + i * 0.2)] == []
    assert eng.stats.head_fixes == 0


def test_top_line_join_needs_the_series_letters_the_whole_reads_show(cfg, store):
    """Whole reads say "HR38" + 2 series letters. The top half reads "HR38" (its trailing letter
    lost) and the bottom "A3075": HR38A3075 validates but has 1 letter, so it is not saved."""
    det = FakeDetector([[two_line_box(400 + 2 * i)] for i in range(5)])
    eng = make(cfg, store, det, FakeOcr(["HR38AA307", "HR38AA307"] + ["A3075", "HR38"] * 3))
    assert [e for i in range(5) for e in eng.process(FRAME, 1000 + i * 0.2)] == []
    assert eng.stats.top_line_fixes == 0


def test_top_line_with_a_letter_needs_whole_reads_behind_it(cfg, store):
    """The bottom line bleeding into the top crop ("TN36B" over "BC8199") would validate as a
    3-letter series (TN36BBC8199). With no whole-plate read showing 3 letters it is refused."""
    det = FakeDetector([[two_line_box(400 + 2 * i)] for i in range(3)])
    eng = make(cfg, store, det, FakeOcr(["BC8199", "TN36B"] * 3))
    assert [e for i in range(3) for e in eng.process(FRAME, 1000 + i * 0.2)] == []


def test_a_tie_in_series_letter_counts_is_not_agreement(cfg, store):
    """One whole read shows 1 series letter, one shows 2: the count is unknown, so no head join."""
    det = FakeDetector([[pbox(400 + 5 * i)] for i in range(5)])
    eng = make(cfg, store, det, FakeOcr(["HR38A43075", "HR38AA307"] + ["A3075", None] * 3))
    assert [e for i in range(5) for e in eng.process(FRAME, 1000 + i * 0.2)] == []
    assert eng.stats.head_fixes == 0


def test_top_line_of_state_and_district_joins_without_whole_reads(cfg, store):
    det = FakeDetector([[two_line_box(400 + 2 * i)] for i in range(3)])
    eng = make(cfg, store, det, FakeOcr(["AA3075", "HR38"] * 3))
    events = [e for i in range(3) for e in eng.process(FRAME, 1000 + i * 0.2)]
    assert [e.plate for e in events] == ["HR38AA3075"]


# ---- playlist (camera.playlist) ------------------------------------------------------------------


def test_playlist_plays_videos_one_after_another_then_waits(fast_cfg, store):
    eng = make(fast_cfg, store, FakeDetector([]), FakeOcr([]))
    opener = Opener(frames=2)
    control = StreamControl(open=opener, default="0", check_every_s=60.0, playlist=("/v/a.mp4", "/v/b.mp4"))
    assert control.target("") == "/v/a.mp4"
    first = FakeSource(3)
    calls = iter(range(10**6))
    run(fast_cfg, first, eng, stop=lambda: next(calls) > 50, source_name="/v/a.mp4", control=control)
    assert first.stopped
    assert [s.address for s in opener.opened] == ["/v/b.mp4"]  # b started by itself, then no more
    st = store.read_status()
    assert st.source == "/v/b.mp4" and st.source_state == "ended" and st.frames == 5


def test_playlist_loops_back_to_the_first_video(fast_cfg, store):
    eng = make(fast_cfg, store, FakeDetector([]), FakeOcr([]))
    opener = Opener(frames=1)
    control = StreamControl(
        open=opener, default="0", check_every_s=60.0, playlist=("/v/a.mp4", "/v/b.mp4"), loop_playlist=True
    )
    control.target("")
    calls = iter(range(10**6))  # bounded, so a broken playlist fails instead of hanging
    run(
        fast_cfg,
        FakeSource(1),
        eng,
        stop=lambda: len(opener.opened) >= 4 or next(calls) > 200,
        control=control,
    )
    assert [s.address for s in opener.opened] == ["/v/b.mp4", "/v/a.mp4", "/v/b.mp4", "/v/a.mp4"]


def test_dashboard_stream_leaves_the_playlist_and_default_returns_to_it(fast_cfg, store):
    eng = make(fast_cfg, store, FakeDetector([]), FakeOcr([]))
    opener = Opener(frames=1)
    control = StreamControl(
        open=opener, default="0", check_every_s=0.0, playlist=("/v/a.mp4", "/v/b.mp4"), loop_playlist=True
    )
    control.target("")
    store.request_source("rtsp://cam/x")
    calls = iter(range(10**6))
    run(fast_cfg, FakeSource(10**6), eng, stop=lambda: next(calls) > 30, control=control)
    assert [s.address for s in opener.opened] == ["rtsp://cam/x"]  # its end does not resume the list
    assert control.playlist_pos is None
    store.request_source("")  # "back to default" = start of the playlist
    calls = iter(range(10**6))
    run(fast_cfg, None, eng, stop=lambda: len(opener.opened) >= 3 or next(calls) > 200, control=control)
    assert [s.address for s in opener.opened][1:] == ["/v/a.mp4", "/v/b.mp4"]


def test_playlist_config_from_yaml(tmp_path):
    from anpr.config import load_config

    p = tmp_path / "c.yaml"
    p.write_text("camera:\n  loop: true\n  playlist:\n    - /v/a.mp4\n    - /v/b.mp4\n")
    assert load_config(p).camera.playlist == ("/v/a.mp4", "/v/b.mp4")


# ---- HSRP check ---------------------------------------------------------------------------------


def make_hsrp(cfg, store, det, ocr, looks):
    crop = np.full((50, 200, 3), 255, np.uint8)
    seen = []

    def look(frame, box):
        seen.append(box)
        v = looks[len(seen) - 1]
        if isinstance(v, Exception):
            raise v
        return v

    eng = Engine(
        cfg, det, ocr, store, motion=lambda f: True, prepare=lambda frame, box, c: crop, hsrp_look=look
    )
    return eng, seen


def test_saved_plate_carries_the_hsrp_result_of_its_frames(cfg, store):
    det = FakeDetector([[pbox(400 + 10 * i)] for i in range(3)])
    eng, seen = make_hsrp(cfg, store, det, FakeOcr(["MH12AB1234"] * 3), ["hsrp", "non_hsrp", "hsrp"])
    events = [e for i in range(3) for e in eng.process(FRAME, 1000 + i * 0.3)]
    assert [e.hsrp for e in events] == ["hsrp"]
    assert store.get_event(events[0].id).hsrp == "hsrp"
    assert seen == [pbox(400), pbox(410), pbox(420)]  # judged on the plate box of each valid read


def test_hsrp_looks_are_per_track_and_forgotten_when_it_ends(cfg, store):
    # track 1: three clean looks -> non-HSRP; the next car (new track) starts from nothing
    frames = [[pbox(400)], [pbox(410)], [pbox(420)], [], [], [], [], [], [], [], [], [], [pbox(900)]]
    det = FakeDetector(frames + [[pbox(910)], [pbox(920)]])
    ocr = FakeOcr(["MH12AB1234"] * 3 + ["KA01MX0001"] * 3)
    eng, _ = make_hsrp(cfg, store, det, ocr, ["non_hsrp"] * 3 + ["hsrp"] * 3)
    events = [e for i in range(len(frames) + 2) for e in eng.process(FRAME, 1000 + i * 0.5)]
    assert [(e.plate, e.hsrp) for e in events] == [("MH12AB1234", "non_hsrp"), ("KA01MX0001", "hsrp")]
    assert eng._hsrp.keys() <= {t for t in eng.tracker.tracks}


def test_failing_hsrp_check_never_costs_the_plate(cfg, store):
    det = FakeDetector([[pbox(400 + 10 * i)] for i in range(3)])
    eng, _ = make_hsrp(cfg, store, det, FakeOcr(["MH12AB1234"] * 3), [RuntimeError("boom")] * 3)
    events = [e for i in range(3) for e in eng.process(FRAME, 1000 + i * 0.3)]
    assert [(e.plate, e.hsrp) for e in events] == [("MH12AB1234", "non_hsrp")]  # mark never seen


@pytest.mark.parametrize(("as_non", "want"), [(True, "non_hsrp"), (False, "unsure")])
def test_plate_that_cannot_be_judged_is_non_hsrp_unless_switched_off(cfg, store, as_non, want):
    # blurred / 2-line plate: no frame could be judged (looks are None)
    cfg = cfg.model_copy(update={"hsrp": cfg.hsrp.model_copy(update={"unsure_as_non_hsrp": as_non})})
    det = FakeDetector([[pbox(400 + 10 * i)] for i in range(3)])
    eng, _ = make_hsrp(cfg, store, det, FakeOcr(["MH12AB1234"] * 3), [None] * 3)
    events = [e for i in range(3) for e in eng.process(FRAME, 1000 + i * 0.3)]
    assert [(e.plate, e.hsrp) for e in events] == [("MH12AB1234", want)]


def test_hsrp_check_can_be_switched_off(tmp_path):
    from anpr.config import HsrpConfig

    cfg = AppConfig(
        storage=StorageConfig(db_path=tmp_path / "a.db", image_dir=tmp_path / "img"),
        vote=IMMEDIATE,
        hsrp=HsrpConfig(enabled=False),
    )
    store = SqliteEventStore(cfg.storage.db_path, cfg.storage.image_dir)
    try:
        det = FakeDetector([[pbox(400 + 10 * i)] for i in range(3)])
        eng = make(cfg, store, det, FakeOcr(["MH12AB1234"] * 3))
        assert eng.hsrp_look is None
        events = [e for i in range(3) for e in eng.process(FRAME, 1000 + i * 0.3)]
        assert [(e.plate, e.hsrp) for e in events] == [("MH12AB1234", None)]
    finally:
        store.close()


def test_real_hsrp_check_runs_by_default(cfg, store):
    det = FakeDetector([[pbox(400 + 10 * i)] for i in range(3)])
    eng = make(cfg, store, det, FakeOcr(["MH12AB1234"] * 3))
    events = [e for i in range(3) for e in eng.process(FRAME, 1000 + i * 0.3)]
    assert [(e.plate, e.hsrp) for e in events] == [("MH12AB1234", "non_hsrp")]  # black frame: mark not seen


def test_hsrp_check_stops_after_enough_looks(tmp_path):
    from anpr.engine import HSRP_MAX_LOOKS

    cfg = AppConfig(storage=StorageConfig(db_path=tmp_path / "a.db", image_dir=tmp_path / "img"))
    store = SqliteEventStore(cfg.storage.db_path, cfg.storage.image_dir)
    try:
        n = HSRP_MAX_LOOKS + 8
        # parked car (no decision): the plate grows for 4 frames, then stays the same size
        widths = [150, 160, 170, 180] + [200] * (n - 4)
        det = FakeDetector([[Box(400, 400, 400 + w, 450, 0.9)] for w in widths])
        cfg_votes = cfg.model_copy(update={"vote": VoteConfig(min_votes=n + 1, end_min_votes=0)})
        looks = ["non_hsrp"] * 4 + ["hsrp"] * (n - 4)  # far frames blurred: no mark seen
        eng, seen = make_hsrp(cfg_votes, store, det, FakeOcr(["MH12AB1234"] * n), looks)
        for i in range(n):
            eng.process(FRAME, 1000 + i * 0.3)
        # first 12 frames judged; later ones (not bigger than the smallest kept) are skipped, but
        # the 4 far-away frames were each replaced by a bigger one before that
        kept = eng._hsrp[next(iter(eng._hsrp))]
        assert len(kept) == HSRP_MAX_LOOKS and min(w for w, _ in kept) == 200
        assert len(seen) == HSRP_MAX_LOOKS + 4
        assert eng._hsrp_result(next(iter(eng._hsrp))) == "hsrp"
    finally:
        store.close()
