"""Two cameras in one process (run_cameras): each pipeline is independent and every saved plate is
tagged with its camera's name, which then appears in the JSON sent to the client's API."""

from __future__ import annotations

import time

import numpy as np
import pytest

from anpr import engine as eng_mod
from anpr import push
from anpr.config import AppConfig, CameraFeed
from anpr.storage import SqliteEventStore
from anpr.types import PlateEvent


@pytest.fixture
def store(tmp_path):
    s = SqliteEventStore(tmp_path / "anpr.db", tmp_path / "images", snapshot_width=64)
    yield s
    s.close()


class FakeSource:
    """A handful of frames, then 'finished' - stands in for a camera or a short video file."""

    def __init__(self, cam):  # built from cfg_i.camera, like make_source
        self.cam = cam
        self._left = 3
        self.finished = False
        self.camera_ok = True

    def start(self):
        pass

    def read(self, timeout=1.0):
        if self._left <= 0:
            self.finished = True
            return None
        self._left -= 1
        return np.zeros((48, 160, 3), np.uint8), time.time()

    def stop(self):
        pass


class StubEngine:
    """Minimal Engine stand-in: each processed frame saves one plate tagged with this camera."""

    def __init__(self, camera):
        self.camera = camera
        self.store = None
        self.stats = type("S", (), {"frames": 0, "events": 0})()
        self._saved = False

    def bind(self, store):
        self.store = store
        return self

    def process(self, frame, ts):
        self.stats.frames += 1
        if not self._saved:  # one plate per camera is enough for the test
            self._saved = True
            self.store.add_event(
                PlateEvent(
                    plate=f"MH12{self.camera[:2].upper()}0001",
                    kind="standard",
                    confidence=0.9,
                    votes=3,
                    track_id=1,
                    first_seen=ts,
                    last_seen=ts,
                    hsrp="non_hsrp",
                    camera=self.camera,
                )
            )
            self.stats.events += 1

    def flush(self):
        pass


def test_two_cameras_tag_each_plate_with_its_camera(store, monkeypatch):
    monkeypatch.setattr(eng_mod, "Engine", None)  # must not build the real pipeline
    monkeypatch.setattr("anpr.camera.make_source", lambda cam: FakeSource(cam))

    cfg = AppConfig()
    feeds = [CameraFeed(source="rtsp://a", name="entry"), CameraFeed(source="rtsp://b", name="exit")]

    def make_engine(cfg_i, name):
        assert cfg_i.camera.source in ("rtsp://a", "rtsp://b")
        return StubEngine(name).bind(store)

    code = eng_mod.run_cameras(
        cfg, feeds, store, stop=lambda: False, exit_at_end=True, make_engine=make_engine
    )
    assert code == 0

    # read straight from the DB to avoid depending on the query API shape
    cams = sorted(e.camera for e in _all_events(store))
    assert cams == ["entry", "exit"]  # both cameras saved exactly one, correctly labelled


def _all_events(store):
    out = []
    i = 0
    while True:
        e = store.get_event(i := i + 1)
        if e is None and i > 10:
            break
        if e is not None:
            out.append(e)
    return out


def test_camera_name_flows_into_the_api_json(store):
    e = store.add_event(
        PlateEvent(
            plate="MH12AB1234", kind="standard", confidence=0.9, votes=3, track_id=1,
            first_seen=time.time() - 1, last_seen=time.time(), hsrp="hsrp", camera="exit",
        )
    )
    payload = push.event_payload(e, "pi-1", store, include_images=False)
    assert payload["camera"] == "exit"
    assert payload["plate"] == "MH12AB1234" and payload["hsrp"] == "hsrp"


def test_single_camera_has_blank_camera_field(store):
    e = store.add_event(
        PlateEvent(
            plate="MH12AB1234", kind="standard", confidence=0.9, votes=3, track_id=1,
            first_seen=time.time() - 1, last_seen=time.time(),
        )
    )
    assert e.camera is None
    assert push.event_payload(e, "pi-1", store, include_images=False)["camera"] == ""
