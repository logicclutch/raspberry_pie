"""Frame sources: video file (lossless + realtime), read semantics, rpicam-vid via a fake command."""

from __future__ import annotations

import sys
import textwrap
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from anpr.camera import OpenCVSource, RpiCamSource, make_source
from anpr.config import CameraConfig

N_FRAMES = 20
W, H = 64, 48
FPS = 20.0


def frame_value(i: int) -> int:
    return 10 + i * 12  # 10..238, well separated so mp4v artefacts can't confuse neighbours


def decode_index(frame: np.ndarray) -> int:
    return round((float(frame.mean()) - 10) / 12)


@pytest.fixture(scope="module")
def video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("vid") / "tiny.mp4"
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    assert wr.isOpened()
    for i in range(N_FRAMES):
        wr.write(np.full((H, W, 3), frame_value(i), np.uint8))
    wr.release()
    return path


def drain(src, deadline_s: float = 5.0) -> list[tuple[np.ndarray, float]]:
    out = []
    end = time.monotonic() + deadline_s
    while not src.finished and time.monotonic() < end:
        item = src.read(timeout=0.5)
        if item is not None:
            out.append(item)
    return out


def test_file_lossless_delivers_every_frame_in_order(video: Path) -> None:
    src = OpenCVSource(CameraConfig(source=str(video), realtime=False))
    src.start()
    try:
        items = drain(src)
        assert src.finished
        assert [decode_index(f) for f, _ in items] == list(range(N_FRAMES))
        assert items[0][0].shape == (H, W, 3)
        # synthetic clock: start_ts + index / fps
        ts = [t for _, t in items]
        assert np.allclose(np.diff(ts), 1.0 / FPS, atol=1e-6)
        assert src.read(timeout=0.05) is None  # finished: returns immediately
    finally:
        src.stop()


@pytest.mark.parametrize("threads", [0, 1, 2])
def test_file_decode_threads_limit_and_still_every_frame(video: Path, threads: int) -> None:
    src = OpenCVSource(CameraConfig(source=str(video), realtime=False, decode_threads=threads))
    cap = src._open_file()
    try:
        assert cap.isOpened()
        if threads:
            assert cap.get(cv2.CAP_PROP_N_THREADS) == threads
    finally:
        cap.release()
    src.start()
    try:
        assert [decode_index(f) for f, _ in drain(src)] == list(range(N_FRAMES))
    finally:
        src.stop()


def test_decode_threads_default_and_bounds() -> None:
    assert CameraConfig().decode_threads == 2
    with pytest.raises(ValueError):
        CameraConfig(decode_threads=-1)


def test_file_realtime_drops_frames_for_slow_consumer(video: Path) -> None:
    src = OpenCVSource(CameraConfig(source=str(video), realtime=True))
    src.start()
    try:
        got = []
        end = time.monotonic() + 5.0
        while not src.finished and time.monotonic() < end:
            item = src.read(timeout=0.5)
            if item is not None:
                got.append(decode_index(item[0]))
                time.sleep(0.15)  # 3 frame periods at 20 fps
        assert src.finished
        assert 2 <= len(got) < N_FRAMES
        assert got == sorted(set(got))  # increasing, no duplicates
    finally:
        src.stop()


def test_read_never_repeats_and_respects_timeout(video: Path) -> None:
    src = OpenCVSource(CameraConfig(source=str(video), realtime=False))
    src.start()
    try:
        first = src.read(timeout=2.0)
        assert first is not None
        # Consumer took frame 0; grabber publishes frame 1 then waits. Take it, then the next ones.
        seen = [decode_index(first[0])]
        for _ in range(3):
            item = src.read(timeout=2.0)
            assert item is not None
            seen.append(decode_index(item[0]))
        assert seen == [0, 1, 2, 3]
    finally:
        src.stop()
    # after stop nothing new arrives, and read returns promptly
    t0 = time.monotonic()
    assert src.read(timeout=0.3) is None
    assert time.monotonic() - t0 < 0.5


def test_read_timeout_when_no_frames() -> None:
    src = RpiCamSource(
        CameraConfig(source="picamera", width=64, height=48),
        command=[sys.executable, "-c", "import time; time.sleep(5)"],
    )
    src.start()
    try:
        t0 = time.monotonic()
        assert src.read(timeout=0.2) is None
        assert 0.15 <= time.monotonic() - t0 < 0.5
        assert not src.camera_ok
    finally:
        src.stop()


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        OpenCVSource(CameraConfig(source=str(tmp_path / "nope.mp4"))).start()


def test_dead_stream_reports_not_ok_and_stops_fast() -> None:
    src = OpenCVSource(CameraConfig(source="rtsp://127.0.0.1:1/none"))
    src.backoff_min = 0.05
    src.backoff_max = 0.1
    src.start()
    try:
        assert src.read(timeout=0.3) is None
        assert not src.camera_ok
        assert not src.finished
    finally:
        t0 = time.monotonic()
        src.stop()
        assert time.monotonic() - t0 < 6.0


def test_make_source() -> None:
    assert isinstance(make_source(CameraConfig(source="picamera")), RpiCamSource)
    assert isinstance(make_source(CameraConfig(source="0")), OpenCVSource)
    cmd = RpiCamSource.default_command(CameraConfig(source="picamera", exposure_us=1000))
    assert cmd[0] == "rpicam-vid" and "--shutter" in cmd and cmd[-2:] == ["-o", "-"]
    assert "--shutter" not in RpiCamSource.default_command(CameraConfig(source="picamera", exposure_us=None))


# -- RpiCamSource with a fake rpicam-vid -------------------------------------------------------------


def fake_rpicam(tmp_path: Path, n_frames: int, interval: float, w: int = W, h: int = H) -> list[str]:
    """Script writing I420 frames: Y = 16 + 20 * (i % 10), U = V = 128 (neutral grey)."""
    script = tmp_path / "fake_rpicam.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import sys, time
            w, h = {w}, {h}
            out = sys.stdout.buffer
            for i in range({n_frames}):
                y = bytes([16 + 20 * (i % 10)]) * (w * h)
                uv = bytes([128]) * (w * h // 2)
                out.write(y + uv)
                out.flush()
                time.sleep({interval})
            """
        )
    )
    return [sys.executable, str(script)]


def test_rpicam_converts_i420_and_keeps_latest(tmp_path: Path) -> None:
    cmd = fake_rpicam(tmp_path, n_frames=10, interval=0.02)
    src = RpiCamSource(CameraConfig(source="picamera", width=W, height=H), command=cmd)
    src.start()
    try:
        first = src.read(timeout=3.0)
        assert first is not None
        frame, ts = first
        assert frame.shape == (H, W, 3) and frame.dtype == np.uint8
        assert abs(ts - time.time()) < 2.0
        # neutral chroma -> B == G == R; BT.601 limited range: Y=16 -> ~0 ... Y=196 -> ~210
        assert int(frame.max()) - int(frame.min()) <= 2
        time.sleep(0.4)  # the fake finishes its 10 frames meanwhile; only the latest is kept
        last = src.read(timeout=1.0)
        assert last is not None
        expected = cv2.cvtColor(
            np.concatenate([np.full((H, W), 16 + 20 * 9, np.uint8), np.full((H // 2, W), 128, np.uint8)]),
            cv2.COLOR_YUV2BGR_I420,
        )
        assert np.array_equal(last[0], expected)
        assert src.read(timeout=0.05) is None  # already taken; the restart backoff is >= 1 s
    finally:
        src.stop()


def test_rpicam_stop_kills_process(tmp_path: Path) -> None:
    cmd = fake_rpicam(tmp_path, n_frames=100000, interval=0.01)
    src = RpiCamSource(CameraConfig(source="picamera", width=W, height=H), command=cmd)
    src.start()
    assert src.read(timeout=3.0) is not None
    assert src.camera_ok
    proc = src._proc
    assert proc is not None and proc.poll() is None
    t0 = time.monotonic()
    src.stop()
    assert time.monotonic() - t0 < 3.0
    assert proc.poll() is not None
    assert not src.camera_ok


def test_rpicam_restarts_after_process_exit(tmp_path: Path) -> None:
    cmd = fake_rpicam(tmp_path, n_frames=2, interval=0.01)
    src = RpiCamSource(CameraConfig(source="picamera", width=W, height=H), command=cmd)
    src.backoff_min = 0.05
    src.backoff_max = 0.1
    src.start()
    try:
        end = time.monotonic() + 5.0
        while src.restarts < 3 and time.monotonic() < end:
            src.read(timeout=0.2)
        assert src.restarts >= 3
        assert src.read(timeout=2.0) is not None  # frames keep coming after restarts
        assert not src.finished
    finally:
        src.stop()


def test_rpicam_missing_binary_retries(tmp_path: Path) -> None:
    src = RpiCamSource(
        CameraConfig(source="picamera", width=W, height=H), command=[str(tmp_path / "no-such-rpicam-vid")]
    )
    src.backoff_min = 0.05
    src.start()
    try:
        assert src.read(timeout=0.2) is None
        assert not src.camera_ok
    finally:
        src.stop()


def test_rpicam_rejects_odd_size() -> None:
    with pytest.raises(ValueError):
        RpiCamSource(CameraConfig(source="picamera", width=63, height=48))


def test_logs_never_show_the_rtsp_password(caplog):
    import logging

    from anpr.camera import OpenCVSource
    from anpr.config import CameraConfig

    src = OpenCVSource(CameraConfig(source="rtsp://admin:hunter2@127.0.0.1:9/x"))
    src.backoff_min = src.backoff_max = 0.05
    with caplog.at_level(logging.INFO, logger="anpr.camera"):
        src.start()
        src.read(timeout=0.5)
        src.stop()
    assert "open failed" in caplog.text
    assert "hunter2" not in caplog.text and "admin:***@127.0.0.1" in caplog.text


def test_file_loop_plays_the_video_again(video: Path) -> None:
    src = OpenCVSource(CameraConfig(source=str(video), realtime=True, loop=True, max_fps=0))
    src.start()
    try:
        got, end = [], time.monotonic() + 6
        while len(got) < N_FRAMES * 2 + 3 and time.monotonic() < end:
            item = src.read(timeout=0.5)
            if item is not None:
                got.append(item)
        assert len(got) >= N_FRAMES + 3 and not src.finished  # went past the end and kept going
        ts = [t for _, t in got]
        assert all(b > a for a, b in zip(ts, ts[1:], strict=False))  # timestamps keep increasing
    finally:
        src.stop()
