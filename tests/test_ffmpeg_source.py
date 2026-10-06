"""camera.backend "ffmpeg": FfmpegSource with fake ffmpeg/ffprobe scripts (no real camera, network or
Pi needed), its helpers, make_source selection, max_fps/output_width for OpenCVSource, config
validation, and real-ffmpeg integration tests that skip when ffmpeg is not installed."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest

import anpr.camera as camera
from anpr.camera import (
    FfmpegSource,
    OpenCVSource,
    RateGate,
    RpiCamSource,
    StreamInfo,
    decoder_args,
    ffmpeg_select_expr,
    make_source,
    mask_text,
    parse_probe,
    parse_rate,
    scaled_size,
)
from anpr.config import AppConfig, CameraConfig

W, H = 32, 24
RTSP = "rtsp://admin:hunter22@127.0.0.1:9/avstream/channel=1/stream=0.sdp"


# -- fakes ---------------------------------------------------------------------------------------------


def _script(path: Path, body: str) -> str:
    path.write_text(f"#!{sys.executable}\n" + textwrap.dedent(body))
    path.chmod(0o755)
    return str(path)


def make_fakes(
    tmp_path: Path,
    *,
    frames: int = 10,
    interval: float = 0.0,
    exit_code: int = 0,
    stderr: str = "",
    fail_hw: bool = False,
    width: int = W,
    height: int = H,
    fps: float = 25.0,
    codec: str = "h264",
    probe_fails: bool = False,
    hang_hw: bool = False,
    hang_at_end: bool = False,
    tail_bytes: int = 0,
    rotation: int = 0,
) -> tuple[str, str, Path]:
    """Fake ffmpeg: reads W:H from the scale filter, writes `frames` BGR frames filled with 1, 2, 3, ...
    (`interval` s apart), then `tail_bytes` of a torn last frame, prints `stderr`, exits with
    `exit_code` (or stays alive without output with `hang_at_end`). With `fail_hw` it prints an error
    and exits 1, with `hang_hw` it hangs silently, if hardware decoding was asked for. Every argv is
    appended to argv.jsonl."""
    argv_log = tmp_path / "argv.jsonl"
    ffmpeg = _script(
        tmp_path / "fake_ffmpeg",
        f"""
        import json, re, sys, time
        args = sys.argv[1:]
        with open({str(argv_log)!r}, "a") as f:
            f.write(json.dumps(args) + "\\n")
        if {fail_hw!r} and ("-hwaccel" in args or "-c:v" in args):
            sys.stderr.write("Could not find a valid device\\n")
            sys.exit(1)
        if {hang_hw!r} and ("-hwaccel" in args or "-c:v" in args):
            time.sleep(3600)
        w, h = map(int, re.search(r"scale=(\\d+):(\\d+)", args[args.index("-vf") + 1]).groups())
        out = sys.stdout.buffer
        for i in range({frames}):
            try:
                out.write(bytes([(i % 250) + 1]) * (w * h * 3))
                out.flush()
            except BrokenPipeError:
                sys.exit(0)
            time.sleep({interval})
        if {tail_bytes}:
            out.write(bytes([255]) * {tail_bytes})
            out.flush()
        if {hang_at_end!r}:
            time.sleep(3600)
        if {stderr!r}:
            sys.stderr.write({stderr!r} + "\\n")
        sys.exit({exit_code})
        """,
    )
    probe = {
        "streams": [
            {"codec_type": "audio", "codec_name": "pcm_alaw"},
            {
                "codec_type": "video",
                "codec_name": codec,
                "width": width,
                "height": height,
                "avg_frame_rate": f"{int(fps * 1000)}/1000",
                "r_frame_rate": f"{int(fps * 1000)}/1000",
                "side_data_list": [{"side_data_type": "Display Matrix", "rotation": rotation}],
            },
        ],
        "format": {"bit_rate": "2000000"},
    }
    ffprobe = _script(
        tmp_path / "fake_ffprobe",
        f"""
        import sys
        if {probe_fails!r}:
            sys.stderr.write(sys.argv[-1] + ": Connection refused\\n")
            sys.exit(1)
        print({json.dumps(probe)!r})
        """,
    )
    return ffmpeg, ffprobe, argv_log


def calls(argv_log: Path) -> list[list[str]]:
    if not argv_log.exists():
        return []
    return [json.loads(line) for line in argv_log.read_text().splitlines()]


def fast(src: FfmpegSource | OpenCVSource) -> FfmpegSource | OpenCVSource:
    src.backoff_min = 0.05
    src.backoff_max = 0.1
    return src


def drain(src, deadline_s: float = 5.0) -> list[tuple[np.ndarray, float]]:
    out = []
    end = time.monotonic() + deadline_s
    while not src.finished and time.monotonic() < end:
        item = src.read(timeout=0.3)
        if item is not None:
            out.append(item)
    return out


@pytest.fixture()
def clip(tmp_path: Path) -> Path:
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"not decoded by the fakes")
    return p


# -- helpers -------------------------------------------------------------------------------------------


def _gate_count(src_fps: float, max_fps: float, seconds: float = 10.0, jitter: float = 0.0) -> int:
    gate = RateGate(max_fps)
    rng = np.random.default_rng(0)
    n = int(src_fps * seconds)
    return sum(gate.ready(i / src_fps + (rng.uniform(-jitter, jitter) if jitter else 0.0)) for i in range(n))


def test_rate_gate_caps_but_never_pads() -> None:
    assert 99 <= _gate_count(25, 10) <= 101
    assert 99 <= _gate_count(15, 10) <= 101
    assert _gate_count(5, 10) == 50  # slower camera: every frame, nothing repeated
    assert _gate_count(10, 10, jitter=0.012) == 100  # arrival jitter doesn't drop frames at the cap
    assert _gate_count(25, 0) == 250  # 0 = no limit


def test_rate_gate_recovers_after_a_stall_without_a_burst() -> None:
    gate = RateGate(10)
    assert gate.ready(0.0)
    assert gate.ready(5.0)  # long gap
    assert not gate.ready(5.04)
    assert gate.ready(5.1)


def test_select_expr_shape() -> None:
    e = ffmpeg_select_expr(10)
    assert "st(0," in e and "ld(0)" in e and "isnan(t)" in e and "0.1" in e and "0.025" in e
    assert "'" not in e  # quoted by the caller


@pytest.mark.parametrize(
    ("w", "h", "out", "expected"),
    [
        (1920, 1080, 1280, (1280, 720)),
        (1920, 1080, None, (1920, 1080)),
        (704, 576, 1280, (704, 576)),  # never upscaled
        (704, 576, 640, (640, 524)),
        (1921, 1081, 1001, (1000, 562)),  # even numbers
    ],
)
def test_scaled_size(w: int, h: int, out: int | None, expected: tuple[int, int]) -> None:
    assert scaled_size(w, h, out) == expected


def test_parse_probe_and_rate() -> None:
    info = parse_probe(
        {
            "streams": [
                {"codec_type": "audio", "codec_name": "pcm_alaw"},
                {
                    "codec_type": "video",
                    "codec_name": "hevc",
                    "width": 704,
                    "height": 576,
                    "avg_frame_rate": "0/0",
                    "r_frame_rate": "4993/1000",
                    "bit_rate": "523333",
                },
            ],
            "format": {"duration": "684.1"},
        }
    )
    assert info == StreamInfo("hevc", 704, 576, pytest.approx(4.993), 523333, pytest.approx(684.1))
    assert parse_rate("25/1") == 25.0 and parse_rate("0/0") is None and parse_rate("90000/1") is None
    assert parse_rate(None) is None and parse_rate("x") is None
    with pytest.raises(ValueError):
        parse_probe('{"streams": [{"codec_type": "audio"}]}')


def test_decoder_args(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(camera, "ffmpeg_decoders", lambda _b="ffmpeg": frozenset({"h264", "h264_v4l2m2m"}))
    pi = {"system": "Linux", "machine": "aarch64"}
    assert decoder_args("auto", "h264", **pi) == (["-c:v", "h264_v4l2m2m"], "v4l2m2m")
    assert decoder_args("auto", "hevc", **pi) == ([], "software")  # Pi 3B+: no H.265 hardware
    assert decoder_args("off", "h264", **pi) == ([], "software")
    assert decoder_args("auto", "h264", system="Linux", machine="x86_64") == ([], "software")
    assert decoder_args("auto", "h264", system="Darwin", machine="x86_64")[1] == "videotoolbox"
    assert decoder_args("videotoolbox", "h264", **pi) == (["-hwaccel", "videotoolbox"], "videotoolbox")
    monkeypatch.setattr(camera, "ffmpeg_decoders", lambda _b="ffmpeg": frozenset({"h264"}))
    assert decoder_args("auto", "h264", **pi) == ([], "software")
    assert decoder_args("v4l2m2m", "h264", **pi) == ([], "software")


def test_mask_text() -> None:
    line = f"[rtsp @ 0x1] {RTSP}: 401 Unauthorized (password hunter22)"
    masked = mask_text(line, RTSP)
    assert "hunter22" not in masked and "admin:***@127.0.0.1:9" in masked
    assert mask_text("rtsp://u:other@10.0.0.1/x failed", RTSP) == "rtsp://u:***@10.0.0.1/x failed"
    assert mask_text("plain line", "/videos/a.mp4") == "plain line"


# -- command -------------------------------------------------------------------------------------------


def _index(cmd: list[str], item: str) -> int:
    return cmd.index(item)


def test_command_live_rtsp() -> None:
    src = FfmpegSource(CameraConfig(source=RTSP, backend="ffmpeg", max_fps=10, decode_threads=2))
    cmd = src.build_command(1280, 720, ["-c:v", "h264_v4l2m2m"])
    i = _index(cmd, "-i")
    assert cmd[i + 1] == RTSP
    assert cmd[:6] == ["ffmpeg", "-hide_banner", "-nostats", "-loglevel", "warning", "-nostdin"]
    assert _index(cmd, "-rtsp_transport") < i and cmd[_index(cmd, "-rtsp_transport") + 1] == "tcp"
    assert cmd[_index(cmd, "-timeout") + 1] == "5000000" and _index(cmd, "-timeout") < i
    assert _index(cmd, "-c:v") < i and "-threads" not in cmd  # hardware: no decoder threads
    assert "-re" not in cmd
    vf = cmd[_index(cmd, "-vf") + 1]
    assert vf.startswith("select='") and vf.endswith(",scale=1280:720:flags=area")
    assert cmd[_index(cmd, "-fps_mode") + 1] == "passthrough"
    assert cmd[-5:] == ["-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]
    assert "-an" in cmd


def test_command_files(clip: Path) -> None:
    rt = FfmpegSource(CameraConfig(source=str(clip), backend="ffmpeg", realtime=True, decode_threads=2))
    cmd = rt.build_command(704, 576, [])
    assert "-re" in cmd and _index(cmd, "-re") < _index(cmd, "-i")
    assert cmd[_index(cmd, "-i") + 1] == "file:" + str(clip)
    assert cmd[_index(cmd, "-threads") + 1] == "2" and "-rtsp_transport" not in cmd
    assert "select=" in cmd[_index(cmd, "-vf") + 1]
    every = FfmpegSource(CameraConfig(source=str(clip), backend="ffmpeg", realtime=False))
    cmd = every.build_command(704, 576, [])
    assert "-re" not in cmd and cmd[_index(cmd, "-vf") + 1] == "scale=704:576:flags=area"
    no_cap = FfmpegSource(CameraConfig(source=RTSP, backend="ffmpeg", max_fps=0))
    assert "select" not in no_cap.build_command(64, 48, [])[-8]


def test_ffmpeg_source_rejects_webcam_and_picamera() -> None:
    for s in ("0", "picamera", ""):
        with pytest.raises(ValueError):
            FfmpegSource(CameraConfig(source=s))


# -- FfmpegSource with fakes ---------------------------------------------------------------------------


def test_live_frames_size_latest_only_and_timestamps(tmp_path: Path) -> None:
    ffmpeg, ffprobe, argv = make_fakes(tmp_path, frames=100000, interval=0.01, width=128, height=96)
    cfg = CameraConfig(source=RTSP, backend="ffmpeg", output_width=64, hw_decode="off")
    src = FfmpegSource(cfg, ffmpeg=ffmpeg, ffprobe=ffprobe)
    src.start()
    try:
        first = src.read(timeout=5.0)
        assert first is not None
        frame, ts = first
        assert frame.shape == (48, 64, 3) and frame.dtype == np.uint8  # 128x96 shrunk to 64x48
        assert abs(ts - time.time()) < 2.0
        assert src.camera_ok and src.frame_size == (64, 48) and src.decoder == "software"
        time.sleep(0.3)
        later = src.read(timeout=1.0)
        assert later is not None and int(later[0][0, 0, 0]) > int(frame[0, 0, 0]) + 5  # skipped ahead
        assert int(later[0].min()) == int(later[0].max())  # a whole frame, not a torn one
    finally:
        src.stop()
    assert "-rtsp_transport" in calls(argv)[0]


def test_live_restarts_with_backoff(tmp_path: Path) -> None:
    ffmpeg, ffprobe, _ = make_fakes(tmp_path, frames=2)
    src = fast(FfmpegSource(CameraConfig(source=RTSP, backend="ffmpeg"), ffmpeg=ffmpeg, ffprobe=ffprobe))
    src.start()
    try:
        end = time.monotonic() + 8.0
        while src.restarts < 3 and time.monotonic() < end:
            src.read(timeout=0.2)
        assert src.restarts >= 3
        assert src.read(timeout=3.0) is not None
        assert not src.finished
    finally:
        src.stop()


def test_stop_kills_ffmpeg(tmp_path: Path) -> None:
    ffmpeg, ffprobe, _ = make_fakes(tmp_path, frames=10**6, interval=0.01)
    src = FfmpegSource(CameraConfig(source=RTSP, backend="ffmpeg"), ffmpeg=ffmpeg, ffprobe=ffprobe)
    src.start()
    assert src.read(timeout=5.0) is not None
    proc = src._proc
    assert proc is not None and proc.poll() is None
    t0 = time.monotonic()
    src.stop()
    assert time.monotonic() - t0 < 4.0
    assert proc.poll() is not None  # reaped: no zombie
    assert not src.camera_ok
    assert src.read(timeout=0.1) is None


def test_stderr_password_is_masked_in_logs(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    ffmpeg, ffprobe, _ = make_fakes(tmp_path, frames=1, exit_code=1, stderr=f"{RTSP}: 401 Unauthorized")
    src = fast(FfmpegSource(CameraConfig(source=RTSP, backend="ffmpeg"), ffmpeg=ffmpeg, ffprobe=ffprobe))
    with caplog.at_level(logging.INFO, logger="anpr.camera"):
        src.start()
        end = time.monotonic() + 5.0
        while "401" not in caplog.text and time.monotonic() < end:
            src.read(timeout=0.1)
        src.stop()
    assert "401 Unauthorized" in caplog.text and "admin:***@127.0.0.1" in caplog.text
    assert "hunter22" not in caplog.text


def test_probe_failure_is_masked_and_retried(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    ffmpeg, ffprobe, argv = make_fakes(tmp_path, probe_fails=True)
    src = fast(FfmpegSource(CameraConfig(source=RTSP, backend="ffmpeg"), ffmpeg=ffmpeg, ffprobe=ffprobe))
    with caplog.at_level(logging.INFO, logger="anpr.camera"):
        src.start()
        end = time.monotonic() + 5.0
        while "Connection refused" not in caplog.text and time.monotonic() < end:
            assert src.read(timeout=0.1) is None
        assert not src.camera_ok and not src.finished
        src.stop()
    assert "Connection refused" in caplog.text and "hunter22" not in caplog.text
    assert calls(argv) == []  # ffmpeg is never started without a probe


def test_hardware_failure_falls_back_to_software(tmp_path: Path) -> None:
    ffmpeg, ffprobe, argv = make_fakes(tmp_path, frames=10**6, interval=0.01, fail_hw=True)
    cfg = CameraConfig(source=RTSP, backend="ffmpeg", hw_decode="videotoolbox")
    src = fast(FfmpegSource(cfg, ffmpeg=ffmpeg, ffprobe=ffprobe))
    src.start()
    try:
        assert src.read(timeout=5.0) is not None
        assert src.decoder == "software"
    finally:
        src.stop()
    runs = calls(argv)
    assert "-hwaccel" in runs[0] and "-hwaccel" not in runs[1]


def test_file_every_frame_in_order(tmp_path: Path, clip: Path) -> None:
    ffmpeg, ffprobe, argv = make_fakes(tmp_path, frames=20, fps=20.0)
    cfg = CameraConfig(source=str(clip), backend="ffmpeg", realtime=False, hw_decode="off")
    src = FfmpegSource(cfg, ffmpeg=ffmpeg, ffprobe=ffprobe)
    src.start()
    try:
        items = drain(src)
        assert src.finished
        assert [int(f[0, 0, 0]) for f, _ in items] == list(range(1, 21))
        assert np.allclose(np.diff([t for _, t in items]), 1 / 20.0, atol=1e-6)
        assert src.file_fps == 20.0
    finally:
        src.stop()
    cmd = calls(argv)[0]
    assert "-re" not in cmd and len(calls(argv)) == 1  # a file is never restarted


def test_file_realtime_drops_for_slow_consumer(tmp_path: Path, clip: Path) -> None:
    ffmpeg, ffprobe, argv = make_fakes(tmp_path, frames=20, interval=0.05, fps=20.0)
    cfg = CameraConfig(source=str(clip), backend="ffmpeg", realtime=True, max_fps=10, hw_decode="off")
    src = FfmpegSource(cfg, ffmpeg=ffmpeg, ffprobe=ffprobe)
    src.start()
    try:
        got = []
        end = time.monotonic() + 5.0
        while not src.finished and time.monotonic() < end:
            item = src.read(timeout=0.3)
            if item is not None:
                got.append((int(item[0][0, 0, 0]), item[1]))
                time.sleep(0.15)
        assert src.finished
        values = [v for v, _ in got]
        assert 2 <= len(values) < 20 and values == sorted(set(values))
        # realtime + max_fps 10: frame k of the capped output is stamped start + k / 10
        stamps = [t for _, t in got]
        steps = np.diff(stamps) / np.maximum(1, np.diff(values))
        assert np.allclose(steps, 0.1, atol=1e-6)
    finally:
        src.stop()
    cmd = calls(argv)[0]
    assert "-re" in cmd and "select=" in cmd[cmd.index("-vf") + 1]


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        FfmpegSource(CameraConfig(source=str(tmp_path / "nope.mp4"), backend="ffmpeg")).start()


def test_file_unreadable_finishes(tmp_path: Path, clip: Path) -> None:
    ffmpeg, ffprobe, _ = make_fakes(tmp_path, probe_fails=True)
    src = FfmpegSource(CameraConfig(source=str(clip), backend="ffmpeg"), ffmpeg=ffmpeg, ffprobe=ffprobe)
    src.start()
    try:
        end = time.monotonic() + 5.0
        while not src.finished and time.monotonic() < end:
            src.read(timeout=0.1)
        assert src.finished
    finally:
        src.stop()


# -- make_source and config ----------------------------------------------------------------------------


def test_make_source_picks_backend(monkeypatch: pytest.MonkeyPatch, clip: Path) -> None:
    monkeypatch.setattr(camera.shutil, "which", lambda b: f"/usr/bin/{b}")
    assert isinstance(make_source(CameraConfig(source=RTSP, backend="ffmpeg")), FfmpegSource)
    assert isinstance(make_source(CameraConfig(source=str(clip), backend="ffmpeg")), FfmpegSource)
    assert isinstance(make_source(CameraConfig(source="0", backend="ffmpeg")), OpenCVSource)
    assert isinstance(make_source(CameraConfig(source="picamera", backend="ffmpeg")), RpiCamSource)
    assert isinstance(make_source(CameraConfig(source=RTSP)), OpenCVSource)  # default backend
    # the dashboard's stream switch opens sources the same way (engine.main: open_source)
    base = CameraConfig(source="0", backend="ffmpeg")
    assert isinstance(make_source(base.model_copy(update={"source": RTSP})), FfmpegSource)
    monkeypatch.setattr(camera.shutil, "which", lambda b: None)
    assert isinstance(make_source(CameraConfig(source=RTSP, backend="ffmpeg")), OpenCVSource)


def test_camera_config_new_keys() -> None:
    c = CameraConfig()
    assert (c.backend, c.hw_decode, c.max_fps, c.output_width) == ("opencv", "auto", 10.0, None)
    ok = CameraConfig(backend="ffmpeg", hw_decode="v4l2m2m", max_fps=12.5, output_width=1280)
    assert ok.output_width == 1280
    for bad in (
        {"backend": "gstreamer"},
        {"hw_decode": "cuda"},
        {"max_fps": -1},
        {"output_width": 10},
    ):
        with pytest.raises(ValueError):
            CameraConfig(**bad)


def test_repo_config_yaml_keeps_opencv_default() -> None:
    from anpr.config import load_config

    cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    assert isinstance(cfg, AppConfig)
    assert cfg.camera.backend == "opencv"


# -- OpenCVSource: max_fps and output_width --------------------------------------------------------------


class _FakeCap:
    """Stands in for cv2.VideoCapture on a 50 fps live stream."""

    def __init__(self) -> None:
        self.grabs = 0
        self.retrieves = 0

    def isOpened(self) -> bool:  # noqa: N802 - cv2 API
        return True

    def grab(self) -> bool:
        time.sleep(0.02)
        self.grabs += 1
        return True

    def retrieve(self) -> tuple[bool, np.ndarray]:
        self.retrieves += 1
        return True, np.full((96, 128, 3), 7, np.uint8)

    def release(self) -> None:
        pass


def test_opencv_live_honours_max_fps_and_output_width(monkeypatch: pytest.MonkeyPatch) -> None:
    cap = _FakeCap()
    src = OpenCVSource(CameraConfig(source="rtsp://127.0.0.1:9/x", max_fps=10, output_width=64))
    monkeypatch.setattr(src, "_open", lambda: cap)
    src.start()
    try:
        got = []
        end = time.monotonic() + 1.5
        while time.monotonic() < end:
            item = src.read(timeout=0.2)
            if item is not None:
                got.append(item[0])
    finally:
        src.stop()
    assert cap.grabs >= 40  # the camera kept delivering ~50 fps
    assert 10 <= len(got) <= 18  # ~10 fps reached the consumer
    assert cap.retrieves <= len(got) + 2  # skipped frames were never converted
    assert got[0].shape == (48, 64, 3)


# -- real ffmpeg -----------------------------------------------------------------------------------------

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg not installed"
)


def _encode(path: Path, size: str, rate: int, seconds: int) -> Path:
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
            "-i", f"testsrc2=size={size}:rate={rate}", "-t", str(seconds),
            "-c:v", "libx264", "-preset", "ultrafast", "-g", str(rate), "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
        timeout=60,
    )  # fmt: skip
    return path


@pytest.fixture(scope="module")
def h264_25fps(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    return _encode(tmp_path_factory.mktemp("ff") / "t25.mp4", "320x240", 25, 2)


@needs_ffmpeg
def test_real_ffmpeg_every_frame_and_downscale(h264_25fps: Path) -> None:
    cfg = CameraConfig(
        source=str(h264_25fps), backend="ffmpeg", realtime=False, output_width=160, hw_decode="off"
    )
    src = FfmpegSource(cfg)
    src.start()
    try:
        items = drain(src, 20.0)
    finally:
        src.stop()
    assert src.finished and len(items) == 50
    assert items[0][0].shape == (120, 160, 3)
    assert src.info is not None and src.info.codec == "h264" and src.file_fps == pytest.approx(25.0)


@needs_ffmpeg
def test_real_ffmpeg_realtime_caps_fps(h264_25fps: Path) -> None:
    cfg = CameraConfig(source=str(h264_25fps), backend="ffmpeg", realtime=True, max_fps=10, hw_decode="off")
    src = FfmpegSource(cfg)
    src.start()
    try:
        items = drain(src, 20.0)
    finally:
        src.stop()
    assert 17 <= len(items) <= 21  # 2 s at max 10 fps (not 50)


@needs_ffmpeg
@pytest.mark.parametrize(("rate", "lo", "hi"), [(5, 10, 10), (10, 20, 20), (25, 20, 21)])
def test_real_ffmpeg_rate_cap_drops_but_never_repeats(tmp_path: Path, rate: int, lo: int, hi: int) -> None:
    """Why select + passthrough and not fps=10: a 5 fps camera must stay 5 fps (fps=10 would send
    every frame twice and the voter would count each read twice); 25 fps becomes 10 fps."""
    clip = _encode(tmp_path / f"t{rate}.mp4", "160x120", rate, 2)
    src = FfmpegSource(CameraConfig(source=str(clip), backend="ffmpeg", realtime=True, max_fps=10))
    cmd = [a for a in src.build_command(160, 120, []) if a != "-re"]  # same filters, full speed
    out = subprocess.run(cmd, capture_output=True, check=True, timeout=60).stdout
    frames = np.frombuffer(out, np.uint8).reshape(-1, 120, 160, 3)
    assert lo <= len(frames) <= hi  # 2 s of video
    assert all(not np.array_equal(a, b) for a, b in zip(frames, frames[1:], strict=False))


@needs_ffmpeg
def test_real_ffprobe_error_never_shows_the_password(caplog: pytest.LogCaptureFixture) -> None:
    src = fast(FfmpegSource(CameraConfig(source="rtsp://admin:hunter22@127.0.0.1:9/x", backend="ffmpeg")))
    with caplog.at_level(logging.INFO, logger="anpr.camera"):
        src.start()
        assert src.read(timeout=1.5) is None
        src.stop()
    assert "ffprobe failed" in caplog.text or "no answer" in caplog.text
    assert "hunter22" not in caplog.text


# -- review fixes: time jumps, stalls, torn frames, rotation, decoder cache ------------------------------


def test_rate_gate_restarts_after_time_goes_backwards() -> None:
    """An IP camera that resets its stream timestamps must not freeze the capped stream."""
    gate = RateGate(10)
    passed = [t for t in (i / 25 for i in range(100)) if gate.ready(t)]  # 0 .. 3.96 s
    assert 39 <= len(passed) <= 41
    after = [t for t in (i / 25 - 60.0 for i in range(100)) if gate.ready(t)]  # jumped back 60 s
    assert 39 <= len(after) <= 41  # still ~10 fps, not zero until the clock catches up


def test_parse_probe_swaps_size_for_rotated_video() -> None:
    def probe(**extra: object) -> dict:
        st = {"codec_type": "video", "codec_name": "h264", "width": 1920, "height": 1080, **extra}
        return {"streams": [st]}

    assert parse_probe(probe()).width == 1920
    rot = parse_probe(probe(side_data_list=[{"side_data_type": "Display Matrix", "rotation": -90}]))
    assert (rot.width, rot.height) == (1080, 1920)
    assert (parse_probe(probe(tags={"rotate": "270"})).width) == 1080  # older ffprobe
    assert parse_probe(probe(side_data_list=[{"rotation": 180}])).width == 1920


def test_decoder_list_is_cached_but_failures_are_not(monkeypatch: pytest.MonkeyPatch) -> None:
    runs: list[list[str]] = []
    outputs = iter(["", " V....D h264_v4l2m2m          V4L2 mem2mem H.264 decoder wrapper\n"])

    def fake_run(cmd: list[str], **_kw: object) -> subprocess.CompletedProcess:
        runs.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, next(outputs), "")

    monkeypatch.setattr(camera, "_DECODERS", {})
    monkeypatch.setattr(camera.subprocess, "run", fake_run)
    assert camera.ffmpeg_decoders("fake-ffmpeg") == frozenset()  # failed/empty: not cached
    assert camera.ffmpeg_decoders("fake-ffmpeg") == frozenset({"h264_v4l2m2m"})
    assert camera.ffmpeg_decoders("fake-ffmpeg") == frozenset({"h264_v4l2m2m"})
    assert len(runs) == 2  # ran once more after the failure, then cached


def test_last_command_never_holds_the_password(tmp_path: Path) -> None:
    ffmpeg, ffprobe, _ = make_fakes(tmp_path, frames=10**6, interval=0.01)
    src = FfmpegSource(CameraConfig(source=RTSP, backend="ffmpeg"), ffmpeg=ffmpeg, ffprobe=ffprobe)
    src.start()
    try:
        assert src.read(timeout=5.0) is not None
    finally:
        src.stop()
    shown = " ".join(src.last_command)
    assert "hunter22" not in shown and "admin:***@127.0.0.1:9" in shown


def test_live_stall_restarts_ffmpeg(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """ffmpeg alive but silent (hung decoder, bad timestamps): the watchdog restarts it."""
    ffmpeg, ffprobe, argv = make_fakes(tmp_path, frames=2, hang_at_end=True)
    src = fast(FfmpegSource(CameraConfig(source=RTSP, backend="ffmpeg"), ffmpeg=ffmpeg, ffprobe=ffprobe))
    src.stall_timeout_s = 0.3
    src.first_frame_timeout_s = 2.0
    with caplog.at_level(logging.INFO, logger="anpr.camera"):
        src.start()
        try:
            end = time.monotonic() + 8.0
            while src.restarts < 2 and time.monotonic() < end:
                src.read(timeout=0.2)
            assert src.restarts >= 2
            assert src.read(timeout=3.0) is not None  # the new ffmpeg delivers again
        finally:
            src.stop()
    assert "no picture from ffmpeg" in caplog.text
    assert len(calls(argv)) >= 2


def test_hung_hardware_decoder_falls_back_to_software(tmp_path: Path) -> None:
    ffmpeg, ffprobe, argv = make_fakes(tmp_path, frames=10**6, interval=0.01, hang_hw=True)
    cfg = CameraConfig(source=RTSP, backend="ffmpeg", hw_decode="videotoolbox")
    src = fast(FfmpegSource(cfg, ffmpeg=ffmpeg, ffprobe=ffprobe))
    src.first_frame_timeout_s = 1.5  # the fake is a Python script: allow for its start-up
    src.start()
    try:
        assert src.read(timeout=8.0) is not None
        assert src.decoder == "software"
    finally:
        src.stop()
    runs = calls(argv)
    assert "-hwaccel" in runs[0] and "-hwaccel" not in runs[1]


def test_torn_last_frame_is_dropped(tmp_path: Path, clip: Path) -> None:
    ffmpeg, ffprobe, _ = make_fakes(tmp_path, frames=3, tail_bytes=W * H)  # a third of a 4th frame
    cfg = CameraConfig(source=str(clip), backend="ffmpeg", realtime=False, hw_decode="off")
    src = FfmpegSource(cfg, ffmpeg=ffmpeg, ffprobe=ffprobe)
    src.start()
    try:
        items = drain(src)
    finally:
        src.stop()
    assert src.finished and [int(f[0, 0, 0]) for f, _ in items] == [1, 2, 3]
    assert all(f.shape == (H, W, 3) for f, _ in items)


@pytest.mark.parametrize(("w", "h"), [(33, 25), (31, 7)])
def test_odd_frame_sizes_are_packed_bgr(tmp_path: Path, clip: Path, w: int, h: int) -> None:
    """bgr24 rawvideo has no row padding, also for odd widths: frame bytes = w * h * 3 exactly."""
    ffmpeg, ffprobe, _ = make_fakes(tmp_path, frames=4, width=w, height=h)
    cfg = CameraConfig(source=str(clip), backend="ffmpeg", realtime=False, hw_decode="off")
    src = FfmpegSource(cfg, ffmpeg=ffmpeg, ffprobe=ffprobe)
    src.start()
    try:
        items = drain(src)
    finally:
        src.stop()
    assert [int(f[0, 0, 0]) for f, _ in items] == [1, 2, 3, 4]
    assert all(f.shape == (h, w, 3) and int(f.min()) == int(f.max()) for f, _ in items)


def test_rotated_file_keeps_its_upright_size(tmp_path: Path, clip: Path) -> None:
    ffmpeg, ffprobe, argv = make_fakes(tmp_path, frames=2, width=64, height=48, rotation=90)
    cfg = CameraConfig(source=str(clip), backend="ffmpeg", realtime=False, hw_decode="off")
    src = FfmpegSource(cfg, ffmpeg=ffmpeg, ffprobe=ffprobe)
    src.start()
    try:
        items = drain(src)
    finally:
        src.stop()
    assert items and items[0][0].shape == (64, 48, 3)
    assert "scale=48:64" in calls(argv)[0][calls(argv)[0].index("-vf") + 1]


@needs_ffmpeg
def test_real_ffmpeg_cap_survives_timestamps_going_backwards() -> None:
    """8 s at 25 fps whose timestamps jump back 60 s after 4 s (camera clock reset): the cap must
    keep delivering ~10 fps instead of dropping everything until the clock catches up."""
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats", "-loglevel", "error", "-f", "lavfi", "-t", "8",
        "-i", "testsrc2=size=64x48:rate=25",
        "-vf", f"setpts='if(gte(N,100),PTS-60/TB,PTS)',select='{ffmpeg_select_expr(10)}'",
        "-fps_mode", "passthrough", "-pix_fmt", "gray", "-f", "rawvideo", "pipe:1",
    ]  # fmt: skip
    out = subprocess.run(cmd, capture_output=True, check=True, timeout=60).stdout
    assert 76 <= len(out) // (64 * 48) <= 84


@needs_ffmpeg
def test_real_ffmpeg_rotated_file(tmp_path: Path, h264_25fps: Path) -> None:
    rotated = tmp_path / "rot.mp4"
    made = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-display_rotation", "90",
         "-i", str(h264_25fps), "-c", "copy", str(rotated)],
        capture_output=True,
        timeout=60,
    )  # fmt: skip
    if made.returncode != 0:
        pytest.skip("this ffmpeg has no -display_rotation (needs >= 6.0)")
    src = FfmpegSource(CameraConfig(source=str(rotated), backend="ffmpeg", realtime=False, hw_decode="off"))
    src.start()
    try:
        items = drain(src, 20.0)
    finally:
        src.stop()
    assert len(items) == 50 and items[0][0].shape == (320, 240, 3)
