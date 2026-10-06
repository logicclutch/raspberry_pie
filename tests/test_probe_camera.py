"""scripts/probe_camera.py: main-stream URL guessing, packet statistics, hints and report text."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from anpr.camera import StreamInfo
from scripts.probe_camera import (
    PlateSample,
    StreamReport,
    candidates,
    format_report,
    guess_main_urls,
    main,
    make_hints,
    parse_packets,
)

SUB = "rtsp://admin:pw1234@192.168.5.53:554/avstream/channel=1/stream=1.sdp"
MAIN = "rtsp://admin:pw1234@192.168.5.53:554/avstream/channel=1/stream=0.sdp"


def test_guess_main_avstream_and_hikvision() -> None:
    assert guess_main_urls(SUB) == [MAIN]
    assert guess_main_urls("rtsp://u:p@10.0.0.2:554/avstream/channel=3/stream=1.sdp") == [
        "rtsp://u:p@10.0.0.2:554/avstream/channel=3/stream=0.sdp"
    ]
    assert guess_main_urls("rtsp://u:p@10.0.0.2/Streaming/Channels/102") == [
        "rtsp://u:p@10.0.0.2/Streaming/Channels/101"
    ]
    assert guess_main_urls("rtsp://u:p@10.0.0.2/Streaming/channels/202?transportmode=unicast") == [
        "rtsp://u:p@10.0.0.2/Streaming/channels/201?transportmode=unicast"
    ]


@pytest.mark.parametrize(
    "url",
    [
        MAIN,  # already the main stream
        "rtsp://u:p@10.0.0.2/Streaming/Channels/101",
        "rtsp://u:p@10.0.0.2/Streaming/Channels/1020",  # not the /102 pattern
        "rtsp://u:p@10.0.0.2/cam/realmonitor?channel=1&subtype=1",  # other brands: not guessed
        "/videos/gate.mp4",
    ],
)
def test_guess_main_leaves_other_addresses_alone(url: str) -> None:
    assert guess_main_urls(url) == []


def test_candidates_order_and_dedupe() -> None:
    assert candidates([SUB], guess_main=False) == [(SUB, None)]
    assert candidates([SUB, MAIN], guess_main=True) == [(SUB, None), (MAIN, SUB)]
    assert candidates([MAIN, SUB], guess_main=True) == [(MAIN, None), (SUB, None)]


def test_parse_packets() -> None:
    # 5 fps, keyframe every 10 frames, 1000 bytes per packet; one row without pts (dts is used)
    rows = []
    for i in range(30):
        pts = "N/A" if i == 3 else f"{i * 0.2:.6f}"
        rows.append(f"{pts},{i * 0.2:.6f},1000,{'K__' if i % 10 == 0 else '___'}")
    st = parse_packets("\n".join(rows) + "\n", None)
    assert st["packets"] == 30
    assert st["measured_fps"] == pytest.approx(5.0)
    assert st["keyframe_every_frames"] == pytest.approx(10.0)
    assert st["keyframe_every_s"] == pytest.approx(2.0)
    assert st["bitrate_kbps"] == pytest.approx(30 * 1000 * 8 / 6.0 / 1000)
    empty = parse_packets("", 25.0)
    assert empty["packets"] == 0 and empty["measured_fps"] is None and empty["bitrate_kbps"] is None


def _report(**kw) -> StreamReport:
    base = dict(
        address="rtsp://admin:***@192.168.5.53:554/avstream/channel=1/stream=1.sdp",
        info=StreamInfo("hevc", 704, 576, 5.0, 523_000),
        packets=50,
        measured_fps=5.0,
        bitrate_kbps=520.0,
        keyframe_every_frames=10.0,
        keyframe_every_s=2.0,
        frames_sampled=20,
        plates=[
            PlateSample(92, 100, 11000.0, 0.9, "1-line", 36.0),
            PlateSample(58, 40, 30.0, 0.8, "2-line", -30.0),
        ],
        min_sharpness=60.0,
    )
    base.update(kw)
    return StreamReport(**base)


def test_hints_for_todays_gate_camera() -> None:
    text = " | ".join(make_hints(_report()))
    assert "H.264" in text and "SUB-stream" in text and "10-15 fps" in text
    assert "1-line plates are 92 px" in text and "2-line plates are 58 px" in text
    assert "slanted 33 deg" in text
    assert "keyframe" not in text  # 2 s is acceptable


def test_hints_for_a_good_main_stream() -> None:
    good = _report(
        info=StreamInfo("h264", 1920, 1080, 12.0, None),
        measured_fps=12.0,
        bitrate_kbps=3000.0,
        keyframe_every_s=1.0,
        plates=[
            PlateSample(150, 40, 900.0, 0.9, "1-line", 5.0),
            PlateSample(95, 60, 700.0, 0.9, "2-line", 8.0),
        ],
    )
    assert make_hints(good) == []
    assert "looks good" in format_report(good)
    blurry = _report(
        info=StreamInfo("h264", 1280, 720, 12.0),
        measured_fps=12.0,
        bitrate_kbps=900.0,
        keyframe_every_s=4.0,
        plates=[PlateSample(130, 30, 20.0, 0.9, "1-line", 0.0)],
    )
    text = " | ".join(make_hints(blurry))
    assert "shutter" in text and "CBR 2-4 Mbit/s" in text and "I-frame interval" in text
    assert "no plate seen" in make_hints(_report(plates=[]))[-1]


def test_format_report() -> None:
    text = format_report(_report(guessed_from="rtsp://admin:***@x/stream=1.sdp"))
    lines = text.splitlines()
    assert lines[0] == "== rtsp://admin:***@192.168.5.53:554/avstream/channel=1/stream=1.sdp"
    assert "guessed main stream of rtsp://admin:***@x/stream=1.sdp" in lines[1]
    assert "   codec            hevc" in lines
    assert "   resolution       704x576" in lines
    assert any(ln.startswith("   fps              declared 5.0, measured 5.0 (50 frames)") for ln in lines)
    assert "   bitrate          520 kbit/s (declared 523)" in lines
    assert "   keyframe every   10 frames, 2.0 s" in lines
    assert any("1-line width   min 92 / median 92 / max 92 px (target >= 120, 1 boxes)" in ln for ln in lines)
    assert any(ln.startswith("   -> ") for ln in lines)
    err = format_report(StreamReport(address="rtsp://u:***@h/x", error="Connection refused"))
    assert err.splitlines() == ["== rtsp://u:***@h/x", "   ERROR: Connection refused"]
    nodet = format_report(_report(detector_used=False, plates=[]))
    assert "not checked (--no-detect), 20 frames sampled" in nodet


needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg not installed"
)


@needs_ffmpeg
def test_main_on_a_file_and_a_dead_camera(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    clip = tmp_path / "t.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
         "testsrc2=size=320x240:rate=10", "-t", "3", "-c:v", "libx264", "-g", "10", "-pix_fmt", "yuv420p",
         str(clip)],
        check=True,
        timeout=60,
    )  # fmt: skip
    dead = "rtsp://admin:hunter22@127.0.0.1:9/avstream/channel=1/stream=1.sdp"
    code = main(
        [
            str(clip),
            dead,
            "--guess-main",
            "--no-detect",
            "--seconds",
            "2",
            "--save-dir",
            str(tmp_path / "out"),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0  # at least one address worked
    assert "hunter22" not in out
    assert "trying rtsp://admin:***@127.0.0.1:9/avstream/channel=1/stream=0.sdp" in out
    assert "   codec            h264" in out and "   resolution       320x240" in out
    assert "keyframe every   10 frames, 1.0 s" in out
    assert "ERROR" in out
    assert list((tmp_path / "out").glob("s0_frame*.jpg"))


def test_check_plates_streams_and_keeps_only_the_best_frames(tmp_path: Path) -> None:
    """Frames come from a generator (never a list: a minute of 1080p would not fit in the Pi's RAM);
    only the MAX_SAVED_FRAMES frames with the widest plates are kept and saved."""
    import numpy as np

    from anpr.types import Box
    from scripts.probe_camera import MAX_SAVED_FRAMES, check_plates

    class WidthDetector:
        def detect(self, frame: np.ndarray) -> list[Box]:
            w = int(frame[0, 0, 0])  # plate width encoded in the frame
            return [Box(10, 10, 10 + w, 30, 0.9)] if w else []

    def frames():
        for w in (0, 50, 90, 70, 0, 120, 60, 110):
            f = np.full((120, 200, 3), w, np.uint8)
            yield f

    r = StreamReport(address="x")
    check_plates(r, frames(), WidthDetector(), tmp_path, "s0")
    assert r.frames_sampled == 8 and len(r.plates) == 6
    saved = sorted(p.name for p in tmp_path.glob("s0_frame*.jpg"))
    assert len(saved) == MAX_SAVED_FRAMES
    assert saved == ["s0_frame002.jpg", "s0_frame005.jpg", "s0_frame007.jpg"]  # widths 90, 120, 110


@needs_ffmpeg
def test_iter_frames_and_output_width(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from scripts.probe_camera import iter_frames

    clip = tmp_path / "w.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
         "testsrc2=size=640x360:rate=10", "-t", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)],
        check=True,
        timeout=60,
    )  # fmt: skip
    got = list(iter_frames(str(clip), (320, 180), 2.0, 0.0, 5.0))
    assert 9 <= len(got) <= 11 and all(f.shape == (180, 320, 3) for f in got)
    assert main([str(clip), "--seconds", "1", "--output-width", "320"]) == 0
    assert "sampled frames at 320x180" in capsys.readouterr().out
