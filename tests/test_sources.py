"""Stream address checks for the dashboard "Add stream" button."""

import pytest

from anpr.sources import SourceError, mask_source, normalize_source, source_kind


@pytest.fixture
def video(tmp_path):
    p = tmp_path / "Gate Cam" / "gate.mp4"
    p.parent.mkdir()
    p.write_bytes(b"\x00")
    return p


def test_empty_means_config_camera():
    assert normalize_source("   ") == ""
    assert source_kind("") == "default"


def test_video_file_is_resolved(video):
    assert normalize_source(f"  {video}  ") == str(video.resolve())
    assert source_kind(str(video)) == "file"


@pytest.mark.parametrize("wrap", ['"{}"', "'{}'"])
def test_pasted_quoted_path_is_accepted(video, wrap):
    assert normalize_source(wrap.format(video)) == str(video.resolve())


def test_terminal_escaped_spaces_are_accepted(video):
    assert normalize_source(str(video).replace(" ", "\\ ")) == str(video.resolve())


@pytest.mark.parametrize(
    ("value", "msg"),
    [
        ("videos/gate.mp4", "full path"),
        ("/definitely/not/here.mp4", "not found"),
        ("/etc/passwd", "not a video"),
        ("http://cam.local/stream.mjpg", "rtsp"),
        ("file:///tmp/x.mp4", "rtsp"),
        ("rtsp://", "no camera address"),
        ("rtsp://1.2.3.4:99999/x", "not a valid"),
        ("rtsp://1.2.3.4/a b", "spaces"),
        ("rtsp://1.2.3.4/\nx", "control"),
        ("123", "webcam index"),
    ],
)
def test_rejected(value, msg):
    with pytest.raises(SourceError, match=msg):
        normalize_source(value)


def test_directory_named_like_a_video_is_rejected(tmp_path):
    (tmp_path / "x.mp4").mkdir()
    with pytest.raises(SourceError, match="not found"):
        normalize_source(str(tmp_path / "x.mp4"))


def test_symlink_to_non_video_is_rejected(tmp_path):
    target = tmp_path / "secret.txt"
    target.write_text("x")
    link = tmp_path / "cam.mp4"
    link.symlink_to(target)
    with pytest.raises(SourceError, match="not a video"):
        normalize_source(str(link))


@pytest.mark.parametrize(
    "link",
    [
        "rtsp://192.168.1.64:554/Streaming/Channels/101",
        "rtsps://admin:p@ss@cam.local/live",
        "RTSP://10.0.0.2/x",
    ],
)
def test_rtsp_links_accepted(link):
    assert normalize_source(link) == link
    assert source_kind(link) == "rtsp"


def test_webcam_and_picamera():
    assert normalize_source("0") == "0" and source_kind("0") == "webcam"
    assert normalize_source("PiCamera") == "picamera" and source_kind("picamera") == "picamera"


@pytest.mark.parametrize(
    ("src", "shown"),
    [
        ("rtsp://admin:hunter2@192.168.1.64:554/ch1", "rtsp://admin:***@192.168.1.64:554/ch1"),
        ("rtsp://:hunter2@cam/x", "rtsp://:***@cam/x"),
        ("rtsp://admin@cam/x", "rtsp://admin@cam/x"),
        ("rtsp://cam/x?token=1", "rtsp://cam/x?token=1"),
        ("/Users/me/gate.mp4", "/Users/me/gate.mp4"),
    ],
)
def test_mask_hides_password(src, shown):
    assert mask_source(src) == shown
    assert "hunter2" not in mask_source(src)
