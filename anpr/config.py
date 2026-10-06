"""Typed configuration loaded from config.yaml. Same file format on Mac and Pi."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CameraConfig(_Section):
    # "0" (webcam index) | path to a video file | rtsp://... | "picamera" (Pi camera module via the
    # rpicam-vid CLI; no picamera2 Python lib, so no apt-numpy vs venv-numpy ABI clash)
    source: str = "0"
    width: int = Field(1280, gt=0)
    height: int = Field(720, gt=0)
    fps: int = Field(15, gt=0)
    # Video files only: True = drop frames to behave like a live camera; False = process every frame.
    realtime: bool = True
    # Video files only: start again from the beginning when the file ends (demo / showroom use).
    loop: bool = False
    # Demo: video files played one after another. Non-empty = used instead of `source`; each file
    # plays once, then the next starts by itself; after the last, back to the first if `loop`.
    playlist: tuple[str, ...] = ()
    # Pi camera only: fixed shutter in microseconds (fast shutter vs motion blur). None = auto.
    exposure_us: int | None = 1000
    # RTSP streams and video files: FFmpeg decoder threads. 0 = FFmpeg default (one per CPU core, and
    # every thread keeps its own full-size frame buffers: ~345 MB for a 1080p file on a 16-core Mac).
    # 2 decodes 1080p H.264 at >100 fps on the Mac and keeps the Pi well under its memory limit.
    decode_threads: int = Field(2, ge=0, le=16)
    # How RTSP streams and video files are decoded (webcam indexes and "picamera" ignore it):
    #   opencv = OpenCV's bundled FFmpeg, software decode only (fine for the 704x576 sub-stream).
    #   ffmpeg = the system `ffmpeg` program in a child process (apt ffmpeg on the Pi). It can use the
    #            Pi's hardware H.264 decoder and drops/scales frames before Python sees them: needed
    #            for a 1080p main stream on a Pi 3B+. See docs/CAMERA_SETUP.md.
    backend: Literal["opencv", "ffmpeg"] = "opencv"
    # backend ffmpeg only. auto = h264_v4l2m2m on a 64-bit Linux Pi when `ffmpeg -decoders` lists it and
    # the stream is H.264; videotoolbox on macOS; else software. Falls back to software by itself if
    # the hardware decoder fails. The Pi 3B+ has NO H.265 hardware decoder: set the camera to H.264.
    hw_decode: Literal["auto", "off", "v4l2m2m", "videotoolbox"] = "auto"
    # Live streams and realtime video files: at most this many frames per second reach the pipeline;
    # extra frames are dropped before conversion (inside ffmpeg / before OpenCV's BGR conversion).
    # Slower streams are never padded with repeated frames. 0 = no limit. Not used when realtime is
    # False (every frame of a file is processed).
    max_fps: float = Field(10.0, ge=0.0, le=120.0)
    # Downscale wider pictures to this width (height keeps the aspect ratio) before the pipeline sees
    # them, e.g. 1280 for a 1920x1080 main stream: a BGR frame is 2.7 MB instead of 6.2 MB. The detector
    # works on 320x320 anyway; plate crops are cut from this frame. None = keep the camera's size.
    output_width: int | None = Field(None, ge=64, le=7680)
    # Multi-camera installs: a short label for this camera, put in each plate's JSON as "camera"
    # (e.g. "entry", "exit"). "" = the single default camera.
    name: str = ""


class CameraFeed(_Section):
    # One camera in a multi-camera install (see AppConfig.cameras). Only `source` and `name` differ per
    # camera; all other settings (fps, backend, resolution, threads...) come from the shared `camera:`.
    source: str
    name: str = ""


class MotionConfig(_Section):
    enabled: bool = True
    downscale_width: int = Field(160, gt=0)
    pixel_threshold: int = Field(25, ge=0, le=255)
    min_changed_ratio: float = Field(0.002, ge=0.0, le=1.0)


class DetectorConfig(_Section):
    # ncnn = target runtime (YOLO11n exported with Ultralytics); onnx = generic YOLO/ONNX baseline.
    backend: Literal["ncnn", "onnx"] = "ncnn"
    model_path: Path = Path("models/detector/plate_det_ncnn_model")  # ncnn: dir; onnx: .onnx file
    input_size: int = Field(320, gt=0)
    conf_threshold: float = Field(0.5, ge=0.0, le=1.0)
    iou_threshold: float = Field(0.45, ge=0.0, le=1.0)
    num_threads: int = Field(4, ge=1)  # Pi 3B+ has 4 cores
    class_ids: list[int] | None = None  # None = all classes (plate model has one class)


class OcrConfig(_Section):
    model_path: Path = Path("models/ocr/plate_ocr.onnx")
    config_path: Path = Path("models/ocr/plate_ocr_config.yaml")  # fast-plate-ocr plate config
    num_threads: int = Field(4, ge=1)
    max_plates_per_frame: int = Field(3, ge=1)  # bounds OCR latency per frame on the Pi


class CropConfig(_Section):
    padding: float = Field(0.08, ge=0.0, le=0.5)
    min_width_px: int = Field(80, ge=1)
    min_sharpness: float = Field(60.0, ge=0.0)  # variance of Laplacian on the gray crop
    clahe_clip: float = Field(2.0, gt=0.0)
    # Off by default: on real video the reader (trained on unwarped crops) did better without it.
    deskew: bool = False
    # Side-mounted cameras: the plate is a parallelogram (upright letters, baseline rising ~25-40 deg);
    # the reader then drops the last character with high confidence. De-shear straightens 1-line
    # plates (2-line plates keep the plain crop). Measured on a 704x576 gate camera: 3 -> 12 of 22.
    deshear: bool = False
    # De-shear only looks inside the detector box; when the box stops short of the plate's end the
    # reader drops the last character. If a de-sheared read fails the format check, re-read once
    # from a box widened by this fraction of its width on each side. 0 = off.
    # Measured on the gate camera + phone video: 13 -> 15 of 17 known gate plates, 0 wrong.
    retry_expand: float = Field(0.10, ge=0.0, le=0.3)
    # 2-line plates: when a read is only a bottom line (series + number, e.g. "BC8199"), also read
    # the band just above the box and join the two ("TN36" + "BC8199"). Voting still needs
    # several agreeing frames.
    top_line_rescue: bool = True


class ValidationConfig(_Section):
    min_char_conf: float = Field(0.6, ge=0.0, le=1.0)
    allow_bh: bool = True
    max_repairs: int = Field(2, ge=0)  # max O<->0 style fixes per read; each one lowers confidence
    repair_penalty: float = Field(0.9, gt=0.0, le=1.0)  # confidence multiplier for a repaired char


class TrackingConfig(_Section):
    iou_match: float = Field(0.3, ge=0.0, le=1.0)
    max_age_s: float = Field(3.0, gt=0.0)


class VoteConfig(_Section):
    min_votes: int = Field(3, ge=1)
    min_avg_conf: float = Field(0.85, ge=0.0, le=1.0)
    min_agreement: float = Field(0.7, gt=0.0, le=1.0)  # winner share of all valid reads on the track
    dedupe_window_s: float = Field(60.0, ge=0.0)
    # When a vehicle leaves without a report (at 5 fps a passing car often gives only 2 clean reads),
    # accept it if either:
    #  - its close-up reads have a leader with >= `end_min_votes` reads and >= two-thirds of all
    #    close-up reads, every other close-up read one character away from it, and the leader's mean
    #    confidence >= `end_min_conf` (all identical is the simplest case), or
    #  - every valid read on the track, near or far, is the same text: >= `min_votes` of them, at
    #    least one close-up, mean confidence >= `min_avg_conf`.
    # 0 disables both. Nothing can contradict the result any more.
    end_min_votes: int = Field(2, ge=0)
    end_min_conf: float = Field(0.95, ge=0.0, le=1.0)
    # Decide from CLOSE-UP reads only. A vehicle first seen far away gives small, blurred plates that
    # the reader can get confidently wrong (TN33BY9603 read as TN33BT9603 at 120 px, right at 230 px).
    # close_ratio: a read votes only if its plate is at least this fraction of the largest plate seen
    #   on the track (0 = every read votes).
    # settle_s: while the plate keeps growing (vehicle approaching) nothing is decided; a decision is
    #   made once the plate has not grown by more than settle_growth for settle_s seconds (vehicle
    #   close or stopped), or when the vehicle leaves. 0 = decide as soon as the votes agree.
    close_ratio: float = Field(0.85, ge=0.0, le=1.0)
    settle_s: float = Field(1.0, ge=0.0)
    settle_growth: float = Field(0.05, ge=0.0)


class HsrpConfig(_Section):
    # Mark each saved plate HSRP / non-HSRP / not sure from the hologram + "IND" left of the text
    # (anpr.hsrp). Judged on the frames read before the plate is saved.
    enabled: bool = True
    min_marked: int = Field(2, ge=1)  # HSRP: frames showing the mark (and >= 30% of judged frames)
    min_clean: int = Field(3, ge=1)  # non-HSRP: frames with a clean space (and <= 10% showing a mark)
    # Report "not sure" as non-HSRP: only a plate where the mark was actually seen counts as HSRP.
    # Plates too small, blurred or 2-line to judge are then saved as non-HSRP.
    unsure_as_non_hsrp: bool = True
    # Also judge 2-line plates (hologram in the top-left corner). Off by default: needs a clear,
    # close view (plate >= ~150 px wide, ideally colour) - on blurry grey video the plate frame and
    # bolt heads there look like a hologram. Off: 2-line plates are "not sure".
    two_line: bool = False


class PushConfig(_Section):
    # Send every confirmed plate to the client's server (anpr.push). When this whole section is present
    # in config.yaml it is the source of truth: these values are applied to the device on every restart
    # ("file wins"), so a fleet of Pis can be set from one file. Leave the section out to manage sending
    # only from the dashboard Settings page instead.
    enabled: bool = False
    url: str = ""  # e.g. https://demo.intelliparks.in/api/deviceAPI/ANPRLogInsert ("" = off)
    header_name: str = "Authorization"  # name of the key/token header ("" = send no key header)
    header_value: str = ""  # the key/token value, e.g. "Bearer abc123" ("" = none). Kept on the device.
    device_id: str = ""  # identifies this Pi in the JSON ("" = the Pi's hostname)
    include_images: bool = True  # include the plate snapshot (base64) in each record


class IngestConfig(_Section):
    # "Push" mode: an ANPR camera (e.g. GVD) POSTs its vehicle image(s) (base64 JSON, multipart, or a raw
    # JPEG) to this Pi. The Pi reads and VOTES across all images of one vehicle (full pipeline, see the
    # grouping fields below) and sends the result to the client API (the `push:` section). No
    # video/RTSP/ffmpeg is used, so this is the lightest mode. With a single image per vehicle the vote has
    # one read, so set vote.min_votes / vote.end_min_votes to 1 (the shipped ingest config does).
    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = Field(8080, gt=0, lt=65536)
    path: str = "/api/v1/ingest"  # the URL path the camera POSTs to
    token: str | None = Field(None, max_length=200)  # if set, the camera must send it as X-Ingest-Token
    # JSON field holding the base64 image. "" = find it automatically (any big base64 JPEG/PNG in the body).
    # Dotted paths work, e.g. "data.image". Set it once you see the camera's real field (capture tool).
    image_field: str = ""
    # Optional JSON field with the camera's vehicle/event id. Used to GROUP images of one vehicle (and
    # echoed in the ack) when respond_with_plate is off. It is not sent to the client API, and not echoed
    # in respond_with_plate mode (each POST is already one vehicle).
    vehicle_id_field: str = ""
    camera_field: str = ""  # optional JSON field naming which camera/lane; stored as the plate's "camera"
    max_body_mb: float = Field(12.0, gt=0.0, le=64.0)  # reject bodies bigger than this
    # The camera sends several images per vehicle. We GROUP them and VOTE across the group (full pipeline:
    # 2-line/handwritten rescue, re-read, HSRP) to get one best result per vehicle - no accuracy loss.
    # Grouping key: the vehicle_id_field if the camera sends one (same id on every image of one vehicle);
    # otherwise images from one sender within group_timeout_s are treated as the same vehicle.
    group_timeout_s: float = Field(3.0, ge=0.0, le=60.0)  # a gap this long ends a vehicle's burst
    max_images_per_vehicle: int = Field(20, ge=1, le=200)  # cap reads per vehicle (memory/CPU bound)
    # Return the read plate in the HTTP response instead of only an ack. Each POST is then treated as one
    # complete vehicle (read+voted inline) and the reply carries "plate"/"plates". Use this when the camera
    # sends ALL of a vehicle's images in ONE POST (the GVD shape: BgImg + PlateImg together). Leave it off
    # for cameras that send one-image-per-POST and rely on time-gap grouping (the result can't be known
    # until the burst ends). The plate is still sent to the client API (push:) either way: the reply's
    # "push" says whether THIS plate was accepted now, or is queued for the background sender (API down,
    # or behind earlier plates). A plate repeated within vote.dedupe_window_s comes back "duplicate": true.
    respond_with_plate: bool = False
    # Second look when a vehicle gives NO valid plate: search zoomed-in parts of each image (2x2 overlapping
    # tiles) and try every plate box found, best score first. The detector sees the whole photo at 320 px,
    # so a plate that is small in a big photo (or sits next to a bigger, unreadable hand-painted one) can be
    # missed or never read. Costs only on failures: ~4 extra detector runs + a few OCR reads per image.
    zoom_fallback: bool = True
    zoom_max_candidates: int = Field(4, ge=1, le=16)  # plate boxes tried per image in the second look


class StorageConfig(_Section):
    db_path: Path = Path("data/anpr.db")
    image_dir: Path = Path("data/images")
    retention_days: float = Field(30.0, gt=0.0)
    snapshot_width: int = Field(640, gt=0)


class WebConfig(_Section):
    station_name: str = Field("Gate 1", min_length=1, max_length=40)  # shown in the dashboard header
    host: str = "0.0.0.0"
    port: int = Field(8000, gt=0, lt=65536)
    auth_token: str | None = None  # if set, required for API/WS as "Authorization: Bearer <token>"
    # Admin password: if set, changing things (Activate licence, Settings, Add stream) also needs it, so
    # people who can only VIEW the dashboard cannot. Sent as header "X-Admin-Token"; the dashboard asks
    # for it once per browser session. Not set: anyone with auth_token is admin, and with no tokens at
    # all a licence can only be activated from the device itself.
    admin_token: str | None = Field(None, min_length=8, max_length=200)
    # View-only dashboard (e.g. shared with a client over the internet): nobody can change the stream.
    read_only: bool = False
    # Headless installs: false = do not run the dashboard/API at all (the .deb installer then disables
    # the web service; `python -m anpr.web` exits). The engine still reads plates and pushes to the API.
    enabled: bool = True


class RuntimeConfig(_Section):
    max_rss_mb: float = Field(600.0, gt=0.0)  # Pi 3B+ has 1 GB total
    status_interval_s: float = Field(2.0, gt=0.0)
    # Dashboard "Live" view: every N seconds the engine saves the latest frame (small JPEG with the
    # detector's boxes drawn on it): in RAM (/dev/shm) on Linux, else next to the image folder. 0 = off.
    preview_interval_s: float = Field(1.0, ge=0.0)
    preview_width: int = Field(640, ge=64, le=1920)
    preview_quality: int = Field(70, ge=20, le=95)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class AppConfig(_Section):
    camera: CameraConfig = Field(default_factory=CameraConfig)
    # Two or more cameras on ONE device (Pi 3B+: keep it to a quiet/medium gate). Each runs the identical
    # pipeline (no accuracy loss per camera) in one process to share memory. Only source + name differ;
    # all other camera settings come from `camera:` above. None/empty = single camera (`camera.source`).
    cameras: list[CameraFeed] | None = None
    motion: MotionConfig = Field(default_factory=MotionConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    ocr: OcrConfig = Field(default_factory=OcrConfig)
    crop: CropConfig = Field(default_factory=CropConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    vote: VoteConfig = Field(default_factory=VoteConfig)
    hsrp: HsrpConfig = Field(default_factory=HsrpConfig)
    push: PushConfig | None = None  # None = not in config.yaml: sending is managed from the dashboard
    # "Push" mode: a camera POSTs vehicle images to this Pi instead of the Pi pulling an RTSP stream.
    ingest: IngestConfig | None = None
    storage: StorageConfig = Field(default_factory=StorageConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    def resolve_paths(self, base: Path) -> AppConfig:
        """Return a copy with every relative path made absolute against `base`."""

        def fix(p: Path) -> Path:
            return p if p.is_absolute() else (base / p).resolve()

        return self.model_copy(
            update={
                "detector": self.detector.model_copy(update={"model_path": fix(self.detector.model_path)}),
                "ocr": self.ocr.model_copy(
                    update={
                        "model_path": fix(self.ocr.model_path),
                        "config_path": fix(self.ocr.config_path),
                    }
                ),
                "storage": self.storage.model_copy(
                    update={
                        "db_path": fix(self.storage.db_path),
                        "image_dir": fix(self.storage.image_dir),
                    }
                ),
            }
        )


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load and validate config. Relative paths resolve against the config file's folder."""
    if path is None:
        return AppConfig().resolve_paths(Path.cwd())
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return AppConfig.model_validate(data).resolve_paths(path.parent.resolve())
