"""Crop / quality gate / deskew / enhance on synthetic plates."""

from __future__ import annotations

import cv2
import numpy as np

from anpr.config import CropConfig
from anpr.preprocess import (
    crop_plate,
    deskew,
    enhance,
    find_plate_quad,
    prepare_plate,
    quad_is_plausible,
    sharpness,
)
from anpr.types import Box

PW, PH = 300, 70  # plate size in px (aspect ~4.3, like a 1-line Indian plate)


def plate_image() -> np.ndarray:
    img = np.full((PH, PW, 3), 245, np.uint8)
    cv2.rectangle(img, (0, 0), (PW - 1, PH - 1), (0, 0, 0), 3)
    cv2.putText(img, "MH12AB1234", (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 0), 3, cv2.LINE_AA)
    return img


def scene_with_plate(transform: str = "none") -> tuple[np.ndarray, Box]:
    """Dark 1280x720 frame with the plate pasted at (500, 300), optionally rotated/warped -> (frame, box)."""
    frame = np.full((720, 1280, 3), 40, np.uint8)
    plate = plate_image()
    src = np.array([[0, 0], [PW, 0], [PW, PH], [0, PH]], np.float32)
    ox, oy = 500.0, 300.0
    if transform == "rotate":
        m = cv2.getRotationMatrix2D((PW / 2, PH / 2), 15, 1.0)
        dst = cv2.transform(src[None], m)[0]
    elif transform == "perspective":
        dst = np.array([[0, 10], [PW, 0], [PW - 15, PH + 5], [10, PH - 5]], np.float32)
    else:
        dst = src.copy()
    dst = dst + np.float32([ox, oy])
    h = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(plate, h, (1280, 720), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((PH, PW), 255, np.uint8), h, (1280, 720))
    frame[mask > 127] = warped[mask > 127]
    x1, y1 = np.floor(dst.min(axis=0)).astype(int)
    x2, y2 = np.ceil(dst.max(axis=0)).astype(int)
    return frame, Box(int(x1), int(y1), int(x2), int(y2), 0.9)


def text_rows_are_horizontal(img: np.ndarray) -> float:
    """Angle (deg) of the dominant dark-text blob's min-area rect, normalised to [-45, 45]."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    inner = gray[h // 6 : h - h // 6, w // 20 : w - w // 20]
    _, ink = cv2.threshold(inner, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    pts = cv2.findNonZero(ink)
    (_, _), (rw, rh), ang = cv2.minAreaRect(pts)
    if rw < rh:
        ang -= 90
    return ((ang + 45) % 90) - 45


def test_straight_plate_passes_and_keeps_aspect() -> None:
    frame, box = scene_with_plate()
    out = prepare_plate(frame, box, CropConfig())
    assert out is not None and out.ndim == 3 and out.dtype == np.uint8
    assert 3.6 < out.shape[1] / out.shape[0] < 5.0


def test_deskew_rotated_plate() -> None:
    frame, box = scene_with_plate("rotate")
    raw = crop_plate(frame, box, CropConfig().padding)
    assert raw is not None
    quad = find_plate_quad(raw)
    assert quad is not None
    top_edge = quad[1] - quad[0]
    assert abs(np.degrees(np.arctan2(top_edge[1], top_edge[0]))) > 12  # really tilted before
    out = deskew(raw)
    assert out is not raw
    assert 3.6 < out.shape[1] / out.shape[0] < 5.0
    assert abs(text_rows_are_horizontal(out)) < 3


def test_deskew_perspective_plate() -> None:
    frame, box = scene_with_plate("perspective")
    out = prepare_plate(frame, box, CropConfig(deskew=True))
    assert out is not None
    assert 3.6 < out.shape[1] / out.shape[0] < 5.0
    assert abs(text_rows_are_horizontal(out)) < 3


def test_deskew_leaves_crop_alone_when_no_plate_found() -> None:
    flat = np.full((60, 200, 3), 128, np.uint8)
    assert find_plate_quad(flat) is None
    assert deskew(flat) is flat
    noise = np.random.default_rng(0).integers(0, 255, (60, 200, 3), dtype=np.uint8)
    out = deskew(noise)
    assert out.ndim == 3 and 1.0 <= out.shape[1] / out.shape[0] <= 7.0


def test_blurry_plate_fails_gate() -> None:
    frame, box = scene_with_plate()
    blurred = cv2.GaussianBlur(frame, (0, 0), 6)
    cfg = CropConfig()
    assert sharpness(frame[box.y1 : box.y2, box.x1 : box.x2]) >= cfg.min_sharpness
    assert sharpness(blurred[box.y1 : box.y2, box.x1 : box.x2]) < cfg.min_sharpness
    assert prepare_plate(blurred, box, cfg) is None


def test_tiny_box_fails_gate() -> None:
    frame, _ = scene_with_plate()
    assert prepare_plate(frame, Box(500, 300, 560, 314, 0.9), CropConfig()) is None


def test_box_outside_frame() -> None:
    frame, _ = scene_with_plate()
    assert prepare_plate(frame, Box(1300, 800, 1500, 850, 0.9), CropConfig()) is None


def test_box_at_edge_is_clipped() -> None:
    frame, _ = scene_with_plate()
    crop = crop_plate(frame, Box(1200, 700, 1300, 740, 0.9), 0.1)
    assert crop is not None and crop.shape[1] <= 90 and crop.shape[0] <= 24


def test_no_deskew_option_and_enhance_gray() -> None:
    frame, box = scene_with_plate("rotate")
    out = prepare_plate(frame, box, CropConfig(deskew=False))
    padded = crop_plate(frame, box, CropConfig().padding)
    assert out is not None and padded is not None and out.shape == padded.shape
    g = enhance(np.full((20, 60), 100, np.uint8), 2.0)
    assert g.shape == (20, 60, 3)


def test_prepare_does_not_modify_frame() -> None:
    frame, box = scene_with_plate("perspective")
    before = frame.copy()
    prepare_plate(frame, box, CropConfig())
    assert np.array_equal(frame, before)


def test_deskew_keeps_a_margin_around_the_plate() -> None:
    frame, box = scene_with_plate("rotate")
    raw = crop_plate(frame, box, CropConfig().padding)
    quad = find_plate_quad(raw)
    out = deskew(raw)
    plate_w = float(np.linalg.norm(quad[1] - quad[0]))
    plate_h = float(np.linalg.norm(quad[3] - quad[0]))
    # Output is the plate plus a border on every side, so edge characters never touch the image edge.
    assert out.shape[1] >= plate_w * 1.1 and out.shape[0] >= plate_h * 1.1


def test_implausible_quads_rejected() -> None:
    rect = np.float32([[0, 0], [100, 0], [100, 25], [0, 25]])
    assert quad_is_plausible(rect)
    sheared = np.float32([[40, 0], [140, 0], [100, 25], [0, 25]])  # italic-making shear (32/148 deg)
    assert not quad_is_plausible(sheared)
    duplicate = np.float32([[0, 0], [100, 0], [100, 0], [0, 25]])  # folded -> flat grey output
    assert not quad_is_plausible(duplicate)
    side_view = np.float32([[0, 0], [100, -10], [100, 35], [0, 25]])  # steep side angle is fine
    assert quad_is_plausible(side_view)


def test_default_config_does_not_deskew() -> None:
    frame, box = scene_with_plate("rotate")
    assert CropConfig().deskew is False
    out = prepare_plate(frame, box, CropConfig())
    raw = crop_plate(frame, box, CropConfig().padding)
    assert out is not None and out.shape == raw.shape  # plain padded crop, no warp


def _slanted_plate_frame(slope_deg: float) -> tuple[np.ndarray, Box]:
    """A white plate with dark 'characters' on a dark road, sheared like a side-mounted camera sees it."""
    plate = np.full((40, 200, 3), 235, np.uint8)
    for i in range(10):
        cv2.rectangle(plate, (8 + i * 19, 9), (19 + i * 19, 31), (20, 20, 20), -1)
    t = float(np.tan(np.radians(slope_deg)))
    out_h = 40 + int(abs(t) * 200) + 4
    m = np.float32([[1, 0, 0], [-t, 1, (out_h - 40) / 2 + t * 100]])  # baseline rises to the right
    sheared = cv2.warpAffine(plate, m, (200, out_h), borderValue=(40, 40, 40))
    frame = np.full((400, 600, 3), 40, np.uint8)
    frame[100 : 100 + out_h, 150:350] = sheared
    return frame, Box(150, 100, 350, 100 + out_h, 0.9)


def test_deshear_straightens_a_30_degree_plate():
    from anpr.preprocess import analyse_plate, deshear_plate

    frame, box = _slanted_plate_frame(30.0)
    gray = cv2.cvtColor(frame[box.y1 : box.y2, box.x1 : box.x2], cv2.COLOR_BGR2GRAY)
    slope, band = analyse_plate(gray)
    assert band is not None
    assert abs(np.degrees(np.arctan(slope)) + 30.0) < 3.0  # found the slant (image y grows down)
    flat = deshear_plate(frame, box)
    assert flat is not None
    h, w = flat.shape[:2]
    assert w / h > 3.0  # a flat 1-line plate, not the tall slanted box (200 x ~160)
    dark_rows = (cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY) < 100).mean(axis=1)
    assert dark_rows.max() > 0.3  # the characters line up on the same rows again


def test_deshear_leaves_two_line_plates_to_the_plain_crop():
    from anpr.preprocess import deshear_plate

    frame = np.full((400, 600, 3), 40, np.uint8)
    cv2.rectangle(frame, (200, 100), (280, 170), (235, 235, 235), -1)  # w/h ~1.1 = 2-line plate
    for row in (110, 140):
        for i in range(4):
            cv2.rectangle(frame, (208 + i * 18, row), (218 + i * 18, row + 20), (20, 20, 20), -1)
    assert deshear_plate(frame, Box(195, 95, 285, 175, 0.9)) is None


def test_prepare_plate_uses_deshear_only_when_enabled():
    frame, box = _slanted_plate_frame(30.0)
    plain = prepare_plate(frame, box, CropConfig(min_sharpness=0.0))
    flat = prepare_plate(frame, box, CropConfig(min_sharpness=0.0, deshear=True))
    assert plain is not None and flat is not None
    assert flat.shape[1] / flat.shape[0] > plain.shape[1] / plain.shape[0] + 1.0


def _cut_plate(frame_box: tuple[np.ndarray, Box], cut: int) -> tuple[np.ndarray, Box]:
    frame, box = frame_box
    return frame, Box(box.x1, box.y1, box.x2 - cut, box.y2, box.score)  # detector box stops short


def test_reread_plate_is_off_without_deshear_or_expand():
    from anpr.preprocess import reread_plate

    frame, box = _slanted_plate_frame(30.0)
    assert reread_plate(frame, box, CropConfig(deshear=False)) is None
    assert reread_plate(frame, box, CropConfig(deshear=True, retry_expand=0.0)) is None


def test_reread_plate_recovers_the_end_a_short_box_cut_off():
    from anpr.preprocess import deshear_plate, reread_plate

    frame, box = _cut_plate(_slanted_plate_frame(30.0), cut=30)  # last ~1.5 characters outside the box
    short = deshear_plate(frame, box)
    wide = reread_plate(frame, box, CropConfig(deshear=True, retry_expand=0.1))
    assert short is not None and wide is not None
    assert wide.shape[1] > short.shape[1] * 1.08  # the cut-off end is back in the crop
    assert wide.shape[1] / wide.shape[0] > 3.0  # still a flat 1-line plate


def test_reread_plate_skips_two_line_plates():
    from anpr.preprocess import reread_plate

    frame = np.full((400, 600, 3), 40, np.uint8)
    cv2.rectangle(frame, (200, 100), (280, 170), (235, 235, 235), -1)
    for row in (110, 140):
        for i in range(4):
            cv2.rectangle(frame, (208 + i * 18, row), (218 + i * 18, row + 20), (20, 20, 20), -1)
    assert reread_plate(frame, Box(195, 95, 285, 175, 0.9), CropConfig(deshear=True)) is None
