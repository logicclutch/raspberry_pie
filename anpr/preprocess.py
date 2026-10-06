"""Crop and fix the plate image (WORKFLOW.md step 5).

prepare_plate(frame, box, cfg): crop from the FULL-resolution frame with padding -> quality gate
(too small / too blurry -> None, better to wait for a better frame) -> deskew (perspective warp from
the plate's corners) -> CLAHE contrast boost. Output is a BGR crop; the OCR module resizes it itself.
"""

from __future__ import annotations

import functools
import math

import cv2
import numpy as np

from anpr.config import CropConfig
from anpr.types import Box

# Plausible plate aspect ratios (width / height) after deskew. Indian plates: 1-line ~500x120 (4.2),
# 2-line ~340x200 (1.7), bikes ~200x100 (2.0). Anything outside means the corner search went wrong.
MIN_PLATE_ASPECT = 1.0
MAX_PLATE_ASPECT = 7.0
# The bright plate blob must cover at least this share of the padded crop. If it covers more than
# MAX_QUAD_AREA_RATIO the plate edges aren't visible (or the crop is flat) -> nothing to straighten.
MIN_QUAD_AREA_RATIO = 0.25
MAX_QUAD_AREA_RATIO = 0.97
# A real plate seen from the side is still roughly a rectangle. Reject corner sets that would shear or
# fold the image (seen on real video: merged bumper blobs gave italic text and flat grey crops).
MIN_CORNER_ANGLE = 60.0
MAX_CORNER_ANGLE = 120.0
MAX_SIDE_RATIO = 2.0  # opposite sides may differ this much (steep side view: near edge looks taller)
# Blank border kept around the warped plate: OCR drops characters that touch the image edge.
WARP_MARGIN = 0.06
MIN_OUTPUT_STD = 12.0  # a warped plate must still have contrast (text on background)


def clip_box(
    box: Box, frame_shape: tuple[int, ...], padding: float = 0.0
) -> tuple[int, int, int, int] | None:
    """Box grown by `padding` (fraction of its width/height on each side), clipped to the frame."""
    h, w = frame_shape[:2]
    px = round(box.width * padding)
    py = round(box.height * padding)
    x1, y1 = max(0, box.x1 - px), max(0, box.y1 - py)
    x2, y2 = min(w, box.x2 + px), min(h, box.y2 + py)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2, y2


def crop_plate(frame: np.ndarray, box: Box, padding: float) -> np.ndarray | None:
    """Padded crop (a view into `frame`, no copy) or None if the box is outside the frame."""
    r = clip_box(box, frame.shape, padding)
    if r is None:
        return None
    x1, y1, x2, y2 = r
    return frame[y1:y2, x1:x2]


def sharpness(img: np.ndarray) -> float:
    """Variance of the Laplacian (higher = sharper). Accepts BGR or gray."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """4x2 points -> (top-left, top-right, bottom-right, bottom-left)."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], dtype=np.float32)


def find_plate_quad(crop: np.ndarray) -> np.ndarray | None:
    """Corners (tl, tr, br, bl) of the plate inside a padded crop, or None if not found reliably.

    Plates are a bright (white/yellow) rectangle: Otsu threshold, close the character holes, take the
    largest blob and reduce its hull to 4 corners (falling back to the min-area rectangle).
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = max(3, (min(h, w) // 8) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if not (MIN_QUAD_AREA_RATIO * h * w <= area <= MAX_QUAD_AREA_RATIO * h * w):
        return None
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    quad = None
    for eps in (0.02, 0.04, 0.06):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            quad = approx
            break
    if quad is None:
        quad = cv2.boxPoints(cv2.minAreaRect(hull))
    return _order_corners(quad)


def quad_is_plausible(quad: np.ndarray) -> bool:
    """4 distinct corners forming a convex, nearly rectangular shape (no shear, no fold)."""
    pts = quad.reshape(4, 2).astype(np.float32)
    for i in range(4):
        for j in range(i + 1, 4):
            if np.linalg.norm(pts[i] - pts[j]) < 4:
                return False
    if not cv2.isContourConvex(pts.reshape(-1, 1, 2)):
        return False
    for i in range(4):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % 4]
        v1, v2 = a - b, c - b
        cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
        if not (MIN_CORNER_ANGLE <= ang <= MAX_CORNER_ANGLE):
            return False
    tl, tr, br, bl = pts
    for s1, s2 in ((tr - tl, br - bl), (bl - tl, br - tr)):
        n1, n2 = float(np.linalg.norm(s1)), float(np.linalg.norm(s2))
        if max(n1, n2) / max(min(n1, n2), 1e-6) > MAX_SIDE_RATIO:
            return False
    return True


def deskew(crop: np.ndarray) -> np.ndarray:
    """Perspective-warp the plate to a straight rectangle. Returns `crop` unchanged when unsure."""
    quad = find_plate_quad(crop)
    if quad is None or not quad_is_plausible(quad):
        return crop
    tl, tr, br, bl = quad
    out_w = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    out_h = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    if out_h < 8 or out_w < 16:
        return crop
    aspect = out_w / out_h
    if not (MIN_PLATE_ASPECT <= aspect <= MAX_PLATE_ASPECT):
        return crop
    w, h = round(out_w), round(out_h)
    mx, my = round(w * WARP_MARGIN), round(h * WARP_MARGIN)
    dst = np.array([[mx, my], [mx + w - 1, my], [mx + w - 1, my + h - 1], [mx, my + h - 1]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(quad, dst)
    out = cv2.warpPerspective(
        crop, m, (w + 2 * mx, h + 2 * my), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY) if out.ndim == 3 else out
    if float(gray.std()) < MIN_OUTPUT_STD:
        return crop
    return out


@functools.lru_cache(maxsize=8)
def _clahe(clip: float) -> cv2.CLAHE:
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(4, 4))


def enhance(crop: np.ndarray, clahe_clip: float) -> np.ndarray:
    """CLAHE on the lightness channel (helps glare, shadow, night); colours are kept. Returns BGR."""
    if crop.ndim == 2:
        return cv2.cvtColor(_clahe(clahe_clip).apply(crop), cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = _clahe(clahe_clip).apply(np.ascontiguousarray(lab[:, :, 0]))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# -- de-shear (side-mounted cameras) ---------------------------------------------------------------
# The plate is seen as a parallelogram: strokes stay ~vertical, the text baseline rises 25-40 deg.
# Find that slope from the dark text pixels inside the bright plate blob, shear it flat, crop.
DESHEAR_WORK_W = 96  # analyse a downscaled copy: cost is O(slopes x text pixels)
DESHEAR_VMARGIN = 0.15  # blank band kept above/below the plate (fraction of plate height)
DESHEAR_HMARGIN = 0.04  # ... left/right (fraction of plate width). >=0.1 produced wrong reads
TWO_LINE_ASPECT = 1.3  # blob w/h below this = 2-line plate -> keep plain crop
MIN_BAND = 0.18  # band height at least this x plate width
_COARSE = np.tan(np.radians(np.arange(-40.0, 40.1, 2.5)))


def _best_slope(ys: np.ndarray, xc: np.ndarray, slopes: np.ndarray, rows: int, off: int) -> float:
    """Slope whose sheared row profile of the text pixels is the most peaked (sum of squares)."""
    r = np.rint(ys[None, :] - slopes[:, None] * xc[None, :]).astype(np.int32) + off
    r += (np.arange(len(slopes), dtype=np.int32) * rows)[:, None]
    prof = np.bincount(r.ravel(), minlength=rows * len(slopes)).reshape(len(slopes), rows)
    return float(slopes[int((prof.astype(np.float32) ** 2).sum(axis=1).argmax())])


def analyse_plate(gray: np.ndarray) -> tuple[float, tuple[float, float, float, float] | None]:
    """-> (slope dy/dx, band (x0, x1, y0, y1) in de-sheared `gray` coords, or None)."""
    full_h, full_w = gray.shape
    s = 1.0
    if full_w > DESHEAR_WORK_W:
        s = DESHEAR_WORK_W / full_w
        gray = cv2.resize(gray, (DESHEAR_WORK_W, max(2, round(full_h * s))), interpolation=cv2.INTER_AREA)
    h, w = gray.shape
    dark = cv2.adaptiveThreshold(gray, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    _, bright = cv2.threshold(cv2.GaussianBlur(gray, (3, 3), 0), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = max(3, (min(h, w) // 8) | 1)
    closed = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    blob = None
    if n > 1:
        big = 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
        if stats[big, cv2.CC_STAT_AREA] >= 0.12 * h * w:
            blob = lab == big
    txt = dark & blob.astype(np.uint8) if blob is not None else dark  # text inside the plate only
    ys, xs = np.nonzero(txt)
    if len(xs) < 20:
        return 0.0, None
    ys = ys.astype(np.float32)
    xc = xs.astype(np.float32) - w / 2.0
    off = int(math.ceil(1.1 * w / 2)) + 2
    rows = h + 2 * off
    t = _best_slope(ys, xc, _COARSE, rows, off)
    a0 = math.degrees(math.atan(t))
    t = _best_slope(ys, xc, np.tan(np.radians(np.arange(a0 - 2.0, a0 + 2.01, 0.5))), rows, off)
    if blob is None:
        return t, None
    by, bx = np.nonzero(blob)
    yp = by - t * (bx - w / 2.0)
    y0, y1 = np.percentile(yp, [2, 98])
    x0, x1 = np.percentile(bx, [1, 99])
    return t, (x0 / s, (x1 + 1) / s, y0 / s, (y1 + 1) / s)


def deshear_plate(frame: np.ndarray, box: Box) -> np.ndarray | None:
    """Straightened plate crop, or None when the plate looks 2-line (caller keeps the plain crop)."""
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = max(0, box.x1), max(0, box.y1), min(fw, box.x2), min(fh, box.y2)
    inner = frame[y1:y2, x1:x2]
    if inner.size == 0:
        return None
    gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY) if inner.ndim == 3 else inner
    h, w = gray.shape
    t, band = analyse_plate(gray)
    if band is None:
        bx0, bx1 = 0.0, float(w)
        bh = max(h - abs(t) * w, MIN_BAND * w)
        by0, by1 = h / 2 - bh / 2, h / 2 + bh / 2
    else:
        bx0, bx1, by0, by1 = band
        if bx1 - bx0 < TWO_LINE_ASPECT * (by1 - by0):
            return None  # 2-line plate
        if by1 - by0 < MIN_BAND * (bx1 - bx0):
            c, half = (by0 + by1) / 2, MIN_BAND * (bx1 - bx0) / 2
            by0, by1 = c - half, c + half
    mx, my = DESHEAR_HMARGIN * (bx1 - bx0), DESHEAR_VMARGIN * (by1 - by0)
    ox0, oy0 = bx0 - mx, by0 - my
    ow, oh = max(8, round(bx1 + mx - ox0)), max(8, round(by1 + my - oy0))
    # output (u, v) samples frame (X, Y) = (u + ox0 + x1, v + oy0 + t*(u + ox0 - w/2) + y1)
    m = np.float32([[1, 0, ox0 + x1], [t, 1, oy0 + t * (ox0 - w / 2.0) + y1]])
    return cv2.warpAffine(
        frame, m, (ow, oh), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE
    )


def reread_plate(frame: np.ndarray, box: Box, cfg: CropConfig) -> np.ndarray | None:
    """Second-chance crop for a read that failed the format check: de-shear from a box widened by
    `cfg.retry_expand` on each side, so a plate end the detector box cut off is included.
    None when there is nothing different to try (retry or de-shear off, or a 2-line plate)."""
    if not cfg.deshear or cfg.retry_expand <= 0:
        return None
    e = round(box.width * cfg.retry_expand)
    if e < 1:
        return None
    flat = deshear_plate(frame, Box(box.x1 - e, box.y1, box.x2 + e, box.y2, box.score))
    return None if flat is None else enhance(flat, cfg.clahe_clip)


TOP_LINE_HEIGHT = 1.05  # 2-line plate: the top line sits in a band about one box-height above
TOP_LINE_XPAD = 3  # px added left/right; the top line is often a bit wider than the box


def top_line_crops(frame: np.ndarray, box: Box, cfg: CropConfig) -> tuple[np.ndarray, np.ndarray] | None:
    """2-line plates (trucks, buses): the detector often boxes only the bottom line ("BC8199")
    and misses the top one ("TN36"). Returns (top line crop, whole plate crop), both enhanced,
    or None when off or the band above the box is outside the frame."""
    if not cfg.top_line_rescue:
        return None
    h, w = frame.shape[:2]
    bh = box.y2 - box.y1
    top_y = box.y1 - round(TOP_LINE_HEIGHT * bh)
    if top_y < 0 or bh < 4:
        return None
    x1, x2 = max(0, box.x1 - TOP_LINE_XPAD), min(w, box.x2 + TOP_LINE_XPAD)
    top = frame[top_y : box.y1 + 1, x1:x2]
    whole = frame[top_y : min(h, box.y2 + 2), x1:x2]
    if top.size == 0 or whole.size == 0:
        return None
    return enhance(top, cfg.clahe_clip), enhance(whole, cfg.clahe_clip)


TWO_LINE_BOX_ASPECT = 2.2  # box w/h below this can hold a 2-line plate (1-line boxes are ~4:1)
IN_BOX_TOP = 0.5  # top line = upper half of the box (0.55 pulled bottom-line tops into the read)


def in_box_top_crops(frame: np.ndarray, box: Box, cfg: CropConfig) -> tuple[np.ndarray, np.ndarray] | None:
    """2-line plate boxed whole, but the reader returned only the bottom line ("A3075" for
    HR38AA3075): the top line is in the upper part of the box. Returns (top line crop, whole plate
    crop), both enhanced, or None when off or the box is too wide for a 2-line plate."""
    if not cfg.top_line_rescue:
        return None
    x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
    bw, bh = x2 - x1, y2 - y1
    if bh < 8 or bw >= TWO_LINE_BOX_ASPECT * bh:
        return None
    h, w = frame.shape[:2]
    x1, x2 = max(0, x1 - TOP_LINE_XPAD), min(w, x2 + TOP_LINE_XPAD)
    y1, y2 = max(0, y1), min(h, y2)
    top = frame[y1 : y1 + round(IN_BOX_TOP * bh), x1:x2]
    whole = frame[y1:y2, x1:x2]
    if top.size == 0 or whole.size == 0:
        return None
    return enhance(top, cfg.clahe_clip), enhance(whole, cfg.clahe_clip)


def prepare_plate(frame: np.ndarray, box: Box, cfg: CropConfig) -> np.ndarray | None:
    """Full-frame BGR + plate box -> clean BGR plate crop, or None if it fails the quality gate."""
    if box.width < cfg.min_width_px:
        return None
    inner = crop_plate(frame, box, 0.0)
    if inner is None or sharpness(inner) < cfg.min_sharpness:
        return None
    if cfg.deshear:
        flat = deshear_plate(frame, box)
        if flat is not None:
            return enhance(flat, cfg.clahe_clip)  # else a 2-line plate: plain crop below
    crop = crop_plate(frame, box, cfg.padding)
    if crop is None:
        return None
    if cfg.deskew:
        crop = deskew(crop)
    return enhance(crop, cfg.clahe_clip)
