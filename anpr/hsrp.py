"""HSRP check: does a plate carry the High Security Registration Plate marks?

Every HSRP plate (India, compulsory for new vehicles since April 2019) has, left of the first
character, a chromium hologram (blue Ashoka Chakra, top-left) with "IND" in blue below it. Older
and home-made plates (painted, handwritten, fancy fonts) have neither.

Per frame, `look()` straightens the plate, finds the row(s) of characters and inspects the space
left of the first character (2-line plates: left of both lines, where the hologram sits top-left
and "IND" runs down the edge; there only a hologram-sized mark counts, since bolt heads sit there):
  - colour video: blue ink there -> HSRP;
  - any video (also black-and-white night/IR): a small dark mark in the upper part of that space
    (the hologram) -> HSRP; a clean, empty plate background there -> not HSRP;
  - characters not found, space cut off by the picture edge, plate too small -> can't tell (None).
`decide()` turns a track's per-frame looks into "hsrp", "non_hsrp" or "unsure".

A camera cannot verify a plate: a look-alike plate with a blue sticker passes. The result means
"looks like HSRP", nothing more.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import cv2
import numpy as np

from anpr.preprocess import crop_plate, deshear_plate
from anpr.types import Box

HsrpLook = Literal["hsrp", "non_hsrp"]
HsrpResult = Literal["hsrp", "non_hsrp", "unsure"]

WORK_W = 240  # the plate is analysed at this width
MIN_CHARS = 4  # need at least this many characters to know where the text starts
MIN_CONTRAST = 25  # grey levels between text and plate; less = too dark/blurred to judge
# A pixel is "mark" when darker than plate - k * (plate - text). The hologram is fainter than the
# text and its contrast varies (grey in colour video, dark under IR): try both levels.
MARK_LEVELS = (0.35, 0.5)
SPACE_W = 1.0  # width searched left of the first character, in character heights
# 2-line plates: only a hologram-sized mark counts. Bolt heads (small round dots) often sit left of
# the text on truck / bike plates, and on a 2-line plate they are where the "IND" letters would be.
MIN_MARK_2LINE = 0.22  # smallest mark, in character heights: hologram >= 20 mm vs 65 mm characters
# (>= 0.3), bolt heads 10-12 mm (0.15-0.18)
# 2-line plates centre their (short) lines, so the hologram can sit far left of the text: the space
# is searched up to the plate's left edge, at most this many character heights.
SPACE_W_2LINE = 3.0


def _left_edge(gray: np.ndarray, x0: int, y0: int, y1: int, dark: float, limit: int) -> int:
    """Walking left from x0, the first column (within `limit`) that is dark top-to-bottom of the
    band: the plate border or the plate's end. A hologram or bolt covers only part of the height,
    so it does not stop the walk."""
    for x in range(x0 - 1, max(-1, x0 - 1 - limit), -1):
        if float(np.median(gray[y0:y1, x])) < dark:
            return x + 1
    return max(0, x0 - limit)


def _plate_image(frame: np.ndarray, box: Box) -> np.ndarray | None:
    """The plate, grey, WORK_W wide: straightened when possible (1-line), else the plain box crop
    (deshear_plate gives up on 2-line plates)."""
    plain = crop_plate(frame, box, 0.0)
    if plain is None or plain.size == 0:
        return None
    img = deshear_plate(frame, box)
    if img is None or img.size == 0:
        img = plain
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h = max(8, round(gray.shape[0] * WORK_W / gray.shape[1]))
    return cv2.resize(gray, (WORK_W, h), interpolation=cv2.INTER_CUBIC)


Row = tuple[int, float, float]  # (left x of the first character, character height, top y)


def _rows(binary: np.ndarray) -> list[Row] | None:
    """Rows of characters in a binary plate (text = 255), top first: one row (1-line plate) or two
    (2-line plate), or None when no clear text is found."""
    h = binary.shape[0]
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cand = []
    for i in range(1, n):
        x, y, w, hh, area = (int(v) for v in stats[i])
        if 0.15 * h <= hh <= 0.95 * h and 0.08 * hh <= w <= 1.0 * hh and area >= 0.15 * w * hh:
            cand.append((x, y, w, hh))
    if len(cand) < MIN_CHARS:
        return None
    m = float(np.median([c[3] for c in cand]))
    chars = [c for c in cand if abs(c[3] - m) <= 0.3 * m]
    # Two characters are on the same line when they are neighbours: side by side (a space between
    # groups, as in "BC 8199", is allowed) and at about
    # the same height. Chaining neighbours follows a tilted line without merging it with the next.
    parent = list(range(len(chars)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(chars):
        for j in range(i + 1, len(chars)):
            b = chars[j]
            gap = max(a[0], b[0]) - min(a[0] + a[2], b[0] + b[2])  # < 0: they overlap side by side
            if gap <= 2.0 * m and gap > -0.5 * m and abs((a[1] + a[3] / 2) - (b[1] + b[3] / 2)) <= 0.45 * m:
                parent[root(i)] = root(j)
    groups: dict[int, list[tuple[int, int, int, int]]] = {}
    for i, c in enumerate(chars):
        groups.setdefault(root(i), []).append(c)
    # every line of an Indian plate has >= 3 characters ("DL 3" / "AB 1234"): a blob or two (bolt,
    # hologram, sticker) is not a line. Lines top first.
    lines = sorted((g for g in groups.values() if len(g) >= 3), key=lambda g: min(c[1] for c in g))
    if len(chars) - sum(len(g) for g in lines) >= 2:
        return None  # text that is not on the line(s): a tilted 2-line plate seen badly, can't tell
    for g in lines:  # a line's centres lie on a straight (maybe tilted) line; two lines mixed don't
        xs = np.array([c[0] + c[2] / 2 for c in g], float)
        ys = np.array([c[1] + c[3] / 2 for c in g], float)
        fit = np.polyval(np.polyfit(xs, ys, 1), xs)
        if float(np.max(np.abs(ys - fit))) > 0.3 * m:
            return None
    if not (len(lines) == 2 or (len(lines) == 1 and len(lines[0]) >= MIN_CHARS)):
        return None  # no text, or 3+ lines
    return [(min(c[0] for c in g), m, float(np.median([c[1] for c in g]))) for g in lines]


def _has_mark(
    gray: np.ndarray,
    region: tuple[int, int, int, int],
    m: float,
    level: float,
    text: float,
    bg: float,
    min_size: float = 0.08,
) -> bool:
    """A hologram / "IND"-sized dark mark (at least `min_size` character heights) inside `region`
    (x0, y0, x1, y1)?"""
    x0, y0, x1, y1 = region
    dark = (gray[y0:y1, x0:x1] < bg - level * (bg - text)).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    lo = min_size * m
    for i in range(1, n):
        x, _, w, h, area = (int(v) for v in stats[i])
        if not (lo <= w <= 0.5 * m and lo <= h <= 0.5 * m and area >= 0.2 * w * h):
            continue  # too small (noise), too big (border, bumper, a character) or a thin line
        if x + w >= x1 - x0 or (x == 0 and w < 0.15 * m):
            continue  # part of the first character, or a sliver of the plate edge
        return True
    return False


def look(frame: np.ndarray, box: Box, two_line: bool = False) -> HsrpLook | None:
    """One frame's verdict for the plate in `box`, or None when it cannot be judged. 2-line plates
    are judged only with `two_line` (config hsrp.two_line): on low-resolution or black-and-white
    video, plate frames and bolt heads at their top-left corner look like a hologram."""
    gray = _plate_image(frame, box)
    if gray is None:
        return None
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    rows = _rows(binary)
    if rows is None:
        return None
    m = rows[0][1]
    x0 = min(r[0] for r in rows)  # 2-line: the marks run down the left edge, beside both lines
    top, bottom = rows[0][2], rows[-1][2] + m
    if x0 < 0.3 * m:
        return None  # no room left of the text in this picture (plate cut off)
    band = gray[int(top) : int(bottom), x0:]
    text, bg = float(np.percentile(band, 5)), float(np.percentile(band, 80))
    if bg - text < MIN_CONTRAST:
        return None
    if len(rows) == 2 and not two_line:
        return None
    two_line = len(rows) == 2
    left = int(x0 - SPACE_W * m)
    if two_line:
        dark = bg - 0.5 * (bg - text)
        left = _left_edge(gray, x0, int(top), int(bottom), dark, int(SPACE_W_2LINE * m))
        if x0 - left < 0.3 * m:
            return None  # the text runs into the border / picture edge: no space to judge
    # 1-line: beside the text. 2-line: only the top-left corner, level with the top line - the
    # hologram's place. Bolts sit lower, between the lines, and must not count as a hologram.
    region_bottom = rows[0][2] + 0.8 * m
    region = (
        max(0, left),
        max(0, int(top - 0.25 * m)),
        int(x0 - 0.05 * m),
        min(gray.shape[0], int(region_bottom)),
    )
    x0r, y0r, x1r, y1r = region
    if float(np.median(gray[y0r:y1r, x0r:x1r])) < bg - 0.5 * (bg - text):
        return None  # the space left of the text is not plate (edge cut off, bumper, shadow)
    min_size = MIN_MARK_2LINE if two_line else 0.08
    if any(_has_mark(gray, region, m, k, text, bg, min_size) for k in MARK_LEVELS):
        return "hsrp"
    return "non_hsrp"


def decide(looks: Sequence[HsrpLook | None], min_marked: int = 2, min_clean: int = 3) -> HsrpResult:
    """A track's per-frame looks -> result. Blur hides the mark far more often than dirt fakes one,
    so HSRP needs the mark in >= 30% of judged frames; non-HSRP needs a clean space in (almost) all."""
    marked = sum(1 for v in looks if v == "hsrp")
    clean = sum(1 for v in looks if v == "non_hsrp")
    judged = marked + clean
    if marked >= min_marked and marked >= 0.3 * judged:
        return "hsrp"
    if clean >= min_clean and marked <= 0.1 * judged:
        return "non_hsrp"
    return "unsure"
