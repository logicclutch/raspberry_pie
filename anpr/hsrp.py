"""HSRP check: does a plate carry the High Security Registration Plate marks?

Every HSRP plate (India, compulsory for new vehicles since April 2019) has, left of the first
character, a chromium hologram (blue Ashoka Chakra, top-left) with "IND" in blue below it. Older
and home-made plates (painted, handwritten, fancy fonts) have neither.

Per frame, `conditions()` straightens the plate, finds the row(s) of characters and checks four
conditions (2-line plates: left of both lines, where the hologram sits top-left and "IND" runs down
the edge; there only a hologram-sized mark counts, since bolt heads sit there):
  - hologram: a small dark mark (hologram / "IND" sized) in the space left of the text;
  - blue: blue ink in that space (colour pictures only);
  - ind: two or more small letter-like marks side by side or stacked below the hologram (2-line:
    down the left edge beside both lines; two round dots such as bolt heads are not letters);
  - font: every character the same height and stroke thickness (standard HSRP font).
Characters not found, space cut off by the picture edge, plate too dark -> can't tell (None).
Text running into the plate's border (no room for the marks at all) -> non-HSRP: old painted plates.
`look()` combines them by a rule (config hsrp.rule, see `passes`): "close" = hologram AND "IND",
"far" = "IND" AND one other, "any" = any one (OR). Each condition also has a weight (WEIGHTS /
config hsrp.weights); the frame's score = the sum of the weights of the conditions that passed,
logged to show how strong the evidence is. Weight 0 switches a condition off.
`decide()` turns a track's per-frame looks into "hsrp", "non_hsrp" or "unsure"; `why_unsure()`
says why in a few words, and `label()` makes the saved value, e.g. "not sure: text not found".

A camera cannot verify a plate: a look-alike plate with a blue sticker passes. The result means
"looks like HSRP", nothing more.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from anpr.config import HsrpRule, HsrpWeights
from anpr.preprocess import crop_plate, deshear_plate
from anpr.types import Box

log = logging.getLogger(__name__)

HsrpLook = Literal["hsrp", "non_hsrp"]
HsrpResult = Literal["hsrp", "non_hsrp", "unsure"]

# Why a frame could not be judged, kept short: shown on the saved plate as "not sure: <reason>"
# (or "hsrp (not sure: <reason>)" / "non_hsrp (not sure: <reason>)", see label()).
WHY_NO_PLATE = "box outside picture"  # the plate box is outside the frame
WHY_NO_TEXT = "text not found"  # characters not found: plate too small, blurred or fancy font
WHY_CUT_OFF = "plate cut off"  # no room left of the number in the plate box (or the box is tight)
WHY_LOW_CONTRAST = "low contrast"  # plate too dark or blurred
WHY_TWO_LINE_OFF = "2-line plate"  # 2-line plate and hsrp.two_line is off
WHY_NOT_PLATE = "left side not plate"  # the space left of the number is bumper, shadow or cut off
WHY_FAILED = "check error"  # the HSRP check raised an error
WHY_NOT_CHECKED = "not checked"  # no image of the plate was checked
WHY_ONE_LINE = "1-line plate"  # only_two_line asked and the plate has one line (never shown)

# The conditions of a plate with no room left of the number for the marks: none can pass.
NO_MARKS = {"hologram": False, "blue": False, "ind": False, "font": False}

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

# HSRP conditions, combined by a rule (`passes`, config hsrp.rule). The weights only make
# the score (how much evidence: all four = 1.0); weight 0 switches a condition off. Change them in
# config.yaml (hsrp.weights).
WEIGHTS = HsrpWeights().model_dump()  # defaults live in config.HsrpWeights
IND_MIN, IND_MAX = 0.08, 0.4  # size of one "IND" letter, in character heights
BLUE_HUE = (90, 130)  # OpenCV hue (0-180) of the blue hologram / "IND" ink
BLUE_MIN_SAT = 70  # 0-255: white plate under a bluish light is not "blue ink"
BLUE_MIN_AREA = 0.01  # smallest blue patch, in character heights squared
FONT_SAMPLE_CHARS = 3  # hsrp.method: font - only these first characters of the top line are measured
FONT_LETTER_HEIGHT_TOL = 0.1  # a blob this far from the line's median letter height is not a letter
FONT_PIXEL_SLACK = 1.5  # font limits widen by this many real pixels (a 20 px letter: +7.5%)
FONT_JOINED_HEIGHT = (0.45, 0.95)  # a top+bottom letter joined into one blob: this much of the plate tall
FONT_JOINED_MAX_W = 0.35  # ... and narrow (the plate's frame is wider)
FONT_SPLIT_BAND = (0.3, 0.7)  # where in such a blob the two letters meet
FONT_MIN_CHARS = 6  # Indian plates have 8-10 characters; fewer found = can't compare them
FONT_MAX_HEIGHT_CV = 0.05  # character heights vary by at most 5%
FONT_MAX_STROKE_CV = 0.15  # stroke thickness varies by at most 15%
FONT_STROKE = (0.10, 0.22)  # stroke thickness / character height of the bold HSRP font


def _left_edge(gray: np.ndarray, x0: int, y0: int, y1: int, dark: float, limit: int) -> int:
    """Walking left from x0, the first column (within `limit`) that is dark top-to-bottom of the
    band: the plate border or the plate's end. A hologram or bolt covers only part of the height,
    so it does not stop the walk."""
    for x in range(x0 - 1, max(-1, x0 - 1 - limit), -1):
        if float(np.median(gray[y0:y1, x])) < dark:
            return x + 1
    return max(0, x0 - limit)


def _plate_image(frame: np.ndarray, box: Box) -> tuple[np.ndarray, np.ndarray | None] | None:
    """The plate, WORK_W wide, as (grey, colour or None for a grey frame): straightened when
    possible (1-line), else the plain box crop (deshear_plate gives up on 2-line plates)."""
    plain = crop_plate(frame, box, 0.0)
    if plain is None or plain.size == 0:
        return None
    img = deshear_plate(frame, box)
    if img is None or img.size == 0:
        img = plain
    h = max(8, round(img.shape[0] * WORK_W / img.shape[1]))
    img = cv2.resize(img, (WORK_W, h), interpolation=cv2.INTER_CUBIC)
    if img.ndim == 2:
        return img, None
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), img


Row = tuple[int, float, float]  # (left x of the first character, character height, top y)
Char = tuple[int, int, int, int]  # (x, y, w, h)


def _rows(m: float, lines: list[list[Char]]) -> list[Row]:
    """Each line of characters (from _lines) -> (left x of its first character, character height, top y)."""
    return [(min(c[0] for c in g), m, float(np.median([c[1] for c in g]))) for g in lines]


def _lines(binary: np.ndarray) -> tuple[float, list[list[Char]]] | None:
    """(character height, the characters of each line, top line first), or None when no clear text."""
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
    return m, lines


def _dark_beside_text(
    gray: np.ndarray, region: tuple[int, int, int, int], m: float, level: float
) -> np.ndarray:
    """Dark pixels of `region`, labelled together with a strip of the text beside it (to the
    right): a mark ending near the region's edge (within 0.15 character heights) but not joined to
    the first character still counts, while one joined to it runs on into the text (the caller
    drops it). On plates with little room before the text the hologram almost touches the text."""
    x0, y0, x1, y1 = region
    ext = min(gray.shape[1], x1 + max(2, int(0.3 * m)))
    return (gray[y0:y1, x0:ext] < level).astype(np.uint8)


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
    dark = _dark_beside_text(gray, region, m, bg - level * (bg - text))
    n, _, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    lo = min_size * m
    for i in range(1, n):
        x, _, w, h, area = (int(v) for v in stats[i])
        if not (lo <= w <= 0.5 * m and lo <= h <= 0.5 * m and area >= 0.2 * w * h):
            continue  # too small (noise), too big (border, bumper, a character) or a thin line
        if x + w > x1 - x0 + 0.15 * m or (x == 0 and w < 0.15 * m):
            continue  # joined to the first character, or a sliver of the plate edge
        return True
    return False


def _has_ind(gray: np.ndarray, region: tuple[int, int, int, int], m: float, text: float, bg: float) -> bool:
    """The small "IND" letters below the hologram: two or more letter-sized dark marks side by side
    or stacked (the letters are often printed turned 90 degrees). Two round dots (bolt heads, dirt)
    are not letters, and bolt heads sit far apart while letters of one word sit close together."""
    x0, y0, x1, y1 = region
    if x1 - x0 < 2 or y1 - y0 < 2:
        return False
    for k in MARK_LEVELS:
        dark = _dark_beside_text(gray, region, m, bg - k * (bg - text))
        n, _, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
        letters = []
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i])
            if x == 0 or x + w > x1 - x0 + 0.15 * m:
                continue  # plate edge / joined to the first character
            if not (IND_MIN * m <= max(w, h) <= IND_MAX * m and area >= 0.15 * w * h):
                continue
            dot = 0.75 <= w / h <= 1.33 and area >= 0.65 * w * h
            letters.append((x + w / 2, y + h / 2, w, h, dot))
        for i, a in enumerate(letters):
            for b in letters[i + 1 :]:
                if a[4] and b[4]:
                    continue  # two dots
                size_a, size_b = max(a[2], a[3]), max(b[2], b[3])
                if max(size_a, size_b) > 2.0 * min(size_a, size_b):
                    continue  # letters of one word are about the same size
                reach = 0.6 * max(size_a, size_b)
                row = (
                    abs(a[1] - b[1]) <= 0.5 * max(a[3], b[3])
                    and abs(a[0] - b[0]) - (a[2] + b[2]) / 2 <= reach
                )
                stack = (
                    abs(a[0] - b[0]) <= 0.5 * max(a[2], b[2])
                    and abs(a[1] - b[1]) - (a[3] + b[3]) / 2 <= reach
                )
                if row or stack:
                    return True
    return False


def _has_blue(color: np.ndarray | None, region: tuple[int, int, int, int], m: float) -> bool:
    """Blue ink (hologram chakra / "IND") in the space left of the text. Colour pictures only: a
    black-and-white (night / IR) frame has no colour, so this condition never passes there."""
    if color is None:
        return False
    x0, y0, x1, y1 = region
    if x1 - x0 < 2 or y1 - y0 < 2:
        return False
    hsv = cv2.cvtColor(color[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    blue = ((hue >= BLUE_HUE[0]) & (hue <= BLUE_HUE[1]) & (sat >= BLUE_MIN_SAT) & (val >= 40)).astype(
        np.uint8
    )
    n, _, stats, _ = cv2.connectedComponentsWithStats(blue, connectivity=8)
    need = max(4, int(BLUE_MIN_AREA * m * m))
    for i in range(1, n):
        x, _, w, _, area = (int(v) for v in stats[i])
        if x == 0 or x + w >= x1 - x0:
            continue  # plate edge / body paint, or a coloured first character
        if area >= need:
            return True
    return False


def _standard_font(
    binary: np.ndarray, lines: list[list[Char]], min_chars: int = FONT_MIN_CHARS, slack: float = 0.0
) -> bool:
    """HSRP characters: one standard, bold, embossed font - every character the same height and the
    same stroke thickness. Painted / fancy plates vary. Weak: a neatly printed non-HSRP plate passes
    it too. `slack`: extra tolerance (fraction of the character height) for a plate so small that one
    pixel is several percent of a letter."""
    chars = [c for g in lines for c in g]
    if len(chars) < min_chars:
        return False
    hs = np.array([c[3] for c in chars], float)
    if hs.std() / hs.mean() > FONT_MAX_HEIGHT_CV + slack:
        return False
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    strokes = []
    for x, y, w, h in chars:
        d = dist[y : y + h, x : x + w]
        d = d[d > 0]
        if d.size == 0:
            return False
        strokes.append(2.0 * float(np.percentile(d, 90)))
    s = np.array(strokes)
    if s.std() / s.mean() > FONT_MAX_STROKE_CV + slack:
        return False
    return bool(FONT_STROKE[0] - slack <= s.mean() / hs.mean() <= FONT_STROKE[1] + slack)


def score(found: dict[str, bool], weights: Mapping[str, float] | None = None) -> float:
    """The weighted score of the conditions that passed (0..1)."""
    w = WEIGHTS if weights is None else weights
    return sum(w.get(name, 0.0) for name, ok in found.items() if ok)


def conditions(frame: np.ndarray, box: Box, two_line: bool = False) -> dict[str, bool] | None:
    """Which HSRP conditions this frame shows ({"hologram", "blue", "ind", "font"} -> passed), or
    None when the plate cannot be judged. 2-line plates are judged only with `two_line` (config
    hsrp.two_line): on low-resolution or black-and-white video, plate frames and bolt heads at their
    top-left corner look like a hologram."""
    found = conditions_or_reason(frame, box, two_line)
    return None if isinstance(found, str) else found


def conditions_or_reason(
    frame: np.ndarray, box: Box, two_line: bool = False, only_two_line: bool = False
) -> dict[str, bool] | str:
    """`conditions()`, but a plate that cannot be judged gives the reason (a WHY_* text) instead
    of None. `only_two_line`: judge 2-line plates only (a 1-line plate gives WHY_ONE_LINE)."""
    images = _plate_image(frame, box)
    if images is None:
        return WHY_NO_PLATE
    gray, color = images
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    found_lines = _lines(binary)
    if found_lines is None:
        return WHY_NO_TEXT
    m, lines = found_lines
    rows = _rows(m, lines)
    x0 = min(r[0] for r in rows)  # 2-line: the marks run down the left edge, beside both lines
    top, bottom = rows[0][2], rows[-1][2] + m
    if x0 < 0.3 * m:
        return WHY_CUT_OFF  # no room left of the text in the plate box (plate cut off, or a tight box)
    band = gray[int(top) : int(bottom), x0:]
    text, bg = float(np.percentile(band, 5)), float(np.percentile(band, 80))
    if bg - text < MIN_CONTRAST:
        return WHY_LOW_CONTRAST
    if len(rows) == 2 and not two_line:
        return WHY_TWO_LINE_OFF
    if only_two_line and len(rows) != 2:
        return WHY_ONE_LINE
    two_line = len(rows) == 2
    # The space left of the text ends at the plate's edge: plates with little room before the first
    # character (the hologram squeezed in) must not have the background behind the plate judged.
    dark = bg - 0.5 * (bg - text)
    limit = SPACE_W_2LINE if two_line else SPACE_W
    left = _left_edge(gray, x0, int(top), int(bottom), dark, int(limit * m))
    if x0 - left < 0.3 * m:
        # The text runs into the plate's border (a dark column top to bottom, which a hologram or
        # bolt is not). An HSRP always leaves room for the hologram and "IND" there, so this is
        # non-HSRP under every rule (old painted plates) - not "can't judge".
        return dict(NO_MARKS)
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
        return WHY_NOT_PLATE  # the space left of the text is not plate (edge cut off, bumper, shadow)
    # "IND" sits below the hologram, down to the bottom of the characters (2-line: of the lower line).
    mark_bottom = min(gray.shape[0], int(top + 1.05 * m))
    ind_bottom = min(gray.shape[0], int(bottom)) if two_line else mark_bottom
    ind_region = (x0r, max(0, int(top + 0.3 * m)), x1r, ind_bottom)
    min_size = MIN_MARK_2LINE if two_line else 0.08
    return {
        "hologram": any(_has_mark(gray, region, m, k, text, bg, min_size) for k in MARK_LEVELS),
        "blue": _has_blue(color, (x0r, y0r, x1r, mark_bottom), m),
        "ind": _has_ind(gray, ind_region, m, text, bg),
        "font": _standard_font(binary, lines),
    }


def look(
    frame: np.ndarray,
    box: Box,
    two_line: bool = False,
    weights: Mapping[str, float] | None = None,
    rule: HsrpRule = "any",
) -> HsrpLook | None:
    """One frame's verdict for the plate in `box`: "hsrp" when the conditions that passed (with a
    weight above 0) satisfy `rule` (see `passes`), else "non_hsrp"; None when it cannot be judged."""
    return judge(frame, box, two_line, weights, rule)[0]


def judge(
    frame: np.ndarray,
    box: Box,
    two_line: bool = False,
    weights: Mapping[str, float] | None = None,
    rule: HsrpRule = "any",
) -> tuple[HsrpLook | None, str]:
    """`look()` plus why it could not be judged: (look, "") or (None, WHY_* reason)."""
    found = conditions_or_reason(frame, box, two_line)
    if isinstance(found, str):
        log.debug("hsrp look: not judged (%s)", found)
        return None, found
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            "hsrp look score=%.2f %s",
            score(found, weights),
            " ".join(f"{k}={'yes' if v else 'no'}" for k, v in found.items()),
        )
    return ("hsrp" if passes(found, rule, weights) else "non_hsrp"), ""


def passes(
    found: Mapping[str, bool], rule: HsrpRule = "any", weights: Mapping[str, float] | None = None
) -> bool:
    """Whether the conditions in `found` make the frame HSRP under `rule` (config hsrp.rule). A
    condition with weight 0 counts as not passed.
      - "any":   any one condition (OR) - the weak font check alone is enough;
      - "close": hologram AND "IND" - clear, close camera view;
      - "far":   "IND" AND one of hologram / blue / font - far camera, night or IR (tiny or grey marks).
    """
    w = WEIGHTS if weights is None else weights
    ok = {k: bool(found.get(k)) and w.get(k, 0.0) > 0 for k in WEIGHTS}
    if rule == "close":
        return ok["hologram"] and ok["ind"]
    if rule == "far":
        return ok["ind"] and (ok["hologram"] or ok["blue"] or ok["font"])
    if rule == "any":
        return any(ok.values())
    raise ValueError(f"unknown HSRP rule {rule!r} (use 'any', 'close' or 'far')")


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


def why_unsure(
    looks: Sequence[HsrpLook | None],
    reasons: Sequence[str] = (),
    min_marked: int = 2,
    min_clean: int = 3,
) -> str:
    """Why `decide(looks, min_marked, min_clean)` came out "unsure", in plain words. `reasons` are
    the per-frame WHY_* texts of the frames that could not be judged (the most common one is told)."""
    marked = sum(1 for v in looks if v == "hsrp")
    clean = sum(1 for v in looks if v == "non_hsrp")
    if marked and clean:
        return f"images disagree ({marked} vs {clean})"  # HSRP vs non-HSRP images
    if marked:
        return f"mark in {marked}/{min_marked} images"  # too few images showed the mark
    if clean:
        return f"clear in {clean}/{min_clean} images"  # too few images with a clean space
    if not looks:
        return WHY_NOT_CHECKED
    told = [r for r in reasons if r]
    if not told:
        return "no image judged"
    return max(told, key=told.count)  # most common; ties -> the first seen


def label(result: HsrpResult, reason: str, as_hsrp: bool = False, as_non_hsrp: bool = False) -> str:
    """The value saved and sent for a plate: "hsrp" / "non_hsrp", or for a plate that could not be
    judged "not sure: <reason>" (e.g. "not sure: text not found"); with config hsrp.unsure_as_hsrp
    "hsrp (not sure: <reason>)", with hsrp.unsure_as_non_hsrp "non_hsrp (not sure: <reason>)"."""
    if result != "unsure":
        return result
    if as_hsrp:
        return f"hsrp (not sure: {reason})"
    if as_non_hsrp:
        return f"non_hsrp (not sure: {reason})"
    return f"not sure: {reason}"


# ---- trained classifier (config hsrp.method: model) ----------------------------------------------

MODEL_H, MODEL_W = 128, 320  # default model input; a model file with another fixed size uses its own
CROP_MX, CROP_MY = 0.12, 0.30  # margin around the plate box: fraction of its width / height (as trained)
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)
WHY_LOW_CONF = "model unsure"  # the classifier's answer was too close to the threshold
WHY_VIEWS_DISAGREE = "images disagree"  # one image weakly HSRP, another not (no image strongly HSRP)
STRONG_P = 0.9  # a classifier answer this sure is HSRP even when another image of the vehicle says not
# (the font check gives no p, so a font answer never counts as this sure)
MIN_CROP_STD = 8.0  # grey-level spread below which a plate crop shows nothing to judge


def model_crop(frame: np.ndarray, box: Box) -> np.ndarray | None:
    """The plate box plus a margin (CROP_MX of its width left and right, CROP_MY of its height above
    and below) - exactly what the classifier was trained on - or None when the box is outside the frame."""
    h, w = frame.shape[:2]
    bw, bh = box.x2 - box.x1, box.y2 - box.y1
    x1, y1 = max(0, int(box.x1 - CROP_MX * bw)), max(0, int(box.y1 - CROP_MY * bh))
    x2, y2 = min(w, int(box.x2 + CROP_MX * bw)), min(h, int(box.y2 + CROP_MY * bh))
    if x2 - x1 < 8 or y2 - y1 < 4:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    return crop


class HsrpModel:
    """HSRP / non-HSRP from a small trained image classifier. Callable like `judge`:
    (frame, box) -> ("hsrp" | "non_hsrp", "") or (None, reason). Thread-safe (one ONNX session)."""

    def __init__(
        self,
        path: str | Path,
        threshold: float = 0.5,
        margin: float = 0.1,
        threads: int = 1,
        two_line_marks: bool = False,
    ):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads  # one plate at a time; keep the Pi's cores for the reader
        opts.inter_op_num_threads = 1
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self._sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        inp = self._sess.get_inputs()[0]
        self._input = inp.name
        h, w = inp.shape[2:4] if len(inp.shape) == 4 else (None, None)
        self._size = (w, h) if isinstance(h, int) and isinstance(w, int) else (MODEL_W, MODEL_H)
        self.threshold, self.margin = threshold, margin
        # Second opinion on 2-line plates (config hsrp.two_line_marks): the classifier learnt mostly
        # from cars and misses many 2-line truck HSRPs, so when it does not say HSRP and the rule
        # check finds both the hologram and "IND" (rule "close") the plate is HSRP. Measured on 58
        # non-HSRP 2-line plates (trucks + Kaggle): never wrongly HSRP. The font check alone is not
        # used: a neatly printed old plate passes it too.
        self.two_line_marks = two_line_marks
        log.info(
            "hsrp: classifier %s threshold=%.2f margin=%.2f two_line_marks=%s",
            path,
            threshold,
            margin,
            two_line_marks,
        )

    def probability(self, frame: np.ndarray, box: Box) -> float | None:
        """Chance (0..1) the plate in `box` is an HSRP, or None when there is no usable crop."""
        crop = model_crop(frame, box)
        if crop is None:
            return None
        x = cv2.resize(crop, self._size, interpolation=cv2.INTER_AREA)[:, :, ::-1]
        x = ((x.astype(np.float32) / 255.0 - _MEAN) / _STD).transpose(2, 0, 1)[None]
        logits = self._sess.run(None, {self._input: np.ascontiguousarray(x)})[0][0]
        e = np.exp(logits - logits.max())
        return float(e[1] / e.sum())

    def __call__(self, frame: np.ndarray, box: Box) -> tuple[HsrpLook | None, str]:
        look, why, _ = self.judge(frame, box)
        return look, why

    def judge(self, frame: np.ndarray, box: Box) -> tuple[HsrpLook | None, str, float | None]:
        """`__call__` plus the classifier's p (None when it did not run), so a vehicle with several images
        can tell a sure HSRP image (p >= STRONG_P) from a weak one."""
        crop = model_crop(frame, box)
        if crop is None:
            return None, WHY_NO_PLATE, None
        # Nothing to see (black / blank / washed-out picture): the classifier would still give an answer.
        if float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).std()) < MIN_CROP_STD:
            return None, WHY_LOW_CONTRAST, None
        p = self.probability(frame, box)
        if p is None:
            return None, WHY_NO_PLATE, None
        log.debug("hsrp model p=%.3f", p)
        if p >= self.threshold + self.margin:
            return "hsrp", "", p
        if self.two_line_marks and self._two_line_marked(frame, box):
            log.debug("hsrp model p=%.3f overruled: 2-line plate shows hologram + IND", p)
            return "hsrp", "", p
        if abs(p - self.threshold) < self.margin:
            return None, WHY_LOW_CONF, p
        return "non_hsrp", "", p

    @staticmethod
    def _two_line_marked(frame: np.ndarray, box: Box) -> bool:
        found = conditions_or_reason(frame, box, two_line=True, only_two_line=True)
        return not isinstance(found, str) and passes(found, "close")


class CombinedCheck:
    """HSRP from the trained classifier and the character font together (config hsrp.method:
    combined). The model decides whenever it is sure - agreeing with the font or not, since the font
    is weak (a neatly printed non-HSRP plate passes it). Only when the model is unsure (p within
    `margin` of the threshold) is the font asked: when it measures the same answer the model leans
    to, that answer is given (with the model's p, so it never counts as a sure image); when it
    disagrees or cannot measure, the plate stays "not sure". Callable like `judge`."""

    def __init__(self, model, font=None) -> None:
        self.model = model
        self.font = font if font is not None else FontCheck()  # no fallback: the model was asked already

    def __call__(self, frame: np.ndarray, box: Box) -> tuple[HsrpLook | None, str]:
        look, why, _ = self.judge(frame, box)
        return look, why

    def judge(self, frame: np.ndarray, box: Box) -> tuple[HsrpLook | None, str, float | None]:
        look, why, p = self.model.judge(frame, box)
        if look is not None or why != WHY_LOW_CONF or p is None:
            return look, why, p
        lean = "hsrp" if p >= self.model.threshold else "non_hsrp"
        font_look, font_why, _ = self.font.judge(frame, box)
        if font_look == lean:
            log.debug("hsrp model unsure p=%.3f, font agrees: %s", p, lean)
            return lean, "", p
        log.debug("hsrp model unsure p=%.3f leans %s, font says %s", p, lean, font_look or font_why)
        return None, WHY_LOW_CONF, p


def _split_lines(binary: np.ndarray) -> tuple[np.ndarray, tuple[float, list[list[Char]]]] | None:
    """A 2-line plate whose lines touch (small, blurred or tilted picture): a top-line letter joins
    the one below it into one blob about two lines tall. Cut each such blob at its thinnest row
    (where the two letters meet), drop what touches the picture's edge and look for the lines
    again. (cut binary, _lines result) or None."""
    h, w = binary.shape
    cut = binary.copy()
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    joined = 0
    for i in range(1, n):
        x, y, bw, bh, _ = (int(v) for v in stats[i])
        if not (FONT_JOINED_HEIGHT[0] * h <= bh <= FONT_JOINED_HEIGHT[1] * h and bw <= FONT_JOINED_MAX_W * w):
            continue  # not two letters on top of each other (a frame edge, a single letter)
        mask = labels[y : y + bh, x : x + bw] == i
        lo, hi = int(FONT_SPLIT_BAND[0] * bh), int(FONT_SPLIT_BAND[1] * bh)
        row = y + lo + int(np.argmin(mask[lo:hi].sum(axis=1)))
        band = cut[row - 1 : row + 2, x : x + bw]
        band[labels[row - 1 : row + 2, x : x + bw] == i] = 0
        joined += 1
    if not joined:
        return None
    # A picture this poor also has background (wall, bumper) in the box: blobs touching its edge are
    # not characters.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cut, connectivity=8)
    for i in range(1, n):
        x, y, bw, bh, _ = (int(v) for v in stats[i])
        if x == 0 or y == 0 or x + bw == w or y + bh == h:
            cut[labels == i] = 0
    found = _lines(cut)
    if found is None or len(found[1]) != 2:
        return None
    return cut, found


class FontCheck:
    """HSRP from the character font alone (config hsrp.method: font): "hsrp" when the first
    FONT_SAMPLE_CHARS characters of the top line are in one uniform, bold standard font
    (_standard_font), "non_hsrp" when they are not, None (with the reason) when they cannot be
    measured. Callable like `judge`; judge() adds p = None: the font check is weak (a neatly
    printed non-HSRP plate passes it too), so a font answer is never a sure image (p >= STRONG_P).
    In whole-vehicle ingest a font "hsrp" makes the vehicle HSRP only when every image says HSRP,
    or when the model fallback is sure of it on some image (anpr.engine). The plate number is
    still taken from an image the font measured before one where the model fallback answered.
    `fallback` (an HsrpModel): its answer when the font cannot be measured (letters not found,
    picture too small or blurred), instead of "not sure"."""

    def __init__(self, fallback=None) -> None:
        self.fallback = fallback

    def __call__(self, frame: np.ndarray, box: Box) -> tuple[HsrpLook | None, str]:
        look, why, _ = self.judge(frame, box)
        return look, why

    def judge(self, frame: np.ndarray, box: Box) -> tuple[HsrpLook | None, str, float | None]:
        look, why, p = self._font(frame, box)
        if look is None and self.fallback is not None:
            look2, why2, p2 = self.fallback.judge(frame, box)
            said = look2 or f"not sure ({why2})"
            log.debug("hsrp font not measured (%s): model decided %s p=%s", why, said, p2)
            return look2, why2, p2
        return look, why, p

    def _font(self, frame: np.ndarray, box: Box) -> tuple[HsrpLook | None, str, float | None]:
        # Only the characters are measured: no room left of the text is needed (that is where the
        # hologram / "IND" would be), so a camera's tight plate crop can be judged too.
        images = _plate_image(frame, box)
        if images is None:
            return None, WHY_NO_PLATE, None
        gray, _ = images
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)
        found_lines = _lines(binary)
        if found_lines is None:
            split = _split_lines(binary)
            if split is None:
                return None, WHY_NO_TEXT, None
            binary, found_lines = split
        if float(gray.std()) < MIN_CROP_STD:
            return None, WHY_LOW_CONTRAST, None
        # Only the first FONT_SAMPLE_CHARS characters of the top line (usually the state letters, "DL1").
        # Blobs that are not letter-height (the hologram or a bolt next to the "D") are skipped first.
        top = found_lines[1][0]
        h_mid = float(np.median([c[3] for c in top]))
        letters = [c for c in top if abs(c[3] - h_mid) <= FONT_LETTER_HEIGHT_TOL * h_mid]
        sample = sorted(letters, key=lambda c: c[0])[:FONT_SAMPLE_CHARS]
        if len(sample) < FONT_SAMPLE_CHARS:
            return None, WHY_NO_TEXT, None
        # Measurement tolerance in REAL pixels: a letter only ~20 px tall in the camera picture is
        # scaled up to WORK_W here, and 1-2 px of blur or dirt there is 5-10% of its height.
        src_height = h_mid * (box.x2 - box.x1) / WORK_W
        slack = FONT_PIXEL_SLACK / src_height if src_height > 0 else 0.0
        if _standard_font(binary, [sample], min_chars=FONT_SAMPLE_CHARS, slack=slack):
            return "hsrp", "", None
        return "non_hsrp", "", None
