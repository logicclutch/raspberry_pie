"""HSRP check (anpr.hsrp) on drawn plates, plus the per-track decision rule."""

import cv2
import numpy as np
import pytest

from anpr import hsrp
from anpr.types import Box

X1, Y1, X2, Y2 = 400, 300, 900, 410


def plate(marks: bool, two_line: bool = False, blur: float = 0.0, ink=(20, 20, 20), bolts=False):
    """A white 1-line plate "HR51CX6945" (or 2-line "TN 36" / "BC 8199") on a dark car; `marks`
    adds the hologram + "IND", `bolts` two small screw heads at the left edge."""
    f = np.full((720, 1280, 3), 40, np.uint8)
    y2 = Y1 + 150 if two_line else Y2  # a 2-line plate is taller
    cv2.rectangle(f, (X1, Y1), (X2, y2), (235, 235, 235), -1)
    cv2.rectangle(f, (X1 + 4, Y1 + 4), (X2 - 4, y2 - 4), ink, 2)
    if two_line:
        for i, ch in enumerate("TN 36"):  # one character at a time: drawn letters must not touch
            cv2.putText(f, ch, (X1 + 140 + 38 * i, Y1 + 62), cv2.FONT_HERSHEY_SIMPLEX, 1.5, ink, 4)
        for i, ch in enumerate("BC 8199"):
            cv2.putText(f, ch, (X1 + 100 + 38 * i, Y1 + 128), cv2.FONT_HERSHEY_SIMPLEX, 1.5, ink, 4)
    else:
        cv2.putText(f, "HR51CX6945", (X1 + 75, Y1 + 85), cv2.FONT_HERSHEY_SIMPLEX, 2.0, ink, 7)
    if marks:
        hs = 12 if two_line else 18  # hologram ~0.3-0.4 character heights
        cv2.rectangle(f, (X1 + 40, Y1 + 22), (X1 + 40 + hs, Y1 + 22 + hs), (120, 90, 60), -1)  # hologram
        cv2.putText(f, "IND", (X1 + 32, Y1 + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 60, 20), 2)
    if bolts:
        for y in (Y1 + 30, Y1 + 80):
            cv2.circle(f, (X1 + 50, y + 20), 3, (60, 60, 60), -1)  # ~7 px vs 38 px characters
    if blur:
        f = cv2.GaussianBlur(f, (0, 0), blur)
    return f, Box(X1 - 10, Y1 - 8, X2 + 10, y2 + 8, 0.9)


@pytest.mark.parametrize("blur", [0.0, 1.5])
def test_hologram_and_ind_mean_hsrp(blur):
    assert hsrp.look(*plate(True, blur=blur)) == "hsrp"


@pytest.mark.parametrize("blur", [0.0, 1.5])
def test_clean_space_before_the_text_means_not_hsrp(blur):
    assert hsrp.look(*plate(False, blur=blur)) == "non_hsrp"


def test_black_and_white_video_works_too():
    f, box = plate(True)
    gray = cv2.cvtColor(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)  # night / IR camera
    assert hsrp.look(gray, box) == "hsrp"


def test_small_plate_still_judged():
    f, _ = plate(True)
    small = cv2.resize(f, (320, 180), interpolation=cv2.INTER_AREA)  # plate ~125 px wide
    assert hsrp.look(small, Box(97, 73, 227, 104, 0.9)) == "hsrp"


def look2(f, box):
    return hsrp.look(f, box, two_line=True)


@pytest.mark.parametrize("marks", [True, False])
def test_two_line_plates_are_not_judged_unless_switched_on(marks):
    assert hsrp.look(*plate(marks, two_line=True)) is None


@pytest.mark.parametrize("blur", [0.0, 1.5])
def test_two_line_plate_with_hologram_is_hsrp(blur):
    assert look2(*plate(True, two_line=True, blur=blur)) == "hsrp"


@pytest.mark.parametrize("blur", [0.0, 1.5])
def test_two_line_plate_with_clean_edge_is_not_hsrp(blur):
    assert look2(*plate(False, two_line=True, blur=blur)) == "non_hsrp"


def test_two_line_bolts_are_not_a_hologram():
    # screw heads left of the text, where "IND" would be on an HSRP plate
    assert look2(*plate(False, two_line=True, bolts=True)) == "non_hsrp"
    assert look2(*plate(True, two_line=True, bolts=True)) == "hsrp"


def test_two_line_black_and_white_and_tilted():
    f, box = plate(True, two_line=True)
    gray = cv2.cvtColor(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    assert look2(gray, box) == "hsrp"
    cx, cy = (X1 + X2) / 2, Y1 + 75
    for marks in (True, False):  # plate turned 6 degrees, as on a truck seen from the side
        g, _ = plate(marks, two_line=True)
        g = cv2.warpAffine(g, cv2.getRotationMatrix2D((cx, cy), 6, 1.0), (1280, 720))
        want = "hsrp" if marks else "non_hsrp"
        assert look2(g, Box(X1 - 20, Y1 - 35, X2 + 20, Y1 + 185, 0.9)) == want


def test_more_than_two_lines_is_not_judged():
    f, _ = plate(False, two_line=True)
    cv2.rectangle(f, (X1, Y1 + 140), (X2, Y1 + 220), (235, 235, 235), -1)  # taller plate, 3rd line
    for i, ch in enumerate("AB 12"):
        cv2.putText(f, ch, (X1 + 140 + 38 * i, Y1 + 194), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 20, 20), 4)
    assert look2(f, Box(X1 - 10, Y1 - 8, X2 + 10, Y1 + 228, 0.9)) is None


def test_cannot_judge_without_text_or_picture():
    f, box = plate(False)
    blank = f.copy()
    cv2.rectangle(blank, (X1, Y1), (X2, Y2), (235, 235, 235), -1)  # plate with nothing on it
    assert hsrp.look(blank, box) is None
    assert hsrp.look(f, Box(1300, 800, 1400, 850, 0.9)) is None  # box off the picture
    assert hsrp.look(np.zeros((720, 1280, 3), np.uint8), box) is None  # black frame


def test_text_cut_off_at_the_left_is_not_judged():
    f, _ = plate(False)
    # box starts inside the first character: no space left of the text in view
    assert hsrp.look(f, Box(X1 + 85, Y1 - 8, X2 + 10, Y2 + 8, 0.9)) is None


@pytest.mark.parametrize(
    ("looks", "want"),
    [
        (["hsrp", "hsrp"], "hsrp"),
        (["hsrp", "non_hsrp", "hsrp", "non_hsrp", "non_hsrp"], "hsrp"),  # blur hides the mark often
        (["hsrp"], "unsure"),  # one frame is not enough
        (["non_hsrp"] * 3, "non_hsrp"),
        (["non_hsrp"] * 2, "unsure"),
        (["non_hsrp"] * 10 + ["hsrp"], "non_hsrp"),  # one stray mark (dirt) is tolerated
        (["non_hsrp"] * 6 + ["hsrp"], "unsure"),  # but not one in seven
        ([None] * 8, "unsure"),
        ([], "unsure"),
        ([None, "hsrp", None, "hsrp", "non_hsrp"], "hsrp"),  # unjudged frames don't count
    ],
)
def test_decide(looks, want):
    assert hsrp.decide(looks) == want


def test_decide_thresholds_are_configurable():
    assert hsrp.decide(["hsrp"], min_marked=1) == "hsrp"
    assert hsrp.decide(["non_hsrp"] * 3, min_clean=5) == "unsure"
