"""HSRP check (anpr.hsrp) on drawn plates, plus the per-track decision rule."""

import logging
from pathlib import Path

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


# The drawn plate's font is perfectly even, so the font condition passes on it: tests of the space
# left of the text switch the font condition off.
NO_FONT = {**hsrp.WEIGHTS, "font": 0.0}


@pytest.mark.parametrize("blur", [0.0, 1.5])
def test_clean_space_before_the_text_means_not_hsrp(blur):
    assert hsrp.look(*plate(False, blur=blur), weights=NO_FONT) == "non_hsrp"


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


# ---- weighted conditions (OR) ----


def test_each_condition_is_reported():
    assert hsrp.conditions(*plate(True)) == {"hologram": True, "blue": True, "ind": True, "font": True}
    found = hsrp.conditions(*plate(False))
    assert not found["hologram"] and not found["blue"] and not found["ind"]


def test_score_adds_the_weights_of_the_conditions_that_passed():
    assert hsrp.score({"hologram": True, "blue": False, "ind": True, "font": False}) == pytest.approx(0.55)
    assert hsrp.score({"hologram": True, "blue": True, "ind": True, "font": True}) == pytest.approx(1.0)
    assert hsrp.score({"font": True}, {"font": 0.5}) == pytest.approx(0.5)


@pytest.mark.parametrize("only", ["hologram", "blue", "ind"])
def test_any_one_decisive_condition_is_enough(only):
    f, box = plate(True)
    weights = {k: (hsrp.WEIGHTS[k] if k == only else 0.0) for k in hsrp.WEIGHTS}
    assert hsrp.look(f, box, weights=weights) == "hsrp"


def test_conditions_are_or_so_the_font_alone_is_enough():
    # a neat printed plate without hologram / IND / blue: only the font passes -> HSRP (OR)
    f, box = plate(False)
    assert hsrp.conditions(f, box) == {"hologram": False, "blue": False, "ind": False, "font": True}
    assert hsrp.score(hsrp.conditions(f, box)) == pytest.approx(0.15)
    assert hsrp.look(f, box) == "hsrp"
    assert hsrp.look(f, box, weights=NO_FONT) == "non_hsrp"  # weight 0 = condition off


def test_ind_letters_alone():
    f, box = plate(False)
    cv2.putText(f, "IND", (X1 + 32, Y1 + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 2)  # grey ink
    assert hsrp.conditions(f, box)["ind"]
    assert hsrp.look(f, box, weights=NO_FONT) == "hsrp"


def _ind_down_the_edge(f):
    """ "IND" printed turned 90 degrees, running down the left edge of the 2-line plate."""
    word = np.full((18, 51, 3), 235, np.uint8)
    for i, ch in enumerate("IND"):  # one letter at a time: drawn letters must not touch
        cv2.putText(word, ch, (2 + 17 * i, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 60, 20), 2)
    word = cv2.rotate(word, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 51 high, 18 wide
    f[Y1 + 60 : Y1 + 111, X1 + 40 : X1 + 58] = word


def test_two_line_ind_letters_alone():
    f, box = plate(False, two_line=True)
    _ind_down_the_edge(f)
    found = hsrp.conditions(f, box, two_line=True)
    assert found["ind"] and not found["hologram"]
    assert hsrp.look(f, box, two_line=True, weights=NO_FONT) == "hsrp"


def test_two_line_bolt_heads_are_not_ind():
    f, box = plate(False, two_line=True, bolts=True)
    assert not hsrp.conditions(f, box, two_line=True)["ind"]


def test_blue_needs_a_colour_picture():
    f, box = plate(True)
    gray = cv2.cvtColor(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    assert hsrp.conditions(f, box)["blue"]
    assert not hsrp.conditions(gray, box)["blue"]


def test_weight_zero_switches_every_condition_off():
    f, box = plate(True)
    assert hsrp.look(f, box, weights={k: 0.0 for k in hsrp.WEIGHTS}) == "non_hsrp"


@pytest.mark.parametrize("marks", [True, False])
def test_little_room_before_the_text(marks):
    # e.g. "DL 14CL 6769": the plate edge ~0.5 character heights before the text and the hologram
    # almost touching the first character; the dark car body left of the plate is not "plate"
    f = np.full((720, 1280, 3), 25, np.uint8)
    cv2.rectangle(f, (X1, Y1), (X2, Y2), (200, 200, 200), -1)
    cv2.putText(f, "DL14CL6769", (X1 + 32, Y1 + 85), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (20, 20, 20), 7)
    if marks:
        cv2.circle(f, (X1 + 17, Y1 + 62), 9, (60, 60, 60), 3)  # ring-shaped chromium hologram
    box = Box(X1 - 40, Y1 - 15, X2 + 10, Y2 + 15, 0.9)
    assert hsrp.conditions(f, box)["hologram"] is marks
    assert hsrp.look(f, box, weights=NO_FONT) == ("hsrp" if marks else "non_hsrp")


# ---- rule (config hsrp.rule): which conditions must pass ----

FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"


def _found(*names):
    return {k: k in names for k in ("hologram", "blue", "ind", "font")}


@pytest.mark.parametrize(
    ("names", "any_", "close", "far"),
    [
        ((), False, False, False),
        (("font",), True, False, False),  # neat painted plate: weak font check only
        (("hologram",), True, False, False),  # bolt head / sticker alone
        (("hologram", "blue"), True, False, False),  # same patch twice: still no "IND"
        (("ind",), True, False, False),  # "IND" needs one more
        (("ind", "font"), True, False, True),
        (("ind", "blue"), True, False, True),
        (("hologram", "ind"), True, True, True),
        (("hologram", "blue", "ind", "font"), True, True, True),
    ],
)
def test_rule_decides_which_conditions_must_pass(names, any_, close, far):
    found = _found(*names)
    assert hsrp.passes(found, "any") is any_
    assert hsrp.passes(found, "close") is close
    assert hsrp.passes(found, "far") is far


def test_rule_ignores_switched_off_conditions():
    found = _found("ind", "font")
    assert hsrp.passes(found, "far")
    assert not hsrp.passes(found, "far", NO_FONT)  # weight 0 = condition off


def test_unknown_rule_is_an_error():
    with pytest.raises(ValueError):
        hsrp.passes(_found("hologram", "ind"), "both")  # type: ignore[arg-type]


@pytest.mark.parametrize("rule", ["close", "far"])
def test_rules_on_drawn_plates(rule):
    assert hsrp.look(*plate(True), rule=rule) == "hsrp"  # hologram + IND
    f, box = plate(False)  # neat printed plate: only the font passes
    assert hsrp.look(f, box, rule="any") == "hsrp"
    assert hsrp.look(f, box, rule=rule) == "non_hsrp"


def test_far_rule_accepts_ind_without_a_hologram():
    f, box = plate(False)
    cv2.putText(f, "IND", (X1 + 32, Y1 + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 2)
    assert hsrp.conditions(f, box)["ind"]
    no_hologram = {**hsrp.WEIGHTS, "hologram": 0.0}  # as if the hologram were glared out
    assert hsrp.look(f, box, weights=no_hologram, rule="far") == "hsrp"  # IND + font
    assert hsrp.look(f, box, weights=no_hologram, rule="close") == "non_hsrp"


def test_real_2line_plate_without_marks_is_not_hsrp():
    # TN37CC3205 (yellow 2-line plate, two bolt heads, no hologram / IND / blue): on a slightly
    # loose box only the font check passes - "any" called it HSRP, "close" and "far" do not.
    img = cv2.imread(str(FIXTURES / "hsrp_2line_non_hsrp_TN37CC3205.jpg"))
    h, w = img.shape[:2]
    box = Box(8, 8, w - 8, h - 8, 0.9)
    assert hsrp.conditions(img, box, two_line=True) == _found("font")
    assert hsrp.look(img, box, two_line=True, rule="any") == "hsrp"
    assert hsrp.look(img, box, two_line=True, rule="close") == "non_hsrp"
    assert hsrp.look(img, box, two_line=True, rule="far") == "non_hsrp"


# ---- "not sure" with the reason ----


def test_unjudged_frames_say_why():
    f, box = plate(True)
    assert hsrp.judge(f, box) == ("hsrp", "")
    assert hsrp.judge(f, Box(1300, 800, 1400, 850, 0.9)) == (None, hsrp.WHY_NO_PLATE)
    assert hsrp.judge(np.zeros((720, 1280, 3), np.uint8), box) == (None, hsrp.WHY_NO_TEXT)
    f2, box2 = plate(True, two_line=True)
    assert hsrp.judge(f2, box2) == (None, hsrp.WHY_TWO_LINE_OFF)
    assert hsrp.conditions(f2, box2) is None  # conditions() still gives None


@pytest.mark.parametrize(
    ("looks", "reasons", "want"),
    [
        ([], [], hsrp.WHY_NOT_CHECKED),
        ([None, None], ["", ""], "no image judged"),
        (
            [None, None, None],
            [hsrp.WHY_NO_TEXT, hsrp.WHY_LOW_CONTRAST, hsrp.WHY_LOW_CONTRAST],
            hsrp.WHY_LOW_CONTRAST,  # the most common reason
        ),
        (["hsrp", "non_hsrp", "non_hsrp"], [], "images disagree (1 vs 2)"),
        (["hsrp", None], ["", hsrp.WHY_NO_TEXT], "mark in 1/2 images"),
        (["non_hsrp"], [], "clear in 1/3 images"),
    ],
)
def test_why_unsure(looks, reasons, want):
    assert hsrp.decide(looks) == "unsure"
    assert hsrp.why_unsure(looks, reasons) == want


def test_label_shows_not_sure_with_the_reason():
    assert hsrp.label("hsrp", "") == "hsrp"
    assert hsrp.label("non_hsrp", "") == "non_hsrp"
    why = hsrp.WHY_LOW_CONTRAST
    assert hsrp.label("unsure", why) == "not sure: low contrast"  # the default
    assert hsrp.label("unsure", why, as_hsrp=True) == f"hsrp (not sure: {why})"
    assert hsrp.label("unsure", why, as_hsrp=True, as_non_hsrp=True) == f"hsrp (not sure: {why})"
    assert hsrp.label("unsure", why, as_non_hsrp=True) == f"non_hsrp (not sure: {why})"


def test_reasons_are_short():
    reasons = [getattr(hsrp, k) for k in dir(hsrp) if k.startswith("WHY_")]
    reasons += [hsrp.why_unsure(["hsrp"] * 9 + ["non_hsrp"] * 9), hsrp.why_unsure([None])]
    assert all(len(r) <= 24 for r in reasons), reasons


def test_text_running_into_the_border_is_non_hsrp():
    """An old painted plate whose text starts right at the plate's border: no room for the hologram
    and "IND", so non-HSRP under every rule (it used to be "not sure")."""
    f = np.full((720, 1280, 3), 40, np.uint8)
    cv2.rectangle(f, (X1, Y1), (X2, Y2), (235, 235, 235), -1)
    cv2.rectangle(f, (X1 + 4, Y1 + 4), (X2 - 4, Y2 - 4), (20, 20, 20), 2)  # the border
    cv2.putText(f, "HR51CX6945", (X1 + 10, Y1 + 85), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (20, 20, 20), 7)
    box = Box(X1 - 10, Y1 - 8, X2 + 10, Y2 + 8, 0.9)
    assert hsrp.conditions(f, box) == hsrp.NO_MARKS
    for rule in ("any", "close", "far"):
        assert hsrp.judge(f, box, rule=rule) == ("non_hsrp", "")


# ---- trained classifier (hsrp.method: model) ----

MODEL = __import__("pathlib").Path(__file__).parent.parent / "models" / "hsrp" / "hsrp_cls.onnx"


def test_model_crop_adds_the_trained_margin():
    f = np.zeros((400, 800, 3), np.uint8)
    crop = hsrp.model_crop(f, Box(200, 100, 400, 150, 0.9))  # 200 x 50 plate
    assert crop.shape[:2] == (50 + 2 * 15, 200 + 2 * 24)
    assert hsrp.model_crop(f, Box(900, 500, 950, 520, 0.9)) is None  # outside the picture


class _FakeModel(hsrp.HsrpModel):
    def __init__(self, p, threshold=0.5, margin=0.1, two_line_marks=False):  # no ONNX file: fixed p
        self.p, self.threshold, self.margin, self.two_line_marks = p, threshold, margin, two_line_marks

    def probability(self, frame, box):
        return self.p


@pytest.mark.parametrize(
    ("p", "want"),
    [(0.97, ("hsrp", "")), (0.61, ("hsrp", "")), (0.55, (None, hsrp.WHY_LOW_CONF)),
     (0.45, (None, hsrp.WHY_LOW_CONF)), (0.2, ("non_hsrp", "")), (None, (None, hsrp.WHY_NO_PLATE))],
)
def test_model_verdict_and_not_sure_band(p, want):
    f, box = plate(True)
    assert _FakeModel(p)(f, box) == want


@pytest.mark.parametrize("p", [0.2, 0.45])
def test_two_line_plate_with_hologram_and_ind_overrules_the_model(p):
    # the model misses many 2-line truck HSRPs: hologram + "IND" on a 2-line plate wins
    f, box = plate(True, two_line=True)
    assert _FakeModel(p, two_line_marks=True)(f, box) == ("hsrp", "")
    assert _FakeModel(p)(f, box)[0] != "hsrp"  # off: the model alone decides


def test_two_line_marks_need_a_2_line_plate_with_both_marks():
    for f, box in (plate(True), plate(False, two_line=True)):  # 1-line with marks / 2-line clean
        assert _FakeModel(0.2, two_line_marks=True)(f, box) == ("non_hsrp", "")


needs_model = pytest.mark.skipif(not MODEL.is_file(), reason="models/hsrp/hsrp_cls.onnx not built")


@needs_model
def test_real_model_tn37cc3205_is_non_hsrp():
    # the yellow 2-line truck plate that started this: no hologram, no IND
    img = cv2.imread(str(FIXTURES / "hsrp_2line_non_hsrp_TN37CC3205.jpg"))
    h, w = img.shape[:2]
    for on in (False, True):
        m = hsrp.HsrpModel(MODEL, two_line_marks=on)
        assert m(img, Box(8, 8, w - 8, h - 8, 0.9))[0] == "non_hsrp"


@needs_model
def test_engine_uses_the_model_when_present(tmp_path):
    from anpr.config import AppConfig, HsrpConfig, StorageConfig
    from anpr.engine import Engine

    cfg = AppConfig(storage=StorageConfig(db_path=tmp_path / "a.db", image_dir=tmp_path / "i"),
                    hsrp=HsrpConfig(model_path=MODEL))
    eng = Engine(cfg, detector=None, ocr=None, store=None)
    assert isinstance(eng.hsrp_look, hsrp.HsrpModel)
    cfg = cfg.model_copy(update={"hsrp": HsrpConfig(model_path=tmp_path / "missing.onnx")})
    assert not isinstance(Engine(cfg, detector=None, ocr=None, store=None).hsrp_look, hsrp.HsrpModel)


@needs_model
def test_model_input_size_is_read_from_the_model_file():
    assert hsrp.HsrpModel(MODEL)._size == (hsrp.MODEL_W, hsrp.MODEL_H)  # the shipped model: 320 x 128


def test_font_check_judges_by_the_characters_only():
    f, box = plate(False)  # one uniform drawn font, no hologram or "IND"
    look, why, p = hsrp.FontCheck().judge(f, box)
    assert (look, why) in (("hsrp", ""), ("non_hsrp", "")) and p is None
    assert hsrp.FontCheck()(f, box) == (look, why)
    blank = np.full_like(f, 40)  # nothing to measure: not sure, with the reason
    assert hsrp.FontCheck()(blank, box)[0] is None


def test_font_method_is_a_config_choice():
    from anpr.config import HsrpConfig

    assert HsrpConfig(method="font").method == "font"
    assert HsrpConfig().method == "model"  # the default stays the classifier


def test_font_check_cuts_2_line_plate_whose_lines_touch():
    # A real 256 x 256 camera picture: tilted and blurred, top-line letters join the ones below
    # ("U" + "U8", "4" + "83") - cut apart where they meet, the font can be measured.
    f = cv2.imread(str(Path(__file__).parent / "fixtures" / "hsrp_2line_touching_UP14DU8839.jpg"))
    box = Box(11, 88, 232, 228, 0.9)
    gray, _ = hsrp._plate_image(f, box)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    assert hsrp._lines(binary) is None  # as found: no clear lines
    _, (_, lines) = hsrp._split_lines(binary)
    assert len(lines) == 2
    assert hsrp.FontCheck().judge(f, box) == ("hsrp", "", None)


def test_font_check_falls_back_to_the_model_only_when_the_font_cannot_be_measured():
    class Model:
        calls = 0

        def judge(self, frame, box):
            Model.calls += 1
            return "hsrp", "", 0.86

    f, box = plate(False)
    measured = hsrp.FontCheck().judge(f, box)
    assert hsrp.FontCheck(Model()).judge(f, box) == measured and Model.calls == 0  # the font decides
    blank = np.full_like(f, 40)
    assert hsrp.FontCheck(Model()).judge(blank, box) == ("hsrp", "", 0.86)  # font can't: the model
    assert hsrp.FontCheck().judge(blank, box)[0] is None  # no model: not sure, as before


def test_font_check_logs_when_the_model_decides(caplog):
    # hsrp.method font: when the font cannot be measured the model answers instead. The log must say
    # so, with the font's reason, so it is clear which method decided the plate.
    class Model:
        def judge(self, frame, box):
            return "hsrp", "", 0.86

    caplog.set_level(logging.DEBUG, logger="anpr.hsrp")
    f, box = plate(False)
    blank = np.full_like(f, 40)
    font_why = hsrp.FontCheck().judge(blank, box)[1]
    assert hsrp.FontCheck(Model()).judge(blank, box) == ("hsrp", "", 0.86)  # the answer is unchanged
    said = [r.getMessage() for r in caplog.records if "font not measured" in r.getMessage()]
    assert len(said) == 1 and "model" in said[0] and font_why in said[0]

    caplog.clear()
    hsrp.FontCheck(Model()).judge(f, box)  # the font can measure this plate: the model is not asked
    assert not [r for r in caplog.records if "font not measured" in r.getMessage()]


def test_a_font_hsrp_image_does_not_outvote_another_image(tmp_path):
    # The font check is weak (a neatly printed non-HSRP plate passes it), so in whole-vehicle ingest
    # one font pass must not count as a SURE HSRP image that overrules the other images of the vehicle.
    from anpr.config import AppConfig, HsrpConfig, StorageConfig
    from anpr.engine import Engine

    img = cv2.imread(str(Path(__file__).parent / "fixtures" / "hsrp_2line_non_hsrp_TN37CC3205.jpg"))
    h, w = img.shape[:2]
    box = Box(8, 8, w - 8, h - 8, 0.9)

    class SureNotModel:  # the fallback, sure the plate is NOT an HSRP
        def judge(self, frame, box):
            return "non_hsrp", "", 0.02

    font = hsrp.FontCheck(SureNotModel())
    assert font.judge(img, box) == ("hsrp", "", None)  # a non-HSRP plate passes the font: no confidence
    cfg = AppConfig(
        storage=StorageConfig(db_path=tmp_path / "a.db", image_dir=tmp_path / "i"),
        hsrp=HsrpConfig(method="font", model_path=tmp_path / "missing.onnx"),
    )
    eng = Engine(cfg, detector=None, ocr=None, store=None, hsrp_look=font)
    eng.hsrp_any_look = True  # whole-vehicle ingest
    eng._hsrp_observe(1, img, box)  # neat non-HSRP plate: the font passes
    eng._hsrp_observe(1, np.full_like(img, 40), box)  # font cannot measure: the model says not HSRP
    assert eng._hsrp_result(1) == "not sure: images disagree"
