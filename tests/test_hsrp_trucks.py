"""The HSRP check on 69 real Indian truck plates marked HSRP / non-HSRP by eye
(tests/fixtures/truck_plates, see its README). The numbers below are today's results with the shipped
settings (rule "close", two_line on): a change to the check must not make them worse. Run this file
directly to see the full table for every rule:  PYTHONPATH=. python tests/test_hsrp_trucks.py
"""

import json
from collections import Counter
from pathlib import Path

import cv2
import pytest

from anpr import hsrp
from anpr.types import Box

DATA = Path(__file__).parent / "fixtures" / "truck_plates"
PLATES = json.loads((DATA / "labels.json").read_text())


def _judge(rule: str) -> list[tuple[dict, str | None, str]]:
    """(plate, verdict or None for not sure, reason) for every marked plate."""
    out = []
    for p in PLATES:
        img = cv2.imread(str(DATA / p["file"]))
        assert img is not None, p["file"]
        look, why = hsrp.judge(img, Box(*p["box"], 1.0), two_line=True, rule=rule)
        out.append((p, look, why))
    return out


def _table(rule: str) -> Counter:
    return Counter((p["label"], look or "not sure") for p, look, _ in _judge(rule))


@pytest.fixture(scope="module")
def close():
    return _table("close")


def test_the_marked_set_is_complete():
    assert Counter(p["label"] for p in PLATES) == {"non_hsrp": 56, "hsrp": 13}


def test_close_rule_never_calls_a_non_hsrp_truck_plate_hsrp(close):
    assert close[("non_hsrp", "hsrp")] == 0


def test_close_rule_finds_at_least_todays_plates(close):
    # Today: 3 of 13 HSRP plates found (the small hologram / "IND" on truck plates is often missed),
    # 36 of 56 non-HSRP plates found, 20 not sure. Raise these when the check gets better.
    assert close[("hsrp", "hsrp")] >= 3
    assert close[("non_hsrp", "non_hsrp")] >= 36
    assert close[("non_hsrp", "not sure")] <= 20


def test_no_space_left_of_the_number_only_hits_non_hsrp_plates(monkeypatch):
    """Text running into the plate's border -> non-HSRP. On these trucks that catches 14 old painted
    plates and no HSRP plate."""
    seen = {}
    rows, left_edge = hsrp._rows, hsrp._left_edge

    def spy_rows(m, lines):
        seen["m"] = m
        return rows(m, lines)

    def spy_left_edge(gray, x0, *a):
        left = left_edge(gray, x0, *a)
        seen["no_space"] = x0 - left < 0.3 * seen["m"]
        return left

    monkeypatch.setattr(hsrp, "_rows", spy_rows)
    monkeypatch.setattr(hsrp, "_left_edge", spy_left_edge)
    hit = Counter()
    for p in PLATES:
        seen.clear()
        found = hsrp.conditions(cv2.imread(str(DATA / p["file"])), Box(*p["box"], 1.0), two_line=True)
        if seen.get("no_space"):
            assert found == hsrp.NO_MARKS
            hit[p["label"]] += 1
    assert hit == {"non_hsrp": 14}


if __name__ == "__main__":
    for rule in ("close", "far", "any"):
        rows = _judge(rule)
        t = Counter((p["label"], look or "not sure") for p, look, _ in rows)
        print(f"rule {rule}:")
        for label in ("hsrp", "non_hsrp"):
            n = sum(v for (lab, _), v in t.items() if lab == label)
            cells = ", ".join(f"{v} {k[1]}" for k, v in sorted(t.items()) if k[0] == label)
            print(f"  {n:2} {label:8} plates -> {cells}")
        wrong = [p["file"] for p, look, _ in rows if look and look != p["label"]]
        print("  wrong:", " ".join(wrong) or "none")


MODEL = Path(__file__).parent.parent / "models" / "hsrp" / "hsrp_cls.onnx"


@pytest.mark.skipif(not MODEL.is_file(), reason="models/hsrp/hsrp_cls.onnx not built")
def test_trained_model_on_trucks():
    """The trained classifier (hsrp.method: model) on these trucks, which it never saw in training.
    Today: every non-HSRP truck plate right (56/56); 5 of 13 HSRP truck plates found - the small, dirty
    hologram on truck plates is its weak spot (the training photos are mostly cars)."""
    m = hsrp.HsrpModel(MODEL)
    t = Counter()
    for p in PLATES:
        look, _ = m(cv2.imread(str(DATA / p["file"])), Box(*p["box"], 1.0))
        t[(p["label"], look or "not sure")] += 1
    assert t[("non_hsrp", "hsrp")] == 0  # a non-HSRP truck plate is never called HSRP
    assert t[("non_hsrp", "non_hsrp")] >= 54
    assert t[("hsrp", "hsrp")] >= 5


@pytest.mark.skipif(not MODEL.is_file(), reason="models/hsrp/hsrp_cls.onnx not built")
def test_trained_model_with_two_line_marks_on_trucks():
    """The shipped setting (hsrp.two_line_marks on): a 2-line plate the model does not call HSRP is HSRP
    when the hologram and "IND" are both found. Today: 6 of 13 HSRP plates found (one more than the
    model alone) and still no non-HSRP plate called HSRP."""
    m = hsrp.HsrpModel(MODEL, two_line_marks=True)
    t = Counter()
    for p in PLATES:
        look, _ = m(cv2.imread(str(DATA / p["file"])), Box(*p["box"], 1.0))
        t[(p["label"], look or "not sure")] += 1
    assert t[("non_hsrp", "hsrp")] == 0
    assert t[("non_hsrp", "non_hsrp")] >= 54
    assert t[("hsrp", "hsrp")] >= 6
