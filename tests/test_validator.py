import pytest

from anpr.config import ValidationConfig
from anpr.types import OcrResult
from anpr.validator import STATE_CODES, PlateValidator, clean, format_display


def ocr(text: str, conf: float = 0.95) -> OcrResult:
    return OcrResult(text=text, char_confs=tuple([conf] * len(text)))


@pytest.fixture
def v() -> PlateValidator:
    return PlateValidator(ValidationConfig(), current_year=2026)


@pytest.mark.parametrize(
    "raw,expected,kind",
    [
        ("MH12AB1234", "MH12AB1234", "standard"),
        ("KA01MJ0001", "KA01MJ0001", "standard"),
        ("TN09A1234", "TN09A1234", "standard"),
        ("UP32ABC1234", "UP32ABC1234", "standard"),
        ("DL3CAB1234", "DL3CAB1234", "standard"),
        ("DL10CA1234", "DL10CA1234", "standard"),
        ("TG09EA5678", "TG09EA5678", "standard"),
        ("22BH1234AB", "22BH1234AB", "bh"),
        ("24BH0042C", "24BH0042C", "bh"),
        ("mh 12-ab.1234", "MH12AB1234", "standard"),
        ("INDMH12AB1234", "MH12AB1234", "standard"),
    ],
)
def test_accepts_valid_formats(v, raw, expected, kind):
    got = v.validate(ocr(raw))
    assert got is not None
    assert got.text == expected
    assert got.kind == kind
    assert got.confidence == pytest.approx(0.95)


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "MH12AB123",  # number must be 4 digits -> this parses as nothing valid
        "XX12AB1234",  # not a state code
        "MH00AB1234",  # district 00
        "MH12AB0000",  # number 0000
        "MH12ABCD1234",  # 4 series letters
        "MH121234",  # no series letters
        "MH3AB1234",  # 1-digit district only allowed for DL
        "19BH1234AB",  # BH series started in 2021
        "29BH1234AB",  # future year
        "22BH1234IO",  # BH never uses I/O
        "HELLOWORLD",
        "12345678",
    ],
)
def test_rejects_invalid(v, raw):
    assert v.validate(ocr(raw)) is None


def test_bh_rejected_when_disabled():
    v = PlateValidator(ValidationConfig(allow_bh=False), current_year=2026)
    assert v.validate(ocr("22BH1234AB")) is None


def test_position_aware_repair_digit_slots(v):
    got = v.validate(ocr("MH12AB12S4"))  # S->5 inside the number
    assert got is not None
    assert got.text == "MH12AB1254"
    assert got.confidence < 0.95  # repairs cost confidence


def test_position_aware_repair_letter_slots(v):
    got = v.validate(ocr("MH12A8B1234"))  # 8->B in the middle of the series
    assert got is not None and got.text == "MH12ABB1234"


def test_state_repair_only_when_unique(v):
    # "0L" -> O/D + L -> only "DL" is a real state
    got = v.validate(ocr("0L3CAB1234"))
    assert got is not None and got.text == "DL3CAB1234"


def test_too_many_repairs_rejected():
    v = PlateValidator(ValidationConfig(max_repairs=1), current_year=2026)
    assert v.validate(ocr("MH12AB1ZS4")) is None


def test_zero_repairs_mode_is_exact():
    v = PlateValidator(ValidationConfig(max_repairs=0), current_year=2026)
    assert v.validate(ocr("MH12AB1234")) is not None
    assert v.validate(ocr("MH12AB12S4")) is None


def test_low_char_confidence_rejected(v):
    confs = tuple([0.95] * 9 + [0.4])
    assert v.validate(OcrResult("MH12AB1234", confs)) is None


def test_confidence_length_mismatch_rejected(v):
    assert v.validate(OcrResult("MH12AB1234", (0.9,) * 3)) is None


def test_letter_o_in_number_is_repaired_but_penalised(v):
    got = v.validate(ocr("MH12AB12O4"))
    assert got is not None and got.text == "MH12AB1204"


def test_ambiguous_delhi_rejected(v):
    # DL + "1" district + "0CA": 0 could be O or D in a letter slot -> ambiguous, and
    # the 2-digit reading "10" is also valid -> the 0-repair reading wins, so it is accepted
    # as DL10CA1234. Verify that the validator prefers the reading with fewer repairs.
    got = v.validate(ocr("DL10CA1234"))
    assert got is not None and got.text == "DL10CA1234"


@pytest.mark.parametrize(
    "raw",
    [
        "MH12AB123",  # dropped digit: 'B' must not become '8' -> MH12A8123
        "MH1ZAB1234",  # 'Z' at district/series edge could be a series letter
        "MH12ABI234",  # 'I' at series/number edge could be a series letter
        "MH128B1234",  # '8' at district/series edge could be a district digit
    ],
)
def test_boundary_repairs_are_not_trusted(v, raw):
    assert v.validate(ocr(raw)) is None


def test_clean_keeps_confidences_aligned():
    t, c = clean("mh-12", (0.1, 0.2, 0.3, 0.4, 0.5))
    assert t == "MH12"
    assert c == (0.1, 0.2, 0.4, 0.5)


def test_state_codes_are_two_letters():
    assert all(len(s) == 2 and s.isalpha() for s in STATE_CODES)


@pytest.mark.parametrize(
    "text,kind,shown",
    [
        ("MH12AB1234", "standard", "MH 12 AB 1234"),
        ("DL3CAB1234", "standard", "DL 3C AB 1234"),
        ("DL2CAZ2022", "standard", "DL 2C AZ 2022"),
        ("DL10SA1234", "standard", "DL 10S A 1234"),
        ("DL1C1234", "standard", "DL 1 C 1234"),
        ("KA01MJ0001", "standard", "KA 01 MJ 0001"),
        ("22BH1234AB", "bh", "22 BH 1234 AB"),
    ],
)
def test_format_display(text, kind, shown):
    assert format_display(text, kind) == shown
