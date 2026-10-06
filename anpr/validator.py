"""Strict Indian number-plate validator.

Accepts exactly two civilian formats:
  * Standard: SS NN X[X][X] NNNN   e.g. MH12AB1234, DL3CAB1234 (Delhi uses a 1-digit district)
  * BH series: YY BH NNNN X[X]     e.g. 22BH1234AB

A read is rejected (None) if it does not fit a format, the state code is not real, any character
confidence is below the threshold, it needs too many character repairs, or two different readings
fit equally well. "Not sure" always means "show nothing".
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass

from anpr.config import ValidationConfig
from anpr.types import OcrResult, PlateKind, ValidPlate

# Real state / union-territory codes, including legacy codes still seen on older vehicles.
STATE_CODES: frozenset[str] = frozenset(
    {
        "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ", "HP", "HR",
        "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "OR",
        "PB", "PY", "RJ", "SK", "TG", "TN", "TR", "TS", "UA", "UK", "UP", "WB",
    }
)  # fmt: skip

# States whose plates print a 1-digit district (DL 3C AB 1234). Everyone else prints 2 digits.
SINGLE_DIGIT_DISTRICT_STATES: frozenset[str] = frozenset({"DL"})

# Position-aware repairs: only applied where the format forces a digit (or a letter).
TO_DIGIT: dict[str, str] = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2",
                            "S": "5", "G": "6", "B": "8"}  # fmt: skip
TO_LETTER: dict[str, tuple[str, ...]] = {"0": ("O", "D"), "1": ("I",), "2": ("Z",),
                                         "5": ("S",), "6": ("G",), "8": ("B",)}  # fmt: skip

_ALNUM = re.compile(r"[A-Z0-9]")
_MIN_LEN, _MAX_LEN = 8, 11  # BH is 9-10, standard is 8-11


@dataclass(frozen=True, slots=True)
class _Parse:
    text: str
    kind: PlateKind
    repaired: tuple[int, ...]  # indices of characters that were changed


def clean(text: str, confs: tuple[float, ...]) -> tuple[str, tuple[float, ...]]:
    """Uppercase, keep only A-Z/0-9 (dropping the matching confidences), strip an 'IND' mark."""
    out_t: list[str] = []
    out_c: list[float] = []
    for ch, c in zip(text.upper(), confs, strict=True):
        if _ALNUM.fullmatch(ch):
            out_t.append(ch)
            out_c.append(c)
    s = "".join(out_t)
    if s.startswith("IND") and len(s) - 3 >= _MIN_LEN:  # no state code starts with "IN"
        return s[3:], tuple(out_c[3:])
    return s, tuple(out_c)


def _as_digit(ch: str) -> str | None:
    if ch.isdigit():
        return ch
    return TO_DIGIT.get(ch)


def _as_letters(ch: str) -> tuple[str, ...]:
    if ch.isalpha():
        return (ch,)
    return TO_LETTER.get(ch, ())


def _coerce_digits(s: str, start: int, repaired: list[int]) -> str | None:
    out = []
    for i, ch in enumerate(s):
        d = _as_digit(ch)
        if d is None:
            return None
        if d != ch:
            repaired.append(start + i)
        out.append(d)
    return "".join(out)


def _coerce_letters(s: str, start: int, repaired: list[int]) -> str | None:
    """Letters only. A digit maps to a letter only when the mapping is unambiguous."""
    out = []
    for i, ch in enumerate(s):
        options = _as_letters(ch)
        if len(options) != 1:
            return None
        if options[0] != ch:
            repaired.append(start + i)
        out.append(options[0])
    return "".join(out)


def _coerce_state(s: str, repaired: list[int]) -> str | None:
    """Two letters that must form a real state code. Tries every repair; must be unique."""
    a_opts, b_opts = _as_letters(s[0]), _as_letters(s[1])
    found = {a + b for a in a_opts for b in b_opts if a + b in STATE_CODES}
    if len(found) != 1:
        return None
    code = found.pop()
    repaired.extend(i for i in (0, 1) if code[i] != s[i])
    return code


def _boundary_safe(raw: str, repaired: list[int], slots: list[tuple[int, int, str]]) -> bool:
    """Reject repairs that could just be a slot boundary in the wrong place.

    If a repaired character sits at the edge of its slot and already has the class the
    neighbouring slot wants (e.g. 'B' right after the series letters in "MH12AB123"), the read
    may simply be missing or gaining a character. Turning it into '8' would invent a different
    plate, so we treat it as "not sure".
    """
    for i in repaired:
        for k, (start, end, _cls) in enumerate(slots):
            if not start <= i < end:
                continue
            cls = "L" if raw[i].isalpha() else "D"
            if i == start and k > 0 and slots[k - 1][2] == cls:
                return False
            if i == end - 1 and k + 1 < len(slots) and slots[k + 1][2] == cls:
                return False
    return True


def _parse_standard(s: str) -> list[_Parse]:
    results: list[_Parse] = []
    n = len(s)
    for district_len in (2, 1):
        series_len = n - 2 - district_len - 4
        if not 1 <= series_len <= 3:
            continue
        repaired: list[int] = []
        state = _coerce_state(s[:2], repaired)
        if state is None:
            continue
        if district_len == 1 and state not in SINGLE_DIGIT_DISTRICT_STATES:
            continue
        district = _coerce_digits(s[2 : 2 + district_len], 2, repaired)
        if district is None or int(district) == 0:
            continue
        ss = 2 + district_len
        series = _coerce_letters(s[ss : ss + series_len], ss, repaired)
        if series is None:
            continue
        number = _coerce_digits(s[n - 4 :], n - 4, repaired)
        if number is None or number == "0000":
            continue
        slots = [(0, 2, "L"), (2, ss, "D"), (ss, ss + series_len, "L"), (n - 4, n, "D")]
        if not _boundary_safe(s, repaired, slots):
            continue
        results.append(_Parse(state + district + series + number, "standard", tuple(repaired)))
    return results


def _parse_bh(s: str, current_year: int) -> list[_Parse]:
    n = len(s)
    if n not in (9, 10):
        return []
    repaired: list[int] = []
    year = _coerce_digits(s[:2], 0, repaired)
    if year is None or not 21 <= int(year) <= (current_year % 100) + 1:
        return []
    bh = _coerce_letters(s[2:4], 2, repaired)
    if bh != "BH":
        return []
    number = _coerce_digits(s[4:8], 4, repaired)
    if number is None or number == "0000":
        return []
    suffix = _coerce_letters(s[8:], 8, repaired)
    if suffix is None or any(ch in "IO" for ch in suffix):  # BH series never uses I or O
        return []
    if not _boundary_safe(s, repaired, [(0, 2, "D"), (2, 4, "L"), (4, 8, "D"), (8, n, "L")]):
        return []
    return [_Parse(year + "BH" + number + suffix, "bh", tuple(repaired))]


class PlateValidator:
    def __init__(self, cfg: ValidationConfig, current_year: int | None = None) -> None:
        self._cfg = cfg
        self._year = current_year or _dt.date.today().year

    def validate(self, ocr: OcrResult) -> ValidPlate | None:
        if len(ocr.text) != len(ocr.char_confs):
            return None
        text, confs = clean(ocr.text, ocr.char_confs)
        if not _MIN_LEN <= len(text) <= _MAX_LEN:
            return None
        if min(confs) < self._cfg.min_char_conf:
            return None

        parses = _parse_standard(text)
        if self._cfg.allow_bh:
            parses += _parse_bh(text, self._year)
        parses = [p for p in parses if len(p.repaired) <= self._cfg.max_repairs]
        if not parses:
            return None

        fewest = min(len(p.repaired) for p in parses)
        best = {p.text: p for p in parses if len(p.repaired) == fewest}
        if len(best) != 1:  # two different plates fit equally well -> not sure
            return None
        p = next(iter(best.values()))

        adjusted = [c * self._cfg.repair_penalty if i in p.repaired else c for i, c in enumerate(confs)]
        return ValidPlate(text=p.text, kind=p.kind, confidence=sum(adjusted) / len(adjusted))


def format_display(text: str, kind: PlateKind) -> str:
    """Human spacing for the dashboard: 'MH 12 AB 1234', 'DL 3C AB 1234', '22 BH 1234 AB'.

    Delhi plates carry a vehicle-class letter right after the district number (C car, S two-wheeler,
    E electric...), written together with it: DL 2C AZ 2022.
    """
    if kind == "bh":
        return f"{text[:2]} BH {text[4:8]} {text[8:]}"
    m = re.fullmatch(r"([A-Z]{2})(\d{1,2})([A-Z]{1,3})(\d{4})", text)
    if not m:
        return text
    state, district, letters, number = m.groups()
    if state == "DL" and len(letters) >= 2:
        return f"DL {district}{letters[0]} {letters[1:]} {number}"
    return f"{state} {district} {letters} {number}"
