"""training/evaluate.py: ground truth parsing, scoring, confusion counting and the gate rule (no models)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from training.evaluate import (
    CropScore,
    ModelScore,
    Shown,
    Vehicle,
    VideoScore,
    align,
    confusions,
    find_leaks,
    gate,
    read_crop_csv,
    read_truth,
    score_video,
)

V = Path("/v/gate.mp4")


def veh(plate: str, sure: bool = True, span: tuple[int, int] | None = None) -> Vehicle:
    ff, lf = span if span else (None, None)
    return Vehicle(V, plate, plate, sure, ff, lf)


# ---- confusions ----------------------------------------------------------------------------------


def test_align_substitution_and_drop() -> None:
    assert confusions("HR51CE1421", "HR51CE1421") == Counter()
    assert confusions("GJ12BX3886", "GJ12 8X3886".replace(" ", "")) == Counter({"B>8": 1})
    # dropped last character
    assert confusions("RJ40GA7317", "RJ40GA731") == Counter({"7>-": 1})
    # extra character
    assert confusions("UP16DY5388", "UP16DY53888") == Counter({"->8": 1})
    assert confusions("HR63D8065", "HR63D8O65") == Counter({"0>O": 1})


def test_align_pairs_cover_both_strings() -> None:
    pairs = align("AB12", "A812X")
    assert "".join(t for t, _ in pairs if t != "-") == "AB12"
    assert "".join(r for _, r in pairs if r != "-") == "A812X"


# ---- video scoring -------------------------------------------------------------------------------


def test_score_video_correct_wrong_missed() -> None:
    vehicles = [veh("RJ40GA7317", span=(60, 100)), veh("HR51CX6945", span=(724, 737)), veh("HR14X3149")]
    shown = [Shown("RJ40GA7317", 70, 95), Shown("HR51CX6948", 725, 736)]
    sc = score_video("gate", shown, vehicles)
    assert (sc.sure, sc.correct, sc.wrong, sc.unscored) == (3, 1, 1, 0)
    assert sc.missed == ["HR14X3149", "HR51CX6945"]
    # the wrong read is attributed to the vehicle it overlaps in time
    assert sc.wrong_reads == [("gate", "HR51CX6945", "HR51CX6948")]
    assert sc.confusions == Counter({"5>8": 1})
    assert sc.precision == pytest.approx(0.5)
    assert sc.recall == pytest.approx(1 / 3)


def test_same_plate_twice_is_wrong() -> None:
    sc = score_video("g", [Shown("HR14X3149", 1, 5), Shown("HR14X3149", 300, 310)], [veh("HR14X3149")])
    assert (sc.correct, sc.wrong) == (1, 1)
    assert sc.wrong_reads[0][1].endswith("(again)")
    assert sc.confusions == Counter()


def test_unsure_vehicles_are_not_scored() -> None:
    vehicles = [veh("HR63D8065", sure=False, span=(1131, 1141)), veh("HR14X3149", span=(1218, 1320))]
    shown = [
        Shown("HR63D8055", 1133, 1140),  # overlaps only the unsure vehicle
        Shown("HR63D8065", 5000, 5001),  # exactly the unsure plate, wherever it is
        Shown("HR14X3149", 1220, 1300),
    ]
    sc = score_video("g", shown, vehicles)
    assert (sc.correct, sc.wrong, sc.unscored) == (1, 0, 2)
    assert sc.sure == 1 and sc.missed == []


def test_read_overlapping_sure_and_unsure_is_wrong() -> None:
    vehicles = [veh("HR63D8065", sure=False, span=(100, 120)), veh("HR14X3149", span=(110, 130))]
    sc = score_video("g", [Shown("HR14X3148", 112, 118)], vehicles)
    assert (sc.wrong, sc.unscored) == (1, 0)
    assert sc.confusions == Counter({"9>8": 1})


def test_wrong_read_without_time_goes_to_most_similar_plate() -> None:
    vehicles = [veh("UP16DY0772"), veh("DL2CAZ2022")]
    sc = score_video("phone", [Shown("DL2CAZ2O22", 10, 20)], vehicles)
    assert sc.wrong_reads == [("phone", "DL2CAZ2022", "DL2CAZ2O22")]


def test_merge_adds_up() -> None:
    a = score_video("a", [Shown("AA11A1111", 0, 1)], [veh("AA11A1111")])
    b = score_video("b", [Shown("BB22B2223", 0, 1)], [veh("BB22B2222")])
    total = VideoScore()
    total.merge(a)
    total.merge(b)
    assert (total.sure, total.correct, total.wrong) == (2, 1, 1)
    assert total.confusions == Counter({"2>3": 1})


# ---- crops ---------------------------------------------------------------------------------------


def test_crop_score() -> None:
    sc = CropScore()
    sc.add("a.png", "GJ12BX3886", "GJ12BX3886", "GJ12BX3886")
    sc.add("b.png", "GJ12BX3886", "GJ128X3886", "GJ12BX3886")  # validator repaired it
    sc.add("c.png", "HR47F1710", "HR47F171", None)
    assert sc.crops == 3
    assert sc.raw_acc == pytest.approx(1 / 3)
    assert sc.valid_acc == pytest.approx(2 / 3)
    assert sc.confusions == Counter({"B>8": 1, "0>-": 1})


# ---- gate ----------------------------------------------------------------------------------------


def vs(correct: int, wrong: int, sure: int = 20) -> VideoScore:
    return VideoScore(sure=sure, correct=correct, wrong=wrong)


def cs(n: int, valid: int) -> CropScore:
    return CropScore(crops=n, raw_exact=valid, valid_exact=valid)


def test_gate_passes_when_better_and_not_worse() -> None:
    ok, lines = gate(ModelScore(vs(15, 1), cs(200, 150)), ModelScore(vs(17, 0), cs(200, 150)))
    assert ok, lines
    assert all(line.startswith("OK") for line in lines)


def test_gate_requires_a_crop_test_by_default() -> None:
    ok, lines = gate(ModelScore(vs(15, 1)), ModelScore(vs(17, 0)))
    assert not ok and any(line.startswith("FAIL") and "crop test ran" in line for line in lines)
    assert gate(ModelScore(vs(15, 1)), ModelScore(vs(17, 0)), require_crops=False)[0]


def test_gate_fails_on_a_new_plate_on_an_unsure_vehicle() -> None:
    base = VideoScore(sure=20, correct=15, unscored=1, unscored_reads=[("gate", "UP16GT6024")])
    same = VideoScore(sure=20, correct=16, unscored=1, unscored_reads=[("gate", "UP16GT6024")])
    new = VideoScore(sure=20, correct=16, unscored=1, unscored_reads=[("gate", "UP16QT6024")])
    assert gate(ModelScore(base), ModelScore(same), require_crops=False)[0]
    ok, lines = gate(ModelScore(base), ModelScore(new), require_crops=False)
    assert not ok and any(line.startswith("FAIL") and "UP16QT6024" in line for line in lines)


def test_gate_fails_on_more_wrong_even_with_more_correct() -> None:
    ok, lines = gate(ModelScore(vs(15, 0)), ModelScore(vs(19, 1)))
    assert not ok
    assert any(line.startswith("FAIL") and "wrong" in line for line in lines)


def test_gate_fails_on_fewer_correct() -> None:
    assert not gate(ModelScore(vs(15, 1)), ModelScore(vs(14, 0)), require_crops=False)[0]


def test_gate_fails_when_only_equal() -> None:
    ok, lines = gate(ModelScore(vs(15, 0)), ModelScore(vs(15, 0)), require_crops=False)
    assert not ok
    assert lines[-1].startswith("FAIL") and "strictly better" in lines[-1]


def test_gate_crop_rules() -> None:
    base = ModelScore(vs(15, 0), cs(200, 150))
    # same video result, better crops -> pass
    assert gate(base, ModelScore(vs(15, 0), cs(200, 170)))[0]
    # better video but worse crops -> fail
    assert not gate(base, ModelScore(vs(16, 0), cs(200, 140)))[0]
    # too few crops -> fail even if better
    small_base = ModelScore(vs(15, 0), cs(40, 20))
    ok, lines = gate(small_base, ModelScore(vs(16, 0), cs(40, 30)), min_crops=100)
    assert not ok and any("crop test set size" in line for line in lines)


def test_gate_min_precision_and_missing_video() -> None:
    kw = {"require_crops": False}
    assert not gate(ModelScore(vs(10, 2)), ModelScore(vs(12, 1)), min_precision=0.995, **kw)[0]
    assert gate(ModelScore(vs(10, 2)), ModelScore(vs(12, 1)), min_precision=0.9, **kw)[0]
    ok, lines = gate(ModelScore(), ModelScore(), **kw)
    assert not ok and "video test ran" in lines[0]


def test_gate_fails_on_leak() -> None:
    ok, lines = gate(ModelScore(vs(15, 1)), ModelScore(vs(20, 0)), leaks=["gate.mp4"], require_crops=False)
    assert not ok and lines[0].startswith("FAIL") and "gate.mp4" in lines[0]


def test_find_leaks_by_video_then_by_source(tmp_path: Path) -> None:
    gate_v, phone_v = tmp_path / "gate.mp4", tmp_path / "phone.mp4"
    source = {gate_v: "gate_cam", phone_v: "phone"}
    info = {"train_videos": [str(phone_v)], "val_videos": [], "train_sources": ["gate_cam", "phone"]}
    assert find_leaks([gate_v, phone_v], source, info) == ["phone.mp4"]
    assert find_leaks([gate_v, phone_v], source, {"train_sources": ["gate_cam"]}) == ["gate_cam (gate.mp4)"]
    assert find_leaks([gate_v], source, {}) == []


def test_find_leaks_by_file_name_and_by_plate(tmp_path: Path) -> None:
    # labels.csv keeps the Mac's absolute path; the ground truth resolves to wherever the repo is now
    gate_v = tmp_path / "repo" / "gate.mp4"
    info = {"train_videos": ["/Users/someone/old-place/gate.mp4"], "val_videos": []}
    assert find_leaks([gate_v], {gate_v: "gate_cam"}, info) == ["gate.mp4"]
    info = {"train_videos": ["/x/other.mp4"], "train_plates": ["HR51CX6945", "MH12AB1234"]}
    leaks = find_leaks([gate_v], {gate_v: "gate_cam"}, info, {"HR51CX6945", "RJ40GA7317"})
    assert leaks == ["test plate(s) HR51CX6945"]


# ---- files ---------------------------------------------------------------------------------------


def test_read_truth_and_crops(tmp_path: Path) -> None:
    (tmp_path / "gate_cam.csv").write_text(
        "# comment line\n"
        "video,vehicle,plate,status,first_frame,last_frame,tracks,note\n"
        "clip.mp4,1,rj40 ga 7317,sure,60,100,1;2,\n"
        "clip.mp4,2,UP16GT6024,unsure,,,3,maybe QT\n",
        encoding="utf-8",
    )
    rows = read_truth(tmp_path / "gate_cam.csv")
    assert [(v.plate, v.sure, v.first_frame, v.last_frame) for v in rows] == [
        ("RJ40GA7317", True, 60, 100),
        ("UP16GT6024", False, None, None),
    ]
    assert rows[0].video == (tmp_path / "clip.mp4").resolve()

    (tmp_path / "bad.csv").write_text(
        "video,vehicle,plate,status\nx.mp4,1,AB12C1234,maybe\n", encoding="utf-8"
    )
    with pytest.raises(ValueError):
        read_truth(tmp_path / "bad.csv")

    (tmp_path / "test.csv").write_text("image_path,plate_text\na.png,HR14X3149\nb.png,\n", encoding="utf-8")
    assert read_crop_csv(tmp_path / "test.csv") == [((tmp_path / "a.png").resolve(), "HR14X3149")]


def test_repo_ground_truth_files_parse() -> None:
    root = Path(__file__).resolve().parents[1] / "training" / "ground_truth"
    gate_rows = read_truth(root / "gate_cam.csv")
    phone_rows = read_truth(root / "phone.csv")
    assert len(gate_rows) == 22 and sum(v.sure for v in gate_rows) == 17
    assert len(phone_rows) == 6 and all(v.sure for v in phone_rows)
    assert all(v.first_frame is not None for v in gate_rows)
