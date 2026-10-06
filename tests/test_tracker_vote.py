import numpy as np
import pytest

from anpr.config import TrackingConfig, VoteConfig
from anpr.tracker import IouTracker, iou
from anpr.types import Box, ValidPlate
from anpr.vote import PlateVoter, decide


def box(x: int, y: int, w: int = 100, h: int = 30) -> Box:
    return Box(x, y, x + w, y + h, 0.9)


def vp(text: str, conf: float = 0.95) -> ValidPlate:
    return ValidPlate(text=text, kind="standard", confidence=conf)


# ---- tracker ----------------------------------------------------------------------------------


def test_iou_basics():
    assert iou(box(0, 0), box(0, 0)) == pytest.approx(1.0)
    assert iou(box(0, 0), box(500, 500)) == 0.0
    assert 0 < iou(box(0, 0), box(50, 0)) < 1


def test_overlapping_box_keeps_track_id():
    t = IouTracker(TrackingConfig())
    (a,), _ = t.update([box(100, 100)], 0.0)
    (b,), _ = t.update([box(110, 102)], 0.3)
    assert a[0] == b[0]


def test_fast_moving_plate_matched_by_centre_distance():
    t = IouTracker(TrackingConfig())
    (a,), _ = t.update([box(100, 100)], 0.0)
    (b,), _ = t.update([box(250, 110)], 0.3)  # no overlap, but within 2 widths
    assert a[0] == b[0]


def test_far_or_different_size_box_is_new_track():
    t = IouTracker(TrackingConfig())
    (a,), _ = t.update([box(100, 100)], 0.0)
    (b,), _ = t.update([box(800, 500)], 0.3)
    (c,), _ = t.update([box(100, 100, w=400, h=120)], 0.6)
    assert len({a[0], b[0], c[0]}) == 3


def test_two_plates_keep_separate_ids():
    t = IouTracker(TrackingConfig())
    out1, _ = t.update([box(100, 100), box(600, 100)], 0.0)
    out2, _ = t.update([box(605, 101), box(104, 99)], 0.3)
    ids1 = {b.x1 // 100: tid for tid, b in out1}
    ids2 = {b.x1 // 100: tid for tid, b in out2}
    assert ids1[1] == ids2[1] and ids1[6] == ids2[6]


def test_tracks_expire_after_max_age():
    t = IouTracker(TrackingConfig(max_age_s=1.0))
    t.update([box(100, 100)], 0.0)
    _, expired = t.update([], 0.5)
    assert not expired and t.has_active()
    _, expired = t.update([], 1.6)
    assert len(expired) == 1 and not t.has_active()


def test_centre_match_does_not_jump_against_the_tracks_motion():
    """NL01AC0056 case (576p gate video): the plate moves right ~15 px a frame, the detector misses
    it for a frame and boxes the "GOODS" painted below-left of it: a new track, not the plate's."""
    t = IouTracker(TrackingConfig())
    plate = [Box(243, 201, 310, 272, 0.9), Box(257, 201, 321, 282, 0.9), Box(270, 203, 336, 284, 0.9)]
    ids = {t.update([b], 0.2 * i)[0][0][0] for i, b in enumerate(plate)}
    assert len(ids) == 1
    (goods,), _ = t.update([Box(217, 289, 310, 379, 0.9)], 0.6)
    assert goods[0] not in ids


def test_fast_plate_on_its_path_keeps_its_track():
    """A jump of over a box width is still the same plate when the track's motion predicts it."""
    t = IouTracker(TrackingConfig())
    ids = {t.update([box(100 + 120 * i, 100)], 0.2 * i)[0][0][0] for i in range(4)}
    assert len(ids) == 1


# ---- vote -------------------------------------------------------------------------------------


CFG = VoteConfig(min_votes=3, min_avg_conf=0.85, min_agreement=0.7)


def test_decide_needs_min_votes():
    assert decide([vp("MH12AB1234")] * 2, CFG) is None
    d = decide([vp("MH12AB1234")] * 3, CFG)
    assert d is not None and d.votes == 3 and d.agreement == 1.0


def test_decide_rejects_disagreement_and_ties():
    assert decide([vp("MH12AB1234")] * 3 + [vp("MH12AB1284")] * 3, CFG) is None
    assert decide([vp("MH12AB1234")] * 3 + [vp("MH12AB1284")] * 2, CFG) is None  # 60% < 70%
    assert decide([vp("MH12AB1234")] * 3 + [vp("MH12AB1284")], CFG) is not None  # 75%


def test_decide_rejects_low_average_confidence():
    assert decide([vp("MH12AB1234", 0.8)] * 5, CFG) is None


def test_voter_reports_once_and_keeps_best_evidence():
    v = PlateVoter(CFG)
    crops = [np.full((2, 2, 3), i, np.uint8) for i in range(4)]
    assert v.add(1, vp("MH12AB1234", 0.90), crops[0]) is None
    assert v.add(1, vp("MH12AB1234", 0.97), crops[1]) is None
    d = v.add(1, vp("MH12AB1234", 0.92), crops[2])
    assert d is not None and d.plate == "MH12AB1234"
    assert v.is_reported(1)
    assert v.add(1, vp("MH12AB1234"), crops[3]) is None  # never twice per track
    crop, _snap = v.evidence(1, "MH12AB1234")
    assert crop is crops[1]


def test_voter_tracks_are_independent_and_droppable():
    v = PlateVoter(CFG)
    for _ in range(2):
        v.add(1, vp("MH12AB1234"), np.zeros((1, 1, 3), np.uint8))
        v.add(2, vp("KA01MJ0001"), np.zeros((1, 1, 3), np.uint8))
    assert len(v.drop(1)) == 2
    assert v.drop(1) == []
    assert v.add(2, vp("KA01MJ0001"), np.zeros((1, 1, 3), np.uint8)) is not None


def test_voter_caps_history():
    v = PlateVoter(VoteConfig(min_votes=1000))
    for _ in range(PlateVoter.MAX_READS + 10):
        v.add(1, vp("MH12AB1234"), np.zeros((1, 1, 3), np.uint8))
    assert len(v.drop(1)) == PlateVoter.MAX_READS


# ---- end-of-track rule ------------------------------------------------------------------------
END = VoteConfig(min_votes=3, min_avg_conf=0.85, min_agreement=0.7, end_min_votes=2, end_min_conf=0.95)


def test_finish_accepts_two_identical_confident_reads():
    v = PlateVoter(END)
    assert v.add(1, vp("HR51CX6945", 0.97), np.zeros((4, 4, 3), np.uint8)) is None
    assert v.add(1, vp("HR51CX6945", 0.99), np.zeros((4, 4, 3), np.uint8)) is None  # only 2 votes
    d = v.finish(1)
    assert d is not None and d.plate == "HR51CX6945" and d.votes == 2 and d.agreement == 1.0
    assert v.is_reported(1) and v.finish(1) is None  # never twice


@pytest.mark.parametrize(
    "reads",
    [
        [("HR51CX6945", 0.99)],  # one read is never enough
        [("HR51CX6945", 0.99), ("HR51CX6946", 0.99)],  # 1 against 1 -> nothing
        [("HR51CX6945", 0.99)] * 2 + [("HR51CX6946", 0.99)] * 2,  # 2 against 2
        [("HR51CX6945", 0.99)] * 2 + [("HR51CX6936", 0.99)],  # rival 2 characters away
        [("HR51CX6945", 0.99)] * 2 + [("HR51CX6946", 0.99), ("HR51CX6948", 0.99)],  # 50%
        [("KA03M1035", 0.99)] * 3 + [("KA03M1005", 0.99)] * 2,  # 60%: the KA03MI0352 misreads
        [("HR51CX6945", 0.96), ("HR51CX6945", 0.90)],  # mean confidence below 0.95
        [("HR51CX6945", 0.96), ("HR51CX6945", 0.90), ("HR51CX6946", 0.99)],  # leader below 0.95
    ],
)
def test_finish_rejects_weak_or_disagreeing_tracks(reads):
    v = PlateVoter(END)
    for text, conf in reads:
        v.add(7, vp(text, conf), np.zeros((4, 4, 3), np.uint8))
    assert v.finish(7) is None


def test_finish_settles_one_disputed_character_by_majority():
    """RJ14UN8156 (gate video): 3 close-up reads, N, N and W at the same position -> N."""
    v = PlateVoter(END)
    for text, conf in [("RJ14UW8156", 0.944), ("RJ14UN8156", 0.938), ("RJ14UN8156", 0.986)]:
        v.add(1, vp(text, conf), np.zeros((4, 4, 3), np.uint8))
    d = v.finish(1)
    assert d is not None and d.plate == "RJ14UN8156" and d.votes == 2
    assert d.agreement == pytest.approx(2 / 3) and d.confidence == pytest.approx(0.962)


FAR_END = VoteConfig(
    min_votes=3, min_avg_conf=0.85, min_agreement=0.7, close_ratio=0.85, end_min_votes=2, end_min_conf=0.95
)


def _approach(v: PlateVoter, reads: dict[int, tuple[str, float]], widths: list[int]) -> None:
    """Sight the plate at each width; frames listed in `reads` also give a valid read."""
    crop = np.zeros((4, 4, 3), np.uint8)
    for i, w in enumerate(widths):
        v.observe(1, w, i * 0.2)
        if i in reads:
            text, conf = reads[i]
            v.add(1, vp(text, conf), crop, width=w)


# DL12CS4288 (gate video): plate widths while it passed; valid reads only at 75, 80 and 98 px (the
# rest dropped the last digit). Floor 0.85 * 98 = 83 px, so only the 98 px read is a close-up.
DL_WIDTHS = [75, 80, 88, 88, 94, 94, 98, 98, 100, 100, 98, 98, 89]


def test_finish_accepts_identical_reads_at_any_distance():
    v = PlateVoter(FAR_END)
    _approach(v, {0: ("DL12CS4288", 0.97), 1: ("DL12CS4288", 0.96), 7: ("DL12CS4288", 0.949)}, DL_WIDTHS)
    d = v.finish(1)
    assert d is not None and d.plate == "DL12CS4288" and d.votes == 3 and d.agreement == 1.0


@pytest.mark.parametrize(
    "reads",
    [
        # a far read that disagrees blocks it: far reads never outvote or join close-up reads
        {0: ("DL12CS4288", 0.97), 1: ("DL12CS4280", 0.96), 7: ("DL12CS4288", 0.949)},
        {0: ("DL12CS4288", 0.97), 7: ("DL12CS4288", 0.949)},  # only 2 reads
        {0: ("DL12CS4288", 0.80), 1: ("DL12CS4288", 0.80), 7: ("DL12CS4288", 0.85)},  # mean < 0.85
    ],
)
def test_finish_identical_reads_rule_stays_strict(reads):
    # (identical reads with none close-up: test_sustained_close_up_still_sets_floor)
    v = PlateVoter(FAR_END)
    _approach(v, reads, DL_WIDTHS)
    assert v.finish(1) is None


def test_finish_disabled_and_unknown_track():
    v = PlateVoter(VoteConfig(end_min_votes=0))
    v.add(1, vp("HR51CX6945", 0.99), np.zeros((4, 4, 3), np.uint8))
    v.add(1, vp("HR51CX6945", 0.99), np.zeros((4, 4, 3), np.uint8))
    assert v.finish(1) is None
    assert PlateVoter(END).finish(99) is None


def test_finish_skips_already_reported_track():
    v = PlateVoter(END)
    for _ in range(3):
        d = v.add(1, vp("HR51CX6945", 0.99), np.zeros((4, 4, 3), np.uint8))
    assert d is not None
    assert v.finish(1) is None


def test_brief_box_spike_does_not_raise_close_up_floor():
    """NL01AC0056: the plate grew 101 -> 119 px with correct reads, then for 3 frames the box took in
    the "C" of painted "CARRIER" (130, 144, 130 px). A single-frame maximum put the floor at 122 px
    and left one read; the median of 3 keeps the floor at 0.85 * 130 and the plate is saved."""
    v = PlateVoter(VoteConfig(min_votes=3, min_avg_conf=0.85, min_agreement=0.7, close_ratio=0.85))
    crop = np.zeros((4, 4, 3), np.uint8)
    ts = 0.0
    for w in [101, 103, 105, 104, 109, 110, 113, 119, 113]:
        v.observe(1, w, ts)
        assert v.add(1, vp("NL01AC0056", 0.97), crop, width=w) is None  # still approaching
        ts += 0.2
    for w in [130, 144, 130, 112]:
        v.observe(1, w, ts)
        ts += 0.2
    d = v.finish(1)
    assert d is not None and d.plate == "NL01AC0056"


def test_sustained_close_up_still_sets_floor():
    """TN33BY9603: far misreads must not vote when the vehicle then stayed big with no valid reads."""
    v = PlateVoter(
        VoteConfig(min_votes=3, min_avg_conf=0.85, min_agreement=0.7, close_ratio=0.85, end_min_votes=2)
    )
    crop = np.zeros((4, 4, 3), np.uint8)
    ts = 0.0
    for w in [120, 122, 124, 126]:
        v.observe(1, w, ts)
        v.add(1, vp("TN33BT9603", 0.97), crop, width=w)
        ts += 0.2
    for w in [160, 180, 200, 230, 260, 270, 270]:
        v.observe(1, w, ts)
        ts += 0.2
    assert v.finish(1) is None
