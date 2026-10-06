"""Gate for a fine-tuned OCR candidate: is it better than the model the Pi runs now?

    .venv/bin/python -m training.evaluate --candidate models/ocr/candidates/<run>
        [--config config.yaml] [--truth training/ground_truth/gate_cam.csv ...] [--crops test.csv]
        [--min-precision 0.0] [--min-crops 100] [--json out.json]

Runs in the RUNTIME venv (.venv): the same detector, tracker, de-shear, validator and voting as the Pi,
with the OCR swapped. BASELINE = config.yaml's ocr.model_path / ocr.config_path; CANDIDATE =
<dir>/plate_ocr.onnx + <dir>/plate_ocr_config.yaml. Exit 0 = PASS, 1 = FAIL. A report (evaluation.md /
evaluation.json) is written into the candidate folder.

1. Videos (the main test): every video listed in a ground-truth CSV (training/ground_truth/*.csv, columns
   video,vehicle,plate,status,first_frame,last_frame,tracks,note; video path relative to the CSV) is played
   frame by frame through the real engine. Each CONFIRMED plate is then:
     correct   - equals a `sure` plate of that video (the first time; a second time counts as wrong)
     unscored  - equals an `unsure` plate, or overlaps in time only with `unsure` vehicles (can't be judged,
                 so a candidate may not show an unscored plate the baseline did not show: see GATE)
     wrong     - anything else (a wrong plate shown to the user: the worst error in strict ANPR)
   missed = sure plates never confirmed. precision = correct / (correct + wrong), recall = correct / sure.
2. Crops (optional, --crops, or the candidate's test split from train_info.json): each labelled OCR-ready crop
   (CSV image_path,plate_text) is read once by each model: exact-match accuracy of the raw read and of the
   validated plate.
Character confusions (truth char -> read char, "-" = dropped / extra) are counted on every wrong read and
every wrong crop, so you can see whether B/8, O/0, A/4, G/6 and dropped last characters got better.

GATE (all must hold for PASS):
  * videos: candidate wrong <= baseline wrong           (never show more wrong plates)
  * videos: candidate correct >= baseline correct       (never miss more plates)
  * videos: every plate the candidate shows on an UNSURE vehicle was also shown by the baseline (it might
    be wrong and nobody can tell: fail closed. Fix: check it by eye and correct / mark sure the ground truth)
  * crops: a crop test set was run (--crops, or the candidate's test split) with >= --min-crops crops
  * crops: candidate validated accuracy >= baseline
  * candidate strictly better in at least one of: video correct, video wrong, crop validated accuracy
  * candidate video precision >= --min-precision        (0 by default; 0.995 = the WORKFLOW A6 ship gate)
  * no leak: the candidate has a train_info.json (train.sh writes it) and, per that file, was not trained
    (train or val split) on crops from a test video (matched by path or file name) nor on a test plate.
    train.sh holds both out.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TRUTH = (REPO / "training/ground_truth/gate_cam.csv", REPO / "training/ground_truth/phone.csv")
MISSING = "-"  # confusion symbol for a dropped or extra character
TIME_MARGIN = 5  # frames: an event this close to an unsure vehicle's span counts as overlapping it

log = logging.getLogger("training.evaluate")


# ---- ground truth ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Vehicle:
    video: Path
    vehicle: str
    plate: str
    sure: bool
    first_frame: int | None = None
    last_frame: int | None = None


def read_truth(csv_path: Path) -> list[Vehicle]:
    """Ground-truth CSV -> vehicles. Lines starting with '#' are comments."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    out: list[Vehicle] = []
    for row in csv.DictReader(lines):
        video = Path(row["video"].strip())
        if not video.is_absolute():
            video = (csv_path.parent / video).resolve()
        status = row["status"].strip().lower()
        if status not in {"sure", "unsure"}:
            raise ValueError(f"{csv_path}: status must be sure/unsure, got {status!r}")
        ff, lf = (row.get("first_frame") or "").strip(), (row.get("last_frame") or "").strip()
        out.append(
            Vehicle(
                video=video,
                vehicle=row["vehicle"].strip(),
                plate=_norm(row["plate"]),
                sure=status == "sure",
                first_frame=int(ff) if ff else None,
                last_frame=int(lf) if lf else None,
            )
        )
    return out


def _norm(text: str) -> str:
    return "".join(c for c in (text or "").upper() if c.isalnum())


# ---- scoring (pure logic, unit-tested) ---------------------------------------------------------


@dataclass(frozen=True)
class Shown:
    """One confirmed plate from the engine, with its frame span in the video."""

    plate: str
    first_frame: int
    last_frame: int


def align(truth: str, read: str) -> list[tuple[str, str]]:
    """Levenshtein alignment -> (truth char, read char) pairs; MISSING marks a dropped / extra char."""
    n, m = len(truth), len(read)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(
                d[i - 1][j - 1] + (truth[i - 1] != read[j - 1]),
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
            )
    pairs: list[tuple[str, str]] = []
    i, j = n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (truth[i - 1] != read[j - 1]):
            pairs.append((truth[i - 1], read[j - 1]))
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            pairs.append((truth[i - 1], MISSING))
            i -= 1
        else:
            pairs.append((MISSING, read[j - 1]))
            j -= 1
    return pairs[::-1]


def confusions(truth: str, read: str) -> Counter[str]:
    """Counter of 'T>R' for every character that differs (T or R may be MISSING)."""
    return Counter(f"{t}>{r}" for t, r in align(truth, read) if t != r)


def _overlaps(s: Shown, v: Vehicle, margin: int = TIME_MARGIN) -> bool:
    if v.first_frame is None or v.last_frame is None:
        return False
    return s.first_frame <= v.last_frame + margin and v.first_frame - margin <= s.last_frame


def _closest(s: Shown, vehicles: list[Vehicle]) -> Vehicle | None:
    """The vehicle a wrong read most likely belongs to: overlapping in time, else the most similar plate."""
    if not vehicles:
        return None
    timed = [v for v in vehicles if _overlaps(s, v, 0)]
    pool = timed or vehicles
    return min(pool, key=lambda v: sum(t != r for t, r in align(v.plate, s.plate)))


@dataclass
class VideoScore:
    sure: int = 0
    correct: int = 0
    wrong: int = 0
    unscored: int = 0
    missed: list[str] = field(default_factory=list)
    wrong_reads: list[tuple[str, str, str]] = field(default_factory=list)  # (video, truth guess, shown)
    unscored_reads: list[tuple[str, str]] = field(default_factory=list)  # (video, shown)
    confusions: Counter[str] = field(default_factory=Counter)

    @property
    def precision(self) -> float:
        judged = self.correct + self.wrong
        return self.correct / judged if judged else 1.0

    @property
    def recall(self) -> float:
        return self.correct / self.sure if self.sure else 1.0

    def merge(self, other: VideoScore) -> None:
        self.sure += other.sure
        self.correct += other.correct
        self.wrong += other.wrong
        self.unscored += other.unscored
        self.missed += other.missed
        self.wrong_reads += other.wrong_reads
        self.unscored_reads += other.unscored_reads
        self.confusions.update(other.confusions)


def score_video(name: str, shown: list[Shown], vehicles: list[Vehicle]) -> VideoScore:
    sure = [v for v in vehicles if v.sure]
    unsure = [v for v in vehicles if not v.sure]
    sc = VideoScore(sure=len(sure))
    matched: set[str] = set()
    sure_plates = {v.plate for v in sure}
    unsure_plates = {v.plate for v in unsure}
    for s in sorted(shown, key=lambda x: x.first_frame):
        if s.plate in sure_plates and s.plate not in matched:
            matched.add(s.plate)
            sc.correct += 1
            continue
        only_unsure = any(_overlaps(s, v) for v in unsure) and not any(_overlaps(s, v) for v in sure)
        if s.plate not in sure_plates and (s.plate in unsure_plates or only_unsure):
            sc.unscored += 1
            sc.unscored_reads.append((name, s.plate))
            continue
        sc.wrong += 1
        guess = _closest(s, sure) if s.plate not in sure_plates else None
        truth = guess.plate if guess else s.plate  # a repeated correct plate: shown twice
        sc.wrong_reads.append((name, truth if guess else f"{truth} (again)", s.plate))
        if guess:
            sc.confusions.update(confusions(guess.plate, s.plate))
    sc.missed = sorted(sure_plates - matched)
    return sc


@dataclass
class CropScore:
    crops: int = 0
    raw_exact: int = 0  # raw OCR text == label
    valid_exact: int = 0  # validated plate == label
    errors: list[tuple[str, str, str]] = field(default_factory=list)  # (image, truth, read)
    confusions: Counter[str] = field(default_factory=Counter)

    @property
    def raw_acc(self) -> float:
        return self.raw_exact / self.crops if self.crops else 0.0

    @property
    def valid_acc(self) -> float:
        return self.valid_exact / self.crops if self.crops else 0.0

    def add(self, image: str, truth: str, raw: str, valid: str | None) -> None:
        self.crops += 1
        self.raw_exact += raw == truth
        self.valid_exact += valid == truth
        if raw != truth:
            self.errors.append((image, truth, raw))
            self.confusions.update(confusions(truth, raw))


@dataclass
class ModelScore:
    video: VideoScore | None = None
    crops: CropScore | None = None


def gate(
    base: ModelScore,
    cand: ModelScore,
    min_precision: float = 0.0,
    min_crops: int = 100,
    leaks: list[str] | None = None,
    require_crops: bool = True,
) -> tuple[bool, list[str]]:
    """-> (passed, one line per rule with OK / FAIL)."""
    lines: list[str] = []
    ok = True
    better = False

    def rule(passed: bool, text: str) -> None:
        nonlocal ok
        ok &= passed
        lines.append(f"{'OK  ' if passed else 'FAIL'} {text}")

    if leaks:
        rule(False, f"no leak: candidate may have been trained on the test data: {'; '.join(leaks)}")
    bv, cv = base.video, cand.video
    if bv is None or cv is None:
        rule(False, "video test ran (no ground-truth videos found)")
    else:
        rule(cv.wrong <= bv.wrong, f"video wrong plates: candidate {cv.wrong} <= baseline {bv.wrong}")
        rule(
            cv.correct >= bv.correct, f"video correct plates: candidate {cv.correct} >= baseline {bv.correct}"
        )
        rule(
            cv.precision >= min_precision,
            f"video precision: candidate {cv.precision:.4f} >= {min_precision}",
        )
        new_unscored = sorted({p for _v, p in set(cv.unscored_reads) - set(bv.unscored_reads)})
        rule(
            not new_unscored,
            "no new plates on unsure vehicles"
            + (
                f": candidate shows {', '.join(new_unscored)} (check by eye, fix / mark sure in ground truth)"
                if new_unscored
                else ""
            ),
        )
        better |= cv.correct > bv.correct or cv.wrong < bv.wrong
    bc, cc = base.crops, cand.crops
    if (bc is None or cc is None) and require_crops:
        rule(False, "crop test ran (no labelled held-out crops: label the test videos' crops, or --crops)")
    if bc is not None and cc is not None:
        rule(cc.crops >= min_crops, f"crop test set size: {cc.crops} >= {min_crops}")
        rule(
            cc.valid_acc >= bc.valid_acc,
            f"crop validated accuracy: candidate {cc.valid_acc:.4f} >= baseline {bc.valid_acc:.4f}",
        )
        better |= cc.valid_acc > bc.valid_acc
    rule(better, "candidate strictly better somewhere (more correct, fewer wrong, or higher crop accuracy)")
    return ok, lines


# ---- running the models ------------------------------------------------------------------------


def _video_events(cfg, video: Path, detector, ocr) -> list[Shown]:  # noqa: ANN001 (anpr types, lazy import)
    """Play one video through the real engine (every frame, like the Pi with realtime off)."""
    from anpr.camera import make_source
    from anpr.config import StorageConfig
    from anpr.engine import Engine
    from anpr.storage import SqliteEventStore

    with tempfile.TemporaryDirectory() as tmp:
        vcfg = cfg.model_copy(
            update={
                "camera": cfg.camera.model_copy(update={"source": str(video), "realtime": False}),
                "storage": StorageConfig(db_path=Path(tmp) / "e.db", image_dir=Path(tmp) / "img"),
            }
        )
        store = SqliteEventStore(vcfg.storage.db_path, vcfg.storage.image_dir)
        src = make_source(vcfg.camera)
        try:
            engine = Engine(vcfg, detector, ocr, store)
            src.start()
            ts0: float | None = None
            events = []
            while True:
                got = src.read(timeout=1.0)
                if got is None:
                    if src.finished:
                        break
                    continue
                frame, ts = got
                ts0 = ts if ts0 is None else ts0
                events += engine.process(frame, ts)
            events += engine.flush()
            fps = getattr(src, "file_fps", None) or float(vcfg.camera.fps)  # known once the file is open
        finally:
            src.stop()
            store.close()
    t0 = ts0 or 0.0
    return [Shown(e.plate, round((e.first_seen - t0) * fps), round((e.last_seen - t0) * fps)) for e in events]


def eval_videos(cfg, truth: dict[Path, list[Vehicle]], detector, ocr) -> VideoScore:  # noqa: ANN001
    total = VideoScore()
    for video, vehicles in truth.items():
        shown = _video_events(cfg, video, detector, ocr)
        total.merge(score_video(video.name, shown, vehicles))
    return total


def read_crop_csv(csv_path: Path) -> list[tuple[Path, str]]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return [
            ((csv_path.parent / r["image_path"]).resolve(), _norm(r["plate_text"]))
            for r in csv.DictReader(f)
            if r.get("image_path") and _norm(r.get("plate_text", ""))
        ]


def eval_crops(cfg, crops: list[tuple[Path, str]], ocr) -> CropScore:  # noqa: ANN001
    import cv2

    from anpr.validator import PlateValidator

    validator = PlateValidator(cfg.validation)
    sc = CropScore()
    for path, truth in crops:
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(path)
        res = ocr.read(img)
        raw = _norm(res.text) if res else ""
        valid = validator.validate(res) if res else None
        sc.add(path.name, truth, raw, valid.text if valid else None)
    return sc


# ---- report ------------------------------------------------------------------------------------


def _top(c: Counter[str], n: int = 12) -> str:
    return ", ".join(f"{k} x{v}" for k, v in c.most_common(n)) or "none"


def report(base: ModelScore, cand: ModelScore, passed: bool, rules: list[str], meta: dict[str, str]) -> str:
    out = [f"# OCR candidate evaluation: {'PASS' if passed else 'FAIL'}", ""]
    out += [f"- {k}: `{v}`" for k, v in meta.items()]
    if base.video and cand.video:
        b, c = base.video, cand.video
        out += [
            "",
            "## Videos (real engine, voting on)",
            "",
            "| | baseline | candidate |",
            "|---|---|---|",
            f"| sure plates | {b.sure} | {c.sure} |",
            f"| correct | {b.correct} | {c.correct} |",
            f"| wrong | {b.wrong} | {c.wrong} |",
            f"| unscored (unsure vehicles) | {b.unscored} | {c.unscored} |",
            f"| missed | {len(b.missed)} | {len(c.missed)} |",
            f"| precision | {b.precision:.3f} | {c.precision:.3f} |",
            f"| recall | {b.recall:.3f} | {c.recall:.3f} |",
            "",
            f"Baseline confusions: {_top(b.confusions)}",
            f"Candidate confusions: {_top(c.confusions)}",
            "",
        ]
        for label, s in (("baseline", b), ("candidate", c)):
            for video, truth, shown in s.wrong_reads:
                out.append(f"- {label} WRONG {video}: truth {truth}, shown {shown}")
            for video, shown in s.unscored_reads:
                out.append(f"- {label} unscored {video}: shown {shown} (check by eye)")
            if s.missed:
                out.append(f"- {label} missed: {' '.join(s.missed)}")
    if base.crops and cand.crops:
        b2, c2 = base.crops, cand.crops
        out += [
            "",
            "## Crops (OCR on labelled crops, no voting)",
            "",
            "| | baseline | candidate |",
            "|---|---|---|",
            f"| crops | {b2.crops} | {c2.crops} |",
            f"| raw exact | {b2.raw_acc:.3f} | {c2.raw_acc:.3f} |",
            f"| validated exact | {b2.valid_acc:.3f} | {c2.valid_acc:.3f} |",
            "",
            f"Baseline confusions: {_top(b2.confusions)}",
            f"Candidate confusions: {_top(c2.confusions)}",
        ]
    out += ["", "## Gate", ""] + [f"    {r}" for r in rules]
    return "\n".join(out) + "\n"


def _as_json(s: ModelScore) -> dict[str, object]:
    d: dict[str, object] = {}
    if s.video:
        d["video"] = asdict(s.video) | {"precision": s.video.precision, "recall": s.video.recall}
    if s.crops:
        c = s.crops
        d["crops"] = {
            "crops": c.crops,
            "raw_acc": c.raw_acc,
            "valid_acc": c.valid_acc,
            "confusions": c.confusions,
        }
    return d


def find_leaks(
    videos: list[Path],
    source: dict[Path, str],
    info: dict[str, object],
    plates: set[str] | None = None,
) -> list[str]:
    """Test videos / plates the candidate saw in training (train_info.json from train.sh). Videos match by
    path OR file name (labels.csv keeps the harvest machine's absolute path; a run on another machine or
    Colab must still be caught); if the info has no video lists, by source name (ground-truth file name ==
    harvest --source-name). Plates: `train_plates` (train + val) that are ground-truth plates."""
    seen = [*info.get("train_videos", []), *info.get("val_videos", [])]  # type: ignore[misc]
    out: list[str] = []
    if seen:
        trained = {str(Path(str(v)).resolve()) for v in seen}
        names = {Path(str(v)).name for v in seen}
        out += sorted(v.name for v in videos if str(v.resolve()) in trained or v.name in names)
    else:
        sources = {*info.get("train_sources", []), *info.get("val_sources", [])}  # type: ignore[misc]
        out += sorted(f"{source[v]} ({v.name})" for v in videos if source.get(v) in sources)
    both = sorted(set(plates or ()) & {str(p) for p in info.get("train_plates", [])})  # type: ignore[union-attr]
    if both:
        out.append(f"test plate(s) {' '.join(both)}")
    return out


def _truth_by_video(paths: list[Path]) -> tuple[dict[Path, list[Vehicle]], dict[Path, str]]:
    """-> ({video: vehicles}, {video: source name taken from the CSV file name})."""
    truth: dict[Path, list[Vehicle]] = {}
    source: dict[Path, str] = {}
    for p in paths:
        for v in read_truth(p):
            truth.setdefault(v.video, []).append(v)
            source[v.video] = p.stem
    return truth, source


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--candidate", type=Path, required=True, help="folder with plate_ocr.onnx + plate_ocr_config.yaml"
    )
    ap.add_argument("--config", type=Path, default=REPO / "config.yaml", help="runtime config (baseline OCR)")
    ap.add_argument(
        "--truth", type=Path, action="append", help="ground-truth CSV (repeatable; default: both)"
    )
    ap.add_argument(
        "--crops", type=Path, help="crop CSV image_path,plate_text (default: candidate's test split)"
    )
    ap.add_argument("--no-crops", action="store_true", help="skip the crop test (it then FAILS the gate)")
    ap.add_argument("--no-videos", action="store_true", help="skip the video test (it then FAILS the gate)")
    ap.add_argument("--min-precision", type=float, default=0.0)
    ap.add_argument("--min-crops", type=int, default=100)
    ap.add_argument("--json", type=Path, help="also write the scores here")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    from anpr.config import load_config
    from anpr.detector import make_detector
    from anpr.ocr import FastPlateOcr

    cand_dir = args.candidate.resolve()
    cand_model, cand_yaml = cand_dir / "plate_ocr.onnx", cand_dir / "plate_ocr_config.yaml"
    for p in (cand_model, cand_yaml):
        if not p.is_file():
            ap.error(f"missing {p}")
    info_path = cand_dir / "train_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}

    cfg = load_config(args.config)
    base_ocr = FastPlateOcr(cfg.ocr)
    cand_ocr = FastPlateOcr(cfg.ocr.model_copy(update={"model_path": cand_model, "config_path": cand_yaml}))
    detector = make_detector(cfg.detector)
    base, cand = ModelScore(), ModelScore()

    leaks: list[str] = []
    truth_files = args.truth or [p for p in DEFAULT_TRUTH if p.is_file()]
    if not args.no_videos and truth_files:
        truth, source = _truth_by_video(truth_files)
        missing = [v for v in truth if not v.is_file()]
        if missing:
            ap.error(f"test video not found: {missing[0]}")
        plates = {v.plate for vs_ in truth.values() for v in vs_}
        leaks = find_leaks(list(truth), source, info, plates)
        if not info:
            leaks.append(f"(unknown: no {info_path.name}, so no proof it never saw the test videos)")
        print(f"videos: {len(truth)} ({', '.join(v.name for v in truth)})", flush=True)
        base.video = eval_videos(cfg, truth, detector, base_ocr)
        print("  baseline done", flush=True)
        cand.video = eval_videos(cfg, truth, detector, cand_ocr)
        print("  candidate done", flush=True)

    crop_csv = args.crops
    if crop_csv is None and info.get("test_csv"):
        crop_csv = Path(info["test_csv"])
        crop_csv = crop_csv if crop_csv.is_absolute() else REPO / crop_csv  # train.sh: relative to the repo
    if not args.no_crops and crop_csv is not None and crop_csv.is_file():
        crops = read_crop_csv(crop_csv)
        if crops:
            print(f"crops: {len(crops)} from {crop_csv}", flush=True)
            base.crops = eval_crops(cfg, crops, base_ocr)
            cand.crops = eval_crops(cfg, crops, cand_ocr)

    if not args.no_crops and base.crops is None:
        print(f"crops: none (test CSV {crop_csv or '-'} missing or empty) -> the gate FAILS", flush=True)
    passed, rules = gate(base, cand, args.min_precision, args.min_crops, leaks)
    meta = {
        "baseline": str(cfg.ocr.model_path),
        "candidate": str(cand_model),
        "config": str(args.config),
        "truth": ", ".join(str(p) for p in truth_files) if not args.no_videos else "(skipped)",
        "crops": str(crop_csv) if base.crops else "(none)",
    }
    text = report(base, cand, passed, rules, meta)
    print(text)
    (cand_dir / "evaluation.md").write_text(text, encoding="utf-8")
    data = {
        "passed": passed,
        "rules": rules,
        "meta": meta,
        "baseline": _as_json(base),
        "candidate": _as_json(cand),
    }
    blob = json.dumps(data, indent=2, default=str)
    (cand_dir / "evaluation.json").write_text(blob, encoding="utf-8")
    if args.json:
        args.json.write_text(blob, encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
