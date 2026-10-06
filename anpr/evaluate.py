"""Accuracy gate (WORKFLOW.md step A6). Strict ANPR is judged mainly on PRECISION:
of all plates the system shows, how many are exactly right. Recall is reported too.

Two modes (see scripts/eval_accuracy.py):
  * images: CSV `image,plate` — one still per row, full pipeline minus voting (worst case).
            An empty plate means "no readable plate in this image": any output is a false positive.
  * video:  CSV `video,plates` — plates separated by ';'. Runs the real engine (with voting) on
            every frame and compares the set of reported plates.
"""

from __future__ import annotations

import csv
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from anpr.config import AppConfig, StorageConfig
from anpr.storage import normalize_query
from anpr.types import PlateDetector, PlateOcr
from anpr.validator import PlateValidator


@dataclass
class Score:
    shown: int = 0  # plates the system output
    correct: int = 0  # ...that exactly match the truth
    expected: int = 0  # plates in the ground truth
    errors: list[tuple[str, str, str]] = field(default_factory=list)  # (item, truth, shown)

    @property
    def precision(self) -> float:
        return self.correct / self.shown if self.shown else 1.0

    @property
    def recall(self) -> float:
        return self.correct / self.expected if self.expected else 1.0

    def add(self, item: str, truth: set[str], shown: list[str]) -> None:
        self.expected += len(truth)
        remaining = Counter(truth)
        for s in shown:
            self.shown += 1
            if remaining[s] > 0:
                remaining[s] -= 1
                self.correct += 1
            else:
                self.errors.append((item, ";".join(sorted(truth)), s))

    def report(self, min_precision: float) -> str:
        ok = "PASS" if self.precision >= min_precision else "FAIL"
        lines = [
            f"expected plates : {self.expected}",
            f"shown plates    : {self.shown}",
            f"correct         : {self.correct}",
            f"precision       : {self.precision:.4f}  (gate >= {min_precision})  {ok}",
            f"recall          : {self.recall:.4f}",
        ]
        if self.expected < 300:
            lines.append(f"WARNING: only {self.expected} plates; the gate needs a held-out set of 300+")
        for item, truth, shown in self.errors[:50]:
            lines.append(f"  WRONG  {item}: truth={truth or '-'} shown={shown}")
        return "\n".join(lines)


def _truth(cell: str) -> set[str]:
    return {normalize_query(p) for p in cell.split(";") if normalize_query(p)}


def read_labels(csv_path: Path) -> list[tuple[Path, set[str]]]:
    rows: list[tuple[Path, set[str]]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or row[0].strip().lower() in {"image", "video", "path"} or row[0].startswith("#"):
                continue
            path = Path(row[0].strip())
            if not path.is_absolute():
                path = (csv_path.parent / path).resolve()
            rows.append((path, _truth(row[1] if len(row) > 1 else "")))
    return rows


def eval_images(
    cfg: AppConfig, labels: list[tuple[Path, set[str]]], detector: PlateDetector, ocr: PlateOcr
) -> Score:
    from anpr.preprocess import prepare_plate

    validator = PlateValidator(cfg.validation)
    score = Score()
    for path, truth in labels:
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(path)
        shown: list[str] = []
        for box in detector.detect(img)[: cfg.ocr.max_plates_per_frame]:
            crop = prepare_plate(img, box, cfg.crop)
            if crop is None:
                continue
            res = ocr.read(crop)
            valid = validator.validate(res) if res else None
            if valid and valid.confidence >= cfg.vote.min_avg_conf:
                shown.append(valid.text)
        score.add(path.name, truth, shown)
    return score


def eval_videos(
    cfg: AppConfig, labels: list[tuple[Path, set[str]]], detector: PlateDetector, ocr: PlateOcr
) -> Score:
    from anpr.camera import make_source
    from anpr.engine import Engine, run
    from anpr.storage import SqliteEventStore

    score = Score()
    for path, truth in labels:
        with tempfile.TemporaryDirectory() as tmp:
            vcfg = cfg.model_copy(
                update={
                    "camera": cfg.camera.model_copy(update={"source": str(path), "realtime": False}),
                    "storage": StorageConfig(db_path=Path(tmp) / "e.db", image_dir=Path(tmp) / "img"),
                }
            )
            store = SqliteEventStore(vcfg.storage.db_path, vcfg.storage.image_dir)
            try:
                engine = Engine(vcfg, detector, ocr, store)
                run(vcfg, make_source(vcfg.camera), engine)
                events, _ = store.list_events(limit=10_000)
            finally:
                store.close()
        score.add(path.name, truth, [e.plate for e in events])
    return score
