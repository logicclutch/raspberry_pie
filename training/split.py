"""Assign train / val / test splits to VERIFIED crops and write the trainer's CSVs.

    .venv/bin/python -m training.split --data training/data [--out DIR] [--test-source NAME ...]
        [--test-video PATH ...] [--val 0.1] [--test 0.1] [--seed 13] [--image raw] [--dry-run]

* Crops are grouped by vehicle pass (source, video, track); groups that share a plate text (the same
  vehicle seen twice) are merged. A group never lands in two splits.
* --test-source / --test-video put whole sources / videos in the test split, so the model is evaluated
  on footage it has never seen. A test video matches by its full path OR its file name (labels.csv keeps
  the absolute path from the harvest machine, so a copy of the project on another machine / Colab must
  still hold the test footage out). --truth <ground-truth CSV> (training/ground_truth/*.csv) = --test-video
  for each of its videos AND holds out its plates (the vehicles the gate evaluation judges): groups
  elsewhere whose plate is a test plate are dropped from train/val (reported) unless --allow-plate-overlap.
* Otherwise the split is --val / --test of the groups (default 80/10/10), deterministic for a seed.
* Synthetic rows (--train-only-source, default `synth`) only ever go to train: val (early stopping) and
  test must measure real camera crops.
* Only rows with status "verified" and a plate_text of at most --max-len characters (the model's plate
  slots, default 10) are used; unreadable / skip / unverified / too long are left out (counted). 2-line
  plates are kept as they are (the model sees the 2-line crop) and counted.
* Writes <out>/{train,val,test}.csv with columns image_path,plate_text (paths relative to the CSV, the
  format fast-plate-ocr's trainer reads) and records the split in labels.csv's `split` column.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from training.dataset import Row, locked, read_rows, row_key, track_key, write_rows

SPLITS = ("train", "val", "test")


class _Groups:
    """Union-find over row indices."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


DEFAULT_MAX_LEN = 10  # training/configs/plate_config.yaml max_plate_slots
TRAIN_ONLY_SOURCES = ("synth",)


def usable(r: Row, max_len: int = DEFAULT_MAX_LEN) -> bool:
    return r["status"] == "verified" and 0 < len(r["plate_text"]) <= max_len


def truth_holdout(paths: list[Path]) -> tuple[list[str], set[str]]:
    """Ground-truth CSVs (training.evaluate format) -> (their videos, all their plates, sure or not)."""
    from training.evaluate import read_truth  # stdlib-only module; anpr is imported lazily there

    videos: list[str] = []
    plates: set[str] = set()
    for p in paths:
        for v in read_truth(Path(p)):
            if str(v.video) not in videos:
                videos.append(str(v.video))
            plates.add(v.plate)
    return videos, plates


def group_rows(rows: list[Row], max_len: int = DEFAULT_MAX_LEN) -> list[list[int]]:
    """Indices of usable rows grouped by vehicle pass, merged when they share a plate text."""
    idx = [i for i, r in enumerate(rows) if usable(r, max_len)]
    uf = _Groups(len(rows))
    first_by: dict[tuple, int] = {}
    for i in idx:
        for key in (("track", *track_key(rows[i])), ("plate", rows[i]["plate_text"])):
            j = first_by.setdefault(key, i)
            uf.union(i, j)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in idx:
        groups[uf.find(i)].append(i)
    return sorted(groups.values(), key=lambda g: min(row_key(rows[i]) for i in g))


def assign_splits(
    rows: list[Row],
    *,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 13,
    test_sources: tuple[str, ...] = (),
    test_videos: tuple[str, ...] = (),
    holdout_plates: frozenset[str] | set[str] = frozenset(),
    train_only_sources: tuple[str, ...] = TRAIN_ONLY_SOURCES,
    max_len: int = DEFAULT_MAX_LEN,
    allow_plate_overlap: bool = False,
) -> tuple[dict[int, str], dict[str, int]]:
    """-> ({row index: split}, notes). Rows not in the map get no split."""
    out: dict[int, str] = {}
    notes: Counter[str] = Counter()
    notes["too_long_rows"] = sum(r["status"] == "verified" and len(r["plate_text"]) > max_len for r in rows)
    fixed_test = bool(test_sources or test_videos)
    tvideos = {str(Path(v).resolve()) for v in test_videos}
    tnames = {Path(v).name for v in test_videos}

    def forced_test(r: Row) -> bool:
        if r["source"] in train_only_sources:
            return False
        return r["source"] in test_sources or r["video"] in tvideos or Path(r["video"]).name in tnames

    if fixed_test:
        test_idx = [i for i, r in enumerate(rows) if usable(r, max_len) and forced_test(r)]
        for i in test_idx:
            out[i] = "test"
        test_plates = {rows[i]["plate_text"] for i in test_idx} | set(holdout_plates)
        rest = [r if not forced_test(r) else {**r, "status": "_test"} for r in rows]
        groups = group_rows(rest, max_len)
        val_share = val_frac / max(1e-9, 1.0 - test_frac) if test_frac < 1 else val_frac
        shares = {"val": min(1.0, val_share), "test": 0.0}
    else:
        test_plates = set(holdout_plates)
        groups = group_rows(rows, max_len)
        shares = {"val": val_frac, "test": test_frac}
    if not allow_plate_overlap and test_plates:
        kept = []
        for g in groups:
            if any(rows[i]["plate_text"] in test_plates for i in g):
                notes["dropped_overlap_rows"] += len(g)
            else:
                kept.append(g)
        groups = kept

    # synthetic groups: always train (val / test must measure real camera crops)
    train_only = [g for g in groups if any(rows[i]["source"] in train_only_sources for i in g)]
    groups = [g for g in groups if not any(rows[i]["source"] in train_only_sources for i in g)]
    for g in train_only:
        for i in g:
            out[i] = "train"

    rng = random.Random(seed)
    order = list(range(len(groups)))
    rng.shuffle(order)
    n = len(groups)
    n_test = round(shares["test"] * n)
    n_val = round(shares["val"] * n)
    if n >= 3:  # every requested split gets at least one vehicle when there are enough
        n_test = max(n_test, 1) if shares["test"] > 0 else 0
        n_val = max(n_val, 1) if shares["val"] > 0 else 0
    n_test = min(n_test, n)
    n_val = min(n_val, n - n_test)
    for k, gi in enumerate(order):
        split = "test" if k < n_test else "val" if k < n_test + n_val else "train"
        for i in groups[gi]:
            out[i] = split
    notes["groups"] = n + len(train_only)
    notes["train_only_rows"] = sum(len(g) for g in train_only)
    return out, dict(notes)


def write_split_csvs(
    rows: list[Row], splits: dict[int, str], data_dir: Path, out_dir: Path, image: str
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    col = "raw_image_path" if image == "raw" else "image_path"
    for name in SPLITS:
        items = [rows[i] for i in sorted(splits) if splits[i] == name]
        tmp = out_dir / f".{name}.csv.tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image_path", "plate_text"])
            for r in items:
                rel = os.path.relpath((data_dir / r[col]).resolve(), out_dir.resolve())
                w.writerow([Path(rel).as_posix(), r["plate_text"]])
        os.replace(tmp, out_dir / f"{name}.csv")


def report(rows: list[Row], splits: dict[int, str], notes: dict[str, int]) -> str:
    lines = [f"{'split':<6} {'crops':>6} {'vehicles':>8} {'2-line':>6}  per source"]
    for name in SPLITS:
        idx = [i for i, s in splits.items() if s == name]
        vehicles = len({track_key(rows[i]) for i in idx})
        two = sum(rows[i]["two_line"] == "1" for i in idx)
        per = Counter(rows[i]["source"] for i in idx)
        src = ", ".join(f"{k}={v}" for k, v in sorted(per.items())) or "-"
        lines.append(f"{name:<6} {len(idx):>6} {vehicles:>8} {two:>6}  {src}")
    status = Counter(r["status"] or "unverified" for r in rows)
    lines.append("labels.csv: " + ", ".join(f"{k}={v}" for k, v in sorted(status.items())))
    if notes.get("dropped_overlap_rows"):
        lines.append(
            f"left out of train/val: {notes['dropped_overlap_rows']} crops whose plate is also in the test "
            "footage (use --allow-plate-overlap to keep them)"
        )
    if notes.get("too_long_rows"):
        lines.append(
            f"left out: {notes['too_long_rows']} verified crops whose plate is longer than the model's "
            "plate slots (--max-len; see RESEARCH.md about 11-character Delhi plates)"
        )
    train_idx = [i for i, sp in splits.items() if sp == "train"]
    synth = sum(rows[i]["source"] in TRAIN_ONLY_SOURCES for i in train_idx)
    if train_idx and synth * 2 > len(train_idx):
        lines.append(
            f"WARNING: {synth} of {len(train_idx)} training crops are synthetic; keep them at most about half"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m training.split", description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, default=Path("training/data"), help="folder with labels.csv")
    ap.add_argument("--out", type=Path, help="where train/val/test.csv go (default: --data)")
    ap.add_argument("--test-source", action="append", default=[], help="whole source -> test (repeatable)")
    ap.add_argument("--test-video", action="append", default=[], help="whole video -> test (repeatable)")
    ap.add_argument(
        "--truth",
        action="append",
        default=[],
        type=Path,
        help="ground-truth CSV (training/ground_truth/*.csv): its videos -> test, its plates held out",
    )
    ap.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN, help="model plate slots (default 10)")
    ap.add_argument("--val", type=float, default=0.1, help="val share of vehicles (default 0.1)")
    ap.add_argument("--test", type=float, default=0.1, help="test share when no --test-source/--test-video")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument(
        "--image",
        choices=("ocr", "raw"),
        default="ocr",
        help="ocr (default) = the exact crop the runtime OCR is fed; raw = experiments only (no de-shear / "
        "CLAHE, so a model trained on it does NOT match anpr/ocr.py's input on the Pi)",
    )
    ap.add_argument("--allow-plate-overlap", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the split, write nothing")
    args = ap.parse_args(argv)
    if not (0 <= args.val < 1 and 0 <= args.test < 1 and args.val + args.test < 1):
        ap.error("--val and --test must be in [0, 1) and sum to < 1")
    out_dir = args.out or args.data
    try:
        truth_videos, truth_plates = truth_holdout(args.truth)
    except (OSError, KeyError, ValueError) as e:
        ap.error(f"--truth: {e}")

    with locked(args.data):
        rows = read_rows(args.data)
        if not rows:
            ap.error(f"no rows in {args.data / 'labels.csv'}")
        splits, notes = assign_splits(
            rows,
            val_frac=args.val,
            test_frac=args.test,
            seed=args.seed,
            test_sources=tuple(args.test_source),
            test_videos=(*args.test_video, *truth_videos),
            holdout_plates=truth_plates,
            max_len=args.max_len,
            allow_plate_overlap=args.allow_plate_overlap,
        )
        if not args.dry_run:
            for i, r in enumerate(rows):
                r["split"] = splits.get(i, "")
            write_rows(args.data, rows)
            write_split_csvs(rows, splits, args.data, out_dir, args.image)
    print(report(rows, splits, notes))
    if not splits:
        print("no verified crops yet: label some with `python -m training.labeler` first", file=sys.stderr)
    elif not args.dry_run:
        print(f"wrote {', '.join(str(out_dir / f'{s}.csv') for s in SPLITS)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
