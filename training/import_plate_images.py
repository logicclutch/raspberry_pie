"""Import a folder/zip of plate images + a labels file into the OCR training set.

Supports 2-line plates: our OCR model reads ONE line (128x64), and the runtime reads a 2-line plate by
reading its top and bottom lines separately and joining them. So with --two-line each image is split into
a top-half and a bottom-half crop, and the text is split top = first --top-chars characters (state+district
for standard plates, NN+BH for Bharat), bottom = the rest - matching how the runtime reads 2-line plates.

Labels file: lines of "<image filename><TAB><PLATE TEXT>" (the Kaggle double-line-Indian set's labels.txt).

    python -m training.import_plate_images --zip set.zip --labels dataset/labels.txt \
        --images-prefix dataset/images --source kag2line --two-line --limit 4500

Rows are added as `verified` (labeler=import). Treat an external synthetic set like synth: keep it at most
about half the training set and judge the result on the REAL test videos (train.sh always holds those out).
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

from training.dataset import IMAGES_DIR, SOURCE_NAME_RE, append_new_rows


def _split_lines(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Top line and bottom line of a 2-line plate image. A small GAP around the centre is dropped so
    neither crop shows a sliver of the other line (cleaner single-line training samples)."""
    h = img.shape[0]
    mid = h // 2
    gap = max(1, h // 14)
    return img[: mid - gap], img[mid + gap :]


def _row(rel: str, source: str, uid: str, text: str, two_line: str) -> dict:
    return {
        "image_path": rel, "raw_image_path": rel, "source": source,
        "video": f"import:{source}", "frame": uid, "track": uid, "box": "", "width_px": "",
        "two_line": two_line, "prep": "import", "sharpness": "", "ocr_text": "", "ocr_conf": "",
        "ocr_valid": "", "ocr_plate": "", "plate_text": text, "status": "verified",
        "split": "", "labeler": "import", "labeled_at": "",
    }


def run(opts) -> tuple[int, int]:
    if not SOURCE_NAME_RE.fullmatch(opts.source):
        raise SystemExit("--source must be letters/digits/_/- (up to 40)")
    zf = zipfile.ZipFile(opts.zip) if opts.zip else None
    labels_raw = (zf.read(opts.labels) if zf else Path(opts.labels).read_bytes()).decode("utf-8", "replace")
    pairs = [ln.split("\t", 1) for ln in labels_raw.splitlines() if "\t" in ln]

    out_dir = opts.data / IMAGES_DIR / opts.source
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    kept = skipped = 0
    for fn, text in pairs:
        text = "".join(ch for ch in text.upper() if ch.isalnum())
        if not text or (opts.exclude_len and len(text) in opts.exclude_len):
            skipped += 1
            continue
        if opts.two_line and len(text) <= opts.top_chars:
            skipped += 1
            continue
        src_name = fn if not opts.images_prefix else f"{opts.images_prefix.rstrip('/')}/{fn}"
        try:
            data = zf.read(src_name) if zf else (opts.dir / fn).read_bytes()
        except KeyError:
            skipped += 1
            continue
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            skipped += 1
            continue
        stem = Path(fn).stem
        if opts.two_line:
            top_img, bot_img = _split_lines(img)
            top_txt, bot_txt = text[: opts.top_chars], text[opts.top_chars :]
            for part, pimg, ptxt in (("t", top_img, top_txt), ("b", bot_img, bot_txt)):
                rel = f"{IMAGES_DIR}/{opts.source}/{stem}_{part}.png"
                cv2.imwrite(str(opts.data / rel), pimg)
                rows.append(_row(rel, opts.source, f"{stem}_{part}", ptxt, "0"))
        else:
            rel = f"{IMAGES_DIR}/{opts.source}/{stem}.png"
            cv2.imwrite(str(opts.data / rel), img)
            rows.append(_row(rel, opts.source, stem, text, "0"))
        kept += 1
        if opts.limit and kept >= opts.limit:
            break
    if zf:
        zf.close()
    added, dup = append_new_rows(opts.data, rows)
    print(f"plates used: {kept}, skipped: {skipped}; rows added: {added} ({dup} already present)")
    return added, dup


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m training.import_plate_images")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--zip", type=str, help="zip file containing the images + labels")
    src.add_argument("--dir", type=Path, help="folder containing the images")
    ap.add_argument("--labels", required=True, help="labels file: '<filename>\\t<TEXT>' per line")
    ap.add_argument("--images-prefix", default="", help="path prefix inside the zip/dir before each filename")
    ap.add_argument("--source", required=True, help="dataset name, e.g. kag2line")
    ap.add_argument("--data", type=Path, default=Path("training/data"))
    ap.add_argument("--two-line", action="store_true", help="split each image into top/bottom line crops")
    ap.add_argument("--top-chars", type=int, default=4, help="characters on the top line (default 4)")
    ap.add_argument("--exclude-len", type=int, nargs="*", default=[11], help="skip plates of these lengths")
    ap.add_argument("--limit", type=int, default=0, help="use at most this many plates (0 = all)")
    return 0 if run(ap.parse_args(argv))[0] >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
