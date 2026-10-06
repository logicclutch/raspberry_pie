"""Synthetic Indian plates for the OCR fine-tune (optional; a supplement, never a replacement for real crops).

    python -m training.synth --data training/data --count 500 [--two-line 0.35] [--seed 1]

Draws plates with valid Indian text (real state codes, SS NN X[X] NNNN), 1-line or 2-line, then makes
them look like the gate camera: grayscale, shear / small rotation, downscaled to 60-170 px wide, blur, noise,
JPEG, and the same CLAHE the runtime applies (anpr.preprocess.enhance). Rows are added to <data>/labels.csv
as source `synth`, status `verified`, labeler `synth`, so training.split treats them like any other vehicle
(1 plate = 1 track).

Limits (first version): OpenCV's Hershey font, not the HSRP font; no IND strip / hologram; no real dirt.
Keep synthetic rows at most about half of the training set and never put them in the test split
(`training.split --test-source <real source>` does that).
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from anpr.preprocess import enhance
from anpr.validator import STATE_CODES
from training.dataset import IMAGES_DIR, append_new_rows

SOURCE = "synth"
_LETTERS = "ABCDEFGHJKLMNPRSTUVWXYZ"  # series letters (I and O are not issued)


def random_plate(rng: random.Random) -> tuple[str, str, str]:
    """-> (full text, line 1, line 2). Standard format SS NN X[X] NNNN, the most common at the gate."""
    state = rng.choice(sorted(STATE_CODES))
    number = f"{rng.randint(1, 9999):04d}"
    if state == "DL":  # Delhi: 1-2 digit district, vehicle-class letter + 0-2 series letters (DL 3C AB 1234)
        d = rng.randint(1, 13)
        district = str(d)
        series = rng.choice("CSPRTVE") + "".join(
            rng.choice(_LETTERS) for _ in range(rng.randint(0, 2 if d < 10 else 1))
        )
        return state + district + series + number, state + district, series + number
    district = f"{rng.randint(1, 99):02d}"
    series = "".join(rng.choice(_LETTERS) for _ in range(rng.choice((1, 2, 2, 2))))
    return state + district + series + number, state + district, series + number


def _text_block(lines: list[str], scale: float, thick: int) -> np.ndarray:
    """White plate with black text lines, tightly framed with a margin and a thin border."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    sizes = [cv2.getTextSize(t, font, scale, thick)[0] for t in lines]
    w = max(s[0] for s in sizes) + 40
    line_h = max(s[1] for s in sizes)
    gap = int(line_h * 0.45)
    h = len(lines) * line_h + (len(lines) - 1) * gap + 36
    img = np.full((h, w), 235, np.uint8)
    y = 18 + line_h
    for t, (tw, _) in zip(lines, sizes, strict=True):
        cv2.putText(img, t, ((w - tw) // 2, y), font, scale, 20, thick, cv2.LINE_AA)
        y += line_h + gap
    cv2.rectangle(img, (3, 3), (w - 4, h - 4), 40, 3)
    return img


def render(text_lines: list[str], rng: random.Random) -> np.ndarray:
    """-> BGR OCR-ready crop, looking roughly like prepare_plate's output on the gate camera."""
    plate = _text_block(text_lines, scale=2.2, thick=rng.choice((5, 6, 7)))
    ph, pw = plate.shape
    bg = rng.randint(30, 120)
    pad = int(0.12 * max(ph, pw))
    canvas = np.full((ph + 2 * pad, pw + 2 * pad), bg, np.uint8)
    canvas[pad : pad + ph, pad : pad + pw] = plate
    ch, cw = canvas.shape
    # residual shear / rotation after the runtime's de-shear (small), plus perspective-ish scale
    shear = rng.uniform(-0.18, 0.18)
    angle = np.deg2rad(rng.uniform(-5, 5))
    ca, sa = np.cos(angle), np.sin(angle)
    m = np.array([[ca, -sa + shear, 0.0], [sa, ca, 0.0]], np.float32)
    m[:, 2] = np.array([cw / 2, ch / 2]) - m[:, :2] @ np.array([cw / 2, ch / 2])
    warped = cv2.warpAffine(canvas, m, (cw, ch), borderMode=cv2.BORDER_CONSTANT, borderValue=bg)
    # contrast / exposure of a cheap sensor at night or in shade
    lo, hi = rng.randint(0, 70), rng.randint(150, 255)
    warped = (lo + warped.astype(np.float32) * (hi - lo) / 255.0).clip(0, 255).astype(np.uint8)
    # down to camera size (the gate camera sees plates 60-170 px wide), then blur, noise and JPEG
    out_w = rng.randint(60, 170)
    out_h = max(12, round(ch * out_w / cw))
    small = cv2.resize(warped, (out_w, out_h), interpolation=cv2.INTER_AREA)
    k = rng.choice((1, 1, 3))
    if k > 1:
        small = cv2.GaussianBlur(small, (k, k), 0)
    noise = np.random.default_rng(rng.randint(0, 2**31)).normal(0, rng.uniform(0, 8), small.shape)
    small = (small.astype(np.float32) + noise).clip(0, 255).astype(np.uint8)
    ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(35, 90)])
    small = cv2.imdecode(jpg, cv2.IMREAD_GRAYSCALE) if ok else small
    return enhance(cv2.cvtColor(small, cv2.COLOR_GRAY2BGR), 2.0)


def generate(data_dir: Path, count: int, two_line: float, seed: int) -> tuple[int, int]:
    rng = random.Random(seed)
    batch = f"seed{seed}"
    out = data_dir / IMAGES_DIR / SOURCE / batch
    out.mkdir(parents=True, exist_ok=True)
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    rows = []
    for i in range(count):
        full, l1, l2 = random_plate(rng)
        is_two = rng.random() < two_line
        img = render([l1, l2] if is_two else [full], rng)
        name = f"s{i:05d}.png"
        cv2.imwrite(str(out / name), img)
        rel = f"{IMAGES_DIR}/{SOURCE}/{batch}/{name}"
        rows.append(
            {
                "image_path": rel,
                "raw_image_path": rel,
                "source": SOURCE,
                "video": f"synth:{batch}",
                "frame": str(i),
                "track": str(i),
                "width_px": str(img.shape[1]),
                "two_line": "1" if is_two else "0",
                "prep": "synth",
                "plate_text": full,
                "status": "verified",
                "labeler": "synth",
                "labeled_at": now,
            }
        )
    return append_new_rows(data_dir, rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("training/data"), help="folder with labels.csv")
    ap.add_argument("--count", type=int, default=500)
    ap.add_argument("--two-line", type=float, default=0.35, help="share of 2-line plates (default 0.35)")
    ap.add_argument("--seed", type=int, default=1, help="a different seed gives a different batch")
    args = ap.parse_args(argv)
    if args.count < 1 or not 0 <= args.two_line <= 1:
        ap.error("--count must be >= 1 and --two-line in [0, 1]")
    added, present = generate(args.data, args.count, args.two_line, args.seed)
    print(f"synth: added {added} rows ({present} already there) to {args.data / 'labels.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
