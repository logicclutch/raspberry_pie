#!/usr/bin/env bash
# Download the fast-plate-ocr cct_xs_v2_global Keras checkpoint (the exact model deployed as
# models/ocr/plate_ocr.onnx) into training/pretrained/ and verify its sha256. Works on macOS and Linux/Colab.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p pretrained
URL=https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates/cct_xs_v2_global.keras
SHA=0716717772b1f8d25b3c227e1e65e7f42e63900ec017059b4a32155488735ffd
OUT=pretrained/cct_xs_v2_global.keras
if [ ! -f "$OUT" ]; then
  curl -fL -o "$OUT.part" "$URL"
  mv "$OUT.part" "$OUT"
fi
if command -v sha256sum >/dev/null 2>&1; then
  echo "$SHA  $OUT" | sha256sum -c -
else
  echo "$SHA  $OUT" | shasum -a 256 -c -
fi
