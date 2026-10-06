#!/usr/bin/env bash
# One command: labelled crops -> fine-tuned OCR candidate -> gate report. Never touches the live model.
#
#   training/train.sh [--name NAME] [--data training/data] [--epochs 150] [--batch-size 64] [--lr 3e-4]
#                     [--patience 30] [--test-video PATH ...] [--test-source NAME ...] [--skip-eval] [--force]
#                     [--python PY] [--runtime-python PY]
#
# Steps (details: training/README.md, background: training/RESEARCH.md):
#   1. fetch + sha256-check the pretrained checkpoint (the exact model the Pi runs now)
#   2. split: VERIFIED rows of <data>/labels.csv -> training/runs/NAME/splits/{train,val,test}.csv.
#      The evaluation videos (training/ground_truth/*.csv) are ALWAYS held out as test, and their plates are
#      kept out of train/val, so the model is never trained on what it is judged on. --test-video /
#      --test-source hold out more footage on top of that.
#   3. fine-tune from the checkpoint (fast-plate-ocr), 4. export ONNX,
#   5. write models/ocr/candidates/NAME/{plate_ocr.onnx,plate_ocr_config.yaml,train_info.json,...}
#   6. check the runtime venv loads it, 7. run the gate: python -m training.evaluate (exit code 0 = PASS).
#
# Mac: training uses .venv-train, the checks use the runtime .venv. Colab: pass --python python; the runtime
# check and the gate need the repo's .venv and the test videos, so they are skipped there (run them on the Mac).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

NAME="$(date +%Y%m%d-%H%M)"
DATA="training/data"
EPOCHS=150
BATCH=64
LR=3e-4
PATIENCE=30
SKIP_EVAL=0
FORCE=0
PY=""
RT=""
TEST_ARGS=()

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
while [ $# -gt 0 ]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --data) DATA="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --patience) PATIENCE="$2"; shift 2 ;;
    --test-video) TEST_ARGS+=(--test-video "$2"); shift 2 ;;
    --test-source) TEST_ARGS+=(--test-source "$2"); shift 2 ;;
    --skip-eval) SKIP_EVAL=1; shift ;;
    --force) FORCE=1; shift ;;
    --python) PY="$2"; shift 2 ;;
    --runtime-python) RT="$2"; shift 2 ;;
    -h|--help) usage 0 ;;
    *) echo "unknown option: $1" >&2; usage 2 ;;
  esac
done
case "$NAME" in
  [A-Za-z0-9]*) case "$NAME" in *[!A-Za-z0-9._-]*) NAME_OK=0 ;; *) NAME_OK=1 ;; esac ;;
  *) NAME_OK=0 ;;
esac
if [ "$NAME_OK" != 1 ]; then  # also refuses "", "." and ".." (rm -rf below uses the name)
  echo "--name: start with a letter or digit, then letters, digits, . _ - only" >&2; exit 2
fi

if [ -z "$PY" ]; then
  if [ -x .venv-train/bin/python ]; then PY=.venv-train/bin/python; else PY=python3; fi
fi
if [ -z "$RT" ] && [ -x .venv/bin/python ]; then RT=.venv/bin/python; fi
FPO=("$PY" -c 'import sys; from fast_plate_ocr.cli.cli import main_cli; sys.argv[0] = "fast-plate-ocr"; main_cli()')
export KERAS_BACKEND=tensorflow
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

RUN="training/runs/$NAME"
CAND="models/ocr/candidates/$NAME"
if [ -e "$CAND" ] && [ "$FORCE" != 1 ]; then
  echo "$CAND already exists (use another --name, or --force to replace it)" >&2; exit 2
fi
if ! "$PY" -c "import fast_plate_ocr.cli.cli" 2>/dev/null; then
  echo "fast-plate-ocr[train] is not installed for $PY: $PY -m pip install -r training/requirements-train.txt" >&2
  exit 2
fi
rm -rf "$RUN"
mkdir -p "$RUN/splits" "$RUN/export"
exec > >(tee "$RUN/train.log") 2>&1
echo "== run $NAME  data=$DATA  python=$PY  runtime=${RT:-none}"

echo "== 1/7 pretrained checkpoint"
bash training/fetch_pretrained.sh

echo "== 2/7 split"
# the ground-truth videos are always the test split and their plates never go into train/val
for t in training/ground_truth/*.csv; do if [ -f "$t" ]; then TEST_ARGS+=(--truth "$t"); fi; done
if [ ${#TEST_ARGS[@]} -eq 0 ]; then
  echo "no training/ground_truth/*.csv and no --test-video/--test-source: nothing to hold out for the test" >&2
  exit 2
fi
"$PY" -m training.split --data "$DATA" --out "$RUN/splits" "${TEST_ARGS[@]}"
rows() { if [ -f "$1" ]; then echo $(( $(wc -l < "$1") - 1 )); else echo 0; fi; }
N_TRAIN=$(rows "$RUN/splits/train.csv"); N_VAL=$(rows "$RUN/splits/val.csv"); N_TEST=$(rows "$RUN/splits/test.csv")
echo "crops: train $N_TRAIN, val $N_VAL, test $N_TEST"
# what the model is trained on, recorded NOW from the split CSVs actually used (not from labels.csv later)
"$PY" - "$DATA" "$RUN" "$NAME" "${TEST_ARGS[@]}" <<'EOF'
import csv, json, os, subprocess, sys
from collections import Counter
from datetime import datetime
from pathlib import Path

data, run, name, held_out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]), sys.argv[3], sys.argv[4:]
with open(data / "labels.csv", newline="", encoding="utf-8") as f:
    by_image = {str((data / r["image_path"]).resolve()): r for r in csv.DictReader(f)}
splits = {}
for s in ("train", "val", "test"):
    p = run / "splits" / f"{s}.csv"
    rows = []
    if p.is_file():
        with open(p, newline="", encoding="utf-8") as f:
            for it in csv.DictReader(f):
                r = by_image.get(str((p.parent / it["image_path"]).resolve()))
                if r is None:
                    sys.exit(f"{p}: {it['image_path']} is not in {data / 'labels.csv'}")
                rows.append(r | {"plate_text": it["plate_text"]})
    splits[s] = rows


def pick(split, key):
    return sorted({r[key] for r in splits[split]})


def count(split):
    rs = splits[split]
    return {
        "crops": len(rs),
        "vehicles": len({(r["source"], r["video"], r["track"]) for r in rs}),
        "two_line": sum(r["two_line"] == "1" for r in rs),
        "per_source": dict(Counter(r["source"] for r in rs)),
    }


try:
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
except OSError:
    rev = ""
info = {
    "name": name,
    "created": datetime.now().astimezone().isoformat(timespec="seconds"),
    "git": rev or None,
    "data": str(data),
    "run_dir": str(run),
    "base_checkpoint": "cct_xs_v2_global.keras (sha256 0716717772b1f8d2...)",
    "held_out": held_out,
    "train_sources": pick("train", "source"),
    "val_sources": pick("val", "source"),
    "test_sources": pick("test", "source"),
    "train_videos": pick("train", "video"),
    "val_videos": pick("val", "video"),
    "test_videos": pick("test", "video"),
    "train_plates": sorted({r["plate_text"] for s in ("train", "val") for r in splits[s]}),
    "test_csv": os.path.relpath(run / "splits" / "test.csv"),  # relative to the repo root
    "counts": {s: count(s) for s in ("train", "val", "test")},
}
(run / "train_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
print(json.dumps(info["counts"], indent=2))
EOF
if [ "$N_TRAIN" -lt 1 ] || [ "$N_VAL" -lt 1 ]; then
  echo "need verified crops in both train and val: label more in the labeler (training/README.md step 2)" >&2
  exit 1
fi
if [ "$N_TRAIN" -lt 500 ]; then
  echo "WARNING: only $N_TRAIN training crops; a useful fine-tune needs >= 2000 crops from >= 500 vehicles"
fi

echo "== 3/7 fine-tune (checks every image + label first; a missing/broken image or a too-long label stops here)"
"${FPO[@]}" train \
  --model-config-file training/configs/model_config.yaml \
  --plate-config-file training/configs/plate_config.yaml \
  --annotations "$RUN/splits/train.csv" --val-annotations "$RUN/splits/val.csv" \
  --validate-dataset error \
  --weights-path training/pretrained/cct_xs_v2_global.keras \
  --lr "$LR" --warmup-fraction 0.05 --batch-size "$BATCH" --epochs "$EPOCHS" \
  --early-stopping-metric val_plate_acc --early-stopping-patience "$PATIENCE" \
  --seed 42 --output-dir "$RUN/trained"
TS_DIR="$(ls -d "$RUN"/trained/*/ | sort | tail -1)"
TS_DIR="${TS_DIR%/}"
[ -f "$TS_DIR/best.keras" ] || { echo "training produced no $TS_DIR/best.keras" >&2; exit 1; }
if [ "$N_TEST" -gt 0 ]; then
  echo "-- held-out crops (Keras, raw OCR, no validator):"
  "${FPO[@]}" valid -m "$TS_DIR/best.keras" --plate-config-file "$TS_DIR/plate_config.yaml" \
    -a "$RUN/splits/test.csv" -b 32 || echo "WARNING: fast-plate-ocr valid failed (the gate below still runs)"
fi

echo "== 4/7 export ONNX"
"${FPO[@]}" export -m "$TS_DIR/best.keras" -f onnx --plate-config-file "$TS_DIR/plate_config.yaml" \
  --save-dir "$RUN/export" --dynamic-batch --simplify

echo "== 5/7 candidate -> $CAND"
rm -rf "$CAND"
mkdir -p "$CAND"
cp "$RUN/export/best.onnx" "$CAND/plate_ocr.onnx"
cp "$TS_DIR/plate_config.yaml" "$CAND/plate_ocr_config.yaml"
for f in hyper_params.json training_log.csv; do if [ -f "$TS_DIR/$f" ]; then cp "$TS_DIR/$f" "$CAND/"; fi; done
cp "$RUN/train_info.json" "$CAND/train_info.json"

if [ -z "$RT" ]; then
  echo "== 6/7, 7/7 skipped: no runtime .venv here. On the Mac run: .venv/bin/python -m training.evaluate --candidate $CAND"
  exit 0
fi
echo "== 6/7 runtime check (anpr.ocr.FastPlateOcr in $RT)"
"$RT" - "$CAND" <<'EOF'
import sys
from pathlib import Path
import numpy as np
from anpr.config import OcrConfig
from anpr.ocr import FastPlateOcr
c = Path(sys.argv[1])
ocr = FastPlateOcr(OcrConfig(model_path=c / "plate_ocr.onnx", config_path=c / "plate_ocr_config.yaml", num_threads=1))
res = ocr.read(np.full((40, 160, 3), 200, np.uint8))
print("runtime loads the candidate OK; blank-crop read:", res)
EOF

if [ "$SKIP_EVAL" = 1 ]; then
  echo "== 7/7 skipped (--skip-eval). Run: $RT -m training.evaluate --candidate $CAND"
  exit 0
fi
echo "== 7/7 gate: candidate vs the model in config.yaml"
set +e
"$RT" -m training.evaluate --candidate "$CAND"
code=$?
set -e
if [ $code -eq 0 ]; then
  echo "PASS. To use it: training/README.md step 6 (point config.yaml ocr.model_path/config_path at $CAND)."
else
  echo "FAIL: keep the current model. Report: $CAND/evaluation.md"
fi
exit $code
