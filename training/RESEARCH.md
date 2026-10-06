# OCR fine-tuning for Indian plates: research and verified recipe

Status 2026-09-29: the tool chain is verified end to end (train → ONNX export → runtime `anpr.ocr.FastPlateOcr`)
on a throwaway synthetic set. **No real training has been done yet**: all 116 harvested crops in
`training/data/labels.csv` are still `unverified`.

> **Use `training/train.sh` now** (see `training/README.md`). It wraps the recipe below (split → fine-tune →
> ONNX export → eval gate) and writes to `models/ocr/candidates/<run>/`. Files named `training/ocr/...` in the
> original notes moved: configs to `training/configs/`, the checkpoint to `training/pretrained/`,
> `requirements-train.txt` and `fetch_pretrained.sh` to `training/`. The manual commands below are kept as a record.

## 1. What is deployed today (verified by hash)

`models/ocr/plate_ocr.onnx` is **byte-identical** to fast-plate-ocr's `cct_xs_v2_global.onnx`
(sha256 `8031afb5…855f44`). `models/ocr/plate_ocr_config.yaml` is identical to `cct_xs_v2_global_plate_config.yaml`
(sha256 `0335c74a…32d0a6`). Earlier notes calling it `cct-s-v1` are wrong. The model is a CCT with about 645 k params,
input uint8 NHWC `[N,64,128,3]` RGB, output `plate` `[N,10,37]` plus `region` `[N,66]`, ONNX opset 15.

## 2. Versions (`.venv-train` only; the runtime `.venv` was not touched)

| package | version | note |
|---|---|---|
| fast-plate-ocr | 1.1.0 | `[train,onnx]` extras |
| tensorflow | 2.16.2 | last release with Intel-mac wheels; pins `ml-dtypes~=0.3.1` |
| keras | 3.15.1 | backend `KERAS_BACKEND=tensorflow` |
| onnx | **1.17.0** | 1.19 (pulled by the extra) crashes export: `ml_dtypes has no attribute float4_e2m1fn`. pip did not flag it |
| tf2onnx | 1.17.0 | |
| onnxslim | 0.1.97 | `--simplify` |
| albumentations | 2.0.8 | |
| runtime `.venv` onnxruntime | 1.20.1 | loads the exported model fine |

Pinned in `training/requirements-train.txt`. The only remaining `pip check` complaint is `ncnn requires portalocker`
(ultralytics side, harmless).

## 3. Checkpoint: fine-tune, not train from scratch

* Release asset: `https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates/cct_xs_v2_global.keras`
  (10 865 307 B, sha256 `0716717772b1f8d25b3c227e1e65e7f42e63900ec017059b4a32155488735ffd`). Fetch it with
  `training/fetch_pretrained.sh`, which verifies the hash.
* This is the exact model deployed today. On 40 real gate crops, Keras vs. the deployed ONNX gives max |Δ| of 8.2e-6 on `plate`
  and 1.8e-6 on `region`, with identical argmax text on 40/40 crops.
* The trainer's `--weights-path` loads with `skip_mismatch=True`. I measured what transfers:

| plate config | head | tensors loaded | not loaded |
|---|---|---|---|
| `training/configs/plate_config.yaml` (no `plate_regions`) | plate only | 135/137 | 2 × `max_blur_pooling2d/blur_kernel` (fixed, non-learned blur constants, identical by construction) |
| original config + `India` region (67 regions) | plate+region | 155/159 | blur kernels and `region/kernel`, `region/bias` |
| `max_plate_slots: 11` | plate only | all except | `token_reducer/query_tokens` (per-slot learned queries → re-initialised) |

  After loading, the no-region model's `plate` output equals the deployed ONNX (Δ 8e-6), so the fine-tune starts
  from exactly the deployed accuracy. In the smoke run, the first batch already had char_acc 0.84, which confirms the
  weights were in use (a random model would be about 0.03).
* **Decision: fine-tune.** Scratch training needs tens of thousands of labelled plates. We will have hundreds to low thousands.

## 4. Config formats

`training/configs/plate_config.yaml` (plate config; must match between training, export and the Pi):
```yaml
max_plate_slots: 10
alphabet: 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_
pad_char: _
img_height: 64
img_width: 128
keep_aspect_ratio: false
interpolation: linear
image_color_mode: rgb
# optional: plate_regions: [..]   -> enables region head, CSV then needs plate_region column
```
`training/configs/model_config.yaml` is an unmodified copy of `cct_xs_v2_global_model_config.yaml`. Top-level keys:
`model: cct`, `rescaling {scale, offset}`, `tokenizer {blocks[], positional_emb, patch_size, patch_mlp}`,
`transformer_encoder {layers:3, heads:2, projection_dim:96, …, normalization: dyt}`. Don't edit it, or the weights won't load.

Annotations CSV (the trainer reads `image_path,plate_text[,plate_region]`, with paths relative to the CSV). `training.split`
already writes exactly this:
```csv
image_path,plate_text
images/gate_cam/…png,MH12AB1234
```
2-line plates: label the text in reading order (top line then bottom line), with no separator. The model reads the 2-line crop as-is.

## 5. Region handling

Recommendation: **no region head** (`plate_regions` removed). Reasons:
* `anpr/ocr.py` reads only the `plate` output (verified: it picks `plate` with and without a `region` output).
* It keeps the CSVs to 2 columns and loads 135/137 tensors.
* All our data is one region, so the region head would learn nothing useful.

Alternative, if a region head is ever wanted: add `India` to `plate_regions`, give every row a `plate_region` column,
and accept that `region/kernel,bias` is re-initialised.

## 6. Verified recipe

```bash
cd /Users/ibm/Desktop/raspberry-main
export KERAS_BACKEND=tensorflow
training/fetch_pretrained.sh                         # checkpoint + sha256 check

# data (existing tools, runtime venv): harvest -> label in the web labeler -> split
.venv/bin/python -m training.harvest --config config.yaml --video gate.mp4 --source-name gate_cam --out training/data
.venv/bin/python -m training.labeler --data training/data
.venv/bin/python -m training.split --data training/data --test-source phone   # writes training/data/{train,val,test}.csv

# sanity checks
.venv-train/bin/fast-plate-ocr validate-dataset -a training/data/train.csv --plate-config-file training/configs/plate_config.yaml
.venv-train/bin/fast-plate-ocr dataset-stats   -a training/data/train.csv -c training/configs/plate_config.yaml

# fine-tune (smoke-tested with --epochs 1/2 --batch-size 16; real-run values below are recommendations)
.venv-train/bin/fast-plate-ocr train \
  --model-config-file training/configs/model_config.yaml \
  --plate-config-file training/configs/plate_config.yaml \
  --annotations training/data/train.csv --val-annotations training/data/val.csv \
  --validate-dataset warn \
  --weights-path training/pretrained/cct_xs_v2_global.keras \
  --lr 3e-4 --warmup-fraction 0.05 --batch-size 64 --epochs 150 \
  --early-stopping-metric val_plate_acc --early-stopping-patience 30 \
  --seed 42 --output-dir trained_models
# -> trained_models/<timestamp>/{best.keras,last.keras,plate_config.yaml,model_config.yaml,hyper_params.json,training_log.csv,train_augmentation.yaml}

# held-out accuracy on the test split
.venv-train/bin/fast-plate-ocr valid -m trained_models/<ts>/best.keras \
  --plate-config-file trained_models/<ts>/plate_config.yaml -a training/data/test.csv -b 32

# export (the --save-dir must already exist)
mkdir -p training/runs/export
.venv-train/bin/fast-plate-ocr export -m trained_models/<ts>/best.keras -f onnx \
  --plate-config-file trained_models/<ts>/plate_config.yaml --save-dir training/runs/export --dynamic-batch --simplify
# prints "ONNX output 'plate' matches Keras ✔"

# deploy: NEVER copy over models/ocr/plate_ocr.onnx. train.sh puts the two files in
# models/ocr/candidates/<run>/, `python -m training.evaluate` must PASS, then point config.yaml's
# ocr.model_path / ocr.config_path at the candidate (training/README.md step 6).
```

Notes:
* With no region head, the checkpoint callback logs `val_acc` (`val_plate_acc` is mapped to it automatically).
* The default augmentation (`train/data/augmentation.py`) already applies Affine rotate ±12°, shear, blur, noise,
  dropout and a ToGray branch. That fits our grayscale, slanted source. A custom pipeline can be passed with `--augmentation-path`
  (a YAML from `A.save(...)`); preview it with `fast-plate-ocr visualize-augmentation -d <imgs> --plate-config-file … -o`.
* **Train on the crops the runtime produces.** `training.harvest` saves `image_path` = the output of `prepare_plate`
  (de-shear + CLAHE, the same pixels the OCR sees at runtime). Keep `--image ocr`, the split.py default.

## 7. `anpr/ocr.py` compatibility: no changes needed

The exported model has input `input` uint8 `[N,64,128,3]` and output `plate` float `[N,10,37]`, is ir 8 / opset 15 (same as
deployed) and has no `region` output. The runtime `.venv` (onnxruntime 1.20.1) loaded it through `FastPlateOcr(OcrConfig(model_path,
config_path))` and read plates correctly. The exported config (no `plate_regions`) is accepted. Latency: 5.08 ms/plate vs.
5.24 ms deployed (1 thread, Intel Mac). The graph is the same size, so Pi latency should be unchanged.

## 8. Risks / open issues

* **11-character Delhi plates** (`DL 10 C AB 1234` → `DL10CAB1234`) are accepted by `anpr/validator.py` but cannot be represented
  with 10 slots. The dataset loader rejects labels longer than `max_plate_slots`. If the site sees such plates, set
  `max_plate_slots: 11` in both the training and deployed configs. That only re-initialises `token_reducer/query_tokens` (the rest transfers),
  and the ONNX output becomes `[N,11,37]`. `ocr.py` reads the slot count from the YAML, but this was not tested at runtime.
* Too little data or one site only → overfitting to one camera. Split by video/source (`--test-source`) as `split.py` does.
* Letters I/O/Q never occur in Indian series, so they get no fine-tune signal. The validator's repair map already handles O/0 and I/1.
* The onnx pin conflict (see §2) comes back if someone runs `pip install -U` in `.venv-train`.

## 9. Expected dataset size and runtime

* Useful minimum: about 500 verified crops from about 150 distinct vehicles (the harvester keeps a few diverse crops per track).
  Target: 2–5 k crops across day/night, with 2-line and slanted plates included. The model is small, so hundreds of
  in-domain samples are usually enough to move a fine-tune.
* Measured on this Intel Mac CPU: about 0.18 s/step at batch 16 (about 11 ms/image/epoch) after about 30 s of graph compilation.
  For 3 k crops × 150 epochs that is **about 80–90 min on this Mac**. Early stopping usually ends it sooner.
* Colab: **not measured.** The same run on a T4 should take minutes. `pip install "fast-plate-ocr[train,onnx]==1.1.0" "onnx==1.17.0"`
  there (Colab's TF version differs, so re-check that the onnx pin works). CPU training on the Mac is fast enough that Colab is optional.

## 10. Public Indian plate data (licence first)

| dataset | size / labels | licence | verdict |
|---|---|---|---|
| Kaggle `umar1103/final-licence` "Indian License Plate Images" https://www.kaggle.com/datasets/umar1103/final-licence | about 30 k plate crops, 728 MB; label = filename; mix of synthetic, real and "external sources" | CC BY 4.0 (per Kaggle metadata) | **Flag:** the licence permits commercial use with attribution, but the images come from unnamed external sources, so provenance is unclear. Usable for pre-fine-tune experiments after legal review. Inspect label quality first |
| Kaggle `kedarsai/indian-license-plates-with-labels` https://www.kaggle.com/datasets/kedarsai/indian-license-plates-with-labels | about 180 + 2 k images, **YOLO boxes, no text** | CC0 (per metadata), but the images were scraped from Google Images | Not an OCR dataset; provenance unclear. Skip |
| Indian_LPR / "Indian Licence Plate Dataset in the wild" (arXiv 2111.06054) https://github.com/sanchit2843/Indian_LPR | 16 192 images, 21 683 plates, 4-point + char labels | **not public** ("legalities") | Not available |
| Datacluster Labs https://github.com/datacluster-labs/Indian-Licence-Plate-Image-Dataset | about 6 k images, detection labels | commercial, contact sales | Paid; text labels unconfirmed |
| IEEE DataPort "VisionPlate ANPR" / "Indian LPR (YOLO & OCR)" | OCR crops + text | IEEE DataPort terms (403 when fetched) | Unclear; needs a subscription and licence check |

Synthetic generation: **worth it as a supplement, not a replacement.** Render HSRP-style plates (Charles Wright–style font,
1-line 500×120 and 2-line 340×200 layouts, blue `IND` strip, state-code/series/number grammar from `anpr/validator.py`),
then degrade them with the camera's look (grayscale, 704×576 scale, 30° shear then de-shear through `prepare_plate`, JPEG, blur).
Its main value is covering rare letters and 2-line layouts that our site footage lacks. Keep it ≤50 % of training rows and
validate only on real crops. `training/synth.py` is a first, simple version (OpenCV Hershey font, not the HSRP font).
