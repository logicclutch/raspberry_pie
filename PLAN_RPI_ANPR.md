# Raspberry Pi 3B+ ANPR (Indian plates): Plan

## Goal
Detect Indian number plates anywhere on a vehicle (front, rear, side, top, at an angle), read the characters,
and show a plate on the dashboard only if it is a valid Indian format. If the system isn't sure about a read,
it drops it. It should never show a wrong plate.

## Hardware limits (Pi 3B+)
- 4x Cortex-A53 at 1.4 GHz, 1 GB RAM, no usable GPU/NPU. Heavy frameworks won't fit: no PyTorch, no PaddleOCR, no EasyOCR.
- Target: 3–6 FPS detection at 320 px and about 30–60 ms of OCR per plate crop. That's enough for gates and parking entry/exit, not for highway speeds.
- Optional speed-up: a Coral USB TPU or Hailo module runs the detector 5–10x faster (same code path, different backend).

## What we reuse from `~/Desktop/Anpr 2/backend`
| Asset | Use |
|---|---|
| `detection/license_plate_yolov11n.pt` (2.59M params, 1 class) | Plate detector. Export to NCNN/TFLite INT8 at imgsz 320 |
| `indian_plate_yolo.pt` (YOLOv8n, `number_plate`) | Test it against the model above on Indian test images; keep the better one |
| `app/plugins/anpr/validator.py` (`PlateValidator`, `STATE_CODES`, BH-series, char fixes like O↔0, I↔1, B↔8) | Strict Indian format check. Port as-is |
| `app/plugins/anpr/fusion.py` (`TemporalFusion`) | Voting across frames. Port as-is |
| `app/plugins/anpr/tracker.py` | Tracks each vehicle so the same plate is only reported once |
| `providers/fastplate_provider.py` (fast-plate-ocr ONNX, `cct-xs-v1-global-model`) | OCR. Runs on the ONNX Runtime CPU build for ARM64 |

## Pipeline
```
Camera (Pi Cam / USB / RTSP) → frame 640x480
 → [1] Plate detector (YOLO11n NCNN INT8, 320px, conf ≥ 0.5)
 → [2] Crop + 8% padding → angle fix (4-point perspective warp from plate corners/contour)
      → CLAHE contrast boost → split two-line plates (bikes/trucks) into rows
 → [3] OCR (fast-plate-ocr ONNX) → text + confidence per character
 → [4] Strict validator: regex + real state/RTO code + BH series; position-aware char repair
 → [5] Temporal fusion per tracked vehicle: accept only if ≥3 agreeing frames,
      average confidence ≥ 0.85, and every character's confidence ≥ 0.6
 → [6] Emit a single event → SQLite → FastAPI + WebSocket → dashboard
```

## Indian formats accepted (strict mode)
- Standard: `SS NN X[XX] NNNN` (e.g. `MH12AB1234`). `SS` must be in `STATE_CODES`.
- BH series: `YY BH NNNN X[X]` (e.g. `22BH1234AB`).
- Legacy plates with no series letters (`DL1C...` variants) are behind a config flag and off by default.
- Anything else gets rejected. No lenient mode.

## Where plates can be on the vehicle
- Detector runs on the full frame, not only the lower part, so plates on the side, top, or at an angle are found.
- Fine-tune the detector on Indian data (Roboflow Indian LP sets + our own snapshots) with augmentation
  for rotation ±30°, perspective, blur, night/IR, and two-line bike plates. Train on the Mac or in Colab, export for the Pi.

## Dashboard
- FastAPI (small footprint, runs on the Pi) that serves one static HTML/JS page.
- Live table: plate, time, camera, confidence, crop image. Search and CSV export.
- Unsure reads never reach the dashboard. They go to a hidden "review" log for tuning.

## Milestones
1. **Setup (day 1):** Raspberry Pi OS 64-bit Lite, Python 3.11 venv, `onnxruntime`, `ncnn`, `opencv-python-headless`, `fastapi`.
2. **Model export (day 1–2):** export `.pt` → NCNN INT8 on the Mac. Benchmark on the Pi (FPS, RAM).
3. **Port core (day 2–3):** validator, fusion, tracker, OCR provider into `raspberry-main/anpr/`, with the PyTorch/CUDA parts removed.
4. **Accuracy harness (day 3–4):** labeled Indian test set of 300+ plates. Measure exact-match precision/recall.
   Release gate: **precision ≥ 99.5%** (wrong plates shown), recall as high as possible.
5. **Fine-tune (day 4–6)** detector and OCR (fast-plate-ocr supports custom training) on Indian plates if the gate fails.
6. **Dashboard + systemd service + watchdog (day 6–7).**
7. **Field test** at the real camera position. Tune the thresholds.

## Honest note on "no mistakes"
No camera ANPR reaches 100%: dirt, glare, and non-standard plates always exist. What we can do is make wrong
output nearly impossible: strict format checks, multi-frame agreement, and per-character confidence gates.
If the system isn't sure, it shows nothing. It never shows a guess.
