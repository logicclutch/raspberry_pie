# Raspberry Pi 3B+ ANPR: How It Works, Step by Step

One camera. Indian number plates only. Strict rule: **if the system isn't sure, it shows nothing.**

There are two workflows:
- **A. Build and training.** Runs on a Mac or in Colab, once. It produces the model files.
- **B. Live system.** Runs on the Pi, all the time. It turns camera frames into plates on the dashboard.

---

## Tech stack

| Layer | Technology | Why |
|---|---|---|
| OS | Raspberry Pi OS Lite **64-bit** (Bookworm) | 64-bit is needed for the ncnn and onnxruntime packages. Lite has no desktop, which saves RAM |
| Language | Python 3.11 | Standard on the Pi, and all the libraries support it |
| Camera | **rpicam-vid** (Pi Camera Module 3, raw frames piped into Python) or **OpenCV** (USB/RTSP, and webcams/video files on the Mac) | Fixed fast shutter so plates aren't blurred. Using the CLI instead of the Picamera2 library avoids a NumPy version clash with our locked venv |
| Image processing | OpenCV + NumPy | Crop, straighten, contrast boost, motion check |
| Plate detector runtime | **ncnn** | Fastest YOLO runtime on a Pi's ARM processor |
| OCR runtime | **ONNX Runtime** (CPU) | Runs fast-plate-ocr models |
| Database | **SQLite** (WAL mode) | No server needed and very light. Snapshots are stored as JPEG files |
| Backend API | **FastAPI + Uvicorn** (1 worker) | REST, plus WebSocket for live updates |
| Dashboard | Plain HTML + CSS + JavaScript, served by FastAPI | No build step and no Node.js on the Pi |
| Service | **systemd** (auto-start, auto-restart) + journald logs | Survives reboots and crashes |
| Tests | pytest | Strict tests for the plate checker and an accuracy harness |
| **Training (Mac/Colab only)** | Ultralytics YOLO (PyTorch), fast-plate-ocr trainer, Label Studio/Roboflow for labeling | Never installed on the Pi |

## Models

| Job | Model | Size | Runs as |
|---|---|---|---|
| Find the plate | **YOLO11n**, 1 class `number_plate`, fine-tuned on Indian plates | ~2.6M params | ncnn, 320×320 input (416 if plates are far away) |
| Read the characters | **fast-plate-ocr (CCT-XS)**, fine-tuned on Indian plates (1-line and 2-line) | small, CPU-friendly | ONNX Runtime |
| Backup OCR (only if the one above fails the accuracy gate) | **LPRNet** trained on Indian plates | ~0.5M params | ONNX Runtime |

---

## A. Build and training workflow (Mac / Colab, one time)

```
A1 Collect images → A2 Label → A3 Train detector → A4 Train OCR → A5 Export → A6 Accuracy gate → A7 Copy to Pi
```

| Step | What happens | Tool | Output |
|---|---|---|---|
| **A1 Collect** | Public Indian plate datasets **plus footage from your own camera at the real install spot** (day, night, rain, bikes, trucks, angled plates) | Pi camera, Kaggle/Roboflow sets | Raw images |
| **A2 Label** | Draw boxes around plates (for the detector). Type the plate text for each crop (for OCR) | Label Studio / Roboflow | `boxes/` and `plates.csv` |
| **A3 Train detector** | Fine-tune YOLO11n at imgsz 320. Augment with rotation ±30°, perspective, blur, low light, glare | Ultralytics | `plate_det.pt` |
| **A4 Train OCR** | Fine-tune fast-plate-ocr. Charset 0-9 A-Z, max 10 characters, 1-line and 2-line plates. **How:** [training/README.md](training/README.md): harvest, label, `training/train.sh`, which also exports and gates the result against the live model | fast-plate-ocr | `models/ocr/candidates/<run>/plate_ocr.onnx` |
| **A5 Export** | Detector `.pt` → ncnn (`yolo export format=ncnn imgsz=320`). Try INT8 later if more speed is needed | Ultralytics / ncnn tools | `plate_det_ncnn/` |
| **A6 Accuracy gate** | Run the full pipeline on a **held-out** test set of 300+ real plates. **Precision ≥ 99.5%** (wrong plates shown ÷ all plates shown ≤ 0.5%). Recall is reported | pytest harness | Pass → deploy. Fail → more data, go back to A1 |
| **A7 Deploy** | Copy models to the Pi and run the speed benchmark | scp + benchmark script | FPS and RAM report |

---

## B. Live workflow on the Pi (every frame)

```
 ┌────────────┐   ┌───────────┐   ┌──────────────┐   ┌───────────┐   ┌──────────────┐
 │1 Capture   │ → │2 Motion   │ → │3 Detect plate│ → │4 Track    │ → │5 Crop + fix  │
 │ rpicam-vid │   │  gate     │   │ YOLO11n/ncnn │   │ IoU track │   │ angle + CLAHE│
 └────────────┘   └───────────┘   └──────────────┘   └───────────┘   └──────┬───────┘
                                                                            ↓
 ┌────────────┐   ┌───────────┐   ┌──────────────┐   ┌───────────┐   ┌──────────────┐
 │10 Dashboard│ ← │9 API +    │ ← │8 Save event  │ ← │7 Vote over│ ← │6 OCR + strict│
 │  browser   │   │ WebSocket │   │ SQLite + JPG │   │  frames   │   │ Indian check │
 └────────────┘   └───────────┘   └──────────────┘   └───────────┘   └──────────────┘
```

### Step 1: Capture
- **Tech:** rpicam-vid (or OpenCV for USB/RTSP/webcam/video file), running in its own thread that always keeps **only the latest frame**, so the system never falls behind. For an IP camera's main stream on the Pi, `camera.backend: ffmpeg` decodes with the Pi's hardware H.264 decoder and drops/shrinks frames inside ffmpeg (`max_fps`, `output_width`); see `docs/CAMERA_SETUP.md`.
- **Settings:** 1280×720 capture, fixed fast shutter (about 1/1000 s) to avoid motion blur, IR camera or light for night.
- **Output:** the latest full-resolution frame.

### Step 2: Motion gate
- **Tech:** OpenCV frame difference on a small 160×120 grayscale copy.
- **Why:** the detector is the slowest part. If nothing moves, it's skipped and the CPU stays free and cool.
- **Output:** "run detector" yes/no. Cost ≈ 2–3 ms.

### Step 3: Detect the plate
- **Model:** YOLO11n (Indian-trained) on **ncnn**, 320×320 input, confidence ≥ 0.5, NMS.
- **It looks at the whole frame**, so plates on the front, back, side or top, or at an angle, are all found.
- **Output:** plate boxes. Estimated 200–400 ms per frame on the 3B+ (to be measured in A7).

### Step 4: Track
- **Tech:** a light IoU tracker (pure NumPy). Each vehicle's plate gets a **track ID**.
- **Why:** the same car appears in many frames. The track lets us vote (step 7) and report it **once**.

### Step 5: Crop and fix the image
- **Tech:** OpenCV.
  1. Crop from the **full-resolution** frame (not the 320 px copy) with a little padding, which gives sharper characters.
  2. **Quality gate:** skip crops that are too small (< ~80 px wide) or blurry (low Laplacian variance). Better to wait for a better frame.
  3. Straighten angled plates (perspective warp from the plate's corners).
  4. CLAHE contrast boost for glare, shadow and night.
- **Output:** a clean, straight plate image resized to the OCR model's input size.

### Step 6: OCR and strict Indian check
- **Model:** fast-plate-ocr (Indian-trained) on ONNX Runtime. Estimated 30–60 ms per plate.
- **Output from the model:** text plus a **confidence for every character**.
- **Strict validator (plain Python, fully unit-tested):**
  1. Uppercase, remove spaces, dots, dashes and the `IND` mark.
  2. **Position-aware repair only:** O→0, I→1, B→8, S→5 only where a digit must be, and the reverse only where a letter must be.
  3. Must match exactly one format:
     - Standard: `SS NN X[X][X] NNNN` → e.g. `MH12AB1234`
     - BH series: `YY BH NNNN X[X]` → e.g. `22BH1234AB`
  4. `SS` must be a **real state/UT code** (MH, DL, KA, TN, …).
  5. Every character's confidence ≥ 0.6.
- If any check fails, the read is dropped (logged for review only).

### Step 7: Vote across frames
- For each track, collect the validated reads.
- **Accept only when** the same plate text appears in **≥ 3 frames**, average confidence ≥ 0.85, and no competing reading comes close.
- Same plate seen again within N seconds → no duplicate event.
- **Output:** one final, confirmed plate per vehicle.

### Step 8: Save
- **Tech:** SQLite (WAL mode): plate, time, confidence, track ID, image paths.
- Save the plate crop and a small vehicle snapshot as JPEG.
- Auto-delete old images after X days so the SD card doesn't fill up.

### Step 9: API
- **Tech:** FastAPI + Uvicorn.
  - `GET /api/v1/plates?q=&from=&to=&page=`: search and history
  - `GET /api/v1/plates/export.csv`: export
  - `WS /ws/plates`: push new plates live
  - `GET /health`: camera OK, FPS, CPU temperature, disk space

### Step 10: Dashboard
- **Tech:** one HTML page with plain JS and CSS, opened from any phone or PC on the same network (`http://<pi-ip>:8000`).
- Shows a live list of confirmed plates (plate in Indian style, time, crop image, confidence), search by plate or date, CSV export, and a system-health strip (FPS, temperature, camera status).

---

## How it runs on the Pi (processes)

| Process | Does | Started by |
|---|---|---|
| `anpr-engine` | Steps 1–8 (capture thread + inference loop) | systemd, auto-restart |
| `anpr-web` | Steps 9–10 (FastAPI + dashboard) | systemd, auto-restart |

The two talk through SQLite, plus a local notify call, so a crash in one doesn't take down the other.

**Hardware notes:** use a heatsink and fan (the 3B+ slows itself down when hot), the official 5V/2.5A power supply, and a good A1/A2 SD card.

## Honest limits
- Estimated speed: about 2–5 detection frames per second. Good for gates, parking and slow lanes; not for highways.
- No ANPR is 100% right. This design makes wrong plates on screen **very rare** by showing nothing when it isn't sure.
- **Camera placement matters as much as the model:** 3–6 m away, under 30° angle, plate at least 80 px wide in the image.
