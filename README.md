# Raspberry Pi 3B+ ANPR (Indian number plates)

One camera → plate detector (YOLO11n, ncnn) → plate reader (fast-plate-ocr, ONNX) → strict Indian
format check → 3-frame vote → SQLite + dashboard. See `WORKFLOW.md` for every step and `PLAN_RPI_ANPR.md`
for the plan.

## Develop and test on the Mac

The Mac venv uses the **same locked package versions** as the Pi (`requirements.lock`), so code that
works here will work there.

```bash
source .venv/bin/activate
pytest                                   # unit tests
ruff check anpr tests scripts            # lint
./scripts/run_dev.sh                     # webcam + dashboard at http://127.0.0.1:8000
./scripts/run_dev.sh clip.mp4            # a recorded video instead
python -m anpr.engine --source clip.mp4 --no-realtime --exit-at-end   # every frame, no dashboard
```

## Changing the camera input from the dashboard

Click **Add stream** (top right) and enter either a **video file path** (full path on the station,
e.g. `/Users/you/Videos/gate.mp4`) or an **RTSP link** (`rtsp://user:password@192.168.1.64:554/...`).
The engine switches within about a second, without a restart. **Use default camera** goes back to
`camera.source` in `config.yaml`.

- The choice is saved in the database, so the Pi reopens the same stream after a restart or reboot.
  `--source` on the command line overrides it for that run.
- When a video file ends, the engine waits for the next stream instead of exiting. Submitting the same
  path again replays it.
- RTSP passwords are stored on the station and shown as `***` everywhere (dashboard, API, logs).
- Viewing needs `web.auth_token` (when set). Changing things (Activate licence, Settings, Add stream)
  also needs the admin password `web.admin_token`, which the Pi installer creates and prints. Without
  any password, a licence can only be activated from the device itself.

**Live view.** The big panel opens on **Live**: the picture the engine is reading right now, updated
about once a second, with the plates it has found drawn on it (amber = still reading, green + text =
confirmed). Use this to check that a new video or camera is really coming through, even when no plate
has been confirmed yet. The badge in the corner shows **LIVE** and the frame rate, or why there is no
picture: *Video ended*, *No signal*, *Can't open*, *Engine off* or *Waiting for picture*
(*Restart needed* means the web/engine services still run code from before the live view: restart
both). The last picture stays on screen, greyed out, once it stops updating. **Last vehicle** shows
the snapshot of the last confirmed plate, as before. The dashboard only asks for pictures while the tab
is open on Live.

- The engine writes one small JPEG every `runtime.preview_interval_s` seconds (default `1.0`; `0`
  switches it off), `runtime.preview_width` pixels wide (640) at JPEG quality
  `runtime.preview_quality` (70), replaced in place, never kept or purged. On the Pi (Linux) it lives in
  RAM (`/dev/shm/anpr-<uid>-<id>/live.jpg`), because ~60 KB a second is ~5 GB a day that would wear out
  the SD card; the engine and web must run as the same user (the systemd units do). On a Mac it is
  `data/live.jpg`, next to `data/images`. It is deleted when the stream changes, so the old stream's
  picture never passes for the new one. Cost: about 2.6 ms per picture on a Mac for the 704x576 toll
  camera (expect several times that on the Pi 3B+, once a second).
- API: `GET /api/v1/live.jpg` (same token as the rest of the API) returns it with `Cache-Control:
  no-store` and `X-Captured-At` (unix time it was written); 404 when there is none yet.
  `/api/v1/source` and `/health` report `preview_age_s` and `preview_fresh`.

Accuracy gate (must pass before deploying, 300+ held-out Indian plates):

```bash
python scripts/eval_accuracy.py --images data/test/labels.csv    # image,plate
python scripts/eval_accuracy.py --videos data/test/videos.csv    # video,plate1;plate2
```

## Build an OCR training set from your own cameras (Mac only)

```bash
# 1. harvest crops (real detector/tracker/prepare_plate/OCR; OCR text is only a suggestion)
.venv/bin/python -m training.harvest --config config.yaml --video gate.mp4 --source-name gate_cam --out training/data
# 2. label them by hand at http://127.0.0.1:8010 (localhost only)
.venv/bin/python -m training.labeler --data training/data
# 3. split verified crops by vehicle (training/train.sh does this for you; shown for manual use)
.venv/bin/python -m training.split --data training/data --test-source phone
```

`training/data/labels.csv` holds one row per crop (`image_path, raw_image_path, source, video, frame, track,
box, width_px, two_line, prep, sharpness, ocr_text, ocr_conf, ocr_valid, ocr_plate, plate_text, status, split,
labeler, labeled_at`). Re-running the harvest never duplicates rows or touches human labels.

## Fine-tune the OCR for Indian plates

After labelling (above), one command trains a new plate reader from your crops and tests it against the current
one on the test videos: `training/train.sh --name gate-v1`. The live model is never replaced automatically.
If it passes, switch `ocr.model_path` / `ocr.config_path` in `config.yaml` to the candidate. Step by step, with
data targets and rollback: [training/README.md](training/README.md).

## HSRP / non-HSRP

Each saved plate gets `hsrp`, `non_hsrp`, or `not sure: <short reason>`. With `hsrp.method: model` (the
default) a small trained image classifier decides (`models/hsrp/hsrp_cls.onnx`, MobileNetV3-small, about 6 MB,
~200 ms per plate on a Pi 3B+). It was trained on the Kaggle "Indian License Plate Dataset" (905 HSRP / non-HSRP
photos) with labels checked by eye; decorative plates with a European-style blue "IND" strip count as
non-HSRP (they have no chakra hologram). Measured on 532 photos each model never trained on (group 5-fold):
HSRP 279/281 (99.3%), non-HSRP 242/251 (96.4%), overall 97.9%. On 69 truck plates (an independent set,
`tests/fixtures/truck_plates`): non-HSRP 56/56, HSRP only 5/13 - small, dirty truck holograms are its weak spot.
`hsrp.two_line_marks` (on by default) adds a second opinion for 2-line plates: when the model does not say HSRP
but the hand-written check finds both the hologram and "IND", the plate is HSRP. Trucks: HSRP 6/13, non-HSRP
still 56/56; no Kaggle non-HSRP plate changes (the font check alone is not used: neat old printed plates pass
it). Camera-push (ingest) mode adds a **second plate check** (`ingest.second_plate_check`, on): trucks often carry an
HSRP on the grille and an old painted copy on the bumper, and the detector may only see the painted one. When a
vehicle comes out not HSRP, a zoomed look searches its first images for another plate of the same vehicle (near it,
about its size, same number - up to 2 letters may differ, never a digit); if that plate is HSRP the vehicle is saved as
`hsrp`, and a confirmed, clearly surer reading of it replaces the number. On 1,106 labelled photos it changed only
the two photos of one such truck (both to the right number and `hsrp`), no non-HSRP vehicle.
`hsrp.method: rules` falls back to the hand-written hologram / "IND" checks. A camera can only tell what a plate *looks like*: a good copy of an HSRP passes.

## Raspberry Pi 3B+ limits

Every service is capped at 750 MB (`MemoryMax`); the engine and the ingest receiver stop cleanly and restart
above `runtime.max_rss_mb` (680 MB) instead of being killed mid-write. Big pictures (phone photos, 4K
snapshots) are decoded at reduced size, at most 1920 px on the long side. Under overload the ingest receiver
answers `503` so the camera retries, instead of queueing without limit. Two cameras run in one process
(`cameras:` in `config.yaml`) and share one HSRP classifier.

The ingest and headless packages install the dashboard but leave it **off** (`web.enabled: false`, about 60 MB
of RAM saved). The client turns it on or off at any time, and the choice survives reboots and upgrades:
`sudo /opt/anpr/bin/anpr dashboard on` / `off` / `status`.

## Deliver to a client (no source code): the `.deb` installer

For clients who should **not** get the source, build a Debian package on the Mac and send only that one
file. It carries the app as **compiled bytecode only** (no `.py`), the arm64 Python environment, and the
models — so the Pi needs no pip, no internet (beyond a few standard apt packages), and no source.

```bash
./deploy/build_deb.sh            # needs Docker Desktop running; builds dist/anpr_<ver>_model_arm64.deb
HEADLESS=1 ./deploy/build_deb.sh # slim, no dashboard: plates -> client API only; supports two cameras
```

The **headless** build (`anpr_<ver>_headless_model_arm64.deb`) has no dashboard or open port: the engine reads
plates and POSTs them to the client's API (set in `config.yaml` `push:`). It is smaller (~63 MB vs 91 MB,
~255 MB vs 382 MB installed), frees RAM, and supports **two cameras on one Pi** (set `cameras:` in the
config; each plate's JSON gets a `camera` field). Licence is activated by command line or by pre-loading
`licence.json`. Suited to a quiet/medium gate — test two cameras on the real Pi first.

The client installs it with one command and gets auto-start, a settings file that survives updates, and
the licence/activation flow — see the 1-page [docs/INSTALL_CLIENT.md](docs/INSTALL_CLIENT.md). Send them
the `.deb` and that guide. `tools/` (licence tool), `training/` and the dev config never go in the package.

## Deploy to the Raspberry Pi 3B+ (from source, e.g. your own Pis)

1. Flash **Raspberry Pi OS Lite 64-bit (Bookworm)**. Fit a heatsink/fan and use the official 5 V 2.5 A supply.
2. Copy this folder to the Pi (e.g. `rsync -a --exclude .venv --exclude .venv-train --exclude data --exclude training --exclude tools --exclude licence.json ./ pi@<ip>:anpr-src/`).
   `training/` (dataset tools and harvested crops) and `tools/` (the vendor's licence tool) never go to the Pi.
3. On the Pi: `cd anpr-src && sudo ./deploy/install_pi.sh`
4. Edit `/opt/anpr/config.yaml`: set `camera.source: picamera` (or `rtsp://...`) and `web.auth_token`,
   then `sudo systemctl restart anpr-engine anpr-web`. (The RTSP camera can also be set later with
   **Add stream** on the dashboard.)
   **IP camera:** follow [docs/CAMERA_SETUP.md](docs/CAMERA_SETUP.md) on site — main stream, H.264,
   10–15 fps, 1/1000 s shutter, plate size — and on the Pi set `camera.backend: ffmpeg` (hardware
   H.264 decoding), `max_fps: 10`, `output_width: 1280`. Check a stream with
   `.venv/bin/python -m scripts.probe_camera '<rtsp link>' --guess-main`.
5. **Licence:** the engine runs only with a valid licence for this Pi. Open the dashboard: it shows the
   machine ID (send it to LogicClutch) and a box for the licence key. Paste the key, press Activate, and
   the engine starts within seconds ([docs/LICENSING.md](docs/LICENSING.md)).
6. Check: `sudo /opt/anpr/deploy/pi_check.sh`, logs with `journalctl -u anpr-engine -f`.

Mac vs Pi differences are only in `config.yaml` (camera source); the code and packages are identical.
