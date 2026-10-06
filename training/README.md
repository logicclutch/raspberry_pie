# Teach the plate reader Indian plates (OCR fine-tuning)

The Pi reads plates with a small OCR model (`models/ocr/plate_ocr.onnx`). It was trained on plates from many
countries, **not India**, so on the gate camera it mixes up **B/8, O/0, A/4, G/6** and sometimes drops the last
character. The fix is to show it a few thousand crops **from your own cameras**, with the correct text typed
by a person. This folder does the whole job. You only harvest, label and run one command.

Nothing here changes what the Pi runs. A new model is only used after it beats the current one on the test videos
**and** you change two lines in `config.yaml` yourself (step 6).

Everything runs on the Mac from the project folder `~/Desktop/raspberry-main`. Only step 4 can run on Colab.

---

## 1. Harvest crops from your camera recordings

Record normal traffic at the real gate: day, evening, night, rain. Include trucks, bikes and 2-line plates.
Then, for each recording:

```bash
.venv/bin/python -m training.harvest --config config.yaml --source-name gate_cam --video ~/Videos/gate-2026-10-01.mp4
```

* `--source-name` = which camera it came from (`gate_cam`, `exit_cam`, ...). Use the same name every time for
  the same camera.
* The video is run through the real Pi pipeline. A few different crops per vehicle are saved in
  `training/data/images/`, with the model's guess as a hint. The guess is **never** used as the answer.
* You can run it again on the same video. It never adds a crop twice and never touches your labels.
* **Do not harvest the two test videos** (`training/ground_truth/*.csv`). They are for judging the model, not for
  training. If you do anyway, `train.sh` keeps them out of training.

### 1b. Collect from a live push / ingest camera (the dataset builds itself)

When the camera POSTs images to the Pi (ingest mode, `anpr.ingest`), every plate the Pi confirms already
saves its OCR-ready crop and a full snapshot in the Pi's `data/` folder. To turn that real traffic into
labelable crops, copy the Pi's data over and import it:

```bash
rsync -a pi@<pi-ip>:/opt/anpr/data/ ./pi-data/          # the Pi's saved crops + anpr.db
.venv/bin/python -m training.import_events --db pi-data/anpr.db --images pi-data/images --source gvd_cam
```

Each event becomes one `unverified` row (the device's voted plate is kept only as a hint, never the answer).
Re-running adds nothing twice (dedup by event id). Then label them exactly like harvested crops (step 2).
So as the camera runs, the fine-tuning set grows on its own - you only label and fine-tune.

### (optional) synthetic Indian plates as a supplement

`python -m training.synth --data training/data --count 1500` adds valid Indian plates (auto-labelled) to
help the common India confusions (B/8, O/0, A/4, G/6). Keep them **at most about half** of the training
set and always pair them with **real labelled crops** - `train.sh` will not train on synthetic alone
(it needs real verified crops in both train and val).

## 2. Label the crops

```bash
.venv/bin/python -m training.labeler --data training/data
```

Open <http://127.0.0.1:8010>. For each crop, type the plate exactly as printed (letters and digits only, no
spaces) and press **Save**. **Apply to track** gives the same text to every crop of that vehicle. Press
**Unreadable** if a person cannot read it either, and **Skip** if unsure. Only saved (verified) crops are used.

How many you need (from `training/RESEARCH.md`):

| | minimum to try | target |
|---|---|---|
| labelled crops | 500 | **2,000 or more** |
| different vehicles | 150 | **500 or more** |
| 2-line plates (vehicles) | 50 | **150 or more** |

More vehicles beat more crops of the same vehicle. Harvest from several days and times of day.

Optional: add synthetic plates for rare letters and 2-line layouts. They are drawn, not photographed, so keep
them to **at most half** of the training crops:

```bash
.venv/bin/python -m training.synth --data training/data --count 1000
```

## 3 + 4 + 5. Split, train and test: one command

```bash
training/train.sh --name gate-v1
```

That's the whole run. It:

1. downloads the current model's training checkpoint (checked with sha256),
2. **splits** your verified crops by vehicle into train / val / test (the same vehicle is never in two parts).
   Crops from the test videos in `training/ground_truth/` always go to test,
3. **fine-tunes** the current model on your crops. It starts from what the model already knows, so it
   needs far less data than training from zero,
4. converts it to the Pi's format (ONNX),
5. saves the candidate to `models/ocr/candidates/gate-v1/` (`plate_ocr.onnx`, `plate_ocr_config.yaml`,
   `train_info.json` with what it was trained on, `training_log.csv`),
6. checks that the Pi's code can load it,
7. **tests it against the current model** (`python -m training.evaluate`, see below) and ends with PASS or FAIL.

Time on this Mac is about 80 to 90 minutes for 3,000 crops (it stops early once it stops improving). Everything
from the run is in `training/runs/gate-v1/` (log: `train.log`).

Useful options: `--epochs 150` (maximum), `--batch-size 64`, `--data <folder>`, `--test-source <camera>` or
`--test-video <file>` (which footage is held out for testing, instead of the default), `--skip-eval`,
`--force` (reuse a name).

### Train on Google Colab instead (optional, faster with a GPU)

Upload the project folder **with** `training/data/` to Google Drive, open a Colab notebook with a GPU runtime and run:

```python
from google.colab import drive; drive.mount('/content/drive')
%cd /content/drive/MyDrive/raspberry-main
!pip install -q -r training/requirements-train.txt
!bash training/train.sh --name gate-v1 --python python
```

Colab has no Pi runtime or test videos, so it stops after step 5. Download
`models/ocr/candidates/gate-v1/` into the same place on the Mac, then run the test there:

```bash
.venv/bin/python -m training.evaluate --candidate models/ocr/candidates/gate-v1
```

(The Colab route has not been tried yet. If the ONNX export fails there, run `!pip install "onnx==1.17.0"` and retry.)

### What the test checks

`training.evaluate` plays each test video through the full Pi pipeline twice, once with the current model and
once with the candidate, and compares the plates each one would have saved with the true plates in
`training/ground_truth/*.csv`. It also reads each held-out labelled crop with both models.

The candidate **passes** only if **all** of these hold:

* it shows **no more wrong plates** than the current model (a wrong plate is the worst error),
* it reads **at least as many plates correctly**,
* on the held-out crops, its accuracy is **not lower**,
* it is **better in at least one** of these,
* there are at least 100 held-out crops,
* it was **not trained on the test videos**.

The report is saved as `models/ocr/candidates/<name>/evaluation.md`. It also lists which characters each model
confuses (for example `B>8 x3` means "B read as 8, three times"; `7>-` means "7 dropped").

The test videos' true plates were read by eye. Plates marked `unsure` are not scored. Correct them in
`training/ground_truth/gate_cam.csv` if you know the real plate. More test videos with checked plates make
the test stronger: add a CSV in the same format.

## 6. Use the new model (only after PASS)

1. Copy the project to the Pi as usual (`deploy/install_pi.sh`; `models/ocr/candidates/` is copied with it).
2. On the Pi, edit `/opt/anpr/config.yaml` (on the Mac: `config.yaml`) and change the two `ocr` lines:

   ```yaml
   ocr:
     model_path: models/ocr/candidates/gate-v1/plate_ocr.onnx
     config_path: models/ocr/candidates/gate-v1/plate_ocr_config.yaml
   ```

3. Restart: `sudo systemctl restart anpr-engine anpr-web`
4. Watch the dashboard for a day. The first plates should look right, and `journalctl -u anpr-engine -f`
   shows no OCR errors.

**Roll back** (any time): put the two lines back to

```yaml
ocr:
  model_path: models/ocr/plate_ocr.onnx
  config_path: models/ocr/plate_ocr_config.yaml
```

and restart again. The original model is never overwritten, so rolling back is always safe.

---

Files: `train.sh` (the one command), `evaluate.py` (the test), `harvest.py`, `labeler/`, `split.py`, `synth.py`,
`configs/` (model settings: do not edit), `ground_truth/` (true plates of the test videos),
`fetch_pretrained.sh`, `requirements-train.txt` (training packages, only for `.venv-train` or Colab, never
the Pi's `.venv`), `RESEARCH.md` (why it is set up this way, measured numbers, risks).
