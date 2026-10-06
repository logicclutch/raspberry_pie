# Camera setup at the gate (installer guide)

This guide is for the person setting up the IP camera and the Raspberry Pi 3B+ on site. Follow it
from top to bottom, then check the result with the probe tool and the dashboard.

## Why this matters

We checked a recording of today's gate stream (`.../avstream/channel=1/stream=1.sdp`):

| What                 | Today                         | Target                               |
|----------------------|-------------------------------|--------------------------------------|
| Stream               | sub-stream (`stream=1`)       | **main stream** (`stream=0`)         |
| Codec                | **H.265 (HEVC)**              | **H.264**                            |
| Resolution           | 704 x 576                     | 1280 x 720 (or 1920 x 1080)          |
| Frame rate           | 5 fps                         | 10–15 fps                            |
| Bitrate              | 0.5 Mbit/s                    | 2–4 Mbit/s, constant (CBR)           |
| Keyframe interval    | every 2 s                     | every 1 s (= the frame rate)         |
| 1-line plate width   | ~92 px (median)               | **120 px or more**                   |
| 2-line plate width   | ~58 px (median)               | **80 px or more**                    |
| Plate text slant     | ~36°                          | under 20°                            |

Small plates, few frames per vehicle and motion blur are why only 13 of 22 vehicles were read
(none were read wrong: the system shows nothing when it is not sure). The camera settings below
fix most of this. The software side is ready: the Pi reads the main stream with its hardware
H.264 decoder (`camera.backend: ffmpeg`).

## 1. Use the MAIN stream

IP cameras send two streams: a sharp **main** stream and a small **sub** stream for phone apps.
Always use the main stream. Common address patterns (check the camera's manual if yours differs):

| Camera address style                                   | Main stream        | Sub stream (don't use) |
|--------------------------------------------------------|--------------------|------------------------|
| `rtsp://…:554/avstream/channel=1/stream=…sdp` (this gate) | `stream=0.sdp`     | `stream=1.sdp`         |
| Hikvision `rtsp://…:554/Streaming/Channels/…`          | `101`              | `102`                  |
| Dahua `rtsp://…/cam/realmonitor?channel=1&subtype=…`   | `subtype=0`        | `subtype=1`            |

For this gate the new link is:

```
rtsp://USER:PASSWORD@192.168.5.53:554/avstream/channel=1/stream=0.sdp
```

The probe tool can try the main stream for you (`--guess-main`, see step 4).

## 2. Camera settings (camera web page, main stream)

Open the camera's web page (type `http://192.168.5.53` in a browser). **Menu and setting names
differ between brands and firmware versions**: the names below (in *italics* or quotes) are the
*typical* ones, not the exact words your camera uses. Look for the setting with the same meaning.
The video settings are usually under something like *Video* / *Encoding* / *Stream* →
**Main stream**:

| Setting                          | Set to                                                     |
|----------------------------------|------------------------------------------------------------|
| Video encoding / codec           | **H.264** (not H.265, H.265+, H.264+, "Smart codec")        |
| Profile                          | Main or High                                                |
| Resolution                       | **1280 x 720** (best for the Pi 3B+), or 1920 x 1080        |
| Frame rate                       | **12** (anything from 10 to 15)                             |
| Bitrate type                     | **CBR** (constant)                                          |
| Bitrate                          | **2 Mbit/s** at 1280 x 720, **4 Mbit/s** at 1920 x 1080     |
| I-frame / GOP / keyframe interval| **same number as the frame rate** (12 → one keyframe a second) |
| Smart codec / ROI / "+" modes    | Off                                                         |

Why H.264: the Raspberry Pi 3B+ has a hardware decoder for **H.264 only**. H.265 has to be
decoded by the CPU; for a 1280x720 or 1920x1080 stream that leaves no time for reading plates.
If the camera offers only H.265 on the main stream, tell us: the engine still runs (software
decoding, with a warning in the log) but the Pi will fall behind.

Then the picture settings, typically under *Image* / *Exposure* / *Camera* (some cameras show
the shutter as "exposure time" in µs or ms: 1/1000 s = 1000 µs = 1 ms):

| Setting                                | Set to                                                        |
|----------------------------------------|---------------------------------------------------------------|
| Shutter / exposure time                | **1/1000 s or faster** (1/1000, 1/2000). Never "Auto" alone.   |
| "Max shutter" / "Shutter limit"        | 1/1000 s if the camera only offers a limit                     |
| "Slow shutter" / "Low-light shutter"   | **Off**                                                        |
| Exposure mode                          | Manual or *shutter priority*; let gain (AGC) go up instead     |
| Day/Night                              | Auto, **IR on** at night                                       |
| WDR                                    | Try on if headlights wash out plates; check moving plates do not get double edges (then use HLC/BLC instead) |
| Noise reduction (3D DNR)               | Low: strong DNR smears moving plates                           |
| Sharpness                              | Default (about 50 %)                                           |

A fast shutter is the only cure for motion blur: at 1/100 s a car at 15 km/h moves about 4 cm
while the picture is taken, at 1/1000 s only 4 mm. The picture gets darker; that is fine, the
plate is reflective and the IR light helps at night.

## 3. Aim and zoom

- **Zoom** so that a 1-line plate is **at least 120 px wide** and a 2-line plate **at least 80 px**
  in the 1280-wide picture. Rule of thumb: the picture should show about **one lane width
  (3–3.5 m)** across at the place where plates are read.
- **Angle:** the plate text should be tilted **less than 20°** in the picture. Today it is ~36°
  because the camera looks at the lane from the side. Move or turn the camera so it looks more
  along the lane (in front of the vehicles), 3–6 m from where plates are read, mounted about
  1–2.5 m high, looking down less than 30°.
- **Focus** on the plate position (not the background), in daylight **and** check again at night
  with IR (IR can shift focus on cheap lenses).
- **On-screen text** (date/time, camera name): move it to a corner where plates never pass.
- **Camera clock:** set NTP (to the Pi or the router) and the time zone to India (UTC+5:30). The
  Pi timestamps the reads itself, but a correct camera clock makes the recordings match.

## 4. Check the stream with the probe tool

On the Pi (or on a Mac with the project and ffmpeg installed):

```bash
cd /opt/anpr
sudo -u anpr .venv/bin/python -m scripts.probe_camera \
    'rtsp://USER:PASSWORD@192.168.5.53:554/avstream/channel=1/stream=1.sdp' \
    --guess-main --seconds 30 --save-dir /tmp/probe
```

Plate widths are measured in the picture the engine will see: wider streams are shrunk to
1280 wide first (`--output-width`, default `camera.output_width` from `config.yaml`, else 1280;
`--output-width 0` = the stream's own size). The probe decodes in software and competes with the
running engine for the CPU; that is fine for a check (frames are handled one at a time, so a long
run needs no extra memory).

It prints, for each address (password shown as `***`): codec, resolution, frame rate (declared and
measured), bitrate, keyframe interval, read errors, and — using the real plate detector on a few
frames per second — plate widths in pixels, crop sharpness and text slant, followed by plain hints.
Drive a car past the camera while it runs (or use `--seconds 60`). `--save-dir` keeps a few frames
(with the plate boxes drawn) and plate crops to look at: letters must have clean edges, not smears.

A good result looks like:

```
== rtsp://USER:***@192.168.5.53:554/avstream/channel=1/stream=0.sdp
   codec            h264
   resolution       1280x720
   fps              declared 12.0, measured 12.0 (360 frames)
   bitrate          2010 kbit/s
   keyframe every   12 frames, 1.0 s
   read errors      0
   plates           14 boxes in 60 sampled frames
     1-line width   min 118 / median 150 / max 190 px (target >= 120, 10 boxes)
     2-line width   min 84 / median 96 / max 110 px (target >= 80, 4 boxes)
   ...
   -> looks good for plate reading
```

Today's recording gives (hints part): *codec is HEVC: set the camera to H.264*, *704x576 is low: this
is probably the SUB-stream*, *5.0 fps is too few frames per vehicle*, *1-line plates are 92 px wide*,
*2-line plates are 58 px wide*, *plate text is slanted 36 deg*.

## 5. Settings on the Pi (`/opt/anpr/config.yaml`)

```yaml
camera:
  source: rtsp://USER:PASSWORD@192.168.5.53:554/avstream/channel=1/stream=0.sdp
  backend: ffmpeg       # system ffmpeg + the Pi's hardware H.264 decoder
  hw_decode: auto       # h264_v4l2m2m on the Pi; falls back to software by itself if it fails
  max_fps: 10           # at most 10 pictures a second reach the plate reader; the rest are dropped early
  output_width: 1280    # a 1920x1080 stream is shrunk to 1280x720 inside ffmpeg (no effect at 720p)
crop:
  min_width_px: 60      # was 40 for the small sub-stream; with the main stream plates are bigger
```

Then restart: `sudo systemctl restart anpr-engine`.

Notes:

- **The dashboard wins over `camera.source`.** If a stream was picked with **Add stream** on the
  dashboard, that one is used after every restart. Either enter the new main-stream link with
  **Add stream**, or press **Use default camera** after editing `config.yaml`. `backend`,
  `max_fps` etc. from `config.yaml` apply to dashboard streams too.
- `install_pi.sh` installs ffmpeg and prints whether the hardware decoder (`h264_v4l2m2m`,
  `/dev/video10`) is usable by the `anpr` user (it is in the `video` group; the engine service has
  `SupplementaryGroups=video render`). `sudo /opt/anpr/deploy/pi_check.sh` checks it again.
- The Pi 3B+ can only process a few pictures a second (the dashboard shows the actual rate); a
  10–15 fps camera makes sure each of those is a fresh picture. If the dashboard shows the engine
  steadily far below `max_fps`, lowering `max_fps` (e.g. 6) saves a little more CPU.
- ffmpeg runs as a child of the engine, so the RTSP link (with password) is visible in the Pi's
  process list to local users — like `config.yaml` itself. Logs and the dashboard never show it.

## 6. Check it works

1. `journalctl -u anpr-engine -f` should show, once per (re)connect:
   `camera rtsp://USER:***@192.168.5.53:554/…/stream=0.sdp: h264 1280x720 @ 12.0 fps -> 1280x720 BGR, v4l2m2m decoding, max 10 fps`
   - `software decoding` + "*hevc is decoded in software*" → the camera is still on H.265.
   - `hardware decoding gave no picture … trying software` → the `anpr` user can't use
     `/dev/video10` (group `video`), or ffmpeg lacks `h264_v4l2m2m`; run `pi_check.sh`.
   - `not reachable, retry` → wrong link, user or password, or network (the engine keeps retrying).
   - `no picture from ffmpeg for 10 s … restarting it` now and then → the camera stalls or the
     network drops packets; check the cable/switch and the `read errors` of `probe_camera`.
2. Dashboard → **Live**: the badge shows **LIVE**, the picture is the new sharp stream, and passing
   plates get amber boxes that turn green with the plate text once confirmed.
3. Let a few vehicles pass (day **and** night) and compare the dashboard list with what passed.

## Quick troubleshooting

| Symptom                                    | Fix                                                                 |
|--------------------------------------------|---------------------------------------------------------------------|
| Plates blurry / smeared letters            | Faster shutter (1/1000 s or 1/2000 s), slow shutter off, lower DNR  |
| Plates too small (< 120 px / < 80 px)      | Main stream; zoom in; move the camera closer to the reading spot    |
| Letters slanted, last character missing    | Aim more along the lane (< 20°); keep `crop.deshear: true` and `crop.retry_expand: 0.10` meanwhile |
| Grey blocks / smears in the picture        | Raise the bitrate (CBR 2–4 Mbit/s); check the cable / switch        |
| Picture freezes a second now and then      | Keyframe interval = frame rate; network errors in `probe_camera`    |
| Engine CPU at 100 %, low frame rate        | H.264 instead of H.265; `backend: ffmpeg`; `output_width: 1280`     |
| Night: plate is a white blob               | Lower IR/exposure, or turn on HLC/BLC (headlight compensation)      |
