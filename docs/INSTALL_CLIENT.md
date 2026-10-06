# ANPR Camera-Push — Setup Guide (Raspberry Pi 3B+)

Your ANPR camera sends each vehicle image to this Raspberry Pi. The Pi reads the number plate and sends
the result (plate number, HSRP, details, and the plate image) to your server/database.

```
  ANPR camera (e.g. GVD)                 Raspberry Pi                         Your server
  detects a vehicle                      reads the plate                      stores the result
        │                                      │                                    ▲
        └── POST vehicle image (base64) ──▶ /api/v1/ingest ── reads + votes ──▶ POST result (JSON) ──┘
```

Two addresses are involved:
- **Address A — into the Pi:** you paste this into the **camera** so it uploads images to the Pi.
- **Address B — out of the Pi:** **your own server's API**, where the Pi posts the finished plate result.

---

## What you need

- **Raspberry Pi 3B+** with a heatsink/fan and the official **5 V / 2.5 A** power supply.
- A **microSD card (16 GB+)** flashed with **Raspberry Pi OS, 64-bit (Bookworm)** — it must be the
  **64-bit** version (32-bit will not work). **Lite or Desktop both work**; **Lite is recommended** on the
  1 GB Pi 3B+ because the Desktop GUI uses ~200–400 MB of RAM. Check with `uname -m` on the Pi — it must
  say `aarch64`. Use **Raspberry Pi Imager**; enable **SSH** and set a username/password so you can reach
  the Pi over the network.
- The Pi on your network (cable or Wi-Fi), reachable by SSH, **and reachable by the camera** (same LAN).
- The **one file** you received: `anpr_<version>_ingest_arm64.deb`.
- Your **ANPR camera** (e.g. GVD) that can POST a vehicle image on a plate event (HTTP upload).

> The Pi needs the internet **once** during install (to fetch a few standard system packages). After
> that it runs offline. The plate software is all inside the `.deb`.

---

## Step 1 — install on the Pi

From your computer (replace `PI-IP` with the Pi's address, e.g. `192.168.1.50`):
```
scp anpr_*_ingest_arm64.deb  pi@PI-IP:~
ssh pi@PI-IP
```
On the Pi:
```
sudo apt update
sudo apt install -y ./anpr_*_ingest_arm64.deb
```
That is the whole install — it sets everything up and **starts automatically on boot**. When it finishes
it prints the **device ID** and the **address the camera posts to** — keep both.

## Step 2 — activate the licence (in the terminal)

The software runs only with a licence for this Pi. There is no dashboard; do it over SSH.

```
sudo /opt/anpr/bin/anpr ingest --machine-id          # shows this Pi's device ID
```
Send that device ID to the supplier (LogicClutch). You get back a key starting with `ANPR1-`. Enter it:
```
sudo /opt/anpr/bin/anpr ingest --activate 'ANPR1-...'
```
Check it is valid (optional):
```
sudo /opt/anpr/bin/anpr ingest --check-licence
```

## Step 3 — point the ANPR camera at the Pi (Address A)

In the **camera's** settings, find the **HTTP upload / event-notification / "listening host"** option and
set the upload URL to:

```
http://<PI-IP>:8080/api/v1/ingest
```

- Trigger it on the **licence-plate / vehicle event**.
- Have it send the **full vehicle image** (full resolution, not a thumbnail).
- The Pi accepts **any common format** — JSON with a base64 image, a multipart file upload, or a plain
  JPEG body — so whatever the camera sends will work.

**What the Pi answers** (each POST is one vehicle; the reply comes back in about a second):
```json
{"ok": true, "images": 2, "plate": "UP83DT0718",
 "plates": [{"plate": "UP83DT0718", "confidence": 0.988, "kind": "standard", "hsrp": "non_hsrp",
             "votes": 1, "duplicate": false}],
 "push": {"enabled": true, "url": "http://192.168.1.72:21300/lane/anpr", "host": "192.168.1.72:21300",
          "plate": "UP83DT0718", "ok": true, "queued": false, "status": 200,
          "message": "accepted (HTTP 200)", "reply": "<your API's answer>"}}
```
If your API cannot be reached, the reply still has the plate, and an `error` (at the top and inside `push`)
names the API address and the plate:
```json
{"ok": true, "plate": "UP83DT0718", ...,
 "push": {"enabled": true, "url": "http://192.168.1.72:21300/lane/anpr", "host": "192.168.1.72:21300",
          "plate": "UP83DT0718", "ok": false, "queued": true, "status": null,
          "message": "cannot reach the server: timed out", "reply": "",
          "error": "API is not connecting: http://192.168.1.72:21300/lane/anpr (host 192.168.1.72:21300): timed out. Plate UP83DT0718 is saved on the Pi and is sent automatically when the API accepts it."},
 "error": "API is not connecting: http://192.168.1.72:21300/lane/anpr (host 192.168.1.72:21300): timed out. Plate UP83DT0718 is saved on the Pi and is sent automatically when the API accepts it."}
```
- `plate: null`: no plate could be read in the image(s).
- `"duplicate": true`: the plate was read, but the same plate was already recorded in the last 60 s
  (for example, the camera sent the vehicle again). It is not stored or sent a second time.
- `push.ok: true`: **this** plate was accepted by your API. `push.queued: true`: it has not been delivered
  yet (your API is down or returned an error, or older plates are being sent first). It is saved on the Pi
  and sent automatically when your API accepts it, even if no more vehicles come.
- `503 busy`: too many uploads at the same moment. The camera should send it again.

## Step 4 — set your server API (Address B)

This is where the Pi sends the finished plate result. Edit the config file on the Pi:
```
sudo nano /opt/anpr/config.yaml
```
Sending is **already switched on** in the installed config, pointing at
`http://192.168.1.72:21300/lane/anpr`. If your API has a different address (or needs a key), change the
`push:` section, then save:
```yaml
push:
  enabled: true
  url: 'https://your-server.com/api/ANPRLogInsert'   # YOUR API that receives the plates
  header_name: 'Authorization'                        # leave '' if your API needs no key
  header_value: 'Bearer YOUR_KEY'                     # leave '' if none
```
Apply it:
```
sudo systemctl restart anpr-ingest
```
The exact JSON the Pi sends (plate, HSRP, confidence, timestamps, and the plate image as base64) is in
`/opt/anpr/docs/CLIENT_API_FORMAT.md`. If your API is down, the Pi keeps the plates and retries — nothing
is lost.

## Step 5 — confirm it is working

```
curl http://<PI-IP>:8080/health          # should return {"ok": true, ...}
journalctl -u anpr-ingest -f             # shows each vehicle as the camera posts it
```
Drive a vehicle past (or trigger the camera). You should see the plate read in the log, and the result
arrive at your server.

---

## How the reading works (good to know)

- The camera usually sends **several images per vehicle**. The Pi **groups them and votes**, then sends
  **one best result per vehicle** — not one message per image. This fixes the odd misread and keeps your
  database clean (no duplicates for you to remove).
- Each result carries the **plate number, type, HSRP / non-HSRP, confidence, timestamps, and the plate
  image** (base64).

## Settings file (`/opt/anpr/config.yaml`)

One file controls everything; it survives updates. Edit, then `sudo systemctl restart anpr-ingest`.

| What you want to change | In the file |
|---|---|
| **Your server API** (where plates go) | `push: url:` |
| **API key / token** (if your API needs one) | `push: header_name:` and `push: header_value:` |
| **Port the camera posts to** (default 8080) | `ingest: port:` |
| **Require a key from the camera** (optional security) | `ingest: token:` → then the camera must send header `X-Ingest-Token` |
| **Include the plate image in the result** | `push: include_images:` (true/false) |
| **How long plates are kept on the Pi** | `storage: retention_days:` |

## Everyday use

- **It starts on its own** after a power cut or reboot — nothing to launch.
- **Health check:**  `sudo /opt/anpr/deploy/pi_check.sh`
- **Watch it live:**  `journalctl -u anpr-ingest -f`  (Ctrl-C to stop watching)
- **Restart:**  `sudo systemctl restart anpr-ingest`

## Updating / renewing

- **New version:**  `sudo apt install -y ./anpr_<new-version>_ingest_arm64.deb` — settings, licence and
  saved plates are kept; it restarts itself.
- **Renew / add a Pi:**  send the Pi's device ID (`anpr ingest --machine-id`) to the supplier and enter
  the new key with `--activate`, like Step 2.

---

## If something is wrong

Run `sudo /opt/anpr/deploy/pi_check.sh` first — it checks the hardware, licence, service and API and says
what is wrong in plain words.

| What you see | What to do |
|---|---|
| Install error: *"package architecture (arm64) does not match"* | The SD card has 32-bit Pi OS. Re-flash with **64-bit** Pi OS Lite. |
| `--check-licence` says *"not for this device"* | The key is for a different Pi. Send **this** Pi's device ID to the supplier. |
| *"licence has ended"* | Ask the supplier for a renewed key and `--activate` again. |
| **Camera posts but nothing happens** | Check `curl http://<PI-IP>:8080/health`; the camera's upload path must be `/api/v1/ingest`, port **8080** reachable from the camera, and it must send the **full image**. Watch `journalctl -u anpr-ingest -f`. |
| **Plates read but your server gets nothing** | Check `push: url:` in `/opt/anpr/config.yaml` is correct and the key (if any) is set; `journalctl -u anpr-ingest -f` shows send errors. |
| **Plates are read wrong often** | Send the camera a **full-resolution, front-on** image; for worn/painted plates, the supplier can improve accuracy from your real images (Phase 2). |

**Support:** send the supplier the output of `sudo /opt/anpr/deploy/pi_check.sh` and the device ID.
