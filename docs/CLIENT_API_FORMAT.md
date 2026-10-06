# Sending plates to the client database (API format)

Each Raspberry Pi sends every confirmed number plate to the client's server
as JSON over HTTPS. This file describes that JSON.

## Where to set it: dashboard → Settings (gear button)

On the admin dashboard, click the **gear button (Settings)** at the top right:

| Field | Example | Meaning |
|---|---|---|
| Send every new plate | on / off | Switches sending on or off. |
| API address (URL) | `https://client-server.com/api/anpr/events` | The client's API. Any `https://` or `http://` address. |
| Key header name | `Authorization` or `X-API-Key` | Name of the header that carries the key. Empty = no key. |
| API key | `Bearer abc123` | Sent exactly as typed. Saved on the Pi, never shown again (only its last 4 characters). Leave empty to keep the saved key. |
| Device ID | `PI-012` | This Pi's ID in the JSON. Empty = the Pi's hostname. |
| Include photos | on / off | Send the plate and vehicle photos (base64) or `null`. |

- **Send test** posts one sample plate (marked `"test": true`) with what is in the form, and shows the
  server's answer. Nothing is saved.
- **Save** takes effect within a few seconds. No restart is needed, and the settings can be changed at any
  time (new client, new key, new address).
- Switching sending on, or changing the address, starts from **that moment**: older plates are not
  sent to the new address. Changing only the key keeps the plates still waiting.
- The view-only (public) dashboard has no Settings button and cannot read the settings.
- The engine does the sending. The Settings box shows the status: plates sent, plates waiting, and the
  last error.

---

## 1. The request

```
POST https://<client-server>/api/anpr/events
Content-Type: application/json
Authorization: Bearer <device-secret-key>
```

- Every Pi has its own secret key. A Pi that is lost or stolen can be blocked
  alone.
- The Pi always sends a **list** (`events`), with 1 to 25 plates, oldest
  first. Normally the list has 1 plate. After the internet comes back, it
  sends the saved plates in batches.
- The **Send test** button adds `"test": true` at the top level of the JSON. Real plates never
  have it.

## 2. The JSON body

```json
{
  "schema_version": "1.0",
  "device": {
    "device_id": "PI-012",
    "station_name": "Gate 12 - Jaipur Warehouse",
    "software_version": "1.4.0"
  },
  "events": [
    {
      "event_id": "PI-012-000629",
      "plate": "HR39E7913",
      "plate_display": "HR 39 E 7913",
      "plate_type": "standard",
      "hsrp": "hsrp",
      "camera": "entry",
      "confidence": 0.9782,
      "votes": 4,
      "first_seen": "2026-09-30T18:01:55+05:30",
      "detected_at": "2026-09-30T18:01:56+05:30",
      "images": {
        "plate_crop_jpeg_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD...",
        "full_snapshot_jpeg_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD..."
      }
    }
  ]
}
```

## 3. What each field means

### `device` (the Pi that sends)

| Field | Type | Example | Meaning |
|---|---|---|---|
| `device_id` | text | `PI-012` | Fixed ID of the Pi (1 of 150). Never changes. |
| `station_name` | text | `Gate 12 - Jaipur Warehouse` | Name of the place, up to 40 characters. |
| `software_version` | text | `1.4.0` | ANPR software version on the Pi. |

### `events` (one item per vehicle)

| Field | Type | Example | Meaning |
|---|---|---|---|
| `event_id` | text | `PI-012-000629` | Unique ID of this plate read (`device_id` + number). **The server uses this to ignore duplicates.** |
| `plate` | text | `HR39E7913` | Plate number. Capital letters and digits only, no spaces. |
| `plate_display` | text | `HR 39 E 7913` | The same number with spaces, for showing on screen. |
| `plate_type` | text | `standard` | `standard` = normal state plate, `bh` = Bharat series (`22 BH 1234 AB`). |
| `hsrp` | text or null | `hsrp` | `hsrp`, `non_hsrp`, `unsure`, or `null` (check switched off). |
| `camera` | text | `entry` | Which camera saw it (multi-camera Pi). `""` on a single-camera Pi. |
| `confidence` | number 0–1 | `0.9782` | How sure the reader is. 1 = fully sure. |
| `votes` | whole number | `4` | How many camera frames agreed on this number. |
| `first_seen` | date-time | `2026-09-30T18:01:55+05:30` | When the vehicle was first seen (Indian time). |
| `detected_at` | date-time | `2026-09-30T18:01:56+05:30` | When the plate was confirmed. **Use this as the event time.** |
| `images.plate_crop_jpeg_base64` | text or null | | Small photo of just the plate (JPEG as base64, about 5 KB). |
| `images.full_snapshot_jpeg_base64` | text or null | | Photo of the whole vehicle (JPEG as base64, 640 px wide, about 100 KB). |

Notes:
- Times are ISO 8601 with the time zone (`+05:30`).
- An image field can be `null` if that photo couldn't be saved. The plate
  number is still sent.

## 4. What the server must reply

**Success (HTTP 200):**

```json
{
  "accepted": ["PI-012-000629"],
  "rejected": []
}
```

**Some plates refused (HTTP 200):**

```json
{
  "accepted": ["PI-012-000629"],
  "rejected": [
    { "event_id": "PI-012-000630", "reason": "plate field is empty" }
  ]
}
```

| Server reply | What the Pi does |
|---|---|
| Any `2xx` (`200`, `201`, ...) | Marks those plates as sent. The `accepted` / `rejected` lists are optional; plates listed in `rejected` are logged and not sent again. |
| `401` / `403` | Wrong key. Keeps the plates, shows the error, retries. |
| `404` / `405` | Wrong address. Keeps the plates, shows the error, retries. |
| `408`, `409`, `425`, `429`, `5xx`, timeout, no internet | Keeps the plates and tries again later (wait 5 s, 10 s, 20 s, up to 5 min). Saving the settings retries at once. |
| `3xx` redirect | Not followed (it would lose the data). Shown as an error: put the new address in Settings. |
| Other `4xx` (`400`, `413`, `422`, ...) | The data was refused. The Pi sends the batch one plate at a time and skips only the plate the server refuses. |

Rules for the server:
1. **Duplicates:** if an `event_id` already exists, put it in `accepted` but
   don't save it again. The Pi may send the same plate twice after a network
   problem.
2. Reply within 10 seconds.
3. Accept requests up to about 4 MB (25 plates with photos).

## 5. Heartbeat (optional, every 1 minute, not built yet)

This tells the client that the Pi and its camera are working, even when no
vehicles pass.

```
POST https://<client-server>/api/anpr/heartbeat
```

```json
{
  "schema_version": "1.0",
  "device_id": "PI-012",
  "station_name": "Gate 12 - Jaipur Warehouse",
  "time": "2026-10-01T10:15:00+05:30",
  "camera_ok": true,
  "fps": 4.2,
  "cpu_temp_c": 61.5,
  "free_disk_mb": 8120,
  "unsent_events": 0
}
```

If a Pi doesn't send a heartbeat for 3 minutes, the client can mark it **offline**.

## 6. Suggested database table (client side)

```sql
CREATE TABLE anpr_events (
    event_id        VARCHAR(40)  PRIMARY KEY,   -- stops duplicates
    device_id       VARCHAR(20)  NOT NULL,
    station_name    VARCHAR(40)  NOT NULL,
    plate           VARCHAR(12)  NOT NULL,
    plate_display   VARCHAR(16)  NOT NULL,
    plate_type      VARCHAR(10)  NOT NULL,      -- standard | bh
    hsrp            VARCHAR(10),                -- hsrp | non_hsrp | unsure | NULL
    camera          VARCHAR(40),                -- which camera (multi-camera Pi); "" if single
    confidence      DECIMAL(5,4) NOT NULL,
    votes           INTEGER      NOT NULL,
    first_seen      TIMESTAMPTZ  NOT NULL,
    detected_at     TIMESTAMPTZ  NOT NULL,
    plate_image     BYTEA,                      -- or save to file storage and keep the path
    vehicle_image   BYTEA,
    received_at     TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX idx_anpr_plate    ON anpr_events (plate);
CREATE INDEX idx_anpr_time     ON anpr_events (detected_at);
CREATE INDEX idx_anpr_device   ON anpr_events (device_id, detected_at);
```
