#!/usr/bin/env bash
# Install / update the ANPR system on a Raspberry Pi 3B+ (Raspberry Pi OS Lite 64-bit, Bookworm).
# Run ON THE PI from the copied project folder:   sudo ./deploy/install_pi.sh
# Re-running is safe: it updates code and packages, and keeps config.yaml and data/.
set -euo pipefail

APP=/opt/anpr
SRC="$(cd "$(dirname "$0")/.." && pwd)"

[[ $EUID -eq 0 ]] || { echo "run with sudo"; exit 1; }
[[ "$(uname -m)" == "aarch64" ]] || { echo "needs 64-bit Raspberry Pi OS (aarch64), got $(uname -m)"; exit 1; }
PY=python3.11
command -v $PY >/dev/null || { echo "python3.11 not found (Bookworm ships it)"; exit 1; }

echo "== system packages"
apt-get update -qq
# libgl1/libglib2.0-0: needed by the opencv-python wheel (ncnn depends on it). rpicam-apps: Pi camera CLI.
# ffmpeg: camera.backend "ffmpeg" (RTSP main stream via the Pi's hardware H.264 decoder, h264_v4l2m2m).
apt-get install -y -qq python3.11-venv libgl1 libglib2.0-0 rpicam-apps sqlite3 ffmpeg

echo "== user and folders"
id anpr >/dev/null 2>&1 || useradd --system --home "$APP" --shell /usr/sbin/nologin anpr
# video: Pi camera + hardware decoder /dev/video10-12; render: /dev/dri (only if the group exists).
for g in video render; do getent group "$g" >/dev/null && usermod -aG "$g" anpr; done
mkdir -p "$APP" "$APP/data"

echo "== hardware video decoder (for camera.backend: ffmpeg)"
# Pi 3B+: H.264 only, through /dev/video10 (bcm2835-codec). There is NO H.265 hardware decoder.
# Capture first: with `set -o pipefail`, `ffmpeg ... | grep -q` fails whenever grep exits early and
# ffmpeg gets SIGPIPE, which reported "no h264_v4l2m2m" every time.
decoders=$(ffmpeg -hide_banner -decoders 2>/dev/null || true)
if grep -qw h264_v4l2m2m <<<"$decoders"; then
  echo "  ffmpeg has h264_v4l2m2m"
else
  echo "  WARN ffmpeg has no h264_v4l2m2m: H.264 will be decoded in software (slow for 1080p)"
fi
if [[ -e /dev/video10 ]]; then
  if runuser -u anpr -- test -r /dev/video10 -a -w /dev/video10; then
    echo "  /dev/video10 usable by user anpr ($(stat -c '%G %a' /dev/video10))"
  else
    echo "  WARN user anpr cannot open /dev/video10 ($(stat -c '%U:%G %a' /dev/video10)); it must be in that group"
  fi
else
  echo "  WARN /dev/video10 missing: no hardware decoder (not a Pi 3/4, or the bcm2835-codec driver is off)"
fi

echo "== code"
# Copy code + models; never overwrite the Pi's own config.yaml, licence.json or data/. tools/ holds the
# vendor's licence tool: it never goes onto a Pi.
tar -C "$SRC" --exclude=.venv --exclude=.venv-train --exclude=training --exclude=data --exclude=__pycache__ \
    --exclude=.pytest_cache --exclude=.ruff_cache --exclude=config.yaml --exclude=licence.json \
    --exclude=tools --exclude=scripts/eval_accuracy.py -cf - . | tar -C "$APP" -xf -
[[ -f "$APP/config.yaml" ]] || cp "$SRC/config.yaml" "$APP/config.yaml"
# Admin password (Activate licence, Settings, Add stream): made once, kept on later re-installs.
NEW_ADMIN=""
if grep -qE '^  admin_token: null' "$APP/config.yaml"; then
  NEW_ADMIN=$(python3 -c 'import secrets; print(secrets.token_urlsafe(12))')
  sed -i -E "s|^  admin_token: null.*|  admin_token: $NEW_ADMIN   # admin password (Activate / Settings / Add stream)|" "$APP/config.yaml"
elif ! grep -qE '^  admin_token:' "$APP/config.yaml"; then
  NEW_ADMIN=$(python3 -c 'import secrets; print(secrets.token_urlsafe(12))')
  sed -i -E "s|^(  auth_token:.*)$|\1\n  admin_token: $NEW_ADMIN   # admin password (Activate / Settings / Add stream)|" "$APP/config.yaml"
fi
chmod 640 "$APP/config.yaml"

echo "== python venv (exact locked versions, binary wheels only)"
[[ -x "$APP/.venv/bin/python" ]] || $PY -m venv "$APP/.venv"
"$APP/.venv/bin/pip" install -q --upgrade pip==25.2
"$APP/.venv/bin/pip" install -q --only-binary=:all: --no-deps -r "$APP/requirements.lock"
"$APP/.venv/bin/pip" check
"$APP/.venv/bin/python" -c "import cv2, ncnn, onnxruntime, numpy, fastapi; print('imports OK', cv2.__version__, numpy.__version__, onnxruntime.__version__)"

chown -R anpr:anpr "$APP"

echo "== services"
install -m 644 "$APP/deploy/anpr-engine.service" /etc/systemd/system/
install -m 644 "$APP/deploy/anpr-web.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable anpr-engine anpr-web
systemctl restart anpr-engine anpr-web

echo "== done"
echo "check:      sudo $APP/deploy/pi_check.sh"
echo "logs:       journalctl -u anpr-engine -f"
echo "dashboard:  http://$(hostname -I | awk '{print $1}'):8000"
echo "device ID:  $(cd "$APP" && "$APP/.venv/bin/python" -m anpr.engine --machine-id)  (send it to the supplier for the licence)"
echo "licence:    open the dashboard and paste the licence key (Activate); the engine starts within seconds"
if [[ -n "$NEW_ADMIN" ]]; then
  echo "admin pass: $NEW_ADMIN   (needed for Activate, Settings, Add stream; also in $APP/config.yaml web.admin_token)"
else
  echo "admin pass: unchanged (see web.admin_token in $APP/config.yaml)"
fi
