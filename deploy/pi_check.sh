#!/usr/bin/env bash
# Quick health check on the Pi: hardware, camera, models, services, API.
APP=/opt/anpr
# .deb install: compiled program behind /opt/anpr/bin/anpr. Script install (install_pi.sh): python -m.
if [[ -x "$APP/bin/anpr" ]]; then ENGINE=("$APP/bin/anpr" engine); PROBE="anpr probe-camera"
else ENGINE=("$APP/.venv/bin/python" -m anpr.engine); PROBE=".venv/bin/python -m scripts.probe_camera"; fi
ok()   { echo "  OK   $*"; }
bad()  { echo "  FAIL $*"; }
warn() { echo "  WARN $*"; }

echo "== hardware"
model=$(tr -d '\0' </proc/device-tree/model 2>/dev/null); echo "  $model"
[[ "$(uname -m)" == "aarch64" ]] && ok "64-bit OS" || bad "not 64-bit ($(uname -m))"
t=$(( $(cat /sys/class/thermal/thermal_zone0/temp) / 1000 )); (( t < 75 )) && ok "CPU ${t}C" || warn "CPU ${t}C (add heatsink/fan)"
thr=$(vcgencmd get_throttled 2>/dev/null | cut -d= -f2); [[ "$thr" == "0x0" ]] && ok "no throttling/under-voltage" || warn "throttled=$thr (check the 5V/2.5A supply and cooling)"
free -m | awk '/Mem:/{printf "  INFO RAM used %d/%d MB\n",$3,$2}'
df -m "$APP/data" | awk 'NR==2{printf "  INFO disk free %d MB\n",$4}'

echo "== camera"
src=$(grep -E '^\s*source:' "$APP/config.yaml" | head -1 | sed -E "s/.*source:\s*['\"]?([^'\"]*)['\"]?/\1/")
echo "  config source: $src"
backend=$(grep -E '^\s*backend:\s*(opencv|ffmpeg)' "$APP/config.yaml" | head -1 | sed -E 's/.*backend:\s*([a-z]+).*/\1/')
echo "  camera backend: ${backend:-opencv}"
if [[ "$backend" == "ffmpeg" ]]; then
  command -v ffmpeg >/dev/null && ok "ffmpeg installed" || bad "ffmpeg missing (sudo apt install ffmpeg)"
  ffmpeg -hide_banner -decoders 2>/dev/null | grep -qw h264_v4l2m2m && ok "h264_v4l2m2m decoder" || warn "no h264_v4l2m2m: software decoding"
  runuser -u anpr -- test -r /dev/video10 -a -w /dev/video10 2>/dev/null && ok "/dev/video10 usable by anpr" || warn "anpr cannot use /dev/video10 (video group?)"
  echo "  stream check: cd $APP && sudo -u anpr $PROBE '<rtsp link>' --guess-main"
fi
if [[ "$src" == "picamera" ]]; then
  rpicam-hello --list-cameras 2>&1 | grep -q ':' && ok "Pi camera detected" || bad "no Pi camera (ribbon cable? camera_auto_detect=1 in /boot/firmware/config.txt?)"
fi

echo "== models"
for p in $(grep -E 'model_path|config_path' "$APP/config.yaml" | sed -E 's/.*: *//'); do
  [[ "$p" = /* ]] || p="$APP/$p"
  [[ -e "$p" ]] && ok "$p" || bad "missing $p"
done

echo "== services"
if out=$(cd "$APP" && "${ENGINE[@]}" --config "$APP/config.yaml" --check-licence 2>&1); then ok "$out"
else bad "$out (device ID: $(cd "$APP" && "${ENGINE[@]}" --machine-id 2>&1))"; fi
# Only check the services this install actually has (ingest receiver, or camera engine +/- dashboard).
for s in anpr-ingest anpr-engine anpr-web; do
  [ -f "/lib/systemd/system/$s.service" ] || [ -f "/etc/systemd/system/$s.service" ] || continue
  systemctl is-active --quiet $s && ok "$s running" || bad "$s not running (journalctl -u $s -n 50)"
done

echo "== api"
if [ -f /lib/systemd/system/anpr-ingest.service ]; then
  port=$(grep -E '^\s*port:' "$APP/config.yaml" | awk '{print $2}' | head -1); port=${port:-8080}
  curl -fsS --max-time 3 "http://127.0.0.1:${port}/health" && echo || bad "ingest /health not answering on :${port}"
elif [ -f /lib/systemd/system/anpr-web.service ]; then
  curl -fsS --max-time 3 http://127.0.0.1:8000/health && echo || bad "/health not answering"
else
  ok "headless (no web API to check)"
fi
