#!/usr/bin/env bash
# Mac dev: run the engine and the dashboard together, the same way systemd runs them on the Pi.
#   ./scripts/run_dev.sh                     # webcam 0 (from config.yaml)
#   ./scripts/run_dev.sh path/to/video.mp4   # a recorded video
# A finished video leaves the engine waiting: pick the next one with "Add stream" on the dashboard.
# Ctrl+C stops both.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
SRC_ARGS=()
[[ $# -ge 1 ]] && SRC_ARGS=(--source "$1")

$PY -m anpr.web --config config.yaml &
WEB=$!
trap 'kill $WEB 2>/dev/null; wait $WEB 2>/dev/null' EXIT
echo "dashboard: http://127.0.0.1:8000"
$PY -m anpr.engine --config config.yaml "${SRC_ARGS[@]+"${SRC_ARGS[@]}"}"
