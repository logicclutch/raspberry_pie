#!/usr/bin/env bash
# Client demo: engine + view-only dashboard (config.demo.yaml) + a public HTTPS link via Cloudflare.
# The link stays up while this Mac is awake and this script runs. Ctrl+C stops everything.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
CFG=config.demo.yaml
PORT=$($PY -c "from anpr.config import load_config; print(load_config('$CFG').web.port)")
TOKEN=$($PY -c "from anpr.config import load_config; print(load_config('$CFG').web.auth_token)")
mkdir -p data/demo
$PY -m anpr.web --config "$CFG" > data/demo/web.log 2>&1 &
WEB=$!
$PY -m anpr.engine --config "$CFG" > data/demo/engine.log 2>&1 &
ENG=$!
caffeinate -dis -w $$ &   # keep the Mac awake while the demo runs
trap 'kill $WEB $ENG 2>/dev/null; wait 2>/dev/null' EXIT
cloudflared tunnel --no-autoupdate --url "http://127.0.0.1:$PORT" 2>&1 | while IFS= read -r line; do
  echo "$line" >> data/demo/tunnel.log
  if [[ "$line" =~ (https://[a-z0-9-]+\.trycloudflare\.com) ]]; then
    echo
    echo "  Client link:  ${BASH_REMATCH[1]}/?token=$TOKEN"
    echo
  fi
done
