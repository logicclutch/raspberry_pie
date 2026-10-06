#!/usr/bin/env bash
# Build the client .deb (Raspberry Pi OS 64-bit Bookworm, arm64) on the Mac.
#   ./deploy/build_deb.sh              normal build  (dashboard + API, RTSP/Pi camera)
#                                                                           -> dist/anpr_<ver>_arm64.deb
#   HEADLESS=1 ./deploy/build_deb.sh   slim headless (no dashboard, API push only, supports two cameras)
#                                                                           -> dist/anpr_<ver>_headless_arm64.deb
#   INGEST=1 ./deploy/build_deb.sh     ingest/push receiver: the ANPR camera POSTs vehicle images to the
#                                      Pi; the Pi reads the plate and sends it on. No RTSP/video. Lightest.
#                                                                           -> dist/anpr_<ver>_ingest_arm64.deb
# Needs Docker with arm64 emulation (Docker Desktop / colima with qemu).
#
# What the package contains:
#   /opt/anpr/lib      the app as COMPILED bytecode only (.pyc, no .py source) + web static + models
#   /opt/anpr/.venv    the Python environment, built for arm64 (ready-to-run wheels; the Pi needs no pip)
#   /opt/anpr/bin/anpr launcher: `anpr engine|web|probe-camera|version`
#   systemd units, config.default.yaml, pi_check.sh, docs
# Always slimmed: unused packages (sympy, mpmath, pip, setuptools) and the detector's .onnx source and
# OCR candidates are dropped. HEADLESS also removes the dashboard (anpr/web + the web Python stack + the
# web service) and ships the headless config (web off, preview off, two-camera ready, more RAM headroom).
# Built INSIDE an arm64 Debian Bookworm container at the Pi's real path (/opt/anpr, python3.11), so the
# venv matches the Pi exactly. Vendor-only files (tools/, training/, licence.json, videos) never enter it.
set -euo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)"
VER_PY=$(sed -nE 's/^version = "([^"]+)"/\1/p' "$SRC/pyproject.toml" | head -1)
VERSION="${VERSION:-${VER_PY}-1}"
HEADLESS="${HEADLESS:-0}"
INGEST="${INGEST:-0}"
# Target OS / Python. Default = Raspberry Pi OS Bookworm (python3.11). For Raspberry Pi OS Trixie set
# DIST=trixie PYVER=3.13. The venv is built in that exact base so the Pi needs no pip.
DIST="${DIST:-bookworm}"
PYVER="${PYVER:-3.11}"
OUT="$SRC/dist"
mkdir -p "$OUT"
# MODE: ingest > headless > normal. ingest and headless are both "no dashboard" (web stack stripped).
if [ "$INGEST" = "1" ]; then MODE=ingest; SUFFIX="_ingest"; HEADLESS=1
elif [ "$HEADLESS" = "1" ]; then MODE=headless; SUFFIX="_headless"
else MODE=normal; SUFFIX=""; fi
# A non-default OS goes in the file name so the two builds never clash (e.g. ..._ingest_trixie_arm64.deb).
[ "$DIST" = "bookworm" ] && DISTTAG="" || DISTTAG="_${DIST}"
DEB_NAME="anpr_${VERSION}${SUFFIX}${DISTTAG}_arm64.deb"
# Debian dependency on the matching interpreter package (its name encodes the minor version).
PYNEXT="3.$(( ${PYVER#3.} + 1 ))"
PYDEP="python${PYVER} (>= ${PYVER}.0), python${PYVER} (<< ${PYNEXT})"

docker info >/dev/null 2>&1 || { echo "Docker is not running. Open Docker Desktop and try again."; exit 1; }

echo "== building $DEB_NAME (MODE=$MODE, DIST=$DIST, PYVER=$PYVER) in an arm64 Debian $DIST container"
docker run --rm --platform linux/arm64 \
  -e VERSION="$VERSION" -e HEADLESS="$HEADLESS" -e MODE="$MODE" -e DEB_NAME="$DEB_NAME" \
  -e PYVER="$PYVER" -e PYDEP="$PYDEP" \
  -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
  -v "$SRC":/src:ro -v "$OUT":/out \
  "debian:$DIST" bash -euo pipefail -c '
APP=/opt/anpr
LIB="$APP/lib"
PKG=/build/pkg
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends "python${PYVER}" "python${PYVER}-venv" libgl1 libglib2.0-0 ca-certificates >/dev/null
[ "$(python${PYVER} -c "import platform;print(platform.machine())")" = "aarch64" ] || { echo "not building on arm64 - enable Docker arm64 emulation"; exit 1; }

echo "== python venv (exact locked versions, binary wheels only, arm64)"
python${PYVER} -m venv "$APP/.venv"
"$APP/.venv/bin/pip" install -q --disable-pip-version-check --upgrade pip==25.2
"$APP/.venv/bin/pip" install -q --disable-pip-version-check --only-binary=:all: --no-deps -r /src/requirements.lock
"$APP/.venv/bin/pip" check

echo "== app files (whitelist: code + web static + models + config)"
mkdir -p "$LIB" "$APP/bin" "$APP/deploy" "$APP/docs" "$LIB/models/detector" "$LIB/models/ocr"
cd /src
tar -cf - --exclude=__pycache__ anpr | tar -C "$LIB" -xf -
cp scripts/probe_camera.py "$LIB/probe_camera.py"
cp deploy/pi_check.sh "$APP/deploy/"
cp docs/CAMERA_SETUP.md docs/CLIENT_API_FORMAT.md "$APP/docs/"
# The detector loads the ncnn model directory; the .onnx is only the conversion source (not shipped).
cp -r models/detector/plate_det_ncnn_model "$LIB/models/detector/"
cp models/ocr/plate_ocr.onnx models/ocr/plate_ocr_config.yaml "$LIB/models/ocr/"

case "$MODE" in
  ingest)   CONF=deploy/deb/config.ingest.yaml ;;
  headless) CONF=deploy/deb/config.headless.yaml ;;
  *)        CONF=deploy/deb/config.client.yaml ;;
esac
# The config template points model paths at lib/models (that is where they live in the package).
sed -e "s#models/detector/plate_det_ncnn_model#lib/models/detector/plate_det_ncnn_model#" \
    -e "s#models/ocr/plate_ocr.onnx#lib/models/ocr/plate_ocr.onnx#" \
    -e "s#models/ocr/plate_ocr_config.yaml#lib/models/ocr/plate_ocr_config.yaml#" \
    "$CONF" > "$APP/config.default.yaml"
install -m 755 deploy/deb/anpr "$APP/bin/anpr"
echo "$VERSION" > "$APP/VERSION"
chmod 755 "$APP/deploy/pi_check.sh"

echo "== slim: drop packages and files not used at runtime"
SITE="$APP/.venv/lib/python${PYVER}/site-packages"
drop_pkg() {  # remove an installed package dir + its dist-info (safe: verified unused at runtime)
  for d in "$SITE/$1" "$SITE/$1".*-info "$SITE/$1"-*.dist-info; do rm -rf $d 2>/dev/null || true; done
}
for p in sympy mpmath pip setuptools pkg_resources _distutils_hack wheel; do drop_pkg "$p"; done
rm -f "$SITE"/*.pth 2>/dev/null || true
if [ "$HEADLESS" = "1" ]; then
  echo "== headless: remove the dashboard (anpr/web) and the web-only packages"
  rm -rf "$LIB/anpr/web"
  # The engine uses Python stdlib urllib for the API POST, not these; verified not imported by the engine.
  for p in fastapi uvicorn starlette websockets httptools h11 click requests urllib3 certifi idna charset_normalizer; do drop_pkg "$p"; done
fi

echo "== compile to bytecode, then delete .py source (no source ships)"
# -b: write foo.pyc beside foo.py (not in __pycache__), so imports work after the .py are removed.
"$APP/.venv/bin/python" -m compileall -q -b -j 0 "$LIB"
PYCOUNT=$(find "$LIB" -name "*.py" | wc -l); PYCCOUNT=$(find "$LIB" -name "*.pyc" | wc -l)
[ "$PYCOUNT" -eq "$PYCCOUNT" ] || { echo "compile mismatch: $PYCOUNT .py vs $PYCCOUNT .pyc"; exit 1; }
find "$LIB" -name "*.py" -delete
find "$LIB" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
[ "$(find "$LIB" -name "*.py" | wc -l)" -eq 0 ] || { echo "ERROR: .py source still present"; exit 1; }

echo "== smoke test (run from compiled bytecode, like the Pi will)"
"$APP/bin/anpr" version
"$APP/bin/anpr" engine --help >/dev/null
"$APP/bin/anpr" ingest --help >/dev/null
"$APP/bin/anpr" probe-camera --help >/dev/null
"$APP/bin/anpr" engine --config "$APP/config.default.yaml" --help >/dev/null
# Import every heavy module from the compiled .pyc, the way the service will on the Pi. --machine-id
# needs real Pi hardware, so it is not part of the test.
WEB_IMPORT="import anpr.web.app;"
if [ "$HEADLESS" = "1" ]; then
  WEB_IMPORT=""
  "$APP/.venv/bin/python" -s -c "import sys; sys.path.insert(0,\"$LIB\");
import importlib.util; assert importlib.util.find_spec(\"anpr.web\") is None, \"web should be gone\"; print(\"headless: web removed OK\")"
else
  "$APP/bin/anpr" web --help >/dev/null
fi
"$APP/.venv/bin/python" -s -c "import sys; sys.path.insert(0,\"$LIB\"); $WEB_IMPORT import anpr.engine, anpr.detector, anpr.ocr, anpr.hsrp, anpr.push, anpr.licence, probe_camera; print(\"compiled imports OK\")"

echo "== package tree"
mkdir -p "$PKG/DEBIAN" "$PKG/opt" "$PKG/lib/systemd/system"
cp -a "$APP" "$PKG/opt/"
# Which systemd services ship depends on the mode.
case "$MODE" in
  ingest)
    # Only the ingest receiver (no camera engine, no dashboard).
    cp /src/deploy/deb/anpr-ingest.service "$PKG/lib/systemd/system/"
    ;;
  headless)
    cp /src/deploy/deb/anpr-engine.service "$PKG/lib/systemd/system/"
    # No dashboard -> the engine can use the RAM the web service would have taken.
    sed -i -E "s/^MemoryMax=.*/MemoryMax=900M/" "$PKG/lib/systemd/system/anpr-engine.service"
    ;;
  *)
    cp /src/deploy/deb/anpr-engine.service /src/deploy/deb/anpr-web.service "$PKG/lib/systemd/system/"
    ;;
esac
chmod 644 "$PKG"/lib/systemd/system/*.service
for f in postinst prerm postrm; do install -m 755 "/src/deploy/deb/$f" "$PKG/DEBIAN/$f"; done
SIZE=$(du -sk --exclude=DEBIAN "$PKG" | cut -f1)
sed -e "s/@VERSION@/$VERSION/" -e "s/@SIZE@/$SIZE/" -e "s|@PYDEP@|$PYDEP|" /src/deploy/deb/control.in > "$PKG/DEBIAN/control"
chmod -R u+rwX,go+rX,go-w "$PKG/opt" "$PKG/lib"

DEB="/out/$DEB_NAME"
dpkg-deb --root-owner-group -Zxz --build "$PKG" "$DEB" >/dev/null
chown "$HOST_UID:$HOST_GID" "$DEB"
echo "----"
dpkg-deb --info "$DEB" | sed -n "1,20p"
echo "installed size on the Pi: $((SIZE / 1024)) MB"
'
echo "========"
ls -lh "$OUT/$DEB_NAME"
( cd "$OUT" && shasum -a 256 "$DEB_NAME" | tee "$DEB_NAME.sha256" )
