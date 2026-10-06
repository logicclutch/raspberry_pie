#!/usr/bin/env python3
"""Capture ONE real POST from the ANPR/CCTV camera so we can see its exact format.

Run this where the camera can reach it (the Pi, or this Mac if the camera is on the same network):

    python3 deploy/capture_camera_post.py            # listens on port 8080, path /
    python3 deploy/capture_camera_post.py 9000        # a different port

Then, in the GVD camera's web settings, set its HTTP upload / event-notification target to:
    http://<this-machine-ip>:8080/
Trigger one vehicle (or wait for one). The tool saves the request to capture/ and prints a summary:
  - the method, path and headers
  - the JSON keys (and where a big base64 value lives)
  - the decoded image (if a base64 image is found) as capture/image_001.jpg so you can confirm it

Send me the printed JSON-keys summary (and whether image_001.jpg looks right). No camera password is needed
and nothing is uploaded anywhere - it only listens and writes to the local capture/ folder.
"""
from __future__ import annotations

import base64
import binascii
import json
import re
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "capture"
OUT.mkdir(exist_ok=True)
_count = 0


def _find_base64_images(obj, path=""):
    """Walk a JSON object and yield (json_path, decoded_bytes) for values that look like base64 images."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _find_base64_images(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _find_base64_images(v, f"{path}[{i}]")
    elif isinstance(obj, str) and len(obj) > 500:
        s = re.sub(r"^data:image/[^;]+;base64,", "", obj.strip())
        try:
            raw = base64.b64decode(s, validate=False)
        except (binascii.Error, ValueError):
            return
        if raw[:3] == b"\xff\xd8\xff" or raw[:8] == b"\x89PNG\r\n\x1a\n":  # JPEG / PNG magic
            yield path, raw


def _summarize_keys(obj, prefix=""):
    lines = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (dict, list)):
                lines += _summarize_keys(v, p)
            elif isinstance(v, str) and len(v) > 200:
                lines.append(f"  {p}  = <{len(v)} chars>  (likely the base64 image)")
            else:
                lines.append(f"  {p}  = {v!r}")
    elif isinstance(obj, list):
        lines.append(f"  {prefix}  = list of {len(obj)}")
        if obj:
            lines += _summarize_keys(obj[0], f"{prefix}[0]")
    return lines


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet the default logging
        pass

    def _handle(self):
        global _count
        _count += 1
        n = f"{_count:03d}"
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length) if length else b""
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print("\n" + "=" * 70)
        print(f"[{stamp}]  REQUEST #{n}  {self.command} {self.path}")
        print("-- headers --")
        for k, v in self.headers.items():
            print(f"  {k}: {v}")
        ctype = self.headers.get("Content-Type", "")
        (OUT / f"raw_{n}.bin").write_bytes(body)
        print(f"-- body: {len(body)} bytes (saved raw_{n}.bin), Content-Type: {ctype or '(none)'}")

        if "json" in ctype.lower() or body[:1] in (b"{", b"["):
            try:
                obj = json.loads(body.decode("utf-8", "replace"))
                (OUT / f"body_{n}.json").write_text(json.dumps(obj, indent=2, ensure_ascii=False))
                print("-- JSON keys (send me this) --")
                print("\n".join(_summarize_keys(obj)) or "  (empty)")
                imgs = list(_find_base64_images(obj))
                for i, (jpath, raw) in enumerate(imgs, 1):
                    f = OUT / f"image_{n}_{i}.jpg"
                    f.write_bytes(raw)
                    print(f"-- found a base64 image at JSON field: '{jpath}'  -> saved {f.name} ({len(raw)} bytes)")
                if not imgs:
                    print("-- no base64 image found in the JSON (maybe it is a URL, or multipart). Send me body_*.json")
            except json.JSONDecodeError:
                print("-- body is not JSON. It may be multipart/form-data. Send me raw_*.bin so I can see the format.")
        else:
            print("-- non-JSON body (likely multipart/form-data with image parts). Send me raw_*.bin")

        # Answer the camera so it does not keep retrying.
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    do_POST = _handle
    do_PUT = _handle
    do_GET = _handle


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"Listening on http://0.0.0.0:{port}/   (point the GVD camera's HTTP upload here)")
    print(f"Saving captures to: {OUT}")
    print("Trigger one vehicle, then press Ctrl+C. Send me the 'JSON keys' summary it prints.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
