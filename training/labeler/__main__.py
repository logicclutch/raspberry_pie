"""Run the labeler: `.venv/bin/python -m training.labeler --data training/data [--port 8010]`.

Binds 127.0.0.1 only (there is deliberately no --host option): labels and crops stay on this machine.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    from training.dataset import labels_path
    from training.labeler.app import create_app

    ap = argparse.ArgumentParser(prog="python -m training.labeler", description="Label harvested plate crops")
    ap.add_argument("--data", type=Path, default=Path("training/data"), help="folder with labels.csv")
    ap.add_argument("--port", type=int, default=8010)
    args = ap.parse_args(argv)
    if not labels_path(args.data).is_file():
        ap.error(f"{labels_path(args.data)} not found: run `python -m training.harvest` first")
    print(f"labeler: http://127.0.0.1:{args.port}/  (data: {args.data.resolve()})", flush=True)
    uvicorn.run(
        create_app(args.data),
        host="127.0.0.1",
        port=args.port,
        workers=1,
        log_level="warning",
        access_log=False,
        proxy_headers=False,
        server_header=False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
