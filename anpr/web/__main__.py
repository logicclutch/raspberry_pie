"""Run the web process (Steps 9-10): `python -m anpr.web --config config.yaml`."""

from __future__ import annotations

import importlib.util
import logging
import sys

log = logging.getLogger("anpr.web")


def main(argv: list[str] | None = None) -> int:
    import argparse

    import uvicorn

    from anpr.config import load_config
    from anpr.web.app import create_app

    ap = argparse.ArgumentParser(prog="python -m anpr.web", description="Serve the ANPR API + dashboard")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--host", help="override web.host")
    ap.add_argument("--port", type=int, help="override web.port")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    logging.basicConfig(
        level=cfg.runtime.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not cfg.web.enabled:
        log.info("web.enabled is false: the dashboard/API is turned off (headless). Exiting.")
        return 0
    if not any(importlib.util.find_spec(m) for m in ("websockets", "wsproto")):
        log.warning(
            "no WebSocket library installed (websockets/wsproto): /ws/plates is unavailable; "
            "the dashboard falls back to polling the REST API"
        )
    if cfg.web.auth_token is None and (args.host or cfg.web.host) not in ("127.0.0.1", "localhost", "::1"):
        log.warning("web.auth_token is not set: anyone on the network can read plates")

    from pathlib import Path

    from anpr.licence import LICENCE_FILE

    # Activation saves the licence where the engine looks for it: next to the config file.
    app = create_app(cfg, licence_path=Path(args.config).resolve().parent / LICENCE_FILE)
    uvicorn.run(
        app,
        host=args.host or cfg.web.host,
        port=args.port or cfg.web.port,
        workers=1,
        log_level=cfg.runtime.log_level.lower(),
        access_log=False,  # saves SD-card/journal writes and keeps ?token= out of the logs
        proxy_headers=False,
        server_header=False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
