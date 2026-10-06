"""Step 9: FastAPI app — REST search/history, CSV export, JPEG serving, live WebSocket, /health.

Reads events only; the engine process is their single writer. The one thing the web writes is the
stream request (POST /api/v1/source), which the engine picks up. Every SQLite call runs off the event loop
(sync endpoints run in Starlette's threadpool; the WebSocket loop uses run_in_threadpool).
"""

from __future__ import annotations

import asyncio
import contextlib
import csv
import dataclasses
import io
import logging
import math
import re
import secrets
import shutil
import time
from collections.abc import AsyncIterator, Callable, Iterator
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

from fastapi import APIRouter, Depends, FastAPI, Query, Request, WebSocket
from fastapi import Path as PathParam
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.websockets import WebSocketDisconnect

from anpr import licence as lic_mod
from anpr import push
from anpr.config import AppConfig
from anpr.licence import STATUS_KEY as LICENCE_STATUS_KEY
from anpr.sources import SourceError, mask_source, normalize_source, source_kind
from anpr.storage import SqliteEventStore
from anpr.types import EngineStatus, PlateEvent
from anpr.validator import format_display

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
API_PREFIX = "/api/v1"
MAX_PER_PAGE = 200
CSV_MAX_ROWS = 100_000
CSV_BATCH = 1000
CSV_COLUMNS = (
    "id",
    "plate",
    "plate_display",
    "kind",
    "confidence",
    "votes",
    "first_seen_iso",
    "last_seen_iso",
    "crop_path",
    "snapshot_path",
    "hsrp",
)
WS_BATCH = 100
WS_POLICY_VIOLATION = 1008
WS_TRY_AGAIN_LATER = 1013
IMAGE_SUFFIXES = {".jpg", ".jpeg"}
PREVIEW_FRESH_S = 6.0  # a live preview older than this (or 4 intervals) means the picture stopped
CPU_TEMP_FILE = Path("/sys/class/thermal/thermal_zone0/temp")

_EPOCH_RE = re.compile(r"[+-]?\d+(\.\d+)?")
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_ERROR_CODES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    422: "validation_error",
}

_SECURITY_HEADERS = [
    (b"x-content-type-options", b"nosniff"),
    (b"referrer-policy", b"no-referrer"),
    (b"x-frame-options", b"DENY"),
    (
        b"content-security-policy",
        b"default-src 'self'; img-src 'self' data:; connect-src 'self' ws: wss:; "
        b"style-src 'self'; script-src 'self'; frame-ancestors 'none'; base-uri 'none'",
    ),
]


# ---- helpers ------------------------------------------------------------------------------------


def iso_local(ts: float) -> str:
    """Unix seconds -> ISO-8601 in the Pi's local time zone, with offset (e.g. +05:30)."""
    return datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")


def parse_time(value: str | None, name: str, *, end_of_day: bool = False) -> float | None:
    """ISO date / ISO datetime (naive = local time) / unix epoch seconds -> unix seconds.

    A bare date means the start of that local day, or its last instant when `end_of_day` is set
    (so `to=2026-09-29` includes the whole day). Raises a 422 on anything else.
    """
    if value is None or not value.strip():
        return None
    s = value.strip()
    try:
        if _EPOCH_RE.fullmatch(s):
            ts = float(s)
        elif _DATE_RE.fullmatch(s):
            day = date.fromisoformat(s)
            if end_of_day:
                ts = datetime.combine(day + timedelta(days=1), datetime.min.time()).timestamp() - 1e-6
            else:
                ts = datetime.combine(day, datetime.min.time()).timestamp()
        else:
            ts = datetime.fromisoformat(s).timestamp()  # naive -> local time
    except (ValueError, OverflowError, OSError) as exc:
        raise HTTPException(422, f"'{name}' must be an ISO-8601 date/datetime or unix epoch seconds") from exc
    if not math.isfinite(ts) or not (0 <= ts < 1e11):
        raise HTTPException(422, f"'{name}' is out of range")
    return ts


def event_json(e: PlateEvent) -> dict[str, Any]:
    return {
        "id": e.id,
        "plate": e.plate,
        "plate_display": format_display(e.plate, e.kind),
        "kind": e.kind,
        "confidence": round(float(e.confidence), 4),
        "votes": e.votes,
        "track_id": e.track_id,
        "first_seen": e.first_seen,
        "last_seen": e.last_seen,
        "first_seen_iso": iso_local(e.first_seen),
        "last_seen_iso": iso_local(e.last_seen),
        "crop_path": e.crop_path,
        "snapshot_path": e.snapshot_path,
        "hsrp": e.hsrp,
        "crop_url": f"/images/{quote(e.crop_path)}" if e.crop_path else None,
        "snapshot_url": f"/images/{quote(e.snapshot_path)}" if e.snapshot_path else None,
    }


def read_cpu_temp_c() -> float | None:
    try:
        return round(int(CPU_TEMP_FILE.read_text().strip()) / 1000.0, 1)
    except (OSError, ValueError):
        return None  # not a Pi (or no thermal zone)


def preview_info(store: SqliteEventStore, interval_s: float, now: float) -> dict[str, Any]:
    """How old the live preview JPEG is. `fresh` = written within a few preview intervals, i.e. the
    engine is still getting frames (it only writes one right after a new frame)."""
    ts = store.preview_mtime()
    age = None if ts is None else max(0.0, now - ts)
    return {
        "preview_ts": ts,
        "preview_age_s": None if age is None else round(age, 1),
        "preview_fresh": age is not None and interval_s > 0 and age <= max(PREVIEW_FRESH_S, 4 * interval_s),
    }


def build_health(
    store: SqliteEventStore,
    stale_after_s: float,
    now: float | None = None,
    preview_interval_s: float = 1.0,
) -> dict[str, Any]:
    now = time.time() if now is None else now
    status: EngineStatus | None = store.read_status()
    age = None if status is None else max(0.0, now - status.ts)
    engine_ok = status is not None and age is not None and age <= stale_after_s
    if status is None:
        state = "no_engine"
    elif not engine_ok:
        state = "stale"
    elif status.source_state == "ended":
        state = "source_ended"  # a video file finished; waiting for a new stream
    elif status.source_state == "error":
        state = "source_error"  # the stream could not be opened
    elif not status.camera_ok:
        state = "camera_down"
    else:
        state = "ok"
    disk: dict[str, float | None] = {"free_mb": None, "total_mb": None, "free_pct": None}
    try:
        du = shutil.disk_usage(store.image_dir)
        disk = {
            "free_mb": round(du.free / 1e6, 1),
            "total_mb": round(du.total / 1e6, 1),
            "free_pct": round(100.0 * du.free / du.total, 1) if du.total else None,
        }
    except OSError:
        pass
    return {
        "status": state,
        "ok": state == "ok",
        "engine_ok": engine_ok,
        "engine_age_s": None if age is None else round(age, 1),
        "stale_after_s": stale_after_s,
        "camera_ok": None if status is None else status.camera_ok,
        "fps": None if status is None else round(status.fps, 2),
        "frames": None if status is None else status.frames,
        "events": None if status is None else status.events,
        "rss_mb": None if status is None else round(status.rss_mb, 1),
        "last_error": None if status is None else status.last_error,
        # No stream address here: /health is open, and addresses (camera IPs, paths) are not.
        "source_state": None if status is None else status.source_state,
        "source_rev": None if status is None else status.source_rev,
        **preview_info(store, preview_interval_s, now),  # times only; the picture needs the token
        "cpu_temp_c": read_cpu_temp_c(),
        "disk_free_mb": disk["free_mb"],
        "disk_total_mb": disk["total_mb"],
        "disk_free_pct": disk["free_pct"],
        "latest_id": store.latest_id(),
        "server_time": now,
    }


class LicenceIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(..., min_length=1, max_length=1_000_000)


class PushIn(BaseModel):
    """Settings form: where to send plates (the client's API)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    url: str = Field("", max_length=push.MAX_URL)
    header_name: str = Field("Authorization", max_length=64)
    header_value: str | None = Field(None, max_length=push.MAX_HEADER_VALUE)  # None/"" = keep the saved key
    device_id: str = Field("", max_length=40)
    include_images: bool = True


class SourceIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field("", max_length=2048)  # "" = the camera from config.yaml


class AdminRequired(HTTPException):
    """The admin password (web.admin_token) is missing or wrong: the dashboard asks for it."""

    def __init__(self, message: str = "admin password required") -> None:
        super().__init__(403, message)


def _error(status: int, message: str, headers: dict[str, str] | None = None, **extra: Any) -> JSONResponse:
    body: dict[str, Any] = {"code": _ERROR_CODES.get(status, "http_error"), "message": message, **extra}
    return JSONResponse({"error": body}, status_code=status, headers=headers)


class _SecurityHeaders:
    """Pure-ASGI middleware (no BaseHTTPMiddleware buffering, so CSV streaming stays streamed)."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message: dict) -> None:
            if message["type"] == "http.response.start":
                names = {k.lower() for k, _ in message.get("headers", [])}
                extra = [(k, v) for k, v in _SECURITY_HEADERS if k not in names]
                message["headers"] = [*message.get("headers", []), *extra]
            await send(message)

        await self.app(scope, receive, send_wrapper)


# ---- app ----------------------------------------------------------------------------------------


def create_app(
    cfg: AppConfig,
    store: SqliteEventStore | None = None,
    *,
    ws_poll_s: float = 1.0,
    ws_status_s: float = 5.0,
    stale_after_s: float | None = None,
    max_ws_clients: int = 16,
    licence_path: Path | None = None,
    machine_id: Callable[[], str] | None = None,
) -> FastAPI:
    """Build the web app. If `store` is None one is opened from cfg.storage and closed on shutdown
    (a store passed in stays owned by the caller). `licence_path`: where Activate saves the licence
    (None = activation not offered)."""
    owns_store = store is None
    if store is None:
        store = SqliteEventStore(cfg.storage.db_path, cfg.storage.image_dir, cfg.storage.snapshot_width)
    st: SqliteEventStore = store
    token = cfg.web.auth_token or None
    stale_s = stale_after_s if stale_after_s is not None else max(10.0, 5 * cfg.runtime.status_interval_s)
    preview_every = cfg.runtime.preview_interval_s

    def health_body() -> dict[str, Any]:
        return build_health(st, stale_s, preview_interval_s=preview_every)

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            if owns_store:
                st.close()

    app = FastAPI(
        title="ANPR",
        version="1",
        lifespan=lifespan,
        docs_url=None,  # nothing extra on the Pi; /openapi.json still documents the API
        redoc_url=None,
    )
    app.state.store = st
    app.state.ws_clients = 0
    app.add_middleware(_SecurityHeaders)

    # ---- errors ---------------------------------------------------------------------------------

    @app.exception_handler(HTTPException)
    async def _http_error(_req: Request, exc: HTTPException) -> JSONResponse:
        extra = {"code": "admin_required"} if isinstance(exc, AdminRequired) else {}
        return _error(exc.status_code, str(exc.detail), headers=getattr(exc, "headers", None), **extra)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_req: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {"loc": [str(p) for p in err.get("loc", ())], "msg": str(err.get("msg", ""))}
            for err in exc.errors()
        ]
        msg = "; ".join(f"{'.'.join(d['loc'][1:]) or d['loc'][0]}: {d['msg']}" for d in details)
        return _error(422, msg or "invalid request", details=details)

    # ---- auth -----------------------------------------------------------------------------------

    def _token_ok(authorization: str | None, query_token: str | None) -> bool:
        if token is None:
            return True
        supplied = None
        if authorization:
            scheme, _, value = authorization.partition(" ")
            if scheme.lower() == "bearer":
                supplied = value.strip()
        if not supplied and query_token:
            supplied = query_token
        return bool(supplied) and secrets.compare_digest(supplied.encode(), token.encode())

    def require_auth(request: Request) -> None:
        if not _token_ok(request.headers.get("authorization"), request.query_params.get("token")):
            raise HTTPException(401, "missing or invalid token", headers={"WWW-Authenticate": "Bearer"})

    # ---- API ------------------------------------------------------------------------------------

    api = APIRouter(prefix=API_PREFIX, dependencies=[Depends(require_auth)])

    def _range(from_: str | None, to: str | None) -> tuple[float | None, float | None]:
        since = parse_time(from_, "from")
        until = parse_time(to, "to", end_of_day=True)
        if since is not None and until is not None and since > until:
            raise HTTPException(422, "'from' must not be after 'to'")
        return since, until

    @api.get("/plates")
    def list_plates(
        q: str | None = Query(None, max_length=32, description="plate substring (spaces/case ignored)"),
        from_: str | None = Query(None, alias="from", max_length=40),
        to: str | None = Query(None, max_length=40),
        page: int = Query(1, ge=1, le=1_000_000),
        per_page: int = Query(50, ge=1, le=MAX_PER_PAGE),
    ) -> dict[str, Any]:
        since, until = _range(from_, to)
        events, total = st.list_events(q, since, until, limit=per_page, offset=(page - 1) * per_page)
        return {
            "data": [event_json(e) for e in events],
            "meta": {
                "total": total,
                "page": page,
                "per_page": per_page,
                "total_pages": math.ceil(total / per_page),
            },
        }

    # Declared BEFORE /plates/{event_id} so "export.csv" is never parsed as an id.
    @api.get("/plates/export.csv")
    def export_csv(
        q: str | None = Query(None, max_length=32),
        from_: str | None = Query(None, alias="from", max_length=40),
        to: str | None = Query(None, max_length=40),
    ) -> StreamingResponse:
        since, until = _range(from_, to)
        max_id = st.latest_id()  # snapshot: rows inserted during the export are left out

        def rows() -> Iterator[str]:
            buf = io.StringIO()
            w = csv.writer(buf, lineterminator="\r\n")

            def flush() -> str:
                out = buf.getvalue()
                buf.seek(0)
                buf.truncate()
                return out

            w.writerow(CSV_COLUMNS)
            yield flush()
            offset = sent = 0
            prev_ids: set[int] = set()
            while sent < CSV_MAX_ROWS:
                limit = min(CSV_BATCH, CSV_MAX_ROWS - sent)
                batch, _ = st.list_events(q, since, until, limit=limit, offset=offset)
                ids: set[int] = set()
                for e in batch:
                    # New rows shift the newest-first offsets: skip them and any repeats they cause.
                    if e.id is None or e.id > max_id or e.id in prev_ids or sent >= CSV_MAX_ROWS:
                        continue
                    ids.add(e.id)
                    sent += 1
                    w.writerow(
                        (
                            e.id,
                            e.plate,
                            format_display(e.plate, e.kind),
                            e.kind,
                            f"{e.confidence:.4f}",
                            e.votes,
                            iso_local(e.first_seen),
                            iso_local(e.last_seen),
                            e.crop_path or "",
                            e.snapshot_path or "",
                            e.hsrp or "",
                        )
                    )
                if ids:
                    yield flush()
                if len(batch) < limit:
                    break
                offset += len(batch)
                prev_ids = ids

        stamp = time.strftime("%Y%m%d-%H%M%S")
        return StreamingResponse(
            rows(),
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="anpr-plates-{stamp}.csv"',
                "Cache-Control": "no-store",
            },
        )

    @api.get("/plates/{event_id}")
    def get_plate(event_id: int = PathParam(..., ge=1, le=2**63 - 1)) -> dict[str, Any]:
        e = st.get_event(event_id)
        if e is None:
            raise HTTPException(404, f"plate event {event_id} not found")
        return {"data": event_json(e)}

    @api.get("/stats")
    def stats(hours: int = Query(24, ge=1, le=168)) -> dict[str, Any]:
        """Dashboard figures: hourly counts for the last `hours` (aligned to whole hours, oldest
        first), today's totals (local midnight onwards) and the most frequent plates in the window."""
        now = time.time()
        # Buckets follow LOCAL clock hours (India is UTC+5:30, so UTC-aligned hours would start at :30).
        hour_start = datetime.fromtimestamp(now).replace(minute=0, second=0, microsecond=0)
        end = (hour_start + timedelta(hours=1)).timestamp()  # end of the current local hour
        start = end - hours * 3600.0
        win = st.stats(start, end, bucket_s=3600.0)
        midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        today = st.stats(midnight, max(now, midnight) + 1.0, bucket_s=86400.0, top_n=0)
        last_hour = st.stats(now - 3600.0, now + 1.0, bucket_s=3600.0, top_n=0)
        return {
            "data": {
                "window": {"start": start, "end": end, "hours": hours, "bucket_s": 3600},
                "buckets": [{"start": start + i * 3600.0, "count": n} for i, n in enumerate(win["buckets"])],
                "window_total": win["total"],
                "window_unique": win["unique"],
                "avg_confidence": win["avg_confidence"],
                "today_total": today["total"],
                "today_unique": today["unique"],
                "last_hour_total": last_hour["total"],
                "top": [{**t, "plate_display": format_display(t["plate"], t["kind"])} for t in win["top"]],
            }
        }

    @api.get("/info")
    def info() -> dict[str, Any]:
        """Static facts for the dashboard header and footer."""
        return {
            "data": {
                "station": cfg.web.station_name,
                "read_only": cfg.web.read_only,
                "admin_required": cfg.web.admin_token is not None,
                "min_votes": cfg.vote.min_votes,
                "min_confidence": cfg.vote.min_avg_conf,
                "retention_days": cfg.storage.retention_days,
                "detector": cfg.detector.backend,
                "preview_interval_s": preview_every,  # 0 = live view switched off in config.yaml
                "licence": _licence_view(),
            }
        }

    def _licence_view() -> dict[str, Any] | None:
        """The engine's last licence check (who it is licensed to, until when), for the footer."""
        v = st.read_setting(LICENCE_STATUS_KEY)
        if not isinstance(v, dict):
            return None
        keep = ("ok", "error", "customer", "expires", "days_left", "licence_id")
        return {k: v.get(k) for k in keep}

    # ---- licence activation (dashboard "Activate") ------------------------------------------------

    read_machine_id = machine_id or lic_mod.machine_id
    machine_cache: list[str] = []

    def _machine() -> str | None:
        if not machine_cache:
            try:
                machine_cache.append(read_machine_id())
            except lic_mod.LicenceError:
                return None
        return machine_cache[0]

    @api.get("/licence")
    def get_licence() -> dict[str, Any]:
        """This device's machine ID (to send to the supplier) and its licence state."""
        return {
            "data": {
                "machine_id": _machine(),
                "status": _licence_view(),
                "can_activate": licence_path is not None and not cfg.web.read_only,
                "admin_required": admin_token is not None,
            }
        }

    @api.post("/licence")
    def activate(body: LicenceIn, request: Request) -> dict[str, Any]:
        """Paste a licence key: checked for THIS device and today, then saved. The engine starts (or
        switches to the new licence) by itself within seconds."""
        _require_admin(request)
        if licence_path is None:
            raise HTTPException(403, "activation is not available on this dashboard")
        if admin_token is None and token is None and not _local_request(request):
            raise HTTPException(
                403,
                "this dashboard has no admin password: activate on the device itself, or set "
                "web.admin_token in config.yaml",
            )
        dev = _machine()
        if dev is None:
            raise HTTPException(422, "cannot read this device's machine ID")
        try:
            lic = lic_mod.install_licence(body.key, licence_path, dev)
        except lic_mod.LicenceError as e:
            raise HTTPException(422, str(e)) from None
        now = time.time()
        st.write_setting(LICENCE_STATUS_KEY, lic_mod.summary(lic, dev, now, None))
        log.info("licence %s for %s activated from the dashboard", lic.licence_id, lic.customer)
        return {
            "data": {
                "machine_id": dev,
                "status": _licence_view(),
                "can_activate": True,
                "admin_required": admin_token is not None,
            }
        }

    # ---- stream (dashboard "Add stream") ---------------------------------------------------------

    # "Back to default" plays the config playlist when there is one, else camera.source.
    default_source = cfg.camera.playlist[0] if cfg.camera.playlist else cfg.camera.source
    default_shown = " -> ".join(mask_source(p) for p in cfg.camera.playlist) or mask_source(cfg.camera.source)

    def _source_view(now: float | None = None) -> dict[str, Any]:
        now = time.time() if now is None else now
        status = st.read_status()
        req = st.read_source_request()
        running = status is not None and now - status.ts <= stale_s
        requested = None if req is None else (req[1] or default_source)
        current = None if status is None else status.source
        return {
            "current": current,
            "current_kind": source_kind(current) if current else None,
            "state": None if status is None else status.source_state,
            "error": status.last_error if status is not None and status.source_state == "error" else None,
            "engine_running": running,
            "camera_ok": None if status is None else status.camera_ok,
            "requested": None if requested is None else mask_source(requested),
            "requested_rev": None if req is None else req[0],
            # True until the engine confirms it switched (or picks it up when it next starts).
            "pending": req is not None and (status is None or status.source_rev < req[0]),
            "default": default_shown,
            "fps": None if status is None else round(status.fps, 2),
            "preview_interval_s": preview_every,
            **preview_info(st, preview_every, now),
        }

    @api.get("/source")
    def get_source() -> dict[str, Any]:
        """The stream the engine is reading now, and the last one asked for on the dashboard."""
        return {"data": _source_view()}

    @api.post("/source", status_code=202)
    def set_source(body: SourceIn, request: Request) -> dict[str, Any]:
        """Switch the engine to a video file path or an rtsp:// link ("" = config camera). The
        engine applies it within about a second and keeps it across restarts."""
        _require_admin(request)
        try:
            address = normalize_source(body.source)
        except SourceError as e:
            raise HTTPException(422, str(e)) from None
        rev = st.request_source(address)
        shown = mask_source(address) if address else default_shown
        log.info("stream change requested (#%d): %s", rev, shown if address else f"config camera {shown}")
        return {"data": {"rev": rev, "source": shown, "kind": source_kind(address or default_source)}}

    # ---- sending plates to the client's API (dashboard "Settings") ---------------------------------

    admin_token = cfg.web.admin_token or None

    def _require_admin(request: Request) -> None:
        """Changing things: not on a view-only dashboard, not from another website, and with the
        admin password when one is set (web.admin_token)."""
        if cfg.web.read_only:
            raise HTTPException(403, "this dashboard is view-only")
        origin = request.headers.get("origin")
        if origin and urlsplit(origin).netloc.lower() != (request.headers.get("host") or "").lower():
            raise HTTPException(403, "cross-site request refused")
        if admin_token is not None:
            given = request.headers.get("x-admin-token", "")
            if not given:
                raise AdminRequired()
            if not secrets.compare_digest(given.encode(), admin_token.encode()):
                raise AdminRequired("wrong admin password")

    def _local_request(request: Request) -> bool:
        host = request.client.host if request.client else ""
        return host in ("127.0.0.1", "::1", "localhost")

    def _push_from_form(body: PushIn, saved: push.PushSettings) -> push.PushSettings:
        try:
            url = push.clean_url(body.url)
            header_name = push.clean_header_name(body.header_name)
            value = push.clean_header_value(body.header_value or "")
            device_id = push.clean_device_id(body.device_id)
        except push.PushError as e:
            raise HTTPException(422, str(e)) from None
        if body.enabled and not url:
            raise HTTPException(422, "enter the client's API address to switch sending on")
        if not header_name:
            value = ""  # no header: no key
        elif not value:
            value = saved.header_value  # field left empty: keep the saved key
        return dataclasses.replace(
            saved,
            enabled=body.enabled,
            url=url,
            header_name=header_name,
            header_value=value,
            device_id=device_id,
            include_images=body.include_images,
        )

    def _push_view() -> dict[str, Any]:
        s = push.load_settings(st)
        ps = push.load_state(st)
        status = st.read_status()
        engine_running = status is not None and time.time() - status.ts <= stale_s
        started = ps.start_rev == s.start_rev  # the engine has picked up the latest switch-on
        cursor = ps.cursor if started else s.start_after_id
        return {
            "settings": {
                "enabled": s.enabled,
                "url": s.url,
                "header_name": s.header_name,
                "header_value_set": bool(s.header_value),
                "header_value_hint": push.secret_hint(s.header_value),
                "device_id": s.device_id,
                "device_id_default": push.default_device_id(),
                "include_images": s.include_images,
            },
            "status": {
                "engine_running": engine_running,
                "waiting": st.count_after(cursor) if s.enabled and s.url else 0,
                "sent_total": ps.sent_total if started else 0,
                "skipped_total": ps.skipped_total if started else 0,
                "last_ok_ts": ps.last_ok_ts if started else None,
                "last_error": ps.last_error if started else None,
                "last_error_ts": ps.last_error_ts if started else None,
                "last_reply": ps.last_reply if started else None,
                "last_reply_status": ps.last_reply_status if started else None,
                "last_reply_ts": ps.last_reply_ts if started else None,
            },
        }

    @api.get("/push")
    def get_push(request: Request) -> dict[str, Any]:
        """Where plates are sent and how sending is going. The saved key is never returned."""
        _require_admin(request)
        return {"data": _push_view()}

    @api.put("/push")
    def set_push(body: PushIn, request: Request) -> dict[str, Any]:
        """Save the client's API settings. Takes effect within a few seconds, no restart. Switching on
        (or changing the address) starts sending from the newest plate at that moment."""
        _require_admin(request)
        saved = push.load_settings(st)
        new = _push_from_form(body, saved)
        if new.enabled and (not saved.enabled or new.url != saved.url):
            new = dataclasses.replace(new, start_rev=saved.start_rev + 1, start_after_id=st.latest_id())
        push.save_settings(st, new)
        log.info("plate sending %s: %s", "on" if new.enabled else "off", new.url or "(no address)")
        return {"data": _push_view()}

    @api.post("/push/test")
    def test_push(body: PushIn, request: Request) -> dict[str, Any]:
        """Send one sample plate ("test": true) with the settings in the form (saved or not)."""
        _require_admin(request)
        s = _push_from_form(body.model_copy(update={"enabled": False}), push.load_settings(st))
        if not s.url:
            raise HTTPException(422, "enter the client's API address first")
        res = push.post_json(s.url, s.header_name, s.header_value, push.test_body(s, cfg.web.station_name))
        return {"data": {"ok": res.ok, "status": res.status, "message": res.message, "reply": res.reply}}

    @api.get(
        "/live.jpg",
        response_class=Response,
        responses={200: {"content": {"image/jpeg": {}}}, 404: {"description": "no live picture yet"}},
    )
    def live_jpg() -> Response:
        """The latest camera picture the engine processed (small JPEG, detector boxes drawn on it),
        refreshed every runtime.preview_interval_s. X-Captured-At = unix time it was written;
        compare with X-Server-Time (or /api/v1/source preview_fresh) to tell a stopped picture."""
        got = st.read_preview()
        if got is None:
            raise HTTPException(404, "no live picture yet: the engine writes one once it gets frames")
        data, ts = got
        now = time.time()
        return Response(
            data,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-store",
                "X-Captured-At": f"{ts:.3f}",
                "X-Server-Time": f"{now:.3f}",
                "X-Preview-Age": f"{max(0.0, now - ts):.1f}",
            },
        )

    app.include_router(api)

    # ---- images ---------------------------------------------------------------------------------

    @app.get("/images/{path:path}", dependencies=[Depends(require_auth)])
    def get_image(path: str) -> FileResponse:
        resolved = st.resolve_image(path) if path and "\x00" not in path else None
        if resolved is None or resolved.suffix.lower() not in IMAGE_SUFFIXES:
            raise HTTPException(404, "image not found")
        # Stored images are immutable (unique names), so let phones cache them.
        return FileResponse(
            resolved, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"}
        )

    # ---- health ---------------------------------------------------------------------------------

    @app.get("/health")
    def health() -> JSONResponse:
        """Open (no token) so monitors/systemd checks can use it. 200 when the engine heartbeat is
        fresh and the camera is OK, else 503 — the body is the same shape either way."""
        body = health_body()
        return JSONResponse(
            body, status_code=200 if body["ok"] else 503, headers={"Cache-Control": "no-store"}
        )

    # ---- live WebSocket -------------------------------------------------------------------------

    @app.websocket("/ws/plates")
    async def ws_plates(ws: WebSocket) -> None:
        if not _token_ok(ws.headers.get("authorization"), ws.query_params.get("token")):
            await ws.close(code=WS_POLICY_VIOLATION, reason="missing or invalid token")
            return
        since_raw = ws.query_params.get("since_id")
        last_id: int | None = None
        if since_raw not in (None, ""):
            try:
                last_id = int(since_raw)
                if last_id < 0:
                    raise ValueError
            except ValueError:
                await ws.close(code=WS_POLICY_VIOLATION, reason="since_id must be an integer >= 0")
                return
        if app.state.ws_clients >= max_ws_clients:
            await ws.close(code=WS_TRY_AGAIN_LATER, reason="too many live clients")
            return

        app.state.ws_clients += 1
        await ws.accept()
        gone = asyncio.Event()

        async def reader() -> None:  # we only need to notice the disconnect; client text is ignored
            try:
                while (await ws.receive())["type"] != "websocket.disconnect":
                    pass
            except Exception:  # noqa: BLE001 - any receive failure means the client is gone
                pass
            finally:
                gone.set()

        reader_task = asyncio.create_task(reader())
        loop = asyncio.get_running_loop()
        try:
            if last_id is None:
                last_id = await run_in_threadpool(st.latest_id)
            await ws.send_json({"type": "hello", "data": {"since_id": last_id}})
            await ws.send_json({"type": "status", "data": await run_in_threadpool(health_body)})
            next_status = loop.time() + ws_status_s
            while not gone.is_set():
                events = await run_in_threadpool(st.events_after, last_id, WS_BATCH)
                for e in events:
                    await ws.send_json({"type": "plate", "data": event_json(e)})
                    last_id = max(last_id, e.id or last_id)
                if len(events) == WS_BATCH:
                    continue  # catching up after a reconnect: drain without waiting
                if loop.time() >= next_status:
                    health_now = await run_in_threadpool(health_body)
                    await ws.send_json({"type": "status", "data": health_now})
                    next_status = loop.time() + ws_status_s
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(gone.wait(), timeout=ws_poll_s)
        except (WebSocketDisconnect, RuntimeError, OSError):
            pass  # client went away mid-send
        except Exception:
            log.exception("live websocket failed")
            with contextlib.suppress(Exception):
                await ws.close(code=1011)
        finally:
            app.state.ws_clients -= 1
            reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await reader_task

    # ---- dashboard ------------------------------------------------------------------------------

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=STATIC_DIR, check_dir=False), name="static")

    return app
