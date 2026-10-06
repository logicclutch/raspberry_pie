"""Local labelling web app for harvested plate crops (see training/harvest.py).

Localhost only: the server binds 127.0.0.1, rejects any other Host header (DNS rebinding) and any
cross-origin or non-JSON POST (CSRF). Images are served only from <data>/images. Every save re-reads
labels.csv under the shared lock, changes only the saved row(s) and replaces the file atomically.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import re
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote, urlsplit

from fastapi import FastAPI, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException

from anpr.config import ValidationConfig
from anpr.types import OcrResult
from anpr.validator import PlateValidator, format_display
from training.dataset import (
    IMAGES_DIR,
    PLATE_TEXT_RE,
    STATUSES,
    Row,
    locked,
    normalize_plate,
    read_rows,
    track_key,
    write_rows,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
IMAGE_SUFFIXES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
LABELER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._-]{0,39}")

_SECURITY_HEADERS = [
    (b"x-content-type-options", b"nosniff"),
    (b"referrer-policy", b"no-referrer"),
    (b"x-frame-options", b"DENY"),
    (b"cache-control", b"no-store"),
    (
        b"content-security-policy",
        b"default-src 'self'; img-src 'self'; connect-src 'self'; style-src 'self'; script-src 'self'; "
        b"frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
    ),
]


class _Guard:
    """Pure-ASGI middleware: localhost Host only, same-origin JSON POSTs only, security headers."""

    def __init__(self, app: Any, allowed_hosts: frozenset[str]) -> None:
        self.app = app
        self.allowed = allowed_hosts

    @staticmethod
    def _hostname(value: str) -> str:
        return (urlsplit("//" + value).hostname or "").lower()

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        host = headers.get("host", "")
        problem = None
        if self._hostname(host) not in self.allowed:
            problem = (400, "bad_host", "only localhost may use the labeler")
        elif scope["method"] not in ("GET", "HEAD"):
            origin = headers.get("origin")
            if origin and urlsplit(origin).netloc.lower() != host.lower():
                problem = (403, "forbidden", "cross-origin request refused")
            elif headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
                problem = (415, "unsupported_media_type", "send JSON")
        if problem is not None:
            resp = JSONResponse(
                {"error": {"code": problem[1], "message": problem[2]}}, status_code=problem[0]
            )
            await resp(scope, receive, self._wrap(send))
            return
        await self.app(scope, receive, self._wrap(send))

    @staticmethod
    def _wrap(send: Any) -> Any:
        async def send_wrapper(message: dict) -> None:
            if message["type"] == "http.response.start":
                names = {k.lower() for k, _ in message.get("headers", [])}
                extra = [(k, v) for k, v in _SECURITY_HEADERS if k not in names]
                message["headers"] = [*message.get("headers", []), *extra]
            await send(message)

        return send_wrapper


class LabelIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=512)  # the row's image_path
    status: Literal["unverified", "verified", "skip", "unreadable"]
    plate_text: str = Field("", max_length=32)
    labeler: str = Field("", max_length=40)
    confirm: bool = False  # verified text that is not a strict Indian format: human confirmed it
    rev: str = Field("", max_length=64)  # the row's `rev` as the client last saw it (see row_rev)


class TrackLabelIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=512)  # any row of the track (always updated)
    status: Literal["verified", "unreadable"] = "verified"
    plate_text: str = Field("", max_length=32)
    labeler: str = Field("", max_length=40)
    confirm: bool = False
    rev: str = Field("", max_length=64)  # rev of the `id` row as the client last saw it


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, **extra: Any) -> None:
        super().__init__(message)
        self.status, self.code, self.message, self.extra = status, code, message, extra


def now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def row_rev(r: Row) -> str:
    """Version of a row's human label. A save must send the rev it was based on: if another tab (or
    person) changed the label meanwhile, the save is refused instead of silently overwriting it."""
    blob = "\x1f".join(r[c] for c in ("status", "plate_text", "labeler", "labeled_at"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def create_app(data_dir: Path, *, allowed_hosts: frozenset[str] = LOCAL_HOSTS) -> FastAPI:
    data_dir = Path(data_dir).resolve()
    images_root = (data_dir / IMAGES_DIR).resolve()
    validator = PlateValidator(ValidationConfig(min_char_conf=0.0, max_repairs=2))

    def check_plate(text: str) -> dict[str, Any]:
        t = normalize_plate(text)
        res = validator.validate(OcrResult(t, (1.0,) * len(t))) if t else None
        valid = res is not None and res.text == t
        out: dict[str, Any] = {"text": t, "valid": valid, "kind": res.kind if valid and res else None}
        if valid and res is not None:
            out["display"] = format_display(res.text, res.kind)
            out["message"] = "Valid Indian format"
        elif not t:
            out["message"] = "Empty"
        elif res is not None:
            out["suggestion"] = res.text
            out["message"] = f"Not a strict Indian format (did you mean {res.text}?)"
        else:
            out["message"] = "Not a strict Indian format (standard SS NN X[X][X] NNNN or YY BH NNNN X[X])"
        return out

    def public(r: Row) -> dict[str, Any]:
        out: dict[str, Any] = dict(r)
        out["id"] = r["image_path"]
        out["image_url"] = "/img/" + quote(r["image_path"])
        out["raw_url"] = "/img/" + quote(r["raw_image_path"]) if r["raw_image_path"] else ""
        out["suggestion"] = r["ocr_plate"] or normalize_plate(r["ocr_text"])
        out["rev"] = row_rev(r)
        return out

    def check_rev(cur: Row, rev: str) -> None:
        # An untouched (unverified, never labelled) row may be saved without a rev; anything a human
        # already decided needs the matching rev, so a stale tab can never overwrite it by accident.
        untouched = cur["status"] == "unverified" and not cur["labeled_at"]
        if (rev or not untouched) and rev != row_rev(cur):
            raise ApiError(
                409,
                "conflict",
                "This crop was changed elsewhere (another tab?) since you loaded it. Check it, save again.",
                row=public(cur),
            )

    def sort_key(r: Row) -> tuple:
        def num(v: str) -> int:
            return int(v) if v.isdigit() else -1

        return (r["source"], r["video"], num(r["track"]), num(r["frame"]))

    def counts(rows: list[Row]) -> dict[str, Any]:
        c = dict.fromkeys(STATUSES, 0)
        for r in rows:
            c[r["status"] if r["status"] in c else "unverified"] += 1
        return {
            "counts": c,
            "total": len(rows),
            "sources": sorted({r["source"] for r in rows}),
            "two_line": sum(r["two_line"] == "1" for r in rows),
        }

    def labeled(r: Row, status: str, text: str, labeler: str) -> None:
        r["status"] = status
        r["plate_text"] = text if status == "verified" else ""
        r["labeler"] = labeler if status != "unverified" else ""
        r["labeled_at"] = now_iso() if status != "unverified" else ""

    def checked_input(status: str, plate_text: str, labeler: str, confirm: bool) -> tuple[str, str]:
        labeler = labeler.strip()
        if status != "unverified" and not LABELER_RE.fullmatch(labeler):
            raise ApiError(422, "labeler_required", "Enter your name (letters, digits, space . _ -)")
        text = normalize_plate(plate_text)
        if status == "verified":
            if not PLATE_TEXT_RE.fullmatch(text):
                raise ApiError(422, "bad_plate", "Type the plate text (A-Z, 0-9, up to 12 characters)")
            chk = check_plate(text)
            if not chk["valid"] and not confirm:
                raise ApiError(409, "needs_confirm", chk["message"], validation=chk)
        return text, labeler

    app = FastAPI(title="Plate labeler", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(_Guard, allowed_hosts=allowed_hosts)

    @app.exception_handler(ApiError)
    async def _api_error(_req: Any, exc: ApiError) -> JSONResponse:
        body = {"code": exc.code, "message": exc.message, **exc.extra}
        return JSONResponse({"error": body}, status_code=exc.status)

    @app.exception_handler(HTTPException)
    async def _http_error(_req: Any, exc: HTTPException) -> JSONResponse:
        return JSONResponse({"error": {"code": "http_error", "message": str(exc.detail)}}, exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_req: Any, exc: RequestValidationError) -> JSONResponse:
        msg = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:5])
        return JSONResponse({"error": {"code": "validation_error", "message": msg}}, status_code=422)

    @app.get("/api/summary")
    def summary() -> dict[str, Any]:
        return counts(read_rows(data_dir))

    @app.get("/api/rows")
    def list_rows(
        status: Literal["all", "unverified", "verified", "skip", "unreadable"] = "unverified",
        source: str = Query("", max_length=64),
        two_line: Literal["any", "0", "1"] = "any",
    ) -> dict[str, Any]:
        rows = read_rows(data_dir)
        sel = [
            r
            for r in rows
            if (status == "all" or r["status"] == status)
            and (not source or r["source"] == source)
            and (two_line == "any" or r["two_line"] == two_line)
        ]
        sel.sort(key=sort_key)
        return {"rows": [public(r) for r in sel], **counts(rows)}

    @app.get("/api/track")
    def get_track(id: str = Query(min_length=1, max_length=512)) -> dict[str, Any]:
        rows = read_rows(data_dir)
        cur = next((r for r in rows if r["image_path"] == id), None)
        if cur is None:
            raise ApiError(404, "not_found", "no such crop")
        tk = track_key(cur)
        return {"rows": [public(r) for r in sorted((r for r in rows if track_key(r) == tk), key=sort_key)]}

    @app.get("/api/validate")
    def validate(text: str = Query("", max_length=32)) -> dict[str, Any]:
        return check_plate(text)

    @app.post("/api/label")
    def save_label(body: LabelIn) -> dict[str, Any]:
        text, labeler = checked_input(body.status, body.plate_text, body.labeler, body.confirm)
        with locked(data_dir):
            rows = read_rows(data_dir)  # re-read: another tab may have saved other rows meanwhile
            cur = next((r for r in rows if r["image_path"] == body.id), None)
            if cur is None:
                raise ApiError(404, "not_found", "no such crop")
            check_rev(cur, body.rev)
            labeled(cur, body.status, text, labeler)
            write_rows(data_dir, rows)
        return {"row": public(cur), **counts(rows)}

    @app.post("/api/label-track")
    def save_track(body: TrackLabelIn) -> dict[str, Any]:
        text, labeler = checked_input(body.status, body.plate_text, body.labeler, body.confirm)
        with locked(data_dir):
            rows = read_rows(data_dir)
            cur = next((r for r in rows if r["image_path"] == body.id), None)
            if cur is None:
                raise ApiError(404, "not_found", "no such crop")
            check_rev(cur, body.rev)
            tk = track_key(cur)
            changed = [r for r in rows if r is cur or (track_key(r) == tk and r["status"] == "unverified")]
            for r in changed:
                labeled(r, body.status, text, labeler)
            write_rows(data_dir, rows)
        return {"updated": len(changed), "rows": [public(r) for r in changed], **counts(rows)}

    @app.get("/img/{path:path}")
    def image(path: str) -> FileResponse:
        target = None
        if path and "\x00" not in path and not path.startswith(("/", "\\")):
            p = (data_dir / path).resolve()
            if p.is_relative_to(images_root) and p.suffix.lower() in IMAGE_SUFFIXES and p.is_file():
                target = p
        if target is None:
            raise ApiError(404, "not_found", "no such image")
        return FileResponse(target, media_type=IMAGE_SUFFIXES[target.suffix.lower()])

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
