"""Licence Manager: the vendor's dashboard for making and keeping customer licences.

    .venv/bin/python tools/licence_dashboard.py          # then open the link it prints

VENDOR ONLY. It uses the private signing key (~/anpr-licensing/vendor_key.json), so it:
  - listens on this computer only (127.0.0.1), never on the network or a tunnel;
  - needs the secret link printed at start (a new one every start);
  - refuses requests that do not come from its own page (other websites, DNS rebinding).
Never put it behind Cloudflare or on a server. Never ship it to a client.

One licence per customer: it lists all of that customer's machine IDs, and the same licence key is
pasted on every one of their devices (dashboard -> Activate). Same signing as tools/licence_tool.py.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hmac
import json
import logging
import secrets
import sys
import webbrowser
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import licence_tool as lt  # noqa: E402
from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, Response  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field  # noqa: E402

from anpr import licence as lc  # noqa: E402

log = logging.getLogger("licence-dashboard")

STATIC = Path(__file__).resolve().parent / "licence_dashboard_static"
LOGO = ROOT / "anpr" / "web" / "static" / "logicclutch-logo.png"
LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]", "::1"}
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self' data:; script-src 'self'; style-src 'self'; "
        "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}
VALIDITY_YEARS = {"1y": 1, "2y": 2, "3y": 3, "5y": 5}


class IssueIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer: str = Field(..., max_length=120)
    machines: str = Field(..., max_length=1_000_000)  # 5000 machines with labels
    key_type: str = Field("customer", pattern=r"^(customer|machine)$")  # one key for all | one per machine
    validity: str = Field("1y", pattern=r"^(1y|2y|3y|5y|date|never)$")
    expires: str | None = Field(None, max_length=10)  # YYYY-MM-DD when validity == "date"
    note: str = Field("", max_length=500)


class MachinesIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    machines: str = Field(..., max_length=1_000_000)  # 5000 machines with labels


class CheckIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(..., max_length=1_000_000)
    machine_id: str = Field("", max_length=80)


def _raw_body(text: str) -> dict:
    """The licence fields of an already-verified key or file, including ones the device ignores."""
    try:
        t = lc.key_to_json(text) if text.strip()[:6].upper() == lc.KEY_PREFIX.upper() else text
        body = json.loads(t).get("licence")
        return body if isinstance(body, dict) else {}
    except (lc.LicenceError, ValueError, AttributeError):
        return {}


def add_years(d: dt.date, years: int) -> dt.date:
    """Same day `years` later (29 Feb -> 28 Feb), minus one day: the licence's last valid day."""
    try:
        end = d.replace(year=d.year + years)
    except ValueError:
        end = d.replace(year=d.year + years, day=28)
    return end - dt.timedelta(days=1)


def end_date(validity: str, expires: str | None, today: dt.date) -> dt.date | None:
    if validity == "never":
        return None
    if validity == "date":
        try:
            return dt.date.fromisoformat(expires or "")
        except ValueError:
            raise lt.IssueError("choose the last valid day") from None
    return add_years(today, VALIDITY_YEARS[validity])


def _status(expires: str | None, today: dt.date) -> tuple[str, int | None]:
    if not expires:
        return "active", None
    days_left = (dt.date.fromisoformat(expires) - today).days + 1  # counting today
    if days_left <= 0:
        return "ended", 0
    return ("expiring" if days_left <= lc.WARN_DAYS else "active"), days_left


def licence_view(rec: dict, today: dt.date) -> dict[str, Any]:
    """One licence for all of a customer's machines (key_type "customer")."""
    body = rec["body"]
    if not rec["valid"] or body is None:
        return {"file": rec["file"].name, "valid": False, "error": rec["error"]}
    expires = body.get("expires")
    status, days_left = _status(expires, today)
    return {
        "valid": True,
        "key_type": "customer",
        "licence_id": body["licence_id"],
        "customer": body["customer"],
        "issued": body["issued"],
        "expires": expires,
        "devices": body["devices"],
        "note": body.get("note", ""),
        "labels": body.get("device_labels") if isinstance(body.get("device_labels"), dict) else {},
        "status": status,
        "days_left": days_left,
        "key": rec["key"],
    }


def batch_view(recs: list[dict], today: dt.date) -> dict[str, Any]:
    """One licence per machine, made together (key_type "machine"): one row, with every machine's key."""
    first = recs[0]["body"]
    items = [
        {
            "licence_id": r["body"]["licence_id"],
            "device": r["body"]["devices"][0],
            "label": r["body"].get("device_label", ""),
            "key": r["key"],
        }
        for r in sorted(recs, key=lambda r: r["body"]["licence_id"])
    ]
    status, days_left = _status(first.get("expires"), today)
    return {
        "valid": True,
        "key_type": "machine",
        "licence_id": first["batch"],
        "customer": first["customer"],
        "issued": first["issued"],
        "expires": first.get("expires"),
        "devices": [i["device"] for i in items],
        "labels": {i["device"]: i["label"] for i in items if i["label"]},
        "note": first.get("note", ""),
        "status": status,
        "days_left": days_left,
        "items": items,
        "key": None,
    }


def grouped(records: list[dict], today: dt.date) -> list[dict[str, Any]]:
    """Issued licences for the list: per-machine keys of one batch as a single entry, newest first."""
    out: list[dict[str, Any]] = []
    batches: dict[str, list[dict]] = {}
    for r in records:
        batch = (r["body"] or {}).get("batch") if r["valid"] else None
        if isinstance(batch, str) and lt._LICENCE_ID_RE.match(batch):
            if batch not in batches:
                batches[batch] = []
                out.append({"_batch": batch})
            batches[batch].append(r)
        else:
            out.append(licence_view(r, today))
    return [batch_view(batches[v["_batch"]], today) if "_batch" in v else v for v in out]


def batch_rows(view: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "customer": view["customer"],
            "device": i["device"],
            "label": i["label"],
            "licence_id": i["licence_id"],
            "expires": view["expires"],
            "key": i["key"],
        }
        for i in view["items"]
    ]


def create_app(key_path: Path, token: str, *, today: Any = dt.date.today) -> FastAPI:
    app = FastAPI(title="Licence Manager", docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def guard(request: Request, call_next):  # noqa: ANN001, ANN202
        host = (request.headers.get("host") or "").rsplit(":", 1)[0].lower()
        if host not in LOCAL_HOSTS:  # DNS rebinding: a website renamed to point at 127.0.0.1
            return JSONResponse({"error": "only usable on this computer"}, status_code=403)
        if request.url.path.startswith("/api/"):
            origin = request.headers.get("origin")
            if origin and urlsplit(origin).netloc.lower() != (request.headers.get("host") or "").lower():
                return JSONResponse({"error": "cross-site request refused"}, status_code=403)
            if not hmac.compare_digest(request.headers.get("x-token", ""), token):
                return JSONResponse(
                    {"error": "open the link printed when the Licence Manager started"}, status_code=401
                )
        resp: Response = await call_next(request)
        for k, v in SECURITY_HEADERS.items():
            resp.headers.setdefault(k, v)
        return resp

    def fail(e: Exception) -> HTTPException:
        return HTTPException(422, str(e))

    @app.exception_handler(HTTPException)
    async def http_error(_req: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    @app.get("/api/info")
    def info() -> dict[str, Any]:
        try:
            _, pub = lt.load_key(key_path)
            key_ok, key_msg = True, None
            matches = pub == lc.vendor_public_key()
        except lt.IssueError as e:
            key_ok, key_msg, matches = False, str(e), False
        return {
            "key_path": str(key_path),
            "key_ok": key_ok,
            "key_error": key_msg,
            "key_matches_software": matches,
            "today": today().isoformat(),
            "warn_days": lc.WARN_DAYS,
        }

    @app.get("/api/licences")
    def licences() -> dict[str, Any]:
        return {"licences": grouped(lt.list_issued(key_path), today())}

    @app.post("/api/machines")
    def machines(body: MachinesIn) -> dict[str, Any]:
        """Live check of the machine ID box while typing."""
        try:
            entries = lt.parse_device_entries(body.machines)
        except lt.IssueError as e:
            return {"ok": False, "count": 0, "labelled": 0, "error": str(e)}
        return {
            "ok": True,
            "count": len(entries),
            "labelled": sum(1 for _, lb in entries if lb),
            "error": None,
        }

    @app.post("/api/licences")
    def issue(body: IssueIn) -> dict[str, Any]:
        t = today()
        try:
            entries = lt.parse_device_entries(body.machines)
            expires = end_date(body.validity, body.expires, t)
            if body.key_type == "machine":
                b = lt.issue_per_machine(key_path, body.customer, entries, expires, note=body.note, today=t)
            else:
                r = lt.issue_licence(
                    key_path,
                    body.customer,
                    [d for d, _ in entries],
                    expires,
                    note=body.note,
                    today=t,
                    labels=dict(entries),
                )
        except lt.IssueError as e:
            raise fail(e) from None
        if body.key_type == "machine":
            recs = [
                {"body": i["body"], "key": i["key"], "valid": True, "error": None, "file": i["record"]}
                for i in b["items"]
            ]
            log.info(
                "issued batch %s for %s (%d keys, one per machine)",
                b["batch"],
                body.customer.strip(),
                len(recs),
            )
            return {"licence": batch_view(recs, t)}
        log.info(
            "issued %s for %s (%d machines)",
            r["body"]["licence_id"],
            r["body"]["customer"],
            len(r["body"]["devices"]),
        )
        rec = {"body": r["body"], "key": r["key"], "valid": True, "error": None, "file": r["record"]}
        return {"licence": licence_view(rec, t)}

    @app.get("/api/licences/{licence_id}/file")
    def download(licence_id: str) -> Response:
        if not lt._LICENCE_ID_RE.match(licence_id):
            raise HTTPException(404, "no such licence")
        f = lt.issued_dir(key_path) / f"{licence_id}.json"
        if not f.is_file():
            raise HTTPException(404, "no such licence")
        return Response(
            f.read_bytes(),
            media_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="licence.json"'},
        )

    def _batch(batch: str) -> dict[str, Any]:
        if not lt._LICENCE_ID_RE.match(batch):
            raise HTTPException(404, "no such batch")
        view = next(
            (
                v
                for v in grouped(lt.list_issued(key_path), today())
                if v.get("key_type") == "machine" and v["licence_id"] == batch
            ),
            None,
        )
        if view is None:
            raise HTTPException(404, "no such batch")
        return view

    def _attachment(view: dict[str, Any], batch: str, ext: str) -> dict[str, str]:
        """File name: ASCII only in filename= (any customer name, e.g. Hindi or Polish, is safe),
        the full name in filename* for browsers that read it."""
        from urllib.parse import quote

        ascii_name = "".join(c if c.isascii() and c.isalnum() else "-" for c in view["customer"])
        ascii_name = "-".join(p for p in ascii_name.split("-") if p)[:40] or "customer"
        full = "".join(c if c.isalnum() else "-" for c in view["customer"])[:40].strip("-") or "customer"
        return {
            "Content-Disposition": (
                f'attachment; filename="licence-keys-{ascii_name}-{batch}.{ext}"; '
                f"filename*=UTF-8''{quote(f'licence-keys-{full}-{batch}.{ext}')}"
            )
        }

    @app.get("/api/batches/{batch}/csv")
    def batch_csv(batch: str) -> Response:
        """All the keys of a per-machine batch as a CSV file."""
        view = _batch(batch)
        return Response(
            lt.keys_csv(batch_rows(view)).encode("utf-8"),
            media_type="text/csv; charset=utf-8",
            headers=_attachment(view, batch, "csv"),
        )

    @app.get("/api/batches/{batch}/xlsx")
    def batch_xlsx(batch: str) -> Response:
        """The same keys as an Excel workbook (every cell text: machine IDs stay exactly as they are)."""
        view = _batch(batch)
        return Response(
            lt.keys_xlsx(batch_rows(view)),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers=_attachment(view, batch, "xlsx"),
        )

    @app.post("/api/check")
    def check(body: CheckIn) -> dict[str, Any]:
        """Open any licence key or licence file text: what is in it, and is it valid (for a device)?"""
        try:
            lic = lc.parse_licence(body.key, lc.vendor_public_key())
        except lc.LicenceError as e:
            return {"valid": False, "error": str(e)}
        raw = _raw_body(body.key)
        out: dict[str, Any] = {
            "valid": True,
            "error": None,
            "licence_id": lic.licence_id,
            "customer": lic.customer,
            "issued": lic.issued.isoformat(),
            "expires": lic.expires.isoformat() if lic.expires else None,
            "devices": sorted(lic.devices),
            "note": lic.note,
            "label": raw.get("device_label", ""),
            "key_type": "machine" if raw.get("batch") else "customer",
            "machine": None,
        }
        if body.machine_id.strip():
            import time

            try:
                lc.check(lic, body.machine_id, time.time())
                out["machine"] = {"ok": True, "message": "valid for this machine"}
            except (lc.LicenceError, ValueError) as e:
                out["machine"] = {"ok": False, "message": str(e)}
        return out

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/logo.png")
    def logo() -> FileResponse:
        return FileResponse(LOGO)

    @app.get("/static/{name}")
    def static(name: str) -> FileResponse:
        if name not in {"app.js", "style.css"}:
            raise HTTPException(404, "not found")
        return FileResponse(STATIC / name)

    return app


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    ap = argparse.ArgumentParser(description="Licence Manager (vendor only, this computer only)")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument(
        "--key", type=Path, default=None, help="signing key (default ~/anpr-licensing/vendor_key.json)"
    )
    ap.add_argument("--no-browser", action="store_true", help="do not open the browser")
    args = ap.parse_args(argv)
    key_path = (args.key or lt.default_key_path()).expanduser()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    token = secrets.token_urlsafe(24)
    url = f"http://127.0.0.1:{args.port}/#token={token}"
    print("\nLicence Manager — this computer only. Keep this window open while you use it.")
    print(f"Open: {url}\n(the link changes every time it starts; press Ctrl+C to stop)\n", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    uvicorn.run(
        create_app(key_path, token),
        host="127.0.0.1",
        port=args.port,
        access_log=False,
        server_header=False,
        log_level="warning",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
