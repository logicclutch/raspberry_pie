"""Send confirmed plates to the client's own server: JSON over HTTP POST (docs/CLIENT_API_FORMAT.md).

The client's API address and key are set on the dashboard (Settings) and kept in the database, so they
can be changed at any time without touching config files or restarting anything. A background thread
in the engine process (`Pusher`) sends every new plate, in order, and remembers how far it got: when
the server or the internet is down it keeps the plates and retries with growing pauses, and after a
restart it continues where it stopped.

Plates confirmed before the sending was switched on (or before the address was changed) are not sent:
the new server gets plates from that moment on.
"""

from __future__ import annotations

import base64
import contextlib
import dataclasses
import json
import logging
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from collections.abc import Callable
from datetime import datetime
from typing import Any
from urllib.parse import urlsplit

from anpr import __version__
from anpr.storage import SqliteEventStore
from anpr.types import PlateEvent
from anpr.validator import format_display

log = logging.getLogger(__name__)

SETTINGS_KEY = "push"  # written by the dashboard
STATE_KEY = "push_state"  # written by the sender (separate key: the two never overwrite each other)
SCHEMA_VERSION = "1.0"
BATCH = 25  # plates per request (with photos about 100 KB each)
TIMEOUT_S = 10.0
POLL_S = 2.0  # how often to look for new plates / changed settings
MIN_BACKOFF_S = 5.0
MAX_BACKOFF_S = 300.0
MAX_URL = 2048
MAX_HEADER_VALUE = 4096
REPLY_SNIPPET = 300  # characters of the server's reply kept for the dashboard
# Inline push (ingest respond_with_plate): the camera is waiting for the HTTP reply, so the attempt made
# for it is short. Anything not delivered in that time stays queued for the background sender.
INLINE_TIMEOUT_S = 4.0
INLINE_LOCK_WAIT_S = 1.0  # how long to wait if the background sender is mid-request
_OUTCOMES_KEPT = 2000  # per-plate send outcomes remembered for push_now()

_HEADER_NAME = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]{1,64}")
_DEVICE_ID = re.compile(r"[A-Za-z0-9._-]{1,40}")
_RETRY_4XX = {401, 403, 404, 405, 408, 409, 425, 429}  # fixable on the server side: keep the plates


class PushError(ValueError):
    """A setting the dashboard sent is not usable (message is shown to the user)."""


# ---- settings -----------------------------------------------------------------------------------


def default_device_id() -> str:
    name = re.sub(r"[^A-Za-z0-9._-]", "-", socket.gethostname().split(".")[0])[:40].strip("-.")
    return name or "anpr-pi"


@dataclasses.dataclass(frozen=True, slots=True)
class PushSettings:
    enabled: bool = False
    url: str = ""
    header_name: str = "Authorization"  # "" = send no key header
    header_value: str = ""  # e.g. "Bearer abc123" (secret: never sent back to the browser)
    device_id: str = ""
    include_images: bool = True
    # Bumped by the dashboard when sending is switched on or the address changes: the sender then
    # starts after event `start_after_id` (the newest plate at that moment).
    start_rev: int = 0
    start_after_id: int = 0

    @classmethod
    def from_dict(cls, d: object) -> PushSettings:
        """Lenient: unknown keys and values of the wrong type are ignored (defaults used)."""
        if not isinstance(d, dict):
            return cls()
        defaults = cls()
        kw: dict[str, Any] = {}
        for f in dataclasses.fields(cls):
            v = d.get(f.name)
            want = type(getattr(defaults, f.name))
            if type(v) is want:  # exact: a bool is not taken for an int
                kw[f.name] = v
        return cls(**kw)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @property
    def device(self) -> str:
        return self.device_id or default_device_id()

    def fingerprint(self) -> tuple:
        return dataclasses.astuple(self)


def load_settings(store: SqliteEventStore) -> PushSettings:
    return PushSettings.from_dict(store.read_setting(SETTINGS_KEY))


def save_settings(store: SqliteEventStore, s: PushSettings) -> None:
    store.write_setting(SETTINGS_KEY, s.to_dict())


def apply_push_config(store: SqliteEventStore, pc: Any) -> None:
    """Make config.yaml's `push:` section the source of truth ("file wins on restart").

    The user-facing fields (on/off, address, key header, device id, images) are taken from `pc` and
    written to the device's settings; the internal send cursor is kept. As with the dashboard, turning
    sending on or changing the address restarts the stream from the newest plate at this moment, so old
    plates are not re-sent. Called once at engine start when a `push:` section is present. `pc` is an
    anpr.config.PushConfig; typed as Any to avoid importing the web/config layer into the sender.
    """
    saved = load_settings(store)
    desired = dataclasses.replace(
        saved,
        enabled=bool(pc.enabled),
        url=clean_url(pc.url),
        header_name=pc.header_name,
        header_value=pc.header_value,
        device_id=pc.device_id,
        include_images=bool(pc.include_images),
    )
    if desired.enabled and (not saved.enabled or desired.url != saved.url):
        desired = dataclasses.replace(
            desired, start_rev=saved.start_rev + 1, start_after_id=store.latest_id()
        )
    if desired != saved:
        save_settings(store, desired)
        log.info(
            "push settings applied from config.yaml: %s %s",
            "on" if desired.enabled else "off",
            desired.url or "(no address)",
        )


def clean_url(url: str) -> str:
    u = url.strip()
    if not u:
        return ""
    if len(u) > MAX_URL:
        raise PushError("the API address is too long")
    if any(c.isspace() for c in u):
        raise PushError("the API address must not contain spaces")
    parts = urlsplit(u)
    if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
        raise PushError("the API address must start with https:// (or http://) and include the server name")
    return u


def clean_header_name(name: str) -> str:
    n = name.strip()
    if n and not _HEADER_NAME.fullmatch(n):
        raise PushError(
            "the header name may only use letters, digits and - (e.g. Authorization or X-API-Key)"
        )
    return n


def clean_header_value(value: str) -> str:
    v = value.strip()
    if len(v) > MAX_HEADER_VALUE:
        raise PushError("the key is too long")
    if any(c in v for c in "\r\n\x00"):
        raise PushError("the key must be on one line")
    return v


def clean_device_id(device_id: str) -> str:
    d = device_id.strip()
    if d and not _DEVICE_ID.fullmatch(d):
        raise PushError("the device ID may only use letters, digits, '.', '_' and '-' (up to 40)")
    return d


def secret_hint(value: str) -> str | None:
    """What the dashboard may show of a saved key: its last 4 characters, only for long keys."""
    if not value:
        return None
    return "…" + value[-4:] if len(value) >= 12 else "…"


# ---- JSON body (docs/CLIENT_API_FORMAT.md) --------------------------------------------------------


def iso_local(ts: float) -> str:
    return datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")


def _image_b64(store: SqliteEventStore, rel: str | None) -> str | None:
    if not rel:
        return None
    path = store.resolve_image(rel)
    if path is None:
        return None
    try:
        return base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return None


def event_payload(
    e: PlateEvent, device: str, store: SqliteEventStore | None, include_images: bool
) -> dict[str, Any]:
    images = {"plate_crop_jpeg_base64": None, "full_snapshot_jpeg_base64": None}
    if include_images and store is not None:
        images = {
            "plate_crop_jpeg_base64": _image_b64(store, e.crop_path),
            "full_snapshot_jpeg_base64": _image_b64(store, e.snapshot_path),
        }
    return {
        "event_id": f"{device}-{int(e.id or 0):06d}",
        "plate": e.plate,
        "plate_display": format_display(e.plate, e.kind),
        "plate_type": e.kind,
        "hsrp": e.hsrp,
        "camera": e.camera or "",
        "confidence": round(float(e.confidence), 4),
        "votes": int(e.votes),
        "first_seen": iso_local(e.first_seen),
        "detected_at": iso_local(e.last_seen),
        "images": images,
    }


def build_body(
    events: list[PlateEvent],
    s: PushSettings,
    station: str,
    store: SqliteEventStore | None,
    *,
    test: bool = False,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "device": {"device_id": s.device, "station_name": station, "software_version": __version__},
        "events": [event_payload(e, s.device, store, s.include_images) for e in events],
    }
    if test:
        body["test"] = True
    return body


def test_body(s: PushSettings, station: str, now: float | None = None) -> dict[str, Any]:
    """A clearly marked sample plate ("test": true) for the dashboard's "Send test" button."""
    now = time.time() if now is None else now
    sample = PlateEvent(
        plate="MH12AB1234",
        kind="standard",
        confidence=0.99,
        votes=5,
        track_id=0,
        first_seen=now - 2,
        last_seen=now,
        hsrp="hsrp",
    )
    body = build_body([sample], s, station, None, test=True)
    body["events"][0]["event_id"] = f"{s.device}-TEST-{int(now)}"
    return body


# ---- HTTP ---------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class SendResult:
    ok: bool
    status: int | None  # HTTP status, None = no answer
    retry: bool  # False = the server refused these plates for good (bad request)
    message: str
    reply: str = ""  # start of the server's answer (for the dashboard)
    rejected: tuple[tuple[str, str], ...] = ()  # (event_id, reason) the server listed as rejected


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        return None  # a redirect would turn the POST into a GET: report it instead


_OPENER = urllib.request.build_opener(_NoRedirect)


def _snippet(raw: bytes) -> str:
    text = raw.decode("utf-8", "replace").strip()
    return text[:REPLY_SNIPPET] + ("…" if len(text) > REPLY_SNIPPET else "")


def _rejected(raw: bytes) -> tuple[tuple[str, str], ...]:
    try:
        d = json.loads(raw)
    except ValueError:
        return ()
    items = d.get("rejected") if isinstance(d, dict) else None
    if not isinstance(items, list):
        return ()
    out = []
    for r in items:
        if isinstance(r, dict):
            out.append((str(r.get("event_id", "")), str(r.get("reason", ""))))
        elif isinstance(r, str):
            out.append((r, ""))
    return tuple(out)


_OK_WORDS = {"success", "successful", "ok", "saved", "inserted", "true"}


def _error_in_reply(raw: bytes) -> str | None:
    """Some APIs answer HTTP 200 even when nothing was saved, and say so in the body. Recognised:
    {"errorCode": "1"|"0", "errorMessage": "success"} style (IntelliParks: 1 = success), {"success": false},
    {"status": "error"|"fail"|"failed"}. Returns a short reason, or None when the reply is OK or unknown."""
    try:
        d = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(d, dict):
        return None
    low = {str(k).lower(): v for k, v in d.items()}
    if "errorcode" in low:
        code = str(low["errorcode"]).strip()
        msg = str(low.get("errormessage", "")).strip()
        if msg.lower() in _OK_WORDS or (not msg and code in ("0", "1")):
            return None
        return f"{msg or 'error'} (errorCode {code})"
    if low.get("success") is False:
        return str(low.get("message") or low.get("error") or "success: false")
    status = low.get("status")
    if isinstance(status, str) and status.strip().lower() in ("error", "fail", "failed", "failure"):
        return str(low.get("message") or low.get("error") or f"status: {status}")
    return None


def post_json(
    url: str, header_name: str, header_value: str, body: dict, timeout: float = TIMEOUT_S
) -> SendResult:
    data = json.dumps(body, separators=(",", ":")).encode()
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": f"anpr-pi/{__version__}",
    }
    if header_name and header_value:
        headers[header_name] = header_value
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with _OPENER.open(req, timeout=timeout) as r:
            status, raw = r.status, r.read(64 * 1024)
    except urllib.error.HTTPError as e:
        status = e.code
        raw = b""
        with contextlib.suppress(Exception):
            raw = e.read(64 * 1024) or b""
        if 300 <= status < 400:
            where = e.headers.get("Location", "") if e.headers else ""
            to = f": {where}" if where else ""
            msg = f"the server redirected to another address{to} (HTTP {status}) - use that address"
            return SendResult(False, status, True, msg, _snippet(raw))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        reason = getattr(e, "reason", e)
        return SendResult(False, None, True, f"cannot reach the server: {reason}")
    reply = _snippet(raw)
    if 200 <= status < 300:
        refused = _error_in_reply(raw)
        if refused:
            # HTTP 200 but the answer says it was not saved (e.g. {"errorCode":"2","errorMessage":"error"}):
            # keep the plates and try again, never count them as sent.
            return SendResult(False, status, True, f"the server answered an error: {refused}", reply)
        return SendResult(True, status, False, f"accepted (HTTP {status})", reply, _rejected(raw))
    if status in (401, 403):
        return SendResult(False, status, True, f"the server refused the key (HTTP {status})", reply)
    if status in (404, 405):
        return SendResult(False, status, True, f"wrong API address (HTTP {status})", reply)
    if status in _RETRY_4XX or status >= 500:
        return SendResult(False, status, True, f"server error (HTTP {status})", reply)
    return SendResult(False, status, False, f"the server refused the data (HTTP {status})", reply)


# ---- background sender --------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class PushState:
    cursor: int = 0  # id of the last plate handled (sent, or refused for good)
    start_rev: int = 0
    sent_total: int = 0
    skipped_total: int = 0  # plates the server refused for good
    last_ok_ts: float | None = None
    last_attempt_ts: float | None = None
    last_error: str | None = None
    last_error_ts: float | None = None
    # The API's answer to the last request with real plates (HTTP status and the start of the body).
    last_reply: str | None = None
    last_reply_status: int | None = None
    last_reply_ts: float | None = None

    @classmethod
    def from_dict(cls, d: object) -> PushState:
        st = cls()
        if not isinstance(d, dict):
            return st
        num = lambda v: isinstance(v, int | float) and not isinstance(v, bool)  # noqa: E731
        for name in ("cursor", "start_rev", "sent_total", "skipped_total"):
            if num(d.get(name)):
                setattr(st, name, int(d[name]))
        for name in ("last_ok_ts", "last_attempt_ts", "last_error_ts", "last_reply_ts"):
            if num(d.get(name)):
                setattr(st, name, float(d[name]))
        if num(d.get("last_reply_status")):
            st.last_reply_status = int(d["last_reply_status"])
        for name in ("last_error", "last_reply"):
            if isinstance(d.get(name), str):
                setattr(st, name, d[name])
        return st


def load_state(store: SqliteEventStore) -> PushState:
    return PushState.from_dict(store.read_setting(STATE_KEY))


class Pusher:
    """Sends new plates, oldest first, `BATCH` per request. One instance per database."""

    def __init__(
        self,
        store: SqliteEventStore,
        station: str,
        *,
        post: Callable[..., SendResult] = post_json,
        clock: Callable[[], float] = time.time,
        poll_s: float = POLL_S,
        batch: int = BATCH,
    ) -> None:
        self._store = store
        self._station = station
        self._post = post
        self._clock = clock
        self._poll_s = poll_s
        self._batch = max(1, batch)
        self._fails = 0
        self._single_until = 0  # after a refused batch: send one by one up to this id
        # When the next background round is due (also pushed back by a failed inline attempt, so the
        # API is not hammered while it is down).
        self._next_at = 0.0
        # Held around every send, so the background sender and an inline push_now() never send the
        # same plates twice (both read and advance the one cursor in the database).
        self.lock = threading.Lock()
        # The last server result from step() (None when nothing was sent).
        self.last_result: SendResult | None = None
        # What happened to each recently handled plate id: (accepted, server result, message). Lets
        # push_now() report THIS plate's real outcome, whichever thread sent it. Bounded.
        self._outcomes: OrderedDict[int, tuple[bool, SendResult, str]] = OrderedDict()

    def _record(self, eid: int, accepted: bool, res: SendResult, message: str) -> None:
        self._outcomes[eid] = (accepted, res, message)
        self._outcomes.move_to_end(eid)
        while len(self._outcomes) > _OUTCOMES_KEPT:
            self._outcomes.popitem(last=False)

    def _outcome_of(self, ids: list[int]) -> PushOutcome | None:
        """The recorded outcome for these plates, or None if any of them has none yet."""
        recs = [self._outcomes.get(i) for i in ids]
        if any(r is None for r in recs):
            return None
        refused = [r for r in recs if not r[0]]
        if refused:
            _, res, msg = refused[0]
            return PushOutcome(False, False, msg, res)
        _, res, msg = recs[-1]
        return PushOutcome(True, False, msg, res)

    def _save(self, st: PushState) -> None:
        self._store.write_setting(STATE_KEY, dataclasses.asdict(st))

    def _backoff(self) -> float:
        return min(MAX_BACKOFF_S, MIN_BACKOFF_S * 2 ** max(0, self._fails - 1))

    def reset_backoff(self) -> None:
        self._fails = 0

    def step(self, s: PushSettings | None = None, *, timeout: float | None = None) -> float:
        """One round. Returns how long to wait before the next one (0 = more plates are waiting).
        Call it holding `self.lock` when another thread may also send."""
        s = load_settings(self._store) if s is None else s
        self.last_result = None
        if not s.enabled or not s.url:
            return self._poll_s
        st = load_state(self._store)
        if s.start_rev < st.start_rev:
            # A snapshot taken before a dashboard save that another thread has already applied: never
            # rewind the cursor to an old stream (it would re-send plates, to the old address).
            s = load_settings(self._store)
            if not s.enabled or not s.url:
                return self._poll_s
        if st.start_rev != s.start_rev:  # switched on / new address: start after the newest plate then
            st.cursor, st.start_rev = s.start_after_id, s.start_rev
            st.last_error = st.last_error_ts = None
            self._single_until = 0
            self._fails = 0
            self._save(st)
        size = 1 if st.cursor < self._single_until else self._batch
        events = self._store.events_after(st.cursor, size)
        if not events:
            return self._poll_s
        now = self._clock()
        kw = {} if timeout is None else {"timeout": timeout}
        plates = ", ".join(f"{e.plate} (#{e.id})" for e in events)
        # Request/response log (one line each), so `journalctl -u anpr-ingest | grep "API "` shows every
        # call: where it went, which plates, and exactly what the API answered. Images are not logged.
        log.info("API request: POST %s - %d plate(s): %s", s.url, len(events), plates)
        res = self._post(
            s.url, s.header_name, s.header_value, build_body(events, s, self._station, self._store), **kw
        )
        log.info(
            "API response: %s - HTTP %s - %s - reply: %s",
            s.url, res.status if res.status is not None else "none (no connection)", res.message,
            res.reply or "(empty)",
        )
        self.last_result = res
        st.last_attempt_ts = now
        st.last_reply, st.last_reply_status, st.last_reply_ts = res.reply, res.status, now
        if res.ok:
            st.cursor = int(events[-1].id or st.cursor)
            st.sent_total += len(events)
            st.last_ok_ts = now
            st.last_error = st.last_error_ts = None
            rejected = dict(res.rejected)
            for e in events:
                key = f"{s.device}-{int(e.id or 0):06d}"
                if key in rejected:
                    why = rejected[key] or "no reason given"
                    self._record(int(e.id or 0), False, res, f"refused by the client API: {why}")
                else:
                    self._record(int(e.id or 0), True, res, res.message)
            for eid, why in res.rejected:
                log.warning("client server rejected %s: %s", eid, why or "no reason given")
            self._fails = 0
            self._save(st)
            log.info("sent %d plate(s) to the client server (up to #%d)", len(events), st.cursor)
            return 0.0 if len(events) == size else self._poll_s
        if res.retry:
            self._fails += 1
            st.last_error, st.last_error_ts = res.message, now
            self._save(st)
            wait = self._backoff()
            log.warning("sending plates failed: %s; retrying in %.0f s", res.message, wait)
            return wait
        # Refused for good (bad request). Find the plate the server dislikes: one by one, then skip it.
        if len(events) > 1:
            self._single_until = int(events[-1].id or 0)
            return 0.0
        st.cursor = int(events[0].id or st.cursor)
        self._record(int(events[0].id or 0), False, res, f"refused by the client API for good: {res.message}")
        st.skipped_total += 1
        st.last_error = f"plate {events[0].plate} (#{events[0].id}) refused and skipped: {res.message}"
        st.last_error_ts = now
        self._save(st)
        log.warning("client server refused plate #%s for good: %s %s", events[0].id, res.message, res.reply)
        return 0.0

    def push_now(
        self,
        ids: list[int],
        s: PushSettings | None = None,  # noqa: ARG002 - kept for callers; re-read under the lock
        *,
        timeout: float = INLINE_TIMEOUT_S,
        lock_wait: float = INLINE_LOCK_WAIT_S,
    ) -> PushOutcome:
        """Try to deliver the just-saved plates `ids` right now, for a caller that reports the outcome
        (the ingest respond_with_plate reply). Bounded: waits at most `lock_wait` for the background
        sender and `timeout` for the API. It never sends out of order and never sends twice: it only
        sends when these plates fit in the next batch from the cursor; otherwise (a backlog, the API in
        backoff, the sender busy) it leaves them queued, and the background sender delivers them.
        `delivered` is True only when the API accepted THESE plates (not listed as rejected)."""
        ids = [int(i) for i in ids if i]
        if not ids:
            return PushOutcome(False, False, "nothing new to send")
        if not self.lock.acquire(timeout=lock_wait):
            return PushOutcome(False, True, "queued: the sender is busy with earlier plates; sent shortly")
        try:
            # Settings are read under the lock, so a dashboard save is never applied out of order.
            s = load_settings(self._store)
            if not s.enabled or not s.url:
                return PushOutcome(False, False, "sending to the client API is off")
            st = load_state(self._store)
            new_stream = st.start_rev != s.start_rev
            # Mirrors step(): a newly saved address/switch-on starts after the newest plate at that time.
            cursor = s.start_after_id if new_stream else st.cursor
            first, top = min(ids), max(ids)
            if top <= cursor:
                done = self._outcome_of(ids)  # the background sender handled it first
                if done is not None:
                    return done
                if new_stream or top <= s.start_after_id:
                    return PushOutcome(False, False, "not sent: saved before sending was switched on")
                return PushOutcome(False, False, "already handled by the background sender")
            now = self._clock()
            if self._fails and now < self._next_at:
                why = st.last_error or "the last attempt failed"
                return PushOutcome(False, True, f"queued: {why}; retrying in {self._next_at - now:.0f} s")
            single = cursor < self._single_until
            pending = self._store.events_after(cursor, 1 if single else self._batch)
            pending_ids = {int(e.id or 0) for e in pending}
            if not pending_ids or first < min(pending_ids) or top not in pending_ids:
                return PushOutcome(False, True, "queued behind earlier unsent plates; sent in order shortly")
            self._next_at = now + self.step(s, timeout=timeout)
            done = self._outcome_of(ids)
            if done is not None:
                return done
            res = self.last_result
            if res is None:
                return PushOutcome(False, True, "queued: not sent yet")
            # Not accepted yet: a retryable error, or a refused batch now being re-sent one by one.
            return PushOutcome(False, True, res.message, res)
        finally:
            self.lock.release()

    def run(self, stop: threading.Event) -> None:
        last_fp: tuple | None = None
        while not stop.is_set():
            try:
                s = load_settings(self._store)
                if s.fingerprint() != last_fp:  # saved on the dashboard: try right away
                    last_fp = s.fingerprint()
                    self._next_at = 0.0
                    self.reset_backoff()
                if self._clock() >= self._next_at:
                    with self.lock:
                        # An inline push_now() may have sent (or backed off) while we waited; and the
                        # settings are re-read under the lock so an older snapshot is never applied.
                        if self._clock() >= self._next_at:
                            self._next_at = self._clock() + self.step(load_settings(self._store))
            except Exception:  # noqa: BLE001 - the sender must never take the engine down
                log.exception("plate sender failed")
                self._next_at = self._clock() + 30.0
            stop.wait(max(0.0, min(self._poll_s, self._next_at - self._clock())))


@dataclasses.dataclass(frozen=True, slots=True)
class PushOutcome:
    """What happened to specific plates in Pusher.push_now()."""

    delivered: bool  # the client API accepted THESE plates
    queued: bool  # not delivered yet, kept in the database; the background sender will retry
    message: str
    result: SendResult | None = None  # the API's answer when a request was made


def start_pusher(
    store: SqliteEventStore, station: str, pusher: Pusher | None = None
) -> tuple[threading.Thread, threading.Event]:
    """Run the background sender. Pass `pusher` to share it with an inline sender (ingest)."""
    stop = threading.Event()
    p = pusher if pusher is not None else Pusher(store, station)
    t = threading.Thread(target=p.run, args=(stop,), name="plate-sender", daemon=True)
    t.start()
    return t, stop
