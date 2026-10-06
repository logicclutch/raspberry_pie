"""Licence check: the engine only runs on devices listed in a licence file signed by the vendor.

The licence file (`licence.json`, next to config.yaml) looks like this:

    {
      "licence": {
        "format": 1,
        "licence_id": "LIC-20261001-7f3a",
        "customer": "Acme Logistics",
        "issued": "2026-10-01",
        "expires": "2027-09-30",            // last valid day; null = never expires
        "devices": ["a1b2c3d4", "9f8e7d6c"], // device IDs (Raspberry Pi serial numbers)
        "note": ""
      },
      "signature": "<base64 Ed25519 signature>"
    }

Only the vendor can make one: the signature is made with the vendor's private key (tools/licence_tool.py,
on the vendor's own computer) and checked here with the public key below. Changing any letter of the
file (another customer, a later date, one more serial) breaks the signature. One file may list many
devices, so the same file can be copied to every Pi of a customer.

Device ID: the Raspberry Pi's serial number (`python -m anpr.engine --machine-id`, or the "Serial" line
of /proc/cpuinfo). Leading zeros and upper/lower case do not matter. On a Mac it is the Mac's serial
number, on other Linux computers /etc/machine-id.

Setting the clock back does not extend a licence: the newest time the engine has seen (per licence) is
kept in the database, and the licence is checked against whichever is later. While running, that time
only moves on as fast as real time passes, so a clock that jumps far ahead (a mistyped year, a bad time
server) cannot lock a licensed device out; a new licence file from the vendor always starts afresh.
A renewed licence (pasted on the dashboard, or copied in) is picked up by a running engine within seconds.
"""

from __future__ import annotations

import base64
import binascii
import datetime as dt
import json
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from anpr import _ed25519

log = logging.getLogger("anpr.licence")

# Public half of the vendor's licence-signing key (LogicClutch Software LLP). The private half is kept
# by the vendor only and never ships with the software.
PUBLIC_KEY_HEX = "59ab4000e535d243c40249562da1d5fea12da7000da33d9ca4e5f340ac3ed3ae"

LICENCE_FILE = "licence.json"
FORMAT = 1
_DOMAIN = b"ANPR-LICENCE-v1\n"  # signatures made for anything else can never pass as a licence
_ID_RE = re.compile(r"^[0-9a-z][0-9a-z-]{0,63}$")
_CLOCK_KEY = "licence_clock"  # settings row: {licence_id: newest time seen} (clock set back = no extra days)
_MAX_RUN_JUMP_S = 86400.0  # within one run, the clock may run ahead of real elapsed time by at most this
_ISSUED_SLACK_S = 86400.0  # a day of slack before the issue date (time zones)
STATUS_KEY = "licence"  # settings row: last check result, shown on the dashboard
WARN_DAYS = 30  # log and dashboard warn this many days before the licence ends
FILE_CHECK_S = 5.0  # a running engine looks this often whether the licence file was replaced
_MAX_FILE_BYTES = 1 << 20


class LicenceError(Exception):
    """The licence is missing, damaged, not for this device, or expired. The message is for people."""


@dataclass(frozen=True)
class Licence:
    licence_id: str
    customer: str
    issued: dt.date
    expires: dt.date | None  # last valid day (local time); None = never expires
    devices: frozenset[str]
    note: str = ""

    def ends_at(self) -> float | None:
        """Unix time when the licence stops working: midnight after the last valid day."""
        if self.expires is None:
            return None
        try:
            return dt.datetime.combine(self.expires + dt.timedelta(days=1), dt.time.min).timestamp()
        except (OverflowError, ValueError, OSError):
            return math.inf  # e.g. 9999-12-31: further away than this computer can count

    def days_left(self, now: float) -> int | None:
        end = self.ends_at()
        if end is None:
            return None
        if math.isinf(end):
            return None
        return max(0, int((end - now) // 86400))


class SettingsStore(Protocol):
    def read_setting(self, key: str) -> object | None: ...
    def write_setting(self, key: str, value: object) -> None: ...


# ---- device ID ------------------------------------------------------------------------------------


def normalize_id(raw: str) -> str:
    """Comparable form of a device ID: lower case, no spaces, no leading zeros."""
    s = re.sub(r"[\s:\x00]", "", str(raw)).lower()
    if not s:
        raise ValueError("empty device ID")
    s = s.lstrip("0") or "0"
    if not _ID_RE.match(s):
        raise ValueError(f"not a valid device ID: {raw!r}")
    return s


def _read(path: str) -> str:
    try:
        return Path(path).read_bytes().decode("ascii", "ignore")
    except OSError:
        return ""


def machine_id() -> str:
    """This device's ID for the licence. Raises LicenceError when it cannot be read."""
    # Raspberry Pi: the SoC serial number (device tree first, /proc/cpuinfo on older kernels).
    raw = _read("/sys/firmware/devicetree/base/serial-number").strip("\x00 \t\r\n")
    if not raw:
        m = re.search(r"^Serial\s*:\s*([0-9A-Fa-f]+)\s*$", _read("/proc/cpuinfo"), re.M)
        raw = m.group(1) if m else ""
    if not raw and sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["/usr/sbin/ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            out = ""
        m = re.search(r'"IOPlatformSerialNumber"\s*=\s*"([^"]+)"', out)
        raw = m.group(1) if m else ""
    if not raw:
        # Other Linux computers (not a Pi): /etc/machine-id, marked so it can never equal a Pi serial.
        mid = _read("/etc/machine-id").strip()
        raw = f"mid-{mid[:32]}" if mid else ""
    try:
        return normalize_id(raw)
    except ValueError:
        raise LicenceError("cannot read this device's serial number") from None


# ---- licence file -----------------------------------------------------------------------------------


def signed_bytes(body: dict[str, Any]) -> bytes:
    """Exactly the bytes that are signed: the `licence` object as canonical JSON."""
    return _DOMAIN + json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _date(v: object, field: str) -> dt.date:
    if not isinstance(v, str):
        raise LicenceError(f"licence file is damaged ({field})")
    try:
        return dt.date.fromisoformat(v)
    except ValueError:
        raise LicenceError(f"licence file is damaged ({field})") from None


# ---- licence key (the same licence as one line of text, to copy and paste) -------------------------

KEY_PREFIX = "ANPR1-"
_MAX_KEY_JSON = 256 * 1024  # a decompressed key larger than this is not a licence (zip bomb guard)


def encode_key(document: dict[str, Any]) -> str:
    """{"licence": ..., "signature": ...} -> "ANPR1-<base64url(zlib(json))>"."""
    raw = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return KEY_PREFIX + base64.urlsafe_b64encode(zlib.compress(raw, 9)).decode().rstrip("=")


def key_to_json(key: str) -> str:
    """The licence document inside a licence key. Spaces and line breaks (from e-mail or chat) are
    ignored. Raises LicenceError."""
    k = "".join(key.split())
    if k[: len(KEY_PREFIX)].upper() != KEY_PREFIX.upper():
        raise LicenceError(f"this is not a licence key (it should start with {KEY_PREFIX})")
    body = k[len(KEY_PREFIX) :]
    try:
        packed = base64.urlsafe_b64decode(body + "=" * (-len(body) % 4))
        d = zlib.decompressobj()
        raw = d.decompress(packed, _MAX_KEY_JSON)
        if d.unconsumed_tail or not d.eof:
            raise ValueError("incomplete or too big")
        return raw.decode("ascii")
    except (binascii.Error, ValueError, zlib.error, UnicodeDecodeError):
        raise LicenceError(
            "the licence key is damaged or incomplete (copy the whole key, it is one long line)"
        ) from None


def parse_licence(text: str, public_key: bytes) -> Licence:
    """Check the signature, then read the fields. `text` is a licence file (JSON) or a licence key.
    Raises LicenceError."""
    if text.lstrip()[: len(KEY_PREFIX)].upper() == KEY_PREFIX.upper():
        text = key_to_json(text)
    try:
        doc = json.loads(text)
    except ValueError:
        raise LicenceError("licence file is not valid JSON (damaged or not a licence)") from None
    if not isinstance(doc, dict) or not isinstance(doc.get("licence"), dict):
        raise LicenceError("this is not a licence file")
    body, sig_txt = doc["licence"], doc.get("signature")
    if not isinstance(sig_txt, str):
        raise LicenceError("licence file has no signature")
    try:
        sig = base64.b64decode(sig_txt, validate=True)
    except (binascii.Error, ValueError):
        raise LicenceError("licence signature is damaged") from None
    if not _ed25519.verify(public_key, signed_bytes(body), sig):
        raise LicenceError(
            "licence signature is not valid (the file was changed, or not issued by the vendor)"
        )
    # Signed by the vendor: the fields can be trusted from here on, but still check their shape.
    if body.get("format") != FORMAT:
        raise LicenceError("licence format not supported by this software version; update the software")
    lid, customer, note = body.get("licence_id"), body.get("customer"), body.get("note", "")
    devices = body.get("devices")
    if not isinstance(lid, str) or not isinstance(customer, str) or not isinstance(note, str):
        raise LicenceError("licence file is damaged (names)")
    if not isinstance(devices, list) or not devices or not all(isinstance(d, str) for d in devices):
        raise LicenceError("licence file is damaged (devices)")
    try:
        ids = frozenset(normalize_id(d) for d in devices)
    except ValueError:
        raise LicenceError("licence file is damaged (devices)") from None
    expires = body.get("expires")
    return Licence(
        licence_id=lid,
        customer=customer,
        issued=_date(body.get("issued"), "issued"),
        expires=None if expires is None else _date(expires, "expires"),
        devices=ids,
        note=note,
    )


def check(lic: Licence, device_id: str, now: float) -> None:
    """Raises LicenceError unless `lic` lets `device_id` run at time `now`."""
    if normalize_id(device_id) not in lic.devices:
        raise LicenceError(
            f"licence {lic.licence_id} is not for this device (device ID {device_id}). "
            "Send this device ID to your supplier to get it added."
        )
    issued_at = dt.datetime.combine(lic.issued, dt.time.min).timestamp()
    if now < issued_at - _ISSUED_SLACK_S:
        raise LicenceError(
            f"the clock of this device is wrong ({time.strftime('%Y-%m-%d', time.localtime(now))}, "
            f"before the licence was issued on {lic.issued}). Set the date and time."
        )
    end = lic.ends_at()
    if end is not None and now >= end:
        raise LicenceError(
            f"licence {lic.licence_id} ended on {lic.expires}. Ask your supplier for a renewed licence."
        )


def read_licence(path: Path, public_key: bytes) -> Licence:
    try:
        if path.stat().st_size > _MAX_FILE_BYTES:
            raise LicenceError(f"{path} is too big to be a licence file")
        text = path.read_text(encoding="utf-8-sig")  # -sig: a file saved by a Windows editor (BOM) too
    except FileNotFoundError:
        raise LicenceError(f"no licence file at {path}") from None
    except (OSError, UnicodeDecodeError) as e:
        raise LicenceError(f"cannot read the licence file {path}: {e}") from None
    return parse_licence(text, public_key)


def vendor_public_key() -> bytes:
    return bytes.fromhex(PUBLIC_KEY_HEX)


def install_licence(
    text: str,
    path: Path,
    device_id: str,
    *,
    now: float | None = None,
    public_key: bytes | None = None,
) -> Licence:
    """Activation: check a pasted licence key (or licence file text) for THIS device and today, then
    save it as the licence file (atomically: a reader never sees half a file). Nothing is written when
    the key is not valid here. Raises LicenceError."""
    if len(text) > _MAX_FILE_BYTES:
        raise LicenceError("that is too long to be a licence key")
    lic = parse_licence(text, public_key or vendor_public_key())
    check(lic, device_id, time.time() if now is None else now)
    doc_text = key_to_json(text) if text.lstrip()[: len(KEY_PREFIX)].upper() == KEY_PREFIX.upper() else text
    pretty = json.dumps(json.loads(doc_text), indent=2) + "\n"
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix=".licence-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(pretty)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except OSError as e:
        Path(tmp).unlink(missing_ok=True)
        raise LicenceError(f"cannot save the licence file {path}: {e}") from None
    return lic


# ---- running engine ---------------------------------------------------------------------------------


def _finite(v: object) -> float | None:
    if isinstance(v, bool) or not isinstance(v, int | float):
        return None
    f = float(v)
    return f if math.isfinite(f) and f > 0 else None


def _stored_clock(store: SettingsStore | None, licence_id: str) -> float | None:
    if store is None:
        return None
    try:
        seen = store.read_setting(_CLOCK_KEY)
    except Exception:  # noqa: BLE001 - a database hiccup must not stop a licensed engine
        log.debug("licence clock not available", exc_info=True)
        return None
    return _finite(seen.get(licence_id)) if isinstance(seen, dict) else None


def _save_clock(store: SettingsStore | None, licence_id: str, t: float) -> None:
    if store is None:
        return
    try:
        seen = store.read_setting(_CLOCK_KEY)
        marks = {k: v for k, v in seen.items() if _finite(v)} if isinstance(seen, dict) else {}
        old = marks.get(licence_id)
        if old is None or t > old + 60:
            marks[licence_id] = t
            store.write_setting(_CLOCK_KEY, marks)
    except Exception:  # noqa: BLE001
        log.debug("cannot save licence clock", exc_info=True)


def summary(lic: Licence | None, device_id: str | None, now: float, error: str | None) -> dict[str, Any]:
    """What the dashboard shows about the licence."""
    return {
        "ok": error is None,
        "error": error,
        "licence_id": lic.licence_id if lic else None,
        "customer": lic.customer if lic else None,
        "expires": lic.expires.isoformat() if lic and lic.expires else None,
        "days_left": lic.days_left(now) if lic else None,
        "device_id": device_id,
        "checked_at": now,
    }


class ReadOnlySettings:
    """The engine's settings table, read only (never writes, never creates files)."""

    def __init__(self, db_path: Path | None) -> None:
        self._db = db_path

    def read_setting(self, key: str) -> object | None:
        if self._db is None or not Path(self._db).exists():
            return None
        import sqlite3

        try:
            con = sqlite3.connect(f"file:{Path(self._db).resolve()}?mode=ro", uri=True, timeout=5)
            try:
                row = con.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
            finally:
                con.close()
            return json.loads(row[0]) if row else None
        except (sqlite3.Error, ValueError, OSError):
            return None

    def write_setting(self, key: str, value: object) -> None:
        return None


def _record(store: SettingsStore | None, body: dict[str, Any]) -> None:
    if store is None:
        return
    try:
        store.write_setting(STATUS_KEY, body)
    except Exception:  # noqa: BLE001
        log.debug("cannot save licence status", exc_info=True)


class LicenceGuard:
    """A checked licence for this device. `expired()` is cheap enough to call every frame: it checks
    the date again (and looks for a renewed licence file) at most every `recheck_s` seconds."""

    def __init__(
        self,
        lic: Licence,
        device_id: str,
        store: SettingsStore | None = None,
        *,
        path: Path | None = None,
        public_key: bytes | None = None,
        recheck_s: float = 600.0,
        clock: Callable[[], float] = time.time,
        mono: Callable[[], float] = time.monotonic,
    ) -> None:
        self.licence = lic
        self.device_id = device_id
        self._store = store
        self._path = path
        self._public_key = public_key
        self._file_sig = self._file_signature()
        self._recheck_s = recheck_s
        self._clock = clock
        self._mono = mono
        self._next = -float("inf")
        self._next_file = -float("inf")
        self._ref: tuple[float, float] | None = None  # (trusted time, monotonic) at the last check
        self.error: str | None = None
        self._evaluate()
        if self.error is not None:
            raise LicenceError(self.error)

    @classmethod
    def open(
        cls,
        path: Path,
        store: SettingsStore | None = None,
        *,
        public_key: bytes | None = None,
        device_id: str | None = None,
        **kw: Any,
    ) -> LicenceGuard:
        """Read, verify and check the licence file for this device. Raises LicenceError."""
        dev = device_id or machine_id()
        key = public_key or vendor_public_key()
        try:
            lic = read_licence(path, key)
        except LicenceError as e:
            _record(store, summary(None, dev, time.time(), str(e)))
            raise
        return cls(lic, dev, store, path=path, public_key=key, **kw)

    def _file_signature(self) -> tuple[int, int] | None:
        if self._path is None:
            return None
        try:
            st = self._path.stat()
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size)

    def _reload_if_changed(self) -> None:
        """A renewed licence copied in while running: use it if it is valid for this device."""
        sig = self._file_signature()
        if self._path is None or sig is None or sig == self._file_sig:
            return
        self._file_sig = sig
        try:
            new = read_licence(self._path, self._public_key or vendor_public_key())
            if normalize_id(self.device_id) not in new.devices:
                raise LicenceError("the new licence file is not for this device")
        except LicenceError as e:
            log.warning(
                "licence file changed but cannot be used (%s); keeping %s", e, self.licence.licence_id
            )
            return
        if new != self.licence:
            log.info("licence file changed: now %s for %s", new.licence_id, new.customer)
            self.licence = new
            self._ref = None  # a new licence starts its own clock

    def expired(self) -> bool:
        if self.error is not None:
            return True  # ended: the engine stops; the service restarts it and waits for a valid licence
        now = self._mono()
        if now < self._next:
            # A renewed key pasted on the dashboard: noticed within seconds, not at the next full check.
            if self._path is None or now < self._next_file:
                return False
            self._next_file = now + FILE_CHECK_S
            if self._file_signature() == self._file_sig:
                return False
        self._evaluate()
        if self.error is not None:
            log.error("licence: %s", self.error)
        return self.error is not None

    def _trusted_now(self) -> float:
        """The time the licence is checked against. At the first check: the clock, or the newest time
        seen before with this licence if that is later (clock set back). After that: the clock, but
        never more than a day ahead of the real time that has passed since the last check (a clock
        jump is ignored for this run) and never behind the last check (clock set back)."""
        wall, mono = self._clock(), self._mono()
        lid = self.licence.licence_id
        if self._ref is None:
            stored = _stored_clock(self._store, lid)
            now = max(wall, stored) if stored is not None else wall
        else:
            last, last_mono = self._ref
            elapsed = max(0.0, mono - last_mono)
            ceiling = last + elapsed + _MAX_RUN_JUMP_S
            if wall > ceiling:
                log.warning(
                    "the clock jumped ahead to %s; using the time passed since the last check instead",
                    time.strftime("%Y-%m-%d %H:%M", time.localtime(wall)),
                )
                now = last + elapsed
            else:
                now = max(wall, last + elapsed) if wall >= last else last + elapsed
        self._ref = (now, mono)
        _save_clock(self._store, lid, now)
        return now

    def _evaluate(self) -> None:
        self._next = self._mono() + self._recheck_s
        self._reload_if_changed()
        now = self._trusted_now()
        try:
            check(self.licence, self.device_id, now)
        except LicenceError as e:
            self.error = str(e)
        else:
            left = self.licence.days_left(now)
            if left is not None and left <= WARN_DAYS:
                log.warning(
                    "licence %s ends in %d day(s), on %s", self.licence.licence_id, left, self.licence.expires
                )
        _record(self._store, summary(self.licence, self.device_id, now, self.error))


def wait_for_licence(
    path: Path,
    store: SettingsStore | None,
    stop: Callable[[], bool],
    *,
    wait: bool = True,
    retry_s: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
    **kw: Any,
) -> LicenceGuard | None:
    """A valid licence for this device, or None when `stop()` (or `wait` is off and there is none).
    With `wait`, a missing or wrong licence is checked again every `retry_s` seconds, and at once when the
    licence file changes (activation on the dashboard), so the engine starts within seconds."""

    def file_sig() -> tuple[int, int] | None:
        try:
            st = Path(path).stat()
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size)

    last_error = None
    while True:
        try:
            guard = LicenceGuard.open(path, store, **kw)
        except LicenceError as e:
            if not wait:
                log.error("licence: %s", e)
                return None
            if str(e) != last_error:  # the same complaint once, not every minute
                log.error("licence: %s", e)
                log.info(
                    "the engine starts once a valid licence is in %s (activate it on the dashboard)", path
                )
                last_error = str(e)
            sig = file_sig()
            waited = 0.0
            while waited < retry_s and not stop() and file_sig() == sig:
                sleep(1.0)
                waited += 1.0
            if stop():
                return None
            continue
        lic = guard.licence
        log.info(
            "licence %s for %s, device %s, %s",
            lic.licence_id,
            lic.customer,
            guard.device_id,
            f"valid until {lic.expires}" if lic.expires else "no end date",
        )
        return guard
