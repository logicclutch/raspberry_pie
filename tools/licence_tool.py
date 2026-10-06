"""Vendor-only licence tool: make the signing key, issue licence files, check them.

NEVER ship this file or the key to a client. The client gets only the licence.json files.

    # once, ever (keep the key file safe and backed up: without it no new licences can be made)
    python tools/licence_tool.py keygen

    # a licence for a customer's devices (IDs from `python -m anpr.engine --machine-id` on each Pi)
    python tools/licence_tool.py issue --customer "Acme Logistics" --devices-file acme_serials.txt \
        --days 365 --out licence.json

    # what is inside a licence, and is it valid for one device?
    python tools/licence_tool.py show licence.json --device 00000000a1b2c3d4

The key lives in ~/anpr-licensing/vendor_key.json (or --key / $ANPR_LICENCE_KEY). Every issued licence is
also copied to ~/anpr-licensing/issued/ as a record of what was sold to whom.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import os
import re
import secrets
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anpr import _ed25519  # noqa: E402
from anpr import licence as lc  # noqa: E402

KEY_KIND = "anpr-licence-signing-key"
_LICENCE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
MAX_EXPIRY_YEAR = 2200
_VERIFIED: dict[tuple[str, str, int, int], dict] = {}  # (public key, path, mtime, size) -> record


class IssueError(ValueError):
    """A licence cannot be made as asked (message is for the person using the tool)."""


def default_key_path() -> Path:
    return Path(os.environ.get("ANPR_LICENCE_KEY") or Path.home() / "anpr-licensing" / "vendor_key.json")


def load_key(path: Path) -> tuple[bytes, bytes]:
    try:
        doc = json.loads(path.read_text())
        seed = bytes.fromhex(doc["private_key"])
    except FileNotFoundError:
        raise IssueError(f"no signing key at {path} (run: licence_tool.py keygen)") from None
    except (ValueError, KeyError, TypeError, OSError):
        raise IssueError(f"{path} is not a licence signing key") from None
    if doc.get("kind") != KEY_KIND or len(seed) != 32:
        raise IssueError(f"{path} is not a licence signing key")
    return seed, _ed25519.public_key(seed)


def cmd_keygen(args: argparse.Namespace) -> int:
    path: Path = args.key
    if path.exists():
        print(f"refusing to overwrite the existing key {path}", file=sys.stderr)
        print("(a new key would make every licence already issued stop working)", file=sys.stderr)
        return 1
    seed = secrets.token_bytes(32)
    pub = _ed25519.public_key(seed)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    body = {
        "kind": KEY_KIND,
        "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "private_key": seed.hex(),
        "public_key": pub.hex(),
        "warning": "SECRET. Anyone with this file can make licences. Keep a backup offline.",
    }
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(body, f, indent=2)
        f.write("\n")
    print(f"signing key saved: {path}  (keep it secret, back it up)")
    print(f'public key — put it in anpr/licence.py:\nPUBLIC_KEY_HEX = "{pub.hex()}"')
    return 0


def parse_device_entries(text: str) -> list[tuple[str, str]]:
    """(device ID, label) from free text: IDs separated by commas, spaces or new lines; '#' starts a
    comment, which becomes the label of the ID on that line ("00000000a1b2c3d4  # Gate 1"); duplicates
    (also with/without leading zeros or other case) counted once. Raises IssueError."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        code, _, comment = line.partition("#")
        label = " ".join(comment.split())[:60]
        parts = [r for r in re.split(r"[,;\s]+", code) if r]
        for r in parts:
            try:
                d = lc.normalize_id(r)
            except ValueError as e:
                raise IssueError(str(e)) from None
            if d not in out:
                out[d] = label if len(parts) == 1 else ""
    return list(out.items())


def parse_device_text(text: str) -> list[str]:
    """Device IDs only (see parse_device_entries). Raises IssueError."""
    return [d for d, _ in parse_device_entries(text)]


def check_expiry(exp: dt.date | None, issued: dt.date) -> dt.date | None:
    if exp is None:
        return None
    if exp < issued:
        raise IssueError("the end date is before today")
    if exp.year > MAX_EXPIRY_YEAR:
        raise IssueError(f"the end date is after {MAX_EXPIRY_YEAR}: choose 'no end date' instead")
    return exp


def issued_dir(key_path: Path) -> Path:
    return key_path.parent / "issued"


def issue_licence(
    key_path: Path,
    customer: str,
    devices: list[str],
    expires: dt.date | None,
    *,
    note: str = "",
    licence_id: str | None = None,
    allow_other_key: bool = False,
    today: dt.date | None = None,
    labels: dict[str, str] | None = None,
) -> dict:
    """Sign ONE licence for all `devices` (one key per customer) and keep a record of it in issued/.
    Returns {"body", "json", "key", "record"}. Raises IssueError (nothing is written then)."""
    seed, pub = _signing_key(key_path, allow_other_key)
    customer, note = _check_names(customer, note)
    if licence_id is not None and not _LICENCE_ID_RE.match(licence_id):
        raise IssueError("the licence ID may only use letters, digits, '.', '_' and '-' (up to 64)")
    ids = _normalize_all(devices)
    issued = today or dt.date.today()
    expires = check_expiry(expires, issued)
    body = _body(
        licence_id or f"LIC-{issued:%Y%m%d}-{secrets.token_hex(3)}", customer, issued, expires, ids, note
    )
    named = {}
    for d, label in (labels or {}).items():
        try:
            n = lc.normalize_id(d)
        except ValueError:
            continue
        label = " ".join(str(label).split())[:60]
        if n in ids and label:
            named[n] = label
    if named:
        body["device_labels"] = named  # for the vendor's records (renewal); devices ignore it
    item = _sign(seed, pub, body)
    _write_records(key_path, [item])
    return item


def issue_per_machine(
    key_path: Path,
    customer: str,
    entries: list[tuple[str, str]],
    expires: dt.date | None,
    *,
    note: str = "",
    allow_other_key: bool = False,
    today: dt.date | None = None,
) -> dict:
    """One licence (and key) per machine, all made together as a batch: `entries` are (device ID,
    label). Returns {"batch", "items": [{"body", "json", "key", "record", "device", "label"}]}. All
    or nothing: raises IssueError and writes nothing if any of them cannot be made."""
    seed, pub = _signing_key(key_path, allow_other_key)
    customer, note = _check_names(customer, note)
    labels: dict[str, str] = {}
    for d, label in entries:
        try:
            n = lc.normalize_id(d)
        except ValueError as e:
            raise IssueError(str(e)) from None
        labels.setdefault(n, " ".join(str(label).split())[:60])
    if not labels:
        raise IssueError("no machine IDs given")
    if len(labels) > 5000:
        raise IssueError("more than 5000 machines in one batch")
    issued = today or dt.date.today()
    expires = check_expiry(expires, issued)
    batch = f"LIC-{issued:%Y%m%d}-{secrets.token_hex(3)}"
    items = []
    for i, (dev, label) in enumerate(labels.items(), 1):
        body = _body(f"{batch}-{i:04d}", customer, issued, expires, [dev], note)
        body["batch"] = batch
        if label:
            body["device_label"] = label
        item = _sign(seed, pub, body)
        item.update(device=dev, label=label)
        items.append(item)
    _write_records(key_path, items)
    return {"batch": batch, "items": items}


def _signing_key(key_path: Path, allow_other_key: bool) -> tuple[bytes, bytes]:
    seed, pub = load_key(key_path)
    if pub != lc.vendor_public_key() and not allow_other_key:
        raise IssueError(
            f"the key {key_path} does not match the public key built into the software "
            "(anpr/licence.py PUBLIC_KEY_HEX): the software would reject this licence. "
            "Use the right key, or --other-key for a test licence."
        )
    return seed, pub


def _check_names(customer: str, note: str) -> tuple[str, str]:
    customer = customer.strip()
    if not customer:
        raise IssueError("the customer name is empty")
    if len(customer) > 120 or len(note) > 500:
        raise IssueError("the customer name or note is too long")
    return customer, note.strip()


def _normalize_all(devices: list[str]) -> list[str]:
    ids: list[str] = []
    for d in devices:
        try:
            n = lc.normalize_id(d)
        except ValueError as e:
            raise IssueError(str(e)) from None
        if n not in ids:
            ids.append(n)
    if not ids:
        raise IssueError("no machine IDs given")
    if len(ids) > 5000:
        raise IssueError("more than 5000 machine IDs in one licence")
    return ids


def _body(
    licence_id: str, customer: str, issued: dt.date, expires: dt.date | None, ids: list[str], note: str
) -> dict:
    return {
        "format": lc.FORMAT,
        "licence_id": licence_id,
        "customer": customer,
        "issued": issued.isoformat(),
        "expires": expires.isoformat() if expires else None,
        "devices": ids,
        "note": note,
    }


def _sign(seed: bytes, pub: bytes, body: dict) -> dict:
    sig = _ed25519.sign(seed, lc.signed_bytes(body))
    doc = {"licence": body, "signature": base64.b64encode(sig).decode()}
    text = json.dumps(doc, indent=2) + "\n"
    key = lc.encode_key(doc)
    lc.parse_licence(key, pub)  # never hand out a licence that does not verify (key = same document)
    return {"body": body, "json": text, "key": key, "record": None}


def _write_records(key_path: Path, items: list[dict]) -> None:
    """Keep a record of each licence in issued/ (owner-only files). All or nothing."""
    folder = issued_dir(key_path)
    folder.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    try:
        for item in items:
            record = folder / f"{item['body']['licence_id']}.json"
            try:
                fd = os.open(record, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                raise IssueError(
                    f"licence ID {item['body']['licence_id']} was already issued ({record}); pick another"
                ) from None
            written.append(record)
            with os.fdopen(fd, "w") as f:
                f.write(item["json"])
            item["record"] = record
            st = record.stat()  # just signed and checked: the list does not need to check it again
            _VERIFIED[(lc.vendor_public_key().hex(), str(record), st.st_mtime_ns, st.st_size)] = {
                "body": item["body"],
                "key": item["key"],
                "json": item["json"],
                "valid": True,
                "error": None,
                "file": record,
            }
    except BaseException:
        for f in written:
            f.unlink(missing_ok=True)
        raise


CSV_HEADER = ["Customer", "Machine ID", "Label", "Licence ID", "Valid until", "Licence key"]


def _csv_cell(v: object) -> str:
    s = "" if v is None else str(v)
    return "'" + s if s[:1] in ("=", "+", "-", "@", "\t", "\r") else s  # no formulas in Excel


def keys_xlsx(rows: list[dict]) -> bytes:
    """The same table as keys_csv, as an Excel workbook where every cell is text (Excel never turns a
    machine ID like 1234e567 or 00012345 into a number). Plain XML, no extra library."""
    import io
    import zipfile
    from xml.sax.saxutils import escape

    def cell(ref: str, v: object) -> str:
        t = escape("" if v is None else str(v))
        return f'<c r="{ref}" t="inlineStr"><is><t xml:space="preserve">{t}</t></is></c>'

    cols = "ABCDEF"
    table = [CSV_HEADER] + [
        [
            r["customer"],
            r["device"],
            r.get("label", ""),
            r["licence_id"],
            r["expires"] or "no end date",
            r["key"],
        ]
        for r in rows
    ]
    lines = [
        f'<row r="{i}">' + "".join(cell(f"{cols[j]}{i}", v) for j, v in enumerate(values)) + "</row>"
        for i, values in enumerate(table, 1)
    ]
    sheet = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<cols><col min="1" max="1" width="28" customWidth="1"/>'
        '<col min="2" max="2" width="22" customWidth="1"/>'
        '<col min="3" max="3" width="18" customWidth="1"/><col min="4" max="4" width="26" customWidth="1"/>'
        '<col min="5" max="5" width="13" customWidth="1"/>'
        '<col min="6" max="6" width="90" customWidth="1"/></cols>'
        "<sheetData>" + "".join(lines) + "</sheetData></worksheet>"
    )
    rels_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
    doc_rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    files = {
        "[Content_Types].xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" '
            'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            "</Types>"
        ),
        "_rels/.rels": (
            f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="{rels_ns}">'
            f'<Relationship Id="rId1" Type="{doc_rel}/officeDocument" '
            'Target="xl/workbook.xml"/></Relationships>'
        ),
        "xl/workbook.xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            f'xmlns:r="{doc_rel}">'
            '<sheets><sheet name="Licence keys" sheetId="1" r:id="rId1"/></sheets></workbook>'
        ),
        "xl/_rels/workbook.xml.rels": (
            f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="{rels_ns}">'
            f'<Relationship Id="rId1" Type="{doc_rel}/worksheet" '
            'Target="worksheets/sheet1.xml"/></Relationships>'
        ),
        "xl/worksheets/sheet1.xml": sheet,
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


def keys_csv(rows: list[dict]) -> str:
    """CSV (opens in Excel) of per-machine keys: rows of {"customer", "device", "label", "licence_id",
    "expires", "key"}. UTF-8 with BOM so Excel shows names correctly."""
    import csv
    import io

    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\r\n")
    w.writerow(CSV_HEADER)
    for r in rows:
        w.writerow(
            [
                _csv_cell(r["customer"]),
                _csv_cell(r["device"]),
                _csv_cell(r.get("label", "")),
                _csv_cell(r["licence_id"]),
                _csv_cell(r["expires"] or "no end date"),
                r["key"],
            ]
        )
    return "\ufeff" + buf.getvalue()


def list_issued(key_path: Path) -> list[dict]:
    """Every licence in issued/, newest first: {"body", "key", "json", "valid", "error", "file"}.
    Each file's signature is checked once and remembered until the file changes (thousands of
    per-machine keys would otherwise take many seconds on every list)."""
    pub = lc.vendor_public_key()
    out = []
    for f in issued_dir(key_path).glob("*.json"):
        try:
            st = f.stat()
            ck = (pub.hex(), str(f), st.st_mtime_ns, st.st_size)
            rec = _VERIFIED.get(ck)
            if rec is None:
                text = f.read_text()
                doc = json.loads(text)
                body = doc["licence"]
                key = lc.encode_key(doc)
                lc.parse_licence(key, pub)
                rec = {"body": body, "key": key, "json": text, "valid": True, "error": None, "file": f}
                if len(_VERIFIED) > 200_000:
                    _VERIFIED.clear()
                _VERIFIED[ck] = rec
            out.append(rec)
        except (OSError, ValueError, KeyError, TypeError, lc.LicenceError) as e:
            out.append({"body": None, "key": None, "json": None, "valid": False, "error": str(e), "file": f})
    out.sort(key=lambda r: (r["body"] or {}).get("issued", ""), reverse=True)
    return out


def _cli_entries(args: argparse.Namespace) -> list[tuple[str, str]]:
    text = "\n".join(args.devices or [])
    if args.devices_file:
        text += "\n" + Path(args.devices_file).read_text()
    entries = parse_device_entries(text)
    if not entries:
        raise IssueError("no device IDs given (--devices or --devices-file)")
    return entries


def _cli_expiry(args: argparse.Namespace, issued: dt.date) -> dt.date | None:
    given = [args.expires is not None, args.days is not None, args.no_expiry]
    if sum(given) != 1:
        raise IssueError("give exactly one of --expires YYYY-MM-DD, --days N or --no-expiry")
    if args.no_expiry:
        return None
    if args.days is not None:
        if args.days < 1:
            raise IssueError("--days must be at least 1")
        return issued + dt.timedelta(days=args.days - 1)  # --days 365: today + 364 more days
    try:
        exp = dt.date.fromisoformat(args.expires)
    except ValueError:
        raise IssueError(f"--expires must be YYYY-MM-DD, not {args.expires!r}") from None
    return check_expiry(exp, issued)


def cmd_issue(args: argparse.Namespace) -> int:
    issued = dt.date.today()
    expires = _cli_expiry(args, issued)
    entries = _cli_entries(args)
    devices = [d for d, _ in entries]
    if args.out is None:
        args.out = "licence-keys.csv" if args.per_machine else "licence.json"
    out = Path(args.out)
    if out.exists() and not args.force:
        raise IssueError(f"{out} exists (use --force to replace it)")
    if out.resolve() == args.key.resolve():
        raise IssueError("--out is the signing key file")
    folder = out.resolve().parent  # can the output be written? (checked before anything is signed)
    if (
        not folder.is_dir()
        or not os.access(folder, os.W_OK)
        or (out.exists() and not os.access(out, os.W_OK))
    ):
        raise IssueError(f"cannot write {out}: no such folder, or no permission")
    if args.per_machine:
        if args.licence_id:
            raise IssueError("--licence-id cannot be used with --per-machine")
        b = issue_per_machine(
            args.key,
            args.customer,
            entries,
            expires,
            note=args.note or "",
            allow_other_key=args.other_key,
            today=issued,
        )
        rows = [
            {
                "customer": i["body"]["customer"],
                "device": i["device"],
                "label": i["label"],
                "licence_id": i["body"]["licence_id"],
                "expires": i["body"]["expires"],
                "key": i["key"],
            }
            for i in b["items"]
        ]
        try:
            if out.suffix.lower() == ".xlsx":
                out.write_bytes(keys_xlsx(rows))
            else:
                out.write_text(keys_csv(rows), encoding="utf-8")
        except OSError as e:
            for i in b["items"]:
                Path(i["record"]).unlink(missing_ok=True)  # nobody received these keys
            raise IssueError(f"cannot write {out}: {e}") from None
        print(f"batch {b['batch']}: {len(rows)} licence keys (one per machine) for {rows[0]['customer']}")
        print(f"written: {out}  (open in Excel; give each device its own key)")
        return 0
    r = issue_licence(
        args.key,
        args.customer,
        devices,
        expires,
        note=args.note or "",
        licence_id=args.licence_id,
        allow_other_key=args.other_key,
        today=issued,
        labels=dict(entries),
    )
    try:
        out.write_text(r["json"])
    except OSError as e:
        Path(r["record"]).unlink(missing_ok=True)  # nobody received this licence
        raise IssueError(f"cannot write {out}: {e}") from None
    body = r["body"]
    print(
        f"licence {body['licence_id']} for {body['customer']}: {len(body['devices'])} device(s), "
        f"{'valid until ' + body['expires'] if body['expires'] else 'no end date'}"
    )
    print(f"written: {out}\nrecord:  {r['record']}")
    print(f"licence key (paste on the device's Activation page):\n{r['key']}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    pub = load_key(args.key)[1] if args.with_key else lc.vendor_public_key()
    try:
        lic = lc.read_licence(Path(args.file), pub)
    except lc.LicenceError as e:
        print(f"NOT VALID: {e}")
        return 2
    print(f"licence:  {lic.licence_id}")
    print(f"customer: {lic.customer}")
    print(f"issued:   {lic.issued}")
    print(f"expires:  {lic.expires or 'never'}")
    print(f"devices:  {len(lic.devices)}")
    for d in sorted(lic.devices):
        print(f"  {d}")
    if lic.note:
        print(f"note:     {lic.note}")
    if args.device:
        try:
            lc.check(lic, args.device, time.time())
        except (lc.LicenceError, ValueError) as e:
            print(f"device {args.device}: NOT VALID: {e}")
            return 2
        print(f"device {args.device}: valid")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="licence_tool.py", description="ANPR licence tool (vendor only)")
    ap.add_argument(
        "--key", type=Path, default=None, help="signing key file (default ~/anpr-licensing/vendor_key.json)"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("keygen", help="make the signing key (once)")

    p = sub.add_parser("issue", help="make a licence file")
    p.add_argument("--customer", required=True)
    p.add_argument("--devices", action="append", help="device IDs, comma separated (repeatable)")
    p.add_argument("--devices-file", help="text file with device IDs (one per line, # comments)")
    p.add_argument("--expires", help="last valid day, YYYY-MM-DD")
    p.add_argument("--days", type=int, help="valid for N days from today")
    p.add_argument("--no-expiry", action="store_true", help="never expires")
    p.add_argument("--licence-id", help="default: LIC-<date>-<random>")
    p.add_argument("--note", help="free text kept in the licence")
    p.add_argument(
        "--out",
        default=None,
        help="licence file (default licence.json); "
        "with --per-machine a .csv (default licence-keys.csv) or .xlsx",
    )
    p.add_argument("--per-machine", action="store_true", help="one key per machine instead of one for all")
    p.add_argument("--force", action="store_true", help="replace --out if it exists")
    p.add_argument("--other-key", action="store_true", help="allow a key that does not match the software")

    s = sub.add_parser("show", help="print and check a licence file")
    s.add_argument("file")
    s.add_argument("--device", help="also check it for this device ID")
    s.add_argument("--with-key", action="store_true", help="verify with --key instead of the built-in key")

    args = ap.parse_args(argv)
    args.key = (args.key or default_key_path()).expanduser()
    try:
        return {"keygen": cmd_keygen, "issue": cmd_issue, "show": cmd_show}[args.cmd](args)
    except IssueError as e:
        raise SystemExit(str(e)) from None


if __name__ == "__main__":
    sys.exit(main())
