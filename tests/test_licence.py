"""Licence system: signatures, licence files, device IDs, dates, the running guard and the vendor tool."""

from __future__ import annotations

import base64
import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from anpr import _ed25519 as ed
from anpr import licence as lc

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import licence_tool  # noqa: E402

# RFC 8032 section 7.1, tests 1-3: (secret, public, message, signature)
RFC_VECTORS = [
    (
        "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
        "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
        "",
        "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b",
    ),
    (
        "4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb",
        "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c",
        "72",
        "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00",
    ),
    (
        "c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7",
        "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025",
        "af82",
        "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac18ff9b538d16f290ae67f760984dc6594a7c15e9716ed28dc027beceea1ec40a",
    ),
]

SEED = bytes(range(32))
PUB = ed.public_key(SEED)
DEV = "a1b2c3d4"


def make_licence(devices=(DEV,), issued="2026-10-01", expires="2027-09-30", seed=SEED, **extra) -> str:
    body = {
        "format": 1,
        "licence_id": "LIC-TEST-0001",
        "customer": "Acme Logistics",
        "issued": issued,
        "expires": expires,
        "devices": list(devices),
        "note": "",
        **extra,
    }
    sig = ed.sign(seed, lc.signed_bytes(body))
    return json.dumps({"licence": body, "signature": base64.b64encode(sig).decode()})


def ts(day: str, hour: int = 12) -> float:
    return dt.datetime.combine(dt.date.fromisoformat(day), dt.time(hour)).timestamp()


class MemStore:
    def __init__(self) -> None:
        self.d: dict[str, object] = {}

    def read_setting(self, key):
        return self.d.get(key)

    def write_setting(self, key, value):
        self.d[key] = json.loads(json.dumps(value))


# ---- Ed25519 ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(("sk", "pk", "msg", "sig"), RFC_VECTORS)
def test_rfc8032_vectors(sk, pk, msg, sig):
    sk, pk, msg, sig = map(bytes.fromhex, (sk, pk, msg, sig))
    assert ed.public_key(sk) == pk
    assert ed.sign(sk, msg) == sig
    assert ed.verify(pk, msg, sig)
    assert not ed.verify(pk, msg + b"!", sig)


def test_verify_rejects_damaged_signatures():
    msg = b"hello"
    sig = ed.sign(SEED, msg)
    for i in (0, 31, 32, 63):
        bad = bytearray(sig)
        bad[i] ^= 1
        assert not ed.verify(PUB, msg, bytes(bad))
    assert not ed.verify(PUB, msg, sig[:63])
    assert not ed.verify(PUB[:31], msg, sig)
    assert not ed.verify(ed.public_key(bytes(32)), msg, sig)  # someone else's key
    # s >= group order (signature malleability) must be refused
    s = int.from_bytes(sig[32:], "little") + ed._Q
    assert s < 2**256
    assert not ed.verify(PUB, msg, sig[:32] + s.to_bytes(32, "little"))


def _openssl_ed25519() -> bool:
    exe = shutil.which("openssl")
    if not exe:
        return False
    r = subprocess.run([exe, "list", "-public-key-algorithms"], capture_output=True, text=True)
    return "ED25519" in r.stdout.upper()


@pytest.mark.skipif(not _openssl_ed25519(), reason="openssl with Ed25519 not available")
def test_matches_openssl(tmp_path):
    key, msg, sig, pub = (tmp_path / n for n in ("k.pem", "m.bin", "s.bin", "p.pem"))
    msg.write_bytes(lc.signed_bytes({"x": 1, "devices": ["a"]}))
    run = lambda *a: subprocess.run(["openssl", *a], check=True, capture_output=True)  # noqa: E731
    run("genpkey", "-algorithm", "ed25519", "-out", str(key))
    run("pkeyutl", "-sign", "-inkey", str(key), "-rawin", "-in", str(msg), "-out", str(sig))
    der = run("pkey", "-in", str(key), "-pubout", "-outform", "DER").stdout
    raw_pub = der[-32:]
    assert ed.verify(raw_pub, msg.read_bytes(), sig.read_bytes())  # OpenSSL signs, we verify
    raw_seed = run("pkey", "-in", str(key), "-outform", "DER").stdout[-32:]
    assert ed.public_key(raw_seed) == raw_pub
    sig.write_bytes(ed.sign(raw_seed, msg.read_bytes()))  # we sign, OpenSSL verifies
    run("pkey", "-in", str(key), "-pubout", "-out", str(pub))
    run("pkeyutl", "-verify", "-pubin", "-inkey", str(pub), "-rawin", "-in", str(msg), "-sigfile", str(sig))


# ---- device ID -------------------------------------------------------------------------------------


def test_normalize_id():
    assert lc.normalize_id("00000000A1B2C3D4") == "a1b2c3d4"
    assert lc.normalize_id("  a1b2c3d4\n") == "a1b2c3d4"
    assert lc.normalize_id("a1b2c3d4\x00") == "a1b2c3d4"
    assert lc.normalize_id("C02FM7AAMD6R") == "c02fm7aamd6r"
    assert lc.normalize_id("0000") == "0"
    for bad in ("", "a/b", "../x", "x" * 65, "ab cd!"):
        with pytest.raises(ValueError):
            lc.normalize_id(bad)


def test_machine_id_reads_the_pi_serial(monkeypatch):
    cpuinfo = (
        "processor\t: 0\nHardware\t: BCM2835\nRevision\t: a020d3\nSerial\t\t: 00000000a1b2c3d4\nModel\t: Pi\n"
    )
    files = {"/proc/cpuinfo": cpuinfo}
    monkeypatch.setattr(lc, "_read", lambda p: files.get(p, ""))
    assert lc.machine_id() == "a1b2c3d4"
    files["/sys/firmware/devicetree/base/serial-number"] = "00000000deadbeef\x00"
    assert lc.machine_id() == "deadbeef"  # device tree wins


def test_machine_id_unreadable(monkeypatch):
    monkeypatch.setattr(lc, "_read", lambda p: "")
    monkeypatch.setattr(lc.sys, "platform", "linux")
    with pytest.raises(lc.LicenceError, match="serial"):
        lc.machine_id()


def test_machine_id_on_this_computer():
    assert lc.normalize_id(lc.machine_id()) == lc.machine_id()


# ---- licence file ----------------------------------------------------------------------------------


def test_good_licence_parses():
    lic = lc.parse_licence(make_licence(devices=["00000000A1B2C3D4", "9f8e7d6c"]), PUB)
    assert lic.customer == "Acme Logistics" and lic.licence_id == "LIC-TEST-0001"
    assert lic.devices == {"a1b2c3d4", "9f8e7d6c"}
    assert lic.issued == dt.date(2026, 10, 1) and lic.expires == dt.date(2027, 9, 30)
    lc.check(lic, "00000000a1b2c3d4", ts("2027-01-01"))


@pytest.mark.parametrize(
    "change",
    [
        lambda b: b.update(expires="2099-12-31"),
        lambda b: b.update(customer="Someone Else"),
        lambda b: b["devices"].append("ffffffff"),
        lambda b: b.update(expires=None),
        lambda b: b.update(extra_field=1),
    ],
)
def test_any_change_breaks_the_signature(change):
    doc = json.loads(make_licence())
    change(doc["licence"])
    with pytest.raises(lc.LicenceError, match="signature is not valid"):
        lc.parse_licence(json.dumps(doc), PUB)


def test_licence_from_another_key_is_refused():
    with pytest.raises(lc.LicenceError, match="signature is not valid"):
        lc.parse_licence(make_licence(seed=bytes(32)), PUB)


def test_reformatted_file_still_valid():
    """Opening and saving the file in an editor (other spacing, key order) does not break it."""
    doc = json.loads(make_licence())
    doc["licence"] = dict(reversed(list(doc["licence"].items())))
    assert lc.parse_licence(json.dumps(doc, indent=4), PUB).customer == "Acme Logistics"


@pytest.mark.parametrize(
    ("text", "msg"),
    [
        ("not json", "not valid JSON"),
        ("[]", "not a licence"),
        ('{"licence": {}}', "no signature"),
        ('{"licence": {}, "signature": "***"}', "signature is damaged"),
        ('{"licence": {}, "signature": "' + base64.b64encode(b"x" * 64).decode() + '"}', "not valid"),
    ],
)
def test_damaged_files(text, msg):
    with pytest.raises(lc.LicenceError, match=msg):
        lc.parse_licence(text, PUB)


def test_signed_but_unsupported_or_malformed():
    with pytest.raises(lc.LicenceError, match="format"):
        lc.parse_licence(make_licence(format=2), PUB)
    with pytest.raises(lc.LicenceError, match="devices"):
        lc.parse_licence(make_licence(devices=[]), PUB)
    with pytest.raises(lc.LicenceError, match="devices"):
        lc.parse_licence(make_licence(devices=["bad id!"]), PUB)
    with pytest.raises(lc.LicenceError, match="expires"):
        lc.parse_licence(make_licence(expires="next year"), PUB)


def test_wrong_device():
    lic = lc.parse_licence(make_licence(), PUB)
    with pytest.raises(lc.LicenceError, match="not for this device.*deadbeef"):
        lc.check(lic, "deadbeef", ts("2027-01-01"))


def test_end_date_is_the_last_valid_day():
    lic = lc.parse_licence(make_licence(expires="2027-09-30"), PUB)
    lc.check(lic, DEV, ts("2027-09-30", 23))  # last day: still fine
    with pytest.raises(lc.LicenceError, match="ended on 2027-09-30"):
        lc.check(lic, DEV, ts("2027-10-01", 0))
    assert lic.days_left(ts("2027-09-30", 0)) == 1
    assert lic.days_left(ts("2028-01-01")) == 0


def test_no_end_date():
    lic = lc.parse_licence(make_licence(expires=None), PUB)
    assert lic.ends_at() is None and lic.days_left(ts("2090-01-01")) is None
    lc.check(lic, DEV, ts("2090-01-01"))


def test_clock_before_issue_date_is_refused():
    lic = lc.parse_licence(make_licence(issued="2026-10-01"), PUB)
    lc.check(lic, DEV, ts("2026-09-30", 12))  # a day of slack
    with pytest.raises(lc.LicenceError, match="clock"):
        lc.check(lic, DEV, ts("2026-09-01"))


def test_read_licence_missing_and_too_big(tmp_path):
    with pytest.raises(lc.LicenceError, match="no licence file"):
        lc.read_licence(tmp_path / "licence.json", PUB)
    big = tmp_path / "big.json"
    big.write_bytes(b" " * (lc._MAX_FILE_BYTES + 1))
    with pytest.raises(lc.LicenceError, match="too big"):
        lc.read_licence(big, PUB)


# ---- running guard ---------------------------------------------------------------------------------


class Clock:
    def __init__(self, t: float) -> None:
        self.t, self.m = t, 0.0

    def __call__(self) -> float:
        return self.t


def guard_for(tmp_path, clock, store=None, text=None, **kw):
    path = tmp_path / "licence.json"
    path.write_text(text or make_licence())
    return lc.LicenceGuard.open(
        path, store, public_key=PUB, device_id=DEV, clock=clock, mono=lambda: clock.m, **kw
    )


def test_guard_ok_records_status(tmp_path):
    store, clock = MemStore(), Clock(ts("2027-01-01"))
    g = guard_for(tmp_path, clock, store)
    assert not g.expired()
    st = store.d[lc.STATUS_KEY]
    assert st["ok"] is True and st["customer"] == "Acme Logistics" and st["expires"] == "2027-09-30"
    assert st["device_id"] == DEV and st["days_left"] == 272


def test_guard_ends_while_running_rechecking_hourly(tmp_path):
    store, clock = MemStore(), Clock(ts("2027-09-30", 22))
    g = guard_for(tmp_path, clock, store, recheck_s=3600)
    clock.t = ts("2027-10-01", 1)  # past the end, but not rechecked yet
    clock.m = 1800
    assert not g.expired()
    clock.m = 3600
    assert g.expired()
    assert "ended on 2027-09-30" in g.error
    assert store.d[lc.STATUS_KEY]["ok"] is False
    clock.t = ts("2027-01-01")
    assert g.expired()  # stays ended for this run


def test_setting_the_clock_back_does_not_help(tmp_path):
    store, clock = MemStore(), Clock(ts("2027-10-05"))
    with pytest.raises(lc.LicenceError, match="ended"):
        guard_for(tmp_path, clock, store)
    clock.t = ts("2027-06-01")  # someone sets the date back
    with pytest.raises(lc.LicenceError, match="ended"):
        guard_for(tmp_path, clock, store)
    assert store.d[lc.STATUS_KEY]["ok"] is False


def test_guard_without_store_and_broken_store(tmp_path):
    class Broken:
        def read_setting(self, key):
            raise RuntimeError("db locked")

        def write_setting(self, key, value):
            raise RuntimeError("db locked")

    clock = Clock(ts("2027-01-01"))
    assert not guard_for(tmp_path, clock, None).expired()
    assert not guard_for(tmp_path, clock, Broken()).expired()


def test_open_records_missing_file(tmp_path):
    store = MemStore()
    with pytest.raises(lc.LicenceError):
        lc.LicenceGuard.open(tmp_path / "nope.json", store, public_key=PUB, device_id=DEV)
    st = store.d[lc.STATUS_KEY]
    assert st["ok"] is False and "no licence file" in st["error"] and st["device_id"] == DEV


def test_wait_for_licence(tmp_path):
    path = tmp_path / "licence.json"
    kw = {"public_key": PUB, "device_id": DEV}
    assert lc.wait_for_licence(path, None, stop=lambda: False, wait=False, **kw) is None

    sleeps: list[float] = []
    assert lc.wait_for_licence(path, None, stop=lambda: len(sleeps) >= 3, sleep=sleeps.append, **kw) is None

    def sleep(s: float) -> None:  # the licence file arrives while the engine waits
        sleeps.append(s)
        if len(sleeps) == 70:
            path.write_text(make_licence(expires=None))

    sleeps.clear()
    g = lc.wait_for_licence(path, None, stop=lambda: False, sleep=sleep, retry_s=60, **kw)
    assert g is not None and g.licence.customer == "Acme Logistics"
    assert 70 <= len(sleeps) <= 120


# ---- engine command line ---------------------------------------------------------------------------


def _cfg(tmp_path) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"storage:\n  db_path: {tmp_path / 'anpr.db'}\n  image_dir: {tmp_path / 'img'}\n")
    return cfg


def test_engine_machine_id(capsys):
    from anpr.engine import main

    assert main(["--machine-id"]) == 0
    assert capsys.readouterr().out.strip() == lc.machine_id()


def test_engine_refuses_to_run_without_licence(tmp_path, monkeypatch, capsys):
    from anpr import engine

    def boom(*a, **k):
        raise AssertionError("must not get this far")

    monkeypatch.setattr("anpr.detector.make_detector", boom)
    cfg = _cfg(tmp_path)
    assert engine.main(["--config", str(cfg), "--check-licence"]) == engine.EXIT_LICENCE
    assert "no licence file" in capsys.readouterr().out
    assert engine.main(["--config", str(cfg), "--exit-at-end"]) == engine.EXIT_LICENCE


def test_engine_check_licence_ok(tmp_path, monkeypatch, capsys):
    from anpr import engine

    monkeypatch.setattr(lc, "PUBLIC_KEY_HEX", PUB.hex())
    monkeypatch.setattr(lc, "machine_id", lambda: DEV)
    cfg = _cfg(tmp_path)
    (tmp_path / "licence.json").write_text(make_licence(expires=None))
    assert engine.main(["--config", str(cfg), "--check-licence"]) == 0
    assert "licence OK: LIC-TEST-0001 for Acme Logistics" in capsys.readouterr().out
    other = tmp_path / "other.json"
    other.write_text(make_licence(devices=["ffff"], expires=None))
    assert engine.main(["--config", str(cfg), "--licence", str(other), "--check-licence"]) == 4


def test_built_in_key_is_a_real_key():
    pub = lc.vendor_public_key()
    assert len(pub) == 32 and pub != bytes(32) and ed._decompress(pub) is not None


# ---- vendor tool -----------------------------------------------------------------------------------


def test_tool_keygen_issue_show(tmp_path, monkeypatch, capsys):
    key = tmp_path / "keys" / "vendor_key.json"
    assert licence_tool.main(["--key", str(key), "keygen"]) == 0
    assert key.stat().st_mode & 0o077 == 0  # only the owner can read it
    assert licence_tool.main(["--key", str(key), "keygen"]) == 1  # never overwritten
    pub = bytes.fromhex(json.loads(key.read_text())["public_key"])
    monkeypatch.setattr(lc, "PUBLIC_KEY_HEX", pub.hex())

    serials = tmp_path / "serials.txt"
    serials.write_text("# gate Pis\n00000000A1B2C3D4\n9f8e7d6c, 0000000011112222  # yard\n\na1b2c3d4\n")
    out = tmp_path / "licence.json"
    args = ["--key", str(key), "issue", "--customer", "Acme", "--devices-file", str(serials), "--days", "365"]
    assert licence_tool.main([*args, "--devices", "abcd", "--out", str(out)]) == 0
    lic = lc.read_licence(out, pub)
    assert lic.devices == {"a1b2c3d4", "9f8e7d6c", "11112222", "abcd"}
    assert lic.expires == dt.date.today() + dt.timedelta(days=364)
    assert (key.parent / "issued" / f"{lic.licence_id}.json").read_text() == out.read_text()
    with pytest.raises(SystemExit, match="exists"):
        licence_tool.main([*args, "--out", str(out)])

    capsys.readouterr()
    assert licence_tool.main(["--key", str(key), "show", str(out), "--device", "00000000a1b2c3d4"]) == 0
    assert "device 00000000a1b2c3d4: valid" in capsys.readouterr().out
    assert licence_tool.main(["--key", str(key), "show", str(out), "--device", "ffffffff"]) == 2
    out.write_text(out.read_text().replace("Acme", "Acme2"))
    assert licence_tool.main(["--key", str(key), "show", str(out)]) == 2
    assert "NOT VALID" in capsys.readouterr().out


def test_tool_issue_argument_checks(tmp_path):
    key = tmp_path / "k.json"
    licence_tool.main(["--key", str(key), "keygen"])
    base = [
        "--key",
        str(key),
        "issue",
        "--customer",
        "Acme",
        "--out",
        str(tmp_path / "l.json"),
        "--other-key",
    ]
    with pytest.raises(SystemExit, match="exactly one"):
        licence_tool.main([*base, "--devices", "abcd"])
    with pytest.raises(SystemExit, match="exactly one"):
        licence_tool.main([*base, "--devices", "abcd", "--days", "3", "--no-expiry"])
    with pytest.raises(SystemExit, match="no device IDs"):
        licence_tool.main([*base, "--days", "3"])
    with pytest.raises(SystemExit, match="not a valid device ID"):
        licence_tool.main([*base, "--devices", "bad/id", "--days", "3"])
    with pytest.raises(SystemExit, match="before today"):
        licence_tool.main([*base, "--devices", "abcd", "--expires", "2000-01-01"])
    with pytest.raises(SystemExit, match="no signing key"):
        licence_tool.main(["--key", str(tmp_path / "none.json"), *base[2:], "--devices", "a", "--days", "3"])
    with pytest.raises(SystemExit, match="after 2200"):
        licence_tool.main([*base, "--devices", "abcd", "--expires", "9999-12-31"])
    for bad_id in ("../vendor_key", "a/b", ".hidden", "x" * 65):
        with pytest.raises(SystemExit, match="licence ID may only"):
            licence_tool.main([*base, "--devices", "abcd", "--days", "3", "--licence-id", bad_id])
    assert json.loads(key.read_text())["kind"] == licence_tool.KEY_KIND  # the key was never touched


def test_tool_refuses_a_key_that_does_not_match_the_software(tmp_path):
    key = tmp_path / "k.json"
    licence_tool.main(["--key", str(key), "keygen"])
    with pytest.raises(SystemExit, match="does not match"):
        licence_tool.main(
            [
                "--key",
                str(key),
                "issue",
                "--customer",
                "A",
                "--devices",
                "ab",
                "--days",
                "3",
                "--out",
                str(tmp_path / "l.json"),
            ]
        )
    assert not (tmp_path / "l.json").exists()


def test_tool_devices_with_spaces_and_reused_ids(tmp_path, monkeypatch):
    key = tmp_path / "k.json"
    licence_tool.main(["--key", str(key), "keygen"])
    monkeypatch.setattr(lc, "PUBLIC_KEY_HEX", json.loads(key.read_text())["public_key"])
    out = tmp_path / "l.json"
    args = ["--key", str(key), "issue", "--customer", "A", "--days", "3", "--licence-id", "LIC-A"]
    assert (
        licence_tool.main([*args, "--devices", "00000000a1b2c3d4 00000000b5c6d7e8\n9f9f", "--out", str(out)])
        == 0
    )
    assert lc.read_licence(out, lc.vendor_public_key()).devices == {"a1b2c3d4", "b5c6d7e8", "9f9f"}
    with pytest.raises(SystemExit, match="already issued"):
        licence_tool.main([*args, "--devices", "ab", "--out", str(tmp_path / "l2.json")])
    with pytest.raises(SystemExit, match="signing key file"):
        licence_tool.main(
            [*args[:-2], "--licence-id", "LIC-B", "--devices", "ab", "--out", str(key), "--force"]
        )


# ---- review fixes ----------------------------------------------------------------------------------


def test_far_future_end_date_does_not_crash(tmp_path):
    lic = lc.parse_licence(make_licence(expires="9999-12-31"), PUB)
    assert lic.days_left(ts("2027-01-01")) is None
    lc.check(lic, DEV, ts("2090-01-01"))
    assert not guard_for(
        tmp_path, Clock(ts("2027-01-01")), MemStore(), text=make_licence(expires="9999-12-31")
    ).expired()


def test_licence_saved_with_a_bom_is_accepted(tmp_path):
    p = tmp_path / "licence.json"
    p.write_bytes(b"\xef\xbb\xbf" + make_licence().encode())
    assert lc.read_licence(p, PUB).customer == "Acme Logistics"


def test_non_pi_linux_id_can_never_equal_a_pi_serial(monkeypatch):
    files = {"/etc/machine-id": "a1b2c3d4\n"}
    monkeypatch.setattr(lc, "_read", lambda p: files.get(p, ""))
    monkeypatch.setattr(lc.sys, "platform", "linux")
    assert lc.machine_id() == "mid-a1b2c3d4"


def test_clock_jump_ahead_while_running_does_not_stop_a_licensed_device(tmp_path):
    """A mistyped year (or a bad time server) during a run: ignored, and nothing is remembered."""
    store, clock = MemStore(), Clock(ts("2027-01-01"))
    g = guard_for(tmp_path, clock, store)
    clock.t, clock.m = ts("2076-01-01"), 600  # jumped 49 years in 10 minutes
    assert not g.expired()
    clock.t, clock.m = ts("2027-01-01", 13), 3600  # fixed again
    assert not g.expired()
    assert guard_for(tmp_path, clock, store) is not None  # a restart is fine too
    assert max(store.d["licence_clock"].values()) < ts("2027-01-02")


def test_real_time_passing_still_ends_the_licence_while_running(tmp_path):
    clock = Clock(ts("2027-09-28"))
    g = guard_for(tmp_path, clock, MemStore())
    for day in range(1, 5):  # four days of running, checked every 10 minutes is not needed: once a day
        clock.t, clock.m = ts("2027-09-28") + day * 86400, day * 86400.0
        if g.expired():
            break
    assert "ended on 2027-09-30" in (g.error or "")


def test_a_wrong_time_remembered_at_startup_is_cleared_by_a_new_licence(tmp_path):
    store = MemStore()
    with pytest.raises(lc.LicenceError, match="ended"):
        guard_for(tmp_path, Clock(ts("2076-01-01")), store)  # booted with a far-future clock
    with pytest.raises(lc.LicenceError, match="ended"):
        guard_for(tmp_path, Clock(ts("2027-01-01")), store)  # remembered: same licence stays refused
    renewed = json.loads(make_licence())
    renewed["licence"]["licence_id"] = "LIC-TEST-0002"
    renewed["signature"] = base64.b64encode(ed.sign(SEED, lc.signed_bytes(renewed["licence"]))).decode()
    assert not guard_for(tmp_path, Clock(ts("2027-01-01")), store, text=json.dumps(renewed)).expired()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "x", True, -5, {"LIC-TEST-0001": float("nan")}])
def test_broken_clock_value_is_ignored_and_rewritten(tmp_path, bad):
    store = MemStore()
    store.d["licence_clock"] = bad  # MemStore keeps the value as is
    clock = Clock(ts("2027-01-01"))
    assert not guard_for(tmp_path, clock, store).expired()
    mark = store.d["licence_clock"]["LIC-TEST-0001"]
    assert mark == pytest.approx(ts("2027-01-01"))


def test_clock_set_back_after_the_end_is_still_refused(tmp_path):
    store = MemStore()
    with pytest.raises(lc.LicenceError):
        guard_for(tmp_path, Clock(ts("2027-10-05")), store)
    with pytest.raises(lc.LicenceError, match="ended"):
        guard_for(tmp_path, Clock(ts("2027-03-01")), store)


def test_renewed_licence_is_picked_up_while_running(tmp_path):
    store, clock = MemStore(), Clock(ts("2027-09-30", 20))
    g = guard_for(tmp_path, clock, store, recheck_s=600)
    renewed = json.loads(make_licence(expires="2028-09-30"))
    renewed["licence"]["licence_id"] = "LIC-TEST-0002"
    renewed["signature"] = base64.b64encode(ed.sign(SEED, lc.signed_bytes(renewed["licence"]))).decode()
    import os

    (tmp_path / "licence.json").write_text(json.dumps(renewed))
    os.utime(tmp_path / "licence.json", ns=(1, 1))  # a different file time, as a copy would give
    clock.t, clock.m = ts("2027-10-01", 2), 6 * 3600
    assert not g.expired()
    assert g.licence.licence_id == "LIC-TEST-0002" and store.d[lc.STATUS_KEY]["expires"] == "2028-09-30"


def test_a_bad_new_file_keeps_the_running_licence(tmp_path):
    clock = Clock(ts("2027-01-01"))
    g = guard_for(tmp_path, clock, MemStore(), recheck_s=600)
    (tmp_path / "licence.json").write_text(make_licence(devices=["ffff"]))
    clock.m = 700
    assert not g.expired() and g.licence.devices == {DEV}


def test_check_licence_sees_the_same_remembered_time(tmp_path, monkeypatch, capsys):
    from anpr import engine
    from anpr.storage import SqliteEventStore

    monkeypatch.setattr(lc, "PUBLIC_KEY_HEX", PUB.hex())
    monkeypatch.setattr(lc, "machine_id", lambda: DEV)
    cfg = _cfg(tmp_path)
    (tmp_path / "licence.json").write_text(make_licence())
    st = SqliteEventStore(tmp_path / "anpr.db", tmp_path / "img")
    st.write_setting("licence_clock", {"LIC-TEST-0001": ts("2027-12-01")})  # the engine saw this time
    st.close()
    assert engine.main(["--config", str(cfg), "--check-licence"]) == engine.EXIT_LICENCE
    assert "ended" in capsys.readouterr().out


# ---- licence key (one line to paste) --------------------------------------------------------------


def key_of(text: str) -> str:
    return lc.encode_key(json.loads(text))


def test_key_round_trip_and_paste_damage():
    k = key_of(make_licence(devices=[f"{i:016x}" for i in range(150)]))
    assert k.startswith("ANPR1-") and "\n" not in k and len(k) < 4000  # 150 machines: still pasteable
    lic = lc.parse_licence(k, PUB)
    assert len(lic.devices) == 150
    wrapped = "  " + "\n".join(k[i : i + 60] for i in range(0, len(k), 60)) + "\n"  # e-mail line breaks
    assert lc.parse_licence(wrapped, PUB) == lic
    assert lc.parse_licence("anpr1-" + k[6:], PUB) == lic


@pytest.mark.parametrize(
    ("key", "msg"),
    [
        ("ANPR1-", "damaged or incomplete"),
        ("ANPR1-@@@@", "damaged or incomplete"),
        ("HELLO-abc", "not a licence"),
    ],
)
def test_bad_keys(key, msg):
    with pytest.raises(lc.LicenceError, match=msg):
        lc.key_to_json(key)


def test_cut_short_or_edited_key_is_refused():
    k = key_of(make_licence())
    with pytest.raises(lc.LicenceError, match="damaged or incomplete"):
        lc.parse_licence(k[:-10], PUB)
    doc = json.loads(make_licence())
    doc["licence"]["expires"] = "2099-12-31"
    with pytest.raises(lc.LicenceError, match="signature is not valid"):
        lc.parse_licence(lc.encode_key(doc), PUB)


def test_key_zip_bomb_is_refused():
    import zlib

    bomb = lc.KEY_PREFIX + base64.urlsafe_b64encode(zlib.compress(b" " * 5_000_000, 9)).decode()
    with pytest.raises(lc.LicenceError, match="damaged or incomplete"):
        lc.key_to_json(bomb)


def test_install_licence_checks_device_and_date_before_writing(tmp_path):
    p = tmp_path / "licence.json"
    k = key_of(make_licence(devices=[DEV, "ffff"]))
    with pytest.raises(lc.LicenceError, match="not for this device"):
        lc.install_licence(k, p, "deadbeef", now=ts("2027-01-01"), public_key=PUB)
    with pytest.raises(lc.LicenceError, match="ended"):
        lc.install_licence(k, p, DEV, now=ts("2028-01-01"), public_key=PUB)
    assert not p.exists() and list(tmp_path.iterdir()) == []  # nothing written, no temp files left
    lic = lc.install_licence(k, p, "00000000A1B2C3D4", now=ts("2027-01-01"), public_key=PUB)
    assert lic.customer == "Acme Logistics"
    assert lc.read_licence(p, PUB) == lic  # saved as an ordinary licence file
    assert json.loads(p.read_text())["licence"]["devices"] == [DEV, "ffff"]
    lc.install_licence(make_licence(expires=None), p, DEV, now=ts("2027-01-01"), public_key=PUB)  # JSON too
    assert lc.read_licence(p, PUB).expires is None


def test_waiting_engine_starts_at_once_when_the_licence_arrives(tmp_path):
    path = tmp_path / "licence.json"
    sleeps: list[float] = []

    def sleep(s: float) -> None:
        sleeps.append(s)
        if len(sleeps) == 3:  # activated on the dashboard 3 s after the engine began waiting
            lc.install_licence(key_of(make_licence(expires=None)), path, DEV, public_key=PUB)

    g = lc.wait_for_licence(
        path, None, stop=lambda: False, sleep=sleep, retry_s=60, public_key=PUB, device_id=DEV
    )
    assert g is not None and len(sleeps) <= 4


def test_engine_activate_command(tmp_path, monkeypatch, capsys):
    from anpr import engine

    monkeypatch.setattr(lc, "PUBLIC_KEY_HEX", PUB.hex())
    monkeypatch.setattr(lc, "machine_id", lambda: DEV)
    cfg = _cfg(tmp_path)
    assert (
        engine.main(
            ["--config", str(cfg), "--activate", key_of(make_licence(devices=["ffff"], expires=None))]
        )
        == 4
    )
    assert "NOT activated" in capsys.readouterr().out and not (tmp_path / "licence.json").exists()
    assert engine.main(["--config", str(cfg), "--activate", key_of(make_licence(expires=None))]) == 0
    assert "activated: licence LIC-TEST-0001 for Acme Logistics" in capsys.readouterr().out
    assert engine.main(["--config", str(cfg), "--check-licence"]) == 0
    src = tmp_path / "from_vendor.json"
    src.write_text(make_licence(expires=None))
    assert engine.main(["--config", str(cfg), "--activate", str(src)]) == 0


# ---- licence tool library --------------------------------------------------------------------------


def test_parse_device_text():
    text = "# Acme\n00000000A1B2C3D4  # gate 1\nb5c6d7e8, 9f9f; abcd\n\na1b2c3d4\n"
    assert licence_tool.parse_device_text(text) == ["a1b2c3d4", "b5c6d7e8", "9f9f", "abcd"]
    with pytest.raises(licence_tool.IssueError, match="not a valid device ID"):
        licence_tool.parse_device_text("ok1\nbad/id")


def test_issue_licence_and_list(tmp_path, monkeypatch):
    key = tmp_path / "vendor_key.json"
    licence_tool.main(["--key", str(key), "keygen"])
    monkeypatch.setattr(lc, "PUBLIC_KEY_HEX", json.loads(key.read_text())["public_key"])
    r = licence_tool.issue_licence(key, " Acme ", ["00000000A1B2C3D4", "a1b2c3d4", "ffff"], None, note="PO 1")
    assert r["body"]["customer"] == "Acme" and r["body"]["devices"] == ["a1b2c3d4", "ffff"]
    assert lc.parse_licence(r["key"], lc.vendor_public_key()).devices == {"a1b2c3d4", "ffff"}
    assert r["record"].stat().st_mode & 0o077 == 0
    (licence_tool.issued_dir(key) / "broken.json").write_text("{")
    listed = licence_tool.list_issued(key)
    good = [x for x in listed if x["valid"]]
    assert [x["body"]["licence_id"] for x in good] == [r["body"]["licence_id"]] and good[0]["key"] == r["key"]
    assert any(not x["valid"] for x in listed)
    with pytest.raises(licence_tool.IssueError, match="no machine IDs"):
        licence_tool.issue_licence(key, "Acme", [], None)
    with pytest.raises(licence_tool.IssueError, match="customer name is empty"):
        licence_tool.issue_licence(key, "  ", ["ab"], None)


def test_cli_issue_prints_the_key(tmp_path, monkeypatch, capsys):
    key = tmp_path / "vendor_key.json"
    licence_tool.main(["--key", str(key), "keygen"])
    monkeypatch.setattr(lc, "PUBLIC_KEY_HEX", json.loads(key.read_text())["public_key"])
    capsys.readouterr()
    licence_tool.main(
        [
            "--key",
            str(key),
            "issue",
            "--customer",
            "Acme",
            "--devices",
            "ab",
            "--no-expiry",
            "--out",
            str(tmp_path / "l.json"),
        ]
    )
    k = capsys.readouterr().out.strip().splitlines()[-1]
    assert lc.parse_licence(k, lc.vendor_public_key()).customer == "Acme"


def test_cli_per_machine_writes_a_csv_of_keys(tmp_path, monkeypatch, capsys):
    import csv
    import io

    key = tmp_path / "vendor_key.json"
    licence_tool.main(["--key", str(key), "keygen"])
    monkeypatch.setattr(lc, "PUBLIC_KEY_HEX", json.loads(key.read_text())["public_key"])
    serials = tmp_path / "serials.txt"
    serials.write_text("00000000a1b2c3d4  # Gate 1\n00000000b5c6d7e8  # Gate 2\n")
    out = tmp_path / "keys.csv"
    assert (
        licence_tool.main(
            [
                "--key",
                str(key),
                "issue",
                "--customer",
                "Acme",
                "--devices-file",
                str(serials),
                "--days",
                "30",
                "--per-machine",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert "2 licence keys (one per machine)" in capsys.readouterr().out
    rows = list(csv.reader(io.StringIO(out.read_text(encoding="utf-8-sig"))))
    assert rows[0] == licence_tool.CSV_HEADER
    assert [(r[1], r[2]) for r in rows[1:]] == [("a1b2c3d4", "Gate 1"), ("b5c6d7e8", "Gate 2")]
    for r in rows[1:]:
        assert lc.parse_licence(r[5], lc.vendor_public_key()).devices == {r[1]}
    with pytest.raises(SystemExit, match="licence-id cannot"):
        licence_tool.main(
            [
                "--key",
                str(key),
                "issue",
                "--customer",
                "A",
                "--devices",
                "ab",
                "--days",
                "3",
                "--per-machine",
                "--licence-id",
                "X1",
                "--out",
                str(tmp_path / "k2.csv"),
            ]
        )


def test_renewal_is_picked_up_within_seconds_not_at_the_hourly_check(tmp_path):
    import os

    store, clock = MemStore(), Clock(ts("2027-09-30", 20))
    g = guard_for(tmp_path, clock, store, recheck_s=3600)
    clock.m = 2.0
    assert not g.expired() and g.licence.licence_id == "LIC-TEST-0001"  # nothing changed: cheap
    renewed = json.loads(make_licence(expires="2028-09-30"))
    renewed["licence"]["licence_id"] = "LIC-TEST-0002"
    renewed["signature"] = base64.b64encode(ed.sign(SEED, lc.signed_bytes(renewed["licence"]))).decode()
    (tmp_path / "licence.json").write_text(json.dumps(renewed))
    os.utime(tmp_path / "licence.json", ns=(2, 2))
    clock.m = 8.0  # seconds later, long before the hourly check
    assert not g.expired()
    assert g.licence.licence_id == "LIC-TEST-0002" and store.d[lc.STATUS_KEY]["licence_id"] == "LIC-TEST-0002"


def test_cli_per_machine_default_name_xlsx_and_no_orphans(tmp_path, monkeypatch, capsys):
    import zipfile

    key = tmp_path / "vendor_key.json"
    licence_tool.main(["--key", str(key), "keygen"])
    monkeypatch.setattr(lc, "PUBLIC_KEY_HEX", json.loads(key.read_text())["public_key"])
    monkeypatch.chdir(tmp_path)
    base = [
        "--key",
        str(key),
        "issue",
        "--customer",
        "Acme",
        "--devices",
        "a1,b2",
        "--days",
        "3",
        "--per-machine",
    ]
    assert licence_tool.main(base) == 0
    assert (tmp_path / "licence-keys.csv").exists() and not (tmp_path / "licence.json").exists()
    assert licence_tool.main([*base, "--out", "keys.xlsx"]) == 0
    assert "xl/worksheets/sheet1.xml" in zipfile.ZipFile(tmp_path / "keys.xlsx").namelist()
    before = sorted(licence_tool.issued_dir(key).glob("*.json"))
    with pytest.raises(SystemExit, match="cannot write"):
        licence_tool.main([*base, "--out", str(tmp_path / "nosuch" / "k.csv")])
    assert sorted(licence_tool.issued_dir(key).glob("*.json")) == before  # nothing half-issued
