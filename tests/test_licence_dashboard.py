"""Licence Manager (vendor dashboard) and the device's Activate page."""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from anpr import _ed25519 as ed
from anpr import licence as lc
from anpr.config import AppConfig
from anpr.storage import SqliteEventStore
from anpr.web.app import create_app

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import licence_dashboard as ld  # noqa: E402
import licence_tool  # noqa: E402

TOKEN = "t0ken-for-tests"
TODAY = dt.date(2026, 10, 1)


@pytest.fixture
def vendor(tmp_path, monkeypatch):
    key = tmp_path / "keys" / "vendor_key.json"
    assert licence_tool.main(["--key", str(key), "keygen"]) == 0
    monkeypatch.setattr(lc, "PUBLIC_KEY_HEX", json.loads(key.read_text())["public_key"])
    app = ld.create_app(key, TOKEN, today=lambda: TODAY)
    with TestClient(app, base_url="http://127.0.0.1:8090", headers={"X-Token": TOKEN}) as c:
        yield c, key


def issue(c, **kw):
    body = {
        "customer": "Acme Logistics",
        "machines": "00000000a1b2c3d4  # gate 1\nb5c6d7e8",
        "validity": "1y",
    }
    body.update(kw)
    return c.post("/api/licences", json=body)


def test_page_and_assets_load_without_token(vendor):
    c, _ = vendor
    r = c.get("/", headers={"X-Token": ""})
    assert r.status_code == 200 and "Licence Manager" in r.text
    assert "default-src 'self'" in r.headers["content-security-policy"]
    assert c.get("/static/app.js").status_code == 200 and c.get("/logo.png").status_code == 200
    assert c.get("/static/../licence_tool.py").status_code == 404


def test_api_needs_the_token_and_this_computer(vendor):
    c, _ = vendor
    assert c.get("/api/info", headers={"X-Token": "wrong"}).status_code == 401
    assert c.get("/api/info", headers={"Host": "evil.example.com"}).status_code == 403  # DNS rebinding
    r = c.post("/api/licences", json={}, headers={"Origin": "https://evil.example.com"})
    assert r.status_code == 403
    info = c.get("/api/info").json()
    assert info["key_ok"] and info["key_matches_software"] and info["today"] == "2026-10-01"


def test_issue_list_download(vendor):
    c, key = vendor
    r = issue(c, note="PO 4512")
    assert r.status_code == 200, r.text
    lic = r.json()["licence"]
    assert lic["customer"] == "Acme Logistics" and lic["devices"] == ["a1b2c3d4", "b5c6d7e8"]
    assert lic["expires"] == "2027-09-30" and lic["status"] == "active" and lic["note"] == "PO 4512"
    parsed = lc.parse_licence(lic["key"], lc.vendor_public_key())
    assert parsed.devices == {"a1b2c3d4", "b5c6d7e8"}
    listed = c.get("/api/licences").json()["licences"]
    assert [x["licence_id"] for x in listed] == [lic["licence_id"]] and listed[0]["key"] == lic["key"]
    f = c.get(f"/api/licences/{lic['licence_id']}/file")
    assert f.status_code == 200 and "attachment" in f.headers["content-disposition"]
    assert lc.parse_licence(f.text, lc.vendor_public_key()) == parsed
    assert c.get("/api/licences/..%2Fvendor_key/file").status_code == 404
    assert c.get("/api/licences/NOPE/file").status_code == 404


@pytest.mark.parametrize(
    ("kw", "expires"),
    [
        ({"validity": "2y"}, "2028-09-30"),
        ({"validity": "never"}, None),
        ({"validity": "date", "expires": "2026-12-31"}, "2026-12-31"),
    ],
)
def test_validity_choices(vendor, kw, expires):
    c, _ = vendor
    assert issue(c, **kw).json()["licence"]["expires"] == expires


@pytest.mark.parametrize(
    ("kw", "msg"),
    [
        ({"customer": "  "}, "customer name"),
        ({"machines": "# only a comment"}, "no machine IDs"),
        ({"machines": "ok\nbad/id"}, "not a valid device ID"),
        ({"validity": "date"}, "last valid day"),
        ({"validity": "date", "expires": "2020-01-01"}, "before today"),
    ],
)
def test_issue_errors(vendor, kw, msg):
    c, key = vendor
    r = issue(c, **kw)
    assert r.status_code == 422 and msg in r.json()["error"]
    assert not list(licence_tool.issued_dir(key).glob("*.json"))


def test_status_soon_and_ended(vendor):
    c, key = vendor
    licence_tool.issue_licence(key, "Soon", ["ab"], dt.date(2026, 10, 20), today=dt.date(2026, 9, 1))
    licence_tool.issue_licence(key, "Gone", ["cd"], dt.date(2026, 9, 30), today=dt.date(2026, 9, 1))
    st = {x["customer"]: (x["status"], x["days_left"]) for x in c.get("/api/licences").json()["licences"]}
    assert st == {"Soon": ("expiring", 20), "Gone": ("ended", 0)}


def test_machine_count_and_check(vendor):
    c, _ = vendor
    assert c.post("/api/machines", json={"machines": "a1\nA1\n00b2"}).json() == {
        "ok": True,
        "count": 2,
        "labelled": 0,
        "error": None,
    }
    assert c.post("/api/machines", json={"machines": "bad/id"}).json()["ok"] is False
    k = issue(c, validity="never").json()["licence"]["key"]
    r = c.post("/api/check", json={"key": k, "machine_id": "00000000A1B2C3D4"}).json()
    assert r["valid"] and r["machine"]["ok"] and r["customer"] == "Acme Logistics"
    r = c.post("/api/check", json={"key": k, "machine_id": "ffff"}).json()
    assert r["valid"] and not r["machine"]["ok"] and "not for this device" in r["machine"]["message"]
    assert c.post("/api/check", json={"key": k[:-8]}).json()["valid"] is False


def test_wrong_signing_key_blocks_issuing(tmp_path, monkeypatch):
    key = tmp_path / "vendor_key.json"
    licence_tool.main(["--key", str(key), "keygen"])
    monkeypatch.setattr(lc, "PUBLIC_KEY_HEX", "11" * 32)
    with TestClient(ld.create_app(key, TOKEN), base_url="http://127.0.0.1", headers={"X-Token": TOKEN}) as c:
        assert c.get("/api/info").json()["key_matches_software"] is False
        r = issue(c)
        assert r.status_code == 422 and "does not match" in r.json()["error"]


def test_add_years():
    assert ld.add_years(dt.date(2026, 10, 1), 1) == dt.date(2027, 9, 30)
    assert ld.add_years(dt.date(2028, 2, 29), 1) == dt.date(2029, 2, 27)


# ---- device side: Activate on the ANPR dashboard ----------------------------------------------------

SEED = bytes(range(32))
PUB = ed.public_key(SEED)


def signed_key(devices, expires=None) -> str:
    body = {
        "format": 1,
        "licence_id": "LIC-X",
        "customer": "Acme",
        "issued": "2026-01-01",
        "expires": expires,
        "devices": devices,
        "note": "",
    }
    import base64

    sig = base64.b64encode(ed.sign(SEED, lc.signed_bytes(body))).decode()
    return lc.encode_key({"licence": body, "signature": sig})


ADMIN = "admin-pass-123"


@pytest.fixture
def device(tmp_path, monkeypatch):
    monkeypatch.setattr(lc, "PUBLIC_KEY_HEX", PUB.hex())
    store = SqliteEventStore(tmp_path / "anpr.db", tmp_path / "img")

    def client(
        read_only=False, licence_path=tmp_path / "licence.json", admin=ADMIN, send_admin=True, local=False
    ):
        cfg = AppConfig.model_validate(
            {
                "storage": {"db_path": str(tmp_path / "anpr.db"), "image_dir": str(tmp_path / "img")},
                "web": {"read_only": read_only, "admin_token": admin},
            }
        )
        app = create_app(cfg, store, licence_path=licence_path, machine_id=lambda: "a1b2c3d4")
        headers = {"X-Admin-Token": admin} if admin and send_admin else {}
        kw = {"client": ("127.0.0.1", 50000)} if local else {}
        return TestClient(app, headers=headers, **kw)

    yield client, store, tmp_path
    store.close()


def test_activation_page_shows_machine_id_then_activates(device):
    client, store, tmp = device
    with client() as c:
        d = c.get("/api/v1/licence").json()["data"]
        assert d == {"machine_id": "a1b2c3d4", "status": None, "can_activate": True, "admin_required": True}
        r = c.post("/api/v1/licence", json={"key": signed_key(["ffff"])})
        assert r.status_code == 422 and "not for this device" in r.json()["error"]["message"]
        assert not (tmp / "licence.json").exists()
        r = c.post("/api/v1/licence", json={"key": signed_key(["00000000A1B2C3D4", "ffff"])})
        assert r.status_code == 200
        st = r.json()["data"]["status"]
        assert st["ok"] is True and st["customer"] == "Acme" and st["licence_id"] == "LIC-X"
        assert lc.read_licence(tmp / "licence.json", PUB).devices == {"a1b2c3d4", "ffff"}
        assert c.get("/api/v1/info").json()["data"]["licence"]["ok"] is True


def test_activation_refused_on_view_only_cross_site_or_without_path(device):
    client, _, tmp = device
    k = signed_key(["a1b2c3d4"])
    with client(read_only=True) as c:
        assert c.get("/api/v1/licence").json()["data"]["can_activate"] is False
        assert c.post("/api/v1/licence", json={"key": k}).status_code == 403
    with client() as c:
        r = c.post("/api/v1/licence", json={"key": k}, headers={"Origin": "https://evil.example.com"})
        assert r.status_code == 403
    with client(licence_path=None) as c:
        assert c.get("/api/v1/licence").json()["data"]["can_activate"] is False
        assert c.post("/api/v1/licence", json={"key": k}).status_code == 403
    assert not (tmp / "licence.json").exists()


# ---- one key per machine ---------------------------------------------------------------------------


def test_per_machine_keys_each_work_only_on_their_machine(vendor):
    c, key = vendor
    r = issue(
        c, key_type="machine", machines="00000000a1b2c3d4  # Gate 1\nb5c6d7e8 # Gate 2\nc9d0e1f2\nA1B2C3D4"
    )
    assert r.status_code == 200, r.text
    b = r.json()["licence"]
    assert b["key_type"] == "machine" and b["key"] is None and b["licence_id"].startswith("LIC-20261001-")
    assert [(i["device"], i["label"]) for i in b["items"]] == [
        ("a1b2c3d4", "Gate 1"),
        ("b5c6d7e8", "Gate 2"),
        ("c9d0e1f2", ""),
    ]
    assert b["labels"] == {"a1b2c3d4": "Gate 1", "b5c6d7e8": "Gate 2"} and b["expires"] == "2027-09-30"
    pub = lc.vendor_public_key()
    for it in b["items"]:
        lic = lc.parse_licence(it["key"], pub)
        assert lic.devices == {it["device"]} and lic.customer == "Acme Logistics"
        assert it["licence_id"].startswith(b["licence_id"] + "-")
    k1 = lc.parse_licence(b["items"][0]["key"], pub)
    with pytest.raises(lc.LicenceError, match="not for this device"):
        lc.check(k1, "b5c6d7e8", dt.datetime(2027, 1, 1).timestamp())
    assert (
        len(b["items"][0]["key"])
        < len(issue(c, machines="a1\nb2\nc3\nd4\ne5").json()["licence"]["key"]) + 200
    )
    assert len(list(licence_tool.issued_dir(key).glob("*.json"))) == 4  # 3 per machine + 1 for all


def test_list_groups_a_batch_into_one_row(vendor):
    c, _ = vendor
    issue(c, customer="One for all")
    b = issue(c, customer="Per machine", key_type="machine", machines="a1\nb2\nc3").json()["licence"]
    rows = c.get("/api/licences").json()["licences"]
    by = {r["customer"]: r for r in rows}
    assert len(rows) == 2 and by["One for all"]["key_type"] == "customer"
    m = by["Per machine"]
    assert m["key_type"] == "machine" and m["licence_id"] == b["licence_id"] and len(m["items"]) == 3
    assert [i["key"] for i in m["items"]] == [i["key"] for i in b["items"]]


def test_batch_csv_for_excel(vendor):
    c, _ = vendor
    b = issue(c, customer="=HYPERLINK(evil)", key_type="machine", machines="a1 # Gate, north\nb2").json()[
        "licence"
    ]
    r = c.get(f"/api/batches/{b['licence_id']}/csv")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    assert (
        "attachment" in r.headers["content-disposition"]
        and b["licence_id"] in r.headers["content-disposition"]
    )
    import csv
    import io

    text = r.content.decode("utf-8")
    assert text.startswith("﻿")  # Excel opens UTF-8 correctly
    rows = list(csv.reader(io.StringIO(text[1:])))
    assert rows[0] == licence_tool.CSV_HEADER
    assert [row[1:3] for row in rows[1:]] == [["a1", "Gate, north"], ["b2", ""]]
    assert all(row[0] == "'=HYPERLINK(evil)" for row in rows[1:])  # no formula runs in Excel
    assert [row[5] for row in rows[1:]] == [i["key"] for i in b["items"]]
    assert c.get("/api/batches/LIC-NOPE/csv").status_code == 404
    assert c.get("/api/batches/..%2Fx/csv").status_code == 404


def test_per_machine_check_shows_label(vendor):
    c, _ = vendor
    b = issue(c, key_type="machine", machines="a1b2c3d4 # Gate 7").json()["licence"]
    r = c.post("/api/check", json={"key": b["items"][0]["key"], "machine_id": "00000000a1b2c3d4"}).json()
    assert r["valid"] and r["key_type"] == "machine" and r["label"] == "Gate 7" and r["machine"]["ok"]


def test_per_machine_is_all_or_nothing(vendor, monkeypatch):
    c, key = vendor
    real = licence_tool.os.open
    calls = {"n": 0}

    def flaky(path, *a, **k):
        if str(path).endswith(".json") and "issued" in str(path):
            calls["n"] += 1
            if calls["n"] == 3:
                raise OSError("disk full")
        return real(path, *a, **k)

    monkeypatch.setattr(licence_tool.os, "open", flaky)
    with pytest.raises(OSError):
        licence_tool.issue_per_machine(key, "Acme", [("a1", ""), ("b2", ""), ("c3", ""), ("d4", "")], None)
    monkeypatch.undo()
    assert not list(licence_tool.issued_dir(key).glob("*.json"))


def test_activate_a_per_machine_key_on_its_device(vendor, tmp_path):
    c, _ = vendor
    b = issue(c, key_type="machine", machines="a1b2c3d4\nffff").json()["licence"]
    keys = {i["device"]: i["key"] for i in b["items"]}
    store = SqliteEventStore(tmp_path / "d.db", tmp_path / "dimg")
    cfg = AppConfig.model_validate(
        {
            "storage": {"db_path": str(tmp_path / "d.db"), "image_dir": str(tmp_path / "dimg")},
            "web": {"admin_token": ADMIN},
        }
    )
    app = create_app(
        cfg, store, licence_path=tmp_path / "licence.json", machine_id=lambda: "00000000a1b2c3d4"
    )
    try:
        with TestClient(app, headers={"X-Admin-Token": ADMIN}) as d:
            r = d.post("/api/v1/licence", json={"key": keys["ffff"]})  # another machine's key
            assert r.status_code == 422 and "not for this device" in r.json()["error"]["message"]
            r = d.post("/api/v1/licence", json={"key": keys["a1b2c3d4"]})
            assert r.status_code == 200 and r.json()["data"]["status"]["ok"] is True
    finally:
        store.close()


# ---- admin password (web.admin_token) ---------------------------------------------------------------


def test_viewers_without_the_admin_password_cannot_activate(device):
    client, _, tmp = device
    k = signed_key(["a1b2c3d4"])
    with client(send_admin=False) as c:
        assert c.get("/api/v1/licence").status_code == 200  # can see the machine ID and status
        r = c.post("/api/v1/licence", json={"key": k})
        assert r.status_code == 403 and r.json()["error"]["code"] == "admin_required"
        r = c.post("/api/v1/licence", json={"key": k}, headers={"X-Admin-Token": "wrong-password"})
        assert r.status_code == 403 and r.json()["error"] == {
            "code": "admin_required",
            "message": "wrong admin password",
        }
    assert not (tmp / "licence.json").exists()
    with client() as c:
        assert c.post("/api/v1/licence", json={"key": k}).status_code == 200


def test_admin_password_also_guards_settings_and_stream(device):
    client, _, _ = device
    with client(send_admin=False) as c:
        for method, path, body in [
            ("GET", "/api/v1/push", None),
            ("PUT", "/api/v1/push", {"enabled": False}),
            ("POST", "/api/v1/push/test", {"url": "https://x.example.com"}),
            ("POST", "/api/v1/source", {"source": ""}),
        ]:
            r = c.request(method, path, json=body)
            assert r.status_code == 403 and r.json()["error"]["code"] == "admin_required", path
        assert c.get("/api/v1/info").json()["data"]["admin_required"] is True
    with client() as c:
        assert c.get("/api/v1/push").status_code == 200
        assert c.post("/api/v1/source", json={"source": ""}).status_code == 202


def test_dashboard_without_any_password_activates_only_on_the_device(device):
    client, _, _ = device
    k = signed_key(["a1b2c3d4"])
    with client(admin=None) as c:
        r = c.post("/api/v1/licence", json={"key": k})
        assert r.status_code == 403 and "on the device itself" in r.json()["error"]["message"]
        assert c.get("/api/v1/info").json()["data"]["admin_required"] is False
    with client(admin=None, local=True) as c:
        assert c.post("/api/v1/licence", json={"key": k}).status_code == 200


def test_short_admin_password_is_refused():
    with pytest.raises(ValueError):
        AppConfig.model_validate({"web": {"admin_token": "short"}})


# ---- review fixes (per-machine keys) ------------------------------------------------------------------


def test_downloads_work_for_any_customer_name(vendor):
    c, _ = vendor
    for name in ("Łódź Logistics", "東京物流", "शर्मा ट्रांसपोर्ट"):
        b = issue(c, customer=name, key_type="machine", machines="a1b2c3d4").json()["licence"]
        for ext in ("csv", "xlsx"):
            r = c.get(f"/api/batches/{b['licence_id']}/{ext}")
            assert r.status_code == 200, (name, ext)
            cd = r.headers["content-disposition"]
            assert cd.isascii() and "filename*=UTF-8''" in cd


def test_excel_file_keeps_machine_ids_as_text(vendor):
    import io
    import zipfile

    c, _ = vendor
    b = issue(c, key_type="machine", machines="1234e567  # Gate & <1>\n00012345\n0x1f").json()["licence"]
    r = c.get(f"/api/batches/{b['licence_id']}/xlsx")
    assert r.headers["content-type"].startswith("application/vnd.openxmlformats-officedocument.spreadsheetml")
    z = zipfile.ZipFile(io.BytesIO(r.content))
    assert set(z.namelist()) >= {"[Content_Types].xml", "xl/workbook.xml", "xl/worksheets/sheet1.xml"}
    sheet = z.read("xl/worksheets/sheet1.xml").decode()
    assert 't="inlineStr"' in sheet and "<v>" not in sheet  # every cell is text, none a number
    assert '<t xml:space="preserve">1234e567</t>' in sheet and "Gate &amp; &lt;1&gt;" in sheet
    assert b["items"][0]["key"] in sheet


def test_search_finds_a_per_machine_licence_id_server_data(vendor):
    c, _ = vendor
    b = issue(c, key_type="machine", machines="a1\nb2").json()["licence"]
    row = next(r for r in c.get("/api/licences").json()["licences"] if r["licence_id"] == b["licence_id"])
    assert {i["licence_id"] for i in row["items"]} == {i["licence_id"] for i in b["items"]}


def test_labels_are_kept_for_one_key_for_all(vendor):
    c, _ = vendor
    lic = issue(c, machines="a1b2c3d4  # Gate 1\nb5c6d7e8").json()["licence"]
    assert lic["labels"] == {"a1b2c3d4": "Gate 1"}
    assert lc.parse_licence(lic["key"], lc.vendor_public_key()).devices == {"a1b2c3d4", "b5c6d7e8"}
    row = c.get("/api/licences").json()["licences"][0]
    assert row["labels"] == {"a1b2c3d4": "Gate 1"}


def test_big_labelled_batch_fits_the_form_limit(vendor):
    c, _ = vendor
    text = "\n".join(f"{i:016x}  # Toll plaza lane number {i}, north side" for i in range(4000))
    assert len(text) > 200_000
    r = c.post("/api/machines", json={"machines": text}).json()
    assert r == {"ok": True, "count": 4000, "labelled": 4000, "error": None}


def test_list_is_fast_after_the_first_time(vendor):
    import time

    c, key = vendor
    licence_tool.issue_per_machine(key, "Big", [(f"{i:08x}", "") for i in range(300)], None, today=TODAY)
    assert len(c.get("/api/licences").json()["licences"][0]["items"]) == 300  # just made: already known
    licence_tool._VERIFIED.clear()  # as after restarting the Licence Manager
    t0 = time.perf_counter()
    c.get("/api/licences")
    first = time.perf_counter() - t0
    t0 = time.perf_counter()
    rows = c.get("/api/licences").json()["licences"]
    again = time.perf_counter() - t0
    assert len(rows[0]["items"]) == 300 and again < first / 3
