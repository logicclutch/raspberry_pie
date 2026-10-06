"""Dataset tools for OCR fine-tuning: harvest, labeler API, split. Fast, no real models needed."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from anpr.config import AppConfig, CropConfig, ValidationConfig
from anpr.types import Box, OcrResult
from anpr.validator import PlateValidator
from training import dataset
from training.dataset import COLUMNS, append_new_rows, read_rows, write_rows
from training.harvest import Candidate, harvest_video, select_diverse
from training.labeler.app import create_app
from training.split import assign_splits
from training.split import main as split_main

BASE = "http://127.0.0.1:8010"


def read_csv(path: Path) -> list[list[str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def read_dicts(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---- synthetic video + fakes --------------------------------------------------------------------


def make_video(path: Path, frames: int = 24, gap: tuple[int, int] | None = None) -> Path:
    """A white plate with black text sliding right. Frames in `gap` show no plate (track ends)."""
    w, h = 240, 140
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (w, h))
    assert vw.isOpened()
    for i in range(frames):
        img = np.full((h, w, 3), 40, np.uint8)
        if not (gap and gap[0] <= i < gap[1]):
            x = 10 + 3 * i
            y = 45 + (i % 3)
            cv2.rectangle(img, (x, y), (x + 120, y + 34), (255, 255, 255), -1)
            cv2.putText(
                img, f"MH12AB{1000 + i}", (x + 4, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )
        vw.write(img)
    vw.release()
    return path


class FakeDetector:
    """Finds the bright plate rectangle."""

    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame: np.ndarray) -> list[Box]:
        self.calls += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, m = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return []
        x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        return [Box(x, y, x + w, y + h, 0.9)] if w > 20 else []


class FakeOcr:
    def __init__(self, text: str = "MH12AB1234") -> None:
        self.text = text
        self.calls = 0

    def read(self, crop: np.ndarray) -> OcrResult | None:
        self.calls += 1
        return OcrResult(self.text, (0.99,) * len(self.text))


def cfg() -> AppConfig:
    return AppConfig(crop=CropConfig(min_width_px=20, min_sharpness=0.0, deshear=True))


def run_harvest(video: Path, out: Path, **kw):
    kw.setdefault("max_per_track", 3)
    kw.setdefault("min_diff", 0.5)  # synthetic frames differ only slightly (dedupe: tested below)
    return harvest_video(
        video,
        kw.pop("source", "cam1"),
        out,
        cfg(),
        FakeDetector(),
        kw.pop("ocr", FakeOcr()),
        PlateValidator(ValidationConfig(), current_year=2026),
        **kw,
    )


# ---- harvest -----------------------------------------------------------------------------------


def test_harvest_saves_diverse_crops_with_unverified_suggestions(tmp_path):
    video = make_video(tmp_path / "v.avi", frames=24, gap=(10, 30))  # plate in frames 0-9 only
    video2 = make_video(tmp_path / "w.avi", frames=12)
    out = tmp_path / "data"
    s = run_harvest(video, out)
    assert s.processed == 24 and s.tracks == 1 and s.crops == 3 and s.added == 3
    rows = read_rows(out)
    assert len(rows) == 3
    assert read_csv(out / "labels.csv")[0] == list(COLUMNS)
    frames = sorted(int(r["frame"]) for r in rows)
    assert len(set(frames)) == 3 and frames[-1] - frames[0] >= 4  # spread over the track
    for r in rows:
        assert r["status"] == "unverified" and r["plate_text"] == "" and r["labeler"] == ""
        assert r["ocr_text"] == "MH12AB1234" and r["ocr_valid"] == "1" and r["ocr_plate"] == "MH12AB1234"
        assert r["source"] == "cam1" and r["video"] == str(video.resolve())
        assert r["prep"] in ("deshear", "plain") and r["two_line"] in ("0", "1")
        assert len(r["box"].split()) == 4 and int(r["width_px"]) > 100
        for col in ("image_path", "raw_image_path"):
            p = out / r[col]
            assert p.suffix == ".png" and p.is_file() and cv2.imread(str(p)) is not None
            assert p.resolve().is_relative_to((out / "images" / "cam1").resolve())
    # a second video in the same source: its own folder, its own rows
    s2 = run_harvest(video2, out, every=2)
    assert s2.processed == 6 and s2.added == 3
    assert len({Path(r["image_path"]).parent for r in read_rows(out)}) == 2


def test_harvest_is_idempotent_and_never_overwrites_labels(tmp_path):
    video = make_video(tmp_path / "v.avi", frames=16)
    out = tmp_path / "data"
    run_harvest(video, out)
    rows = read_rows(out)
    rows[0].update(
        plate_text="MH12AB9999", status="verified", labeler="ann", labeled_at="2026-09-29T10:00:00"
    )
    rows[1].update(status="unreadable", labeler="ann")
    write_rows(out, rows)
    first = (out / rows[0]["image_path"]).stat().st_mtime_ns

    ocr = FakeOcr("DIFFERENT1")
    s = run_harvest(video, out, ocr=ocr)
    assert s.added == 0 and s.existing == 3 and ocr.calls == 0  # nothing re-read, nothing rewritten
    again = read_rows(out)
    assert again == rows
    assert (out / rows[0]["image_path"]).stat().st_mtime_ns == first

    s = run_harvest(video, out, max_per_track=6)  # more crops per track: only new ones are added
    after = read_rows(out)
    assert s.added == len(after) - 3 > 0
    assert after[0]["plate_text"] == "MH12AB9999" and after[0]["status"] == "verified"
    assert after[1]["status"] == "unreadable"
    assert len({dataset.row_key(r) for r in after}) == len(after)


def test_append_new_rows_skips_existing_keys(tmp_path):
    base = {"source": "s", "video": "/v.mp4", "frame": "1", "track": "2", "status": "unverified"}
    assert append_new_rows(tmp_path, [base]) == (1, 0)
    assert append_new_rows(tmp_path, [{**base, "ocr_text": "X"}, {**base, "frame": "3"}]) == (1, 1)
    assert [r["frame"] for r in read_rows(tmp_path)] == ["1", "3"]
    assert not list(tmp_path.glob(".labels.*"))  # no temp files left behind


def _cand(frame: int, sharp: float, value: int) -> Candidate:
    img = np.full((20, 40), value, np.uint8)
    from training.harvest import thumb

    return Candidate(frame, Box(0, 0, 40, 20, 1.0), img, img, sharp, False, "plain", thumb(img))


def test_select_diverse_spreads_and_skips_near_duplicates():
    cands = [_cand(i, float(100 - i), 10 * i) for i in range(12)]
    kept = select_diverse(cands, 4, min_diff=6.0)
    assert [c.frame for c in kept] == [0, 3, 6, 9]  # sharpest of each quarter, in frame order
    dups = [_cand(i, float(i), 100) for i in range(8)]  # identical pixels
    assert len(select_diverse(dups, 4)) == 1
    assert select_diverse([], 4) == [] and select_diverse(cands, 0) == []


# ---- labeler ------------------------------------------------------------------------------------


def _png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((20, 60, 3), 200, np.uint8))


def make_data(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    rows = []
    for track, frames in ((1, (5, 6, 7)), (2, (20,))):
        for f in frames:
            rel = f"images/cam/v_abc/t{track:04d}_f{f:06d}.png"
            _png(data / rel)
            _png(data / rel.replace(".png", "_raw.png"))
            rows.append(
                {
                    "image_path": rel,
                    "raw_image_path": rel.replace(".png", "_raw.png"),
                    "source": "cam",
                    "video": "/videos/v.mp4",
                    "frame": str(f),
                    "track": str(track),
                    "two_line": "1" if track == 2 else "0",
                    "ocr_text": "MH12AB1234",
                    "ocr_valid": "1",
                    "ocr_plate": "MH12AB1234",
                    "status": "unverified",
                }
            )
    write_rows(data, rows)
    (data / "secret.txt").write_text("nope")
    return data


@pytest.fixture
def client(tmp_path):
    data = make_data(tmp_path)
    c = TestClient(create_app(data), base_url=BASE)
    c.data_dir = data  # type: ignore[attr-defined]
    return c


def label(c, **body):
    body.setdefault("labeler", "ann")
    return c.post("/api/label", json=body)


def test_labeler_lists_and_saves_atomically(client):
    r = client.get("/")
    assert r.status_code == 200 and "script-src 'self'" in r.headers["content-security-policy"]
    data = client.get("/api/rows").json()
    assert data["total"] == 4 and data["counts"]["unverified"] == 4 and len(data["rows"]) == 4
    assert data["sources"] == ["cam"] and data["two_line"] == 1
    first = data["rows"][0]
    assert first["suggestion"] == "MH12AB1234" and first["image_url"].startswith("/img/images/")
    assert len(client.get("/api/rows", params={"two_line": "1"}).json()["rows"]) == 1

    r = label(client, id=first["id"], status="verified", plate_text="mh 12 ab-1234")
    assert r.status_code == 200, r.text
    row = r.json()["row"]
    assert row["plate_text"] == "MH12AB1234" and row["status"] == "verified" and row["labeler"] == "ann"
    assert row["labeled_at"] and r.json()["counts"]["verified"] == 1
    saved = read_rows(client.data_dir)
    assert saved[0]["plate_text"] == "MH12AB1234" and saved[0]["status"] == "verified"
    assert not list(client.data_dir.glob(".labels.*"))
    assert len(client.get("/api/rows").json()["rows"]) == 3  # unverified only

    # another tab (or the harvester) changed a different row meanwhile: that change survives
    rows = read_rows(client.data_dir)
    rows[3].update(status="skip", labeler="bob")
    write_rows(client.data_dir, rows)
    assert label(client, id=rows[1]["image_path"], status="unreadable").status_code == 200
    final = read_rows(client.data_dir)
    assert final[3]["status"] == "skip" and final[3]["labeler"] == "bob"
    assert final[1]["status"] == "unreadable" and final[1]["plate_text"] == ""
    assert final[0]["plate_text"] == "MH12AB1234"

    # undo (a human decision can only be changed with the rev the client saw)
    rev = next(
        x
        for x in client.get("/api/rows", params={"status": "all"}).json()["rows"]
        if x["id"] == rows[1]["image_path"]
    )["rev"]
    r = label(client, id=rows[1]["image_path"], status="unverified", rev=rev)
    assert r.json()["row"]["status"] == "unverified" and r.json()["row"]["labeler"] == ""


def test_labeler_refuses_stale_saves_over_a_human_label(client):
    # tab A and tab B both load the same unverified crop
    row = client.get("/api/rows").json()["rows"][0]
    # tab B verifies it
    r = label(client, id=row["id"], status="verified", plate_text="MH12AB1234", rev=row["rev"], labeler="bob")
    assert r.status_code == 200
    new_rev = r.json()["row"]["rev"]
    assert new_rev != row["rev"]
    # tab A (stale) tries to mark it skip / change the text / apply to the track: refused, label kept
    for path, body in (
        ("/api/label", {"id": row["id"], "status": "skip", "rev": row["rev"]}),
        ("/api/label", {"id": row["id"], "status": "verified", "plate_text": "MH12AB1284"}),
        ("/api/label-track", {"id": row["id"], "plate_text": "MH12AB1284", "rev": row["rev"]}),
    ):
        r = client.post(path, json={"labeler": "ann", **body})
        assert r.status_code == 409 and r.json()["error"]["code"] == "conflict", r.text
        assert r.json()["error"]["row"]["plate_text"] == "MH12AB1234"
    saved = read_rows(client.data_dir)[0]
    assert (saved["plate_text"], saved["labeler"]) == ("MH12AB1234", "bob")
    # with the current rev the edit is allowed
    r = label(client, id=row["id"], status="verified", plate_text="MH12AB1284", rev=new_rev)
    assert r.status_code == 200 and read_rows(client.data_dir)[0]["plate_text"] == "MH12AB1284"


def test_labels_csv_round_trips_awkward_paths_and_keeps_its_mode(tmp_path):
    data = tmp_path / "d"
    row = {c: "" for c in COLUMNS} | {
        "image_path": "images/cam/v/t0001_f000001.png",
        "source": "cam",
        "video": '/Videos/Anpr 2/rtsp___1.2.3.4_554_avstream_channel=1,stream=1 "x" é.sdp-.mp4',
        "frame": "1",
        "track": "1",
        "ocr_text": "=HYPERLINK(1)",
    }
    write_rows(data, [row])
    assert (data / "labels.csv").stat().st_mode & 0o777 == 0o644
    (data / "labels.csv").chmod(0o640)
    assert read_rows(data) == [row]
    write_rows(data, read_rows(data))
    assert (data / "labels.csv").stat().st_mode & 0o777 == 0o640
    assert read_rows(data) == [row]


def test_labeler_requires_confirm_for_non_standard_and_a_name(client):
    rid = client.get("/api/rows").json()["rows"][0]["id"]
    r = label(client, id=rid, status="verified", plate_text="ABC123")
    assert r.status_code == 409 and r.json()["error"]["code"] == "needs_confirm"
    assert read_rows(client.data_dir)[0]["status"] == "unverified"
    r = label(client, id=rid, status="verified", plate_text="ABC123", confirm=True)
    assert r.status_code == 200 and r.json()["row"]["plate_text"] == "ABC123"
    assert label(client, id=rid, status="verified", plate_text="MH12AB1234", labeler="").status_code == 422
    assert label(client, id=rid, status="verified", plate_text="=-+", labeler="x").status_code == 422
    assert label(client, id=rid, status="verified", plate_text="", labeler="x").status_code == 422
    assert (
        label(client, id=rid, status="verified", plate_text="MH12AB1234", labeler="=HYPERLINK").status_code
        == 422
    )
    assert label(client, id="images/nope.png", status="skip").status_code == 404
    assert label(client, id=rid, status="bogus").status_code == 422


def test_labeler_validate_endpoint(client):
    v = client.get("/api/validate", params={"text": "mh12ab1234"}).json()
    assert v["valid"] and v["kind"] == "standard" and v["display"] == "MH 12 AB 1234"
    assert client.get("/api/validate", params={"text": "22BH1234AB"}).json()["valid"]
    assert client.get("/api/validate", params={"text": "DL3CAB1234"}).json()["valid"]
    v = client.get("/api/validate", params={"text": "MH12AB12O4"}).json()
    assert not v["valid"] and v["suggestion"] == "MH12AB1204"
    v = client.get("/api/validate", params={"text": "XX12AB1234"}).json()
    assert not v["valid"] and "suggestion" not in v


def test_labeler_track_apply_only_touches_unverified_crops_of_that_track(client):
    rows = client.get("/api/rows").json()["rows"]
    t1 = [r for r in rows if r["track"] == "1"]
    label(client, id=t1[1]["id"], status="unreadable")
    r = client.post(
        "/api/label-track", json={"id": t1[0]["id"], "plate_text": "HR26DK8337", "labeler": "ann"}
    )
    assert r.status_code == 200 and r.json()["updated"] == 2
    by_frame = {x["frame"]: x for x in read_rows(client.data_dir)}
    assert by_frame["5"]["plate_text"] == "HR26DK8337" and by_frame["7"]["plate_text"] == "HR26DK8337"
    assert by_frame["6"]["status"] == "unreadable"  # a human decision is not overwritten
    assert by_frame["20"]["status"] == "unverified"  # other track untouched
    tr = client.get("/api/track", params={"id": t1[0]["id"]}).json()["rows"]
    assert [x["frame"] for x in tr] == ["5", "6", "7"]
    r = client.post("/api/label-track", json={"id": t1[0]["id"], "plate_text": "AB1", "labeler": "ann"})
    assert r.status_code == 409


def test_labeler_image_serving_blocks_traversal(client):
    r = client.get("/img/images/cam/v_abc/t0001_f000005.png")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    for bad in (
        "/img/labels.csv",
        "/img/secret.txt",
        "/img/images/../secret.txt",
        "/img/images/%2e%2e/secret.txt",
        "/img/images/..%2fsecret.txt",
        "/img/%2Fetc%2Fpasswd",
        "/img//etc/passwd",
        "/img/images/cam/v_abc/missing.png",
        "/img/images/cam/v_abc/t0001_f000005.png%00.txt",
    ):
        assert client.get(bad).status_code == 404, bad


def test_labeler_is_localhost_only_and_same_origin(client):
    assert client.get("/api/summary", headers={"host": "evil.example"}).status_code == 400
    rid = client.get("/api/rows").json()["rows"][0]["id"]
    body = {"id": rid, "status": "skip", "labeler": "ann"}
    r = client.post("/api/label", json=body, headers={"origin": "http://evil.example"})
    assert r.status_code == 403
    r = client.post("/api/label", content='{"id":"x"}', headers={"content-type": "text/plain"})
    assert r.status_code == 415
    assert client.post("/api/label", json=body, headers={"origin": BASE}).status_code == 200


# ---- split --------------------------------------------------------------------------------------


def split_rows() -> list[dict[str, str]]:
    rows = []
    plates = [f"MH12AB{i:04d}" for i in range(20)]
    for t in range(20):
        src = "phone" if t >= 16 else "gate"
        for f in range(3):
            rows.append(
                {c: "" for c in COLUMNS}
                | {
                    "image_path": f"images/{src}/v/t{t}_f{f}.png",
                    "raw_image_path": f"images/{src}/v/t{t}_f{f}_raw.png",
                    "source": src,
                    "video": f"/v/{src}.mp4",
                    "frame": str(f),
                    "track": str(t),
                    "two_line": "1" if t % 4 == 0 else "0",
                    "plate_text": plates[t],
                    "status": "verified",
                }
            )
    rows[0]["status"] = "unreadable"
    rows[0]["plate_text"] = ""
    rows[3]["status"] = "skip"
    rows[6]["status"] = "unverified"
    rows[6]["plate_text"] = ""
    return rows


def test_split_groups_by_track_and_is_deterministic():
    rows = split_rows()
    s1, _ = assign_splits(rows, seed=1)
    assert s1 == assign_splits(rows, seed=1)[0]
    assert {0, 3, 6}.isdisjoint(s1)  # unreadable / skip / unverified are left out
    per_track: dict[str, set[str]] = {}
    for i, sp in s1.items():
        per_track.setdefault(rows[i]["track"], set()).add(sp)
    assert all(len(v) == 1 for v in per_track.values())
    assert set(s1.values()) == {"train", "val", "test"}


def test_split_merges_same_plate_across_tracks_and_test_by_source():
    rows = split_rows()
    for r in rows:  # the gate vehicle of track 5 comes back as track 17 in the phone video
        if r["track"] == "17":
            r["plate_text"] = rows[15]["plate_text"]
    s, notes = assign_splits(rows, test_sources=("phone",), seed=3)
    for i, sp in s.items():
        assert (sp == "test") == (rows[i]["source"] == "phone")
    assert not any(rows[i]["track"] == "5" for i in s)  # same plate as a test vehicle: left out
    assert notes["dropped_overlap_rows"] == 3
    s2, _ = assign_splits(rows, seed=3)
    assert len({s2[i] for i, r in enumerate(rows) if r["track"] in ("5", "17") and i in s2}) == 1


def test_split_matches_test_video_by_file_name_and_holds_out_truth_plates():
    rows = split_rows()
    # the project moved (other machine / Colab): same file name, different absolute path
    s, notes = assign_splits(
        rows, test_videos=("/content/drive/repo/phone.mp4",), holdout_plates={rows[15]["plate_text"]}, seed=3
    )
    for i, sp in s.items():
        assert (sp == "test") == (rows[i]["source"] == "phone")
    assert not any(rows[i]["track"] == "5" for i in s)  # a ground-truth plate never goes to train/val
    assert notes["dropped_overlap_rows"] == 3


def test_split_keeps_synthetic_rows_in_train_and_drops_too_long_labels():
    rows = split_rows()
    for t in range(4):  # 4 synthetic "vehicles"
        rows.append(
            {c: "" for c in COLUMNS}
            | {
                "image_path": f"images/synth/seed1/s{t}.png",
                "source": "synth",
                "video": "synth:seed1",
                "frame": str(t),
                "track": str(t),
                "plate_text": f"KA01AB{t:04d}",
                "status": "verified",
            }
        )
    rows[9]["plate_text"] = "DL10CAB1234"  # 11 characters: more than the model's 10 slots
    for seed in range(5):
        s, notes = assign_splits(rows, seed=seed)
        assert all(sp == "train" for i, sp in s.items() if rows[i]["source"] == "synth")
        assert 9 not in s and notes["too_long_rows"] == 1
        s, _ = assign_splits(rows, test_sources=("phone",), seed=seed)
        assert all(sp == "train" for i, sp in s.items() if rows[i]["source"] == "synth")


def test_split_cli_truth_file(tmp_path, capsys):
    data = tmp_path / "data"
    write_rows(data, split_rows())
    truth = tmp_path / "gt" / "phone.csv"
    truth.parent.mkdir()
    truth.write_text(
        "# comment\nvideo,vehicle,plate,status,first_frame,last_frame,tracks,note\n"
        "/elsewhere/phone.mp4,1,MH12AB0003,unsure,,,,\n",
        encoding="utf-8",
    )
    assert split_main(["--data", str(data), "--truth", str(truth), "--out", str(tmp_path / "out")]) == 0
    capsys.readouterr()
    saved = read_rows(data)
    assert {r["split"] for r in saved if r["source"] == "phone" and r["status"] == "verified"} == {"test"}
    assert {r["split"] for r in saved if r["track"] == "3"} == {""}  # its plate is a ground-truth plate


def test_split_cli_writes_trainer_csvs(tmp_path, capsys):
    data = tmp_path / "data"
    write_rows(data, split_rows())
    assert (
        split_main(["--data", str(data), "--test-video", "/v/phone.mp4", "--out", str(tmp_path / "out")]) == 0
    )
    text = capsys.readouterr().out
    assert "train" in text and "2-line" in text
    out = tmp_path / "out"
    test_rows = read_dicts(out / "test.csv")
    assert len(test_rows) == 12 and all(
        r["image_path"].startswith("../data/images/phone/") for r in test_rows
    )
    train = read_dicts(out / "train.csv")
    assert train and list(train[0]) == ["image_path", "plate_text"]
    saved = read_rows(data)
    assert {r["split"] for r in saved if r["source"] == "phone"} == {"test"}
    assert saved[0]["split"] == "" and saved[0]["status"] == "unreadable"
