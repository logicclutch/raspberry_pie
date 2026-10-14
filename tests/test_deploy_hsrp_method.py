"""The .deb sets hsrp.method in the Pi's kept config.yaml (deploy/deb/postinst, deploy/build_deb.sh)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from anpr.config import load_config

ROOT = Path(__file__).resolve().parents[1]

# The hsrp section as shipped in 0.1.0-1 / 0.1.0-2: no "  method:" line. Pis installed from those
# builds keep this config.yaml on upgrade.
OLD_CONFIG = """\
# ANPR ingest config
hsrp:
  # Mark each saved plate HSRP / Non-HSRP / Not sure (hologram + "IND" left of the number).
  enabled: true
  min_marked: 2
  min_clean: 3
  unsure_as_hsrp: true  # plates that can't be judged ("not sure") are saved as HSRP
  two_line: true
push:
  url: http://example.invalid/api
"""

NEW_CONFIG = OLD_CONFIG.replace("  enabled: true\n", "  enabled: true\n  method: model\n")


def _set_hsrp_method():
    # Imported here so a missing helper fails each test on its own, not the whole file.
    from anpr.config import set_hsrp_method

    return set_hsrp_method


def _without_method(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["hsrp"].pop("method", None)
    return data


def test_set_hsrp_method_adds_missing_key(tmp_path: Path) -> None:
    conf = tmp_path / "config.yaml"
    conf.write_text(OLD_CONFIG, encoding="utf-8")
    before = _without_method(conf)

    _set_hsrp_method()(conf, "font")

    assert load_config(conf).hsrp.method == "font"
    assert _without_method(conf) == before  # other hsrp keys and push: unchanged
    assert "# Mark each saved plate HSRP" in conf.read_text(encoding="utf-8")  # comments kept


def test_set_hsrp_method_replaces_existing(tmp_path: Path) -> None:
    conf = tmp_path / "config.yaml"
    conf.write_text(NEW_CONFIG, encoding="utf-8")
    before = _without_method(conf)

    _set_hsrp_method()(conf, "font")
    assert load_config(conf).hsrp.method == "font"
    _set_hsrp_method()(conf, "model")  # a later model build switches it back

    assert load_config(conf).hsrp.method == "model"
    assert _without_method(conf) == before
    assert conf.read_text(encoding="utf-8").count("method:") == 1


def test_no_hsrp_section_appends(tmp_path: Path) -> None:
    conf = tmp_path / "config.yaml"
    conf.write_text("push:\n  url: http://example.invalid/api\n", encoding="utf-8")

    _set_hsrp_method()(conf, "font")

    assert load_config(conf).hsrp.method == "font"
    assert yaml.safe_load(conf.read_text(encoding="utf-8"))["push"] == {"url": "http://example.invalid/api"}


def test_set_hsrp_method_rejects_unknown_method(tmp_path: Path) -> None:
    conf = tmp_path / "config.yaml"
    conf.write_text(NEW_CONFIG, encoding="utf-8")

    with pytest.raises(ValueError, match="hsrp.method"):
        _set_hsrp_method()(conf, "fonts")
    assert conf.read_text(encoding="utf-8") == NEW_CONFIG  # file untouched


def test_build_deb_always_pins_hsrp_method() -> None:
    text = (ROOT / "deploy" / "build_deb.sh").read_text(encoding="utf-8")
    m = re.search(r'HSRP_METHOD="\$\{HSRP_METHOD:-([a-z]*)\}"', text)
    assert m is not None
    # Every build ships /opt/anpr/HSRP_METHOD, so a plain build after a font build switches the Pi back.
    assert m.group(1) == "model"


def test_postinst_uses_helper_and_reports_config_value() -> None:
    text = (ROOT / "deploy" / "deb" / "postinst").read_text(encoding="utf-8")
    assert "set_hsrp_method" in text
    # No sed edit of method: (it silently does nothing when the line is missing).
    assert not re.search(r"sed .*method:", text)
    # The install message shows what config.yaml says, not what the build asked for.
    lines = [ln for ln in text.splitlines() if "HSRP check:" in ln]
    assert lines
    assert all("HSRP_METHOD" not in ln for ln in lines)
