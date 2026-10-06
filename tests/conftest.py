import pytest

import anpr.storage


@pytest.fixture(autouse=True)
def _preview_on_disk(monkeypatch):
    """Keep test stores' live previews in their tmp folders, not in the Linux RAM folder
    (/dev/shm), so tests behave the same on the Mac and the Pi and leave nothing behind."""
    monkeypatch.setattr(anpr.storage, "PREVIEW_RAM_ROOT", None)
