"""Motion gate on synthetic frames."""

from __future__ import annotations

import numpy as np

from anpr.config import MotionConfig
from anpr.motion import MotionGate


def scene(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(60, 120, size=(9, 16, 3), dtype=np.uint8)
    return np.repeat(np.repeat(base, 80, axis=0), 80, axis=1)  # 720x1280, blocky static scene


def test_first_frame_is_motion_then_static_is_not() -> None:
    gate = MotionGate(MotionConfig())
    f = scene()
    assert gate.update(f) is True
    assert gate.update(f.copy()) is False
    assert gate.last_ratio == 0.0


def test_sensor_noise_is_not_motion() -> None:
    gate = MotionGate(MotionConfig())
    f = scene()
    gate.update(f)
    rng = np.random.default_rng(1)
    noisy = np.clip(f.astype(np.int16) + rng.integers(-8, 9, size=f.shape), 0, 255).astype(np.uint8)
    assert gate.update(noisy) is False


def test_moving_object_is_motion() -> None:
    gate = MotionGate(MotionConfig())
    f = scene()
    gate.update(f)
    g = f.copy()
    g[300:420, 500:700] = 250  # a bright "car" appears (~2.6 % of the frame)
    assert gate.update(g) is True
    assert gate.update(g.copy()) is False  # it stopped


def test_small_change_below_ratio_ignored() -> None:
    gate = MotionGate(MotionConfig(min_changed_ratio=0.05))
    f = scene()
    gate.update(f)
    g = f.copy()
    g[300:420, 500:700] = 250
    assert gate.update(g) is False


def test_disabled_always_true_and_size_change_resets() -> None:
    assert MotionGate(MotionConfig(enabled=False)).update(scene()) is True
    gate = MotionGate(MotionConfig())
    gate.update(scene())
    small = np.zeros((240, 320, 3), np.uint8)
    assert gate.update(small) is True
    assert gate.update(small) is False


def test_grayscale_input() -> None:
    gate = MotionGate(MotionConfig())
    gray = scene()[:, :, 0].copy()
    assert gate.update(gray) is True
    assert gate.update(gray) is False
