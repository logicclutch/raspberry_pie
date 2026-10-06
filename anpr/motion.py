"""Motion gate (WORKFLOW.md step 2): cheap frame difference on a tiny grayscale copy.

The detector is the slowest stage; when nothing in the scene changes we skip it and the Pi stays cool.
"""

from __future__ import annotations

import cv2
import numpy as np

from anpr.config import MotionConfig


class MotionGate:
    """`update(frame) -> bool`: True when enough pixels changed since the previous frame.

    The frame is shrunk to `downscale_width` px wide (aspect kept, e.g. 1280x720 -> 160x90) and turned
    to gray; a small blur removes sensor noise, then the absolute difference against the previous small
    frame is thresholded at `pixel_threshold` and motion = changed-pixel ratio >= `min_changed_ratio`.
    The first frame (or a frame whose size changed) always counts as motion. Disabled -> always True.
    """

    def __init__(self, cfg: MotionConfig) -> None:
        self.cfg = cfg
        self._prev: np.ndarray | None = None
        self._diff: np.ndarray | None = None  # reused buffer
        self.last_ratio = 1.0  # changed-pixel ratio of the last update (for debugging/tuning)

    def reset(self) -> None:
        self._prev = None
        self._diff = None

    def _small_gray(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        sw = min(self.cfg.downscale_width, w)
        sh = max(1, round(h * sw / w))
        # Resize first (INTER_AREA averages -> also denoises), then convert only the tiny image.
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        if small.ndim == 3:
            small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(small, (3, 3), 0)

    def update(self, frame: np.ndarray) -> bool:
        if not self.cfg.enabled:
            return True
        gray = self._small_gray(frame)
        prev = self._prev
        self._prev = gray
        if prev is None or prev.shape != gray.shape:
            self._diff = None
            self.last_ratio = 1.0
            return True
        self._diff = cv2.absdiff(gray, prev, dst=self._diff)
        changed = int(np.count_nonzero(self._diff > self.cfg.pixel_threshold))
        self.last_ratio = changed / self._diff.size
        return self.last_ratio >= self.cfg.min_changed_ratio
