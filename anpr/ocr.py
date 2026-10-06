"""Plate OCR: fast-plate-ocr CCT models run directly with ONNX Runtime.

The `fast-plate-ocr` package is NOT a runtime dependency (it's only used on the dev machine to
download / train models). This module re-implements its tiny inference path — read the plate config
YAML, resize the crop, feed uint8 NHWC, argmax per slot — so the Pi needs only onnxruntime + OpenCV.

Model output: (batch, max_plate_slots, len(alphabet)) softmax probabilities, optionally with a second
"region" head (v2 models), which we ignore.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from anpr.config import OcrConfig
from anpr.types import OcrResult

log = logging.getLogger(__name__)

_INTERPOLATION = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos4": cv2.INTER_LANCZOS4,
}


@dataclass(frozen=True, slots=True)
class PlateModelConfig:
    """The subset of fast-plate-ocr's plate config we need for inference."""

    max_plate_slots: int
    alphabet: str
    pad_char: str
    img_height: int
    img_width: int
    keep_aspect_ratio: bool = False
    interpolation: str = "linear"
    image_color_mode: str = "rgb"  # "rgb" | "grayscale"
    padding_color: tuple[int, int, int] = (114, 114, 114)

    @classmethod
    def load(cls, path: Path) -> PlateModelConfig:
        if not Path(path).is_file():
            raise FileNotFoundError(f"OCR plate config not found: {path}")
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        try:
            pad = data.get("padding_color", (114, 114, 114))
            pad = (int(pad),) * 3 if isinstance(pad, int | float) else tuple(int(c) for c in pad)
            cfg = cls(
                max_plate_slots=int(data["max_plate_slots"]),
                alphabet=str(data["alphabet"]),
                pad_char=str(data["pad_char"]),
                img_height=int(data["img_height"]),
                img_width=int(data["img_width"]),
                keep_aspect_ratio=bool(data.get("keep_aspect_ratio", False)),
                interpolation=str(data.get("interpolation", "linear")),
                image_color_mode=str(data.get("image_color_mode", "rgb")),
                padding_color=pad,  # type: ignore[arg-type]
            )
        except KeyError as e:
            raise ValueError(f"{path}: missing key {e}") from e
        if cfg.pad_char not in cfg.alphabet or len(cfg.pad_char) != 1:
            raise ValueError(f"{path}: pad_char {cfg.pad_char!r} must be one character of the alphabet")
        if cfg.interpolation not in _INTERPOLATION:
            raise ValueError(f"{path}: unknown interpolation {cfg.interpolation!r}")
        if cfg.image_color_mode not in ("rgb", "grayscale"):
            raise ValueError(f"{path}: unknown image_color_mode {cfg.image_color_mode!r}")
        return cfg

    @property
    def channels(self) -> int:
        return 3 if self.image_color_mode == "rgb" else 1


def preprocess(crop: np.ndarray, m: PlateModelConfig) -> np.ndarray:
    """BGR (or gray) crop -> uint8 batch (1, H, W, C) exactly as fast-plate-ocr prepares it."""
    if crop.ndim == 3 and crop.shape[2] == 1:
        crop = crop[:, :, 0]
    if m.image_color_mode == "rgb":
        img = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB if crop.ndim == 2 else cv2.COLOR_BGR2RGB)
        border: int | tuple[int, int, int] = m.padding_color
    else:
        img = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        border = int(m.padding_color[0])
    interp = _INTERPOLATION[m.interpolation]
    if not m.keep_aspect_ratio:
        img = cv2.resize(img, (m.img_width, m.img_height), interpolation=interp)
    else:
        h, w = img.shape[:2]
        r = min(m.img_height / h, m.img_width / w)
        nw, nh = max(1, round(w * r)), max(1, round(h * r))
        img = cv2.resize(img, (nw, nh), interpolation=interp)
        dw, dh = (m.img_width - nw) / 2, (m.img_height - nh) / 2
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=border)
    if img.ndim == 2:
        img = img[:, :, None]
    return np.ascontiguousarray(img[None], dtype=np.uint8)


def decode(probs: np.ndarray, m: PlateModelConfig) -> OcrResult | None:
    """(slots, alphabet) probabilities -> text without pad chars + one confidence per kept char."""
    probs = np.asarray(probs, dtype=np.float32).reshape(m.max_plate_slots, len(m.alphabet))
    idx = probs.argmax(axis=1)
    conf = probs[np.arange(len(idx)), idx]
    chars, confs = [], []
    for i, c in zip(idx, conf, strict=True):
        ch = m.alphabet[int(i)]
        if ch == m.pad_char:
            continue
        chars.append(ch)
        confs.append(float(c))
    if not chars:
        return None
    return OcrResult(text="".join(chars), char_confs=tuple(confs))


class FastPlateOcr:
    """PlateOcr implementation (see anpr.types.PlateOcr)."""

    def __init__(self, cfg: OcrConfig) -> None:
        import onnxruntime as ort

        self.cfg = cfg
        self.model = PlateModelConfig.load(Path(cfg.config_path))
        path = Path(cfg.model_path)
        if not path.is_file():
            raise FileNotFoundError(f"OCR ONNX model not found: {path}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = cfg.num_threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")  # don't burn Pi cores
        self._sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])

        inp = self._sess.get_inputs()[0]
        self._input = inp.name
        want = [self.model.img_height, self.model.img_width, self.model.channels]
        got = list(inp.shape[1:])
        if len(got) != 3 or any(isinstance(g, int) and g != w for g, w in zip(got, want, strict=True)):
            raise ValueError(f"{path} input {inp.shape} does not match plate config (N, {want})")
        outs = [o.name for o in self._sess.get_outputs()]
        self._output = "plate" if "plate" in outs else outs[0]
        shape = next(o.shape for o in self._sess.get_outputs() if o.name == self._output)
        tail = [d for d in shape[1:] if isinstance(d, int)]
        if tail and tail != [self.model.max_plate_slots, len(self.model.alphabet)]:
            raise ValueError(
                f"{path} output {shape} does not match plate config "
                f"({self.model.max_plate_slots} slots x {len(self.model.alphabet)} chars)"
            )
        log.info(
            "ocr: %s slots=%d input=%dx%d %s threads=%d",
            path,
            self.model.max_plate_slots,
            self.model.img_width,
            self.model.img_height,
            self.model.image_color_mode,
            cfg.num_threads,
        )

    def read(self, crop: np.ndarray) -> OcrResult | None:
        if crop is None or crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
            return None
        x = preprocess(crop, self.model)
        (probs,) = self._sess.run([self._output], {self._input: x})
        return decode(probs[0], self.model)
