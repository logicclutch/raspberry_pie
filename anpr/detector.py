"""Plate detector: YOLO11n (Ultralytics export) on ncnn (the Pi target) or ONNX Runtime (baseline).

Both backends share the same letterbox pre-processing and YOLO decode + NMS, so they return the
same boxes for the same frame (up to float noise). Output boxes are in FULL-FRAME pixel coordinates,
sorted by score (best first).

Ultralytics YOLOv8/11 detect head output: (1, 4 + num_classes, num_anchors), rows = cx, cy, w, h
(in letterboxed input pixels) followed by per-class scores (already sigmoid-ed, no objectness).
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from anpr.config import DetectorConfig
from anpr.types import Box, PlateDetector

log = logging.getLogger(__name__)

LETTERBOX_COLOR = (114, 114, 114)  # Ultralytics pad value
NCNN_INPUT = "in0"  # blob names written by `yolo export format=ncnn`
NCNN_OUTPUT = "out0"


def letterbox(frame: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    """Resize `frame` (BGR) to fit a size x size square, keeping aspect ratio, centred on grey.

    Returns (image uint8 BGR size x size x 3, scale, pad_x, pad_y) where
    input_px = frame_px * scale + pad.
    """
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    h, w = frame.shape[:2]
    scale = min(size / h, size / w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    if (nw, nh) != (w, h):
        # Shrinking (the usual case: 720p/1080p -> 320) must average pixels: INTER_LINEAR samples
        # and aliases, so small far plates break up. Measured on the gate video at 1080p: 23 -> 63
        # frames with a plate found. 4x/6x shrinks hit OpenCV's fast integer INTER_AREA path.
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        frame = cv2.resize(frame, (nw, nh), interpolation=interp)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    out = cv2.copyMakeBorder(
        frame,
        pad_y,
        size - nh - pad_y,
        pad_x,
        size - nw - pad_x,
        cv2.BORDER_CONSTANT,
        value=LETTERBOX_COLOR,
    )
    return out, scale, pad_x, pad_y


def decode(
    pred: np.ndarray,
    *,
    conf_threshold: float,
    iou_threshold: float,
    class_ids: list[int] | None,
    scale: float,
    pad_x: int,
    pad_y: int,
    frame_w: int,
    frame_h: int,
) -> list[Box]:
    """Turn a raw YOLO head output into NMS-filtered full-frame boxes (best score first)."""
    pred = np.asarray(pred, dtype=np.float32)
    pred = pred.reshape(pred.shape[-2], pred.shape[-1]) if pred.ndim == 3 else pred
    if pred.ndim != 2 or min(pred.shape) < 5:
        raise ValueError(f"unexpected detector output shape {pred.shape}")
    # (4 + C, N) -> (N, 4 + C). Anchors always outnumber channels for real models.
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T
    cls_scores = pred[:, 4:]
    if class_ids is not None:
        keep_cls = [c for c in class_ids if 0 <= c < cls_scores.shape[1]]
        if not keep_cls:
            return []
        masked = np.full_like(cls_scores, -1.0)
        masked[:, keep_cls] = cls_scores[:, keep_cls]
        cls_scores = masked
    scores = cls_scores.max(axis=1)
    mask = scores >= conf_threshold
    if not mask.any():
        return []
    xywh, scores = pred[mask, :4], scores[mask]

    # Back to full-frame pixels (float), then clip.
    x1 = (xywh[:, 0] - xywh[:, 2] / 2 - pad_x) / scale
    y1 = (xywh[:, 1] - xywh[:, 3] / 2 - pad_y) / scale
    x2 = (xywh[:, 0] + xywh[:, 2] / 2 - pad_x) / scale
    y2 = (xywh[:, 1] + xywh[:, 3] / 2 - pad_y) / scale
    x1, x2 = np.clip(x1, 0, frame_w), np.clip(x2, 0, frame_w)
    y1, y2 = np.clip(y1, 0, frame_h), np.clip(y2, 0, frame_h)

    rects = [
        [float(a), float(b), float(c - a), float(d - b)] for a, b, c, d in zip(x1, y1, x2, y2, strict=True)
    ]
    idx = cv2.dnn.NMSBoxes(rects, scores.astype(float).tolist(), conf_threshold, iou_threshold)
    boxes: list[Box] = []
    for i in np.asarray(idx, dtype=int).reshape(-1):
        bx1, by1 = int(np.floor(x1[i])), int(np.floor(y1[i]))
        bx2, by2 = int(np.ceil(x2[i])), int(np.ceil(y2[i]))
        if bx2 > bx1 and by2 > by1:
            boxes.append(Box(bx1, by1, bx2, by2, float(scores[i])))
    boxes.sort(key=lambda b: b.score, reverse=True)
    return boxes


class _YoloBase:
    def __init__(self, cfg: DetectorConfig) -> None:
        self.cfg = cfg

    def _infer(self, img_bgr: np.ndarray) -> np.ndarray:  # letterboxed uint8 BGR -> raw head output
        raise NotImplementedError

    def detect(self, frame: np.ndarray) -> list[Box]:
        h, w = frame.shape[:2]
        img, scale, pad_x, pad_y = letterbox(frame, self.cfg.input_size)
        pred = self._infer(img)
        return decode(
            pred,
            conf_threshold=self.cfg.conf_threshold,
            iou_threshold=self.cfg.iou_threshold,
            class_ids=self.cfg.class_ids,
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
            frame_w=w,
            frame_h=h,
        )


class NcnnDetector(_YoloBase):
    """YOLO11n exported with `yolo export format=ncnn` (a directory with model.ncnn.param/.bin)."""

    def __init__(self, cfg: DetectorConfig) -> None:
        super().__init__(cfg)
        import ncnn  # local import: the ONNX baseline must not need ncnn

        self._ncnn = ncnn
        param, weights = _ncnn_files(Path(cfg.model_path))
        net = ncnn.Net()
        net.opt.use_vulkan_compute = False  # Pi 3B+ has no usable Vulkan GPU
        net.opt.num_threads = cfg.num_threads
        net.opt.lightmode = True  # free intermediate blobs early (1 GB RAM)
        if net.load_param(str(param)) != 0 or net.load_model(str(weights)) != 0:
            raise RuntimeError(f"ncnn failed to load {param} / {weights}")
        self._net = net
        ins, outs = list(net.input_names()), list(net.output_names())
        self._in = ins[0] if len(ins) == 1 else NCNN_INPUT
        self._out = outs[0] if len(outs) == 1 else NCNN_OUTPUT
        self._norm = [1 / 255.0] * 3

    def _infer(self, img_bgr: np.ndarray) -> np.ndarray:
        ncnn = self._ncnn
        s = img_bgr.shape[0]
        mat = ncnn.Mat.from_pixels(np.ascontiguousarray(img_bgr), ncnn.Mat.PixelType.PIXEL_BGR2RGB, s, s)
        mat.substract_mean_normalize([], self._norm)
        ex = self._net.create_extractor()
        try:
            ex.input(self._in, mat)
            ret, out = ex.extract(self._out)
            if ret != 0:
                raise RuntimeError(f"ncnn extract failed ({ret})")
            return np.array(out)  # copy: `out` is owned by the extractor
        finally:
            ex.clear()

    def close(self) -> None:
        self._net.clear()


class OnnxDetector(_YoloBase):
    """Generic YOLOv8/11 ONNX (`yolo export format=onnx`), run with ONNX Runtime on CPU."""

    def __init__(self, cfg: DetectorConfig) -> None:
        super().__init__(cfg)
        import onnxruntime as ort

        path = Path(cfg.model_path)
        if not path.is_file():
            raise FileNotFoundError(f"detector ONNX model not found: {path}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = cfg.num_threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self._sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        inp = self._sess.get_inputs()[0]
        self._input = inp.name
        dims = [d for d in inp.shape[2:] if isinstance(d, int)]
        if dims and any(d != cfg.input_size for d in dims):
            raise ValueError(f"{path} expects input {inp.shape}, config says input_size={cfg.input_size}")

    def _infer(self, img_bgr: np.ndarray) -> np.ndarray:
        x = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return self._sess.run(None, {self._input: x})[0]

    def close(self) -> None:
        pass


def _ncnn_files(path: Path) -> tuple[Path, Path]:
    if path.is_dir():
        param, weights = path / "model.ncnn.param", path / "model.ncnn.bin"
    elif path.suffix == ".param":
        param, weights = path, path.with_suffix(".bin")
    else:
        raise FileNotFoundError(f"ncnn model must be an export directory or a .param file: {path}")
    for f in (param, weights):
        if not f.is_file():
            raise FileNotFoundError(f"ncnn model file not found: {f}")
    return param, weights


def make_detector(cfg: DetectorConfig) -> PlateDetector:
    det: PlateDetector = NcnnDetector(cfg) if cfg.backend == "ncnn" else OnnxDetector(cfg)
    log.info(
        "detector: %s %s input=%d threads=%d", cfg.backend, cfg.model_path, cfg.input_size, cfg.num_threads
    )
    return det
