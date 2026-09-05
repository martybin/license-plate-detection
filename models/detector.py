from __future__ import annotations

from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import torch
from ultralytics import YOLO

BBox = Tuple[int, int, int, int]


class Detection(NamedTuple):
    crop: np.ndarray
    confidence: float
    bbox: BBox


class PlateDetector:
    def __init__(
        self,
        model_path: str | Path,
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        img_size: int = 640,
        device: str = "cuda",
        pad_ratio: float = 0.06,
        max_det: int = 8,
        half: bool = True,
    ) -> None:
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.model = YOLO(str(model_path))
        self.model.to(self.device)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.pad_ratio = pad_ratio
        self.max_det = max_det
        self.use_half = half and self.device == "cuda"

        # Ultralytics renamed `half` to `quantize` in 8.4 and warns on every
        # predict call for the old name -- 30 lines a second into journalctl on a
        # live gate. Resolve the supported name once and only pass it when fp16
        # is actually wanted.
        self._predict_extra = {}
        if self.use_half:
            self._predict_extra[self._half_kwarg()] = True

        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
        self._warmup()

    @staticmethod
    def _half_kwarg() -> str:
        try:
            from ultralytics.cfg import DEFAULT_CFG_DICT

            return "quantize" if "quantize" in DEFAULT_CFG_DICT else "half"
        except ImportError:
            return "half"

    def _warmup(self) -> None:
        """Run one dummy frame so the first real vehicle is not the slow one."""
        dummy = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        try:
            self.model.predict(
                source=dummy,
                imgsz=self.img_size,
                device=self.device,
                verbose=False,
                **self._predict_extra,
            )
        except Exception:
            # A warmup failure is never fatal; the real call will surface it.
            pass

    def _pad_box(self, x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> BBox:
        """Grow the box slightly so tight boxes do not clip the outer glyphs."""
        pad_x = int((x2 - x1) * self.pad_ratio)
        pad_y = int((y2 - y1) * self.pad_ratio)
        return (
            int(max(0, x1 - pad_x)),
            int(max(0, y1 - pad_y)),
            int(min(w, x2 + pad_x)),
            int(min(h, y2 + pad_y)),
        )

    def detect(self, image: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            device=self.device,
            max_det=self.max_det,
            verbose=False,
            **self._predict_extra,
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []

        boxes = results[0].boxes
        h, w = image.shape[:2]
        # Pull the whole tensor across the PCIe bus once instead of calling
        # .item() per box, which forces a separate GPU sync each time.
        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()

        detections: List[Detection] = []
        for (x1, y1, x2, y2), conf in zip(xyxy, confs):
            x1, y1, x2, y2 = self._pad_box(x1, y1, x2, y2, w, h)
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(Detection(image[y1:y2, x1:x2].copy(), float(conf), (x1, y1, x2, y2)))
        return detections

    def detect_best(self, image: np.ndarray) -> Optional[Detection]:
        detections = self.detect(image)
        if not detections:
            return None
        # Prefer the largest confident plate: at a gate the nearest truck is the
        # one at the barrier, and a bigger crop also reads more reliably.
        return max(
            detections,
            key=lambda d: d.confidence * ((d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1])) ** 0.5,
        )
