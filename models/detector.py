from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from ultralytics import YOLO


class PlateDetector:
    def __init__(
        self,
        model_path: str | Path,
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        img_size: int = 640,
        device: str = "cuda",
    ) -> None:
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.model = YOLO(str(model_path))
        self.model.to(self.device)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size

    def detect(self, image: np.ndarray) -> List[Tuple[np.ndarray, float, Tuple[int, int, int, int]]]:
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            device=self.device,
            verbose=False,
        )
        detections: List[Tuple[np.ndarray, float, Tuple[int, int, int, int]]] = []
        if not results:
            return detections

        result = results[0]
        if result.boxes is None:
            return detections

        h, w = image.shape[:2]
        for box in result.boxes:
            conf = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image[y1:y2, x1:x2].copy()
            detections.append((crop, conf, (x1, y1, x2, y2)))
        return detections

    def detect_best(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, float, Tuple[int, int, int, int]]]:
        dets = self.detect(image)
        if not dets:
            return None
        return max(dets, key=lambda x: x[1])
