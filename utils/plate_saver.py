from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class PlateSaver:
    def __init__(
        self,
        output_dir: str | Path = "captures",
        save_raw: bool = True,
        save_enhanced: bool = True,
        min_interval_seconds: float = 3.0,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_raw = save_raw
        self.save_enhanced = save_enhanced
        self.min_interval = min_interval_seconds
        self._last_saved: dict[str, float] = {}

    def _should_save(self, plate: str, now: float) -> bool:
        last = self._last_saved.get(plate)
        if last is None:
            return True
        return (now - last) >= self.min_interval

    def save(
        self,
        plate: str,
        raw_crop: Optional[np.ndarray],
        enhanced_crop: Optional[np.ndarray] = None,
        ocr_conf: float = 0.0,
        det_conf: float = 0.0,
    ) -> Optional[Path]:
        if not plate or raw_crop is None or raw_crop.size == 0:
            return None

        now = time.monotonic()
        if not self._should_save(plate, now):
            return None

        ts = time.strftime("%Y%m%d_%H%M%S")
        safe_plate = "".join(c if c.isalnum() else "_" for c in plate)
        base = self.output_dir / f"{ts}_{safe_plate}_d{det_conf:.2f}_o{ocr_conf:.2f}"

        saved = None
        if self.save_raw:
            path = Path(str(base) + "_raw.jpg")
            cv2.imwrite(str(path), raw_crop)
            saved = path

        if self.save_enhanced and enhanced_crop is not None and enhanced_crop.size > 0:
            path = Path(str(base) + "_enh.jpg")
            cv2.imwrite(str(path), enhanced_crop)
            if saved is None:
                saved = path

        self._last_saved[plate] = now
        return saved
