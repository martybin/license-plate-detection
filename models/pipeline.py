from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from models.detector import PlateDetector
from models.recognizer import PlateRecognizer
from utils.image_processing import enhance_plate, correct_perspective
from utils.plate_utils import normalize_iran_plate, is_valid_iran_plate
from utils.database import VehicleDB


class LPRPipeline:
    def __init__(
        self,
        detector: PlateDetector,
        recognizer: PlateRecognizer,
        db: VehicleDB,
        deskew: bool = True,
    ) -> None:
        self.detector = detector
        self.recognizer = recognizer
        self.db = db
        self.deskew = deskew

    def process(
        self, frame: np.ndarray
    ) -> Tuple[Optional[str], Optional[dict], Optional[Tuple[int, int, int, int]], float]:
        best = self.detector.detect_best(frame)
        if best is None:
            return None, None, None, 0.0

        crop, conf, bbox = best
        if self.deskew:
            crop = correct_perspective(crop)
        crop = enhance_plate(crop)

        raw_text = self.recognizer.recognize(crop)
        if raw_text is None:
            return None, None, bbox, conf

        plate = normalize_iran_plate(raw_text)
        if not is_valid_iran_plate(plate):
            return plate, None, bbox, conf

        info = self.db.lookup(plate)
        return plate, info, bbox, conf
