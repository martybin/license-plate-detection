from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Deque, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from models.detector import BBox, PlateDetector
from models.recognizer import PlateRecognizer, Recognition
from utils.database import VehicleDB
from utils.image_processing import correct_perspective, enhance_plate
from utils.plate_utils import is_valid_iran_plate, repair_plate


class PlateResult(NamedTuple):
    plate: Optional[str]
    info: Optional[dict]
    bbox: Optional[BBox]
    det_conf: float
    ocr_conf: float
    confirmed: bool


class PlateVoter:
    """Fuses readings across consecutive frames before trusting a plate.

    A single frame can be ruined by dust, a headlight flare or motion blur, but
    those failures are uncorrelated between frames while the true plate is not.
    Accumulating confidence-weighted votes over a short window is what turns a
    good-per-frame reader into a reliable gate.
    """

    def __init__(self, window_seconds: float = 2.0, min_votes: int = 3, min_score: float = 1.5) -> None:
        self.window_seconds = window_seconds
        self.min_votes = min_votes
        self.min_score = min_score
        self._votes: Deque[Tuple[float, str, float]] = deque()

    def add(self, plate: str, confidence: float, now: Optional[float] = None) -> None:
        self._votes.append((now if now is not None else time.monotonic(), plate, confidence))

    def _expire(self, now: float) -> None:
        while self._votes and now - self._votes[0][0] > self.window_seconds:
            self._votes.popleft()

    def best(self, now: Optional[float] = None) -> Optional[Tuple[str, float, int]]:
        now = now if now is not None else time.monotonic()
        self._expire(now)
        if not self._votes:
            return None

        scores: Dict[str, float] = defaultdict(float)
        counts: Dict[str, int] = defaultdict(int)
        for _, plate, conf in self._votes:
            scores[plate] += conf
            counts[plate] += 1

        plate = max(scores, key=lambda p: scores[p])
        return plate, scores[plate], counts[plate]

    def confirmed(self, now: Optional[float] = None) -> Optional[str]:
        best = self.best(now)
        if best is None:
            return None
        plate, score, votes = best
        if votes >= self.min_votes and score >= self.min_score:
            return plate
        return None

    def reset(self) -> None:
        self._votes.clear()


class LPRPipeline:
    def __init__(
        self,
        detector: PlateDetector,
        recognizer: PlateRecognizer,
        db: VehicleDB,
        deskew: bool = True,
        enhance_params: Optional[dict] = None,
        voter: Optional[PlateVoter] = None,
    ) -> None:
        self.detector = detector
        self.recognizer = recognizer
        self.db = db
        self.deskew = deskew
        self.enhance_params = enhance_params or {}
        self.voter = voter or PlateVoter()

    def _build_variants(self, crop: np.ndarray) -> List[np.ndarray]:
        """Produce a few renderings of the same plate for the reader to choose from.

        Enhancement helps a dusty or backlit plate but can hurt an already-clean
        one, so we submit both and let the OCR confidence decide rather than
        committing to a single preprocessing chain up front.
        """
        variants = [crop]
        enhanced = enhance_plate(crop, **self.enhance_params)
        variants.append(enhanced)
        if self.deskew:
            warped = correct_perspective(crop)
            if warped is not crop and warped.size > 0:
                variants.append(enhance_plate(warped, **self.enhance_params))
        return variants

    def _read_plate(self, crop: np.ndarray) -> Optional[Recognition]:
        variants = self._build_variants(crop)
        # One batched forward pass for every variant: three crops cost barely
        # more than one on the GPU, and far less than three separate calls.
        readings = [r for r in self.recognizer.recognize_batch(variants) if r is not None]
        if not readings:
            return None

        repaired = [Recognition(repair_plate(r.text), r.confidence) for r in readings]
        valid = [r for r in repaired if is_valid_iran_plate(r.text)]
        # A structurally valid plate always beats a higher-confidence garbage read.
        return max(valid or repaired, key=lambda r: r.confidence)

    def process(self, frame: np.ndarray) -> PlateResult:
        best = self.detector.detect_best(frame)
        if best is None:
            confirmed = self.voter.confirmed()
            return PlateResult(confirmed, self.db.lookup(confirmed) if confirmed else None, None, 0.0, 0.0, bool(confirmed))

        reading = self._read_plate(best.crop)
        if reading is None:
            return PlateResult(None, None, best.bbox, best.confidence, 0.0, False)

        if is_valid_iran_plate(reading.text):
            self.voter.add(reading.text, reading.confidence)

        confirmed = self.voter.confirmed()
        if confirmed is None:
            # Show the provisional read so the operator sees the system working,
            # but withhold the driver record until the vote settles.
            return PlateResult(reading.text, None, best.bbox, best.confidence, reading.confidence, False)

        return PlateResult(
            confirmed,
            self.db.lookup(confirmed),
            best.bbox,
            best.confidence,
            reading.confidence,
            True,
        )
