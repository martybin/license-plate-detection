"""Multi-frame voting and the detect -> enhance -> read -> confirm flow."""
from __future__ import annotations

from typing import NamedTuple

import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch", reason="models.recognizer needs torch for Recognition")

from models.pipeline import LPRPipeline, PlateVoter  # noqa: E402
from models.recognizer import Recognition  # noqa: E402

PLATE = "12ب34567"
OTHER = "34ج67890"


class FakeDetection(NamedTuple):
    crop: object
    confidence: float
    bbox: tuple


class Img:
    """Stands in for a crop; carries a tag so we can assert which one was used."""

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.size = 100

    def __repr__(self) -> str:
        return self.tag

    def __eq__(self, other) -> bool:
        return isinstance(other, Img) and other.tag == self.tag


CROP = Img("CROP")
FRAME = Img("FRAME")


class FakeDetector:
    def __init__(self, detecting: bool = True) -> None:
        self.detecting = detecting

    def detect_best(self, frame):
        return FakeDetection(CROP, 0.88, (10, 20, 200, 70)) if self.detecting else None


class ScriptedRecognizer:
    """Returns a fixed list of readings, one per variant submitted."""

    def __init__(self, readings) -> None:
        self.readings = readings
        self.batch_sizes = []

    def recognize_batch(self, images):
        self.batch_sizes.append(len(images))
        return list(self.readings)[: len(images)]


class SpySaver:
    def __init__(self) -> None:
        self.calls = []

    def save(self, **kwargs):
        self.calls.append(kwargs)
        return True


@pytest.fixture(autouse=True)
def _passthrough_enhancement(monkeypatch):
    """Keep the image maths out of these tests; tag the variants instead."""
    import models.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "enhance_plate", lambda img, **k: Img(f"ENH({img.tag})"))
    monkeypatch.setattr(pipeline_module, "correct_perspective", lambda img: Img(f"WARP({img.tag})"))


def build(recognizer, db, saver=None, deskew=True, **voter_kwargs):
    kwargs = {"window_seconds": 5.0, "min_votes": 2, "min_score": 1.0}
    kwargs.update(voter_kwargs)
    return LPRPipeline(
        detector=FakeDetector(),
        recognizer=recognizer,
        db=db,
        deskew=deskew,
        saver=saver,
        voter=PlateVoter(**kwargs),
    )


class TestPlateVoter:
    def test_empty_confirms_nothing(self):
        assert PlateVoter().confirmed(1000.0) is None

    def test_needs_minimum_vote_count(self):
        voter = PlateVoter(2.0, min_votes=3, min_score=1.5)
        t = 1000.0
        voter.add(PLATE, 0.9, t)
        voter.add(PLATE, 0.9, t + 0.1)
        assert voter.confirmed(t + 0.2) is None
        voter.add(PLATE, 0.9, t + 0.2)
        assert voter.confirmed(t + 0.3) == PLATE

    def test_needs_minimum_score_not_just_count(self):
        """Four weak readings should not outrank the confidence threshold."""
        voter = PlateVoter(2.0, min_votes=3, min_score=1.5)
        t = 1000.0
        for i in range(4):
            voter.add(PLATE, 0.3, t + i * 0.1)
        assert voter.confirmed(t + 0.5) is None
        voter.add(PLATE, 0.9, t + 0.5)
        assert voter.confirmed(t + 0.6) == PLATE

    def test_single_confident_outlier_loses(self):
        """One frame ruined by glare must not beat the accumulated true plate."""
        voter = PlateVoter(5.0, min_votes=3, min_score=1.5)
        t = 1000.0
        for i in range(3):
            voter.add(PLATE, 0.9, t + i * 0.1)
        voter.add(OTHER, 0.99, t + 0.4)
        assert voter.confirmed(t + 0.5) == PLATE

    def test_votes_expire(self):
        voter = PlateVoter(2.0, min_votes=2, min_score=1.0)
        t = 1000.0
        voter.add(PLATE, 0.9, t)
        voter.add(PLATE, 0.9, t + 0.1)
        assert voter.confirmed(t + 0.2) == PLATE
        assert voter.confirmed(t + 10.0) is None

    def test_reset(self):
        voter = PlateVoter(5.0, min_votes=1, min_score=0.1)
        voter.add(PLATE, 0.9, 1000.0)
        voter.reset()
        assert voter.confirmed(1000.1) is None

    def test_best_reports_score_and_count(self):
        voter = PlateVoter(5.0)
        t = 1000.0
        voter.add(PLATE, 0.5, t)
        voter.add(PLATE, 0.7, t + 0.1)
        plate, score, votes = voter.best(t + 0.2)
        assert plate == PLATE
        assert score == pytest.approx(1.2)
        assert votes == 2


class TestVariantSelection:
    def test_all_variants_go_in_one_batch(self, db):
        recognizer = ScriptedRecognizer([Recognition(PLATE, 0.9)] * 3)
        pipe = build(recognizer, db)
        pipe.process(FRAME)
        assert recognizer.batch_sizes == [3], "variants must share one forward pass"

    def test_deskew_disabled_drops_the_warped_variant(self, db):
        recognizer = ScriptedRecognizer([Recognition(PLATE, 0.9)] * 3)
        pipe = build(recognizer, db, deskew=False)
        pipe.process(FRAME)
        assert recognizer.batch_sizes == [2]

    def test_valid_reading_beats_more_confident_garbage(self, db):
        recognizer = ScriptedRecognizer(
            [Recognition("XXXX", 0.99), Recognition(PLATE, 0.80), Recognition("YY", 0.95)]
        )
        pipe = build(recognizer, db)
        reading, _ = pipe._read_plate(CROP)
        assert reading.text == PLATE

    def test_saved_variant_matches_the_reading_despite_a_none(self, db):
        """A dropped reading used to shift the index, saving the wrong variant.

        `readings` was filtered before indexing back into `variants`, so with
        variant 0 unreadable the audit image came from the wrong preprocessing.
        """
        recognizer = ScriptedRecognizer(
            [None, Recognition("XXXX", 0.99), Recognition(PLATE, 0.80)]
        )
        pipe = build(recognizer, db)
        reading, variant = pipe._read_plate(CROP)
        assert reading.text == PLATE
        assert variant == Img("ENH(WARP(CROP))")

    def test_all_unreadable(self, db):
        pipe = build(ScriptedRecognizer([None, None, None]), db)
        assert pipe._read_plate(CROP) == (None, None)


class TestProcess:
    def test_provisional_before_confirmation(self, db):
        db.seed_demo()
        pipe = build(ScriptedRecognizer([Recognition(PLATE, 0.9)] * 3), db, min_votes=3)
        result = pipe.process(FRAME)
        assert result.plate == PLATE
        assert result.confirmed is False
        assert result.info is None, "driver record must wait for the vote to settle"

    def test_driver_appears_once_confirmed(self, db):
        db.seed_demo()
        pipe = build(ScriptedRecognizer([Recognition(PLATE, 0.9)] * 3), db, min_votes=2)
        pipe.process(FRAME)
        result = pipe.process(FRAME)
        assert result.confirmed is True
        assert result.info["driver_name"] == "علی محمدی"

    def test_no_detection_returns_empty_result(self, db):
        pipe = build(ScriptedRecognizer([]), db)
        pipe.detector = FakeDetector(detecting=False)
        result = pipe.process(FRAME)
        assert result.plate is None and result.bbox is None and result.det_conf == 0.0

    def test_confirmed_plate_survives_a_dropped_frame(self, db):
        """The truck is still there even if one frame missed it."""
        db.seed_demo()
        pipe = build(ScriptedRecognizer([Recognition(PLATE, 0.9)] * 3), db, min_votes=2)
        pipe.process(FRAME)
        pipe.process(FRAME)
        pipe.detector = FakeDetector(detecting=False)
        result = pipe.process(FRAME)
        assert result.plate == PLATE and result.confirmed is True

    def test_unreadable_crop_reports_the_box_only(self, db):
        pipe = build(ScriptedRecognizer([None, None, None]), db)
        result = pipe.process(FRAME)
        assert result.plate is None
        assert result.bbox == (10, 20, 200, 70)
        assert result.det_conf == pytest.approx(0.88)

    def test_invalid_reading_never_votes(self, db):
        pipe = build(ScriptedRecognizer([Recognition("XXXX", 0.99)] * 3), db, min_votes=1, min_score=0.1)
        for _ in range(5):
            result = pipe.process(FRAME)
        assert result.confirmed is False


class TestSaverIntegration:
    def test_saver_receives_full_frame_and_box(self, db):
        """The vehicle photo is the artefact being audited, not just the crop."""
        db.seed_demo()
        saver = SpySaver()
        pipe = build(ScriptedRecognizer([Recognition(PLATE, 0.9)] * 3), db, saver=saver,
                     min_votes=1, min_score=0.5)
        pipe.process(FRAME)

        assert len(saver.calls) == 1
        call = saver.calls[0]
        assert call["plate"] == PLATE
        assert call["full_frame"] == FRAME
        assert call["bbox"] == (10, 20, 200, 70)
        assert call["raw_crop"] == CROP
        assert call["det_conf"] == pytest.approx(0.88)

    def test_nothing_saved_before_confirmation(self, db):
        saver = SpySaver()
        pipe = build(ScriptedRecognizer([Recognition(PLATE, 0.9)] * 3), db, saver=saver, min_votes=5)
        pipe.process(FRAME)
        assert saver.calls == []

    def test_pipeline_works_without_a_saver(self, db):
        pipe = build(ScriptedRecognizer([Recognition(PLATE, 0.9)] * 3), db, saver=None,
                     min_votes=1, min_score=0.5)
        assert pipe.process(FRAME).confirmed is True
