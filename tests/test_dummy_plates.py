"""Handling the IR-LPR 'dummy' subset: Latin letters and a corrupted panel."""
from __future__ import annotations

import pytest

from tests.conftest import make_voc_xml
from training.prepare_dataset import (
    _overlaps,
    extract_plate_from_xml,
    split_body_and_panel,
    usable_region,
)
from utils.plate_utils import PLATE_LETTERS, is_valid_iran_plate, normalize_iran_plate

# A clean plate: 2 digits, letter, 3 digits, then the 2-digit region panel.
CLEAN = [
    ("5", 57, 26, 110, 100), ("9", 110, 28, 158, 98), ("گ", 158, 12, 270, 120),
    ("8", 270, 28, 330, 110), ("1", 330, 28, 385, 110), ("3", 385, 28, 445, 110),
    ("7", 470, 30, 520, 110), ("5", 520, 30, 570, 110),
]

# The same plate as the dummy_gaf group ships it: the region panel is rendered
# several times on top of itself, so those two digits have no correct label.
CORRUPT_PANEL = CLEAN[:6] + [
    ("7", 470, 30, 495, 110), ("5", 495, 30, 520, 110), ("7", 508, 30, 537, 110),
    ("5", 535, 30, 560, 110), ("3", 537, 30, 568, 110), ("5", 560, 30, 590, 110),
]


class TestLatinPlateLetters:
    @pytest.mark.parametrize("letter", ["D", "S"])
    def test_are_plate_letters(self, letter):
        """Diplomatic and political plates use a Latin letter, same layout."""
        assert letter in PLATE_LETTERS

    @pytest.mark.parametrize("plate", ["63D11319", "55S28179"])
    def test_validate_like_any_other_plate(self, plate):
        assert is_valid_iran_plate(normalize_iran_plate(plate))

    def test_survive_normalisation(self):
        assert normalize_iran_plate("63D11319") == "63D11319"

    def test_persian_plates_still_work(self):
        assert is_valid_iran_plate(normalize_iran_plate("12ع44964"))


class TestTashrifatIsExcluded:
    def test_word_is_not_folded_to_a_letter(self):
        """'تشریفات' is a word printed across the plate, not a letter glyph.

        Folding it to 'ت' would teach the model that a wide word image is that
        letter. The label stays long instead, so the length filter drops it.
        """
        label = normalize_iran_plate("تشریفات6170")
        assert label != "ت6170"
        assert not (5 <= len(label) <= 10), "should fall outside the usable length band"


class TestOverlapDetection:
    def test_clean_annotation_has_no_overlaps(self):
        boxes = [type("B", (), dict(xmin=a, xmax=b, ymin=0, ymax=10, name="x"))()
                 for a, b in [(0, 10), (12, 22), (24, 34)]]
        assert not _overlaps(boxes)

    def test_stacked_boxes_are_detected(self):
        boxes = [type("B", (), dict(xmin=a, xmax=b, ymin=0, ymax=10, name="x"))()
                 for a, b in [(0, 10), (5, 15)]]
        assert _overlaps(boxes)


class TestUsableRegion:
    def test_clean_plate_is_kept_whole(self, tmp_path):
        from training.prepare_dataset import parse_xml

        boxes, _ = parse_xml(make_voc_xml(CLEAN, tmp_path / "a.xml"))
        trusted, cropped = usable_region(boxes)
        assert len(trusted) == 8 and cropped is False

    def test_corrupted_panel_is_dropped(self, tmp_path):
        from training.prepare_dataset import parse_xml

        boxes, _ = parse_xml(make_voc_xml(CORRUPT_PANEL, tmp_path / "a.xml"))
        trusted, cropped = usable_region(boxes)
        assert cropped is True
        assert "".join(b.name for b in trusted) == "59گ813"

    def test_split_finds_the_panel_divider(self, tmp_path):
        from training.prepare_dataset import parse_xml

        boxes, _ = parse_xml(make_voc_xml(CLEAN, tmp_path / "a.xml"))
        body, panel = split_body_and_panel(boxes)
        assert len(body) == 6 and len(panel) == 2


class TestExtractedLabels:
    def test_clean_plate_yields_the_full_number(self, tmp_path):
        assert extract_plate_from_xml(make_voc_xml(CLEAN, tmp_path / "a.xml")) == "59گ81375"

    def test_corrupted_panel_yields_the_trustworthy_prefix(self, tmp_path):
        """Better a short true label than an invented region code."""
        label = extract_plate_from_xml(make_voc_xml(CORRUPT_PANEL, tmp_path / "a.xml"))
        assert label == "59گ813"

    def test_partial_label_is_still_trainable(self, tmp_path):
        """It fails the 8-char plate format but the OCR loader accepts it."""
        charset = "0123456789" + PLATE_LETTERS
        label = extract_plate_from_xml(make_voc_xml(CORRUPT_PANEL, tmp_path / "a.xml"))
        assert not is_valid_iran_plate(label)
        assert 5 <= len(label) <= 10 and all(c in charset for c in label)

    def test_the_rare_letter_survives(self, tmp_path):
        """گ is the whole reason this group is worth salvaging."""
        assert "گ" in extract_plate_from_xml(make_voc_xml(CORRUPT_PANEL, tmp_path / "a.xml"))
