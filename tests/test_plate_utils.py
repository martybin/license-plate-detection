"""Plate text normalisation, validation and repair.

The raw strings here are the exact object names found in the IR-LPR XML files,
not idealised versions of them -- several bugs in this module survived earlier
review precisely because they were tested against cleaned-up input.
"""
from __future__ import annotations

import pytest

from utils.plate_utils import (
    PLATE_LETTERS,
    format_plate_display,
    is_valid_iran_plate,
    normalize_iran_plate,
    parse_iran_plate,
    repair_plate,
)

ZWJ = "‍"


class TestNormalize:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("12ب34567", "12ب34567"),
            ("۱۲ب۳۴۵۶۷", "12ب34567"),  # Persian digits
            ("١٢ب٣٤٥٦٧", "12ب34567"),  # Arabic-Indic digits
            ("12 ب 345 67", "12ب34567"),  # separators
            ("12-ب-345-67", "12ب34567"),
            ("", ""),
        ],
    )
    def test_digits_and_separators(self, raw, expected):
        assert normalize_iran_plate(raw) == expected

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("12ك34567", "12ک34567"),  # Arabic kaf -> Persian kaf
            ("12ي34567", "12ی34567"),  # Arabic yeh -> Persian yeh
        ],
    )
    def test_arabic_letters_folded(self, raw, expected):
        assert normalize_iran_plate(raw) == expected

    def test_zwj_stripped(self):
        """1632 objects in the dataset are 'ه' followed by a zero-width joiner."""
        assert normalize_iran_plate(f"52ه{ZWJ}55768") == "52ه55768"

    def test_alef_spelled_out(self):
        """78 objects name the letter as 'الف'; kept raw it becomes three letters."""
        assert normalize_iran_plate("11الف12898") == "11ا12898"

    @pytest.mark.parametrize(
        "raw",
        [
            "11ژ (معلولین و جانبازان)14452",  # exact form in the XML
            "11ژمعلولینوجانبازان14452",  # already-stripped form
            "11معلولینوجانبازان14452",  # word only, glyph absent
        ],
    )
    def test_disabled_veteran_plate(self, raw):
        """The class is annotated as a phrase alongside its glyph; both collapse to ژ."""
        assert normalize_iran_plate(raw) == "11ژ14452"

    def test_repeated_letter_collapsed_but_digits_kept(self):
        assert normalize_iran_plate("11بب22333") == "11ب22333"
        assert normalize_iran_plate("11ب22333") == "11ب22333"  # repeated digits survive

    def test_unknown_characters_dropped(self):
        assert normalize_iran_plate("12ب34567!@#$") == "12ب34567"
        assert normalize_iran_plate("ABC") == ""


class TestValidate:
    @pytest.mark.parametrize("plate", ["12ب34567", "11ژ14452", "99ی99999", "10ا10010"])
    def test_accepts_well_formed(self, plate):
        assert is_valid_iran_plate(plate)

    @pytest.mark.parametrize(
        "plate, why",
        [
            ("", "empty"),
            ("12ب3456", "too short"),
            ("12ب345678", "too long"),
            ("02ب34567", "leading zero in prefix"),
            ("12ب34505", "region code cannot start with zero"),
            ("1234567", "no letter"),
            ("ببببببب", "no digits"),
            ("12ب345ب7", "two letters"),
        ],
    )
    def test_rejects_malformed(self, plate, why):
        assert not is_valid_iran_plate(plate), why

    def test_every_charset_letter_is_accepted(self):
        for letter in PLATE_LETTERS:
            assert is_valid_iran_plate(f"12{letter}34567"), letter

    def test_parse_returns_parts(self):
        parts = parse_iran_plate("12ب34567")
        assert (parts.prefix, parts.letter, parts.serial, parts.region) == ("12", "ب", "345", "67")
        assert parts.plate == "12ب34567"

    def test_parse_returns_none_when_invalid(self):
        assert parse_iran_plate("nonsense") is None


class TestRepair:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("12ب34567", "12ب34567"),  # already valid, untouched
            ("12 ب 345 67", "12ب34567"),
            ("۱۲-ب-۳۴۵-۶۷", "12ب34567"),
        ],
    )
    def test_repairs_recoverable(self, raw, expected):
        assert repair_plate(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "1ب2345677",  # letter in the wrong position: ambiguous
            "12ب3456",  # a digit is missing outright
            "12ب345ب67",  # two letters
        ],
    )
    def test_leaves_ambiguous_alone(self, raw):
        """Guessing here would invent a plate; validation must reject it instead."""
        assert not is_valid_iran_plate(repair_plate(raw))


class TestDisplay:
    def test_formats_valid_plate(self):
        assert format_plate_display("12ب34567") == "12 ب 345 | 67"

    def test_passes_through_invalid(self):
        assert format_plate_display("garbage") == "garbage"


class TestAgainstRealDataset:
    """Regression guard against the actual filenames, when the dataset is present."""

    def test_dataset_labels_normalise(self, charset):
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent / "data" / "ocr_dataset"
        if not root.is_dir():
            pytest.skip("data/ocr_dataset not built")

        files = [p for p in root.iterdir() if p.suffix.lower() == ".jpg"]
        if not files:
            pytest.skip("data/ocr_dataset is empty")

        usable = 0
        for path in files:
            stem = path.stem
            if "_" in stem:
                head, _, tail = stem.rpartition("_")
                if head and tail.isdigit():
                    stem = head
            label = normalize_iran_plate(stem)
            if 5 <= len(label) <= 10 and all(c in charset for c in label):
                usable += 1

        ratio = usable / len(files)
        assert ratio > 0.99, f"only {ratio:.1%} of labels are usable"
