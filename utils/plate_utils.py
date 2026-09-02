from __future__ import annotations

import re
from typing import List, NamedTuple, Optional

PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
ENGLISH_DIGITS = "0123456789"
PLATE_LETTERS = "ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی"

# Arabic code points that render identically to their Persian counterparts and
# routinely leak in from XML annotations and keyboard input.
ARABIC_TO_PERSIAN = {"ك": "ک", "ي": "ی", "ﻻ": "لا", "ة": "ه", "أ": "ا", "إ": "ا", "آ": "ا"}

_DIGIT_TABLE = str.maketrans(PERSIAN_DIGITS + ARABIC_DIGITS, ENGLISH_DIGITS * 2)
_LETTER_TABLE = str.maketrans(ARABIC_TO_PERSIAN)

# Some letters are annotated by their spelled-out name rather than the glyph, and
# special plate classes are annotated as a whole phrase. Longest first, so a
# phrase is consumed before any shorter key can match inside it.
WORD_TO_LETTER = {
    "معلولینوجانبازان": "ژ",
    "معلولین": "ژ",
    "جانبازان": "ژ",
    "تشریفات": "ت",
    # 'الف' is the *name* of the letter ا; without this it survives as three
    # separate plate letters and every such label is silently malformed.
    "الف": "ا",
}

_STRIP_PATTERN = re.compile(f"[^0-9{PLATE_LETTERS}]")

# First group and region code both run 10-99: neither carries a leading zero.
# Verified against all 23302 well-formed labels in data/ocr_dataset.
IRAN_PLATE_PATTERN = re.compile(
    f"^([1-9][0-9])([{PLATE_LETTERS}])([0-9]{{3}})([1-9][0-9])$"
)


class PlateParts(NamedTuple):
    prefix: str
    letter: str
    serial: str
    region: str

    @property
    def plate(self) -> str:
        return f"{self.prefix}{self.letter}{self.serial}{self.region}"


def _collapse_repeated_letters(plate: str) -> str:
    """Drop an immediately repeated letter.

    A plate carries exactly one letter, so a run can only be an artefact -- the
    annotations spell special plates as both the glyph and the word ('ژ' plus
    'معلولین و جانبازان'), which would otherwise normalise to 'ژژ'.
    """
    out: List[str] = []
    for char in plate:
        if out and char == out[-1] and char in PLATE_LETTERS:
            continue
        out.append(char)
    return "".join(out)


def normalize_iran_plate(text: str) -> str:
    """Fold digits/letters to canonical Persian forms and drop everything else."""
    if not text:
        return ""
    text = text.translate(_DIGIT_TABLE).translate(_LETTER_TABLE)
    # Strip separators *before* matching the word forms: the annotations spell
    # the class as 'ژ (معلولین و جانبازان)' with spaces, parentheses and
    # zero-width joiners, none of which appear in the lookup keys.
    text = _STRIP_PATTERN.sub("", text)
    for word, letter in WORD_TO_LETTER.items():
        text = text.replace(word, letter)
    return _collapse_repeated_letters(text)


def parse_iran_plate(plate: str) -> Optional[PlateParts]:
    match = IRAN_PLATE_PATTERN.match(plate or "")
    if not match:
        return None
    return PlateParts(*match.groups())


def is_valid_iran_plate(plate: str) -> bool:
    return parse_iran_plate(plate) is not None


def repair_plate(text: str) -> str:
    """Rebuild a canonical plate from a slightly malformed OCR reading.

    CTC readings often carry a duplicated or dropped glyph. When the reading has
    exactly one letter and exactly seven digits, the layout is unambiguous, so we
    can reassemble it rather than throw the detection away. Anything less certain
    is returned untouched for the validity check to reject.
    """
    plate = normalize_iran_plate(text)
    if is_valid_iran_plate(plate):
        return plate

    letters = [(i, c) for i, c in enumerate(plate) if c in PLATE_LETTERS]
    digits = [c for c in plate if c.isdigit()]
    if len(letters) != 1 or len(digits) != 7:
        return plate

    letter_pos = letters[0][0]
    digits_before = sum(1 for c in plate[:letter_pos] if c.isdigit())
    # Trust the letter's position only when it sits where a plate letter belongs.
    if digits_before != 2:
        return plate

    candidate = f"{digits[0]}{digits[1]}{letters[0][1]}{''.join(digits[2:5])}{digits[5]}{digits[6]}"
    return candidate if is_valid_iran_plate(candidate) else plate


def format_plate_display(plate: str) -> str:
    parts = parse_iran_plate(plate)
    if parts is None:
        return plate
    return f"{parts.prefix} {parts.letter} {parts.serial} | {parts.region}"
