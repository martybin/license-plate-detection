from __future__ import annotations

import re

PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ENGLISH_DIGITS = "0123456789"
PERSIAN_TO_ENGLISH = str.maketrans(PERSIAN_DIGITS, ENGLISH_DIGITS)

IRAN_PLATE_PATTERN = re.compile(
    r"^(\d{2})([ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی])(\d{3})(\d{2})$"
)


def normalize_iran_plate(text: str) -> str:
    if not text:
        return ""
    text = text.translate(PERSIAN_TO_ENGLISH)
    text = re.sub(r"[^0-9ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی]", "", text)
    return text


def is_valid_iran_plate(plate: str) -> bool:
    if not plate:
        return False
    return bool(IRAN_PLATE_PATTERN.match(plate))


def format_plate_display(plate: str) -> str:
    m = IRAN_PLATE_PATTERN.match(plate)
    if not m:
        return plate
    return f"{m.group(1)} {m.group(2)} {m.group(3)} | {m.group(4)}"
