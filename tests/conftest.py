"""Shared fixtures.

Tests are split by how heavy their dependencies are: the plate-text and database
suites are pure stdlib and always run, the image/pipeline suites need numpy and
OpenCV, and only the model suites need torch. Each module skips itself when its
dependency is absent so a partial environment still runs everything it can.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CHARSET = "0123456789ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی"


class FakeRecognition(NamedTuple):
    text: str
    confidence: float


@pytest.fixture
def charset() -> str:
    return CHARSET


@pytest.fixture
def voc_xml():
    """Factory writing a Pascal-VOC file shaped like the IR-LPR annotations.

    Exposed as a fixture rather than an importable helper on purpose: `tests` is
    not a package here, and ultralytics ships a top-level package of that name,
    so `from tests.conftest import ...` silently resolves to theirs.
    """
    return make_voc_xml


@pytest.fixture
def db(tmp_path):
    """A VehicleDB on a throwaway file, closed afterwards."""
    from utils.database import VehicleDB

    database = VehicleDB(tmp_path / "vehicles.db")
    yield database
    database.close()


@pytest.fixture
def plate_image():
    """A synthetic 200x60 BGR plate-ish image."""
    np = pytest.importorskip("numpy")
    image = np.full((60, 200, 3), 220, dtype=np.uint8)
    image[10:50, 10:190] = 255
    for x in range(20, 180, 25):  # dark blocks standing in for glyphs
        image[18:44, x : x + 14] = 30
    return image


def make_voc_xml(objects, path: Path, size=None) -> Path:
    """Write a Pascal-VOC file shaped like the IR-LPR annotations.

    `objects` is a list of (name, xmin, ymin, xmax, ymax). The real IR-LPR files
    carry no <size> element, so it is omitted unless explicitly requested.
    """
    parts = ["<annotation><filename>x.jpg</filename>"]
    if size is not None:
        parts.append(f"<size><width>{size[0]}</width><height>{size[1]}</height></size>")
    for name, x1, y1, x2, y2 in objects:
        parts.append(
            f"<object><name>{name}</name><bndbox>"
            f"<xmin>{x1}</xmin><ymin>{y1}</ymin><xmax>{x2}</xmax><ymax>{y2}</ymax>"
            f"</bndbox></object>"
        )
    parts.append("</annotation>")
    path.write_text("".join(parts), encoding="utf-8")
    return path
