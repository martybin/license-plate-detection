"""Dataset preparation: annotation parsing and YOLO label geometry."""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import make_voc_xml
from training.prepare_dataset import (
    _is_plate_object,
    extract_plate_from_xml,
    find_xml,
    image_size,
    parse_xml,
    plate_regions,
    union_box,
    write_yolo_label,
)

# A plate reading '12ز81' laid out left to right, as the character objects appear.
CHAR_OBJECTS = [
    ("1", 59, 56, 101, 171),
    ("2", 96, 63, 149, 181),
    ("ز", 157, 97, 211, 199),
    ("8", 219, 113, 273, 216),
    ("1", 285, 124, 325, 245),
]


class TestPlateObjectNames:
    @pytest.mark.parametrize(
        "name", ["کل ناحیه پلاک", "plate", "Plate", "license plate", "LP", "پلاک"]
    )
    def test_recognised_as_whole_plate(self, name):
        assert _is_plate_object(name)

    @pytest.mark.parametrize("name", ["1", "0", "ژ", "ب", "الف"])
    def test_characters_are_not_plate_objects(self, name):
        assert not _is_plate_object(name)


class TestParsing:
    def test_reads_objects_without_size_element(self, tmp_path):
        """IR-LPR annotations carry no <size>; the size must come from the image."""
        xml = make_voc_xml(CHAR_OBJECTS, tmp_path / "a.xml")
        boxes, size = parse_xml(xml)
        assert len(boxes) == 5
        assert size is None

    def test_reads_size_when_present(self, tmp_path):
        xml = make_voc_xml(CHAR_OBJECTS, tmp_path / "a.xml", size=(580, 317))
        _, size = parse_xml(xml)
        assert size == (580, 317)

    def test_malformed_xml_returns_empty(self, tmp_path):
        bad = tmp_path / "bad.xml"
        bad.write_text("<annotation><object>", encoding="utf-8")
        assert parse_xml(bad) == ([], None)

    def test_missing_file_returns_empty(self, tmp_path):
        assert parse_xml(tmp_path / "nope.xml") == ([], None)

    def test_objects_with_bad_coordinates_are_skipped(self, tmp_path):
        xml = tmp_path / "a.xml"
        xml.write_text(
            "<annotation>"
            "<object><name>1</name><bndbox><xmin>x</xmin><ymin>1</ymin>"
            "<xmax>2</xmax><ymax>3</ymax></bndbox></object>"
            "<object><name>2</name><bndbox><xmin>1</xmin><ymin>1</ymin>"
            "<xmax>2</xmax><ymax>3</ymax></bndbox></object>"
            "</annotation>",
            encoding="utf-8",
        )
        boxes, _ = parse_xml(xml)
        assert [b.name for b in boxes] == ["2"]

    def test_find_xml_handles_both_cases(self, tmp_path):
        img = tmp_path / "a.jpg"
        img.write_bytes(b"x")
        assert find_xml(img) is None
        (tmp_path / "a.XML").write_text("<annotation/>", encoding="utf-8")
        assert find_xml(img) is not None


class TestPlateText:
    def test_reads_left_to_right_by_xmin(self, tmp_path):
        shuffled = [CHAR_OBJECTS[i] for i in (3, 0, 4, 2, 1)]
        xml = make_voc_xml(shuffled, tmp_path / "a.xml")
        assert extract_plate_from_xml(xml) == "12ز81"

    def test_ignores_the_whole_plate_object(self, tmp_path):
        """Car annotations carry a plate box alongside the characters."""
        objects = [("کل ناحیه پلاک", 50, 50, 340, 250)] + CHAR_OBJECTS
        xml = make_voc_xml(objects, tmp_path / "a.xml")
        assert extract_plate_from_xml(xml) == "12ز81"

    def test_too_short_returns_none(self, tmp_path):
        xml = make_voc_xml([("1", 1, 1, 2, 2)], tmp_path / "a.xml")
        assert extract_plate_from_xml(xml) is None

    def test_no_objects_returns_none(self, tmp_path):
        xml = make_voc_xml([], tmp_path / "a.xml")
        assert extract_plate_from_xml(xml) is None


class TestPlateRegions:
    def test_prefers_explicit_plate_box(self, tmp_path):
        objects = [("کل ناحیه پلاک", 627, 770, 780, 816)] + CHAR_OBJECTS
        regions, _, source = plate_regions(make_voc_xml(objects, tmp_path / "a.xml"))
        assert source == "explicit"
        assert regions == [(627.0, 770.0, 780.0, 816.0)]

    def test_returns_every_plate_in_the_frame(self, tmp_path):
        """~14% of the car images contain more than one plate."""
        objects = [
            ("کل ناحیه پلاک", 100, 100, 200, 140),
            ("کل ناحیه پلاک", 400, 300, 520, 350),
        ]
        regions, _, source = plate_regions(make_voc_xml(objects, tmp_path / "a.xml"))
        assert source == "explicit"
        assert len(regions) == 2

    def test_falls_back_to_character_union(self, tmp_path):
        regions, _, source = plate_regions(make_voc_xml(CHAR_OBJECTS, tmp_path / "a.xml"))
        assert source == "union"
        assert len(regions) == 1
        x1, y1, x2, y2 = regions[0]
        assert x1 < 59 and y1 < 56 and x2 > 325 and y2 > 245  # padded outward

    def test_empty_annotation(self, tmp_path):
        regions, _, source = plate_regions(make_voc_xml([], tmp_path / "a.xml"))
        assert regions == [] and source == "none"

    def test_union_box_padding_is_proportional(self):
        box = union_box([type("B", (), {"xmin": 0, "ymin": 0, "xmax": 100, "ymax": 20})()], margin=0.1)
        assert box[0] == pytest.approx(-10.0)
        assert box[2] == pytest.approx(110.0)


class TestYoloLabels:
    def test_normalised_centre_and_size(self, tmp_path):
        out = tmp_path / "a.txt"
        write_yolo_label(out, [(100.0, 50.0, 300.0, 150.0)], (400, 200))
        cls, cx, cy, bw, bh = out.read_text().split()
        assert cls == "0"
        assert float(cx) == pytest.approx(0.5)
        assert float(cy) == pytest.approx(0.5)
        assert float(bw) == pytest.approx(0.5)
        assert float(bh) == pytest.approx(0.5)

    def test_boxes_are_clipped_in_pixel_space(self, tmp_path):
        """Clamping the normalised values independently describes a different box.

        A union box padded past the image edge used to yield centre 0.47 with
        width 1.0, which is not the annotated rectangle at all.
        """
        out = tmp_path / "a.txt"
        write_yolo_label(out, [(-40.0, -20.0, 440.0, 260.0)], (400, 200))
        _, cx, cy, bw, bh = map(float, out.read_text().split())
        assert cx == pytest.approx(0.5) and cy == pytest.approx(0.5)
        assert bw == pytest.approx(1.0) and bh == pytest.approx(1.0)

    def test_every_box_stays_inside_the_image(self, tmp_path):
        out = tmp_path / "a.txt"
        write_yolo_label(out, [(-50.0, -50.0, 100.0, 60.0), (350.0, 150.0, 500.0, 260.0)], (400, 200))
        for line in out.read_text().strip().splitlines():
            _, cx, cy, bw, bh = map(float, line.split())
            assert cx - bw / 2 >= -1e-6 and cx + bw / 2 <= 1 + 1e-6
            assert cy - bh / 2 >= -1e-6 and cy + bh / 2 <= 1 + 1e-6

    def test_one_line_per_plate(self, tmp_path):
        out = tmp_path / "a.txt"
        write_yolo_label(out, [(10.0, 10.0, 50.0, 30.0), (100.0, 60.0, 160.0, 90.0)], (400, 200))
        assert len(out.read_text().strip().splitlines()) == 2

    def test_degenerate_boxes_dropped(self, tmp_path):
        out = tmp_path / "a.txt"
        write_yolo_label(out, [(10.0, 10.0, 10.5, 10.5), (500.0, 500.0, 600.0, 600.0)], (400, 200))
        assert out.read_text().strip() == ""


class TestImageSize:
    def test_reads_jpeg_header(self, tmp_path):
        cv2 = pytest.importorskip("cv2")
        np = pytest.importorskip("numpy")
        path = tmp_path / "a.jpg"
        cv2.imwrite(str(path), np.zeros((123, 456, 3), dtype=np.uint8))
        assert image_size(path) == (456, 123)

    def test_reads_png_header(self, tmp_path):
        cv2 = pytest.importorskip("cv2")
        np = pytest.importorskip("numpy")
        path = tmp_path / "a.png"
        cv2.imwrite(str(path), np.zeros((77, 88, 3), dtype=np.uint8))
        assert image_size(path) == (88, 77)

    def test_unreadable_file_returns_none(self, tmp_path):
        path = tmp_path / "a.jpg"
        path.write_bytes(b"not an image")
        assert image_size(path) is None

    def test_missing_file_returns_none(self, tmp_path):
        assert image_size(tmp_path / "nope.jpg") is None
