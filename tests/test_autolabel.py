"""Turning unlabelled site photos into training data."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("torch")

from tools.autolabel_plates import (  # noqa: E402
    cmd_import,
    next_free_name,
    read_with_variants,
    seed_counter,
)
from models.recognizer import Recognition  # noqa: E402

PLATE = "12ع44964"


def write_img(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", np.full((60, 240, 3), 200, dtype=np.uint8))
    assert ok
    path.write_bytes(buf.tobytes())
    return path


class Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class TestFileNaming:
    def test_first_copy_is_the_bare_plate(self, tmp_path):
        counter = defaultdict(int)
        assert next_free_name(tmp_path, PLATE, ".jpg", counter).name == f"{PLATE}.jpg"

    def test_repeats_get_the_underscore_suffix(self, tmp_path):
        counter = defaultdict(int)
        names = [next_free_name(tmp_path, PLATE, ".jpg", counter).name for _ in range(3)]
        assert names == [f"{PLATE}.jpg", f"{PLATE}_2.jpg", f"{PLATE}_3.jpg"]

    def test_never_overwrites_an_existing_file(self, tmp_path):
        write_img(tmp_path / f"{PLATE}.jpg")
        write_img(tmp_path / f"{PLATE}_2.jpg")
        counter = defaultdict(int)
        assert next_free_name(tmp_path, PLATE, ".jpg", counter).exists() is False

    def test_seed_counter_reads_the_existing_dataset(self, tmp_path):
        write_img(tmp_path / f"{PLATE}.jpg")
        write_img(tmp_path / f"{PLATE}_2.jpg")
        write_img(tmp_path / "34ج67890.jpg")
        counter = seed_counter(tmp_path)
        assert counter[PLATE] == 2 and counter["34ج67890"] == 1

    def test_seed_counter_on_missing_dir(self, tmp_path):
        assert seed_counter(tmp_path / "nope") == {}

    def test_continues_numbering_after_a_previous_run(self, tmp_path):
        write_img(tmp_path / f"{PLATE}.jpg")
        write_img(tmp_path / f"{PLATE}_2.jpg")
        counter = seed_counter(tmp_path)
        assert next_free_name(tmp_path, PLATE, ".jpg", counter).name == f"{PLATE}_3.jpg"


class FakeRecognizer:
    def __init__(self, readings):
        self.readings = readings

    def recognize_batch(self, images):
        return list(self.readings)[: len(images)]


class TestReadWithVariants:
    def _crop(self):
        return np.full((60, 240, 3), 200, dtype=np.uint8)

    def test_reports_agreement_across_variants(self):
        rec = FakeRecognizer([Recognition(PLATE, 0.95)] * 3)
        reading, agreement = read_with_variants(rec, self._crop(), {})
        assert reading.text == PLATE and agreement >= 2

    def test_disagreement_is_visible(self):
        rec = FakeRecognizer(
            [Recognition(PLATE, 0.95), Recognition("34ج67890", 0.90), Recognition("55د11122", 0.80)]
        )
        _, agreement = read_with_variants(rec, self._crop(), {})
        assert agreement == 1

    def test_valid_plate_beats_confident_garbage(self):
        rec = FakeRecognizer(
            [Recognition("XXXX", 0.99), Recognition(PLATE, 0.70), Recognition("YY", 0.98)]
        )
        reading, _ = read_with_variants(rec, self._crop(), {})
        assert reading.text == PLATE

    def test_all_unreadable(self):
        reading, agreement = read_with_variants(FakeRecognizer([None, None, None]), self._crop(), {})
        assert reading is None and agreement == 0


class TestImportCorrections:
    def _csv(self, tmp_path, rows):
        path = tmp_path / "review.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["file", "suggested_plate", "corrected_plate"])
            w.writeheader()
            w.writerows(rows)
        return path

    def test_imports_corrected_rows(self, tmp_path, capsys):
        img = write_img(tmp_path / "raw" / "17235223248496-1.jpg")
        out = tmp_path / "ocr"
        csv_path = self._csv(tmp_path, [{"file": str(img), "suggested_plate": "", "corrected_plate": PLATE}])

        cmd_import(Args(csv_path=str(csv_path), out=str(out), dry_run=False))

        assert (out / f"{PLATE}.jpg").exists()
        assert "imported 1" in capsys.readouterr().out

    def test_blank_corrections_are_skipped(self, tmp_path, capsys):
        img = write_img(tmp_path / "raw" / "a.jpg")
        out = tmp_path / "ocr"
        csv_path = self._csv(tmp_path, [{"file": str(img), "suggested_plate": PLATE, "corrected_plate": ""}])

        cmd_import(Args(csv_path=str(csv_path), out=str(out), dry_run=False))

        assert not list(out.glob("*.jpg")) if out.exists() else True
        assert "left blank 1" in capsys.readouterr().out

    def test_invalid_plate_is_rejected(self, tmp_path, capsys):
        img = write_img(tmp_path / "raw" / "a.jpg")
        out = tmp_path / "ocr"
        csv_path = self._csv(tmp_path, [{"file": str(img), "suggested_plate": "", "corrected_plate": "NONSENSE"}])

        cmd_import(Args(csv_path=str(csv_path), out=str(out), dry_run=False))
        assert "skipped 1" in capsys.readouterr().out

    def test_persian_digits_are_normalised_on_import(self, tmp_path):
        """An operator typing on a Persian keyboard must not be punished for it."""
        img = write_img(tmp_path / "raw" / "a.jpg")
        out = tmp_path / "ocr"
        csv_path = self._csv(tmp_path, [{"file": str(img), "suggested_plate": "", "corrected_plate": "۱۲ع۴۴۹۶۴"}])

        cmd_import(Args(csv_path=str(csv_path), out=str(out), dry_run=False))
        assert (out / f"{PLATE}.jpg").exists()

    def test_missing_source_file_is_skipped(self, tmp_path, capsys):
        out = tmp_path / "ocr"
        csv_path = self._csv(tmp_path, [{"file": str(tmp_path / "gone.jpg"), "suggested_plate": "", "corrected_plate": PLATE}])

        cmd_import(Args(csv_path=str(csv_path), out=str(out), dry_run=False))
        assert "skipped 1" in capsys.readouterr().out

    def test_dry_run_writes_nothing(self, tmp_path):
        img = write_img(tmp_path / "raw" / "a.jpg")
        out = tmp_path / "ocr"
        csv_path = self._csv(tmp_path, [{"file": str(img), "suggested_plate": "", "corrected_plate": PLATE}])

        cmd_import(Args(csv_path=str(csv_path), out=str(out), dry_run=True))
        assert not out.exists() or not list(out.glob("*.jpg"))

    def test_csv_without_required_columns(self, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("a,b\n1,2\n", encoding="utf-8")
        with pytest.raises(SystemExit, match="corrected_plate"):
            cmd_import(Args(csv_path=str(path), out=str(tmp_path / "o"), dry_run=False))

    def test_missing_csv(self, tmp_path):
        with pytest.raises(SystemExit, match="not found"):
            cmd_import(Args(csv_path=str(tmp_path / "nope.csv"), out=str(tmp_path), dry_run=False))
