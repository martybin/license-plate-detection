"""Capture archive: one image per vehicle pass, off the hot path, bounded on disk."""
from __future__ import annotations

import csv
import os
import queue
import time

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from utils.plate_saver import PlateSaver  # noqa: E402

PLATE = "12ب34567"
OTHER = "34ج67890"


@pytest.fixture
def frame():
    return np.full((120, 200, 3), 60, dtype=np.uint8)


@pytest.fixture
def crop():
    return np.full((40, 140, 3), 200, dtype=np.uint8)


def make(tmp_path, **kwargs):
    kwargs.setdefault("min_interval_seconds", 0.5)
    return PlateSaver(tmp_path / "captures", **kwargs)


def images(saver):
    return sorted(p for p in saver.output_dir.rglob("*.jpg"))


def read_csv(saver):
    with saver.csv_path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


class TestWhatGetsWritten:
    def test_writes_full_frame_and_both_crops(self, tmp_path, frame, crop):
        saver = make(tmp_path)
        saver.save(PLATE, crop, crop, frame, (10, 20, 150, 60), 0.88, 0.91)
        saver.close()

        names = [p.name for p in images(saver)]
        assert len(names) == 3
        assert any(n.endswith("_full.jpg") for n in names), "the vehicle photo is the point"
        assert any(n.endswith("_raw.jpg") for n in names)
        assert any(n.endswith("_enh.jpg") for n in names)

    def test_plate_text_is_in_the_filename(self, tmp_path, frame, crop):
        saver = make(tmp_path)
        saver.save(PLATE, crop, crop, frame)
        saver.close()
        assert all(PLATE in p.name for p in images(saver))

    def test_files_land_in_a_day_folder(self, tmp_path, frame, crop):
        saver = make(tmp_path)
        saver.save(PLATE, crop, crop, frame)
        saver.close()
        day = images(saver)[0].parent
        assert day.parent == saver.output_dir
        assert len(day.name) == len("2026-01-01")

    def test_toggles_are_respected(self, tmp_path, frame, crop):
        saver = make(tmp_path, save_raw=False, save_enhanced=False)
        saver.save(PLATE, crop, crop, frame)
        saver.close()
        assert [p.name.split("_")[-1] for p in images(saver)] == ["full.jpg"]

    def test_full_frame_can_be_disabled(self, tmp_path, frame, crop):
        saver = make(tmp_path, save_full_frame=False)
        saver.save(PLATE, crop, crop, frame)
        saver.close()
        assert not any(p.name.endswith("_full.jpg") for p in images(saver))

    def test_written_image_is_decodable(self, tmp_path, frame, crop):
        """Round-trip through a Persian filename, which cv2.imread cannot open."""
        from utils.image_processing import imread_unicode

        saver = make(tmp_path)
        saver.save(PLATE, crop, crop, frame, (10, 20, 150, 60))
        saver.close()
        full = next(p for p in images(saver) if p.name.endswith("_full.jpg"))
        decoded = imread_unicode(full)
        assert decoded is not None and decoded.shape == frame.shape

    def test_csv_paths_use_forward_slashes(self, tmp_path, frame, crop):
        """The log should read identically on Windows and Linux."""
        saver = make(tmp_path)
        saver.save(PLATE, crop, crop, frame)
        saver.close()
        row = read_csv(saver)[0]
        assert "\\" not in row["full_frame"]

    def test_annotation_does_not_mutate_the_caller_frame(self, tmp_path, frame, crop):
        original = frame.copy()
        saver = make(tmp_path, annotate=True)
        saver.save(PLATE, crop, crop, frame, (10, 20, 150, 60))
        saver.close()
        assert np.array_equal(frame, original), "the live frame must not be drawn on"


class TestEventDeduplication:
    def test_one_capture_per_vehicle_pass(self, tmp_path, frame, crop):
        """The pipeline calls save() on every confirmed frame while the truck waits."""
        saver = make(tmp_path, min_interval_seconds=10.0)
        accepted = [saver.save(PLATE, crop, crop, frame) for _ in range(30)]
        saver.close()
        assert sum(accepted) == 1
        assert len(images(saver)) == 3

    def test_a_different_vehicle_is_its_own_event(self, tmp_path, frame, crop):
        saver = make(tmp_path, min_interval_seconds=10.0)
        assert saver.save(PLATE, crop, crop, frame) is True
        assert saver.save(OTHER, crop, crop, frame) is True
        saver.close()

    def test_same_plate_saves_again_after_the_gap(self, tmp_path, frame, crop):
        saver = make(tmp_path, min_interval_seconds=0.2)
        assert saver.save(PLATE, crop, crop, frame) is True
        time.sleep(0.25)
        assert saver.save(PLATE, crop, crop, frame) is True
        saver.close()

    def test_empty_plate_rejected(self, tmp_path, frame, crop):
        saver = make(tmp_path)
        assert saver.save("", crop, crop, frame) is False
        saver.close()

    def test_dedup_table_is_bounded(self, tmp_path, frame, crop):
        saver = make(tmp_path, min_interval_seconds=0.0)
        for i in range(5000):
            saver._is_new_event(f"plate{i}", time.monotonic())
        assert len(saver._last_seen) <= 4096
        saver.close()


class TestCsvLog:
    def test_header_and_row(self, tmp_path, frame, crop):
        saver = make(tmp_path)
        saver.save(PLATE, crop, crop, frame, (1, 2, 3, 4), ocr_conf=0.91, det_conf=0.88)
        saver.close()

        rows = read_csv(saver)
        assert len(rows) == 1
        row = rows[0]
        assert row["plate"] == PLATE
        assert float(row["ocr_conf"]) == pytest.approx(0.91)
        assert float(row["det_conf"]) == pytest.approx(0.88)

    def test_paths_in_csv_resolve(self, tmp_path, frame, crop):
        saver = make(tmp_path)
        saver.save(PLATE, crop, crop, frame)
        saver.close()
        row = read_csv(saver)[0]
        for key in ("full_frame", "plate_raw", "plate_enhanced"):
            assert (saver.output_dir / row[key]).exists(), key

    def test_appends_without_repeating_the_header(self, tmp_path, frame, crop):
        saver = make(tmp_path, min_interval_seconds=0.0)
        saver.save(PLATE, crop, crop, frame)
        saver.save(OTHER, crop, crop, frame)
        saver.close()
        assert len(read_csv(saver)) == 2


class TestRetention:
    def test_max_files_evicts_oldest(self, tmp_path, frame, crop):
        saver = make(tmp_path, min_interval_seconds=0.0, save_enhanced=False, max_files=6)
        for i in range(10):
            saver.save(f"1{i}ب34567", crop, None, frame)
        saver.close()
        assert len(images(saver)) <= 6

    def test_max_age_removes_stale_files(self, tmp_path, frame, crop):
        saver = make(tmp_path, min_interval_seconds=0.0, max_age_days=1.0)
        saver.save("11ب34567", crop, crop, frame)
        saver._queue.join()

        stale = time.time() - 3 * 86400
        for path in images(saver):
            os.utime(path, (stale, stale))

        saver.save("22ج34567", crop, crop, frame)
        saver.close()

        names = [p.name for p in images(saver)]
        assert not any("11ب34567" in n for n in names)
        assert any("22ج34567" in n for n in names)

    def test_retention_off_by_default(self, tmp_path, frame, crop):
        saver = make(tmp_path, min_interval_seconds=0.0)
        for i in range(8):
            saver.save(f"1{i}ب34567", crop, crop, frame)
        saver.close()
        assert len(images(saver)) == 24


class TestHotPath:
    def test_save_never_blocks_when_the_queue_is_full(self, tmp_path, frame, crop):
        """A slow disk must cost an audit image, never a dropped frame at the gate."""
        saver = make(tmp_path, min_interval_seconds=0.0, queue_size=2)
        saver.close()  # stop the writer thread so nothing drains the queue
        while True:  # jam it
            try:
                saver._queue.put_nowait(None)
            except queue.Full:
                break

        started = time.monotonic()
        results = [saver.save(f"9{i}ق12345", crop, crop, frame) for i in range(50)]
        elapsed = time.monotonic() - started

        assert elapsed < 0.5
        assert results.count(True) == 0
        assert saver.dropped == 50

    def test_close_drains_pending_writes(self, tmp_path, frame, crop):
        saver = make(tmp_path, min_interval_seconds=0.0)
        for i in range(5):
            saver.save(f"1{i}ب34567", crop, crop, frame)
        saver.close()
        assert saver.written == 5
        assert len(images(saver)) == 15

    def test_context_manager_closes(self, tmp_path, frame, crop):
        with make(tmp_path) as saver:
            saver.save(PLATE, crop, crop, frame)
        assert saver.written == 1
