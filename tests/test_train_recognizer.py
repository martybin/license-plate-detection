"""Dataset loading and the harsh-condition augmentation."""
from __future__ import annotations

import random
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
torch = pytest.importorskip("torch")

from training.train_recognizer import (  # noqa: E402
    PlateOCRDataset,
    augment_plate,
    label_from_filename,
    scan_samples,
)

CHARSET = "0123456789ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی"


def write_plate(path: Path, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    img = np.full((60, 240, 3), 220, dtype=np.uint8)
    for x in range(20, 220, 28):
        img[15:45, x : x + 16] = int(rng.integers(0, 80))
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img)
    path.write_bytes(buf.tobytes())
    assert ok
    return path


class TestLabelFromFilename:
    @pytest.mark.parametrize(
        "stem, expected",
        [
            ("12ب34567", "12ب34567"),
            ("12ب34567_2", "12ب34567"),   # duplicate suffix stripped
            ("12ب34567_15", "12ب34567"),
            ("11الف12898", "11ا12898"),
        ],
    )
    def test_recovers_the_plate(self, tmp_path, stem, expected):
        assert label_from_filename(tmp_path / f"{stem}.jpg") == expected

    def test_keeps_a_non_numeric_suffix(self, tmp_path):
        assert label_from_filename(tmp_path / "12ب34567_x.jpg") == "12ب34567"


class TestScanSamples:
    def test_accepts_duplicates_that_the_old_loader_dropped(self, tmp_path):
        write_plate(tmp_path / "12ب34567.jpg")
        write_plate(tmp_path / "12ب34567_2.jpg")
        write_plate(tmp_path / "34ج67890.jpg")
        samples = scan_samples(tmp_path, CHARSET)
        assert len(samples) == 3
        assert [lbl for _, lbl in samples].count("12ب34567") == 2

    def test_rejects_labels_outside_the_charset(self, tmp_path):
        write_plate(tmp_path / "12ب34567.jpg")
        write_plate(tmp_path / "HELLO.jpg")
        assert len(scan_samples(tmp_path, CHARSET)) == 1


class TestAugmentation:
    def test_returns_a_same_shaped_uint8_image(self):
        img = np.full((60, 240, 3), 200, dtype=np.uint8)
        out = augment_plate(img, random.Random(0))
        assert out.shape == img.shape and out.dtype == np.uint8

    def test_different_draws_give_different_images(self):
        img = np.full((60, 240, 3), 200, dtype=np.uint8)
        rng = random.Random(1)
        seen = {augment_plate(img, rng).tobytes() for _ in range(12)}
        assert len(seen) > 1, "augmentation must actually vary"

    def test_tiny_crop_survives(self):
        img = np.full((12, 40, 3), 180, dtype=np.uint8)
        assert augment_plate(img, random.Random(3)).shape == img.shape


class TestAugmentationVariesAcrossEpochs:
    """The whole point of augmentation is a *different* view each epoch.

    Seeding from the sample index made it a pure function of that index, so all
    60 epochs trained on one fixed dusty/blurred variant per plate.
    """

    def _epoch(self, ds, indices):
        return tuple(round(float(ds[i][0].sum()), 1) for i in indices)

    def test_same_sample_differs_between_passes(self, tmp_path):
        for i in range(3):
            write_plate(tmp_path / f"1{i}ب3456{i}.jpg", seed=i)
        ds = PlateOCRDataset(scan_samples(tmp_path, CHARSET), charset=CHARSET,
                             img_height=64, img_width=192, augment=True)
        indices = list(range(len(ds)))
        passes = {self._epoch(ds, indices) for _ in range(4)}
        assert len(passes) > 1, "every epoch produced identical augmented images"

    def test_augment_disabled_is_deterministic(self, tmp_path):
        write_plate(tmp_path / "12ب34567.jpg")
        ds = PlateOCRDataset(scan_samples(tmp_path, CHARSET), charset=CHARSET,
                             img_height=64, img_width=192, augment=False)
        assert float(ds[0][0].sum()) == float(ds[0][0].sum())


class TestDatasetItem:
    def test_shapes_and_target(self, tmp_path):
        write_plate(tmp_path / "12ب34567.jpg")
        ds = PlateOCRDataset(scan_samples(tmp_path, CHARSET), charset=CHARSET,
                             img_height=64, img_width=192, augment=False)
        image, target, length = ds[0]
        assert image.shape == (3, 64, 192)
        assert length == 8 and target.numel() == 8

    def test_unreadable_file_warns_and_continues(self, tmp_path):
        bad = tmp_path / "12ب34567.jpg"
        bad.write_bytes(b"not an image")
        ds = PlateOCRDataset([(bad, "12ب34567")], charset=CHARSET,
                             img_height=64, img_width=192, augment=False)
        with pytest.warns(RuntimeWarning):
            image, _, _ = ds[0]
        assert image.shape == (3, 64, 192)
        assert ds.read_failures == 1
