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


class TestSplitByPlate:
    """Validation must hold out whole plates, not individual files.

    prepare_dataset writes repeat sightings as `<plate>_2.jpg`. Splitting by file
    put near-identical photos of one vehicle on both sides, so the model scored
    itself on images it had effectively memorised.
    """

    def _samples(self, distinct=200, repeats=40, seed=0):
        rng = random.Random(seed)
        plates = [f"{11 + i % 89}ب{i % 1000:03d}{11 + i % 89}" for i in range(distinct)]
        plates = list(dict.fromkeys(plates))
        out = [(Path(f"{p}.jpg"), p) for p in plates]
        for i in range(repeats):
            p = plates[rng.randrange(len(plates))]
            out.append((Path(f"{p}_{i}.jpg"), p))
        return out

    def test_no_plate_on_both_sides(self):
        from training.train_recognizer import split_by_plate

        train, val = split_by_plate(self._samples(), 0.2, seed=1)
        assert set(l for _, l in train).isdisjoint(set(l for _, l in val))

    def test_every_file_is_used_exactly_once(self):
        from training.train_recognizer import split_by_plate

        samples = self._samples()
        train, val = split_by_plate(samples, 0.2, seed=1)
        assert len(train) + len(val) == len(samples)
        assert {p for p, _ in train} | {p for p, _ in val} == {p for p, _ in samples}

    def test_repeat_sightings_stay_together(self):
        from training.train_recognizer import split_by_plate

        samples = [
            (Path("11ب22233.jpg"), "11ب22233"),
            (Path("11ب22233_2.jpg"), "11ب22233"),
            (Path("11ب22233_3.jpg"), "11ب22233"),
            (Path("44ج55566.jpg"), "44ج55566"),
        ]
        train, val = split_by_plate(samples, 0.5, seed=0)
        for side in (train, val):
            labels = [l for _, l in side]
            if "11ب22233" in labels:
                assert labels.count("11ب22233") == 3

    def test_val_size_is_close_to_requested(self):
        from training.train_recognizer import split_by_plate

        samples = self._samples(distinct=500, repeats=100)
        _, val = split_by_plate(samples, 0.1, seed=2)
        assert 0.05 <= len(val) / len(samples) <= 0.2

    def test_deterministic_for_a_given_seed(self):
        from training.train_recognizer import split_by_plate

        samples = self._samples()
        a = split_by_plate(samples, 0.2, seed=7)
        b = split_by_plate(samples, 0.2, seed=7)
        assert [p.name for p, _ in a[1]] == [p.name for p, _ in b[1]]

    def test_single_plate_dataset_does_not_crash(self):
        from training.train_recognizer import split_by_plate

        samples = [(Path("11ب22233.jpg"), "11ب22233")]
        train, val = split_by_plate(samples, 0.5, seed=0)
        assert len(train) + len(val) == 1
