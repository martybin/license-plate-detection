"""Plate conditioning: enhancement, deskew and perspective correction."""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from utils.image_processing import (  # noqa: E402
    adjust_gamma,
    correct_perspective,
    deskew_plate,
    enhance_plate,
    estimate_quality,
    imread_unicode,
    imwrite_unicode,
    normalize_illumination,
    suppress_glare,
    unsharp_mask,
)


def synthetic_plate(width=240, height=60, bg=235, fg=20):
    """A light plate with dark blocks standing in for glyphs."""
    image = np.full((height, width, 3), bg, dtype=np.uint8)
    for x in range(20, width - 20, 28):
        image[height // 4 : 3 * height // 4, x : x + 16] = fg
    return image


class TestQuality:
    def test_reports_brightness_and_contrast(self):
        dark = np.full((40, 100, 3), 20, dtype=np.uint8)
        bright = np.full((40, 100, 3), 240, dtype=np.uint8)
        assert estimate_quality(dark)["brightness"] < 30
        assert estimate_quality(bright)["brightness"] > 230

    def test_detects_glare(self):
        image = synthetic_plate()
        image[:, :120] = 255
        assert estimate_quality(image)["glare"] > 0.3

    def test_blur_score_drops_when_blurred(self):
        sharp = synthetic_plate()
        blurred = cv2.GaussianBlur(sharp, (0, 0), 3)
        assert estimate_quality(blurred)["blur"] < estimate_quality(sharp)["blur"]

    def test_accepts_grayscale(self):
        gray = np.full((40, 100), 128, dtype=np.uint8)
        assert estimate_quality(gray)["brightness"] == pytest.approx(128, abs=1)


class TestEnhance:
    def test_returns_three_channel_bgr(self):
        """Training reads colour images; a grayscale return would mismatch it."""
        out = enhance_plate(synthetic_plate())
        assert out.ndim == 3 and out.shape[2] == 3
        assert out.dtype == np.uint8

    def test_accepts_grayscale_input(self):
        out = enhance_plate(np.full((40, 100), 128, dtype=np.uint8))
        assert out.ndim == 3 and out.shape[2] == 3

    def test_preserves_size(self):
        image = synthetic_plate(200, 50)
        assert enhance_plate(image).shape[:2] == image.shape[:2]

    def test_empty_input_is_survivable(self):
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        assert enhance_plate(empty).size == 0

    def test_lifts_a_dark_plate(self):
        dark = (synthetic_plate().astype(np.float32) * 0.2).astype(np.uint8)
        assert estimate_quality(enhance_plate(dark))["brightness"] > estimate_quality(dark)["brightness"]

    def test_improves_contrast_on_a_hazy_plate(self):
        hazy = cv2.addWeighted(synthetic_plate(), 0.35, np.full((60, 240, 3), 140, np.uint8), 0.65, 0)
        out = enhance_plate(hazy)
        assert estimate_quality(out)["contrast"] > estimate_quality(hazy)["contrast"]

    def test_auto_off_still_works(self):
        out = enhance_plate(synthetic_plate(), auto=False)
        assert out.ndim == 3 and out.shape[2] == 3

    def test_tiny_crop_does_not_crash(self):
        assert enhance_plate(np.full((6, 12, 3), 128, dtype=np.uint8)).shape[:2] == (6, 12)


class TestPrimitives:
    def test_gamma_below_one_darkens(self):
        image = np.full((10, 10, 3), 128, dtype=np.uint8)
        assert adjust_gamma(image, 0.5).mean() < image.mean()

    def test_gamma_above_one_brightens(self):
        image = np.full((10, 10, 3), 128, dtype=np.uint8)
        assert adjust_gamma(image, 2.0).mean() > image.mean()

    def test_gamma_one_is_identity(self):
        image = synthetic_plate()
        assert np.array_equal(adjust_gamma(image, 1.0), image)

    def test_unsharp_increases_local_contrast(self):
        blurred = cv2.GaussianBlur(synthetic_plate(), (0, 0), 2)
        assert estimate_quality(unsharp_mask(blurred, 1.5))["blur"] > estimate_quality(blurred)["blur"]

    def test_glare_suppression_reduces_blown_pixels(self):
        image = synthetic_plate()
        image[10:30, 40:90] = 255
        assert estimate_quality(suppress_glare(image))["glare"] < estimate_quality(image)["glare"]

    def test_glare_suppression_skips_when_mostly_blown(self):
        """Inpainting a fully blown plate would invent glyphs rather than restore them."""
        image = np.full((60, 240, 3), 255, dtype=np.uint8)
        assert np.array_equal(suppress_glare(image), image)

    def test_illumination_flattening_reduces_the_gradient(self):
        image = synthetic_plate()
        ramp = np.linspace(-70, 70, image.shape[1], dtype=np.float32)
        lit = np.clip(image.astype(np.float32) + ramp[None, :, None], 0, 255).astype(np.uint8)

        def halves_gap(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            half = gray.shape[1] // 2
            return abs(gray[:, :half].mean() - gray[:, half:].mean())

        assert halves_gap(normalize_illumination(lit)) < halves_gap(lit)


class TestGeometry:
    def test_deskew_straightens_a_rotated_plate(self):
        image = synthetic_plate()
        h, w = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 8.0, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

        def tilt(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            angle = cv2.minAreaRect(cv2.findNonZero(binary))[-1]
            # minAreaRect's convention differs across OpenCV versions; fold into
            # [-45, 45] so a straight plate reads as 0 either way.
            return abs((angle + 45.0) % 90.0 - 45.0)

        assert tilt(rotated) > 5.0, "the fixture should actually be tilted"
        assert tilt(deskew_plate(rotated)) < 1.0

    def test_deskew_leaves_a_straight_plate_alone(self):
        image = synthetic_plate()
        assert deskew_plate(image).shape == image.shape

    def test_perspective_does_not_rotate_by_90_degrees(self):
        """The destination corners were ordered differently from the source ones,
        which turned every crop on its side."""
        image = synthetic_plate(240, 60)
        out = correct_perspective(image)
        assert out.shape[1] > out.shape[0], "a plate must stay wider than it is tall"

    def test_perspective_keeps_a_plate_like_aspect(self):
        image = synthetic_plate(240, 60)
        out = correct_perspective(image)
        assert 1.5 < out.shape[1] / out.shape[0] < 9.0

    def test_perspective_rejects_a_noise_contour(self):
        """With nothing plate-shaped to lock onto it must fall back, not warp."""
        rng = np.random.default_rng(0)
        noise = rng.integers(0, 255, (60, 240, 3), dtype=np.uint8)
        out = correct_perspective(noise)
        assert out.shape[1] > out.shape[0]

    def test_empty_input_returned_unchanged(self):
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        assert correct_perspective(empty).size == 0


class TestUnicodeIo:
    """Every dataset file and every capture is named after a Persian plate."""

    def test_round_trip_through_a_persian_filename(self, tmp_path):
        path = tmp_path / "12ب34567.jpg"
        image = synthetic_plate()
        assert imwrite_unicode(path, image)
        assert path.exists(), "the file must be at the exact path requested"
        decoded = imread_unicode(path)
        assert decoded is not None and decoded.shape == image.shape

    def test_plain_imread_cannot_do_this_on_windows(self, tmp_path):
        """Documents why the helpers exist; cv2 may succeed on Linux."""
        path = tmp_path / "34ج67890.jpg"
        imwrite_unicode(path, synthetic_plate())
        assert path.exists()
        assert imread_unicode(path) is not None

    def test_read_missing_file_returns_none(self, tmp_path):
        assert imread_unicode(tmp_path / "nope.jpg") is None

    def test_read_corrupt_file_returns_none(self, tmp_path):
        path = tmp_path / "bad.jpg"
        path.write_bytes(b"not an image")
        assert imread_unicode(path) is None

    def test_read_empty_file_returns_none(self, tmp_path):
        path = tmp_path / "empty.jpg"
        path.write_bytes(b"")
        assert imread_unicode(path) is None

    def test_write_to_missing_directory_fails_cleanly(self, tmp_path):
        assert imwrite_unicode(tmp_path / "nope" / "a.jpg", synthetic_plate()) is False
