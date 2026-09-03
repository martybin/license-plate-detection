"""Recognizer geometry and CTC decoding.

The timestep tests are the important ones: a stock ResNet leaves only 8 CTC
timesteps for a 256px plate, which is fewer than the 8-9 characters an Iranian
plate carries, so the loss is unreachable and training silently never converges.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from models.recognizer import (  # noqa: E402
    ResNetCRNN,
    letterbox_plate,
)
from tests.conftest import CHARSET  # noqa: E402

NUM_CLASSES = len(CHARSET) + 1
LONGEST_LABEL = 9  # 2 digits + letter + 3 digits + 2-digit region


def timesteps(model, height, width):
    with torch.no_grad():
        return model(torch.zeros(1, 3, height, width)).shape[1]


@pytest.fixture(scope="module")
def model():
    return ResNetCRNN(num_classes=NUM_CLASSES, backbone="resnet18", pretrained=False).eval()


class TestCtcCapacity:
    @pytest.mark.parametrize("width, expected", [(128, 16), (192, 24), (256, 32)])
    def test_width_is_divided_by_eight(self, model, width, expected):
        assert timesteps(model, 64, width) == expected

    @pytest.mark.parametrize("width", [128, 192, 256])
    def test_enough_timesteps_for_a_full_plate(self, model, width):
        assert timesteps(model, 64, width) >= LONGEST_LABEL

    def test_configured_width_has_headroom(self, model):
        """CTC needs comfortably more timesteps than characters, not just equal."""
        import yaml
        from pathlib import Path

        cfg = yaml.safe_load(
            (Path(__file__).resolve().parent.parent / "configs" / "config.yaml").read_text(
                encoding="utf-8"
            )
        )
        rec = cfg["recognizer"]
        got = timesteps(model, rec["img_height"], rec["img_width"])
        assert got >= 2 * LONGEST_LABEL, f"only {got} timesteps for {LONGEST_LABEL} chars"

    def test_output_width_matches_class_count(self, model):
        with torch.no_grad():
            out = model(torch.zeros(2, 3, 64, 192))
        assert out.shape[0] == 2 and out.shape[2] == NUM_CLASSES


class TestBackbones:
    @pytest.mark.parametrize("backbone", ["resnet18", "resnet34"])
    def test_supported_backbones_agree_on_shape(self, backbone):
        net = ResNetCRNN(NUM_CLASSES, backbone=backbone, pretrained=False).eval()
        assert timesteps(net, 64, 192) == 24

    def test_unknown_backbone_rejected(self):
        with pytest.raises(ValueError, match="Unsupported backbone"):
            ResNetCRNN(NUM_CLASSES, backbone="resnet99", pretrained=False)


class TestLetterbox:
    def test_exact_target_size(self):
        out = letterbox_plate(np.zeros((30, 300, 3), dtype=np.uint8), 64, 192)
        assert out.shape[:2] == (64, 192)

    def test_aspect_ratio_preserved(self):
        """A 4:1 plate scaled into a 3:1 box must letterbox, not stretch."""
        image = np.zeros((50, 200, 3), dtype=np.uint8)
        image[:, :] = 255
        out = letterbox_plate(image, 64, 192)
        assert out.shape[:2] == (64, 192)

    @pytest.mark.parametrize("shape", [(22, 90, 3), (163, 350, 3), (317, 580, 3), (10, 46, 3)])
    def test_handles_the_real_crop_size_range(self, shape):
        out = letterbox_plate(np.zeros(shape, dtype=np.uint8), 64, 192)
        assert out.shape[:2] == (64, 192)

    def test_zero_sized_input(self):
        out = letterbox_plate(np.zeros((0, 0, 3), dtype=np.uint8), 64, 192)
        assert out.shape[:2] == (64, 192)


class TestCtcDecode:
    @pytest.fixture
    def decoder(self):
        """A PlateRecognizer with the decode logic but no checkpoint loaded."""
        from models.recognizer import PlateRecognizer

        obj = PlateRecognizer.__new__(PlateRecognizer)
        obj.charset = CHARSET
        obj.idx_to_char = {i: c for i, c in enumerate(CHARSET)}
        obj.blank_idx = len(CHARSET)
        return obj

    def _logits(self, indices, confidence=0.9):
        """One-hot-ish logits that argmax to `indices` with the given probability."""
        logits = torch.full((1, len(indices), NUM_CLASSES), 0.0)
        for t, idx in enumerate(indices):
            logits[0, t, :] = -10.0
            logits[0, t, idx] = 10.0
        return logits

    def test_collapses_repeats_and_drops_blanks(self, decoder):
        blank = len(CHARSET)
        # "12ب" spelled with repeats and blanks between them
        seq = [1, 1, blank, 2, blank, CHARSET.index("ب"), CHARSET.index("ب")]
        assert decoder._ctc_decode(self._logits(seq))[0].text == "12ب"

    def test_repeated_char_separated_by_blank_is_kept(self, decoder):
        blank = len(CHARSET)
        assert decoder._ctc_decode(self._logits([1, blank, 1]))[0].text == "11"

    def test_all_blank_yields_empty(self, decoder):
        blank = len(CHARSET)
        reading = decoder._ctc_decode(self._logits([blank] * 8))[0]
        assert reading.text == "" and reading.confidence == 0.0

    def test_confidence_is_between_zero_and_one(self, decoder):
        reading = decoder._ctc_decode(self._logits([1, 2, 3]))[0]
        assert 0.0 < reading.confidence <= 1.0

    def test_one_weak_glyph_drags_the_score_down(self, decoder):
        """Geometric mean: a single bad character must not be averaged away."""
        strong = torch.full((1, 3, NUM_CLASSES), -10.0)
        for t, idx in enumerate([1, 2, 3]):
            strong[0, t, idx] = 10.0

        weak = strong.clone()
        weak[0, 1, 2] = 0.1  # make the middle glyph uncertain

        assert decoder._ctc_decode(weak)[0].confidence < decoder._ctc_decode(strong)[0].confidence

    def test_decodes_a_whole_batch(self, decoder):
        logits = torch.cat([self._logits([1, 2, 3]), self._logits([4, 5, 6])], dim=0)
        readings = decoder._ctc_decode(logits)
        assert [r.text for r in readings] == ["123", "456"]
