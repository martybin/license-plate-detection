"""Restricting the plate letter at decode time, and balancing it at train time.

A site whose fleet is one plate class knows the letter before the model runs.
Letting the network pick a letter that cannot occur there throws that away: on
20 real mine photos the unconstrained model chose the letter wrong every time.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from models.recognizer import PlateRecognizer  # noqa: E402

# Defined locally, not imported from conftest: `tests` is not a package here and
# ultralytics ships a top-level package of that name that would shadow it.
CHARSET = "0123456789ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی"
NUM_CLASSES = len(CHARSET) + 1
BLANK = len(CHARSET)
AIN = CHARSET.index("ع")
JIM = CHARSET.index("ج")


def make_decoder(allowed: str = "") -> PlateRecognizer:
    """A PlateRecognizer with decode wiring only -- no checkpoint needed."""
    obj = PlateRecognizer.__new__(PlateRecognizer)
    obj.charset = CHARSET
    obj.idx_to_char = {i: c for i, c in enumerate(CHARSET)}
    obj.blank_idx = len(CHARSET)
    obj.allowed_letters = allowed
    obj._blocked_idx = [
        i for i, c in enumerate(CHARSET) if not c.isdigit() and allowed and c not in allowed
    ]
    return obj


def logits_for(frames, peak=10.0):
    """Build logits from a per-frame spec.

    Each frame is either a class index (one clear winner) or a {index: logit}
    mapping, which is how a real network behaves: the letter it prefers scores
    highest and the true letter is a close runner-up. That distinction matters --
    masking only removes classes, it does not promote a letter, so with a flat
    runner-up field the winner after masking is whatever index sorts first.
    """
    out = torch.full((1, len(frames), NUM_CLASSES), -10.0)
    for t, frame in enumerate(frames):
        if isinstance(frame, dict):
            for idx, value in frame.items():
                out[0, t, idx] = value
        else:
            out[0, t, frame] = peak
    return out


# The letter frame as a real network emits it: 'ج' preferred, 'ع' a close second.
LETTER_FRAME = {JIM: 10.0, AIN: 6.0}


class TestBlockedIndices:
    def test_empty_allowed_blocks_nothing(self):
        assert make_decoder("")._blocked_idx == []

    def test_digits_are_never_blocked(self):
        blocked = set(make_decoder("ع")._blocked_idx)
        for d in "0123456789":
            assert CHARSET.index(d) not in blocked

    def test_only_the_allowed_letter_survives(self):
        blocked = set(make_decoder("ع")._blocked_idx)
        assert AIN not in blocked
        assert JIM in blocked

    def test_multiple_allowed_letters(self):
        blocked = set(make_decoder("عت")._blocked_idx)
        assert AIN not in blocked and CHARSET.index("ت") not in blocked
        assert JIM in blocked


class TestDecoding:
    def test_unconstrained_returns_the_networks_choice(self):
        reading = make_decoder("")._ctc_decode(logits_for([1, 2, LETTER_FRAME, 3, 4, 5, 6, 7]))[0]
        assert reading.text == "12ج34567"

    def test_constrained_swaps_in_the_site_letter(self):
        """The exact failure seen on real mine photos: 12ج44964 -> 12ع44964."""
        reading = make_decoder("ع")._ctc_decode(logits_for([1, 2, LETTER_FRAME, 4, BLANK, 4, 9, 6, 4]))[0]
        assert reading.text == "12ع44964"

    def test_digits_are_untouched_by_the_constraint(self):
        free = make_decoder("")._ctc_decode(logits_for([1, 2, LETTER_FRAME, 3, 4, 5, 6, 7]))[0]
        held = make_decoder("ع")._ctc_decode(logits_for([1, 2, LETTER_FRAME, 3, 4, 5, 6, 7]))[0]
        assert [c for c in free.text if c.isdigit()] == [c for c in held.text if c.isdigit()]

    def test_confidence_is_renormalised_upward(self):
        """Masking impossible letters concentrates the probability mass."""
        seq = [1, 2, LETTER_FRAME, 4, BLANK, 4, 9, 6, 4]
        assert make_decoder("ع")._ctc_decode(logits_for(seq))[0].confidence > make_decoder(
            ""
        )._ctc_decode(logits_for(seq))[0].confidence

    def test_already_correct_letter_is_left_alone(self):
        reading = make_decoder("ع")._ctc_decode(logits_for([1, 2, AIN, 4, BLANK, 4, 9, 6, 4]))[0]
        assert reading.text == "12ع44964"

    def test_masking_does_not_invent_a_letter(self):
        """Masking removes rival letters; it does not force the letter position.

        If the network has no real preference for the allowed letter, a digit can
        win that frame instead. On the 20 real mine photos the allowed letter did
        win every time, but that is a property of the images, not a guarantee.
        """
        flat = [1, 2, {JIM: 10.0}, 3, 4, 5, 6, 7]  # no support for 'ع' at all
        reading = make_decoder("ع")._ctc_decode(logits_for(flat))[0]
        assert "ع" not in reading.text

    def test_no_nan_when_every_letter_is_blocked_in_a_frame(self):
        reading = make_decoder("ع")._ctc_decode(logits_for([1, 2, 3]))[0]
        assert reading.text == "123"
        assert not np.isnan(reading.confidence)


class TestLetterBalancedWeights:
    def _samples(self):
        from pathlib import Path

        # 20 plates with 'د', 2 with 'ع' -- the real imbalance, exaggerated
        out = [(Path(f"1{i%9}د2223{i%9}"), f"1{i%9}د2223{i%9}") for i in range(20)]
        out += [(Path(f"1{i}ع4496{i}"), f"1{i}ع4496{i}") for i in range(2)]
        return out

    def test_rare_letter_gets_more_weight(self):
        from training.train_recognizer import letter_balanced_weights

        samples = self._samples()
        w = letter_balanced_weights(samples, strength=1.0)
        rare = [w[i] for i, (_, l) in enumerate(samples) if "ع" in l]
        common = [w[i] for i, (_, l) in enumerate(samples) if "د" in l]
        assert min(rare) > max(common)

    def test_strength_zero_is_uniform(self):
        from training.train_recognizer import letter_balanced_weights

        w = letter_balanced_weights(self._samples(), strength=0.0)
        assert len(set(round(x, 9) for x in w)) == 1

    def test_boost_multiplies_the_named_letter(self):
        from training.train_recognizer import letter_balanced_weights

        samples = self._samples()
        plain = letter_balanced_weights(samples, 1.0, boost="")
        boosted = letter_balanced_weights(samples, 1.0, boost="ع")
        idx = next(i for i, (_, l) in enumerate(samples) if "ع" in l)
        assert boosted[idx] == pytest.approx(plain[idx] * 2.0)

    def test_expected_share_after_balancing(self):
        from training.train_recognizer import letter_balanced_weights

        samples = self._samples()
        w = letter_balanced_weights(samples, 1.0)
        total = sum(w)
        share = sum(w[i] for i, (_, l) in enumerate(samples) if "ع" in l) / total
        # 2 of 22 samples naturally (9%); balancing lifts it to roughly half
        assert 0.4 < share < 0.6


class TestPlateLetter:
    @pytest.mark.parametrize("text, expected", [
        ("12ع44964", "ع"), ("12ب34567", "ب"), ("12345678", ""), ("", ""),
    ])
    def test_extracts_the_single_letter(self, text, expected):
        from training.train_recognizer import plate_letter

        assert plate_letter(text) == expected
