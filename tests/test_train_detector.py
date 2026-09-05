"""Locating the checkpoint ultralytics actually produced."""
from __future__ import annotations

from pathlib import Path

import pytest

from training.train_detector import _count_images, _locate_best


class _Trainer:
    def __init__(self, best=None, save_dir=None):
        if best is not None:
            self.best = best
        if save_dir is not None:
            self.save_dir = save_dir


class _Model:
    def __init__(self, trainer=None):
        self.trainer = trainer


class _Results:
    def __init__(self, save_dir=None):
        if save_dir is not None:
            self.save_dir = save_dir


def make_run(root: Path, rel: str) -> Path:
    """Create <root>/<rel>/weights/best.pt and return the checkpoint path."""
    weights = root / rel / "weights"
    weights.mkdir(parents=True, exist_ok=True)
    best = weights / "best.pt"
    best.write_bytes(b"checkpoint")
    return best


class TestLocateBest:
    def test_prefers_the_trainer_path(self, tmp_path):
        best = make_run(tmp_path, "runs/detect/iran_plate")
        found = _locate_best(_Model(_Trainer(best=best)), _Results())
        assert found == best

    def test_falls_back_to_the_results_save_dir(self, tmp_path):
        best = make_run(tmp_path, "runs/detect/iran_plate")
        found = _locate_best(_Model(_Trainer()), _Results(save_dir=best.parent.parent))
        assert found == best

    def test_falls_back_to_the_trainer_save_dir(self, tmp_path):
        best = make_run(tmp_path, "runs/detect/iran_plate")
        found = _locate_best(_Model(_Trainer(save_dir=best.parent.parent)), _Results())
        assert found == best

    def test_finds_a_nested_run_directory(self, tmp_path, monkeypatch):
        """Ultralytics 8.4 resolves `project` against its own runs dir, producing
        runs/detect/runs/detect/iran_plate -- the hardcoded path missed it and the
        trained model was silently never copied into weights/."""
        best = make_run(tmp_path, "runs/detect/runs/detect/iran_plate")
        monkeypatch.chdir(tmp_path)
        found = _locate_best(_Model(None), _Results())
        assert found is not None
        assert found.resolve() == best.resolve()

    def test_picks_the_newest_when_several_runs_exist(self, tmp_path, monkeypatch):
        import os
        import time

        old = make_run(tmp_path, "runs/detect/iran_plate")
        new = make_run(tmp_path, "runs/detect/iran_plate2")
        past = time.time() - 3600
        os.utime(old, (past, past))
        monkeypatch.chdir(tmp_path)
        assert _locate_best(_Model(None), _Results()).resolve() == new.resolve()

    def test_returns_none_when_nothing_was_written(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert _locate_best(_Model(None), _Results()) is None

    def test_ignores_a_directory_without_weights(self, tmp_path, monkeypatch):
        (tmp_path / "runs" / "detect" / "iran_plate").mkdir(parents=True)
        monkeypatch.chdir(tmp_path)
        assert _locate_best(_Model(None), _Results()) is None


class TestCountImages:
    def test_counts_only_images(self, tmp_path):
        for name in ("a.jpg", "b.PNG", "c.jpeg", "d.bmp", "notes.txt", "labels.cache"):
            (tmp_path / name).write_bytes(b"x")
        assert _count_images(tmp_path) == 4

    def test_missing_folder_is_zero(self, tmp_path):
        assert _count_images(tmp_path / "nope") == 0
