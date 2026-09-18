"""Exercise precision options through constructor, warmup and real predict calls."""
from unittest.mock import MagicMock

import numpy as np
import pytest
import ultralytics.cfg

import models.detector as detector_module


@pytest.mark.parametrize("modern", [False, True])
@pytest.mark.parametrize("cuda,half", [(True, True), (True, False), (False, True)])
def test_precision_options(monkeypatch, modern, cuda, half):
    model = MagicMock()
    model.predict.return_value = []
    monkeypatch.setattr(detector_module, "YOLO", lambda _: model)
    monkeypatch.setattr(detector_module.torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(ultralytics.cfg, "DEFAULT_CFG_DICT", {"quantize": None} if modern else {"half": False})
    monkeypatch.setattr(detector_module.torch.backends.cudnn, "benchmark", False)
    detector = detector_module.PlateDetector("unused.pt", half=half)
    assert detector.detect(np.zeros((32, 32, 3), dtype=np.uint8)) == []
    assert model.predict.call_count == 2
    for call in model.predict.call_args_list:
        if cuda and half:
            key = "quantize" if modern else "half"
            assert call.kwargs[key] == ("fp16" if modern else True)
            assert ("half" if modern else "quantize") not in call.kwargs
        else:
            assert "quantize" not in call.kwargs and "half" not in call.kwargs


def test_installed_ultralytics_accepts_fp16():
    from ultralytics.cfg import DEFAULT_CFG_DICT, get_cfg

    modern = "quantize" in DEFAULT_CFG_DICT
    options = {"quantize": "fp16"} if modern else {"half": True}
    cfg = get_cfg(overrides=options)
    assert cfg.quantize == 16 if modern else cfg.half is True
