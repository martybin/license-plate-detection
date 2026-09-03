"""Model package.

Submodules are exposed lazily (PEP 562). Importing `models.recognizer` used to
pull in `models.detector` through this file, and with it ultralytics -- so the
CRNN could not be loaded, tested or run without the whole YOLO stack present.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

_EXPORTS = {
    "Detection": "models.detector",
    "PlateDetector": "models.detector",
    "LPRPipeline": "models.pipeline",
    "PlateResult": "models.pipeline",
    "PlateVoter": "models.pipeline",
    "PlateRecognizer": "models.recognizer",
    "Recognition": "models.recognizer",
    "ResNetCRNN": "models.recognizer",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'models' has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_name), name)


def __dir__() -> list[str]:
    return __all__


if TYPE_CHECKING:
    from .detector import Detection, PlateDetector
    from .pipeline import LPRPipeline, PlateResult, PlateVoter
    from .recognizer import PlateRecognizer, Recognition, ResNetCRNN
