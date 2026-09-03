"""Utility package.

Submodules are exposed lazily (PEP 562) so that importing a light one does not
drag in the heavy ones. `utils.plate_utils` and `utils.database` are pure stdlib;
eagerly re-exporting the image helpers here would force an OpenCV import on
anything that touches them -- including `tools/register_vehicle.py`, which only
adds rows to SQLite and has no use for OpenCV at all.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

_EXPORTS = {
    "VehicleDB": "utils.database",
    "correct_perspective": "utils.image_processing",
    "deskew_plate": "utils.image_processing",
    "enhance_plate": "utils.image_processing",
    "estimate_quality": "utils.image_processing",
    "normalize_illumination": "utils.image_processing",
    "suppress_glare": "utils.image_processing",
    "unsharp_mask": "utils.image_processing",
    "TextRenderer": "utils.overlay",
    "draw_panel": "utils.overlay",
    "shape_persian": "utils.overlay",
    "PlateSaver": "utils.plate_saver",
    "PlateParts": "utils.plate_utils",
    "format_plate_display": "utils.plate_utils",
    "is_valid_iran_plate": "utils.plate_utils",
    "normalize_iran_plate": "utils.plate_utils",
    "parse_iran_plate": "utils.plate_utils",
    "repair_plate": "utils.plate_utils",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'utils' has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_name), name)


def __dir__() -> list[str]:
    return __all__


if TYPE_CHECKING:  # keep editors and type checkers aware of the real symbols
    from .database import VehicleDB
    from .image_processing import (
        correct_perspective,
        deskew_plate,
        enhance_plate,
        estimate_quality,
        normalize_illumination,
        suppress_glare,
        unsharp_mask,
    )
    from .overlay import TextRenderer, draw_panel, shape_persian
    from .plate_saver import PlateSaver
    from .plate_utils import (
        PlateParts,
        format_plate_display,
        is_valid_iran_plate,
        normalize_iran_plate,
        parse_iran_plate,
        repair_plate,
    )
