from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:  # optional, but required for correctly joined Persian glyphs
    import arabic_reshaper
    from bidi.algorithm import get_display

    _HAS_SHAPER = True
except ImportError:  # pragma: no cover - depends on deployment environment
    _HAS_SHAPER = False

# Fonts that actually carry the Persian glyph set, in preference order.
_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/vazir/Vazir.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "C:/Windows/Fonts/tahoma.ttf",
    "C:/Windows/Fonts/arial.ttf",
)

TextItem = Tuple[str, Tuple[int, int], int, Tuple[int, int, int]]


def shape_persian(text: str) -> str:
    """Join and reorder Persian text for rendering.

    Without reshaping, Persian renders as disconnected, left-to-right letters.
    """
    if not _HAS_SHAPER or not text:
        return text
    return get_display(arabic_reshaper.reshape(text))


class TextRenderer:
    """Draws Persian and Latin text onto BGR frames via PIL.

    cv2.putText only handles ASCII, so every Persian driver name in the registry
    previously rendered as a row of question marks on the gate monitor.
    """

    def __init__(self, font_path: Optional[str] = None) -> None:
        self.font_path = self._resolve_font(font_path)
        self._fonts: Dict[int, ImageFont.FreeTypeFont] = {}
        if self.font_path is None:
            warnings.warn(
                "No Unicode font found; Persian text will not render. "
                "Install a Persian font and set display.font_path in config.yaml.",
                RuntimeWarning,
            )
        elif not _HAS_SHAPER:
            warnings.warn(
                "arabic_reshaper/python-bidi not installed; Persian letters will "
                "render disconnected. Install them for correct shaping.",
                RuntimeWarning,
            )

    @staticmethod
    def _resolve_font(font_path: Optional[str]) -> Optional[str]:
        if font_path and Path(font_path).exists():
            return font_path
        for candidate in _FONT_CANDIDATES:
            if Path(candidate).exists():
                return candidate
        return None

    def _font(self, size: int) -> Optional[ImageFont.FreeTypeFont]:
        if self.font_path is None:
            return None
        if size not in self._fonts:
            self._fonts[size] = ImageFont.truetype(self.font_path, size)
        return self._fonts[size]

    def render(self, frame: np.ndarray, items: Iterable[TextItem]) -> np.ndarray:
        """Draw every text item in one pass.

        Converting between OpenCV and PIL is the expensive part, so all lines for
        a frame are drawn inside a single conversion rather than one per line.
        """
        items = list(items)
        if not items:
            return frame
        if self.font_path is None:
            return self._render_ascii_fallback(frame, items)

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        for text, (x, y), size, color in items:
            font = self._font(size)
            # PIL takes RGB; the caller works in OpenCV's BGR.
            draw.text((x, y), shape_persian(text), font=font, fill=(color[2], color[1], color[0]))
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _render_ascii_fallback(frame: np.ndarray, items: List[TextItem]) -> np.ndarray:
        for text, (x, y), size, color in items:
            cv2.putText(
                frame,
                text.encode("ascii", "replace").decode("ascii"),
                (x, y + size),
                cv2.FONT_HERSHEY_SIMPLEX,
                size / 30.0,
                color,
                2,
                cv2.LINE_AA,
            )
        return frame


def draw_panel(
    frame: np.ndarray,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    color: Tuple[int, int, int] = (0, 0, 0),
    alpha: float = 0.55,
) -> np.ndarray:
    """Translucent backdrop so text stays legible over a bright mine scene."""
    x1, y1 = top_left
    x2, y2 = bottom_right
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return frame
    region = frame[y1:y2, x1:x2]
    overlay = np.full_like(region, color, dtype=np.uint8)
    frame[y1:y2, x1:x2] = cv2.addWeighted(overlay, alpha, region, 1 - alpha, 0)
    return frame
