from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


def imread_unicode(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """Read an image whose path may contain non-ASCII characters.

    cv2.imread goes through the platform's narrow-character API, so on Windows a
    Persian filename silently returns None. Every file in data/ocr_dataset is
    named after its plate, so this is the normal case here, not an edge case.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path: str | Path, image: np.ndarray, params: Optional[List[int]] = None) -> bool:
    """Write an image to a path that may contain non-ASCII characters."""
    path = Path(path)
    try:
        ok, buffer = cv2.imencode(path.suffix or ".jpg", image, params or [])
        if not ok:
            return False
        path.write_bytes(buffer.tobytes())
        return True
    except (cv2.error, OSError):
        return False

# An Iranian plate is roughly 4.5:1. Anything far outside this band is not a
# plate boundary, so a warp derived from it would be actively harmful.
MIN_PLATE_ASPECT = 1.8
MAX_PLATE_ASPECT = 8.0


def estimate_quality(image: np.ndarray) -> Dict[str, float]:
    """Cheap per-crop quality probe used to pick the right enhancement.

    Returns mean brightness (0-255), a blur score (variance of Laplacian, lower
    is blurrier) and the fraction of blown-out pixels caused by direct sun.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return {
        "brightness": float(gray.mean()),
        "blur": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "glare": float((gray >= 250).mean()),
        "contrast": float(gray.std()),
    }


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def adjust_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    if abs(gamma - 1.0) < 1e-3:
        return image
    inv = 1.0 / max(gamma, 1e-3)
    table = np.clip(((np.arange(256) / 255.0) ** inv) * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(image, table)


def normalize_illumination(image: np.ndarray, strength: float = 0.6) -> np.ndarray:
    """Flatten the low-frequency lighting gradient across the plate.

    Direct sun on one half of a plate and shade on the other is the single most
    common failure at a mine gate. Dividing by a heavily blurred copy removes
    that gradient while leaving the glyph edges intact.
    """
    bgr = _to_bgr(image)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0].astype(np.float32)

    ksize = max(3, (min(bgr.shape[:2]) // 2) | 1)
    background = cv2.GaussianBlur(l_channel, (ksize, ksize), 0)
    flattened = l_channel / np.maximum(background, 1.0) * float(np.mean(background))
    lab[:, :, 0] = np.clip(
        l_channel * (1.0 - strength) + flattened * strength, 0, 255
    ).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def suppress_glare(image: np.ndarray, threshold: int = 250) -> np.ndarray:
    """Inpaint specular highlights so blown-out glyphs can be reconstructed."""
    bgr = _to_bgr(image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray >= threshold).astype(np.uint8)
    if mask.mean() < 0.01 or mask.mean() > 0.5:
        # Nothing to fix, or so much is blown out that inpainting would invent
        # glyphs rather than restore them.
        return bgr
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)


def unsharp_mask(image: np.ndarray, amount: float = 1.0, radius: int = 3) -> np.ndarray:
    """Gentler and less ringing-prone than a hard 3x3 Laplacian kernel."""
    blurred = cv2.GaussianBlur(image, (0, 0), radius)
    return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)


def enhance_plate(
    image: np.ndarray,
    clip_limit: float = 3.0,
    bilateral_d: int = 9,
    sigma: int = 75,
    sharpen: bool = True,
    auto: bool = True,
) -> np.ndarray:
    """Condition a plate crop for OCR, adapting to dust, glare, night and blur.

    Returns a 3-channel BGR image. The previous version returned grayscale,
    which mismatched the colour images the recognizer is trained on.
    """
    bgr = _to_bgr(image)
    if bgr.size == 0:
        return bgr

    quality = estimate_quality(bgr) if auto else {}

    if auto and quality["glare"] > 0.02:
        bgr = suppress_glare(bgr)

    if auto and (quality["contrast"] < 50.0 or quality["glare"] > 0.02):
        # Dust haze and side-lighting both show up as low global contrast.
        bgr = normalize_illumination(bgr)

    if auto and quality["brightness"] < 80.0:
        # Night footage: lift the shadows before CLAHE so it has signal to work
        # with, scaling the lift by how dark the crop actually is.
        bgr = adjust_gamma(bgr, 1.0 + (80.0 - quality["brightness"]) / 80.0)
    elif auto and quality["brightness"] > 200.0:
        bgr = adjust_gamma(bgr, 0.7)

    # Denoise before CLAHE, otherwise CLAHE amplifies the sensor noise that night
    # and dusty scenes are full of. Bilateral keeps the glyph edges sharp.
    if bilateral_d > 0:
        bgr = cv2.bilateralFilter(bgr, bilateral_d, sigma, sigma)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    tile = max(2, min(8, min(bgr.shape[:2]) // 8))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if sharpen:
        # Scale sharpening to how blurry the crop is, so an already-crisp plate
        # is not over-sharpened into ringing artefacts.
        amount = 1.4 if auto and quality["blur"] < 100.0 else 0.7
        bgr = unsharp_mask(bgr, amount=amount)

    return bgr


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def deskew_plate(image: np.ndarray, max_angle: float = 25.0) -> np.ndarray:
    """Rotate out in-plane tilt using the orientation of the glyph mass."""
    bgr = _to_bgr(image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = cv2.findNonZero(binary)
    if coords is None or len(coords) < 20:
        return bgr

    # minAreaRect's angle convention has changed between OpenCV releases (it
    # reports (0, 90] on 4.5+ and [-90, 0) on 5.x), and which edge it calls the
    # width also varies. Folding into [-45, 45] gives the same tilt either way;
    # testing only `angle > 45` silently missed real tilts on some versions.
    angle = cv2.minAreaRect(coords)[-1]
    angle = (angle + 45.0) % 90.0 - 45.0
    if abs(angle) < 0.5 or abs(angle) > max_angle:
        return bgr

    h, w = bgr.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(bgr, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def correct_perspective(image: np.ndarray) -> np.ndarray:
    """Flatten a plate photographed off-axis, falling back to a plain deskew.

    The warp is only applied when the detected quadrilateral actually looks like
    a plate border. The previous version accepted any contour and mapped the
    corners in the wrong order, which rotated every crop by 90 degrees.
    """
    if image is None or image.size == 0:
        return image

    bgr = _to_bgr(image)
    h_img, w_img = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return deskew_plate(bgr)

    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    (rect_w, rect_h) = rect[1]
    if rect_w < 10 or rect_h < 10:
        return deskew_plate(bgr)

    # The quad must cover most of the crop; otherwise we locked onto a single
    # glyph or a bolt rather than the plate boundary.
    if (rect_w * rect_h) < 0.35 * (w_img * h_img):
        return deskew_plate(bgr)

    width = int(max(rect_w, rect_h))
    height = int(min(rect_w, rect_h))
    if height <= 0 or not (MIN_PLATE_ASPECT <= width / height <= MAX_PLATE_ASPECT):
        return deskew_plate(bgr)

    src = _order_points(cv2.boxPoints(rect).astype(np.float32))
    # Destination corners in the same TL, TR, BR, BL order as src.
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(bgr, matrix, (width, height), flags=cv2.INTER_CUBIC)
