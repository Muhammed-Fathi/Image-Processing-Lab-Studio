"""Image loading, saving and the bundled demonstration sample registry.

Images stay as NumPy arrays end to end (BGR for color, 2-D for gray),
which is the natural OpenCV representation.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

SAMPLES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "samples"
)

# Human readable description of each bundled sample (used by the UI and
# documented in the README). All samples are generated deterministically
# by scripts/generate_samples.py, so the project ships its own assets and
# never needs internet access at runtime.
SAMPLE_INFO: dict[str, str] = {
    "color_scene.png": "Synthetic color landscape (sky, sun, mountains, barn) - general RGB demos",
    "dark_scene.png": "Underexposed (dark) version of the scene - arithmetic enhancement demos",
    "midtone_gradient.png": "Smooth mid-tone gradient with shapes - negative / solarization demos",
    "color_cast.png": "Scene with a strong warm color cast - channel elimination demos",
    "low_contrast.png": "Scene compressed into a narrow intensity band - histogram enhancement demos",
    "detail_scene.png": "Scene with fine line detail - noise / sharpening demos (clean reference)",
    "geometric_edges.png": "High-contrast geometric shapes and text - edge detection demos",
}


def list_samples() -> list[str]:
    """Sorted list of sample file names currently present in assets/samples."""
    if not os.path.isdir(SAMPLES_DIR):
        return []
    return sorted(
        f
        for f in os.listdir(SAMPLES_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )


def sample_path(name: str) -> str:
    """Absolute path of a bundled sample image."""
    return os.path.join(SAMPLES_DIR, name)


def load_image_from_path(path: str) -> np.ndarray | None:
    """Load an image file as a BGR (or 2-D gray) uint8 array.

    Returns None when the file cannot be read (corrupt file, unsupported
    format, empty file, ...).
    """
    if not os.path.isfile(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 4:  # RGBA -> BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    if img.ndim == 3 and img.shape[2] != 3:
        return None
    if img.dtype != np.uint8:
        img = cv2.convertScaleAbs(img)
    return img


def load_image_from_bytes(data: bytes) -> np.ndarray | None:
    """Decode raw image bytes (from a file upload) with OpenCV.

    Returns None when the bytes are not a readable image.
    """
    arr = np.frombuffer(data, dtype=np.uint8)
    if arr.size == 0:
        return None
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    if img.ndim > 3 or (img.ndim == 3 and img.shape[2] != 3):
        return None
    if img.dtype != np.uint8:
        img = cv2.convertScaleAbs(img)
    return img


def is_valid_image_bytes(data: bytes) -> bool:
    """Quick Pillow-based sanity check of an upload before decoding.

    Pillow is used as a second, independent validator: it confirms the
    bytes really are a decodable image of a supported format.
    """
    import io

    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def encode_png(img: np.ndarray) -> bytes:
    """Encode an image as PNG bytes (used for the download buttons)."""
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


def ensure_color(bgr_or_gray: np.ndarray) -> np.ndarray:
    """Return a 3-channel BGR image, promoting gray images when needed."""
    if bgr_or_gray.ndim == 2:
        return cv2.cvtColor(bgr_or_gray, cv2.COLOR_GRAY2BGR)
    return bgr_or_gray


def to_gray(img: np.ndarray) -> np.ndarray:
    """Convert to 8-bit grayscale (per-channel luminance, standard weights)."""
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def odd_kernel(k: int, minimum: int = 3, maximum: int = 31) -> int:
    """Clamp `k` to a valid odd kernel size in [minimum, maximum]."""
    k = int(k)
    if k % 2 == 0:
        k += 1
    return max(minimum, min(maximum, k))
