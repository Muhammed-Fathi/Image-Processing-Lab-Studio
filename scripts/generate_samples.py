"""Generate all bundled demonstration images for Image Processing Lab Studio.

Run from the project root:

    python scripts/generate_samples.py

Every image is created programmatically (fixed drawing code, no random
components), so the assets are 100 % reproducible and owned by this
project - no external downloads, no third-party licenses to track.
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.image_operations import make_low_contrast  # noqa: E402

OUT_DIR = os.path.join(ROOT, "assets", "samples")
W, H = 640, 400
HORIZON = 240


def gradient_band(img: np.ndarray, y0: int, y1: int, top: np.ndarray, bot: np.ndarray) -> None:
    """Fill rows [y0, y1) of img with a vertical linear gradient (BGR)."""
    for y in range(y0, y1):
        t = (y - y0) / max(1, (y1 - y0) - 1)
        img[y, :] = np.clip(top + (bot - top) * t, 0, 255).astype(np.uint8)


def base_scene() -> np.ndarray:
    """A small synthetic landscape: sky, sun, mountains, barn, trees."""
    img = np.zeros((H, W, 3), dtype=np.uint8)

    # Sky: cool blue at the top -> warm light near the horizon
    gradient_band(img, 0, HORIZON, np.array([235, 190, 140]), np.array([250, 228, 195]))
    # Ground: two shades of green
    gradient_band(img, HORIZON, H, np.array([115, 160, 85]), np.array([60, 118, 52]))

    # Sun
    cv2.circle(img, (520, 70), 34, (90, 220, 255), -1)

    # Mountains with snow caps
    cv2.fillPoly(img, [np.array([[0, HORIZON], [140, 92], [300, HORIZON]], np.int32)], (125, 132, 112))
    cv2.fillPoly(img, [np.array([[180, HORIZON], [360, 122], [560, HORIZON]], np.int32)], (95, 112, 100))
    cv2.fillPoly(img, [np.array([[140, 92], [118, 118], [163, 118]], np.int32)], (245, 245, 245))
    cv2.fillPoly(img, [np.array([[360, 122], [338, 148], [383, 148]], np.int32)], (245, 245, 245))

    # Barn
    cv2.rectangle(img, (120, 280), (260, 360), (58, 52, 150), -1)
    cv2.fillPoly(img, [np.array([[110, 280], [190, 235], [270, 280]], np.int32)], (45, 42, 95))
    cv2.rectangle(img, (175, 315), (215, 360), (48, 42, 85), -1)

    # Trees
    for x in (330, 400, 470):
        cv2.rectangle(img, (x - 6, 320), (x + 6, 360), (72, 62, 42), -1)
        cv2.circle(img, (x, 300), 28, (125, 152, 52), -1)
    return img


def detail_scene() -> np.ndarray:
    """Scene plus fine line detail (fence, small shapes) for the
    noise-injection and sharpening experiments - the clean reference."""
    img = base_scene().copy()
    # Wooden fence: three thin horizontal rails
    for y in (368, 376, 384):
        cv2.line(img, (320, y), (640, y), (200, 180, 160), 2)
    for x in range(340, 640, 40):
        cv2.line(img, (x, 362), (x, 392), (150, 130, 110), 3)
    # Fine detail: small shapes and text in the sky
    cv2.rectangle(img, (60, 40), (95, 65), (60, 60, 60), 1)
    cv2.circle(img, (130, 52), 14, (220, 120, 60), 2)
    cv2.putText(img, "DETAIL TEST", (60, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2)
    for i, x in enumerate(range(60, 260, 20)):
        cv2.line(img, (x, 160), (x + 12, 185), (30, 30, 200), 1)
    return img


def midtone_gradient() -> np.ndarray:
    """Horizontal mid-tone gradient (40 -> 200) with mid-level shapes.

    Ideal for negative / solarization: lots of pixels around the middle
    of the intensity range, so partial inversion is clearly visible.
    """
    img = np.zeros((H, W, 3), dtype=np.uint8)
    for x in range(W):
        v = int(40 + 160 * x / (W - 1))
        img[:, x] = v
    # Mid-tone objects
    cv2.circle(img, (320, 200), 90, (128, 128, 128), -1)
    cv2.rectangle(img, (420, 120), (580, 280), (90, 90, 90), -1)
    cv2.circle(img, (140, 280), 55, (170, 170, 170), -1)
    cv2.putText(img, "MIDTONES", (230, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 3)
    return img


def color_cast_scene() -> np.ndarray:
    """The base scene with a strong warm (orange/red) color cast:
    red boosted, blue strongly reduced - a classic 'white balance' flaw."""
    img = base_scene().astype(np.float32)
    img[:, :, 0] = img[:, :, 0] * 0.55  # B
    img[:, :, 1] = img[:, :, 1] * 0.95  # G
    img[:, :, 2] = img[:, :, 2] * 1.25  # R
    return np.clip(img, 0, 255).astype(np.uint8)


def geometric_edges() -> np.ndarray:
    """High-contrast black shapes on white: ideal for edge detectors."""
    img = np.full((H, W, 3), 255, dtype=np.uint8)
    c = (25, 25, 25)
    cv2.rectangle(img, (60, 70), (260, 300), c, 4)          # square outline
    cv2.circle(img, (430, 180), 95, c, 4)                   # circle outline
    cv2.line(img, (300, 330), (600, 60), c, 5)              # diagonal
    cv2.circle(img, (560, 320), 25, c, -1)                  # filled dot
    cv2.line(img, (60, 350), (600, 350), c, 2)              # thin line
    cv2.putText(img, "EDGE TEST", (180, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.4, c, 3)
    return img


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    scene = base_scene()

    samples = {
        "color_scene.png": scene,
        "dark_scene.png": np.clip(scene.astype(np.float32) * 0.35, 0, 255).astype(np.uint8),
        "midtone_gradient.png": midtone_gradient(),
        "color_cast.png": color_cast_scene(),
        "low_contrast.png": make_low_contrast(scene, band=(105, 165)),
        "detail_scene.png": detail_scene(),
        "geometric_edges.png": geometric_edges(),
    }

    for name, img in samples.items():
        path = os.path.join(OUT_DIR, name)
        ok = cv2.imwrite(path, img)
        status = "ok" if ok else "FAILED"
        print(f"[{status}] {name}  shape={img.shape}  dtype={img.dtype}")
        if not ok:
            sys.exit(1)
    print(f"\nAll {len(samples)} samples written to {OUT_DIR}")


if __name__ == "__main__":
    main()
