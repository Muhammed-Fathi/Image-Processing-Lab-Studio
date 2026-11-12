"""Core image processing algorithms for Image Processing Lab Studio.

Every function here is a pure NumPy/OpenCV function: no Streamlit, no
file I/O. Images are NumPy arrays:

    * color images  : (H, W, 3) uint8 in BGR order (OpenCV convention)
    * gray images   : (H, W) uint8

All functions work in floating point internally and only convert back to
uint8 after clipping, so unsigned overflow / underflow cannot corrupt
results.
"""

from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Task 1 - RGB channel separation
# ---------------------------------------------------------------------------

def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert a BGR image (OpenCV order) to RGB order.

    OpenCV loads color images as BGR. Displaying them directly would swap
    the red and blue channels, so every task that needs RGB starts here.
    """
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def separate_channels(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split an RGB image into its Red, Green and Blue single-channel arrays."""
    return cv2.split(rgb)


def isolate_channel(rgb: np.ndarray, channel: int) -> np.ndarray:
    """Return a color image in which ONLY `channel` (0=R,1=G,2=B) is kept.

    The other two channels are set to zero, which shows what the scene
    looks like lit only by that primary color.
    """
    r, g, b = cv2.split(rgb)
    r, g, b = r.copy(), g.copy(), b.copy()
    isolated = [r, g, b]
    for i in range(3):
        if i != channel:
            isolated[i] = np.zeros_like(isolated[i])
    return cv2.merge(isolated)


# ---------------------------------------------------------------------------
# Task 2 - weighted grayscale conversion
# ---------------------------------------------------------------------------

def weighted_grayscale(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to grayscale with the ITU-R BT.601 luminance formula.

        Gray = 0.299*R + 0.587*G + 0.114*B

    The arithmetic is done in float32 and the result is clipped to
    [0, 255] before being cast to uint8. This is deliberately NOT
    cv2.cvtColor(..., COLOR_RGB2GRAY) and NOT a plain average.
    """
    if rgb.ndim == 2:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    gray = np.clip(gray, 0.0, 255.0)
    return gray.astype(np.uint8)


# ---------------------------------------------------------------------------
# Task 3 - arithmetic intensity operations
# ---------------------------------------------------------------------------

def arithmetic_operation(img: np.ndarray, operation: str, c: float) -> np.ndarray:
    """Apply an arithmetic intensity operation to every pixel.

    Supported operations:
        "add"     : I + C
        "sub"     : I - C
        "mul"     : I * C
        "div"     : I / C

    `c` is the scalar constant. Computation happens in float32 and the
    result is clipped to [0, 255], so uint8 overflow (e.g. 250 + 10) can
    never wrap around to a small value. Division by zero is prevented by
    clamping C away from zero.
    """
    c = float(c)
    if operation == "div" and abs(c) < 0.05:
        c = 0.05  # hard guard: never divide by (near) zero
    arr = img.astype(np.float32)
    if operation == "add":
        out = arr + c
    elif operation == "sub":
        out = arr - c
    elif operation == "mul":
        out = arr * c
    elif operation == "div":
        out = arr / c
    else:
        raise ValueError(f"Unknown arithmetic operation: {operation}")
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Task 4 - negative and solarization
# ---------------------------------------------------------------------------

def negative(img: np.ndarray) -> np.ndarray:
    """Image complement (negative): out = 255 - I for every pixel."""
    return cv2.bitwise_not(img)


def solarize(img: np.ndarray, threshold: int = 127) -> np.ndarray:
    """Threshold-controlled partial inversion (solarization).

    Exact rule, applied to every pixel:

        out = 255 - I   if I >= threshold
        out = I         if I <  threshold

    Bright pixels are inverted into darks while dark pixels stay put,
    which produces the characteristic "solarized" highlight look.
    """
    t = int(np.clip(threshold, 0, 255))
    out = img.astype(np.float32).copy()
    out[img >= t] = 255.0 - out[img >= t]
    return out.astype(np.uint8)


# ---------------------------------------------------------------------------
# Task 5 - channel elimination
# ---------------------------------------------------------------------------

def eliminate_channel(rgb: np.ndarray, channel: int) -> np.ndarray:
    """Zero out exactly one color channel (0=R, 1=G, 2=B) of an RGB image.

    The two remaining channels are left completely unchanged. This is a
    channel-removal experiment, not a color-correction: the result is
    intentionally a different (often unnatural) color appearance.
    """
    r, g, b = cv2.split(rgb)
    r, g, b = r.copy(), g.copy(), b.copy()
    if channel == 0:
        r = np.zeros_like(r)
    elif channel == 1:
        g = np.zeros_like(g)
    elif channel == 2:
        b = np.zeros_like(b)
    else:
        raise ValueError("channel must be 0 (R), 1 (G) or 2 (B)")
    return cv2.merge((r, g, b))


# ---------------------------------------------------------------------------
# Task 6 - histogram enhancement
# ---------------------------------------------------------------------------

def _stretch_gray(gray: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Linearly stretch a gray image: out = (I - min) * 255 / (max - min).

    If the image is constant (max == min) the division would be by zero;
    in that case the input is returned unchanged so no crash happens.
    """
    i_min = int(gray.min())
    i_max = int(gray.max())
    if i_max == i_min:
        return gray.copy(), i_min, i_max
    out = (gray.astype(np.float32) - i_min) * (255.0 / float(i_max - i_min))
    return np.clip(out, 0.0, 255.0).astype(np.uint8), i_min, i_max


def histogram_stretch(img: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Histogram stretching (contrast stretch) for gray or color images.

    Gray images are stretched directly. Color images are stretched in the
    Y (luminance) channel of the YCrCb color space, which preserves hue
    sensibly instead of stretching every RGB channel independently.
    Returns (result, original_min, original_max).
    """
    if img.ndim == 2:
        return _stretch_gray(img)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y_out, i_min, i_max = _stretch_gray(ycrcb[:, :, 0])
    ycrcb[:, :, 0] = y_out
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR), i_min, i_max


def histogram_equalization(img: np.ndarray) -> np.ndarray:
    """Histogram equalization for gray or color images.

    Gray images use cv2.equalizeHist directly. Color images are
    equalized on the Y (luminance) channel of YCrCb and the original
    chrominance (Cr, Cb) is kept, which avoids the color distortion you
    get by equalizing R, G and B independently.
    """
    if img.ndim == 2:
        return cv2.equalizeHist(img)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def adjust_brightness(img: np.ndarray, delta: float) -> np.ndarray:
    """Add a brightness offset (may be negative) to every channel."""
    out = img.astype(np.float32) + float(delta)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def make_low_contrast(img: np.ndarray, band: tuple[int, int] = (105, 165)) -> np.ndarray:
    """Compress the intensity range of an image into a narrow band.

    Used to create a demonstrably low-contrast version of any image:
    each channel is linearly mapped onto [band[0], band[1]].
    """
    lo, hi = float(band[0]), float(band[1])
    out = img.astype(np.float32)
    if out.ndim == 2:
        c_min, c_max = float(out.min()), float(out.max())
    else:
        c_min, c_max = float(out.min(axis=(0, 1)).min()), float(out.max(axis=(0, 1)).max())
    if c_max == c_min:
        return out.clip(lo, hi).astype(np.uint8)
    out = (out - c_min) * (hi - lo) / (c_max - c_min) + lo
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Task 7 - noise and smoothing filters
# ---------------------------------------------------------------------------

def add_salt_and_pepper(img: np.ndarray, ratio: float, seed: int = 42) -> np.ndarray:
    """Inject salt-and-pepper noise with NumPy (reproducible via `seed`).

    `ratio` is the fraction of corrupted pixels (0..1). Half of them are
    forced to white (salt), half to black (pepper). The whole pixel value
    is replaced on every channel, as in a classic impulse-noise model.
    """
    ratio = float(np.clip(ratio, 0.0, 1.0))
    rng = np.random.default_rng(seed)
    out = img.astype(np.float32)
    n = int(round(ratio * out.size))
    if n > 0:
        idx = rng.choice(out.size, size=n, replace=False)
        half = n // 2
        out.flat[idx[:half]] = 255.0
        out.flat[idx[half:]] = 0.0
    return out.astype(np.uint8)


def add_gaussian_noise(img: np.ndarray, sigma: float, seed: int = 42) -> np.ndarray:
    """Inject additive Gaussian noise N(0, sigma^2) on every channel.

    Reproducible for a fixed `seed`. The result is clipped to [0, 255]
    (the clipping slightly truncates the tails of the noise distribution,
    which is the standard practical behaviour for 8-bit images).
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, float(sigma), size=img.shape)
    out = img.astype(np.float32) + noise.astype(np.float32)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def median_filter(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Median filtering - excellent against impulse (salt & pepper) noise."""
    k = int(ksize) if int(ksize) % 2 == 1 else int(ksize) + 1
    k = max(3, k)
    return cv2.medianBlur(img, k)


def gaussian_filter(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Gaussian smoothing - good against Gaussian (additive) noise."""
    k = int(ksize) if int(ksize) % 2 == 1 else int(ksize) + 1
    k = max(3, k)
    return cv2.GaussianBlur(img, (k, k), 0)


def mean_filter(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Box (mean) filtering - simple averaging, for comparison only."""
    k = int(ksize) if int(ksize) % 2 == 1 else int(ksize) + 1
    k = max(3, k)
    return cv2.blur(img, (k, k))


SMOOTHING_FILTERS = {
    "Median filter": median_filter,
    "Gaussian filter": gaussian_filter,
    "Mean filter": mean_filter,
}


# ---------------------------------------------------------------------------
# Task 8 - blur and sharpening
# ---------------------------------------------------------------------------

def gaussian_blur(img: np.ndarray, ksize: int = 7) -> np.ndarray:
    """Gaussian blur with an odd kernel size (auto-corrected if needed)."""
    k = int(ksize) if int(ksize) % 2 == 1 else int(ksize) + 1
    k = max(3, k)
    return cv2.GaussianBlur(img, (k, k), 0)


SHARPEN_LAPLACIAN_KERNEL = np.array(
    [[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32
)
"""4-neighbour Laplacian. Convolution gives the second derivative, which
is negative at edges; subtracting it from the image reinforces edges."""

SHARPEN_CUSTOM_KERNEL = np.array(
    [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32
)
"""Classic 3x3 sharpening kernel (kernel sum = 1, flat areas unchanged)."""


def sharpen_laplacian(img: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Laplacian sharpening: out = I - alpha * Laplacian(I).

    The Laplacian is a band-pass (second derivative) operator; it is
    negative in regions where the pixel is brighter than its neighbours,
    so subtracting it pushes those regions brighter - edges get stronger.
    """
    lap = cv2.filter2D(img, -1, SHARPEN_LAPLACIAN_KERNEL)
    out = img.astype(np.float32) - float(alpha) * lap
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def sharpen_custom_kernel(img: np.ndarray, intensity: float = 1.0) -> np.ndarray:
    """Sharpen with the custom 3x3 kernel [[0,-1,0],[-1,5,-1],[0,-1,0]].

    The kernel is multiplied by `intensity` before convolution. The
    kernel values are fixed and safe (sum = 1, no division involved), so
    the output stays in a well-behaved numeric range and is clipped to
    [0, 255].
    """
    k = SHARPEN_CUSTOM_KERNEL * float(intensity)
    out = cv2.filter2D(img, -1, k)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def unsharp_mask(img: np.ndarray, amount: float = 1.0, ksize: int = 7) -> np.ndarray:
    """Unsharp masking: out = I + amount * (I - blurred(I)).

    The high-pass detail is (I - blurred(I)); adding a scaled copy back
    increases local contrast around edges.
    """
    blurred = gaussian_blur(img, ksize)
    out = img.astype(np.float32) + float(amount) * (
        img.astype(np.float32) - blurred.astype(np.float32)
    )
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Task 9 - edge detection
# ---------------------------------------------------------------------------

def sobel_edges(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Sobel gradient magnitude (first derivative) visualized as 0..255.

    gx and gy are computed in float32, combined into the true gradient
    magnitude sqrt(gx^2 + gy^2) and then normalized so the strongest
    gradient maps to 255. Using the magnitude (instead of the raw signed
    gradients) avoids truncating negative values incorrectly.
    """
    k = int(ksize) if int(ksize) % 2 == 1 else int(ksize) + 1
    k = max(3, k)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=k)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=k)
    mag = cv2.magnitude(gx, gy)  # sqrt(gx^2 + gy^2)
    m_max = float(mag.max())
    if m_max <= 1e-6:  # constant image -> no edges, avoid division by zero
        return np.zeros_like(gray)
    return (mag * (255.0 / m_max)).astype(np.uint8)


def canny_edges(gray: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    """Canny edge detector (thin edges via non-maximum suppression +
    double threshold hysteresis). `low` and `high` are the thresholds;
    `low` is clamped below `high` automatically.
    """
    low_i, high_i = int(low), int(high)
    if low_i >= high_i:
        high_i = low_i + 10
    return cv2.Canny(gray, low_i, high_i)


def laplacian_edges(gray: np.ndarray) -> np.ndarray:
    """Laplacian edge detection (second derivative, isotropic).

    The absolute value of the response is shown, so both rising and
    falling edges appear bright on black.
    """
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return np.clip(np.abs(lap), 0.0, 255.0).astype(np.uint8)
