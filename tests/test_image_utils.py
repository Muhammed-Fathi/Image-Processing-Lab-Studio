"""Tests for loading / saving utilities (src/image_utils.py)."""

import numpy as np
import cv2
import pytest

from src import image_utils as iu


@pytest.fixture()
def sample_png(tmp_path):
    """A valid 4x4 color PNG file in a temp directory."""
    img = (np.random.default_rng(1).random((4, 4, 3)) * 255).astype(np.uint8)
    path = tmp_path / "sample.png"
    assert cv2.imwrite(str(path), img)
    return str(path)


def test_load_image_from_path(sample_png):
    img = iu.load_image_from_path(sample_png)
    assert img is not None
    assert img.shape == (4, 4, 3)
    assert img.dtype == np.uint8


def test_load_image_from_path_missing_file():
    assert iu.load_image_from_path("/nonexistent/image.png") is None


def test_load_image_from_bytes_roundtrip(sample_png):
    with open(sample_png, "rb") as f:
        data = f.read()
    img = iu.load_image_from_bytes(data)
    assert img is not None
    assert img.shape == (4, 4, 3)


def test_load_image_from_bytes_garbage_returns_none():
    assert iu.load_image_from_bytes(b"this is definitely not an image") is None
    assert iu.load_image_from_bytes(b"") is None


# ---------------------------------------------------------------------------
# REGRESSION (audit issue 5): 4-channel images are BGRA in OpenCV, not RGBA
# ---------------------------------------------------------------------------

@pytest.fixture()
def bgra_png(tmp_path):
    """A 2x2 BGRA PNG whose four channels all differ, saved with cv2
    (which writes PNGs from BGRA arrays)."""
    img = np.zeros((2, 2, 4), dtype=np.uint8)
    img[0, 0] = [10, 20, 30, 255]   # B, G, R, A
    img[1, 1] = [200, 100, 50, 128]
    path = tmp_path / "bgra.png"
    assert cv2.imwrite(str(path), img)
    return str(path), img


def test_bgra_png_decoding_from_path(bgra_png):
    path, _ = bgra_png
    loaded = iu.load_image_from_path(path)
    assert loaded is not None
    assert loaded.shape == (2, 2, 3)
    # BGR order must be preserved: (B=10, G=20, R=30) and (B=200, G=100, R=50)
    assert loaded[0, 0].tolist() == [10, 20, 30]
    assert loaded[1, 1].tolist() == [200, 100, 50]


def test_bgra_png_decoding_from_bytes(bgra_png):
    path, _ = bgra_png
    with open(path, "rb") as f:
        data = f.read()
    loaded = iu.load_image_from_bytes(data)
    assert loaded is not None
    assert loaded.shape == (2, 2, 3)
    assert loaded[0, 0].tolist() == [10, 20, 30]
    assert loaded[1, 1].tolist() == [200, 100, 50]


def test_is_valid_image_bytes(sample_png):
    with open(sample_png, "rb") as f:
        good = f.read()
    assert iu.is_valid_image_bytes(good) is True
    assert iu.is_valid_image_bytes(b"\x00\x01\x02 not an image") is False


def test_encode_png_roundtrip(sample_png):
    img = cv2.imread(sample_png)
    data = iu.encode_png(img)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic header
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == img.shape


def test_ensure_color_and_to_gray():
    gray = np.full((4, 4), 120, dtype=np.uint8)
    color = iu.ensure_color(gray)
    assert color.shape == (4, 4, 3)
    assert iu.to_gray(color).shape == (4, 4)
    assert iu.to_gray(gray).shape == (4, 4)


def test_odd_kernel_clamps():
    assert iu.odd_kernel(4) == 5
    assert iu.odd_kernel(3) == 3
    assert iu.odd_kernel(2) == 3
    assert iu.odd_kernel(99) == 31  # maximum
    assert iu.odd_kernel(1) == 3    # minimum


def test_list_samples_returns_sorted_strings():
    names = iu.list_samples()
    assert isinstance(names, list)
    assert names == sorted(names)
