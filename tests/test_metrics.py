"""Tests for the metrics module (SNR from scratch + intensity statistics)."""

import numpy as np
import pytest

from src import metrics


def test_snr_known_value():
    # S = [10, 10, 10, 10] -> signal power = 400
    # X = [12, 12, 12, 12] -> noise power = 4 * 4 = 16
    # SNR = 10 * log10(400 / 16) = 10 * log10(25) ~= 3.9794 dB
    s = np.full(4, 10.0)
    x = np.full(4, 12.0)
    snr = metrics.snr_db(s, x)
    assert snr == pytest.approx(10 * np.log10(25), abs=1e-9)


def test_snr_zero_noise_is_infinite():
    s = np.array([10, 20, 30], dtype=np.float64)
    assert metrics.snr_db(s, s.copy()) == float("inf")


def test_snr_zero_signal_nonzero_noise_is_minus_infinity():
    """REGRESSION (audit issue 8): a zero-energy signal with non-zero
    error has ratio 0 -> 10*log10(0) = -infinity. We must report -inf,
    never a fake 0 dB."""
    s = np.zeros(4)
    x = np.zeros(4) + 1
    assert metrics.snr_db(s, x) == float("-inf")


def test_snr_shape_mismatch_raises():
    with pytest.raises(ValueError):
        metrics.snr_db(np.ones(4), np.ones(9))


def test_snr_rejects_same_size_different_shape():
    """REGRESSION (audit issue 8): shapes are checked BEFORE flattening,
    so same-size transposes or flat-vs-2-D arrays must be rejected."""
    with pytest.raises(ValueError):
        metrics.snr_db(np.ones((2, 3)), np.ones((3, 2)))
    with pytest.raises(ValueError):
        metrics.snr_db(np.ones(6), np.ones((2, 3)))


def test_snr_uses_power_not_psnr():
    # PSNR of a perfect copy is undefined/infinite like our SNR, but for a
    # small error the two numbers differ a lot: check a concrete case.
    s = np.full(100, 100.0)
    x = s + 1.0
    snr = metrics.snr_db(s, x)
    # signal power = 100 * 100^2 = 1e6, noise power = 100 * 1 = 100
    assert snr == pytest.approx(10 * np.log10(1e6 / 100), abs=1e-9)
    assert snr > 10  # clearly a power-ratio value, not an MSE/255^2 value


def test_snr_parts_returns_intermediates():
    s = np.full(4, 10.0)
    x = s + 2.0
    parts = metrics.snr_parts(s, x)
    assert parts["signal_power"] == pytest.approx(400.0)
    assert parts["noise_power"] == pytest.approx(16.0)
    assert parts["snr_db"] == pytest.approx(10 * np.log10(25))


def test_snr_works_on_color_images_flattened():
    s = np.full((4, 4, 3), 100.0)
    x = s + 2.0
    # 48 pixels: signal power = 48 * 100^2, noise power = 48 * 4
    assert metrics.snr_db(s, x) == pytest.approx(
        10 * np.log10(48 * 10000 / (48 * 4)), abs=1e-9
    )


def test_intensity_stats_known_values():
    img = np.array([0, 100, 200, 300], dtype=np.float64)
    s = metrics.intensity_stats(img)
    assert s["mean"] == pytest.approx(150.0)
    assert s["min"] == 0.0
    assert s["max"] == 300.0
    assert s["std"] == pytest.approx(np.std(np.array([0, 100, 200, 300])))


def test_channel_stats_shape_and_values():
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[:, :, 0] = 10
    rgb[:, :, 1] = 20
    rgb[:, :, 2] = 30
    rows = metrics.channel_stats(rgb)
    assert len(rows) == 3
    assert [r["Mean"] for r in rows] == [10.0, 20.0, 30.0]
    assert [r["Channel"] for r in rows] == ["Red", "Green", "Blue"]


def test_histogram_total_and_length():
    img = (np.random.default_rng(0).random((10, 10)) * 256).astype(np.uint8)
    counts, edges = metrics.histogram(img)
    assert len(counts) == 256
    assert int(counts.sum()) == 100
    assert len(edges) == 257
