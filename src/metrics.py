"""Quality metrics and intensity statistics.

The SNR function is implemented from scratch on purpose (no library
PSNR/SNR): the university task requires the student to be able to show
the exact calculation.

    Signal power  = sum(S^2)          over all pixels of the clean
                                  reference image S
    Noise power   = sum((S - X)^2)    over all pixels of the evaluated
                                  image X (noisy or denoised)
    SNR (dB)      = 10 * log10(signal power / noise power)

This is a power-ratio SNR, NOT PSNR. PSNR would scale by 255^2 and use
MSE; we deliberately do not do that here.
"""

from __future__ import annotations

import numpy as np


def snr_db(signal: np.ndarray, evaluated: np.ndarray) -> float:
    """Compute SNR in dB of `evaluated` against the clean `signal`.

    * Both images must have the same shape.
    * All math is done in float64 for precision.
    * If `evaluated` equals `signal` exactly, the noise power is zero and
      the SNR is infinite (perfect reconstruction) -> returns inf.
    * If the signal itself has zero power (all-black image), returns 0.0.
    """
    s = np.asarray(signal, dtype=np.float64).ravel()
    e = np.asarray(evaluated, dtype=np.float64).ravel()
    if s.shape != e.shape:
        raise ValueError("signal and evaluated images must have the same shape")

    signal_power = float(np.sum(s * s))
    noise_power = float(np.sum((s - e) * (s - e)))

    if noise_power <= 0.0:
        return float("inf")  # no error at all: define SNR as infinite
    if signal_power <= 0.0:
        return 0.0  # no signal energy; ratio would be 0 -> -inf, report 0

    return float(10.0 * np.log10(signal_power / noise_power))


def snr_parts(signal: np.ndarray, evaluated: np.ndarray) -> dict:
    """Return the intermediate quantities, so the UI can show the math.

    Keys: signal_power, noise_power, snr_db.
    """
    s = np.asarray(signal, dtype=np.float64).ravel()
    e = np.asarray(evaluated, dtype=np.float64).ravel()
    signal_power = float(np.sum(s * s))
    noise_power = float(np.sum((s - e) * (s - e)))
    return {
        "signal_power": signal_power,
        "noise_power": noise_power,
        "snr_db": snr_db(s, e),
    }


def intensity_stats(img: np.ndarray) -> dict:
    """Mean / std / min / max intensity statistics over the whole image."""
    a = np.asarray(img, dtype=np.float64)
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def histogram(img: np.ndarray, bins: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Intensity histogram (256 bins) of a grayscale or luminance image."""
    counts, bin_edges = np.histogram(img, bins=bins, range=(0, 256))
    return counts, bin_edges


def channel_stats(rgb: np.ndarray) -> list[dict]:
    """Per-channel (R, G, B) intensity statistics for a 3-channel image."""
    names = ("Red", "Green", "Blue")
    out = []
    for i, name in enumerate(names):
        ch = np.asarray(rgb[:, :, i], dtype=np.float64)
        out.append(
            {
                "Channel": name,
                "Mean": round(float(ch.mean()), 2),
                "Std": round(float(ch.std()), 2),
                "Min": int(ch.min()),
                "Max": int(ch.max()),
            }
        )
    return out
