"""Spike detection via robust threshold crossing and snippet extraction."""

from __future__ import annotations

import numpy as np

SNIPPET_LEN = 45
TARGET_TROUGH_INDEX = 15


def _channel_threshold(channel: np.ndarray, threshold_multiplier: float) -> float:
    sigma_hat = np.median(np.abs(channel)) / 0.6745
    return -threshold_multiplier * sigma_hat


def detect_spikes(
    voltage: np.ndarray,
    fs: float,
    threshold_multiplier: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect threshold crossings and return `(snippets, timestamps)`.
    Parameters:
    ----------
        voltage:
            Voltage array with shape ``(n_channels, n_samples)``.
        fs:
            Sampling frequency in Hz.
        threshold_multiplier:
            Multiplier for robust median threshold.
    """
    if voltage.ndim != 2:
        raise ValueError("voltage must be 2D with shape (n_channels, n_samples)")

    n_channels, n_samples = voltage.shape
    pre = TARGET_TROUGH_INDEX

    snippets: list[np.ndarray] = []
    timestamps: list[float] = []

    for ch in range(n_channels):
        signal = voltage[ch]
        thr = _channel_threshold(signal, threshold_multiplier)
        crossings = np.where((signal[:-1] > thr) & (signal[1:] <= thr))[0] + 1

        for idx in crossings:
            start = idx - pre
            end = start + SNIPPET_LEN
            if start < 0 or end > n_samples:
                continue
            snippets.append(signal[start:end])
            timestamps.append(idx / fs)

    if not snippets:
        return np.empty((0, SNIPPET_LEN), dtype=float), np.empty((0,), dtype=float)

    return np.asarray(snippets, dtype=float), np.asarray(timestamps, dtype=float)
