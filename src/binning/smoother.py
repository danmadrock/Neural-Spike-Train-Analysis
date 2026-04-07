"""Temporal smoothing for binned spike counts."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d


def smooth_binned(
    binned: np.ndarray,
    sigma_bins: float,
) -> np.ndarray:
    """Apply Gaussian smoothing along the time axis.

    Args:
        binned: Input array with shape ``(T_bins, N_units)``.
        sigma_bins: Gaussian sigma expressed in number of bins.

    Returns:
        Smoothed array with exactly the same shape as ``binned``.
    """
    if sigma_bins < 0:
        msg = "sigma_bins must be >= 0"
        raise ValueError(msg)

    arr = np.asarray(binned, dtype=float)
    if arr.ndim != 2:
        msg = "binned must be a 2D array of shape (T_bins, N_units)"
        raise ValueError(msg)

    if sigma_bins == 0:
        return arr.copy()

    smoothed = gaussian_filter1d(arr, sigma=sigma_bins, axis=0, mode="nearest")
    return smoothed
