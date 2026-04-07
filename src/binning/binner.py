"""Spike train fixed-window binning utilities."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

ArrayLike1D = np.ndarray | Sequence[float]


def bin_spikes(
    spike_trains: Sequence[ArrayLike1D],
    bin_width_ms: float,
    t_stop: float,
) -> np.ndarray:
    """Bin per-unit spike timestamps into fixed-width count bins.
    Args:
        spike_trains: One timestamp array/list per unit (in seconds).
        bin_width_ms: Bin width in milliseconds.
        t_stop: End of decode/training window in seconds (start is assumed 0.0).
    Returns:
        A ``(T_bins, N_units)`` matrix of spike counts.
    """
    if bin_width_ms <= 0:
        msg = "bin_width_ms must be > 0"
        raise ValueError(msg)
    if t_stop <= 0:
        msg = "t_stop must be > 0"
        raise ValueError(msg)

    bin_width_s = bin_width_ms / 1000.0
    n_bins = int(math.ceil(t_stop / bin_width_s))
    n_units = len(spike_trains)

    binned = np.zeros((n_bins, n_units), dtype=np.int64)
    if n_units == 0:
        return binned

    edges = np.arange(0.0, n_bins + 1, dtype=float) * bin_width_s

    for unit_idx, unit_spikes in enumerate(spike_trains):
        spikes = np.asarray(unit_spikes, dtype=float)
        if spikes.size == 0:
            continue

        # Keep spikes inside [0, t_stop) so bin sums map to window-contained spikes.
        in_window = (spikes >= 0.0) & (spikes < t_stop)
        if not np.any(in_window):
            continue

        counts, _ = np.histogram(spikes[in_window], bins=edges)
        binned[:, unit_idx] = counts

    return binned
