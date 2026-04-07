from __future__ import annotations

import numpy as np

TARGET_TROUGH_INDEX = 15


def _parabolic_offset(left: float, center: float, right: float) -> float:
    denom = left - 2.0 * center + right
    if np.isclose(denom, 0.0):
        return 0.0
    offset = 0.5 * (left - right) / denom
    return float(np.clip(offset, -1.0, 1.0))


def align_snippets(
    snippets: np.ndarray,
    target_index: int = TARGET_TROUGH_INDEX,
) -> tuple[np.ndarray, np.ndarray]:
    """Align snippets so their trough is at `target_index`.

    Returns aligned snippets and estimated sub-sample trough positions.
    """
    if snippets.ndim != 2:
        raise ValueError("snippets must be shape (n_spikes, n_samples)")

    n_spikes, n_samples = snippets.shape
    x = np.arange(n_samples, dtype=float)
    aligned = np.zeros_like(snippets, dtype=float)
    trough_positions = np.zeros(n_spikes, dtype=float)

    for i, snip in enumerate(snippets):
        min_idx = int(np.argmin(snip))
        if 0 < min_idx < n_samples - 1:
            offset = _parabolic_offset(
                snip[min_idx - 1],
                snip[min_idx],
                snip[min_idx + 1],
            )
        else:
            offset = 0.0

        trough_pos = min_idx + offset
        trough_positions[i] = trough_pos

        shift = trough_pos - target_index
        sample_points = x + shift
        aligned[i] = np.interp(sample_points, x, snip, left=0.0, right=0.0)

    return aligned, trough_positions
