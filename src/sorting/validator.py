"""Quality metrics for sorted units."""

from __future__ import annotations

import numpy as np

from src.sorting.clusterer import SortedUnit


def _isi_violation_rate(spike_times: np.ndarray, refractory_ms: float) -> float:
    if spike_times.size < 2:
        return 0.0
    isi = np.diff(np.sort(spike_times))
    return float(np.mean(isi < (refractory_ms / 1000.0)))


def _snr(unit: SortedUnit) -> float:
    noise_source = unit.waveforms - unit.mean_waveform[None, :]
    noise_std = (
        float(np.std(noise_source))
        if noise_source.size
        else float(np.std(unit.mean_waveform))
    )
    if np.isclose(noise_std, 0.0):
        return float("inf")
    return float(np.ptp(unit.mean_waveform) / (2.0 * noise_std))


def _isolation_distance(
    target: SortedUnit, all_features: np.ndarray, labels: np.ndarray
) -> float:
    unit_features = target.pca_features
    if unit_features.shape[0] < 2:
        return 0.0

    mu = unit_features.mean(axis=0)
    cov = np.cov(unit_features, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    cov += np.eye(cov.shape[0]) * 1e-6
    inv_cov = np.linalg.pinv(cov)

    centered = all_features - mu
    d2 = np.einsum("ij,jk,ik->i", centered, inv_cov, centered)

    other = d2[labels != target.unit_id]
    n_unit = unit_features.shape[0]
    if other.size < n_unit:
        return float("inf")
    return float(np.sort(other)[n_unit - 1])


def validate_units(
    units: list[SortedUnit],
    refractory_ms: float = 1.5,
    isi_threshold: float = 0.01,
    snr_threshold: float = 3.0,
    isolation_threshold: float = 10.0,
) -> dict:
    """Return quality report while keeping all units."""
    if not units:
        return {"n_units": 0, "units": [], "bad_unit_ids": []}

    all_features = np.vstack([u.pca_features for u in units])
    labels = np.concatenate(
        [np.full(u.pca_features.shape[0], u.unit_id, dtype=int) for u in units]
    )

    rows = []
    bad_ids = []
    for unit in units:
        isi_rate = _isi_violation_rate(unit.spike_times, refractory_ms)
        snr = _snr(unit)
        isolation_distance = _isolation_distance(unit, all_features, labels)

        flags = {
            "isi": isi_rate >= isi_threshold,
            "snr": snr <= snr_threshold,
            "isolation": isolation_distance <= isolation_threshold,
        }
        is_bad = any(flags.values())
        if is_bad:
            bad_ids.append(unit.unit_id)

        rows.append(
            {
                "unit_id": unit.unit_id,
                "n_spikes": int(unit.spike_times.size),
                "isi_violation_rate": isi_rate,
                "snr": snr,
                "isolation_distance": isolation_distance,
                "flags": flags,
                "bad": is_bad,
            }
        )

    return {
        "n_units": len(units),
        "units": rows,
        "bad_unit_ids": bad_ids,
    }
