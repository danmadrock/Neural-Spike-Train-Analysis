from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from src.sorting.aligner import align_snippets
from src.sorting.detector import detect_spikes


@dataclass
class SortedUnit:
    unit_id: int
    spike_times: np.ndarray
    mean_waveform: np.ndarray
    waveforms: np.ndarray = field(repr=False)
    pca_features: np.ndarray = field(repr=False)


def cluster_waveforms(
    aligned_snippets: np.ndarray,
    timestamps: np.ndarray,
    max_components: int = 8,
    random_state: int = 0,
) -> list[SortedUnit]:
    if aligned_snippets.size == 0:
        return []

    n_spikes = aligned_snippets.shape[0]
    n_components = min(3, aligned_snippets.shape[1], n_spikes)
    features = PCA(n_components=n_components, random_state=random_state).fit_transform(
        aligned_snippets
    )

    upper = min(max_components, n_spikes)
    best_model: GaussianMixture | None = None
    best_bic = np.inf
    for k in range(1, upper + 1):
        model = GaussianMixture(n_components=k, random_state=random_state)
        model.fit(features)
        bic = model.bic(features)
        if bic < best_bic:
            best_bic = bic
            best_model = model

    assert best_model is not None
    labels = best_model.predict(features)

    units: list[SortedUnit] = []
    for unit_id in sorted(np.unique(labels)):
        mask = labels == unit_id
        units.append(
            SortedUnit(
                unit_id=int(unit_id),
                spike_times=np.asarray(timestamps[mask], dtype=float),
                mean_waveform=np.mean(aligned_snippets[mask], axis=0),
                waveforms=np.asarray(aligned_snippets[mask], dtype=float),
                pca_features=np.asarray(features[mask], dtype=float),
            )
        )
    return units


def spike_cluster(voltage: np.ndarray, fs: float) -> list[SortedUnit]:
    snippets, timestamps = detect_spikes(voltage=voltage, fs=fs)
    aligned, _ = align_snippets(snippets)
    return cluster_waveforms(aligned, timestamps)
