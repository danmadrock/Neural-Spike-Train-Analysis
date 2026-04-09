import numpy as np

from src.sorting.aligner import align_snippets
from src.sorting.clusterer import SortedUnit, cluster_waveforms
from src.sorting.detector import detect_spikes
from src.sorting.validator import validate_units


def test_detector_snippet_count_matches_injected_spikes():
    rng = np.random.default_rng(1)
    fs = 30_000
    voltage = rng.normal(0.0, 10e-6, size=(2, fs))
    injected = [0.1, 0.3, 0.5, 0.7]
    waveform = np.hanning(45) * 200e-6

    for ch in range(voltage.shape[0]):
        for t in injected:
            idx = int(t * fs)
            voltage[ch, idx : idx + 45] -= waveform

    snippets, times = detect_spikes(voltage, fs)
    expected = len(injected) * voltage.shape[0]
    assert snippets.shape[0] >= expected
    assert times.shape[0] == snippets.shape[0]
    assert snippets.shape[1] == 45


def test_aligner_trough_position(synthetic_voltage):
    voltage, fs = synthetic_voltage
    snippets, _ = detect_spikes(voltage, fs)
    aligned, _ = align_snippets(snippets)
    troughs = np.argmin(aligned, axis=1)
    assert np.all(np.abs(troughs - 15) <= 1)


def test_clusterer_returns_sorted_units(synthetic_voltage):
    voltage, fs = synthetic_voltage
    snippets, times = detect_spikes(voltage, fs)
    aligned, _ = align_snippets(snippets)
    units = cluster_waveforms(aligned, times)
    assert isinstance(units, list)
    assert all(isinstance(unit, SortedUnit) for unit in units)


def test_validator_flags_bad_unit():
    rng = np.random.default_rng(0)
    waveforms = rng.normal(0.0, 1e-6, size=(20, 45))
    mean_waveform = waveforms.mean(axis=0)

    bad = SortedUnit(
        unit_id=7,
        spike_times=np.array([0.0000, 0.0005, 0.0010, 0.0012]),
        mean_waveform=mean_waveform,
        waveforms=waveforms,
        pca_features=np.ones((20, 3)),
    )
    good = SortedUnit(
        unit_id=1,
        spike_times=np.array([0.1, 0.2, 0.4, 0.8]),
        mean_waveform=np.sin(np.linspace(0, np.pi, 45)),
        waveforms=np.tile(np.sin(np.linspace(0, np.pi, 45)), (20, 1)),
        pca_features=rng.normal(size=(20, 3)) + 10.0,
    )

    report = validate_units([good, bad])
    assert 7 in report["bad_unit_ids"]
    assert any(row["unit_id"] == 7 and row["bad"] for row in report["units"])
