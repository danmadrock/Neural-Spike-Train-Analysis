import numpy as np

from src.binning.binner import bin_spikes
from src.binning.smoother import smooth_binned


def test_bin_spikes_shape():
    spikes = [
        np.array([0.01, 0.06, 0.11, 0.23]),
        np.array([0.02, 0.03]),
        np.array([], dtype=float),
    ]
    out = bin_spikes(spikes, bin_width_ms=50, t_stop=0.25)
    assert out.shape == (5, 3)


def test_bin_spikes_sum_matches_input_spike_count_in_window():
    spikes = [
        np.array([0.01, 0.06, 0.11, 0.23, 0.30]),
        np.array([0.02, 0.03, 0.40]),
    ]
    t_stop = 0.25
    out = bin_spikes(spikes, bin_width_ms=50, t_stop=t_stop)

    expected = sum(int(np.sum((u >= 0.0) & (u < t_stop))) for u in spikes)
    assert int(out.sum()) == expected


def test_smooth_output_non_negative():
    binned = np.array(
        [
            [0, 2, 1],
            [1, 0, 0],
            [0, 1, 3],
            [4, 0, 2],
        ],
        dtype=float,
    )
    smoothed = smooth_binned(binned, sigma_bins=1.0)
    assert smoothed.shape == binned.shape
    assert np.all(smoothed >= 0.0)


def test_smoothing_does_not_inflate_global_peak():
    binned = np.array(
        [
            [0, 0],
            [5, 1],
            [0, 0],
            [0, 0],
        ],
        dtype=float,
    )
    smoothed = smooth_binned(binned, sigma_bins=1.0)
    assert smoothed.max() <= binned.max() + 1e-9


def test_edge_cases_empty_spike_train_and_single_spike():
    spikes = [np.array([], dtype=float), np.array([0.10])]
    out = bin_spikes(spikes, bin_width_ms=50, t_stop=0.20)

    assert out.shape == (4, 2)
    assert np.all(out[:, 0] == 0)
    assert int(out[:, 1].sum()) == 1
