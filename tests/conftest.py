import numpy as np
import pytest


@pytest.fixture
def synthetic_voltage():
    """Raw extracellular voltage: 4 channels, 30 kHz, 2 seconds."""
    rng = np.random.default_rng(42)
    n_channels, fs, duration = 4, 30_000, 2
    noise = rng.normal(0, 50e-6, (n_channels, fs * duration))  # 50 uV RMS
    # later inject 5 synthetic spikes per channel at known times
    spike_times = [0.1, 0.4, 0.7, 1.1, 1.6]
    waveform = np.hanning(45) * 200e-6  # simple spike shape
    for ch in range(n_channels):
        for t in spike_times:
            idx = int(t * fs)
            noise[ch, idx : idx + 45] -= waveform
    return noise, fs


@pytest.fixture
def synthetic_spike_trains():
    """Three sorted units with Poisson spike trains, 2 seconds."""
    rng = np.random.default_rng(42)
    firing_rates = [20.0, 35.0, 12.0]  # Hz
    duration = 2.0
    trains = []
    for rate in firing_rates:
        n_spikes = rng.poisson(rate * duration)
        times = np.sort(rng.uniform(0, duration, n_spikes))
        trains.append(times)
    return trains


@pytest.fixture
def synthetic_rate_matrix():
    """Binned firing-rate matrix: 40 bins x 3 units (50 ms bins, 2 s)."""
    rng = np.random.default_rng(42)
    return rng.poisson(lam=2.0, size=(40, 3)).astype(float)


@pytest.fixture
def synthetic_trial():
    """PCA latents + ground-truth trajectory for one 2-second trial."""
    rng = np.random.default_rng(42)
    T, K = 40, 10
    t = np.linspace(0, 2 * np.pi, T)
    # Ground truth: a smooth figure-8 trajectory
    x = np.sin(t)
    y = np.sin(2 * t)
    trajectory = np.stack([x, y], axis=1)
    # Latents: correlated with trajectory + noise
    latents = trajectory[:, :1] @ rng.normal(size=(1, K)) + rng.normal(0, 0.1, (T, K))
    return {"latents": latents, "trajectory": trajectory, "T": T, "K": K}
