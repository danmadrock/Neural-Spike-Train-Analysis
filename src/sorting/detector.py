"""
Spike detection via median-based threshold crossing.
Extracts waveform snippets from raw extracellular voltage when
the signal exceeds a noise-derived threshold.
"""


def detect_spikes(voltage, fs: float, threshold_multiplier: float = 4.0):
    """Detect threshold crossing events and return snippet array + timestamps."""
    raise NotImplementedError
