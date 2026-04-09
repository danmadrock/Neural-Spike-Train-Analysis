"""Inference decode pipeline used by REST and WebSocket handlers."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch
from pydantic import TypeAdapter

from src.api.schemas import SpikeBuffer, TrajectoryPoint
from src.binning.binner import bin_spikes
from src.binning.smoother import smooth_binned
from src.uncertainty.mc_dropout import mc_predict

TRAJECTORY_POINTS = TypeAdapter(list[TrajectoryPoint])


class PCAProtocol(Protocol):
    def transform(self, matrix: np.ndarray) -> np.ndarray: ...


def decode(
    spike_buffer: SpikeBuffer,
    model: torch.nn.Module,
    pca: PCAProtocol,
) -> list[TrajectoryPoint]:
    """Run binning -> smoothing -> PCA -> MC Dropout and return trajectory points."""
    bin_width_ms = float(getattr(model, "bin_width_ms", 50.0))
    smooth_sigma_ms = float(getattr(model, "smoothing_sigma_ms", 25.0))

    shifted_spikes = [
        [float(ts) - spike_buffer.t_start for ts in unit_times]
        for unit_times in spike_buffer.spike_times
    ]
    duration_s = spike_buffer.t_stop - spike_buffer.t_start

    binned = bin_spikes(
        shifted_spikes,
        bin_width_ms=bin_width_ms,
        t_stop=duration_s,
    )
    sigma_bins = smooth_sigma_ms / bin_width_ms
    smoothed = smooth_binned(binned, sigma_bins=sigma_bins)

    latents_np = pca.transform(smoothed)
    latents = torch.as_tensor(latents_np, dtype=torch.float32).unsqueeze(0)
    uncertainty = mc_predict(model, latents, n_samples=spike_buffer.n_mc_samples)

    mean = uncertainty.mean[0]
    std = uncertainty.std[0]

    t_ms = np.arange(mean.shape[0], dtype=float) * bin_width_ms
    raw_points: list[dict[str, float]] = []
    for idx in range(mean.shape[0]):
        x = float(mean[idx, 0])
        y = float(mean[idx, 1])
        x_std = float(std[idx, 0])
        y_std = float(std[idx, 1])
        confidence = float(1.0 / (1.0 + x_std + y_std))
        raw_points.append(
            {
                "t_ms": float(t_ms[idx]),
                "x": x,
                "y": y,
                "x_std": x_std,
                "y_std": y_std,
                "confidence": max(0.0, min(1.0, confidence)),
            }
        )

    return TRAJECTORY_POINTS.validate_python(raw_points)
