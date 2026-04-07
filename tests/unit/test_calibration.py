from pathlib import Path

import numpy as np

from src.uncertainty.calibration import (
    calibration_diagnostics,
    heteroscedastic_error_correlation,
)


def test_calibration_diagnostics_runs_on_synthetic_data(tmp_path: Path):
    rng = np.random.default_rng(42)
    n = 400
    y_true = rng.normal(size=(n, 2))
    pred_std = np.full((n, 2), 0.3)
    pred_mean = y_true + rng.normal(scale=pred_std, size=(n, 2))

    result = calibration_diagnostics(
        pred_mean=pred_mean,
        pred_std=pred_std,
        y_true=y_true,
        output_path=tmp_path / "reliability.png",
    )

    assert result.confidence_levels.shape == result.observed_coverage.shape
    assert 0 <= result.ece <= 1
    assert result.artifact_path.exists()


def test_heteroscedastic_sigma_correlates_with_error(tmp_path: Path):
    rng = np.random.default_rng(0)
    n = 500
    base = rng.uniform(0.05, 0.6, size=(n, 2))
    noise = rng.normal(scale=base, size=(n, 2))
    y_true = rng.normal(size=(n, 2))
    pred_mean = y_true + noise
    pred_log_var = np.log(base**2)

    result = heteroscedastic_error_correlation(
        pred_mean=pred_mean,
        pred_log_var=pred_log_var,
        y_true=y_true,
        output_path=tmp_path / "sigma_vs_error.png",
    )

    assert result.predicted_sigma.shape == pred_mean.shape
    assert result.absolute_error.shape == pred_mean.shape
    assert result.pearson_r > 0.3
    assert result.artifact_path is not None and result.artifact_path.exists()
