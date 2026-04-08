"""Uncertainty calibration diagnostics and plots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
from matplotlib import pyplot as plt


@dataclass(slots=True)
class CalibrationResult:
    """Reliability and expected calibration error summary."""

    confidence_levels: np.ndarray
    observed_coverage: np.ndarray
    ece: float
    artifact_path: Path


@dataclass(slots=True)
class HeteroscedasticCalibrationResult:
    """Correlation between predicted sigma and actual absolute error."""

    predicted_sigma: np.ndarray
    absolute_error: np.ndarray
    pearson_r: float
    artifact_path: Path | None = None


def heteroscedastic_error_correlation(
    pred_mean: np.ndarray,
    pred_log_var: np.ndarray,
    y_true: np.ndarray,
    output_path: str | Path | None = None,
) -> HeteroscedasticCalibrationResult:
    """Check whether predicted sigma correlates with actual error."""

    pred_mean = np.asarray(pred_mean, dtype=float)
    pred_log_var = np.asarray(pred_log_var, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    predicted_sigma = np.exp(0.5 * pred_log_var)
    absolute_error = np.abs(pred_mean - y_true)

    sigma_flat = predicted_sigma.reshape(-1)
    error_flat = absolute_error.reshape(-1)

    sigma_std = np.std(sigma_flat)
    error_std = np.std(error_flat)
    if sigma_std <= 1e-12 or error_std <= 1e-12:
        pearson_r = 0.0
    else:
        pearson_r = float(np.corrcoef(sigma_flat, error_flat)[0, 1])

    artifact_path: Path | None = None
    if output_path is not None:
        artifact_path = Path(output_path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(sigma_flat, error_flat, alpha=0.35, s=12)
        ax.set_title(f"Predicted sigma vs absolute error (r={pearson_r:.3f})")
        ax.set_xlabel("predicted_sigma")
        ax.set_ylabel("absolute_error")
        fig.tight_layout()
        fig.savefig(artifact_path, dpi=160)
        plt.close(fig)

        if mlflow.active_run() is not None:
            mlflow.log_artifact(str(artifact_path), artifact_path="diagnostics")
            mlflow.log_metric("heteroscedastic_sigma_abs_error_pearson_r", pearson_r)

    return HeteroscedasticCalibrationResult(
        predicted_sigma=predicted_sigma,
        absolute_error=absolute_error,
        pearson_r=pearson_r,
        artifact_path=artifact_path,
    )


def calibration_diagnostics(
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    y_true: np.ndarray,
    confidence_levels: np.ndarray | None = None,
    output_path: str | Path = "reliability_diagram.png",
) -> CalibrationResult:
    """Build reliability curve and compute expected calibration error (ECE)."""
    pred_mean = np.asarray(pred_mean, dtype=float)
    pred_std = np.asarray(pred_std, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    if confidence_levels is None:
        confidence_levels = np.linspace(0.1, 0.9, 9)
    confidence_levels = np.asarray(confidence_levels, dtype=float)

    if np.any((confidence_levels <= 0) | (confidence_levels >= 1)):
        msg = "confidence_levels must be strictly between 0 and 1"
        raise ValueError(msg)

    # Gaussian central interval z-values without scipy dependency.
    from statistics import NormalDist

    norm = NormalDist()
    observed: list[float] = []

    safe_std = np.maximum(pred_std, 1e-8)
    residual = np.abs(y_true - pred_mean)

    for p in confidence_levels:
        z = norm.inv_cdf((1 + float(p)) / 2)
        inside = residual <= (z * safe_std)
        observed.append(float(np.mean(inside)))

    observed_cov = np.asarray(observed, dtype=float)
    ece = float(np.mean(np.abs(observed_cov - confidence_levels)))

    artifact_path = Path(output_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="ideal")
    ax.plot(confidence_levels, observed_cov, marker="o", label="model")
    ax.set_title(f"Reliability Diagram (ECE={ece:.4f})")
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Observed coverage")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(artifact_path, dpi=160)
    plt.close(fig)

    if mlflow.active_run() is not None:
        mlflow.log_artifact(str(artifact_path), artifact_path="diagnostics")
        mlflow.log_metric("ece", ece)

    return CalibrationResult(
        confidence_levels=confidence_levels,
        observed_coverage=observed_cov,
        ece=ece,
        artifact_path=artifact_path,
    )
