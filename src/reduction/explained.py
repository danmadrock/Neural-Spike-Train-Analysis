"""Variance explained diagnostics for PCA reduction"""

from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np

from src.reduction.pca import NeuralPCA


def generate_variance_diagnostics(
    neural_pca: NeuralPCA,
    variance_threshold: float,
    output_path: str | Path,
) -> int:
    """Save scree plot, log to MLflow, and return 1-indexed threshold component."""
    if not 0 < variance_threshold <= 1:
        msg = "variance_threshold must be in (0, 1]"
        raise ValueError(msg)

    cumulative = np.cumsum(neural_pca.explained_variance_ratio_)
    threshold_idx = int(
        np.searchsorted(cumulative, variance_threshold, side="left") + 1
    )

    artifact_path = neural_pca.plot_scree(output_path)
    if mlflow.active_run() is None:
        with mlflow.start_run(nested=True):
            mlflow.log_artifact(str(artifact_path), artifact_path="diagnostics")
            mlflow.log_metric("pca_variance_threshold", variance_threshold)
            mlflow.log_metric("pca_n_components_for_threshold", threshold_idx)
    else:
        mlflow.log_artifact(str(artifact_path), artifact_path="diagnostics")
        mlflow.log_metric("pca_variance_threshold", variance_threshold)
        mlflow.log_metric("pca_n_components_for_threshold", threshold_idx)

    return threshold_idx
