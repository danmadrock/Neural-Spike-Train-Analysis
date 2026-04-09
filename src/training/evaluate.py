"""Evaluation helpers: LOO-CV model comparison and reporting."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlflow
import numpy as np

from src.models.wiener import WienerFilter
from src.training.loo_cv import loo_cv


def _format_table(rows: list[dict[str, float | str]]) -> str:
    header = ["model", "R² mean", "R² std", "RMSE", "velocity RMSE", "ECE"]
    lines = [" | ".join(header), " | ".join(["---"] * len(header))]
    for row in rows:
        lines.append(
            " | ".join(
                [
                    str(row["model"]),
                    f"{float(row['r2_mean']):.4f}",
                    f"{float(row['r2_std']):.4f}",
                    f"{float(row['rmse_mean']):.4f}",
                    f"{float(row['velocity_rmse_mean']):.4f}",
                    f"{float(row['ece']):.4f}",
                ]
            )
        )
    return "\n".join(lines)


def run_comparison_table(
    neural_trials: list[np.ndarray],
    trajectory_trials: list[np.ndarray],
    *,
    pca_components: int,
    lstm_model_factory: Callable[[], Any],
    lstm_fit_fn: Callable[[Any, np.ndarray, np.ndarray], None],
    lstm_predict_fn: Callable[[Any, np.ndarray], np.ndarray],
    lstm_ece: float = float("nan"),
    output_path: str | Path = "comparison_table.md",
) -> str:
    """Run identical LOO-CV for Wiener and LSTM and log a comparison artifact."""
    wiener_scores = loo_cv(
        neural_trials,
        trajectory_trials,
        pca_components=pca_components,
        model_factory=WienerFilter,
        fit_fn=lambda model, x, y: model.fit(x, y),
        predict_fn=lambda model, x: model.predict(x),
    )
    lstm_scores = loo_cv(
        neural_trials,
        trajectory_trials,
        pca_components=pca_components,
        model_factory=lstm_model_factory,
        fit_fn=lstm_fit_fn,
        predict_fn=lstm_predict_fn,
    )

    rows: list[dict[str, float | str]] = [
        {
            "model": "wiener",
            "r2_mean": float(wiener_scores["r2_mean"]),
            "r2_std": float(wiener_scores["r2_std"]),
            "rmse_mean": float(wiener_scores["rmse_mean"]),
            "velocity_rmse_mean": float(wiener_scores["velocity_rmse_mean"]),
            "ece": float("nan"),
        },
        {
            "model": "lstm",
            "r2_mean": float(lstm_scores["r2_mean"]),
            "r2_std": float(lstm_scores["r2_std"]),
            "rmse_mean": float(lstm_scores["rmse_mean"]),
            "velocity_rmse_mean": float(lstm_scores["velocity_rmse_mean"]),
            "ece": float(lstm_ece),
        },
    ]
    table = _format_table(rows)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(table)
    Path(output.with_suffix(".json")).write_text(json.dumps(rows, indent=2))

    print(table)
    if mlflow.active_run() is not None:
        mlflow.log_artifact(str(output), artifact_path="evaluation")
        mlflow.log_artifact(
            str(output.with_suffix(".json")), artifact_path="evaluation"
        )
    return table
