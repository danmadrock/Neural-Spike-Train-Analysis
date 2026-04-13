import json
from pathlib import Path

import numpy as np
import pytest

from src.models.wiener import WienerFilter
from src.training.evaluate import _format_table, run_comparison_table


def test_wiener_fit_predict_recovers_linear_mapping() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 5))
    w = rng.normal(size=(5, 2))
    y = x @ w + 0.2

    model = WienerFilter().fit(x, y)
    pred = model.predict(x)

    assert pred.shape == y.shape
    assert np.mean((pred - y) ** 2) < 1e-10


def test_wiener_predict_requires_fit() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        WienerFilter().predict(np.zeros((3, 2)))


def test_format_table_contains_expected_headers() -> None:
    table = _format_table(
        [
            {
                "model": "x",
                "r2_mean": 0.1,
                "r2_std": 0.2,
                "rmse_mean": 0.3,
                "velocity_rmse_mean": 0.4,
                "ece": 0.5,
            }
        ]
    )
    assert "R² mean" in table
    assert "x | 0.1000" in table


def test_run_comparison_table_writes_markdown_and_json(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    neural_trials = [rng.normal(size=(30, 4)) for _ in range(4)]
    traj_trials = [rng.normal(size=(30, 2)) for _ in range(4)]

    class DummyModel:
        def fit(self, x: np.ndarray, y: np.ndarray) -> None:
            self.mean = y.mean(axis=0)

        def predict(self, x: np.ndarray) -> np.ndarray:
            return np.repeat(self.mean[None, :], x.shape[0], axis=0)

    out_path = tmp_path / "comparison.md"
    table = run_comparison_table(
        neural_trials,
        traj_trials,
        pca_components=3,
        lstm_model_factory=DummyModel,
        lstm_fit_fn=lambda m, x, y: m.fit(x, y),
        lstm_predict_fn=lambda m, x: m.predict(x),
        lstm_ece=0.12,
        output_path=out_path,
    )

    assert "wiener" in table
    assert out_path.exists()
    json_path = out_path.with_suffix(".json")
    payload = json.loads(json_path.read_text())
    assert len(payload) == 2
    assert payload[1]["ece"] == pytest.approx(0.12)
