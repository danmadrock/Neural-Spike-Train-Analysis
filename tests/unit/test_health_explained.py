from pathlib import Path

import numpy as np
import pytest

from src.api import health
from src.reduction import NeuralPCA
from src.reduction.explained import generate_variance_diagnostics


def test_health_helpers() -> None:
    health.request_latency_ms.clear()
    health.request_count.clear()

    health.record_latency("/predict", 10.0)
    health.record_latency("/predict", 20.0)

    assert health.health_payload() == {"status": "ok"}
    assert health.ready_payload(True) == ({"status": "ready"}, 200)
    assert health.ready_payload(False) == (
        {"status": "not_ready", "reason": "model loading"},
        503,
    )

    metrics = health.metrics_payload()
    assert metrics["predict_requests_total"] == 2
    assert metrics["predict_latency_ms_avg"] == pytest.approx(15.0)


def test_generate_variance_diagnostics_outputs_plot(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(120, 6))
    pca = NeuralPCA(n_components=4).fit(x)

    out = tmp_path / "scree.png"
    threshold_idx = generate_variance_diagnostics(
        pca,
        variance_threshold=0.8,
        output_path=out,
    )

    assert out.exists()
    assert 1 <= threshold_idx <= 4


def test_generate_variance_diagnostics_rejects_invalid_threshold(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(5)
    x = rng.normal(size=(100, 5))
    pca = NeuralPCA(n_components=3).fit(x)

    with pytest.raises(ValueError, match="variance_threshold"):
        generate_variance_diagnostics(
            pca,
            variance_threshold=1.2,
            output_path=tmp_path / "x.png",
        )
