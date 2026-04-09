from __future__ import annotations

from collections.abc import Generator

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from src.api.schemas import SpikeBuffer


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bin_width_ms = 50.0
        self.smoothing_sigma_ms = 25.0

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, t_steps, _ = x.shape
        timeline = torch.linspace(0.0, 1.0, t_steps, device=x.device)
        mean = torch.zeros((batch, t_steps, 2), dtype=torch.float32, device=x.device)
        mean[:, :, 0] = timeline
        mean[:, :, 1] = -timeline
        log_var = torch.zeros_like(mean)
        return mean, log_var


class _FakePCA:
    def transform(self, matrix: np.ndarray) -> np.ndarray:
        return np.asarray(matrix, dtype=np.float32)


@pytest.fixture
def test_client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    import src.api.main as main

    monkeypatch.setattr(main, "load_model_and_pca", lambda: (_FakeModel(), _FakePCA()))
    with TestClient(main.app) as client:
        yield client


def _sample_buffer() -> dict[str, object]:
    return {
        "spike_times": [[0.01, 0.08, 0.12], [0.03, 0.09], []],
        "unit_ids": [0, 1, 2],
        "t_start": 0.0,
        "t_stop": 0.5,
        "n_mc_samples": 4,
    }


def test_health(test_client: TestClient) -> None:
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready(test_client: TestClient) -> None:
    response = test_client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_predict_with_synthetic_buffer(test_client: TestClient) -> None:
    response = test_client.post("/predict", json=_sample_buffer())
    assert response.status_code == 200
    payload = response.json()
    assert "timesteps" in payload
    assert len(payload["timesteps"]) == 10

    point = payload["timesteps"][0]
    assert {"x", "y", "x_std", "y_std", "confidence", "t_ms"}.issubset(point)


def test_websocket_round_trip(test_client: TestClient) -> None:
    with test_client.websocket_connect("/ws/decode") as websocket:
        websocket.send_json(_sample_buffer())
        received = websocket.receive_json()
        assert {"x", "y", "x_std", "y_std", "confidence", "t_ms"}.issubset(received)


def test_decode_returns_expected_length() -> None:
    from src.api.decoder import decode

    points = decode(
        SpikeBuffer.model_validate(_sample_buffer()), _FakeModel(), _FakePCA()
    )
    assert len(points) == 10
