"""FastAPI application entrypoint for trajectory decoding."""

from __future__ import annotations

import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts
from mlflow.pytorch import load_model as load_pytorch_model

from src.api.decoder import decode
from src.api.health import (
    health_payload,
    metrics_payload,
    ready_payload,
    record_latency,
)
from src.api.schemas import SpikeBuffer, TrajectoryResponse
from src.api.websocket import router as websocket_router
from src.reduction.pca import NeuralPCA

logger = logging.getLogger(__name__)


def load_model_and_pca() -> tuple[object, NeuralPCA]:
    """Load champion model and companion PCA artifact from MLflow."""
    model_name = os.getenv("MLFLOW_MODEL_NAME", "neural-spiketrain-analysis")
    model_uri = os.getenv("MLFLOW_MODEL_URI", f"models:/{model_name}@champion")
    pca_artifact_path = os.getenv("MLFLOW_PCA_ARTIFACT_PATH", "artifacts/pca.pkl")

    model = load_pytorch_model(model_uri)

    client = MlflowClient()
    mv = client.get_model_version_by_alias(model_name, "champion")
    local_pca_path = download_artifacts(
        run_id=mv.run_id,
        artifact_path=pca_artifact_path,
        dst_path=tempfile.mkdtemp(prefix="pca_artifacts_"),
    )
    pca = NeuralPCA.load(Path(local_pca_path))
    return model, pca


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    app.state.pca = None
    try:
        model, pca = load_model_and_pca()
        app.state.model = model
        app.state.pca = pca
    except Exception:
        logger.exception("Failed to load model and PCA during startup")
    yield


app = FastAPI(title="Neural Trajectory Decoder", lifespan=lifespan)
app.include_router(websocket_router)


@app.get("/health")
def health() -> dict[str, str]:
    return health_payload()


@app.get("/ready")
def ready() -> JSONResponse:
    is_ready = app.state.model is not None and app.state.pca is not None
    payload, status = ready_payload(is_ready)
    return JSONResponse(content=payload, status_code=status)


@app.post("/predict", response_model=TrajectoryResponse)
def predict(spike_buffer: SpikeBuffer) -> TrajectoryResponse:
    if app.state.model is None or app.state.pca is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    points = decode(spike_buffer, app.state.model, app.state.pca)
    latency_ms = (time.perf_counter() - start) * 1000.0
    record_latency("/predict", latency_ms)
    logger.info("predict latency_ms=%.3f", latency_ms)

    return TrajectoryResponse(timesteps=points)


@app.get("/metrics")
def metrics() -> dict[str, float | int]:
    return metrics_payload()
