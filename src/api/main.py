# src/api/main.py
from contextlib import asynccontextmanager

from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    # model loading will go here later
    yield


app = FastAPI(title="Neural Trajectory Decoder", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    return {"status": "not_ready", "reason": "model not loaded"}
