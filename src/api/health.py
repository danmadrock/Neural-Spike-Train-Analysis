"""Health and readiness helpers."""

from __future__ import annotations

from collections import defaultdict

request_latency_ms: list[float] = []
request_count: dict[str, int] = defaultdict(int)


def record_latency(endpoint: str, latency_ms: float) -> None:
    request_count[endpoint] += 1
    request_latency_ms.append(float(latency_ms))


def health_payload() -> dict[str, str]:
    return {"status": "ok"}


def ready_payload(is_ready: bool) -> tuple[dict[str, str], int]:
    if is_ready:
        return {"status": "ready"}, 200
    return {"status": "not_ready", "reason": "model loading"}, 503


def metrics_payload() -> dict[str, float | int]:
    avg_latency = (
        sum(request_latency_ms) / len(request_latency_ms) if request_latency_ms else 0.0
    )
    return {
        "predict_requests_total": request_count["/predict"],
        "predict_latency_ms_avg": avg_latency,
    }
