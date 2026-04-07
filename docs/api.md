# API Reference

## Base URL and versioning

```
http://localhost:8000
```

All endpoints are unversioned in the current release. A `/v1/` prefix will be introduced when breaking changes are required; the unversioned path will remain available for one release cycle after any such change.

The OpenAPI specification is served at `/docs` (Swagger UI) and `/openapi.json` (raw schema).

---

## Authentication

The API is unauthenticated in local and Docker Compose deployments. When deploying behind a reverse proxy in a shared environment, add an `X-API-Key` header at the proxy layer. The application itself does not validate credentials, this is intentional to keep the inference path fast.

---

## Content types

All request bodies must be `application/json`. All responses are `application/json` unless otherwise noted. WebSocket messages are UTF-8 encoded JSON strings.

---

## Error format

All error responses follow a single envelope:

```json
{
  "error": {
    "code": "INVALID_SPIKE_TIMES",
    "message": "spike_times contains timestamps outside [t_start, t_stop]",
    "detail": {
      "unit_id": 3,
      "offending_time": 1.823,
      "t_stop": 1.5
    }
  }
}
```

| Field | Type | Description |
|---|---|---|
| `error.code` | string | Machine-readable error identifier. See [Error codes](#error-codes). |
| `error.message` | string | Human-readable description. Safe to surface to end users. |
| `error.detail` | object \| null | Optional structured context. Shape varies by error code. |

---

## Endpoints

### POST /predict

Decode a hand-movement trajectory from a buffer of neural spike times. Applies the full inference pipeline: binning → Gaussian smoothing → PCA projection → LSTM forward pass → MC Dropout uncertainty estimation.

**Request body** — `SpikeBuffer`

```json
{
  "spike_times": [
    [0.012, 0.034, 0.089, 0.201],
    [0.021, 0.067, 0.143],
    []
  ],
  "unit_ids": [0, 1, 2],
  "t_start": 0.0,
  "t_stop": 0.5,
  "n_mc_samples": 50
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `spike_times` | `float[][]` | yes | — | Outer array: one list per unit. Inner array: spike timestamps in seconds, sorted ascending. |
| `unit_ids` | `int[]` | yes | — | Unit identifiers. Must match the unit IDs used during training. Length must equal `len(spike_times)`. |
| `t_start` | `float` | no | `0.0` | Start of the decoding window in seconds. Must be ≥ 0. |
| `t_stop` | `float` | yes | — | End of the decoding window in seconds. Must be > `t_start`. Maximum window: 10.0 s. |
| `n_mc_samples` | `int` | no | `50` | Number of Monte Carlo Dropout forward passes for uncertainty estimation. Range: 1–200. Higher values give tighter confidence intervals at the cost of latency (roughly linear scaling). Use `1` to disable uncertainty and return point estimates only. |

**Response** — `TrajectoryResponse`

```json
{
  "timesteps": [
    {
      "t_ms": 0.0,
      "x": 0.124,
      "y": -0.341,
      "x_std": 0.021,
      "y_std": 0.018,
      "confidence": 0.94
    },
    {
      "t_ms": 50.0,
      "x": 0.187,
      "y": -0.298,
      "x_std": 0.038,
      "y_std": 0.041,
      "confidence": 0.81
    }
  ],
  "n_timesteps": 10,
  "bin_width_ms": 50.0,
  "model_version": "3",
  "n_mc_samples": 50,
  "latency_ms": 18.4
}
```

| Field | Type | Description |
|---|---|---|
| `timesteps` | `TrajectoryPoint[]` | Decoded trajectory. One point per time bin. Length = `ceil((t_stop - t_start) / bin_width_s)`. |
| `timesteps[].t_ms` | `float` | Bin centre time relative to `t_start`, in milliseconds. |
| `timesteps[].x` | `float` | Decoded x-position. Units depend on training data (typically cm, normalised to the workspace). |
| `timesteps[].y` | `float` | Decoded y-position. Same units as `x`. |
| `timesteps[].x_std` | `float` | Standard deviation of x across MC Dropout samples. Zero when `n_mc_samples=1`. |
| `timesteps[].y_std` | `float` | Standard deviation of y across MC Dropout samples. Zero when `n_mc_samples=1`. |
| `timesteps[].confidence` | `float` | Scalar confidence in [0, 1]. Derived from the heteroscedastic log-variance head: `confidence = exp(-mean_logvar)` clipped to [0, 1]. Independent of MC Dropout; available even when `n_mc_samples=1`. |
| `n_timesteps` | `int` | Number of decoded bins. |
| `bin_width_ms` | `float` | Bin width used for decoding. Reflects the value the @champion model was trained with. |
| `model_version` | `string` | MLflow model version string of the @champion model currently loaded. |
| `n_mc_samples` | `int` | Actual number of MC passes performed. Echoes the request value. |
| `latency_ms` | `float` | Server-side wall-clock time for the inference call, excluding network. |

**Status codes**

| Code | Meaning |
|---|---|
| `200` | Successful decode. |
| `400` | Malformed request. See `error.code` for specifics. |
| `503` | Model not loaded. `/ready` returns `not_ready`. |

---

### POST /predict/batch

Decode multiple independent spike buffers in a single request. Internally runs buffers sequentially (not parallelised). Useful for offline analysis where minimising round-trips matters more than minimising total latency.

**Request body**

```json
{
  "buffers": [
    { "spike_times": [[...], [...]], "unit_ids": [0, 1], "t_start": 0.0, "t_stop": 0.5 },
    { "spike_times": [[...], [...]], "unit_ids": [0, 1], "t_start": 0.5, "t_stop": 1.0 }
  ],
  "n_mc_samples": 20
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `buffers` | `SpikeBuffer[]` | yes | List of spike buffers. `n_mc_samples` from each buffer is ignored; the top-level value is used. Maximum 100 buffers per request. |
| `n_mc_samples` | `int` | no | Applied uniformly across all buffers. Default: `50`. |

**Response**

```json
{
  "results": [ { ...TrajectoryResponse... }, { ...TrajectoryResponse... } ],
  "total_buffers": 2,
  "total_latency_ms": 34.7
}
```

---

### GET /health

Liveness probe. Returns 200 as long as the process is running, regardless of model state. Use this for container liveness checks.

**Response**

```json
{ "status": "ok" }
```

---

### GET /ready

Readiness probe. Returns 200 only when the @champion model and its companion PCA object are both loaded and the API can serve predictions. Use this for container readiness checks and as a pre-flight check before sending `/predict` requests.

**Response — ready**

```json
{
  "status": "ready",
  "model_version": "3",
  "model_alias": "champion",
  "n_units_expected": 64,
  "pca_components": 15,
  "bin_width_ms": 50.0
}
```

**Response — not ready** (HTTP 503)

```json
{
  "status": "not_ready",
  "reason": "model loading"
}
```

| `reason` value | Meaning |
|---|---|
| `"model loading"` | Application is still starting; the MLflow model fetch has not completed. |
| `"pca loading"` | Model loaded but the companion PCA artifact fetch failed or is in progress. |
| `"no champion"` | MLflow registry has no model registered under the `@champion` alias. |

---

### GET /model/info

Returns metadata about the currently loaded model. Does not trigger inference.

**Response**

```json
{
  "model_name": "neural-spiketrain-analysis",
  "model_version": "3",
  "model_alias": "champion",
  "run_id": "a3f82b19c04d47e8b6f12d3c5e917a40",
  "architecture": "lstm",
  "hidden_size": 256,
  "num_layers": 2,
  "dropout": 0.3,
  "pca_components": 15,
  "bin_width_ms": 50.0,
  "training_r2_mean": 0.847,
  "training_r2_std": 0.031,
  "trained_at": "2024-03-15T14:22:01Z",
  "mlflow_tracking_uri": "http://mlflow:5000"
}
```

---

### GET /metrics

Prometheus-compatible plain-text metrics. Mount this on a Prometheus scrape target.

**Response** (text/plain)

```
# HELP ntd_requests_total Total inference requests
# TYPE ntd_requests_total counter
ntd_requests_total{endpoint="/predict",status="200"} 1482
ntd_requests_total{endpoint="/predict",status="400"} 3
ntd_requests_total{endpoint="/predict/batch",status="200"} 67

# HELP ntd_latency_ms_histogram Inference latency in milliseconds
# TYPE ntd_latency_ms_histogram histogram
ntd_latency_ms_histogram_bucket{le="10"} 312
ntd_latency_ms_histogram_bucket{le="25"} 1201
ntd_latency_ms_histogram_bucket{le="50"} 1479
ntd_latency_ms_histogram_bucket{le="+Inf"} 1482
ntd_latency_ms_histogram_sum 26841.3
ntd_latency_ms_histogram_count 1482

# HELP ntd_mc_samples_mean Mean MC Dropout samples per request
# TYPE ntd_mc_samples_mean gauge
ntd_mc_samples_mean 47.2

# HELP ntd_model_version Currently loaded model version
# TYPE ntd_model_version gauge
ntd_model_version{version="3"} 1
```

---

### WebSocket /ws/decode

Real-time streaming endpoint. The client sends spike buffer chunks as JSON messages and receives decoded trajectory points as they are computed, one JSON message per time bin.

**Connection**

```
ws://localhost:8000/ws/decode
```

No authentication headers required. The connection is accepted immediately; the server does not wait for the model to load — if the model is not ready the server sends an `error` frame and closes the connection with code 1011.

**Client → server message** (send one per decoding window)

```json
{
  "type": "decode",
  "spike_times": [[0.012, 0.089], [0.021, 0.143]],
  "unit_ids": [0, 1],
  "t_start": 0.0,
  "t_stop": 0.5,
  "n_mc_samples": 30
}
```

**Server → client messages**

The server streams one `point` message per decoded bin, followed by a single `done` message at the end of the window.

```json
{ "type": "point", "t_ms": 0.0,  "x": 0.124, "y": -0.341, "x_std": 0.021, "y_std": 0.018, "confidence": 0.94 }
{ "type": "point", "t_ms": 50.0, "x": 0.187, "y": -0.298, "x_std": 0.038, "y_std": 0.041, "confidence": 0.81 }
{ "type": "done",  "n_points": 10, "latency_ms": 21.3 }
```

**Error frame**

```json
{ "type": "error", "code": "MODEL_NOT_READY", "message": "No champion model loaded" }
```

After sending an error frame the server closes the connection with WebSocket close code 1011.

**Disconnect behaviour**

If the client disconnects mid-stream, the server cancels the remaining bins and releases the inference resources immediately. No partial results are buffered server-side.

**JavaScript client example**

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/decode');
const points = [];

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'decode',
    spike_times: [[0.012, 0.089, 0.201], [0.021, 0.143]],
    unit_ids: [0, 1],
    t_start: 0.0,
    t_stop: 0.5,
    n_mc_samples: 30
  }));
};

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  if (msg.type === 'point') {
    points.push(msg);
    renderPoint(msg);           // draw incrementally
  } else if (msg.type === 'done') {
    console.log(`Decoded ${msg.n_points} bins in ${msg.latency_ms} ms`);
    ws.close();
  } else if (msg.type === 'error') {
    console.error(msg.message);
    ws.close();
  }
};
```

---

## Data schemas

### SpikeBuffer

```
SpikeBuffer
├── spike_times:    float[][]   One list of timestamps per unit. Sorted ascending. Seconds.
├── unit_ids:       int[]       Unit identifiers. Length must equal len(spike_times).
├── t_start:        float       Window start. Seconds. Default: 0.0.
├── t_stop:         float       Window end. Seconds. Must be > t_start.
└── n_mc_samples:   int         MC Dropout passes. Range 1–200. Default: 50.
```

### TrajectoryPoint

```
TrajectoryPoint
├── t_ms:           float       Bin centre time from t_start. Milliseconds.
├── x:              float       Decoded x-position. Training-space units.
├── y:              float       Decoded y-position. Training-space units.
├── x_std:          float       MC Dropout std dev of x. 0 when n_mc_samples=1.
├── y_std:          float       MC Dropout std dev of y. 0 when n_mc_samples=1.
└── confidence:     float       Heteroscedastic confidence. Range [0, 1].
```

### TrajectoryResponse

```
TrajectoryResponse
├── timesteps:      TrajectoryPoint[]
├── n_timesteps:    int
├── bin_width_ms:   float
├── model_version:  str
├── n_mc_samples:   int
└── latency_ms:     float
```

---

## Latency targets

All targets measured on CPU (Intel Xeon, 2.4 GHz, single thread) with a 500 ms spike buffer and `n_mc_samples=50`.

| Stage | Target | Notes |
|---|---|---|
| Spike binning | < 1 ms | NumPy vectorised; scales linearly with buffer length |
| Gaussian smoothing | < 1 ms | `scipy.ndimage.gaussian_filter1d` on (T, N) array |
| PCA transform | < 1 ms | sklearn `transform` on (T, 15) |
| LSTM forward pass | < 5 ms | Single sequence, CPU inference |
| MC Dropout × 50 | < 40 ms | 50 sequential forward passes |
| JSON serialisation | < 2 ms | Pydantic model dump |
| **Total /predict** | **< 50 ms** | End-to-end server-side wall clock |

Setting `n_mc_samples=1` reduces total latency to approximately 10 ms by eliminating the MC loop. Use this when confidence intervals are not needed (e.g., during rapid prototyping or when feeding a downstream controller that ignores uncertainty).

---

## Error codes

| Code | HTTP | Meaning | Resolution |
|---|---|---|---|
| `UNIT_ID_MISMATCH` | 400 | `len(unit_ids) != len(spike_times)` | Ensure both arrays have the same length. |
| `UNKNOWN_UNIT_ID` | 400 | A `unit_id` was not seen during model training. | Only pass unit IDs from the session used to train the loaded champion. |
| `INVALID_SPIKE_TIMES` | 400 | Timestamps not sorted, negative, or outside `[t_start, t_stop]`. | Sort timestamps and clip to the window. |
| `WINDOW_TOO_LONG` | 400 | `t_stop - t_start > 10.0` seconds. | Split into shorter windows or use `/predict/batch`. |
| `WINDOW_TOO_SHORT` | 400 | Window produces fewer than 2 bins. | Increase `t_stop` or decrease `bin_width_ms` via model config. |
| `MC_SAMPLES_OUT_OF_RANGE` | 400 | `n_mc_samples` < 1 or > 200. | Use a value in [1, 200]. |
| `MODEL_NOT_READY` | 503 | No champion model loaded. | Check `/ready` before sending requests. |
| `BATCH_TOO_LARGE` | 400 | More than 100 buffers in `/predict/batch`. | Split into multiple requests. |
| `INTERNAL_ERROR` | 500 | Unexpected server-side failure. | Check server logs. File a bug report with the `latency_ms` and request payload. |

---

## Rate limits

No rate limiting is applied at the application layer. In production environments, apply rate limiting at the reverse proxy (e.g., nginx `limit_req`, Traefik middleware). Recommended starting point: 100 requests/second per client for `/predict`, unlimited for `/health` and `/ready`.

---

## Examples

### Minimal predict call (curl)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "spike_times": [
      [0.012, 0.089, 0.201, 0.334],
      [0.021, 0.143, 0.287],
      [0.055, 0.178, 0.312, 0.445]
    ],
    "unit_ids": [0, 1, 2],
    "t_stop": 0.5
  }'
```

### Point-estimate only (n_mc_samples=1)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "spike_times": [[0.012, 0.089], [0.021, 0.143]],
    "unit_ids": [0, 1],
    "t_stop": 0.5,
    "n_mc_samples": 1
  }'
```

### Python client using httpx

```python
import httpx

client = httpx.Client(base_url="http://localhost:8000", timeout=5.0)

# Pre-flight check
ready = client.get("/ready")
assert ready.json()["status"] == "ready", "Model not loaded"

# Decode
response = client.post("/predict", json={
    "spike_times": [[0.012, 0.089, 0.201], [0.021, 0.143]],
    "unit_ids": [0, 1],
    "t_stop": 0.5,
    "n_mc_samples": 50,
})

trajectory = response.json()
for point in trajectory["timesteps"]:
    print(f"t={point['t_ms']:5.0f} ms  x={point['x']:+.3f}  y={point['y']:+.3f}  "
          f"confidence={point['confidence']:.2f}")
```

### Batch decode (Python)

```python
buffers = [
    {"spike_times": [[0.012, 0.089], [0.021]], "unit_ids": [0, 1],
     "t_start": 0.0, "t_stop": 0.5},
    {"spike_times": [[0.534, 0.612], [0.501, 0.678]], "unit_ids": [0, 1],
     "t_start": 0.5, "t_stop": 1.0},
]

response = client.post("/predict/batch", json={
    "buffers": buffers,
    "n_mc_samples": 20,
})

for i, result in enumerate(response.json()["results"]):
    print(f"Buffer {i}: {result['n_timesteps']} timesteps, "
          f"latency={result['latency_ms']:.1f} ms")
```
