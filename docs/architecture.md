# Architecture

## Contents

- [System overview](#system-overview)
- [Component map](#component-map)
- [Data flow](#data-flow)
  - [Training path](#training-path)
  - [Inference path](#inference-path)
- [Component reference](#component-reference)
  - [Spike sorting (`src/sorting/`)](#spike-sorting-srcsorting)
  - [Spike binning (`src/binning/`)](#spike-binning-srcbinning)
  - [PCA reduction (`src/reduction/`)](#pca-reduction-srcreduction)
  - [Model layer (`src/models/`)](#model-layer-srcmodels)
  - [Uncertainty (`src/uncertainty/`)](#uncertainty-srcuncertainty)
  - [Training and validation (`src/training/`)](#training-and-validation-srctraining)
  - [Inference API (`src/api/`)](#inference-api-srcapi)
  - [Orchestration (`flows/`)](#orchestration-flows)
- [Serialisation contract](#serialisation-contract)
- [The training/serving skew boundary](#the-trainingserving-skew-boundary)
- [Infrastructure topology](#infrastructure-topology)
  - [Local development](#local-development)
  - [Docker Compose](#docker-compose)
- [Configuration system](#configuration-system)
- [Experiment tracking](#experiment-tracking)
- [Data versioning](#data-versioning)
- [Key design decisions](#key-design-decisions)
- [Dependency graph](#dependency-graph)

---

## System overview

The system has two operational modes that share critical components:

**Training mode** processes raw neural recordings through a seven-stage pipeline — spike sorting, binning, dimensionality reduction, model training, uncertainty calibration, validation, and model registration — producing an MLflow-tracked model artifact that is promoted to a named registry alias.

**Inference mode** loads that artifact at startup and exposes it through a FastAPI server that accepts raw spike buffers and returns decoded trajectories with per-timestep uncertainty bounds.

The two modes are deliberately decoupled. The training pipeline has no dependency on FastAPI; the inference API has no dependency on the training code. They communicate exclusively through the MLflow model registry — training writes to it, inference reads from it.

```
TRAINING                                    INFERENCE
────────                                    ─────────
Raw recordings                              Spike buffer (API request)
     │                                           │
Sorting → Binning → PCA → LSTM              Binning → PCA (loaded) → LSTM (loaded)
                              │                  │
                         MLflow registry    Trajectory + uncertainty
                              │
                         FastAPI (reads @champion at startup)
```

---

## Component map

```
neural-spiketrain-analysis/
│
├── src/
│   ├── sorting/          ← Stage 1: raw voltage → sorted spike trains
│   │   ├── detector.py   ← threshold crossing, snippet extraction
│   │   ├── aligner.py    ← sub-sample trough alignment
│   │   ├── clusterer.py  ← PCA waveforms → GMM unit labels
│   │   └── validator.py  ← ISI, SNR, isolation distance QC
│   │
│   ├── binning/          ← Stage 2: spike trains → firing-rate matrix
│   │   ├── binner.py     ← fixed-window spike counting
│   │   └── smoother.py   ← Gaussian kernel convolution
│   │
│   ├── reduction/        ← Stage 3: (T, N) → (T, K) via PCA
│   │   ├── pca.py        ← NeuralPCA wrapper; fit/transform/save/load
│   │   └── explained.py  ← scree plot, variance threshold logic
│   │
│   ├── models/           ← Stage 4: sequence decoder
│   │   ├── lstm_decoder.py  ← stacked LSTM + dual output heads
│   │   ├── gru_decoder.py   ← GRU variant, identical interface
│   │   ├── wiener.py        ← linear baseline decoder
│   │   └── losses.py        ← NLL + velocity + L1 composite loss
│   │
│   ├── uncertainty/      ← Stage 5: uncertainty quantification
│   │   ├── mc_dropout.py ← N-sample MC Dropout inference
│   │   └── calibration.py ← reliability diagrams, ECE
│   │
│   ├── training/         ← Stage 6: training loop + evaluation
│   │   ├── train.py      ← Lightning module, MLflow logging
│   │   ├── loo_cv.py     ← leave-one-trial-out CV splitter
│   │   ├── metrics.py    ← R², RMSE, correlation, velocity error
│   │   └── register.py   ← MLflow registry promotion with quality gate
│   │
│   └── api/              ← Stage 7: inference server
│       ├── main.py       ← FastAPI app, lifespan model loading
│       ├── schemas.py    ← Pydantic request/response types
│       ├── decoder.py    ← bin → PCA → LSTM → MC dropout pipeline
│       ├── websocket.py  ← streaming WebSocket endpoint
│       └── health.py     ← /health, /ready, /metrics
│
├── flows/
│   ├── training_flow.py      ← Prefect: stages 1–6 as a single flow
│   └── batch_decode_flow.py  ← Prefect: offline batch decoding
│
├── configs/              ← Hydra configuration tree
├── data/                 ← DVC-tracked data assets
├── docker/               ← Dockerfiles and compose file
├── tests/                ← Unit and integration tests
└── notebooks/            ← EDA and analysis notebooks
```

---

## Data flow

### Training path

```
data/raw/
  (EDF / NEV / MAT files)
         │
         │  src/sorting/detector.py
         │  src/sorting/aligner.py
         │  src/sorting/clusterer.py
         │  src/sorting/validator.py
         ▼
data/sorted/
  (one .npy timestamp array per sorted unit)
         │
         │  src/binning/binner.py
         │  src/binning/smoother.py
         ▼
data/binned/
  (T_bins × N_units firing-rate matrices, one per trial)
         │
         │  src/reduction/pca.py          ← fit on train trials only
         ▼
data/pca/
  (NeuralPCA fitted object + projected latents)
         │
         │  src/models/lstm_decoder.py    ← trained on latents
         │  src/models/losses.py
         │  src/training/train.py
         │  src/training/loo_cv.py        ← LOO-CV evaluation
         ▼
  MLflow run (metrics, artifacts, model)
         │
         │  src/training/register.py      ← quality gate + promotion
         ▼
  MLflow registry @champion
```

Each arrow represents a DVC stage with declared dependencies and outputs. Running `dvc repro` re-executes only the stages whose inputs have changed since the last run.

### Inference path

```
HTTP POST /predict
  { spike_times, unit_ids, t_start, t_stop, n_mc_samples }
         │
         │  src/api/decoder.py
         │
         ├── src/binning/binner.py        ← spike_times → rate_matrix
         ├── src/binning/smoother.py      ← Gaussian smooth
         ├── app.state.pca.transform()   ← (T, N) → (T, K)
         ├── app.state.model forward()   ← (T, K) → (T, 2) mean + logvar
         └── src/uncertainty/mc_dropout  ← N forward passes → std
         │
         ▼
HTTP 200
  { timesteps: [{ t_ms, x, y, x_std, y_std, confidence }], ... }
```

The inference path shares `src/binning/` and `src/reduction/` code directly with the training path. This is not incidental — it is the explicit mechanism that prevents training/serving skew. See [The training/serving skew boundary](#the-trainingserving-skew-boundary).

---

## Component reference

### Spike sorting (`src/sorting/`)

**Responsibility:** Convert raw extracellular voltage recordings into per-neuron spike timestamp arrays.

**`detector.py`**
Scans each channel for threshold crossings using the robust median estimator `threshold = -4 × median(|x| / 0.6745)`. Extracts a 45-sample (1.5 ms at 30 kHz) waveform snippet at each crossing. Returns `(snippets: ndarray[N, 45], times: ndarray[N])`.

**`aligner.py`**
Fits a parabola to the three samples around the minimum of each snippet and shifts the extraction window so the trough lands at sample index 15. This sub-sample alignment step is critical — misaligned waveforms increase within-cluster variance and degrade GMM separation.

**`clusterer.py`**
Projects each aligned snippet into 3 PCA dimensions (waveform PCA, distinct from the population PCA in `src/reduction/`). Fits a Gaussian Mixture Model with the number of components chosen by BIC minimisation over a range of 1–8 clusters. Returns `list[SortedUnit]` where each `SortedUnit` carries spike timestamps, mean waveform, and quality metrics.

**`validator.py`**
Computes per-unit quality metrics and flags units that fail acceptance thresholds. Does not drop flagged units — flagging is informational so the user can decide whether to exclude them.

| Metric | Threshold | Description |
|---|---|---|
| ISI violation rate | < 1% | Fraction of inter-spike intervals < 1.5 ms (refractory period). High values indicate multi-unit contamination. |
| SNR | > 3.0 | Peak-to-peak waveform amplitude / (2 × noise std). |
| Isolation distance | > 10.0 | Mahalanobis-distance-based cluster separation metric. |

**External dependency:** `spikeinterface` is available as an alternative backend for labs using Kilosort2 or other commercial sorters. The `SortedUnit` dataclass is compatible with SpikeInterface output via an adapter in `clusterer.py`.

---

### Spike binning (`src/binning/`)

**Responsibility:** Convert discrete spike timestamps into a continuous (T, N) firing-rate matrix.

**`binner.py`**
Implements a vectorised spike counter using `numpy.searchsorted` for speed. Accepts a list of spike timestamp arrays (one per sorted unit), a bin width in milliseconds, and the window boundaries. Returns an `ndarray[T_bins, N_units]` where each cell is the raw spike count in that bin for that unit.

**`smoother.py`**
Applies `scipy.ndimage.gaussian_filter1d` along the time axis (axis=0) with `sigma = smoothing_sigma_ms / bin_width_ms`. Operates in-place on a copy; the unsmoothed matrix is preserved for diagnostic purposes. A z-score normalisation step follows using per-unit mean and std computed on the training set.

**Invariant:** The smoothed, normalised matrix produced by these two modules must be byte-for-byte identical when called on the same input regardless of whether the call originates from the training pipeline or the inference API. This is enforced by the integration test `tests/integration/test_skew.py`.

---

### PCA reduction (`src/reduction/`)

**Responsibility:** Compress the (T, N) firing-rate matrix to a (T, K) latent matrix, isolating the low-dimensional manifold of population activity.

**`pca.py`** — `NeuralPCA` class

| Method | Description |
|---|---|
| `fit(X)` | Calls `sklearn.decomposition.PCA.fit(X)`. Stores the fitted sklearn object internally. May only be called once per instance. |
| `transform(X)` | Applies the fitted projection. Callable on any split (train, test, live data). |
| `plot_scree(path)` | Saves cumulative variance explained curve as PNG. |
| `save(path)` | Pickles the fitted sklearn PCA object. |
| `load(path)` | Class method. Loads and wraps a pickled object. |
| `n_components_for_threshold(threshold)` | Returns the minimum number of components explaining ≥ threshold fraction of variance. |

**`explained.py`**
Diagnostic utilities. Generates the scree plot and logs it as an MLflow artifact. Also implements the auto-increase logic: if `n_components` components explain less than `variance_threshold` of variance, the component count is increased to the minimum necessary and a warning is logged.

---

### Model layer (`src/models/`)

**Responsibility:** Define the LSTM/GRU sequence decoders and the linear baseline.

**`lstm_decoder.py`** — `LSTMDecoder(nn.Module)`
Stacked LSTM with dual output heads. `forward(x)` accepts `(batch, T, K)` and returns `(mean, log_var)`, both `(batch, T, 2)`. The module is designed so that Dropout layers remain active during MC Dropout inference without any modification to the module itself — the caller controls this by setting `model.train()` vs `model.eval()`.

**`gru_decoder.py`** — `GRUDecoder(nn.Module)`
Structurally identical to `LSTMDecoder` with `nn.GRU`. Shares the same `forward` signature. Training code uses duck typing — either decoder can be passed without modification.

**`wiener.py`** — `WienerFilter`
Wraps `scipy.linalg.lstsq` in a scikit-learn–compatible interface with `fit(X, y)` and `predict(X)`. Returns only a mean prediction; no uncertainty output. Used as the mandatory baseline for all LOO-CV comparisons.

**`losses.py`** — `trajectory_loss(pred_mean, pred_logvar, target, model, ...)`
Stateless function. Takes model predictions, ground truth, and the model reference (for L1 norm computation). Returns a scalar tensor with gradient attached. Separated from the model module so it can be tested independently.

---

### Uncertainty (`src/uncertainty/`)

**Responsibility:** Quantify both epistemic and aleatoric uncertainty at inference time.

**`mc_dropout.py`** — `mc_predict(model, latents, n_samples)`
Sets `model.train()`, runs `n_samples` forward passes collecting `mean` and `log_var` tensors, then restores `model.eval()`. Returns an `UncertaintyResult` dataclass:

```python
@dataclass
class UncertaintyResult:
    mean:     np.ndarray   # shape (T, 2) — average across MC samples
    std:      np.ndarray   # shape (T, 2) — std across MC samples
    logvar:   np.ndarray   # shape (T, 2) — mean log-variance from heteroscedastic head
    samples:  np.ndarray   # shape (n_samples, T, 2) — raw MC samples, if retain_samples=True
```

**`calibration.py`**
Takes LOO-CV predictions with uncertainty and ground-truth trajectories. Bins the prediction intervals by confidence level and measures empirical coverage. Computes ECE. Saves a reliability diagram as PNG. Logs ECE to MLflow.

---

### Training and validation (`src/training/`)

**Responsibility:** Execute the training loop, run LOO-CV evaluation, compute metrics, and manage the MLflow model registry.

**`train.py`** — `TrajectoryDecoderModule(LightningModule)`
PyTorch Lightning module. `training_step`, `validation_step`, and `configure_optimizers` are standard Lightning methods. MLflow autologging is disabled in favour of explicit per-epoch metric logging, which gives finer control over what is tracked.

**`loo_cv.py`** — `LeaveOneTrialOut`
Custom cross-validation splitter. Implements the scikit-learn splitter protocol (`split(trials)` yields `(train_indices, test_index)` pairs). Critically, the splitter is trial-aware — it understands that each "trial" may consist of multiple time bins, and ensures that all bins of a given trial are either in train or test, never split across both.

**`metrics.py`**
Pure functions, no side effects, no MLflow calls. All metric functions accept `(y_true: ndarray, y_pred: ndarray)` and return `float`. Tests in `tests/unit/training/test_metrics.py` verify each function against known-answer inputs.

**`register.py`**
Implements a quality gate before promotion. Fetches the candidate run's metrics from MLflow, evaluates against configurable thresholds, runs the paired t-test against the current Wiener baseline run, and promotes only if all criteria pass. The `@champion` alias is updated atomically in the MLflow client — there is no window where the registry has no champion.

---

### Inference API (`src/api/`)

**Responsibility:** Serve trajectory predictions over HTTP and WebSocket.

**`main.py`** — FastAPI application
Uses the `lifespan` context manager (FastAPI 0.95+ pattern) for startup/shutdown logic. On startup: (1) fetch `@champion` model from MLflow registry, (2) load companion PCA artifact from the same MLflow run, (3) store both on `app.state`. The `/ready` endpoint returns `not_ready` until both loads complete successfully.

**`schemas.py`** — Pydantic v2 models
All request and response types. Pydantic validation runs before any business logic — malformed requests are rejected at the schema boundary with structured error messages before reaching `decoder.py`.

**`decoder.py`** — `decode(spike_buffer, model, pca) → list[TrajectoryPoint]`
The single function that implements the inference pipeline. Calls `binner.py`, `smoother.py`, `pca.transform`, and `mc_dropout.py` in sequence. Has no FastAPI dependency — it is a pure Python function that can be called from tests, notebooks, or batch scripts without starting a server.

**`websocket.py`**
Implements the `/ws/decode` endpoint. Uses `asyncio` to run the decode function in a thread pool executor, preventing the LSTM forward pass from blocking the event loop. Streams individual `TrajectoryPoint` JSON messages as each bin is decoded.

**`health.py`**
Implements `/health` (always 200), `/ready` (200 only when model loaded), and `/metrics` (Prometheus text format). Metric counters are stored in module-level variables updated by middleware — no external metrics library required.

---

### Orchestration (`flows/`)

**Responsibility:** Wrap the pipeline stages in Prefect tasks and compose them into deployable flows.

**`training_flow.py`**
Defines `@task` wrappers around the stage functions in `src/`. The `@flow` itself is the composition of these tasks with retry policies and result caching. Key design principle: the task wrappers contain no logic — they call the `src/` functions directly. This keeps `src/` independently testable without Prefect.

**`batch_decode_flow.py`**
Offline batch trajectory decoding for analysing entire recording sessions. Loads the champion model once and passes multiple trial buffers through `decoder.decode()`. Results are saved to `data/decoded/` as structured numpy arrays.

---

## Serialisation contract

Two objects are serialised at training time and loaded at inference time:

| Object | Serialisation | Saved by | Loaded by |
|---|---|---|---|
| `NeuralPCA` (fitted sklearn PCA wrapper) | `pickle` | `src/training/register.py` | `src/api/main.py` lifespan |
| `LSTMDecoder` / `GRUDecoder` (PyTorch model) | `mlflow.pytorch.log_model` | `src/training/register.py` | `src/api/main.py` lifespan |

Both are logged as artifacts on the same MLflow run. They are always fetched together — the model is meaningless without the PCA that was fitted alongside it.

**Serialisation rules (must not be violated):**

1. No lambda functions in `NeuralPCA` — lambdas are not picklable.
2. No module-level mutable state in `LSTMDecoder` — state not in `self` is not captured by `state_dict`.
3. No `sklearn` transformers that use `partial_fit` — these carry internal batch counts that cause unexpected behaviour when loaded.
4. No references to file paths inside serialised objects — paths change between training and inference environments.

These rules are enforced by the integration test `tests/integration/test_serialisation.py`, which round-trips both objects through `mlflow.pytorch.log_model` / `load_model` and `pickle.dumps` / `loads`, then verifies that predictions before and after serialisation are numerically identical (within float32 tolerance).

---

## The training/serving skew boundary

Training/serving skew is the silent failure mode of ML pipelines: the data transformation applied during training differs subtly from the transformation applied at inference, causing the model to receive inputs it was never trained on.

In this project, the skew boundary is explicitly defined and tested:

```
┌─────────────────────────────────────────────────────┐
│  Training path                                      │
│                                                     │
│  raw voltage → sort → bin_spikes() → smooth() →    │
│  pca.transform() → LSTM forward → loss              │
└──────────────────────┬──────────────────────────────┘
                       │  These three functions must be
                       │  byte-for-byte identical in both paths
┌──────────────────────▼──────────────────────────────┐
│  Inference path                                     │
│                                                     │
│  spike_buffer → bin_spikes() → smooth() →           │
│  pca.transform() → LSTM forward → TrajectoryPoint   │
└─────────────────────────────────────────────────────┘
```

The three shared functions — `bin_spikes`, `gaussian_smooth`, and `pca.transform` — are imported from the same modules in both paths. There is no "training version" and "serving version" — there is one implementation.

The z-score normalisation statistics (per-unit mean and std) are computed on the training set, saved with the PCA object, and applied identically at inference time. This is the most common site of skew in real deployments — forgetting to save and apply the normalisation statistics.

`tests/integration/test_skew.py` verifies this by: (1) running the full training path on a synthetic dataset, (2) running the inference path on the same input, and (3) asserting that the intermediate tensors at each step are numerically identical.

---

## Infrastructure topology

### Local development

```
Developer machine
├── Python process: uvicorn src.api.main:app --reload     (port 8000)
├── Python process: mlflow server --port 5000              (port 5000)
└── Python process: prefect server start                   (port 4200)

Data:
├── data/                                                  (local DVC working dir)
├── mlruns/                                                (MLflow artifact store)
└── /tmp/dvc-store                                         (DVC remote, local)
```

### Docker Compose

```
docker-compose.yml
│
├── service: api
│   ├── image: ntd:api (Dockerfile.api, python:3.11-slim)
│   ├── ports: 8000:8000
│   ├── env: MLFLOW_TRACKING_URI=http://mlflow:5000
│   └── depends_on: mlflow (condition: service_healthy)
│
├── service: mlflow
│   ├── image: ghcr.io/mlflow/mlflow:latest
│   ├── ports: 5000:5000
│   ├── command: mlflow server --backend-store-uri sqlite:///mlruns.db
│   │            --default-artifact-root /mlartifacts --host 0.0.0.0
│   └── volumes: mlruns_data:/mlruns.db, mlartifacts:/mlartifacts
│
└── service: prefect-agent
    ├── image: ntd:train (Dockerfile.train, python:3.11)
    ├── command: prefect agent start --pool default-agent-pool
    └── env: MLFLOW_TRACKING_URI=http://mlflow:5000, PREFECT_API_URL=http://prefect:4200/api

volumes: mlruns_data, mlartifacts
```

**Why two Dockerfiles?**

`Dockerfile.train` installs the full scientific stack: PyTorch with CUDA support, SpikeInterface (which pulls in Kilosort2 dependencies), MLflow, Prefect, and all dev tools. This image is large (~4 GB) but is only used for training runs.

`Dockerfile.api` installs only what inference needs: FastAPI, Uvicorn, PyTorch (CPU only), NumPy, SciPy, and the MLflow client. This image is small (~600 MB) and is what gets deployed. Keeping it small reduces attack surface, cold start time, and egress costs.

The two images share the same `src/` code via a volume mount in development and via `COPY` at build time in production.

---

## Configuration system

Hydra composes the runtime configuration from a tree of YAML files:

```
configs/
├── config.yaml          ← root composer: defaults list + project-level keys
├── data/
│   └── default.yaml     ← recording format, channel count, trial structure
├── sorting/
│   └── default.yaml     ← threshold_multiplier, min_isi_ms, n_pca_components
├── binning/
│   └── default.yaml     ← bin_width_ms, smoothing_sigma_ms, normalize
├── pca/
│   └── default.yaml     ← n_components, variance_threshold, random_state
├── model/
│   ├── lstm.yaml        ← hidden_size, num_layers, dropout, bidirectional
│   └── gru.yaml         ← same keys, different values
└── training/
    └── default.yaml     ← lr, weight_decay, l1_lambda, max_epochs, patience
```

The full resolved config is logged to MLflow as a structured parameter dict at the start of every training run. This makes every experiment fully reproducible: the MLflow run ID carries both the model artifact and the exact config that produced it.

DVC's `params.yaml` mirrors the subset of Hydra config that DVC tracks as stage dependencies. When a parameter in `params.yaml` changes, DVC marks the downstream stages as stale and re-executes them on `dvc repro`.

---

## Experiment tracking

MLflow is used as the single source of truth for experiment history.

**What is tracked per run:**

| Category | Items |
|---|---|
| Parameters | Full Hydra config (flattened to dot-notation keys) |
| Metrics | Per-epoch: train_loss, val_loss, val_r2, val_rmse. Final: loo_cv_r2_mean, loo_cv_r2_std, wiener_r2_mean, paired_ttest_p, ece |
| Artifacts | Fitted PCA object, LSTM model, scree plot PNG, trajectory visualisations, reliability diagram, confusion matrices per LOO fold |
| Tags | architecture (lstm/gru), recording_session, git_commit_sha |

**Model registry:**

All production-quality models are registered under the name `neural-spiketrain-analysis`. The `@champion` alias is the only alias the API reads. Previous champions are preserved as numbered versions and can be restored by updating the alias without deleting any run.

---

## Data versioning

DVC tracks three categories of data:

**Raw inputs** (`data/raw/`) — original recording files. These are the source of truth. Never modified, only read.

**Intermediate outputs** (`data/sorted/`, `data/binned/`, `data/pca/`) — deterministic transformations of the raw inputs given the params in `params.yaml`. These can always be regenerated with `dvc repro` but are cached to avoid re-running expensive spike sorting on every experiment.

**`dvc.lock`** — the lockfile. Committed to git. Contains the content hash of every tracked file at every pipeline stage. Combined with a git commit SHA, this completely specifies a reproducible data state.

**Remote storage** — a DVC remote (S3, GCS, or local path) stores the actual data files. The git repository stores only the `.dvc` pointer files and `dvc.lock`. Collaborators run `dvc pull` to fetch the data matching the current `dvc.lock`.

---

## Key design decisions

**PCA fit is inside the LOO-CV loop, not before it.**
If PCA were fitted on all trials before the split, the test trial's variance structure would influence the projection used to encode it — a form of data leakage that inflates reported R². The current design makes leakage physically impossible: `NeuralPCA.fit()` can only be called once per instance, so fitting on the full dataset and then splitting cannot happen accidentally.

**`decoder.py` is a pure function, not a method on the FastAPI app.**
This means it can be tested without starting a server, called from notebooks, and used in the batch Prefect flow — all without any FastAPI dependency. The function signature is `decode(spike_buffer, model, pca)` and it has no global state.

**The API uses `app.state` for loaded models, not module-level globals.**
Module-level globals are unreachable from tests without importing the module (which triggers the load). `app.state` is injectable in tests via the TestClient `app.state.model = mock_model` pattern, making the API fully testable without a real MLflow server.

**Dropout layers are on the full LSTM output sequence, not just inter-layer.**
Standard PyTorch LSTM dropout only applies between layers, not to the output of the final layer. If dropout only applied between layers, MC Dropout inference would produce zero variance (all samples identical) from the final layer. The current implementation adds an explicit `nn.Dropout` after each LSTM layer's full output sequence, which ensures MC Dropout produces meaningful sample variance.

**The Wiener filter is a first-class model, not a footnote.**
It is implemented with the same interface as `LSTMDecoder`, runs under the same LOO-CV scheme, and its results are logged to the same MLflow experiment. The promotion gate in `register.py` requires the LSTM to significantly outperform it. This prevents promoting a model that is merely as good as a linear decoder — at the cost of LSTM complexity.

**`register.py` refuses to demote.** Promoting a new champion requires it to have a higher LOO-CV R² than the current champion, not just a positive absolute R². This prevents accidental model regression during automated retraining.

---

## Dependency graph

Module-level import dependencies (arrows mean "imports from"):

```
src/api/main.py
  └── src/api/decoder.py
        ├── src/binning/binner.py
        ├── src/binning/smoother.py
        ├── src/reduction/pca.py         (loaded from MLflow artifact)
        ├── src/models/lstm_decoder.py   (loaded from MLflow artifact)
        └── src/uncertainty/mc_dropout.py

src/training/train.py
  ├── src/models/lstm_decoder.py
  ├── src/models/losses.py
  ├── src/training/loo_cv.py
  │     └── src/reduction/pca.py
  └── src/training/metrics.py

flows/training_flow.py
  ├── src/sorting/detector.py
  ├── src/sorting/aligner.py
  ├── src/sorting/clusterer.py
  ├── src/sorting/validator.py
  ├── src/binning/binner.py
  ├── src/binning/smoother.py
  ├── src/reduction/pca.py
  └── src/training/train.py
```

No circular imports. The `src/api/` package does not import from `src/training/` or `flows/`. The `src/models/` package does not import from `src/api/` or `src/training/`. Dependencies flow in one direction: orchestration → training → models → utilities → API reads from registry.
