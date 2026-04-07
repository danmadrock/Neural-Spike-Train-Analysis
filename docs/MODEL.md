# Model Documentation

## Contents

- [Problem formulation](#problem-formulation)
- [Model family](#model-family)
- [Architecture](#architecture)
  - [LSTMDecoder](#lstmdecoder)
  - [GRUDecoder](#grudecoder)
  - [Output heads](#output-heads)
- [Input representation](#input-representation)
  - [Spike binning](#spike-binning)
  - [PCA compression](#pca-compression)
- [Loss function](#loss-function)
- [Regularisation](#regularisation)
- [Training procedure](#training-procedure)
- [Uncertainty estimation](#uncertainty-estimation)
  - [MC Dropout — epistemic uncertainty](#mc-dropout--epistemic-uncertainty)
  - [Heteroscedastic head — aleatoric uncertainty](#heteroscedastic-head--aleatoric-uncertainty)
  - [Confidence score derivation](#confidence-score-derivation)
  - [Calibration](#calibration)
- [Baseline — Wiener filter](#baseline--wiener-filter)
- [Evaluation methodology](#evaluation-methodology)
  - [Leave-one-trial-out cross-validation](#leave-one-trial-out-cross-validation)
  - [Metrics](#metrics)
  - [Significance testing](#significance-testing)
- [Hyperparameter reference](#hyperparameter-reference)
- [Known limitations](#known-limitations)
- [Model versioning and registry](#model-versioning-and-registry)

---

## Problem formulation

Given multi-channel neural spike trains recorded from motor cortex during a reaching task, decode the continuous 2D hand trajectory (x, y position as a function of time).

This is a **sequence-to-sequence regression** problem:

```
Input:   X ∈ R^(T × K)   — T time bins of K PCA latent dimensions
Output:  Y ∈ R^(T × 2)   — decoded (x, y) position at each bin
```

where T is the number of time bins in the decoding window, and K = 15 (the number of PCA components retained). Both input and output are time-aligned — the model must produce one position estimate per input bin.

The task differs from classification in three fundamental ways that drive every design decision:

1. **Output is continuous**, not discrete. Loss is MSE/NLL, not cross-entropy.
2. **Temporal structure is load-bearing.** The sequence of neural states encodes motor preparation, execution, and deceleration. A model that ignores temporal order (e.g., a feedforward network applied bin-by-bin) loses this information.
3. **Uncertainty matters for safety.** A deployed BCI controller must know when to trust its own output. Returning a point estimate without uncertainty is insufficient for any application involving physical movement.

---

## Model family

Two architectures are provided with identical interfaces:

| Model | Class | Default hidden | Params (approx.) | Notes |
|---|---|---|---|---|
| LSTM decoder | `LSTMDecoder` | 256 | ~800 K | Slightly higher R² on most datasets; slower to train |
| GRU decoder | `GRUDecoder` | 256 | ~600 K | Trains ~30% faster; matches LSTM on short sequences |

Both are selected via Hydra config (`model=lstm` or `model=gru`) and share the same training loop, loss function, and evaluation code. The champion model in the registry may be either architecture — check `/model/info` to confirm.

---

## Architecture

### LSTMDecoder

```
Input: (batch, T, K)
│
├── LSTM layer 1
│     input_size:   K  (= n_pca_components, default 15)
│     hidden_size:  H  (default 256)
│     batch_first:  True
│     Output: (batch, T, H)
│
├── Dropout(p=0.3)
│     Applied to the full output sequence, not just the final state.
│     This is essential for MC Dropout at inference time —
│     if dropout only applied to inter-layer connections it
│     would not produce sample variance across forward passes.
│
├── LSTM layer 2
│     input_size:   H
│     hidden_size:  H
│     batch_first:  True
│     Output: (batch, T, H)
│
├── Dropout(p=0.3)
│
├── Mean head: Linear(H → 2)
│     Output: (batch, T, 2)  — predicted (x, y) per bin
│
└── Log-variance head: Linear(H → 2)
      Output: (batch, T, 2)  — log σ² per (x, y) per bin
```

The two output heads share the same LSTM backbone but have independent weight matrices. The mean head is optimised directly for prediction accuracy; the log-variance head is optimised to capture residual uncertainty not explained by the mean prediction.

**Why two LSTM layers?** A single layer can model the direct relationship between PCA latents and position, but misses higher-order temporal dependencies (e.g., velocity-dependent corrections, anticipatory deceleration). A second layer models the dynamics of the first layer's hidden states — empirically this adds 3–5% R² on typical motor cortex datasets.

**Why not bidirectional?** A bidirectional LSTM uses future time bins to inform current predictions, which is inappropriate for real-time decoding where future spikes are unavailable. The `bidirectional=False` default is intentional and must not be changed for online deployment.

### GRUDecoder

Identical structure to `LSTMDecoder` with `nn.GRU` replacing `nn.LSTM`. GRU has no cell state — it uses a single gated recurrent unit instead of the LSTM's input/forget/output gate trio. On shorter sequences (< 100 bins) GRU performs comparably to LSTM. On longer sequences (> 200 bins, rare in reaching tasks) LSTM tends to outperform.

### Output heads

Both heads are single `nn.Linear` layers applied to the full sequence output `(batch, T, H)`:

```python
mean_head    = nn.Linear(hidden_size, 2)   # predicts (x, y)
logvar_head  = nn.Linear(hidden_size, 2)   # predicts (log σ²_x, log σ²_y)
```

The log-variance parameterisation (rather than variance directly) is numerically stable and ensures the predicted variance is always positive: `σ² = exp(log σ²) > 0` for any finite output.

---

## Input representation

### Spike binning

Raw spike trains (lists of timestamps per neuron) are converted to a firing-rate matrix before being passed to the model.

**Algorithm:**

1. Divide `[t_start, t_stop]` into non-overlapping bins of width `bin_width_ms` (default 50 ms).
2. For each neuron and each bin, count the number of spikes falling within the bin boundaries.
3. Apply a Gaussian kernel smoother along the time axis: `scipy.ndimage.gaussian_filter1d(rate_matrix, sigma=sigma_bins, axis=0)`, where `sigma_bins = smoothing_sigma_ms / bin_width_ms` (default `sigma_bins = 0.5`).
4. Z-score each neuron's channel across the training set mean and standard deviation. These statistics are saved with the model and applied at inference time.

**Bin width choice:** 50 ms is standard for reaching tasks in motor cortex because it balances temporal resolution against the Poisson noise floor of low-firing neurons (~5–10 Hz baseline). Decreasing to 20 ms increases temporal resolution but introduces more noise; increasing to 100 ms smooths over movement-relevant modulations. The bin width is a training-time decision baked into the loaded model — the API reports the active value in `/ready` and `/model/info`.

**Output shape:** `(T_bins, N_units)` where `T_bins = ceil((t_stop - t_start) / bin_width_s)`.

### PCA compression

The `(T_bins, N_units)` firing-rate matrix is projected into a low-dimensional latent space before being fed to the LSTM.

**Motivation:** Motor cortex population activity lies on a low-dimensional manifold. The top 10–20 principal components typically explain >90% of task-relevant variance. The remaining components are noise dimensions that the LSTM would otherwise attempt to model, degrading generalisation. Compressing with PCA first provides three benefits: noise reduction, dimensionality reduction (smaller LSTM input size), and a canonical latent space that is consistent across sessions.

**Fitting protocol:**

- The PCA object is fitted **only on training data** within each LOO-CV fold.
- The fitted object is applied (transform only) to test data.
- This is enforced in `src/training/loo_cv.py` — the PCA fit call is inside the fold loop, after the train/test split.
- The final production PCA object (saved to MLflow) is fitted on all available trials after the LOO-CV evaluation confirms the model quality.

**Component selection:** The number of components `K` is set by `pca.n_components` in the config. A secondary check is applied: if `K` components explain less than `pca.variance_threshold` (default 0.90) of the variance, a warning is logged and K is increased automatically until the threshold is met. This prevents silent quality loss when the number of recorded neurons changes between sessions.

**Saved artefact:** The fitted `sklearn.decomposition.PCA` object is saved as a pickle alongside the model in MLflow. Both must be loaded together at inference time — the model is meaningless without the PCA that was fitted with it.

---

## Loss function

The total training loss combines three terms:

```
L_total = L_NLL + λ_vel × L_vel + λ_L1 × L_L1
```

### Heteroscedastic NLL (primary term)

```
L_NLL = (1/2T) Σ_t [ log_var_t + (y_t - ŷ_t)² / exp(log_var_t) ]
```

where `ŷ_t` is the mean head output and `log_var_t` is the log-variance head output at timestep t. This is the negative log-likelihood of a Gaussian distribution with learned variance. It simultaneously trains both output heads: the mean head minimises prediction error, and the log-variance head learns to predict when the mean head will be wrong.

When the model is accurate (small residual), the loss penalises large predicted variance. When the model is inaccurate (large residual), the loss penalises small predicted variance. This forces the log-variance head to learn genuine uncertainty rather than collapsing to a constant.

### Velocity smoothness term

```
L_vel = (1/(T-1)) Σ_t [ (ŷ_{t+1} - ŷ_t) - (y_{t+1} - y_t) ]²
```

MSE between predicted velocity and true velocity. Without this term the model can achieve low position MSE while producing jerky, biologically implausible trajectories (high-frequency oscillations in the decoded path). The velocity term regularises the temporal dynamics of the output.

Default weight: `λ_vel = 0.1`. If trajectories appear over-smoothed (rounded corners, delayed onset), reduce this weight. If trajectories are noisy, increase it.

### L1 regularisation

```
L_L1 = Σ_i |w_i|
```

Sum of absolute values of all model parameters. Encourages sparse weight matrices, which improves generalisation on small neural datasets where the number of training samples is limited. Applied as an explicit loss term because PyTorch's AdamW does not support L1 natively (it applies L2 via weight decay).

Default weight: `λ_L1 = 1e-5`. This is intentionally small — L1 is a supplementary regulariser, not the primary mechanism.

---

## Regularisation

Three regularisation mechanisms are applied simultaneously:

| Mechanism | Config key | Default | Applied to |
|---|---|---|---|
| L2 (weight decay) | `training.weight_decay` | `1e-4` | All parameters via AdamW |
| L1 (explicit loss term) | `training.l1_lambda` | `1e-5` | All parameters |
| Dropout | `model.dropout` | `0.3` | Between LSTM layers and on full output sequence |

**L2 vs L1:** L2 penalises large weights quadratically, shrinking all weights toward zero proportionally. L1 penalises linearly, which causes small weights to go exactly to zero (sparse solutions). Both are useful for small datasets; L2 handles the large-weight problem, L1 handles the many-small-weights problem.

**Dropout rate:** 0.3 is standard for sequence models on small datasets. At 0.3, roughly one third of hidden units are zeroed per forward pass, forcing the network to learn redundant representations. Values above 0.5 typically hurt performance on short sequences by destroying too much temporal information.

**Early stopping:** Not strictly regularisation but acts as one. Training stops when validation R² fails to improve for `patience=20` consecutive epochs. The best checkpoint (highest validation R²) is restored at the end of training.

---

## Training procedure

1. **Data split:** Trials are divided per the LOO-CV scheme (see [Evaluation methodology](#evaluation-methodology)). For the final production training run, all trials are used.

2. **Initialisation:** LSTM weights use PyTorch default (uniform, scaled by `1/sqrt(hidden_size)`). Output head weights are initialised with `nn.init.xavier_uniform_`. The log-variance head bias is initialised to 0.0, so the model starts by predicting unit variance.

3. **Optimiser:** AdamW with `lr=1e-3`, `weight_decay=1e-4`, `betas=(0.9, 0.999)`.

4. **Learning rate schedule:** `ReduceLROnPlateau` monitoring validation R², factor=0.5, patience=10. If validation R² stalls, the learning rate is halved. This typically fires 2–3 times per training run.

5. **Gradient clipping:** `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`. Prevents exploding gradients in the early epochs when the log-variance head is still learning.

6. **Batch construction:** Sequences are padded to the longest trial in the batch. A mask is applied so padded positions do not contribute to the loss.

7. **Training loop:** Implemented as a PyTorch Lightning `LightningModule`. All epoch-level metrics (train loss, val loss, val R², val RMSE) are logged to MLflow at the end of each epoch via `mlflow.log_metrics`.

8. **Early stopping:** `EarlyStopping(monitor='val_r2', mode='max', patience=20)`. The best model checkpoint is saved via `ModelCheckpoint`.

9. **Final model:** After early stopping, the best checkpoint is reloaded, registered in the MLflow model registry, and the PCA artefact is attached to the same MLflow run.

---

## Uncertainty estimation

Two complementary uncertainty mechanisms are implemented. They measure different things and are both included in every prediction response.

### MC Dropout — epistemic uncertainty

**What it measures:** Model uncertainty due to limited training data. High when the input is far from the training distribution.

**How it works:** At inference time, Dropout layers are kept active by calling `model.train()` instead of `model.eval()`. The forward pass is run `n_mc_samples` times (default 50) with different random dropout masks each time. The mean of the `n_mc_samples` predictions is the point estimate; the standard deviation is the epistemic uncertainty.

```
ŷ_MC = (1/N) Σ_n ŷ_n          — point estimate (mean of samples)
σ_MC = std({ ŷ_n })            — epistemic uncertainty (std of samples)
```

**Interpretation:** `x_std` and `y_std` in the API response are `σ_MC` per axis. A value of 0.05 (in training-space units) means that 68% of the MC samples fell within ±0.05 of the mean. If training-space units are centimetres, this corresponds to ±5 mm uncertainty.

**Computational cost:** Each MC sample is one full LSTM forward pass. 50 samples ≈ 40 ms on CPU. Use `n_mc_samples=1` to disable the MC loop (returns `x_std=0`, `y_std=0`).

### Heteroscedastic head — aleatoric uncertainty

**What it measures:** Irreducible uncertainty intrinsic to the neural signal — noise that cannot be reduced by collecting more training data.

**How it works:** The log-variance head predicts `log σ²` directly at each timestep. This is trained with the NLL loss, which forces the head to learn when the mean head's predictions will be wrong. Unlike MC Dropout, this does not require multiple forward passes — it adds essentially zero latency.

**Sources of high aleatoric uncertainty:**
- Movement onset and offset (neural state transitions)
- Periods of electrode noise or artifact
- Low-firing neurons providing weak signal
- Novel movement directions not well-represented in training

### Confidence score derivation

The `confidence` field in the API response is derived from the heteroscedastic head alone (not MC Dropout):

```
mean_logvar = (log_var_x + log_var_y) / 2
confidence  = clip(exp(-mean_logvar), 0, 1)
```

When `log_var` is low (model confident), `exp(-log_var) > 1` → clipped to 1. When `log_var` is high (model uncertain), `exp(-log_var) → 0`. The result is a scalar in [0, 1] that provides a fast, single-pass confidence signal without the latency of MC Dropout.

**When to use which signal:**

| Signal | Use when |
|---|---|
| `confidence` | Fast go/no-go decision. Is the current prediction trustworthy? |
| `x_std`, `y_std` | Quantitative uncertainty bounds. How far off might the prediction be? |
| Both together | `confidence` is low AND `x_std` is high → definitely distrust. One high alone → less conclusive. |

### Calibration

A calibrated uncertainty estimate means: when the model says 90% confidence interval, 90% of true values actually fall within that interval.

Calibration is measured with reliability diagrams and the Expected Calibration Error (ECE). After training, run `src/uncertainty/calibration.py` on the LOO-CV predictions to generate:

- A reliability diagram (predicted confidence level vs observed coverage)
- ECE value (lower is better; < 0.05 is considered well-calibrated)

If ECE > 0.10, the model is miscalibrated. Common causes:
- Log-variance head not converging (check NLL term in loss curves)
- Training/test distribution mismatch (check LOO-CV vs final training R² gap)
- `n_mc_samples` too low (< 20 samples gives noisy variance estimates)

---

## Baseline — Wiener filter

The Wiener filter is a linear regression decoder that maps the binned firing-rate matrix directly to trajectory:

```
Ŷ = X_binned @ W + b
```

where `W ∈ R^(N_units × 2)` and `b ∈ R^2` are fitted with ordinary least squares on the training set.

The Wiener filter is always trained and evaluated under the same LOO-CV scheme as the LSTM. The LSTM result is only claimed as an improvement when:

1. LSTM LOO-CV R² > Wiener LOO-CV R² on every fold (or at least on the mean)
2. A paired t-test across fold scores is significant at p < 0.05

If the LSTM does not beat the Wiener filter, the most likely causes are:
- PCA leakage (PCA fitted before the train/test split)
- Too few training trials for the LSTM to generalise
- Overfitting — try increasing `training.weight_decay` or `model.dropout`
- The dataset has inherently linear structure — the Wiener filter is the correct model

---

## Evaluation methodology

### Leave-one-trial-out cross-validation

Each unique trial in the dataset serves as the test set exactly once. For a dataset with N trials:

```
for i in 0 .. N-1:
    train_trials = all trials except trial i
    test_trial   = trial i

    # CRITICAL: PCA must be fit inside the loop
    pca   = NeuralPCA(n_components=K).fit(rate_matrices[train_trials])
    model = LSTMDecoder(...)
    train(model, pca.transform(rate_matrices[train_trials]), trajectories[train_trials])
    pred  = decode(model, pca.transform(rate_matrices[test_trial]))
    scores[i] = r2_score(trajectories[test_trial], pred)

report mean(scores), std(scores)
```

LOO-CV is preferred over k-fold CV for neural decoding because:
- Neural datasets are small (typically 100–300 trials). LOO-CV uses maximally large training sets.
- The decoder must generalise across the natural trial-to-trial variability of motor cortex activity, and LOO-CV directly measures this.
- It avoids the ambiguity of which k to choose.

### Metrics

All metrics are computed per LOO-CV fold and reported as mean ± std across folds.

**R² (variance explained) — primary metric**

```
R² = 1 - SS_res / SS_tot
   = 1 - Σ(y - ŷ)² / Σ(y - ȳ)²
```

Reported separately for x and y axes, and as the average of both. R² = 0 means the model performs no better than predicting the mean trajectory; R² = 1 is perfect decoding. Published motor cortex BCI decoders typically achieve R² = 0.70–0.90 on well-isolated, stable recordings.

**RMSE (cm)**

```
RMSE = sqrt((1/T) Σ(y - ŷ)²)
```

In physical units (centimetres, if the training data was in cm). Interpretable as average position error per timestep.

**Pearson correlation coefficient**

```
r = Σ(y - ȳ)(ŷ - ŷ̄) / sqrt(Σ(y - ȳ)² × Σ(ŷ - ŷ̄)²)
```

Reported per axis. Less sensitive to global bias than R². A model that decodes the shape of the trajectory but has a consistent positional offset will have high r but low R².

**Velocity RMSE (cm/s)**

```
RMSE_vel = sqrt((1/(T-1)) Σ((ŷ_{t+1} - ŷ_t)/Δt - (y_{t+1} - y_t)/Δt)²)
```

Measures the quality of decoded movement dynamics independently of position accuracy.

**Expected Calibration Error (ECE)**

```
ECE = Σ_b (|B_b| / N) × |accuracy(B_b) - confidence(B_b)|
```

Measures uncertainty calibration. Computed on LOO-CV predictions. Should be < 0.05 for a well-calibrated model.

### Significance testing

Before claiming the LSTM beats the Wiener filter, perform a paired t-test on the per-fold R² scores:

```python
from scipy import stats
t_stat, p_value = stats.ttest_rel(lstm_r2_scores, wiener_r2_scores)
# Claim improvement only if p_value < 0.05 and mean(lstm) > mean(wiener)
```

This test is automatically run in `src/training/metrics.py` and the result is logged to MLflow.

---

## Hyperparameter reference

All hyperparameters are set in `configs/` and tracked in `params.yaml`. Every training run logs the full config to MLflow.

| Parameter | Config key | Default | Tuning notes |
|---|---|---|---|
| PCA components | `pca.n_components` | 15 | Increase if scree plot shows elbow beyond 15. Decrease if training R² >> LOO-CV R² (overfitting to noise). |
| PCA variance threshold | `pca.variance_threshold` | 0.90 | Auto-increases n_components until this fraction of variance is explained. |
| Bin width | `binning.bin_width_ms` | 50 | Decrease to 20 for higher temporal resolution. Increases noise. |
| Smoothing sigma | `binning.smoothing_sigma_ms` | 25 | Decrease for less smoothing. 0 disables smoothing. |
| Hidden size | `model.hidden_size` | 256 | 128 for small datasets (< 50 trials). 512 for large datasets (> 500 trials). |
| LSTM layers | `model.num_layers` | 2 | 1 layer is often sufficient. 3+ rarely helps and slows training. |
| Dropout | `model.dropout` | 0.3 | Increase to 0.5 if LOO-CV R² << training R². |
| Learning rate | `training.lr` | 1e-3 | Reduce to 5e-4 if loss is unstable early. |
| L2 weight decay | `training.weight_decay` | 1e-4 | Increase to 1e-3 if strongly overfitting. |
| L1 lambda | `training.l1_lambda` | 1e-5 | Increase to 1e-4 if weights are large and not sparse. |
| Velocity loss weight | `training.vel_weight` | 0.1 | Increase to 0.5 if trajectories are jerky. |
| Max epochs | `training.max_epochs` | 200 | Rarely reached due to early stopping. |
| Early stopping patience | `training.patience` | 20 | Reduce to 10 for fast experiments. |
| Gradient clip | `training.gradient_clip_val` | 1.0 | Increase to 5.0 if gradients are frequently clipped early. |
| Batch size | `training.batch_size` | 32 | Reduce if GPU OOM. Increase to 64 for larger datasets. |
| MC Dropout samples | (API parameter) | 50 | 20 sufficient for qualitative uncertainty; 100 for publication figures. |

---

## Known limitations

**Session-specificity.** The model is trained on sorted units from a specific recording session. Unit IDs are not stable across days — electrode drift changes which neurons are recorded and their waveform shapes. The model must be retrained for each new session. Cross-session transfer is not implemented.

**Spike sorting dependency.** Model quality is bounded by the quality of the upstream spike sorting. If the sorting has high ISI violation rates (> 2%) or low isolation distance (< 10), the signal-to-noise of the firing-rate matrix degrades and R² will be lower. Check `src/sorting/validator.py` output before blaming the decoder.

**Stationary assumption.** The model assumes the neural code is stationary within the session. If neural drift occurs over the recording period (gradual change in tuning curves due to tissue response), the decoder will degrade over time. No online adaptation is implemented.

**Maximum decoding window.** The LSTM was trained on trials of a fixed duration. Feeding windows significantly longer than the training trial length will produce unreliable outputs as the LSTM hidden state evolves into an uncharted region of state space. The API enforces a 10-second maximum window as a conservative safeguard.

**CPU-only inference.** The production Docker image (`Dockerfile.api`) installs CPU-only PyTorch. GPU inference is supported in the training image. For latency-sensitive applications requiring `n_mc_samples > 100`, rebuild `Dockerfile.api` with CUDA PyTorch.

---

## Model versioning and registry

Models are stored in the MLflow Model Registry under the name `neural-spiketrain-analysis`. The `@champion` alias points to the version currently served by the API.

**Promotion criteria** (enforced in `src/training/register.py`):

- LOO-CV R² mean > 0.60 (configurable threshold)
- Paired t-test vs Wiener filter: p < 0.05
- ECE < 0.10
- ISI violation rate of training units < 2%

A model that does not meet all four criteria cannot be promoted to `@champion`. This prevents accidentally deploying a worse model.

**Version history** is preserved in MLflow. To roll back to a previous version:

```bash
python src/training/register.py --run-id <previous_run_id> --force
```

The `--force` flag bypasses the promotion criteria check and is intended only for emergency rollbacks.
