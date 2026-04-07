"""Model training entrypoint with MLflow logging."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import lightning as L
import mlflow
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from src.models.gru_decoder import GRUDecoder
from src.models.losses import trajectory_loss
from src.models.lstm_decoder import LSTMDecoder
from src.models.wiener import evaluate_wiener_and_log
from src.training.metrics import r2_score_np, r2_score_torch, rmse_torch


@dataclass
class TrialDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    latents: np.ndarray
    trajectory: np.ndarray

    def __post_init__(self) -> None:
        self.x = torch.as_tensor(self.latents, dtype=torch.float32)
        self.y = torch.as_tensor(self.trajectory, dtype=torch.float32)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class DecoderLightningModule(L.LightningModule):
    """Lightning module that supports both LSTMDecoder and GRUDecoder."""

    def __init__(self, model: torch.nn.Module, cfg: DictConfig) -> None:
        super().__init__()
        self.model = model
        self.cfg = cfg

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(x)

    def _shared_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        pred_mean, pred_logvar = self(x)
        loss = trajectory_loss(
            pred_mean,
            pred_logvar,
            y,
            self.model,
            velocity_lambda=float(self.cfg.training.velocity_lambda),
            l1_lambda=float(self.cfg.training.l1_lambda),
        )
        return loss, pred_mean, y

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, _, _ = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, pred_mean, y = self._shared_step(batch)
        r2 = r2_score_torch(y, pred_mean)
        rmse = rmse_torch(y, pred_mean)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_r2", r2, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_rmse", rmse, prog_bar=False, on_epoch=True, on_step=False)
        return loss

    def on_train_start(self) -> None:
        cfg_container = OmegaConf.to_container(self.cfg, resolve=True)
        if not isinstance(cfg_container, dict):
            raise TypeError("Expected Hydra config container to be a dictionary")
        flat_cfg = _flatten_dict(cfg_container)
        mlflow.log_params(flat_cfg)

    def on_validation_epoch_end(self) -> None:
        metrics = self.trainer.callback_metrics
        payload = {
            "train_loss": float(
                metrics.get("train_loss_epoch", torch.tensor(float("nan"))).item()
            ),
            "val_loss": float(
                metrics.get("val_loss", torch.tensor(float("nan"))).item()
            ),
            "val_r2": float(metrics.get("val_r2", torch.tensor(float("nan"))).item()),
            "val_rmse": float(
                metrics.get("val_rmse", torch.tensor(float("nan"))).item()
            ),
        }
        mlflow.log_metrics(payload, step=int(self.current_epoch))

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=float(self.cfg.training.lr),
            weight_decay=float(self.cfg.training.weight_decay),
        )


def _flatten_dict(d: Mapping[Any, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, Mapping):
            out.update(_flatten_dict(v, key))
        elif isinstance(v, list):
            out[key] = json.dumps(v)
        elif v is None:
            out[key] = "null"
        else:
            out[key] = v
    return out


def _build_decoder(cfg: DictConfig, input_size: int) -> torch.nn.Module:
    kwargs = {
        "input_size": input_size,
        "hidden_size": int(cfg.model.hidden_size),
        "num_layers": int(cfg.model.num_layers),
        "dropout": float(cfg.model.dropout),
    }
    if cfg.model.type == "gru":
        return GRUDecoder(**kwargs)
    return LSTMDecoder(**kwargs)


def _load_or_synthesize_data(cfg: DictConfig) -> tuple[np.ndarray, np.ndarray]:
    pca_path = Path(cfg.training.latents_path)
    traj_path = Path(cfg.training.trajectory_path)
    if pca_path.exists() and traj_path.exists():
        return np.load(pca_path), np.load(traj_path)

    rng = np.random.default_rng(int(cfg.project.seed))
    n_trials = int(cfg.training.synthetic_trials)
    seq_len = int(cfg.training.synthetic_seq_len)
    k = int(cfg.training.synthetic_k)
    latents = rng.normal(size=(n_trials, seq_len, k))
    trajectory = np.zeros((n_trials, seq_len, 2), dtype=np.float32)
    w = rng.normal(size=(k, 2))
    for trial_idx in range(n_trials):
        for t in range(1, seq_len):
            linear_drive = latents[trial_idx, t] @ w
            nonlinear_drive = np.array(
                [
                    np.sin(latents[trial_idx, t - 1, 0] * 1.5),
                    np.tanh(latents[trial_idx, t - 1, 1] + latents[trial_idx, t, 2]),
                ]
            )
            trajectory[trial_idx, t] = (
                0.88 * trajectory[trial_idx, t - 1]
                + 0.06 * linear_drive
                + 0.20 * nonlinear_drive
            )
        trajectory[trial_idx] += 0.03 * rng.normal(size=(seq_len, 2))
    return latents.astype(np.float32), trajectory.astype(np.float32)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def run_training(cfg: DictConfig) -> None:
    latents, trajectory = _load_or_synthesize_data(cfg)
    n_trials = latents.shape[0]
    split = max(1, int(0.8 * n_trials))

    train_x, val_x = latents[:split], latents[split:]
    train_y, val_y = trajectory[:split], trajectory[split:]

    train_ds = TrialDataset(train_x, train_y)
    val_ds = TrialDataset(val_x, val_y)
    train_loader = DataLoader(
        train_ds, batch_size=int(cfg.training.batch_size), shuffle=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(cfg.training.batch_size), shuffle=False
    )

    mlflow.set_experiment(cfg.project.name)

    with mlflow.start_run(run_name="wiener_baseline"):
        train_flat_x = train_x.reshape(-1, train_x.shape[-1])
        train_flat_y = train_y.reshape(-1, 2)
        val_flat_x = val_x.reshape(-1, val_x.shape[-1])
        val_flat_y = val_y.reshape(-1, 2)
        _, wiener_r2 = evaluate_wiener_and_log(
            train_flat_x,
            train_flat_y,
            val_flat_x,
            val_flat_y,
        )

    with mlflow.start_run(run_name=f"{cfg.model.type}_decoder"):
        model = _build_decoder(cfg, input_size=latents.shape[-1])
        module = DecoderLightningModule(model, cfg)
        trainer = L.Trainer(
            max_epochs=int(cfg.training.max_epochs),
            callbacks=[
                EarlyStopping(
                    monitor="val_loss", mode="min", patience=int(cfg.training.patience)
                )
            ],
            enable_checkpointing=False,
            logger=False,
            deterministic=True,
        )
        trainer.fit(module, train_loader, val_loader)

        model.eval()
        with torch.no_grad():
            val_pred, _ = model(torch.as_tensor(val_x, dtype=torch.float32))
        val_r2 = r2_score_np(val_y, val_pred.numpy())
        mlflow.log_metric("final_val_r2", float(val_r2))
        mlflow.log_metric("beats_wiener", float(val_r2 > wiener_r2))

        output = {
            "wiener_r2": float(wiener_r2),
            "decoder_r2": float(val_r2),
            "decoder_type": str(cfg.model.type),
        }
        Path("training_summary.json").write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    run_training()
