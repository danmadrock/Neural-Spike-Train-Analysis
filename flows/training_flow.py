"""Prefect orchestration entrypoint for pipeline stages."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import mlflow
import numpy as np
from prefect import flow, get_run_logger, task

from src.binning import bin_spikes
from src.reduction import NeuralPCA, generate_variance_diagnostics
from src.sorting.aligner import align_snippets
from src.sorting.clusterer import cluster_waveforms
from src.sorting.detector import detect_spikes
from src.sorting.validator import validate_units
from src.training.register import promote_run_to_champion


@task
def sort_task(
    raw_dir: str = "data/raw",
    out_dir: str = "data/sorted",
    fs: float = 30_000.0,
    threshold_multiplier: float = 4.0,
) -> dict:
    logger = get_run_logger()
    raw_path = Path(raw_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(raw_path.glob("*.npy")) if raw_path.exists() else []
    if not npy_files:
        logger.warning(
            "No raw .npy files found in %s; creating empty sorted output.",
            raw_dir,
        )
        summary = {"sessions": 0, "units_written": 0}
        (out_path / "sorting_summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    units_written = 0
    for session_idx, file in enumerate(npy_files):
        voltage = np.load(file)
        if voltage.ndim == 1:
            voltage = voltage[None, :]

        snippets, timestamps = detect_spikes(
            voltage=voltage, fs=fs, threshold_multiplier=threshold_multiplier
        )
        aligned, _ = align_snippets(snippets)
        units = cluster_waveforms(aligned, timestamps)
        report = validate_units(units)

        for unit in units:
            unit_path = (
                out_path / f"session{session_idx:03d}_unit{unit.unit_id:03d}.npy"
            )
            np.save(unit_path, unit.spike_times)
            units_written += 1

        report_path = out_path / f"session{session_idx:03d}_quality.json"
        report_path.write_text(json.dumps(report, indent=2))

    summary = {"sessions": len(npy_files), "units_written": units_written}
    (out_path / "sorting_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


@task
def bin_task(
    sorted_dir: str = "data/sorted",
    output_dir: str = "data/binned",
    bin_width_ms: float = 50.0,
    t_stop: float | None = None,
) -> Path:
    """Load sorted unit spikes, bin them, and save one matrix to data/binned."""
    logger = get_run_logger()
    sorted_path = Path(sorted_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    spike_files = sorted(sorted_path.glob("*.npy"))
    if not spike_files:
        logger.warning(
            "No sorted spikes found in %s; writing empty matrix", sorted_path
        )
        out_file = out_path / "binned_matrix.npy"
        np.save(out_file, np.zeros((0, 0), dtype=np.float32))
        return out_file

    spike_trains = [np.load(path, allow_pickle=False) for path in spike_files]
    inferred_t_stop = max(
        (float(np.max(x)) for x in spike_trains if x.size > 0), default=0.0
    )
    final_t_stop = t_stop if t_stop is not None else inferred_t_stop
    if final_t_stop <= 0:
        msg = "Could not infer positive t_stop from spike data; pass --t-stop"
        raise ValueError(msg)

    binned = bin_spikes(spike_trains, bin_width_ms=bin_width_ms, t_stop=final_t_stop)
    out_file = out_path / "binned_matrix.npy"
    np.save(out_file, binned)

    logger.info("saved binned matrix to %s with shape %s", out_file, binned.shape)
    return out_file


@task
def reduce_task(
    binned_path: str = "data/binned/binned_matrix.npy",
    output_dir: str = "data/pca",
    n_components: int = 15,
    variance_threshold: float = 0.90,
) -> dict:
    logger = get_run_logger()
    binned_matrix = np.load(Path(binned_path), allow_pickle=False)
    if binned_matrix.ndim != 2:
        msg = f"Expected 2D binned matrix at {binned_path}, got {binned_matrix.shape}"
        raise ValueError(msg)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if (
        binned_matrix.size == 0
        or binned_matrix.shape[0] < 2
        or binned_matrix.shape[1] < 2
    ):
        logger.warning(
            "Binned matrix empty/too small; using deterministic synthetic matrix"
        )
        rng = np.random.default_rng(42)
        binned_matrix = rng.normal(size=(120, max(n_components, 16))).astype(np.float32)

    pca = NeuralPCA(n_components=n_components).fit(binned_matrix)
    pca_file = pca.save(out_path / "neural_pca.pkl")
    scree_file = out_path / "scree.png"
    threshold_component = generate_variance_diagnostics(
        neural_pca=pca,
        variance_threshold=variance_threshold,
        output_path=scree_file,
    )

    latents = pca.transform(binned_matrix).astype(np.float32)
    np.save(out_path / "latents.npy", latents)

    trajectory = np.zeros((latents.shape[0], 2), dtype=np.float32)
    trajectory[:, 0] = np.cumsum(latents[:, 0]) * 0.01
    trajectory[:, 1] = np.cumsum(latents[:, 1]) * 0.01
    np.save(out_path / "trajectory.npy", trajectory)

    logger.info(
        "saved fitted PCA to %s; component for %.2f variance=%d",
        pca_file,
        variance_threshold,
        threshold_component,
    )
    return {
        "pca_path": str(pca_file),
        "scree_path": str(scree_file),
        "variance_threshold_component": threshold_component,
    }


@task
def train_task() -> dict:
    logger = get_run_logger()
    subprocess.run(["python", "src/training/train.py"], check=True)

    summary_path = Path("training_summary.json")
    if not summary_path.exists():
        msg = "training_summary.json was not produced by training script"
        raise FileNotFoundError(msg)
    summary = json.loads(summary_path.read_text())

    run_id = summary.get("run_id")
    if not run_id:
        msg = "training summary does not include run_id"
        raise ValueError(msg)

    logger.info("Training complete with run_id=%s", run_id)
    return summary


@task
def register_task(
    run_id: str,
    model_name: str = "neural-spiketrain-analysis",
    min_r2_threshold: float = -1.0,
    metric_key: str = "loo_cv_r2_mean",
) -> dict:
    version = promote_run_to_champion(
        run_id,
        model_name=model_name,
        min_r2_threshold=min_r2_threshold,
        metric_key=metric_key,
    )
    return {"run_id": run_id, "model_version": version, "alias": "champion"}


@flow(name="ntd-training-flow")
def training_pipeline(
    stage: str = "all", bin_width_ms: float = 50.0, t_stop: float | None = None
) -> dict:
    stage = stage.lower()
    mlflow_tracking_uri = mlflow.get_tracking_uri()
    if stage == "sort":
        return {"sort": sort_task()}
    if stage == "bin":
        return {"bin": str(bin_task(bin_width_ms=bin_width_ms, t_stop=t_stop))}
    if stage == "reduce":
        return {"reduce": reduce_task()}
    if stage == "train":
        return {"train": train_task()}
    if stage == "register":
        train_result = train_task()
        return {
            "register": register_task(
                run_id=str(train_result["run_id"]),
                model_name="neural-spiketrain-analysis",
            )
        }
    if stage == "all":
        sort_task()
        binned_path = bin_task(bin_width_ms=bin_width_ms, t_stop=t_stop)
        reduce_task(binned_path=str(binned_path))
        train_result = train_task()
        register_result = register_task(
            run_id=str(train_result["run_id"]), model_name="neural-spiketrain-analysis"
        )
        return {
            "mlflow_tracking_uri": mlflow_tracking_uri,
            "train": train_result,
            "register": register_result,
        }

    msg = f"Unknown stage: {stage}"
    raise ValueError(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Prefect training pipeline stages")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["sort", "bin", "reduce", "train", "register", "all"],
    )
    parser.add_argument("--bin-width-ms", type=float, default=50.0)
    parser.add_argument("--t-stop", type=float, default=None)
    parser.add_argument(
        "--prefect-engine",
        action="store_true",
        help="Run via Prefect engine (requires reachable Prefect API).",
    )
    args = parser.parse_args()

    training_pipeline(
        stage=args.stage, bin_width_ms=args.bin_width_ms, t_stop=args.t_stop
    )
    if args.prefect_engine:
        training_pipeline(
            stage=args.stage, bin_width_ms=args.bin_width_ms, t_stop=args.t_stop
        )
    else:
        training_pipeline.fn(
            stage=args.stage, bin_width_ms=args.bin_width_ms, t_stop=args.t_stop
        )
