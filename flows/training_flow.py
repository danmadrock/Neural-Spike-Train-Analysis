"""Prefect orchestration entrypoint for pipeline stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from prefect import flow, get_run_logger, task

from src.binning import bin_spikes
from src.sorting.aligner import align_snippets
from src.sorting.clusterer import cluster_waveforms
from src.sorting.detector import detect_spikes
from src.sorting.validator import validate_units


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
        msg = f"No sorted spike files found in {sorted_path}"
        raise FileNotFoundError(msg)

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
def reduce_task() -> dict:
    Path("data/pca").mkdir(parents=True, exist_ok=True)
    return {"status": "stub"}


@task
def train_task() -> dict:
    return {"status": "stub"}


@flow
def training_pipeline(
    stage: str, bin_width_ms: float = 50.0, t_stop: float | None = None
) -> None:
    stage = stage.lower()

    if stage == "sort":
        sort_task()
    elif stage == "bin":
        bin_task(bin_width_ms=bin_width_ms, t_stop=t_stop)
    elif stage in {"reduce", "train"}:
        logger = get_run_logger()
        logger.info("stage '%s' is currently a placeholder", stage)
    else:
        msg = f"Unknown stage: {stage}"
        raise ValueError(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Prefect training pipeline stages")
    parser.add_argument(
        "--stage", required=True, choices=["sort", "bin", "reduce", "train"]
    )
    parser.add_argument("--bin-width-ms", type=float, default=50.0)
    parser.add_argument("--t-stop", type=float, default=None)
    args = parser.parse_args()

    training_pipeline(
        stage=args.stage, bin_width_ms=args.bin_width_ms, t_stop=args.t_stop
    )
