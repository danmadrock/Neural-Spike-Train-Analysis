from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from prefect import flow, get_run_logger, task

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
def bin_task() -> dict:
    Path("data/binned").mkdir(parents=True, exist_ok=True)
    return {"status": "stub"}


@task
def reduce_task() -> dict:
    Path("data/pca").mkdir(parents=True, exist_ok=True)
    return {"status": "stub"}


@task
def train_task() -> dict:
    return {"status": "stub"}


@flow
def training_flow(stage: str = "all") -> dict:
    if stage == "sort":
        return {"sort": sort_task()}
    if stage == "bin":
        return {"bin": bin_task()}
    if stage == "reduce":
        return {"reduce": reduce_task()}
    if stage == "train":
        return {"train": train_task()}

    return {
        "sort": sort_task(),
        "bin": bin_task(),
        "reduce": reduce_task(),
        "train": train_task(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "sort", "bin", "reduce", "train"],
    )
    args = parser.parse_args()
    training_flow(stage=args.stage)
