"""Model registry promotion helpers with quality gates."""

from __future__ import annotations

import argparse

import mlflow
from mlflow import MlflowClient


def _metric_from_run(run_id: str, metric_key: str) -> float:
    run = MlflowClient().get_run(run_id)
    metrics = run.data.metrics
    if metric_key not in metrics:
        msg = f"Run {run_id} does not contain required metric '{metric_key}'"
        raise ValueError(msg)
    return float(metrics[metric_key])


def _resolve_model_version_for_run(run_id: str, model_name: str) -> str:
    client = MlflowClient()
    versions = client.search_model_versions(f"name = '{model_name}'")
    for version in versions:
        if version.run_id == run_id:
            return str(version.version)
    msg = (
        "No registered model version found for "
        f"run_id={run_id} and model={model_name}"
    )
    raise ValueError(msg)


def _current_champion_score(model_name: str, metric_key: str) -> float | None:
    client = MlflowClient()
    try:
        champion = client.get_model_version_by_alias(model_name, "champion")
    except Exception:
        return None
    if champion.run_id is None:
        return None
    return _metric_from_run(champion.run_id, metric_key)


def promote_run_to_champion(
    run_id: str,
    *,
    model_name: str,
    min_r2_threshold: float,
    metric_key: str = "loo_cv_r2_mean",
) -> str:
    """Promote run to @champion if it passes threshold and beats current alias."""
    candidate_score = _metric_from_run(run_id, metric_key)
    if candidate_score < min_r2_threshold:
        msg = (
            f"Refusing promotion: {metric_key}={candidate_score:.4f} "
            f"is below minimum {min_r2_threshold:.4f}"
        )
        raise ValueError(msg)

    current_champion_score = _current_champion_score(model_name, metric_key)
    if current_champion_score is not None and candidate_score < current_champion_score:
        msg = (
            "Refusing promotion: candidate underperforms current champion "
            f"({candidate_score:.4f} < {current_champion_score:.4f})"
        )
        raise ValueError(msg)

    version = _resolve_model_version_for_run(run_id, model_name)
    client = MlflowClient()
    client.set_registered_model_alias(model_name, "champion", version)
    return version


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote an MLflow run to @champion")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--min-r2-threshold", type=float, required=True)
    parser.add_argument("--metric-key", default="loo_cv_r2_mean")
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args()

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    version = promote_run_to_champion(
        args.run_id,
        model_name=args.model_name,
        min_r2_threshold=args.min_r2_threshold,
        metric_key=args.metric_key,
    )
    print(
        f"Promoted run {args.run_id} to {args.model_name}@champion "
        f"(version={version}, metric={args.metric_key})"
    )


if __name__ == "__main__":
    main()
