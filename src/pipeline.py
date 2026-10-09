"""One-call orchestration of the whole flow.

The Streamlit app and the CLI both want the same sequence:

    load CSV -> clean -> engineer features -> baselines -> train/evaluate
    -> impact simulation

Keeping it here means the app cannot drift away from the offline pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.decision import simulate_impact
from src.model import (
    DEFAULT_MODEL_PATH,
    data_signature,
    load_bundle,
    save_bundle,
    train_and_evaluate,
)
from src.preprocess import PreparedData, preprocess


def run_pipeline(
    source: str | Path | pd.DataFrame,
    capacity: float | None = None,
    train_fraction: float = 0.80,
) -> dict[str, Any]:
    """Run preprocessing, baselines, training and the impact simulation."""
    prepared: PreparedData = preprocess(source, capacity=capacity)
    bundle = train_and_evaluate(prepared, train_fraction=train_fraction)

    impact = simulate_impact(
        bundle["test_frame"],
        bundle["test_frame"]["pred"].to_numpy(),
        bundle["capacity"],
    )

    return {
        "prepared": prepared,
        "bundle": bundle,
        "impact": impact,
        "loaded_from_disk": False,
    }


def run_and_save(
    source: str | Path | pd.DataFrame,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    capacity: float | None = None,
) -> dict[str, Any]:
    """Same as :func:`run_pipeline` but also persists the bundle with joblib."""
    artefacts = run_pipeline(source, capacity=capacity)
    save_bundle(artefacts["bundle"], model_path)
    artefacts["model_path"] = Path(model_path)
    return artefacts


def load_or_train(
    source: str | Path | pd.DataFrame,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    capacity: float | None = None,
) -> dict[str, Any]:
    """Reuse a saved bundle for the *same* dataset, otherwise train and save."""
    prepared = preprocess(source, capacity=capacity)
    signature = data_signature(prepared.features, prepared.capacity)

    try:
        bundle = load_bundle(model_path)
        if bundle.get("signature") == signature:
            impact = simulate_impact(
                bundle["test_frame"], bundle["test_frame"]["pred"].to_numpy(), bundle["capacity"]
            )
            return {
                "prepared": prepared,
                "bundle": bundle,
                "impact": impact,
                "loaded_from_disk": True,
                "model_path": Path(model_path),
            }
    except (FileNotFoundError, KeyError, OSError):
        pass  # fall through and retrain

    artefacts = run_pipeline(source, capacity=capacity)
    save_bundle(artefacts["bundle"], model_path)
    artefacts["model_path"] = Path(model_path)
    return artefacts


__all__ = [
    "run_pipeline",
    "run_and_save",
    "load_or_train",
    "preprocess",
    "train_and_evaluate",
    "simulate_impact",
]
