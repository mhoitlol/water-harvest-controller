"""Non-ML baselines. A model is only impressive if it beats these.

Two baselines are provided:

1. **Persistence** - "tomorrow looks like today". The next tank level is simply
   the current one. Tank levels are slow-moving, so this is a genuinely hard
   baseline to beat on MAE.
2. **Physics** - the water balance: next level = current + expected inflow -
   average usage, clipped to [0, capacity]. This is the same rule the tank
   obeys in the real world, so it should do well when sensors are healthy -
   which is exactly why it is also used as the **fallback** when a sensor fault
   is detected (see :mod:`src.decision`).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CAPACITY_SAFETY = 1e-9


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict:
    """Return MAE / RMSE / R2 in a tidy dict."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))),
        "r2": float(r2_score(y_true_arr, y_pred_arr)),
        "n": int(len(y_true_arr)),
    }


def persistence_baseline(current_level: Sequence[float]) -> np.ndarray:
    """Predicted next level == current level."""
    return np.asarray(current_level, dtype=float)


def physics_baseline(
    current_level: Sequence[float],
    expected_inflow: Sequence[float],
    average_usage: Sequence[float],
    capacity: float,
) -> np.ndarray:
    """Predicted next level == current + expected inflow - average usage (clipped)."""
    predicted = (
        np.asarray(current_level, dtype=float)
        + np.asarray(expected_inflow, dtype=float)
        - np.asarray(average_usage, dtype=float)
    )
    return np.clip(predicted, 0.0, float(capacity) - CAPACITY_SAFETY)


def baseline_predictions(
    frame: pd.DataFrame,
    capacity: float,
    average_usage_window: str = "usage_roll_mean_24",
) -> dict[str, np.ndarray]:
    """Compute both baseline prediction vectors for a feature frame.

    The frame must contain ``tank_level_litres`` (current level),
    ``inflow_litres`` (the best available expectation of next-hour inflow) and
    ``usage_roll_mean_24`` (typical recent demand).
    """
    if average_usage_window not in frame.columns:
        # Fall back to the plain mean if the rolling column is unavailable.
        average_usage_window = "usage_litres"

    return {
        "persistence": persistence_baseline(frame["tank_level_litres"]),
        "physics": physics_baseline(
            frame["tank_level_litres"],
            frame["inflow_litres"],
            frame[average_usage_window],
            capacity,
        ),
    }


def evaluate_baselines(
    frame: pd.DataFrame,
    capacity: float,
    target_column: str = "target_level_next",
) -> dict[str, dict]:
    """Evaluate both baselines on `frame` and report which one is best."""
    y_true = frame[target_column].to_numpy(dtype=float)
    results: dict[str, dict] = {}

    for name, predicted in baseline_predictions(frame, capacity).items():
        metrics = regression_metrics(y_true, predicted)
        metrics["name"] = name
        metrics["predictions"] = predicted
        results[name] = metrics

    best_name = min(results, key=lambda key: results[key]["mae"])
    results["best"] = {"name": best_name, "mae": results[best_name]["mae"]}
    return results


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    # Imported here (not at module level) to avoid a circular import: model.py
    # imports this module for the metrics helpers.
    from src.model import time_based_split
    from src.preprocess import preprocess

    prepared = preprocess("data/tank_data.csv")
    # Score on the same held-out test window the model is scored on, so the
    # numbers here line up with `python -m src.model`.
    split = time_based_split(prepared.features)
    baseline_results = evaluate_baselines(split["test"], prepared.capacity)

    print(
        f"Baselines on the held-out test window "
        f"({split['test_rows']} rows, {split['test_period'][0]} .. {split['test_period'][1]}):"
    )
    for key in ("persistence", "physics"):
        row = baseline_results[key]
        print(f"  {key:12s} MAE={row['mae']:8.2f}  RMSE={row['rmse']:8.2f}  R2={row['r2']:.4f}")
    print("  best baseline:", baseline_results["best"]["name"],
          f"(MAE {baseline_results['best']['mae']:.2f} L)")
