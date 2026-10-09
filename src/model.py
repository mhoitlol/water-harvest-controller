"""Model training, evaluation and persistence.

Design choices worth calling out:

* **Time-based split, never shuffled.** The first 80% of the timeline trains the
  model, the last 20% tests it. Random shuffling would let the model peek at
  the future (an hour either side of every test point would be in the training
  set), which inflates the score and would not survive contact with production.
* **A deliberately light model.** A Random Forest with a modest number of trees
  is accurate, robust to non-linear water dynamics, and fast enough to retrain
  inside the Streamlit app. Ridge regression is trained alongside it as an
  honest linear reference point.
* **Overflow warnings get their own metric.** Predicting the exact level is
  nice; *not missing an overflow* is what actually saves water, so we report
  precision/recall for the binary "next level > 90% capacity" event.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.baseline import evaluate_baselines, physics_baseline, regression_metrics
from src.estimator import TARGET_MODE, TankLevelModel
from src.preprocess import FEATURES, TARGET, PreparedData

RANDOM_SEED = 42
DEFAULT_MODEL_PATH = Path("models") / "tank_model.joblib"
OVERFLOW_WARNING_FRACTION = 0.90


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------
def data_signature(frame: pd.DataFrame, capacity: float) -> str:
    """Cheap fingerprint of a prepared dataset.

    A saved bundle is only reusable if it was trained on the same data - this
    is what stops the app from silently serving a model trained on a different
    upload.
    """
    return (
        f"rows={len(frame)};"
        f"start={frame['timestamp'].min()};"
        f"end={frame['timestamp'].max()};"
        f"capacity={float(capacity):.4f}"
    )


def time_based_split(
    frame: pd.DataFrame, train_fraction: float = 0.80
) -> dict[str, Any]:
    """Chronological 80/20 split. No shuffling, ever."""
    ordered = frame.sort_values("timestamp").reset_index(drop=True)
    cut = int(len(ordered) * train_fraction)
    train, test = ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()

    return {
        "train": train,
        "test": test,
        "X_train": train[FEATURES],
        "y_train": train[TARGET],
        "X_test": test[FEATURES],
        "y_test": test[TARGET],
        "train_fraction": train_fraction,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_period": (str(train["timestamp"].min()), str(train["timestamp"].max())),
        "test_period": (str(test["timestamp"].min()), str(test["timestamp"].max())),
    }


# --------------------------------------------------------------------------
# Candidate models
# --------------------------------------------------------------------------
def make_models(seed: int = RANDOM_SEED) -> dict[str, Any]:
    """Return the candidate models, keyed by display name."""
    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            # 20 samples per leaf is a deliberate amount of smoothing: the tank
            # level is close to a random walk, so a noisy forest that overshoots
            # would lose to the persistence baseline on MAE.
            min_samples_leaf=20,
            random_state=seed,
            n_jobs=-1,
        ),
        "Ridge": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0, random_state=None)),
            ]
        ),
    }


def feature_importance_of(model: Any, feature_names: list[str]) -> pd.Series:
    """Extract importances (tree) or absolute standardised coefficients (linear)."""
    estimator = model.estimator if isinstance(model, TankLevelModel) else model
    if isinstance(estimator, Pipeline):
        estimator = estimator.steps[-1][1]

    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_, dtype=float)).ravel()
    else:  # pragma: no cover - defensive
        values = np.zeros(len(feature_names), dtype=float)

    series = pd.Series(values, index=feature_names, name="importance")
    return series.sort_values(ascending=False)


def overflow_warning_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    capacity: float,
    threshold_fraction: float = OVERFLOW_WARNING_FRACTION,
) -> dict[str, float]:
    """Binary precision/recall for the "will the tank overflow?" warning.

    A missed overflow (false negative) is the costly error: the tank spills and
    the water is gone, whereas a false alarm only costs a needless drawdown.
    """
    limit = capacity * threshold_fraction
    actual = np.asarray(y_true, dtype=float) >= limit
    predicted = np.asarray(y_pred, dtype=float) >= limit

    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predicted, average="binary", zero_division=0
    )
    tn, fp, fn, tp = 0, 0, 0, 0
    for a, p in zip(actual, predicted):
        if a and p:
            tp += 1
        elif a and not p:
            fn += 1
        elif not a and p:
            fp += 1
        else:
            tn += 1

    return {
        "threshold_litres": float(limit),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "actual_positive": int(actual.sum()),
        "predicted_positive": int(predicted.sum()),
    }


# --------------------------------------------------------------------------
# Training + evaluation
# --------------------------------------------------------------------------
def train_and_evaluate(
    prepared: PreparedData,
    train_fraction: float = 0.80,
    seed: int = RANDOM_SEED,
    run_cv: bool = True,
    cv_splits: int = 5,
) -> dict[str, Any]:
    """Train every candidate model, score it, and pick a winner."""
    split = time_based_split(prepared.features, train_fraction)
    capacity = prepared.capacity
    X_train, y_train = split["X_train"], split["y_train"]
    X_test, y_test = split["X_test"], split["y_test"]

    # --- non-ML baselines, evaluated on exactly the same test rows ---------
    baselines = evaluate_baselines(split["test"], capacity)
    best_baseline_name = baselines["best"]["name"]
    best_baseline_mae = float(baselines[best_baseline_name]["mae"])

    # --- candidate models --------------------------------------------------
    # What we actually fit is the one-hour *change*; the current level is added
    # back by TankLevelModel.predict (see the note next to TARGET_MODE).
    delta_train = y_train.to_numpy(dtype=float) - X_train["tank_level_litres"].to_numpy(dtype=float)

    metrics: dict[str, dict] = {}
    overflow: dict[str, dict] = {}
    predictions: dict[str, np.ndarray] = {}
    fitted: dict[str, TankLevelModel] = {}
    estimators: dict[str, Any] = {}
    importance: dict[str, pd.Series] = {}

    for name, estimator in make_models(seed).items():
        estimator.fit(X_train, delta_train)
        model = TankLevelModel(estimator, list(FEATURES), capacity)
        y_pred = model.predict(X_test)

        scores = regression_metrics(y_test, y_pred)
        scores["name"] = name
        scores["improvement_pct"] = (
            (best_baseline_mae - scores["mae"]) / best_baseline_mae * 100.0
            if best_baseline_mae
            else 0.0
        )
        metrics[name] = scores
        overflow[name] = overflow_warning_metrics(y_test, y_pred, capacity)
        predictions[name] = y_pred
        fitted[name] = model
        estimators[name] = estimator
        importance[name] = feature_importance_of(model, list(FEATURES))

    # --- why the change-target instead of the raw level? -------------------
    direct_estimator = clone(estimators["RandomForest"])
    direct_estimator.fit(X_train, y_train)
    direct_pred = np.clip(direct_estimator.predict(X_test), 0.0, capacity)
    direct_scores = regression_metrics(y_test, direct_pred)
    direct_scores["improvement_pct"] = (
        (best_baseline_mae - direct_scores["mae"]) / best_baseline_mae * 100.0
    )
    target_mode_comparison = {
        "direct": {**direct_scores, "label": "RandomForest on the level directly"},
        "delta": {
            **metrics["RandomForest"],
            "label": "RandomForest on the hourly change (used)",
        },
    }

    # Baselines get overflow metrics too, so the comparison is apples-to-apples.
    for baseline_name, payload in baselines.items():
        if baseline_name == "best":
            continue
        overflow[baseline_name] = overflow_warning_metrics(
            y_test, payload["predictions"], capacity
        )

    best_model_name = min(metrics, key=lambda key: metrics[key]["mae"])

    # --- optional pass validation: honest walk-forward CV on the train set --
    cv: dict[str, Any] = {"ran": False}
    if run_cv and len(X_train) > cv_splits * 10:
        splitter = TimeSeriesSplit(n_splits=cv_splits)
        fold_scores: list[float] = []
        for fold, (tr_idx, va_idx) in enumerate(splitter.split(X_train), start=1):
            fold_X_tr, fold_X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            fold_delta = (
                y_train.iloc[tr_idx].to_numpy(dtype=float)
                - fold_X_tr["tank_level_litres"].to_numpy(dtype=float)
            )
            fold_model = clone(estimators[best_model_name])
            fold_model.fit(fold_X_tr, fold_delta)
            fold_pred = np.clip(
                fold_X_va["tank_level_litres"].to_numpy(dtype=float)
                + fold_model.predict(fold_X_va),
                0.0,
                capacity,
            )
            fold_scores.append(
                float(regression_metrics(y_train.iloc[va_idx], fold_pred)["mae"])
            )
        cv = {
            "ran": True,
            "splits": cv_splits,
            "model": best_model_name,
            "fold_mae": [round(score, 3) for score in fold_scores],
            "mean_mae": float(np.mean(fold_scores)),
            "std_mae": float(np.std(fold_scores)),
        }

    # --- tidy test frame for charts and the impact simulation --------------
    test_frame = split["test"][
        [
            "timestamp",
            "tank_level_litres",
            "rainfall_mm",
            "inflow_litres",
            "usage_litres",
            "usage_roll_mean_24",
            "rain_roll_3",
            "rain_roll_24",
            "fill_pct",
            TARGET,
        ]
    ].copy()
    test_frame = test_frame.rename(
        columns={"tank_level_litres": "current_level", TARGET: "actual_level"}
    )
    for baseline_name, payload in baselines.items():
        if baseline_name == "best":
            continue
        test_frame[f"pred_{baseline_name}"] = payload["predictions"]
    for model_name, y_pred in predictions.items():
        test_frame[f"pred_{model_name}"] = y_pred
    test_frame["pred"] = predictions[best_model_name]
    test_frame["pred_fill_pct"] = test_frame["pred"] / capacity * 100.0
    test_frame["error"] = test_frame["pred"] - test_frame["actual_level"]

    return {
        "model_name": best_model_name,
        "model": fitted[best_model_name],
        "all_models": fitted,
        "target_mode": TARGET_MODE,
        "target_mode_comparison": target_mode_comparison,
        "features": list(FEATURES),
        "capacity": capacity,
        "metrics": metrics,
        "overflow": overflow,
        "baselines": {
            key: {k: v for k, v in value.items() if k != "predictions"}
            for key, value in baselines.items()
        },
        "best_baseline": best_baseline_name,
        "improvement_pct": metrics[best_model_name]["improvement_pct"],
        "feature_importance": importance[best_model_name],
        "all_importances": importance,
        "cv": cv,
        "split": {k: v for k, v in split.items() if k not in ("train", "test", "X_train", "X_test", "y_train", "y_test")},
        "test_frame": test_frame,
        "signature": data_signature(prepared.features, capacity),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
    }


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------
def save_bundle(bundle: dict[str, Any], path: str | Path = DEFAULT_MODEL_PATH) -> Path:
    """Persist the trained bundle with joblib (the assignment's format of choice)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_bundle(path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    """Load a bundle written by :func:`save_bundle`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No trained bundle at {path}. Run `python -m src.model` first.")
    return joblib.load(path)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _print_report(bundle: dict[str, Any]) -> None:
    capacity = bundle["capacity"]
    print("=" * 74)
    print("RAINWATER HARVESTING TANK CONTROLLER - TRAINING REPORT")
    print("=" * 74)
    split = bundle["split"]
    print(
        f"Rows: {split['train_rows'] + split['test_rows']} | "
        f"train {split['train_rows']} (first {split['train_fraction']:.0%}) | "
        f"test {split['test_rows']} (last {1 - split['train_fraction']:.0%})"
    )
    print(f"Test period : {split['test_period'][0]} .. {split['test_period'][1]}")
    print(f"Capacity    : {capacity:,.0f} L | TARGET = next-hour level\n")

    print(f"{'model':<14}{'MAE (L)':>12}{'RMSE (L)':>12}{'R2':>10}{'vs best baseline':>20}")
    print("-" * 74)
    for name, scores in bundle["baselines"].items():
        if name == "best":
            continue
        print(f"{name:<14}{scores['mae']:>12.2f}{scores['rmse']:>12.2f}{scores['r2']:>10.4f}{'(baseline)':>20}")
    for name, scores in bundle["metrics"].items():
        print(
            f"{name:<14}{scores['mae']:>12.2f}{scores['rmse']:>12.2f}"
            f"{scores['r2']:>10.4f}{scores['improvement_pct']:>19.1f}%"
        )

    print("\nOverflow warning (next level > 90% of capacity):")
    for name, scores in bundle["overflow"].items():
        print(
            f"  {name:<14} precision={scores['precision']:.3f} "
            f"recall={scores['recall']:.3f} f1={scores['f1']:.3f} "
            f"(missed={scores['false_negatives']}, false alarms={scores['false_positives']})"
        )

    if bundle["cv"].get("ran"):
        cv = bundle["cv"]
        print(
            f"\nWalk-forward CV on training rows ({cv['model']}, "
            f"{cv['splits']} TimeSeriesSplit folds): mean MAE = {cv['mean_mae']:.2f} L "
            f"(+/- {cv['std_mae']:.2f})"
        )

    comparison = bundle.get("target_mode_comparison")
    if comparison:
        print("\nWhy the model predicts the hourly *change* rather than the level:")
        for key in ("direct", "delta"):
            row = comparison[key]
            print(f"  {row['label']:<42} MAE={row['mae']:>8.2f} L  R2={row['r2']:.4f}")

    print(f"\nWinner: {bundle['model_name']} -> {bundle['improvement_pct']:.1f}% better MAE "
          f"than the best baseline ({bundle['best_baseline']})")
    top = bundle["feature_importance"].head(8)
    print("\nTop features:")
    for name, value in top.items():
        print(f"  {name:<22}{value:.4f}")
    print("=" * 74)


def main() -> None:
    from src.preprocess import preprocess

    prepared = preprocess("data/tank_data.csv")
    bundle = train_and_evaluate(prepared)
    path = save_bundle(bundle)
    _print_report(bundle)
    print(f"\nSaved bundle -> {path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
