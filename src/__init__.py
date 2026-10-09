"""Rainwater Harvesting Tank Controller - reusable modelling package.

Modules
-------
preprocess : loading, cleaning, outlier handling, feature engineering
baseline   : non-ML persistence and physics baselines
estimator  : the fitted-model wrapper (predicts the hourly change, then recombines)
model      : model training, evaluation and persistence (joblib)
decision   : recommendation rules, sensor-fault fallback and impact metrics
pipeline   : one-call helper that wires the whole flow together
"""

from __future__ import annotations

__all__ = ["preprocess", "baseline", "estimator", "model", "decision", "pipeline"]
