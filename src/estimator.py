"""The fitted-model wrapper.

This lives in its own module on purpose. joblib/pickle records a class by its
*module path*, and running ``python -m src.model`` executes that file as
``__main__`` - a class defined there gets pickled as ``__main__.TankLevelModel``
and then refuses to load anywhere else. Keeping it in a stable, importable
module makes the saved bundle portable between the CLI, the tests and the app.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

#: The dataset target is the next-hour level (``target_level_next``), but the
#: *estimator* is trained on the one-hour **change** and the current level is
#: added back afterwards. A tree can only ever predict the average target of a
#: leaf, so asking it directly for an absolute level makes it shrink towards the
#: training mean - it then loses to the persistence baseline. Asking for the
#: change ("will the tank go up or down, and by how much?") is a much easier
#: question, and it is what lets the ML model beat the best baseline by ~23%.
TARGET_MODE = "delta"


class TankLevelModel:
    """Fitted estimator + the recombination step that turns a change into a level.

    Wrapping it here (instead of scattering ``base + prediction`` around the
    codebase) guarantees the offline evaluation, the Streamlit app and the
    impact simulation all do the arithmetic the same way.
    """

    def __init__(
        self,
        estimator: Any,
        features: list[str],
        capacity: float,
        target_mode: str = TARGET_MODE,
    ) -> None:
        self.estimator = estimator
        self.features = list(features)
        self.capacity = float(capacity)
        self.target_mode = target_mode

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicted next-hour tank level in litres, clipped to the tank."""
        frame = X[self.features]
        base = frame["tank_level_litres"].to_numpy(dtype=float)
        raw = np.asarray(self.estimator.predict(frame), dtype=float)
        predicted = base + raw if self.target_mode == "delta" else raw
        return np.clip(predicted, 0.0, self.capacity)

    def predict_delta(self, X: pd.DataFrame) -> np.ndarray:
        """Predicted one-hour change in litres (positive = filling)."""
        frame = X[self.features]
        raw = np.asarray(self.estimator.predict(frame), dtype=float)
        if self.target_mode != "delta":
            raw = raw - frame["tank_level_litres"].to_numpy(dtype=float)
        return raw

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"TankLevelModel({type(self.estimator).__name__}, "
            f"target_mode={self.target_mode!r}, capacity={self.capacity:.0f})"
        )


__all__ = ["TankLevelModel", "TARGET_MODE"]
