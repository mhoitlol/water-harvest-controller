"""Expose the synthetic data generator as an importable package.

The Streamlit app imports :func:`data.generate_data.generate_dataset` so the
demo can build the sample dataset on the fly when no CSV is uploaded.
"""

from __future__ import annotations

from .generate_data import (
    generate_dataset,
    inject_duplicate_timestamps,
    inject_missing_values,
    inject_outliers,
    main,
    save_dataset,
)

__all__ = [
    "generate_dataset",
    "save_dataset",
    "inject_missing_values",
    "inject_outliers",
    "inject_duplicate_timestamps",
    "main",
]
