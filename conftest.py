"""Pytest bootstrap: make the project root importable as a package root.

This lets the test suite (and `python -m src.model`) import `src.*` without
having to install the project.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
