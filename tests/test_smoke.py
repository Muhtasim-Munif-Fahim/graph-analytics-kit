"""Smoke test: package is importable and reports a version."""
from __future__ import annotations

import graph_analytics_kit


def test_version() -> None:
    assert isinstance(graph_analytics_kit.__version__, str)
    assert graph_analytics_kit.__version__ == "0.1.0"
