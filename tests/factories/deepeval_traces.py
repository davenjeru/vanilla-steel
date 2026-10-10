"""Capture the trace DeepEval would send to Confident AI, without sending it."""

from collections.abc import Callable
from typing import Any

import pytest
from deepeval.tracing.trace_test_manager import trace_testing_manager


def confident_ai_payload(traced_call: Callable[[], object], monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Run an @observe'd call in DeepEval's trace-testing mode: the trace body is stored, never posted."""
    monkeypatch.setattr(trace_testing_manager, "test_name", "capture")
    monkeypatch.setattr(trace_testing_manager, "test_dict", None)
    traced_call()
    return trace_testing_manager.test_dict or {}
