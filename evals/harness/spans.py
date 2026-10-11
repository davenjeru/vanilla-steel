"""Attach eval metrics to a span the product recorded, chosen by name.

The product names its spans (steel.tracing); the eval decides what to check on them, so no metric
lives in product code. When the test's golden is asserted (`assert_test` under `deepeval test run`),
DeepEval scores a span's metrics on that span's own input and output: for "write_cover_letter" that
is the quote the writer saw and the letter it wrote, the same two things the stand-alone letter eval
scores.
"""

from collections.abc import Callable

from deepeval.metrics import BaseMetric
from deepeval.tracing import current_trace_context
from deepeval.tracing.types import BaseSpan


def attach_metrics_to_spans(span_name: str, metrics_for_each: Callable[[], list[BaseMetric]]) -> int:
    """Attach fresh metrics to every span of that name in the live trace; returns how many spans matched.

    Fresh per span, because a metric instance keeps its last score. Outside a live trace nothing is attached.
    """
    trace = current_trace_context.get()
    if trace is None:
        return 0
    matched = [span for span in _all_spans(trace.root_spans or []) if span.name == span_name]
    for span in matched:
        span.metrics = [*(span.metrics or []), *metrics_for_each()]
    return len(matched)


def _all_spans(spans: list[BaseSpan]) -> list[BaseSpan]:
    found: list[BaseSpan] = []
    for span in spans:
        found.append(span)
        found.extend(_all_spans(span.children or []))
    return found
