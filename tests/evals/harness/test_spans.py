"""Attaching eval metrics to a span the product recorded, by name."""

from deepeval.metrics import BaseMetric
from deepeval.tracing import current_trace_context, observe
from deepeval.tracing.types import BaseSpan

from evals.harness.spans import attach_metrics_to_spans
from evals.metrics.quote_letter.names_no_supplier import NamesNoSupplier
from steel.tracing import set_span_input, set_span_output


@observe(type="llm", name="write_cover_letter")
def write_letter() -> None:
    set_span_input('{"language": "en", "quote": {}}')
    set_span_output("Dear customer, ...")


@observe(type="tool", name="draft_quote")
def draft_quote() -> None:
    write_letter()


def metrics_on(span: BaseSpan) -> list[str]:
    return [metric.__name__ for metric in span.metrics or []]


def test_metrics_land_on_the_named_span_however_deep_it_sits() -> None:
    attached: dict[str, list[str]] = {}

    @observe(name="enquiry")
    def run() -> int:
        draft_quote()
        matched = attach_metrics_to_spans("write_cover_letter", lambda: [NamesNoSupplier()])
        trace = current_trace_context.get()
        assert trace is not None
        [tool] = trace.root_spans[0].children
        [letter] = tool.children
        attached.update({tool.name: metrics_on(tool), letter.name: metrics_on(letter)})
        return matched

    assert (run(), attached) == (1, {"draft_quote": [], "write_cover_letter": ["Names no supplier"]})


def test_each_matching_span_gets_its_own_metric_instances() -> None:
    """A metric keeps its last score, so two letters must never share one."""
    instances: list[BaseMetric] = []

    @observe(name="enquiry")
    def run() -> int:
        write_letter()
        write_letter()
        return attach_metrics_to_spans(
            "write_cover_letter", lambda: instances.append(NamesNoSupplier()) or instances[-1:]
        )

    assert (run(), len(instances), instances[0] is instances[1]) == (2, 2, False)


def test_outside_a_live_trace_nothing_is_attached() -> None:
    assert attach_metrics_to_spans("write_cover_letter", lambda: [NamesNoSupplier()]) == 0
