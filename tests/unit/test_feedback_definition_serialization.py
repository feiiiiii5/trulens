"""Round-trip tests for feedback definitions that use OTEL selectors."""

from trulens.core import Metric
from trulens.core.feedback.selector import Selector
from trulens.core.schema import feedback as feedback_schema
from trulens.core.utils import serial as serial_utils
from trulens.otel.semconv.trace import SpanAttributes


def metric_implementation(query: str) -> float:
    """Importable implementation so the metric can be serialized."""
    return 1.0


def _otel_selector() -> Selector:
    return Selector(
        span_type=SpanAttributes.SpanType.RETRIEVAL,
        span_attribute=SpanAttributes.RETRIEVAL.QUERY_TEXT,
        ignore_none_values=True,
        collect_list=False,
    )


def _assert_selector_round_tripped(selector) -> None:
    assert isinstance(selector, Selector), type(selector)
    assert selector.span_type == SpanAttributes.SpanType.RETRIEVAL
    assert selector.span_attribute == SpanAttributes.RETRIEVAL.QUERY_TEXT
    assert selector.ignore_none_values is True
    assert selector.collect_list is False


def test_metric_with_otel_selector_round_trips_through_json() -> None:
    metric = Metric(
        implementation=metric_implementation,
        name="selector_metric",
        selectors={"query": _otel_selector()},
    )

    restored = Metric.model_validate_json(metric.model_dump_json())

    _assert_selector_round_tripped(restored.selectors["query"])


def test_metric_with_otel_selector_round_trips_through_model_validate() -> None:
    metric = Metric(
        implementation=metric_implementation,
        name="selector_metric",
        selectors={"query": _otel_selector()},
    )

    restored = Metric.model_validate(metric.model_dump())

    _assert_selector_round_tripped(restored.selectors["query"])


def test_feedback_definition_with_otel_selector_round_trips() -> None:
    definition = feedback_schema.FeedbackDefinition(
        selectors={"query": _otel_selector()},
        implementation=None,
        aggregator=None,
    )

    restored = feedback_schema.FeedbackDefinition.model_validate(
        definition.model_dump()
    )

    _assert_selector_round_tripped(restored.selectors["query"])


def test_legacy_lens_selector_round_trip_is_unchanged() -> None:
    lens = serial_utils.Lens().record["query"]
    definition = feedback_schema.FeedbackDefinition(
        selectors={"query": lens},
        implementation=None,
        aggregator=None,
    )

    restored = feedback_schema.FeedbackDefinition.model_validate(
        definition.model_dump()
    )

    restored_selector = restored.selectors["query"]
    assert isinstance(restored_selector, serial_utils.Lens)
    assert str(restored_selector) == str(lens)
