"""Regression test: LangChain feedback calls record token usage and cost.

Feedback functions on the LangChain provider reach the model through
`chain.invoke`, and a chat model's `AIMessage` reports its usage in
`usage_metadata`.  `LangchainEndpoint` never instrumented that call, was
missing from `Endpoint.ENDPOINT_SETUPS`, and its `handle_wrapped_call` was a
stub calling `handle_generation(response=None)`, so a feedback call produced a
tally with `n_requests == 0`, `n_tokens == 0` and `cost == 0.0` while the
response it had just received carried the usage.
"""

from __future__ import annotations

import unittest
from unittest import mock

try:
    from langchain_core.language_models.fake_chat_models import (
        GenericFakeChatModel,
    )
    from langchain_core.messages import AIMessage
    from trulens.core.feedback import endpoint as core_endpoint
    from trulens.core.schema import base as base_schema
    from trulens.otel.semconv.trace import SpanAttributes
    from trulens.providers.langchain import Langchain
    from trulens.providers.langchain import endpoint as langchain_endpoint
except Exception:  # pragma: no cover
    GenericFakeChatModel = None
    AIMessage = None
    core_endpoint = None
    base_schema = None
    SpanAttributes = None
    Langchain = None
    langchain_endpoint = None

_MODEL_NAME = "gpt-4o-mini"
_USAGE = {"input_tokens": 1200, "output_tokens": 300, "total_tokens": 1500}
# Prices in USD per token, in the shape of the LiteLLM pricing table.
_PRICING = {"input_cost_per_token": 1e-6, "output_cost_per_token": 2e-6}
_EXPECTED_COST = 1200 * 1e-6 + 300 * 2e-6


def _fake_chat_model(replies: int = 10) -> "GenericFakeChatModel":
    """A chat model that answers with usage metadata and makes no request."""

    def reply() -> "AIMessage":
        return AIMessage(
            content="3",
            usage_metadata=dict(_USAGE),
            response_metadata={"model_name": _MODEL_NAME},
        )

    return GenericFakeChatModel(
        messages=iter([reply() for _ in range(replies)])
    )


def _fake_reply() -> "AIMessage":
    return _fake_chat_model(replies=1).invoke("unused")


def _patched_pricing():
    """Price `_MODEL_NAME` from a table this test controls.

    `LiteLLM` is not a dependency of the LangChain provider package, so the
    table may not exist; `create=True` keeps this test measuring the tally
    rather than the presence of the module attribute.
    """
    return mock.patch.object(
        langchain_endpoint,
        "LITELLM_MODEL_COSTS_TABLE",
        {_MODEL_NAME: dict(_PRICING)},
        create=True,
    )


class TestLangchainCostTracking(unittest.TestCase):
    def setUp(self):
        if Langchain is None:
            self.skipTest("trulens-providers-langchain not available.")

    def test_feedback_call_records_tokens_and_cost(self):
        """A tracked feedback call must report the usage its response carried."""
        provider = Langchain(chain=_fake_chat_model())

        with _patched_pricing():
            _, tally = core_endpoint.Endpoint.track_all_costs_tally(
                provider.relevance,
                prompt="What is the capital of France?",
                response="Paris.",
            )

        cost = tally()
        self.assertEqual(cost.n_requests, 1)
        self.assertEqual(cost.n_tokens, 1500)
        self.assertEqual(cost.n_prompt_tokens, 1200)
        self.assertEqual(cost.n_completion_tokens, 300)
        self.assertAlmostEqual(cost.cost, _EXPECTED_COST)
        self.assertEqual(cost.cost_currency, "USD")

    def test_each_call_is_recorded_once(self):
        """Two calls must add up to two calls' worth of usage, not four."""
        provider = Langchain(chain=_fake_chat_model(replies=2))

        def two_calls():
            provider.relevance(prompt="a?", response="b.")
            provider.relevance(prompt="c?", response="d.")

        with _patched_pricing():
            _, tally = core_endpoint.Endpoint.track_all_costs_tally(two_calls)

        cost = tally()
        self.assertEqual(cost.n_requests, 2)
        self.assertEqual(cost.n_tokens, 3000)
        self.assertAlmostEqual(cost.cost, 2 * _EXPECTED_COST)

    def test_langchain_is_in_endpoint_setups(self):
        """Routing a feedback call through the shared machinery needs the entry."""
        setups = [
            setup
            for setup in core_endpoint.Endpoint.ENDPOINT_SETUPS
            if setup.class_name == "LangchainEndpoint"
        ]
        self.assertEqual(len(setups), 1)
        self.assertEqual(
            setups[0].module_name, "trulens.providers.langchain.endpoint"
        )
        self.assertEqual(setups[0].arg_flag, "with_langchain")

    def test_langchain_tracking_can_be_switched_off(self):
        """`with_langchain=False` must leave the LangChain endpoint out."""
        provider = Langchain(chain=_fake_chat_model())

        with _patched_pricing():
            _, tally = core_endpoint.Endpoint.track_all_costs_tally(
                provider.relevance,
                with_langchain=False,
                prompt="What is the capital of France?",
                response="Paris.",
            )

        cost = tally()
        self.assertEqual(cost.n_requests, 0)
        self.assertEqual(cost.n_tokens, 0)

    def test_response_without_usage_counts_the_request_only(self):
        """A response with no usage must not look like a free priced call."""
        callback = langchain_endpoint.LangchainCallback.model_construct(
            cost=base_schema.Cost()
        )

        callback.handle_generation("plain string from a BaseLLM chain")

        self.assertEqual(callback.cost.n_requests, 1)
        self.assertEqual(callback.cost.n_tokens, 0)
        self.assertEqual(callback.cost.cost, 0.0)


class TestLangchainCostComputer(unittest.TestCase):
    def setUp(self):
        if langchain_endpoint is None:
            self.skipTest("trulens-providers-langchain not available.")

    def test_handle_response_reads_usage_metadata(self):
        cost_info = langchain_endpoint.LangchainCostComputer.handle_response(
            _fake_reply()
        )

        self.assertEqual(cost_info[SpanAttributes.COST.NUM_TOKENS], 1500)
        self.assertEqual(cost_info[SpanAttributes.COST.NUM_PROMPT_TOKENS], 1200)
        self.assertEqual(
            cost_info[SpanAttributes.COST.NUM_COMPLETION_TOKENS], 300
        )
        self.assertEqual(cost_info[SpanAttributes.COST.MODEL], _MODEL_NAME)

    def test_known_model_includes_positive_cost(self):
        with _patched_pricing():
            cost_info = (
                langchain_endpoint.LangchainCostComputer.handle_response(
                    _fake_reply()
                )
            )

        self.assertAlmostEqual(
            cost_info[SpanAttributes.COST.COST], _EXPECTED_COST
        )

    def test_real_pricing_table_prices_a_known_model(self):
        """The shipped pricing table must price a model it knows."""
        table = getattr(langchain_endpoint, "LITELLM_MODEL_COSTS_TABLE", None)
        if not table or _MODEL_NAME not in table:
            self.skipTest("LiteLLM pricing data is not available.")

        cost_info = langchain_endpoint.LangchainCostComputer.handle_response(
            _fake_reply()
        )

        self.assertGreater(cost_info[SpanAttributes.COST.COST], 0)

    def test_unknown_model_omits_cost_but_keeps_tokens(self):
        """An unpriced model must not be recorded as a free call."""
        with mock.patch.object(
            langchain_endpoint,
            "LITELLM_MODEL_COSTS_TABLE",
            {},
            create=True,
        ):
            cost_info = (
                langchain_endpoint.LangchainCostComputer.handle_response(
                    AIMessage(
                        content="3",
                        usage_metadata=dict(_USAGE),
                        response_metadata={"model_name": "not-a-priced-model"},
                    )
                )
            )

        self.assertNotIn(SpanAttributes.COST.COST, cost_info)
        self.assertEqual(cost_info[SpanAttributes.COST.NUM_TOKENS], 1500)

    def test_response_without_usage_metadata_returns_none(self):
        self.assertIsNone(
            langchain_endpoint.LangchainCostComputer.handle_response(
                AIMessage(content="3")
            )
        )


if __name__ == "__main__":
    unittest.main()
