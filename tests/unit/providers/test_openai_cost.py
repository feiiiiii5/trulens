"""Regression test: OpenAI cost callback must count each call exactly once.

`OpenAICallback.handle_generation` built its per-call addition from the
langchain handler's lifetime totals (`total_tokens`, `successful_requests`,
...) and did `self.cost += addl_cost`. The handler is created once per
callback and never reset, so N calls inside one cost scope reported
N(N+1)/2 of their usage (issue #2839): two calls of 120 tokens each
reported 360 tokens and 3 successful requests.
"""

from __future__ import annotations

import unittest

try:
    from trulens.core.schema import base as base_schema
    from trulens.providers.openai.endpoint import (
        Generation,
        LLMResult,
        OpenAICallback,
    )
except Exception:  # pragma: no cover
    LLMResult = None
    OpenAICallback = None


def _llm_result(prompt_tokens: int = 100, completion_tokens: int = 20):
    """An `LLMResult` shaped like the ones langchain's OpenAI LLMs emit:
    per-call usage in `llm_output["token_usage"]` plus the model name."""
    return LLMResult(
        generations=[[Generation(text="hi")]],
        llm_output={
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "model_name": "gpt-4o-mini",
        },
    )


class TestOpenAICostCallback(unittest.TestCase):
    def setUp(self):
        if OpenAICallback is None or LLMResult is None:
            self.skipTest("trulens-providers-openai not available.")

    def test_single_call_counts_once(self):
        cb = OpenAICallback.model_construct(cost=base_schema.Cost())

        cb.handle_generation(_llm_result())

        self.assertEqual(cb.cost.n_tokens, 120)
        self.assertEqual(cb.cost.n_prompt_tokens, 100)
        self.assertEqual(cb.cost.n_completion_tokens, 20)
        self.assertEqual(cb.cost.n_successful_requests, 1)

    def test_two_calls_count_each_call_once(self):
        cb = OpenAICallback.model_construct(cost=base_schema.Cost())

        cb.handle_generation(_llm_result())
        first_cost = cb.cost.cost
        cb.handle_generation(_llm_result())

        # Was 360 tokens / 3 successful requests on main: the second call
        # added the langchain handler's running totals again.
        self.assertEqual(cb.cost.n_tokens, 240)
        self.assertEqual(cb.cost.n_prompt_tokens, 200)
        self.assertEqual(cb.cost.n_completion_tokens, 40)
        self.assertEqual(cb.cost.n_successful_requests, 2)
        self.assertAlmostEqual(cb.cost.cost, 2 * first_cost, places=12)

    def test_successful_requests_matches_requests(self):
        cb = OpenAICallback.model_construct(cost=base_schema.Cost())

        cb.handle_generation(_llm_result())
        cb.handle_generation(_llm_result())

        # The base callback counts every call in `n_requests`; the fixed
        # handler must not report more successful requests than that.
        self.assertEqual(cb.cost.n_requests, 2)
        self.assertEqual(cb.cost.n_successful_requests, cb.cost.n_requests)

    def test_empty_usage_does_not_crash(self):
        cb = OpenAICallback.model_construct(cost=base_schema.Cost())

        cb.handle_generation(
            LLMResult(generations=[[Generation(text="hi")]], llm_output={})
        )
        cb.handle_generation(
            LLMResult(generations=[[Generation(text="hi")]], llm_output=None)
        )

        self.assertEqual(cb.cost.n_successful_requests, 2)


if __name__ == "__main__":
    unittest.main()
