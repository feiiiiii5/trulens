import inspect
import logging
from typing import Any, Callable, Dict, Optional, Tuple, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM
from trulens.core.feedback import endpoint as core_endpoint
from trulens.core.schema import base as base_schema
from trulens.otel.semconv.trace import SpanAttributes

logger = logging.getLogger(__name__)

try:
    # `litellm` is not a dependency of this provider package: price calls from
    # its table when it is installed and record tokens without a cost when it
    # is not.
    from litellm import model_cost as LITELLM_MODEL_COSTS_TABLE
except ImportError:  # pragma: no cover - depends on the environment
    LITELLM_MODEL_COSTS_TABLE = {}


def _get_model_pricing(model_name: str) -> Optional[Tuple[float, float]]:
    """Get per-token input and output pricing from LiteLLM.

    Returns None when the model is not in the pricing table so that callers
    can leave any already-recorded cost alone rather than stamping a
    misleading zero. This mirrors the behaviour of the Anthropic endpoint's
    pricing helper.

    Reference: https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json
    """
    if not model_name:
        return None

    pricing = LITELLM_MODEL_COSTS_TABLE.get(model_name)
    if pricing is None:
        logger.warning("Model %s not found in LiteLLM pricing data", model_name)
        return None

    return (
        pricing.get("input_cost_per_token", 0.0),
        pricing.get("output_cost_per_token", 0.0),
    )


class LangchainCostComputer:
    """Computes usage and cost from a LangChain response."""

    @staticmethod
    def handle_response(response: Any) -> Optional[Dict[str, Any]]:
        """Read the usage a LangChain response reports.

        Args:
            response: what `chain.invoke` returned. A chat model's
                `AIMessage` carries `usage_metadata` and names the model in
                its `response_metadata`.

        Returns:
            The attributes describing the call, or None when the response
            carries no usage information (a plain string from a BaseLLM
            chain, for instance), in which case there is nothing to price.
        """
        usage = getattr(response, "usage_metadata", None)
        if not usage:
            return None

        n_prompt_tokens = usage.get("input_tokens") or 0
        n_completion_tokens = usage.get("output_tokens") or 0
        n_tokens = usage.get("total_tokens") or (
            n_prompt_tokens + n_completion_tokens
        )

        response_metadata = getattr(response, "response_metadata", None) or {}
        model_name = (
            response_metadata.get("model_name")
            or response_metadata.get("model")
            or ""
        )

        cost_info: Dict[str, Any] = {
            SpanAttributes.COST.CURRENCY: "USD",
            SpanAttributes.COST.NUM_TOKENS: n_tokens,
            SpanAttributes.COST.NUM_PROMPT_TOKENS: n_prompt_tokens,
            SpanAttributes.COST.NUM_COMPLETION_TOKENS: n_completion_tokens,
            SpanAttributes.COST.MODEL: model_name,
        }

        pricing = _get_model_pricing(model_name)
        if pricing is not None:
            input_price, output_price = pricing
            cost_info[SpanAttributes.COST.COST] = (
                n_prompt_tokens * input_price
                + n_completion_tokens * output_price
            )

        return cost_info


def _build_cost(cost_info: Dict[str, Any]) -> base_schema.Cost:
    """Build a Cost from what LangchainCostComputer.handle_response read.

    SpanAttributes.COST.COST is absent when the model was not found in the
    pricing table; the returned Cost then keeps the 0.0 default so an unknown
    price is not recorded as a free call.
    """
    addl_cost = base_schema.Cost(
        cost_currency=cost_info.get(SpanAttributes.COST.CURRENCY, "USD"),
        n_tokens=cost_info.get(SpanAttributes.COST.NUM_TOKENS, 0),
        n_prompt_tokens=cost_info.get(SpanAttributes.COST.NUM_PROMPT_TOKENS, 0),
        n_completion_tokens=cost_info.get(
            SpanAttributes.COST.NUM_COMPLETION_TOKENS, 0
        ),
    )
    cost_val = cost_info.get(SpanAttributes.COST.COST)
    if cost_val is not None:
        addl_cost = addl_cost.model_copy(update={"cost": cost_val})
    return addl_cost


class LangchainCallback(core_endpoint.EndpointCallback):
    def handle_classification(self, response: Dict) -> None:
        super().handle_classification(response)

    def handle_generation(self, response: Any) -> None:
        super().handle_generation(response)

        cost_info = LangchainCostComputer.handle_response(response)
        if cost_info is None:
            # A response without usage still counts as a request, but it
            # cannot be priced: do not record it as a free call either.
            logger.debug(
                "LangChain response carried no usage information:\n%s",
                response,
            )
            return

        self.cost += _build_cost(cost_info)


class LangchainEndpoint(core_endpoint.Endpoint):
    """
    LangChain endpoint.
    """

    # Cannot validate BaseLLM / BaseChatModel as they are pydantic v1 and there
    # is some bug involving their use within pydantic v2.
    # https://github.com/langchain-ai/langchain/issues/10112
    chain: Any  # Union[BaseLLM, BaseChatModel]

    def handle_wrapped_call(
        self,
        func: Callable,
        bindings: inspect.BoundArguments,
        response: Any,
        callback: Optional[core_endpoint.EndpointCallback],
    ) -> Any:
        self.global_callback.handle_generation(response=response)

        if callback is not None:
            callback.handle_generation(response=response)

        return response

    def __init__(self, chain: Union[BaseLLM, BaseChatModel], *args, **kwargs):
        if chain is None:
            raise ValueError("`chain` must be specified.")

        if not (isinstance(chain, BaseLLM) or isinstance(chain, BaseChatModel)):
            raise ValueError(
                f"`chain` must be of type {BaseLLM.__name__} or {BaseChatModel.__name__}. "
                f"If you are using DEFERRED mode, this may be due to our inability to serialize `chain`."
            )

        kwargs["chain"] = chain
        kwargs["callback_class"] = LangchainCallback

        super().__init__(*args, **kwargs)

        # Feedback functions reach the model through chain.invoke, so that is
        # where the usage a response reports gets read off.
        self._instrument_class(type(chain), "invoke")
