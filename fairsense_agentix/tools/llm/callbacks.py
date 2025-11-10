"""LangChain callbacks for telemetry integration.

This module provides callback handlers that bridge LangChain's callback system
to our telemetry service. Callbacks observe LLM calls and track metrics like:
- Token usage (prompt tokens, completion tokens)
- Cost estimation (provider-specific pricing)
- Latency (request duration)
- Error rates

Callbacks are non-invasive—they observe without modifying behavior.

Examples
--------
    >>> from langchain_openai import ChatOpenAI
    >>> from fairsense_agentix.tools.llm.callbacks import TelemetryCallback
    >>>
    >>> callback = TelemetryCallback(
    ...     telemetry=telemetry,
    ...     provider="openai",
    ...     model="gpt-4",
    ... )
    >>>
    >>> # Pass to LangChain model
    >>> model = ChatOpenAI(
    ...     model="gpt-4",
    ...     callbacks=[callback],
    ... )
    >>>
    >>> # All calls automatically tracked
    >>> response = model.invoke("Hello!")
    >>> # Telemetry now has token count, cost, latency
"""

import time
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from fairsense_agentix.shared.telemetry import Telemetry


class TelemetryCallback(BaseCallbackHandler):
    """Callback handler bridging LangChain to our telemetry service.

    This callback observes LangChain LLM calls and tracks detailed metrics:
    - Token usage (input/output tokens)
    - Cost estimation using provider-specific pricing
    - Request latency
    - Success/failure rates

    Pricing (as of 2025):
    - GPT-4: $0.03/1K input, $0.06/1K output
    - GPT-4-turbo: $0.01/1K input, $0.03/1K output
    - GPT-3.5-turbo: $0.001/1K input, $0.002/1K output
    - Claude-3-opus: $0.015/1K input, $0.075/1K output
    - Claude-3-sonnet: $0.003/1K input, $0.015/1K output

    Parameters
    ----------
    telemetry : Telemetry
        Telemetry service for tracking events
    provider : str
        Provider name ("openai", "anthropic", "google", etc.)
    model : str
        Model identifier (e.g., "gpt-4", "claude-3-5-sonnet-20241022")

    Attributes
    ----------
    telemetry : Telemetry
        Telemetry service
    provider : str
        Provider name
    model : str
        Model identifier
    start_time : float | None
        Timestamp when LLM call started (for latency calculation)

    Examples
    --------
    >>> callback = TelemetryCallback(
    ...     telemetry=telemetry,
    ...     provider="openai",
    ...     model="gpt-4",
    ... )
    >>> # Use with LangChain model
    >>> from langchain_openai import ChatOpenAI
    >>> model = ChatOpenAI(model="gpt-4", callbacks=[callback])
    >>> response = model.invoke("Analyze this text...")
    >>> # Telemetry automatically tracks tokens, cost, latency
    """

    def __init__(self, telemetry: Telemetry, provider: str, model: str) -> None:
        """Initialize telemetry callback.

        Parameters
        ----------
        telemetry : Telemetry
            Telemetry service
        provider : str
            Provider name (e.g., "openai", "anthropic")
        model : str
            Model identifier (e.g., "gpt-4")
        """
        super().__init__()
        self.telemetry = telemetry
        self.provider = provider
        self.model = model
        self.start_time: float | None = None

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> None:
        """Record LLM start time for latency calculation.

        Records start time for latency calculation.

        Parameters
        ----------
        serialized : dict[str, Any]
            Serialized LLM configuration
        prompts : list[str]
            Input prompts
        **kwargs : Any
            Additional LangChain parameters
        """
        self.start_time = time.time()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Track LLM completion metrics and costs.

        Extracts token usage, calculates cost and latency, and tracks in telemetry.

        Parameters
        ----------
        response : LLMResult
            LLM response with generations and metadata
        **kwargs : Any
            Additional LangChain parameters
        """
        # Calculate latency
        latency_ms = 0
        if self.start_time:
            latency_ms = int((time.time() - self.start_time) * 1000)
            self.start_time = None  # Reset for next call

        # Extract token usage from LLM response
        prompt_tokens = 0
        completion_tokens = 0

        if response.llm_output and isinstance(response.llm_output, dict):
            token_usage = response.llm_output.get("token_usage", {})
            # OpenAI format
            prompt_tokens = token_usage.get("prompt_tokens", 0)
            completion_tokens = token_usage.get("completion_tokens", 0)

            # Anthropic format (uses different keys)
            if prompt_tokens == 0:
                prompt_tokens = token_usage.get("input_tokens", 0)
            if completion_tokens == 0:
                completion_tokens = token_usage.get("output_tokens", 0)

        # Calculate cost based on provider and model
        cost_usd = self._estimate_cost(prompt_tokens, completion_tokens)

        # Track detailed metrics
        self.telemetry.track_event(
            "llm_call_detailed",
            {
                "provider": self.provider,
                "model": self.model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "latency_ms": latency_ms,
                "cost_usd": cost_usd,
            },
        )

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """Track LLM generation errors.

        Tracks error type and resets timer.

        Parameters
        ----------
        error : BaseException
            The exception that occurred
        **kwargs : Any
            Additional LangChain parameters
        """
        # Reset timer
        self.start_time = None

        # Track error (high-level tracking already in adapter)
        # This provides callback-level details
        self.telemetry.track_event(
            "llm_callback_error",
            {
                "provider": self.provider,
                "model": self.model,
                "error": str(error),
                "error_type": type(error).__name__,
            },
        )

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost in USD for this API call.

        Implements provider-specific pricing. Prices are as of 2025 and should
        be updated when providers change pricing.

        Parameters
        ----------
        prompt_tokens : int
            Number of input tokens
        completion_tokens : int
            Number of output tokens

        Returns
        -------
        float
            Estimated cost in USD

        Examples
        --------
        >>> callback = TelemetryCallback(telemetry, "openai", "gpt-4")
        >>> cost = callback._estimate_cost(1000, 500)
        >>> # For GPT-4: (1000 * 0.03 + 500 * 0.06) / 1000 = $0.06
        """
        # Pricing per 1000 tokens (input, output)
        pricing: dict[str, tuple[float, float]] = {}

        # OpenAI pricing
        if self.provider == "openai":
            pricing = {
                "gpt-4": (0.03, 0.06),
                "gpt-4-turbo": (0.01, 0.03),
                "gpt-4-turbo-preview": (0.01, 0.03),
                "gpt-3.5-turbo": (0.001, 0.002),
                "gpt-3.5-turbo-16k": (0.003, 0.004),
            }

        # Anthropic pricing
        elif self.provider == "anthropic":
            pricing = {
                "claude-3-5-sonnet-20241022": (0.003, 0.015),
                "claude-3-5-sonnet": (0.003, 0.015),
                "claude-3-opus": (0.015, 0.075),
                "claude-3-opus-20240229": (0.015, 0.075),
                "claude-3-sonnet": (0.003, 0.015),
                "claude-3-sonnet-20240229": (0.003, 0.015),
                "claude-3-haiku": (0.00025, 0.00125),
                "claude-3-haiku-20240307": (0.00025, 0.00125),
            }

        # Get pricing for model (default to most expensive if unknown)
        input_price, output_price = pricing.get(
            self.model,
            (0.03, 0.06),  # Default to GPT-4 pricing as conservative estimate
        )

        # Calculate cost
        input_cost = (prompt_tokens / 1000) * input_price
        output_cost = (completion_tokens / 1000) * output_price

        return input_cost + output_cost
