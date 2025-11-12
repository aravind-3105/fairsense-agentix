"""LangChain adapter implementing LLMTool protocol.

This module provides a thin adapter that wraps LangChain chat models
(ChatOpenAI, ChatAnthropic) to satisfy our LLMTool protocol interface.

Architecture:
- Graphs call `llm.predict()` from LLMTool protocol
- LangChainLLMAdapter translates to LangChain's `.invoke()` method
- Pydantic output parser validates responses
- Callbacks bridge to our telemetry system

Benefits:
- Zero graph changes (protocol compliance)
- Dependency injection preserved (registry configures adapter)
- All LangChain features available (retry, caching, streaming)
- Easy to add new providers (just wrap new LangChain chat model)

Examples
--------
    >>> from langchain_openai import ChatOpenAI
    >>> from langchain_core.output_parsers import PydanticOutputParser
    >>> from fairsense_agentix.tools.llm.output_schemas import BiasAnalysisOutput
    >>>
    >>> # Create LangChain model
    >>> chat_model = ChatOpenAI(model="gpt-4", temperature=0.3)
    >>>
    >>> # Wrap with adapter
    >>> adapter = LangChainLLMAdapter(
    ...     langchain_model=chat_model,
    ...     telemetry=telemetry,
    ...     output_parser=PydanticOutputParser(pydantic_object=BiasAnalysisOutput),
    ... )
    >>>
    >>> # Use via protocol
    >>> result = adapter.predict("Analyze this text for bias: ...")
    >>> assert isinstance(result, BiasAnalysisOutput)
"""

from typing import Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import BaseLLMOutputParser
from pydantic import BaseModel

from fairsense_agentix.configs.settings import Settings
from fairsense_agentix.services.telemetry import TelemetryService
from fairsense_agentix.tools.exceptions import LLMError


class LangChainLLMAdapter:
    """Adapter wrapping LangChain models to satisfy LLMTool protocol.

    This adapter provides a bridge between LangChain's chat model interface
    and our application's LLMTool protocol. It handles:
    - Parameter translation (protocol params → LangChain config)
    - Output parsing (raw text → validated Pydantic objects)
    - Telemetry integration (LangChain callbacks → our telemetry service)
    - Error handling (LangChain exceptions → our LLMError)

    Parameters
    ----------
    langchain_model : BaseLanguageModel
        The underlying LangChain chat model (ChatOpenAI, ChatAnthropic, etc.)
    telemetry : Telemetry
        Telemetry service for tracking usage
    settings : Settings
        Application configuration
    output_parser : BaseLLMOutputParser | None, optional
        Parser for structured outputs (e.g., PydanticOutputParser), by default None
    provider_name : str, optional
        Provider identifier for telemetry ("openai", "anthropic", etc.),
        by default "unknown"

    Attributes
    ----------
    model : BaseLanguageModel
        The LangChain chat model being wrapped
    telemetry : Telemetry
        Telemetry service
    settings : Settings
        Application settings
    output_parser : BaseLLMOutputParser | None
        Output parser for structured responses
    provider_name : str
        Provider name for telemetry events

    Examples
    --------
    >>> from langchain_openai import ChatOpenAI
    >>> model = ChatOpenAI(model="gpt-4", temperature=0.3)
    >>> adapter = LangChainLLMAdapter(
    ...     langchain_model=model,
    ...     telemetry=telemetry,
    ...     settings=settings,
    ...     provider_name="openai",
    ... )
    >>> response = adapter.predict("Hello, world!")
    """

    def __init__(
        self,
        langchain_model: BaseLanguageModel,
        telemetry: TelemetryService,
        settings: Settings,
        output_parser: BaseLLMOutputParser | None = None,
        provider_name: str = "unknown",
    ) -> None:
        """Initialize LangChain adapter.

        Parameters
        ----------
        langchain_model : BaseLanguageModel
            LangChain chat model to wrap
        telemetry : Telemetry
            Telemetry service
        settings : Settings
            Application configuration
        output_parser : BaseLLMOutputParser | None, optional
            Output parser, by default None
        provider_name : str, optional
            Provider identifier, by default "unknown"
        """
        self.model = langchain_model
        self.telemetry = telemetry
        self.settings = settings
        self.output_parser = output_parser
        self.provider_name = provider_name

    def predict(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int | None = None,
    ) -> str | BaseModel:
        """Generate completion from prompt (LLMTool protocol method).

        This method implements the LLMTool protocol interface. It translates
        our protocol parameters to LangChain's configuration format, invokes
        the model, and optionally parses structured outputs.

        Parameters
        ----------
        prompt : str
            Input prompt for the LLM
        temperature : float | None, optional
            Sampling temperature (0.0-1.0). If None, uses settings default.
        max_tokens : int | None, optional
            Maximum tokens to generate. If None, uses settings default.
        timeout : int | None, optional
            Request timeout in seconds. If None, uses settings default.

        Returns
        -------
        str | BaseModel
            If output_parser is set: Returns validated Pydantic model instance
            If no parser: Returns raw string response

        Raises
        ------
        LLMError
            If prediction fails (timeout, rate limit, invalid response, etc.)

        Examples
        --------
        >>> # Without parser (returns string)
        >>> result = adapter.predict("What is bias?")
        >>> assert isinstance(result, str)
        >>>
        >>> # With Pydantic parser (returns BiasAnalysisOutput)
        >>> adapter_with_parser = LangChainLLMAdapter(
        ...     langchain_model=model,
        ...     telemetry=telemetry,
        ...     settings=settings,
        ...     output_parser=PydanticOutputParser(pydantic_object=BiasAnalysisOutput),
        ... )
        >>> result = adapter_with_parser.predict("Analyze: ...")
        >>> assert isinstance(result, BiasAnalysisOutput)
        """
        # Resolve defaults from settings
        temperature = (
            temperature if temperature is not None else self.settings.llm_temperature
        )
        max_tokens = (
            max_tokens if max_tokens is not None else self.settings.llm_max_tokens
        )
        timeout = timeout if timeout is not None else self.settings.llm_timeout_seconds

        # Validate inputs
        if not prompt or not prompt.strip():
            msg = "Prompt cannot be empty"
            raise LLMError(msg)

        try:
            # Create message
            # LangChain chat models expect Message objects, not raw strings
            message = HumanMessage(content=prompt)

            # Bind temperature and max_tokens to model for this call
            # This is the LangChain way to set per-call parameters
            bound_model = self.model.bind(
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Build config for timeout
            config: dict[str, Any] = {
                "timeout": timeout,
            }

            # Invoke LangChain model
            # If output_parser is set, chain: model → parser
            # Otherwise just invoke model directly
            if self.output_parser:
                # Create chain: bound_model | parser
                # LangChain's Runnable types are complex, ignore type checking
                chain = bound_model | self.output_parser  # type: ignore[operator,var-annotated]
                result = chain.invoke(message, config=config)  # type: ignore[arg-type]
            else:
                # Invoke model directly (returns AIMessage)
                response = bound_model.invoke(message, config=config)  # type: ignore[arg-type]
                result = response.content

            # Track successful call
            # Note: LangChain callbacks handle detailed telemetry
            # This is a high-level success event
            self.telemetry.log_info(
                "llm_call_success",
                provider=self.provider_name,
                model=getattr(self.model, "model_name", "unknown"),
                has_parser=self.output_parser is not None,
            )

            return result

        except Exception as e:
            # Track failure
            self.telemetry.log_info(
                "llm_call_failure",
                provider=self.provider_name,
                model=getattr(self.model, "model_name", "unknown"),
                error=str(e),
                error_type=type(e).__name__,
            )

            # Wrap all exceptions as LLMError for consistent error handling
            msg = f"LLM prediction failed: {e}"
            raise LLMError(msg) from e

    def get_token_count(self, text: str) -> int:
        """Estimate token count for text (LLMTool protocol method).

        Uses LangChain's built-in token counting if available, otherwise
        falls back to word-based heuristic.

        Parameters
        ----------
        text : str
            Text to count tokens for

        Returns
        -------
        int
            Estimated number of tokens

        Examples
        --------
        >>> count = adapter.get_token_count("Hello, world!")
        >>> assert count > 0
        """
        if not text:
            return 0

        # Try LangChain's built-in token counting
        try:
            if hasattr(self.model, "get_num_tokens"):
                return self.model.get_num_tokens(text)
        except Exception:
            pass

        # Fallback: word-based heuristic (tokens ≈ words × 1.3)
        word_count = len(text.split())
        return int(word_count * 1.3)
