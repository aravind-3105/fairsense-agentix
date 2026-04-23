"""LangChain-based text summarizer implementation.

This module provides an LLM-powered summarizer that condenses detailed analysis
outputs into concise summaries. It reuses the existing LangChain model infrastructure
to maintain cache sharing and telemetry consistency.

Design Choices
--------------
1. **Why reuse LangChain model**: Shares SQLite cache, avoids duplicate API calls
2. **Why prompt templates**: Centralized, versioned prompts for maintainability
3. **Why character-based truncation**: Simple, deterministic, no token counting overhead
4. **Why no structured output**: Summaries are free-form text, JSON would add overhead

What This Enables
-----------------
- Consistent LLM usage across all tools (same cache, same telemetry)
- Cost savings through cache hits (multiple summaries of same content)
- Easy prompt iteration (just update template file)
- Simple integration with existing bias/risk workflows

Examples
--------
    >>> from langchain_openai import ChatOpenAI
    >>> model = ChatOpenAI(model="gpt-4", temperature=0.3)
    >>> summarizer = LLMSummarizer(
    ...     langchain_model=model,
    ...     telemetry=telemetry,
    ...     settings=settings,
    ...     prompt_loader=PromptLoader(),
    ... )
    >>> summary = summarizer.summarize(
    ...     "Detailed bias analysis: [500 words]...", max_length=200
    ... )
    >>> print(summary)
    'HIGH severity gender bias detected in job posting...'
"""

import logging

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage

from fairsense_agentix.configs.settings import Settings
from fairsense_agentix.prompts.prompt_loader import PromptLoader
from fairsense_agentix.services.telemetry import TelemetryService
from fairsense_agentix.tools.exceptions import LLMError


logger = logging.getLogger(__name__)


class LLMSummarizer:
    """LLM-based text summarizer using LangChain models.

    This summarizer wraps an existing LangChain chat model and uses it to
    generate concise summaries of longer texts. It's designed to reuse the
    same model instance as the main LLM tool to benefit from shared caching
    and consistent telemetry tracking.

    Design Philosophy:
    - **Model reuse**: Shares cache with main LLM (cost savings)
    - **Template-based prompts**: Centralized prompt management
    - **Character truncation**: Simple, deterministic output sizing
    - **Telemetry integration**: Tracks usage and performance

    Parameters
    ----------
    langchain_model : BaseLanguageModel
        LangChain chat model (ChatOpenAI, ChatAnthropic, etc.)
        Should be the same instance used by the main LLM tool for cache sharing
    telemetry : TelemetryService
        Telemetry service for tracking usage and performance
    settings : Settings
        Application configuration
    prompt_loader : PromptLoader
        Loader for prompt templates

    Attributes
    ----------
    model : BaseLanguageModel
        The LangChain chat model being used
    telemetry : TelemetryService
        Telemetry service
    settings : Settings
        Application settings
    prompt_loader : PromptLoader
        Prompt template loader

    Examples
    --------
    >>> # Typical usage in registry resolver
    >>> llm_tool = _resolve_llm_tool(settings)  # Get main LLM
    >>> summarizer = LLMSummarizer(
    ...     langchain_model=llm_tool.model,  # Reuse model for cache sharing
    ...     telemetry=llm_tool.telemetry,
    ...     settings=settings,
    ...     prompt_loader=PromptLoader(),
    ... )

    Notes
    -----
    The summarizer uses the "summarize_v1" prompt template by default.
    To customize prompts, create new versions (summarize_v2.txt, etc.)
    and update the template name in the summarize() method.
    """

    def __init__(
        self,
        langchain_model: BaseLanguageModel,
        telemetry: TelemetryService,
        settings: Settings,
        prompt_loader: PromptLoader,
    ) -> None:
        """Initialize LLM summarizer.

        Parameters
        ----------
        langchain_model : BaseLanguageModel
            LangChain chat model to use for summarization
        telemetry : TelemetryService
            Telemetry service for tracking
        settings : Settings
            Application configuration
        prompt_loader : PromptLoader
            Prompt template loader
        """
        self.model = langchain_model
        self.telemetry = telemetry
        self.settings = settings
        self.prompt_loader = prompt_loader

        _mname = getattr(langchain_model, "model_name", "unknown")
        logger.info(f"LLMSummarizer initialized (model={_mname})")

    def summarize(
        self,
        text: str,
        max_length: int = 200,
    ) -> str:
        """Condense text into a shorter summary.

        Generates a concise summary of the input text using an LLM. The summary
        preserves key findings and maintains the tone of the original text while
        fitting within the specified length constraint.

        Parameters
        ----------
        text : str
            Text to summarize (typically a detailed analysis or report)
        max_length : int, optional
            Maximum summary length in characters, by default 200

        Returns
        -------
        str
            Condensed summary of the input text, truncated to max_length

        Raises
        ------
        LLMError
            If summarization fails (empty input, API error, timeout, etc.)

        Examples
        --------
        >>> summary = summarizer.summarize(
        ...     "Detailed analysis: Gender bias detected in 3 instances...",
        ...     max_length=150,
        ... )
        >>> len(summary) <= 150
        True

        Notes
        -----
        The summarizer may produce output longer than max_length, but it will
        always be truncated to exactly max_length characters in the return value.
        This is a simple, deterministic truncation - no token counting involved.
        """
        # Validate inputs
        if not text or not text.strip():
            raise LLMError(
                "Cannot summarize empty or whitespace-only text",
                context={
                    "text_length": len(text),
                    "max_length": max_length,
                },
            )

        if max_length <= 0:
            raise LLMError(
                "max_length must be positive",
                context={"max_length": max_length},
            )

        logger.debug(
            f"Summarizing text (input_length={len(text)}, max_length={max_length})",
        )

        try:
            # Load prompt template and render with substitutions
            prompt = self.prompt_loader.load(
                "summarize_v1",
                text=text,
                max_length=max_length,
            )

            # Call LLM with telemetry tracking
            with self.telemetry.timer(
                "summarize",
                model=getattr(self.model, "model_name", "unknown"),
                input_length=len(text),
                max_length=max_length,
            ):
                # Invoke model (uses cache if available)
                response = self.model.invoke([HumanMessage(content=prompt)])

                # Extract text from response
                # Handle both AIMessage (normal) and Pydantic models (structured output)
                if hasattr(response, "content"):
                    # AIMessage with .content attribute
                    raw_summary = response.content
                elif isinstance(response, str):
                    # Direct string response
                    raw_summary = response
                else:
                    # Fallback: convert to string
                    raw_summary = str(response)

                # Truncate to max_length (simple character-based truncation)
                summary = raw_summary[:max_length]

            logger.debug(
                f"Summary generated (raw_length={len(raw_summary)}, "
                f"truncated_length={len(summary)}, "
                f"truncated={len(raw_summary) > max_length})",
            )

            return summary

        except LLMError:
            # Re-raise our own errors unchanged
            raise

        except Exception as e:
            # Wrap all other exceptions as LLMError
            msg = "Text summarization failed"
            raise LLMError(
                msg,
                context={
                    "text_length": len(text),
                    "max_length": max_length,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            ) from e
