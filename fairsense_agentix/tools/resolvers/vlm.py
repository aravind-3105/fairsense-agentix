"""Resolver for VLM tool."""

from fairsense_agentix.configs.settings import Settings
from fairsense_agentix.tools.exceptions import ToolConfigurationError
from fairsense_agentix.tools.interfaces import VLMTool


def _resolve_vlm_tool(settings: Settings) -> VLMTool:
    """Resolve Vision-Language Model tool using same provider as LLM.

    Uses the unified provider model - the same llm_provider setting
    controls both text LLM and vision VLM operations. This eliminates
    the need for separate API keys and configuration.

    Design Choice: Unified Provider
    --------------------------------
    VLM tool routes to OpenAI or Anthropic based on llm_provider setting:
        - llm_provider="openai" → GPT-4o Vision
        - llm_provider="anthropic" → Claude Sonnet Vision
        - llm_provider="fake" → FakeVLMTool (no API calls)

    No separate vlm_provider or vlm_api_key needed!

    What This Enables:
        - Single API key for all AI operations (text + vision)
        - Consistent provider across all analysis workflows
        - Simpler configuration (one provider choice)
        - Lower costs (no duplicate services)

    Why This Way:
        - Both OpenAI and Anthropic support vision + structured output
        - LangChain provides unified interface across providers
        - User configures once, uses everywhere
        - Pattern matches existing LLM tool resolution

    Parameters
    ----------
    settings : Settings
        Application configuration with llm_provider and llm_api_key

    Returns
    -------
    VLMTool
        Configured VLM tool instance (routes to OpenAI or Anthropic)

    Raises
    ------
    ToolConfigurationError
        If VLM cannot be initialized (unsupported provider, missing deps, etc.)

    Examples
    --------
    >>> settings = Settings(llm_provider="openai", llm_api_key="sk-...")
    >>> vlm = _resolve_vlm_tool(settings)
    >>> # Uses GPT-4o Vision with OpenAI API key

    >>> settings = Settings(llm_provider="anthropic", llm_api_key="sk-ant-...")
    >>> vlm = _resolve_vlm_tool(settings)
    >>> # Uses Claude Sonnet Vision with Anthropic API key
    """
    if settings.llm_provider == "fake":
        from fairsense_agentix.tools.vlm import FakeVLMTool  # noqa: PLC0415

        return FakeVLMTool()

    # Real VLM (OpenAI or Anthropic) - routes based on llm_provider
    try:
        from fairsense_agentix.tools.vlm import UnifiedVLMTool  # noqa: PLC0415

        return UnifiedVLMTool(settings)

    except Exception as e:
        # Provide helpful error message for unsupported providers
        if settings.llm_provider not in ("openai", "anthropic"):
            raise ToolConfigurationError(
                f"VLM requires llm_provider to be 'openai' or 'anthropic', "
                f"got '{settings.llm_provider}'",
                context={
                    "llm_provider": settings.llm_provider,
                    "supported_providers": ["openai", "anthropic", "fake"],
                },
            ) from e

        # Other initialization errors
        raise ToolConfigurationError(
            f"Failed to initialize VLM tool: {e}",
            context={
                "llm_provider": settings.llm_provider,
                "llm_model_name": settings.llm_model_name,
                "error_type": type(e).__name__,
            },
        ) from e
