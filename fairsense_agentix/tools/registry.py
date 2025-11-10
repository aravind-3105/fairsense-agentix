"""Tool registry for dependency injection and configuration.

This module provides a factory system that constructs and wires all tool
implementations based on configuration settings. The registry enables:
    - Swappable implementations (fake vs real tools)
    - Dependency injection (graphs receive tools, not hard-coded imports)
    - Configuration-driven tool selection
    - Single source of truth for tool instances

The registry pattern centralizes tool construction logic and makes it easy
to switch between fake tools (for testing) and real tools (for production).

Examples
--------
    >>> from fairsense_agentix.tools.registry import create_tool_registry
    >>> from fairsense_agentix.configs import Settings
    >>>
    >>> settings = Settings(ocr_tool="fake", llm_provider="fake")
    >>> registry = create_tool_registry(settings)
    >>> text = registry.ocr.extract(b"image")
    >>> print(text)
    'Extracted text from 5 byte image'
"""

from dataclasses import dataclass
from pathlib import Path

from fairsense_agentix.configs.settings import Settings
from fairsense_agentix.tools.exceptions import ToolConfigurationError
from fairsense_agentix.tools.interfaces import (
    CaptionTool,
    EmbedderTool,
    FAISSIndexTool,
    FormatterTool,
    LLMTool,
    OCRTool,
    PersistenceTool,
    SummarizerTool,
)


@dataclass
class ToolRegistry:
    """Container for all resolved tool instances.

    The registry holds instances of all tools needed by the platform,
    initialized from configuration. Tools are accessed by attribute name.

    Attributes
    ----------
    ocr : OCRTool
        Optical character recognition tool
    caption : CaptionTool
        Image captioning tool
    llm : LLMTool
        Large language model tool
    summarizer : SummarizerTool
        Text summarization tool
    embedder : EmbedderTool
        Text embedding tool
    faiss_risks : FAISSIndexTool
        FAISS index for AI risks
    faiss_rmf : FAISSIndexTool
        FAISS index for AI-RMF recommendations
    formatter : FormatterTool
        Output formatting tool (HTML, tables)
    persistence : PersistenceTool
        File persistence tool (CSV, JSON)

    Examples
    --------
    >>> registry = ToolRegistry(ocr=FakeOCRTool(), caption=FakeCaptionTool(), ...)
    >>> text = registry.ocr.extract(b"image")
    >>> caption = registry.caption.caption(b"image")
    """

    ocr: OCRTool
    caption: CaptionTool
    llm: LLMTool
    summarizer: SummarizerTool
    embedder: EmbedderTool
    faiss_risks: FAISSIndexTool
    faiss_rmf: FAISSIndexTool
    formatter: FormatterTool
    persistence: PersistenceTool


# ============================================================================
# Tool Resolver Functions
# ============================================================================


def _resolve_ocr_tool(settings: Settings) -> OCRTool:
    """Resolve OCR tool from settings.

    Parameters
    ----------
    settings : Settings
        Application configuration

    Returns
    -------
    OCRTool
        Configured OCR tool instance

    Raises
    ------
    ToolConfigurationError
        If ocr_tool setting is invalid or implementation not available
    """
    if settings.ocr_tool == "fake":
        from fairsense_agentix.tools.fake import FakeOCRTool  # noqa: PLC0415

        return FakeOCRTool()

    if settings.ocr_tool == "tesseract":
        # Phase 5: Import real TesseractOCR
        raise ToolConfigurationError(
            "Tesseract OCR not yet implemented (Phase 5)",
            context={"ocr_tool": settings.ocr_tool},
        )

    if settings.ocr_tool == "paddleocr":
        # Phase 5: Import real PaddleOCR
        raise ToolConfigurationError(
            "PaddleOCR not yet implemented (Phase 5)",
            context={"ocr_tool": settings.ocr_tool},
        )

    raise ToolConfigurationError(
        f"Unknown ocr_tool: {settings.ocr_tool}",
        context={
            "ocr_tool": settings.ocr_tool,
            "valid_options": ["fake", "tesseract", "paddleocr"],
        },
    )


def _resolve_caption_tool(settings: Settings) -> CaptionTool:
    """Resolve caption tool from settings.

    Parameters
    ----------
    settings : Settings
        Application configuration

    Returns
    -------
    CaptionTool
        Configured caption tool instance

    Raises
    ------
    ToolConfigurationError
        If caption_model setting is invalid or implementation not available
    """
    if settings.caption_model == "fake":
        from fairsense_agentix.tools.fake import FakeCaptionTool  # noqa: PLC0415

        return FakeCaptionTool()

    if settings.caption_model in ["blip", "blip2"]:
        # Phase 5: Import real BLIP caption tool
        raise ToolConfigurationError(
            f"{settings.caption_model.upper()} not yet implemented (Phase 5)",
            context={"caption_model": settings.caption_model},
        )

    if settings.caption_model == "llava":
        # Phase 5: Import real LLAVA caption tool
        raise ToolConfigurationError(
            "LLAVA not yet implemented (Phase 5)",
            context={"caption_model": settings.caption_model},
        )

    raise ToolConfigurationError(
        f"Unknown caption_model: {settings.caption_model}",
        context={
            "caption_model": settings.caption_model,
            "valid_options": ["fake", "blip", "blip2", "llava"],
        },
    )


def _resolve_llm_tool(settings: Settings) -> LLMTool:  # noqa: PLR0915
    """Resolve LLM tool from settings.

    Parameters
    ----------
    settings : Settings
        Application configuration

    Returns
    -------
    LLMTool
        Configured LLM tool instance

    Raises
    ------
    ToolConfigurationError
        If llm_provider setting is invalid or implementation not available
    """
    if settings.llm_provider == "fake":
        from fairsense_agentix.tools.fake import FakeLLMTool  # noqa: PLC0415

        return FakeLLMTool()

    if settings.llm_provider == "openai":
        # Phase 5.2: Real OpenAI LLM via LangChain
        try:
            from langchain_core.output_parsers import (  # noqa: PLC0415
                PydanticOutputParser,
            )
            from langchain_openai import ChatOpenAI  # noqa: PLC0415

            from fairsense_agentix.shared.services import get_telemetry  # noqa: PLC0415
            from fairsense_agentix.tools.llm.callbacks import (  # noqa: PLC0415
                TelemetryCallback,
            )
            from fairsense_agentix.tools.llm.langchain_adapter import (  # noqa: PLC0415
                LangChainLLMAdapter,
            )
            from fairsense_agentix.tools.llm.output_schemas import (  # noqa: PLC0415
                BiasAnalysisOutput,
            )

            # Get telemetry service
            telemetry = get_telemetry(settings)

            # Create telemetry callback
            callback = TelemetryCallback(
                telemetry=telemetry,
                provider="openai",
                model=settings.llm_model_name,
            )

            # Enable LangChain caching if configured
            # This sets up a global SQLite cache for all LLM calls
            if settings.llm_cache_enabled:
                from langchain.globals import set_llm_cache  # noqa: PLC0415
                from langchain_community.cache import SQLiteCache  # noqa: PLC0415

                # Ensure cache directory exists
                settings.llm_cache_path.parent.mkdir(parents=True, exist_ok=True)

                # Set global LangChain cache
                # Content-addressed: cache key = SHA256(model, prompt, params)
                set_llm_cache(SQLiteCache(database_path=str(settings.llm_cache_path)))

            # Create LangChain ChatOpenAI model with retry
            # Note: temperature, max_tokens, timeout set per-call in adapter
            from pydantic import SecretStr  # noqa: PLC0415

            base_model = ChatOpenAI(
                model=settings.llm_model_name,
                api_key=SecretStr(settings.llm_api_key)
                if settings.llm_api_key
                else None,
                callbacks=[callback],
            )

            langchain_model = base_model.with_retry(
                stop_after_attempt=3,  # Retry up to 3 times
            )

            # Create output parser for structured responses
            output_parser = PydanticOutputParser(pydantic_object=BiasAnalysisOutput)

            # Wrap in adapter to satisfy LLMTool protocol
            # LangChain types after .with_retry() are complex, requiring type ignores
            return LangChainLLMAdapter(  # type: ignore[return-value]
                langchain_model=langchain_model,  # type: ignore[arg-type]
                telemetry=telemetry,
                settings=settings,
                output_parser=output_parser,
                provider_name="openai",
            )

        except Exception as e:
            msg = f"Failed to initialize OpenAI LLM: {e}"
            raise ToolConfigurationError(
                msg, context={"llm_provider": settings.llm_provider}
            ) from e

    if settings.llm_provider == "anthropic":
        # Phase 5.2: Real Anthropic LLM via LangChain
        try:
            from langchain_anthropic import ChatAnthropic  # noqa: PLC0415
            from langchain_core.output_parsers import (  # noqa: PLC0415
                PydanticOutputParser,
            )

            from fairsense_agentix.shared.services import get_telemetry  # noqa: PLC0415
            from fairsense_agentix.tools.llm.callbacks import (  # noqa: PLC0415
                TelemetryCallback,
            )
            from fairsense_agentix.tools.llm.langchain_adapter import (  # noqa: PLC0415
                LangChainLLMAdapter,
            )
            from fairsense_agentix.tools.llm.output_schemas import (  # noqa: PLC0415
                BiasAnalysisOutput,
            )

            # Get telemetry service
            telemetry = get_telemetry(settings)

            # Create telemetry callback
            callback = TelemetryCallback(
                telemetry=telemetry,
                provider="anthropic",
                model=settings.llm_model_name,
            )

            # Enable LangChain caching if configured
            # This sets up a global SQLite cache for all LLM calls
            if settings.llm_cache_enabled:
                from langchain.globals import set_llm_cache  # noqa: PLC0415
                from langchain_community.cache import SQLiteCache  # noqa: PLC0415

                # Ensure cache directory exists
                settings.llm_cache_path.parent.mkdir(parents=True, exist_ok=True)

                # Set global LangChain cache
                # Content-addressed: cache key = SHA256(model, prompt, params)
                set_llm_cache(SQLiteCache(database_path=str(settings.llm_cache_path)))

            # Create LangChain ChatAnthropic model with retry
            # Note: temperature, max_tokens, timeout set per-call in adapter
            from pydantic import SecretStr  # noqa: PLC0415

            # Ensure API key is present (should be guaranteed by Settings validator)
            if not settings.llm_api_key:
                msg = "API key required for Anthropic provider"
                raise ToolConfigurationError(
                    msg, context={"llm_provider": settings.llm_provider}
                )

            anthropic_model = ChatAnthropic(  # type: ignore[call-arg]
                model_name=settings.llm_model_name,  # ChatAnthropic uses model_name not model
                api_key=SecretStr(settings.llm_api_key),
                callbacks=[callback],
            )

            langchain_model = anthropic_model.with_retry(
                stop_after_attempt=3,  # Retry up to 3 times
            )

            # Create output parser for structured responses
            output_parser = PydanticOutputParser(pydantic_object=BiasAnalysisOutput)

            # Wrap in adapter to satisfy LLMTool protocol
            # LangChain types after .with_retry() are complex, requiring type ignores
            return LangChainLLMAdapter(  # type: ignore[return-value]
                langchain_model=langchain_model,  # type: ignore[arg-type]
                telemetry=telemetry,
                settings=settings,
                output_parser=output_parser,
                provider_name="anthropic",
            )

        except Exception as e:
            msg = f"Failed to initialize Anthropic LLM: {e}"
            raise ToolConfigurationError(
                msg, context={"llm_provider": settings.llm_provider}
            ) from e

    if settings.llm_provider == "local":
        # Phase 5+: Real local LLM (vLLM, llama.cpp, etc.) - not yet implemented
        raise ToolConfigurationError(
            "Local LLM not yet implemented (future phase)",
            context={"llm_provider": settings.llm_provider},
        )

    raise ToolConfigurationError(
        f"Unknown llm_provider: {settings.llm_provider}",
        context={
            "llm_provider": settings.llm_provider,
            "valid_options": ["fake", "openai", "anthropic", "local"],
        },
    )


def _resolve_summarizer_tool(settings: Settings) -> SummarizerTool:
    """Resolve summarizer tool from settings.

    In most cases, the summarizer uses the same LLM backend as the main
    LLM tool, just with a different prompt.

    Parameters
    ----------
    settings : Settings
        Application configuration

    Returns
    -------
    SummarizerTool
        Configured summarizer tool instance

    Raises
    ------
    ToolConfigurationError
        If summarizer cannot be constructed
    """
    if settings.llm_provider == "fake":
        from fairsense_agentix.tools.fake import FakeSummarizerTool  # noqa: PLC0415

        return FakeSummarizerTool()

    # Phase 5: Use LLM-based summarizer
    raise ToolConfigurationError(
        "LLM-based summarizer not yet implemented (Phase 5)",
        context={"llm_provider": settings.llm_provider},
    )


def _resolve_embedder_tool(settings: Settings) -> EmbedderTool:
    """Resolve embedder tool from settings.

    Parameters
    ----------
    settings : Settings
        Application configuration

    Returns
    -------
    EmbedderTool
        Configured embedder tool instance

    Raises
    ------
    ToolConfigurationError
        If embedder cannot be constructed
    """
    # Check if using fake embedder
    if settings.embedding_model == "fake":
        from fairsense_agentix.tools.fake import FakeEmbedderTool  # noqa: PLC0415

        return FakeEmbedderTool(dimension=settings.embedding_dimension)

    # Phase 5: Use real sentence-transformers embedder
    try:
        from fairsense_agentix.tools.embeddings import (  # noqa: PLC0415
            SentenceTransformerEmbedder,
        )

        return SentenceTransformerEmbedder(
            model_name=settings.embedding_model,
            dimension=settings.embedding_dimension,
        )
    except Exception as e:
        raise ToolConfigurationError(
            f"Failed to create SentenceTransformerEmbedder: {e}",
            context={
                "model_name": settings.embedding_model,
                "dimension": settings.embedding_dimension,
                "error_type": type(e).__name__,
            },
        ) from e


def _resolve_faiss_tool(
    index_path: Path,
    embedder: EmbedderTool,
) -> FAISSIndexTool:
    """Resolve FAISS index tool.

    Parameters
    ----------
    index_path : Path
        Path to FAISS index file
    embedder : EmbedderTool
        Embedder tool for text-based search

    Returns
    -------
    FAISSIndexTool
        Configured FAISS index tool instance

    Raises
    ------
    ToolConfigurationError
        If FAISS index cannot be loaded
    """
    # Check if index file exists - if not, use fake
    if not index_path.exists():
        from fairsense_agentix.tools.fake import FakeFAISSIndexTool  # noqa: PLC0415

        return FakeFAISSIndexTool(index_path=index_path, embedder=embedder)

    # Phase 5: Use real FAISS index
    try:
        from fairsense_agentix.tools.faiss_index import (  # noqa: PLC0415
            FAISSIndexTool as RealFAISSIndexTool,
        )

        return RealFAISSIndexTool(index_path=index_path, embedder=embedder)
    except Exception as e:
        raise ToolConfigurationError(
            f"Failed to load FAISS index: {e}",
            context={
                "index_path": str(index_path),
                "error_type": type(e).__name__,
            },
        ) from e


def _resolve_formatter_tool() -> FormatterTool:
    """Resolve formatter tool.

    The formatter is simple HTML/CSS generation with no external dependencies,
    so we can use a real implementation even in Phase 4.

    Returns
    -------
    FormatterTool
        Configured formatter tool instance
    """
    # Phase 4: Use fake formatter (simple HTML generation)
    from fairsense_agentix.tools.fake import FakeFormatterTool  # noqa: PLC0415

    return FakeFormatterTool()

    # Phase 5: Can use real formatter if more sophisticated templates needed
    # from fairsense_agentix.tools.real.formatter import HTMLFormatter
    # return HTMLFormatter()


def _resolve_persistence_tool(output_dir: Path) -> PersistenceTool:
    """Resolve persistence tool.

    Parameters
    ----------
    output_dir : Path
        Directory for saved files

    Returns
    -------
    PersistenceTool
        Configured persistence tool instance
    """
    # Phase 4: Use fake persistence (doesn't actually write files by default)
    from fairsense_agentix.tools.fake import FakePersistenceTool  # noqa: PLC0415

    return FakePersistenceTool(output_dir=output_dir, actually_write=False)

    # Phase 5: Use real persistence
    # from fairsense_agentix.tools.real.persistence import FileWriter
    # return FileWriter(output_dir=output_dir)


# ============================================================================
# Registry Factory
# ============================================================================


def create_tool_registry(settings: Settings | None = None) -> ToolRegistry:
    """Create fully wired tool registry from configuration.

    This is the main factory function that constructs all tool instances
    and wires them together. Tools are resolved based on settings, enabling
    configuration-driven tool selection (fake vs real, different providers).

    Parameters
    ----------
    settings : Settings | None, optional
        Application configuration. If None, uses global singleton settings,
        by default None

    Returns
    -------
    ToolRegistry
        Fully initialized registry with all tools

    Raises
    ------
    ToolConfigurationError
        If any required tool cannot be constructed

    Examples
    --------
    >>> # Use global settings
    >>> registry = create_tool_registry()
    >>> registry.ocr.extract(b"image")

    >>> # Use custom settings
    >>> settings = Settings(ocr_tool="fake", llm_provider="fake")
    >>> registry = create_tool_registry(settings)
    >>> registry.llm.predict("prompt")
    """
    # Use global settings singleton if not provided
    if settings is None:
        from fairsense_agentix.configs import (  # noqa: PLC0415
            settings as global_settings,
        )

        settings = global_settings

    try:
        # Resolve each tool type
        ocr = _resolve_ocr_tool(settings)
        caption = _resolve_caption_tool(settings)
        llm = _resolve_llm_tool(settings)
        summarizer = _resolve_summarizer_tool(settings)
        embedder = _resolve_embedder_tool(settings)

        # FAISS indexes require embedder for text-based search
        faiss_risks = _resolve_faiss_tool(
            index_path=settings.faiss_risks_index_path,
            embedder=embedder,
        )
        faiss_rmf = _resolve_faiss_tool(
            index_path=settings.faiss_rmf_index_path,
            embedder=embedder,
        )

        # Formatter and persistence are straightforward
        formatter = _resolve_formatter_tool()
        persistence = _resolve_persistence_tool(output_dir=Path("outputs"))

        # Construct registry with all tools
        return ToolRegistry(
            ocr=ocr,
            caption=caption,
            llm=llm,
            summarizer=summarizer,
            embedder=embedder,
            faiss_risks=faiss_risks,
            faiss_rmf=faiss_rmf,
            formatter=formatter,
            persistence=persistence,
        )

    except Exception as e:
        # Wrap any errors in ToolConfigurationError
        if isinstance(e, ToolConfigurationError):
            raise
        raise ToolConfigurationError(
            f"Failed to create tool registry: {e!s}",
            context={"error_type": type(e).__name__},
        ) from e


# ============================================================================
# Global Registry Singleton
# ============================================================================

# Module-level singleton (lazy-initialized)
_global_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    """Get or create the global tool registry singleton.

    This provides convenient access to tools throughout the application
    without needing to pass registry instances around. The registry is
    initialized once from global settings and reused.

    Returns
    -------
    ToolRegistry
        Global tool registry instance

    Examples
    --------
    >>> from fairsense_agentix.tools.registry import get_tool_registry
    >>> registry = get_tool_registry()
    >>> text = registry.ocr.extract(b"image")

    >>> # Subsequent calls return same instance
    >>> registry2 = get_tool_registry()
    >>> assert registry is registry2
    """
    global _global_registry  # noqa: PLW0603

    if _global_registry is None:
        _global_registry = create_tool_registry()

    return _global_registry


def reset_tool_registry() -> None:
    """Reset the global tool registry singleton.

    Clears the global registry instance, forcing the next call to
    get_tool_registry() to create a fresh registry. This is used
    exclusively in tests to ensure test isolation - each test gets
    a clean registry without pollution from previous tests.

    **NOT for production use** - resetting the registry is expensive
    and unnecessary in production where a single registry instance
    should be reused throughout the application lifetime.

    Use Cases
    ---------
    - Test setup/teardown for isolation
    - Testing different configuration settings
    - Pytest fixtures that need fresh registry state

    Examples
    --------
    >>> # In test setup
    >>> class TestSomething:
    ...     def setup_method(self):
    ...         reset_tool_registry()  # Fresh registry per test
    ...
    ...     def test_workflow(self):
    ...         registry = get_tool_registry()  # New instance
    """
    global _global_registry  # noqa: PLW0603
    _global_registry = None
