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
    VLMTool,
)
from fairsense_agentix.tools.resolvers import (
    _resolve_caption_tool,
    _resolve_embedder_tool,
    _resolve_faiss_tool,
    _resolve_formatter_tool,
    _resolve_llm_tool,
    _resolve_ocr_tool,
    _resolve_persistence_tool,
    _resolve_summarizer_tool,
    _resolve_vlm_tool,
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
    vlm : VLMTool
        Vision-Language Model for image analysis with CoT reasoning

    Examples
    --------
    >>> registry = ToolRegistry(ocr=FakeOCRTool(), caption=FakeCaptionTool(), ...)
    >>> text = registry.ocr.extract(b"image")
    >>> caption = registry.caption.caption(b"image")
    >>> result = registry.vlm.analyze_image(
    ...     b"image", "prompt", BiasVisualAnalysisOutput
    ... )
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
    vlm: "VLMTool"


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
        # Optimize: Skip loading heavy OCR/Caption models when using VLM mode
        # VLM mode doesn't need OCR/Caption (direct image analysis)
        # This saves 5-10s initialization time and prevents OOM on small systems
        ocr: OCRTool
        caption: CaptionTool
        if settings.image_analysis_mode == "vlm":
            # Use lightweight fake tools (no model loading)
            from fairsense_agentix.tools.fake import (  # noqa: PLC0415
                FakeCaptionTool,
                FakeOCRTool,
            )

            ocr = FakeOCRTool()
            caption = FakeCaptionTool()
        else:
            # Traditional mode: Load real OCR/Caption tools
            ocr = _resolve_ocr_tool(settings)
            caption = _resolve_caption_tool(settings)

        # Core tools always loaded (needed by all modes)
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

        # Formatter and persistence use settings configuration
        formatter = _resolve_formatter_tool(settings)
        persistence = _resolve_persistence_tool(output_dir=settings.output_dir)

        # VLM uses same provider as LLM (unified configuration)
        vlm = _resolve_vlm_tool(settings)

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
            vlm=vlm,
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
