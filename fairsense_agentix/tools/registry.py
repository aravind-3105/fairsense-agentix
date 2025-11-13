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
    """Resolve OCR tool with GPU-adaptive selection.

    Design Choices:
    - **"auto" mode**: Detects GPU and picks optimal tool automatically
    - **Graceful fallback**: PaddleOCR unavailable → try Tesseract → use Fake
    - **GPU detection**: Uses torch.cuda.is_available() from existing dependency
    - **Logging transparency**: User sees what was selected and why

    What This Enables:
    - Zero configuration: Works optimally on any hardware
    - Automatic optimization: GPU → PaddleOCR (fast), CPU → Tesseract (reliable)
    - Easy testing: force_cpu flag tests CPU path on GPU machines
    - Clear debugging: Logs show selection rationale

    Why This Way:
    - "auto" eliminates 90% of configuration decisions
    - Fallback chain handles missing dependencies gracefully
    - Logging makes behavior transparent (no magic)
    - Pattern matches _resolve_llm_tool() (consistency)

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
    import logging  # noqa: PLC0415

    logger = logging.getLogger(__name__)

    if settings.ocr_tool == "fake":
        from fairsense_agentix.tools.fake import FakeOCRTool  # noqa: PLC0415

        return FakeOCRTool()

    # GPU detection for "auto" mode
    # Design Choice: Check GPU once, use for both auto mode and use_gpu parameter
    # What This Enables: Consistent GPU detection across all code paths
    # Why This Way: Single source of truth, can be mocked in tests
    has_gpu = False
    if not settings.ocr_force_cpu:
        try:
            import torch  # noqa: PLC0415

            has_gpu = torch.cuda.is_available()
        except ImportError:
            logger.debug("torch not available, GPU detection skipped")

    # Auto mode: Select based on GPU availability
    # Design Choice: Separate auto selection from manual selection
    # What This Enables: Clear code flow, easy to understand selection logic
    # Why This Way: Explicit is better than implicit (Zen of Python)
    if settings.ocr_tool == "auto":
        selected_tool = "paddleocr" if has_gpu else "tesseract"
        logger.info(
            f"Auto-selected OCR tool: {selected_tool} "
            f"(GPU={'available' if has_gpu else 'not available'})"
        )
    else:
        selected_tool = settings.ocr_tool

    # Load PaddleOCR (GPU-accelerated)
    # Design Choice: Try/except with fallback to Tesseract
    # What This Enables: Graceful degradation if PaddleOCR not installed
    # Why This Way: Users shouldn't need to install everything upfront
    if selected_tool == "paddleocr":
        try:
            from fairsense_agentix.tools.ocr.paddleocr_tool import (  # noqa: PLC0415
                PaddleOCRTool,
            )

            return PaddleOCRTool(use_gpu=has_gpu)
        except ImportError as e:
            logger.warning(f"PaddleOCR not available ({e}), falling back to Tesseract")
            selected_tool = "tesseract"

    # Load Tesseract (CPU fallback)
    # Design Choice: Second fallback, most compatible
    # What This Enables: Works even without PaddleOCR installed
    # Why This Way: Tesseract is widely available (apt-get, brew)
    if selected_tool == "tesseract":
        try:
            from fairsense_agentix.tools.ocr.tesseract_tool import (  # noqa: PLC0415
                TesseractOCRTool,
            )

            return TesseractOCRTool()
        except ImportError as e:
            msg = "Tesseract OCR not available"
            raise ToolConfigurationError(
                msg,
                context={
                    "error": str(e),
                    "install": "pip install pytesseract && apt-get install tesseract-ocr",
                },
            ) from e

    raise ToolConfigurationError(
        f"Unknown ocr_tool: {selected_tool}",
        context={
            "ocr_tool": settings.ocr_tool,
            "valid_options": ["auto", "paddleocr", "tesseract", "fake"],
        },
    )


def _resolve_caption_tool(settings: Settings) -> CaptionTool:
    """Resolve caption tool with GPU-adaptive selection.

    Design Choices:
    - **"auto" mode**: Detects GPU and picks optimal model automatically
    - **Model selection**: BLIP-2 on GPU (fast), BLIP on CPU (acceptable)
    - **Preload option**: User controls startup vs first-call latency
    - **GPU detection**: Reuses torch.cuda.is_available() check

    What This Enables:
    - Zero configuration: Works optimally on any hardware
    - Automatic optimization: GPU → BLIP-2 (100ms), CPU → BLIP (1s)
    - Flexible deployment: API servers preload, dev iteration lazy loads
    - Clear debugging: Logs show selection rationale

    Why This Way:
    - "auto" eliminates model selection complexity
    - BLIP-2 is 10-40x faster on GPU than CPU (worth the distinction)
    - Preload choice gives control over latency profile
    - Pattern matches _resolve_ocr_tool() (consistency)

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
    import logging  # noqa: PLC0415

    logger = logging.getLogger(__name__)

    if settings.caption_model == "fake":
        from fairsense_agentix.tools.fake import FakeCaptionTool  # noqa: PLC0415

        return FakeCaptionTool()

    # GPU detection for "auto" mode
    # Design Choice: Check GPU once, use for both auto mode and use_gpu parameter
    # What This Enables: Consistent GPU detection, single source of truth
    # Why This Way: Reuses pattern from _resolve_ocr_tool(), can be mocked in tests
    has_gpu = False
    if not settings.caption_force_cpu:
        try:
            import torch  # noqa: PLC0415

            has_gpu = torch.cuda.is_available()
        except ImportError:
            logger.debug("torch not available, GPU detection skipped")

    # Auto mode: Select based on GPU availability
    # Design Choice: BLIP-2 on GPU (excellent + fast), BLIP on CPU (good + acceptable)
    # What This Enables: Optimal quality/speed for each hardware configuration
    # Why This Way: BLIP-2 too slow on CPU (2-5s), BLIP too slow on GPU vs BLIP-2
    if settings.caption_model == "auto":
        selected_model = "blip2" if has_gpu else "blip"
        logger.info(
            f"Auto-selected caption model: {selected_model} "
            f"(GPU={'available' if has_gpu else 'not available'})"
        )
    else:
        selected_model = settings.caption_model

    # Load BLIP-2 (GPU-accelerated, state-of-the-art)
    # Design Choice: No fallback (BLIP-2 requires transformers like BLIP does)
    # What This Enables: Production-grade quality on GPU servers
    # Why This Way: If BLIP-2 unavailable, BLIP will also be unavailable (same deps)
    if selected_model == "blip2":
        try:
            from fairsense_agentix.tools.caption.blip2_tool import (  # noqa: PLC0415
                BLIP2CaptionTool,
            )

            return BLIP2CaptionTool(use_gpu=has_gpu, preload=settings.caption_preload)
        except ImportError as e:
            msg = "BLIP-2 not available (transformers required)"
            raise ToolConfigurationError(
                msg,
                context={
                    "error": str(e),
                    "install": "pip install transformers torch accelerate",
                },
            ) from e

    # Load BLIP (CPU-friendly fallback)
    # Design Choice: Lighter model (476M vs 2.7B params)
    # What This Enables: Acceptable speed on CPU-only machines
    # Why This Way: BLIP fast enough on CPU (500ms-1s), BLIP-2 too slow (2-5s)
    if selected_model == "blip":
        try:
            from fairsense_agentix.tools.caption.blip_tool import (  # noqa: PLC0415
                BLIPCaptionTool,
            )

            return BLIPCaptionTool(use_gpu=has_gpu, preload=settings.caption_preload)
        except ImportError as e:
            msg = "BLIP not available (transformers required)"
            raise ToolConfigurationError(
                msg,
                context={
                    "error": str(e),
                    "install": "pip install transformers torch",
                },
            ) from e

    raise ToolConfigurationError(
        f"Unknown caption_model: {selected_model}",
        context={
            "caption_model": settings.caption_model,
            "valid_options": ["auto", "blip2", "blip", "fake"],
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
            from langchain_openai import ChatOpenAI  # noqa: PLC0415

            from fairsense_agentix.services import telemetry  # noqa: PLC0415
            from fairsense_agentix.tools.llm.callbacks import (  # noqa: PLC0415
                TelemetryCallback,
            )
            from fairsense_agentix.tools.llm.langchain_adapter import (  # noqa: PLC0415
                LangChainLLMAdapter,
            )
            from fairsense_agentix.tools.llm.output_schemas import (  # noqa: PLC0415
                BiasAnalysisOutput,
            )

            # Create telemetry callback
            callback = TelemetryCallback(
                telemetry=telemetry,
                provider="openai",
                model=settings.llm_model_name,
            )

            # Enable LangChain caching if configured
            # This sets up a global SQLite cache for all LLM calls
            if settings.llm_cache_enabled:
                from langchain_community.cache import SQLiteCache  # noqa: PLC0415
                from langchain_core.globals import set_llm_cache  # noqa: PLC0415

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

            # Apply structured output mode BEFORE retry wrapper
            # OpenAI's native JSON mode is more reliable than prompt-based parsing
            structured_model = base_model.with_structured_output(BiasAnalysisOutput)

            # Add retry after structured output configuration
            langchain_model = structured_model.with_retry(
                stop_after_attempt=3,  # Retry up to 3 times
            )

            # No output parser needed - structured_output handles parsing
            # Wrap in adapter to satisfy LLMTool protocol
            # LangChain types after .with_retry() are complex, requiring type ignores
            return LangChainLLMAdapter(  # type: ignore[return-value]
                langchain_model=langchain_model,  # type: ignore[arg-type]
                telemetry=telemetry,
                settings=settings,
                output_parser=None,  # Not needed with structured_output
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

            from fairsense_agentix.services import telemetry  # noqa: PLC0415
            from fairsense_agentix.tools.llm.callbacks import (  # noqa: PLC0415
                TelemetryCallback,
            )
            from fairsense_agentix.tools.llm.langchain_adapter import (  # noqa: PLC0415
                LangChainLLMAdapter,
            )
            from fairsense_agentix.tools.llm.output_schemas import (  # noqa: PLC0415
                BiasAnalysisOutput,
            )

            # Create telemetry callback
            callback = TelemetryCallback(
                telemetry=telemetry,
                provider="anthropic",
                model=settings.llm_model_name,
            )

            # Enable LangChain caching if configured
            # This sets up a global SQLite cache for all LLM calls
            if settings.llm_cache_enabled:
                from langchain_community.cache import SQLiteCache  # noqa: PLC0415
                from langchain_core.globals import set_llm_cache  # noqa: PLC0415

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

            # Apply structured output mode BEFORE retry wrapper
            # Anthropic's native JSON mode is more reliable than prompt-based parsing
            structured_model = anthropic_model.with_structured_output(
                BiasAnalysisOutput
            )

            # Add retry after structured output configuration
            langchain_model = structured_model.with_retry(
                stop_after_attempt=3,  # Retry up to 3 times
            )

            # No output parser needed - structured_output handles parsing
            # Wrap in adapter to satisfy LLMTool protocol
            # LangChain types after .with_retry() are complex, requiring type ignores
            return LangChainLLMAdapter(  # type: ignore[return-value]
                langchain_model=langchain_model,  # type: ignore[arg-type]
                telemetry=telemetry,
                settings=settings,
                output_parser=None,  # Not needed with structured_output
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
    LLM tool, just with a different prompt. This ensures cache sharing and
    consistent telemetry tracking.

    Design Choice: Reuse LLM Model
    -------------------------------
    The summarizer reuses the SAME LangChain model instance as the main LLM
    tool. This is critical for:
    - **Cache sharing**: Avoid duplicate API calls (cost savings)
    - **Telemetry consistency**: Track all LLM usage in one place
    - **Connection pooling**: Single HTTP client for all requests

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
    import logging  # noqa: PLC0415

    logger = logging.getLogger(__name__)

    if settings.llm_provider == "fake":
        from fairsense_agentix.tools.fake import FakeSummarizerTool  # noqa: PLC0415

        return FakeSummarizerTool()

    # Phase 5.4: Use LLM-based summarizer
    # NOTE: Creates separate model instance without structured output
    # (cannot reuse bias analysis model which has structured output configured)
    try:
        from fairsense_agentix.prompts import PromptLoader  # noqa: PLC0415
        from fairsense_agentix.services import telemetry  # noqa: PLC0415
        from fairsense_agentix.tools.summarizer import LLMSummarizer  # noqa: PLC0415

        # Create a plain text model for summarization (no structured output)
        if settings.llm_provider == "openai":
            from langchain_openai import ChatOpenAI  # noqa: PLC0415
            from pydantic import SecretStr  # noqa: PLC0415

            base_model = ChatOpenAI(
                model=settings.llm_model_name,
                api_key=SecretStr(settings.llm_api_key)
                if settings.llm_api_key
                else None,
            )
            # Add retry (no structured output for summarizer)
            langchain_model = base_model.with_retry(stop_after_attempt=3)

        elif settings.llm_provider == "anthropic":
            from langchain_anthropic import ChatAnthropic  # noqa: PLC0415
            from pydantic import SecretStr  # noqa: PLC0415

            # Ensure API key is present (should be validated by Settings)
            if not settings.llm_api_key:
                raise ToolConfigurationError(
                    "API key required for Anthropic provider",
                    context={"llm_provider": settings.llm_provider},
                )

            base_model = ChatAnthropic(  # type: ignore[call-arg,assignment]
                model_name=settings.llm_model_name,
                api_key=SecretStr(settings.llm_api_key),
            )
            # Add retry (no structured output for summarizer)
            langchain_model = base_model.with_retry(stop_after_attempt=3)

        else:
            raise ToolConfigurationError(
                f"Unsupported LLM provider for summarizer: {settings.llm_provider}",
                context={"llm_provider": settings.llm_provider},
            )

        logger.info(
            f"Creating LLMSummarizer (plain text model from {settings.llm_provider})"
        )

        return LLMSummarizer(
            langchain_model=langchain_model,  # type: ignore[arg-type]
            telemetry=telemetry,
            settings=settings,
            prompt_loader=PromptLoader(),
        )

    except Exception as e:
        raise ToolConfigurationError(
            f"Failed to create LLMSummarizer: {e}",
            context={
                "llm_provider": settings.llm_provider,
                "error_type": type(e).__name__,
            },
        ) from e


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


def _resolve_formatter_tool(settings: Settings) -> FormatterTool:
    """Resolve formatter tool.

    The formatter generates HTML with bias highlighting and data tables.
    It uses settings for bias color configuration.

    Parameters
    ----------
    settings : Settings
        Application configuration (includes bias color mappings)

    Returns
    -------
    FormatterTool
        Configured formatter tool instance

    Raises
    ------
    ToolConfigurationError
        If formatter cannot be constructed
    """
    # Phase 5.4: Use real HTML formatter
    try:
        from fairsense_agentix.tools.formatter import HTMLFormatter  # noqa: PLC0415

        # Get bias type colors from settings
        color_map = settings.get_bias_type_colors()

        return HTMLFormatter(color_map=color_map)

    except Exception as e:
        raise ToolConfigurationError(
            f"Failed to create HTMLFormatter: {e}",
            context={"error_type": type(e).__name__},
        ) from e


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

    Raises
    ------
    ToolConfigurationError
        If persistence tool cannot be constructed
    """
    # Phase 5.4: Use real CSV writer
    try:
        from fairsense_agentix.tools.persistence import CSVWriter  # noqa: PLC0415

        return CSVWriter(output_dir=output_dir)

    except Exception as e:
        raise ToolConfigurationError(
            f"Failed to create CSVWriter: {e}",
            context={
                "output_dir": str(output_dir),
                "error_type": type(e).__name__,
            },
        ) from e


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

        # Formatter and persistence use settings configuration
        formatter = _resolve_formatter_tool(settings)
        persistence = _resolve_persistence_tool(output_dir=settings.output_dir)

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
