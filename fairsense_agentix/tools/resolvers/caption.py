"""Resolver for caption tool."""

from fairsense_agentix.configs.settings import Settings
from fairsense_agentix.tools.exceptions import ToolConfigurationError
from fairsense_agentix.tools.interfaces import CaptionTool


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
