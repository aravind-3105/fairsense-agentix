"""Image captioning tools for generating natural language descriptions.

This module provides caption implementations with GPU-adaptive selection:
- BLIP: Fast, lightweight, good quality (476M params, 500ms-1s on CPU)
- BLIP-2: State-of-the-art, excellent quality (2.7B params, 50-200ms on GPU)

The registry automatically selects the best model based on GPU availability.
"""

from fairsense_agentix.tools.caption.blip_tool import BLIPCaptionTool


__all__ = ["BLIPCaptionTool"]

# BLIP2CaptionTool exported conditionally (requires more resources)
try:
    from fairsense_agentix.tools.caption.blip2_tool import BLIP2CaptionTool

    __all__.append("BLIP2CaptionTool")
except ImportError:
    # Silently ignore if BLIP-2 dependencies not installed
    # This is expected behavior - BLIP-2 is optional (requires transformers, torch)
    # Registry will fallback to BLIP (CPU-friendly) or FakeCaptionTool
    pass
